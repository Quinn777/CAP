import time
import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

AverageMeter = importlib.import_module(f"src.utils.metrics").AverageMeter
ProgressMeter = importlib.import_module(f"src.utils.metrics").ProgressMeter
accuracy = importlib.import_module(f"src.utils.metrics").accuracy


def show_assignments(a, b, P):
    norm_P = P / P.max()
    norm_P = norm_P.detach().cpu().numpy()
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            plt.arrow(a[i, 0], a[i, 1], b[j, 0] - a[i, 0], b[j, 1] - a[i, 1],
                      alpha=norm_P[0, i, j])
    plt.title('Assignments')
    plt.scatter(a[:, 0], a[:, 1])
    plt.scatter(b[:, 0], b[:, 1])
    plt.axis('off')
    plt.show()


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def show_distance_mat(C, type):
    # Distance matrix
    C = C.detach().cpu().numpy()
    C = normalization(C)
    plt.imshow(C, cmap=plt.cm.GnBu)
    plt.title(type)
    # plt.colorbar()
    plt.show()


def _jensen_shannon_div(logit1, logit2, T=1.):
    prob1 = F.softmax(logit1 / T, dim=1)
    prob2 = F.softmax(logit2 / T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)

    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
    jsd = F.kl_div(logsoftmax, prob1, reduction='batchmean')
    jsd += F.kl_div(logsoftmax, prob2, reduction='batchmean')
    return jsd * 0.5


def consistency_loss(
        model,
        x_natural,
        y,
        optimizer,
        step_size,
        epsilon,
        perturb_steps,
        beta,
        clip_min,
        clip_max,
        device,
        distance="l_inf",
        natural_criterion=nn.CrossEntropyLoss(),
        model_name=""
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = (
            x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    )
    if distance == "l_inf":
        for _ in (range(perturb_steps)):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(
                    F.log_softmax(model(x_adv), dim=1),
                    F.softmax(model(x_natural), dim=1),
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
            )
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif distance == "l_2":
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)
        for _ in range(perturb_steps):
            adv = x_natural + delta
            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(
                    F.log_softmax(model(adv), dim=1), F.softmax(model(x_natural), dim=1)
                )

            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(
                    delta.grad[grad_norms == 0]
                )
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(clip_min, clip_max).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    optimizer.zero_grad()

    outputs = model(x_natural)
    outputs_adv = model(x_adv)
    loss_ce = F.cross_entropy(outputs, y.repeat(2))
    loss_adv = beta * criterion_kl(F.log_softmax(outputs_adv, dim=1), F.softmax(outputs, dim=1))

    # consistency regularization
    outputs_adv1, outputs_adv2 = outputs_adv.chunk(2)
    loss_con = 1.0 * _jensen_shannon_div(outputs_adv1, outputs_adv2, 0.5)

    # total loss
    loss = loss_ce + loss_con + loss_adv
    return loss, 0, 0


def train(model, dataloader, optimizer, args, epoch, device, logger, es, AttackPolicy):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    nat_losses = AverageMeter("Natural Loss", ":.4f")
    rob_losses = AverageMeter("Robust Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, top1],
        logger,
        prefix="Epoch: [{}]".format(epoch),
    )
    end = time.time()
    model.train()
    for i, data in enumerate(dataloader):
        x_natural = data[0][0].to(device)
        x_aug = data[0][1].to(device)
        images_pair = torch.cat([x_natural, x_aug], dim=0)
        label = data[1][0].to(device)
        # images = torch.cat(data[0], dim=0).to(device)
        # labels = torch.cat(data[1], dim=0).to(device)
        # calculate robust loss
        nat_output = model(x_natural)
        loss, nat_loss, rob_loss = consistency_loss(
            model=model,
            x_natural=images_pair,
            y=label,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            beta=args.beta,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            device=device,
            distance=args.distance,
            model_name=args.model_name
        )

        acc1, _ = accuracy(nat_output, label, topk=(1,))
        losses.update(loss.item(), x_natural.size(0))
        nat_losses.update(nat_loss, x_natural.size(0))
        rob_losses.update(rob_loss, x_natural.size(0))
        top1.update(acc1[0].item(), x_natural.size(0))
        batch_time.update(time.time() - end)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return es, model, losses.avg, nat_losses.avg, rob_losses.avg, top1.avg
