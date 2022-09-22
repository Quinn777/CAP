# Adversarial Vertex Mixup: Toward Better Adversarially Robust Generalization
# https://github.com/Saehyung-Lee/cifar10_challenge
import importlib
import torch.nn as nn
AverageMeter = importlib.import_module(f"src.utils.metrics").AverageMeter
ProgressMeter = importlib.import_module(f"src.utils.metrics").ProgressMeter
accuracy = importlib.import_module(f"src.utils.metrics").accuracy


def label_smoothing(onehot, n_classes, factor):
    return onehot * factor + (onehot - 1.) * ((factor - 1) / float(n_classes - 1))


def trades_mixup_loss(
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
    num_classes,
    gamma,
    lambda1,
    lambda2,
    distance="l_inf",
    natural_criterion=nn.CrossEntropyLoss(),
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

    # generate avmixup examples
    labels_one_hot = F.one_hot(y, num_classes).float()
    perturb = (x_adv - x_natural) * gamma
    x_vertex = x_natural + perturb
    x_vertex = x_vertex.clamp(clip_min, clip_max)
    y_natural = label_smoothing(labels_one_hot, num_classes, lambda1)
    y_vertex = label_smoothing(labels_one_hot, num_classes, lambda2)
    policy_x = torch.from_numpy(np.random.beta(1.0, 1.0, [x_natural.size(0), 1, 1, 1])).float().to(device)
    policy_y = policy_x.view(x_natural.size(0), -1)
    x_adv = policy_x * x_natural + (1 - policy_x) * x_vertex
    # not used
    y_adv = policy_y * y_natural + (1 - policy_y) * y_vertex

    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = natural_criterion(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(model(x_adv), dim=1), F.softmax(model(x_natural), dim=1)
    )
    loss = loss_natural + beta * loss_robust
    return loss, loss_natural, loss_robust*beta


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
        images = data[0].to(device)
        labels = data[1].to(device)

        outputs = model(images)
        # calculate robust loss
        loss, nat_loss, rob_loss = trades_mixup_loss(
            model=model,
            x_natural=images,
            y=labels,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            beta=args.beta,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            device=device,
            num_classes=args.num_classes,
            gamma=args.avmixup_gamma,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            distance=args.distance,
        )

        acc1, _ = accuracy(outputs, labels, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        nat_losses.update(nat_loss.item(), images.size(0))
        rob_losses.update(rob_loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        batch_time.update(time.time() - end)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    progress.display(i)

    return es, model, losses.avg, nat_losses.avg, rob_losses.avg, top1.avg
import time

from tqdm import tqdm
import torch
import importlib
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

AverageMeter = importlib.import_module(f"src.utils.metrics").AverageMeter
ProgressMeter = importlib.import_module(f"src.utils.metrics").ProgressMeter
accuracy = importlib.import_module(f"src.utils.metrics").accuracy
show_tensor = importlib.import_module("src.utils.visualize").show_tensor
get_attack_amp = importlib.import_module("src.utils.schedules").get_attack_amp

def label_smoothing(onehot, n_classes, factor):
    return onehot * factor + (onehot - 1.) * ((factor - 1) / float(n_classes - 1))

def trades_mixup_loss(
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
    num_classes,
    gamma,
    lambda1,
    lambda2,
    distance="l_inf",
    natural_criterion=nn.CrossEntropyLoss(),
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

    # generate avmixup examples
    labels_one_hot = F.one_hot(y, num_classes).float()
    perturb = (x_adv - x_natural) * gamma
    x_vertex = x_natural + perturb
    x_vertex = x_vertex.clamp(clip_min, clip_max)
    y_natural = label_smoothing(labels_one_hot, num_classes, lambda1)
    y_vertex = label_smoothing(labels_one_hot, num_classes, lambda2)
    policy_x = torch.from_numpy(np.random.beta(1.0, 1.0, [x_natural.size(0), 1, 1, 1])).float().to(device)
    policy_y = policy_x.view(x_natural.size(0), -1)
    x_adv = policy_x * x_natural + (1 - policy_x) * x_vertex
    #not used
    y_adv = policy_y * y_natural + (1 - policy_y) * y_vertex

    model.train()
    # show_tensor(x_natural[0], "natural")
    # show_tensor(x_adv[0], "adv")
    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = natural_criterion(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(model(x_adv), dim=1), F.softmax(model(x_natural), dim=1)
    )
    loss = loss_natural + beta * loss_robust
    # print(f"natural loss: {loss_natural}  robust loss: {loss_robust}")
    return loss, loss_natural, loss_robust*beta


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

    epsilon, step_size, num_steps = get_attack_amp(args, AttackPolicy, epoch, es)
    logger.info(f"epsilon: {epsilon}        step_size: {step_size}        num_steps: {num_steps}")
    end = time.time()
    model.train()
    for i, data in enumerate(dataloader):
        images = data[0].to(device)
        labels = data[1].to(device)

        outputs = model(images)
        # calculate robust loss
        loss, nat_loss, rob_loss = trades_mixup_loss(
            model=model,
            x_natural=images,
            y=labels,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            beta=args.beta,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            device=device,
            num_classes=args.num_classes,
            gamma=args.avmixup_gamma,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            distance=args.distance,
        )

        acc1, _ = accuracy(outputs, labels, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        nat_losses.update(nat_loss.item(), images.size(0))
        rob_losses.update(rob_loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        batch_time.update(time.time() - end)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    progress.display(i)

    return es, model, losses.avg, nat_losses.avg, rob_losses.avg, top1.avg
