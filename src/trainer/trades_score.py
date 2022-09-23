# Robustness and Accuracy Could Be Reconcilable by (Proper) Definition

import time
import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
AverageMeter = importlib.import_module(f"src.utils.metrics").AverageMeter
ProgressMeter = importlib.import_module(f"src.utils.metrics").ProgressMeter
accuracy = importlib.import_module(f"src.utils.metrics").accuracy


def trades_loss(
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
    num_classes,
    device,
    distance="l_inf",
    clip_score=0.1,
    label_smoothing=0.1
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
                loss_lse = torch.sum((F.log_softmax(model(x_adv), dim=1) - F.softmax(model(x_natural), dim=1)) ** 2)

            grad = torch.autograd.grad(loss_lse, [x_adv])[0]
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
    y_onehot = (1 - num_classes * label_smoothing /
                (num_classes - 1)) * F.one_hot(y, num_classes=num_classes) + label_smoothing / (num_classes - 1)

    logits_natural = F.softmax(model(x_natural), dim=1)
    logits_adv = F.softmax(model(x_adv), dim=1)
    loss_natural = torch.sum((logits_natural - y_onehot) ** 2, dim=-1)
    loss_robust = torch.sum((logits_adv - logits_natural) ** 2, dim=-1)
    loss_robust = F.relu(loss_robust-clip_score)  # clip loss value
    loss = loss_natural.mean() + beta * loss_robust.mean()

    return loss, loss_natural.mean(), loss_robust.mean()


def train(model, dataloader, optimizer, args, epoch, device, logger, AttackPolicy):
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
        x_natural = data[0].to(device)
        labels = data[1].to(device)

        nat_output = model(x_natural)
        loss, nat_loss, rob_loss = trades_loss(
            model=model,
            x_natural=x_natural,
            y=labels,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            beta=args.beta,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            num_classes=args.num_classes,
            device=device,
            distance=args.distance,
            clip_score=args.clip_score
        )
        acc1, _ = accuracy(nat_output, labels, topk=(1, ))
        losses.update(loss.item(), x_natural.size(0))
        nat_losses.update(nat_loss.item(), x_natural.size(0))
        rob_losses.update(rob_loss.item(), x_natural.size(0))
        top1.update(acc1[0].item(), x_natural.size(0))
        batch_time.update(time.time() - end)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return model, losses.avg, nat_losses.avg, rob_losses.avg, top1.avg
