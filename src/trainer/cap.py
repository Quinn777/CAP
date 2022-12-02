import time
import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

get_attack_amp = importlib.import_module("src.utils.schedules").get_attack_amp
AverageMeter = importlib.import_module(f"src.utils.metrics").AverageMeter
ProgressMeter = importlib.import_module(f"src.utils.metrics").ProgressMeter
accuracy = importlib.import_module(f"src.utils.metrics").accuracy


def CAP_loss(
        model,
        x_natural,
        x_aug,
        y,
        optimizer,
        step_size,
        epsilon,
        perturb_steps,
        beta,
        clip_min,
        clip_max,
        device,
        args,
        epoch,
        cap_adversary,
        distance="l_inf",
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

    if epoch >= args.cap_warmup:
        cap = cap_adversary.calc_cap(inputs_aug=x_aug,
                                     inputs_clean=x_natural,
                                     targets=y,
                                     beta=args.beta)
        cap_adversary.perturb(cap)

    # zero gradient
    optimizer.zero_grad()

    y_onehot = F.one_hot(y, args.num_classes).float()
    logits_aug = F.softmax(model(x_aug), dim=1)

    y_smooth = epsilon * logits_aug + (1 - epsilon) * y_onehot

    logits_natural = F.softmax(model(x_natural), dim=1)
    logits_adv = F.softmax(model(x_adv), dim=1)

    loss_natural = torch.sum((logits_natural - y_smooth) ** 2, dim=-1)
    loss_robust = torch.sum((logits_adv - logits_natural) ** 2, dim=-1)
    loss_robust = F.relu(loss_robust)  # clip loss value

    loss = loss_natural.mean() + beta * loss_robust.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= args.cap_warmup:
        cap_adversary.restore(cap)

    return loss, loss_natural.mean(), loss_robust.mean(),


def train(model, dataloader, optimizer, args, epoch, device, logger, cap_adversary, AttackPolicy):
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
    epsilon, step_size, num_steps = get_attack_amp(args, AttackPolicy, epoch)
    end = time.time()
    model.train()
    for i, data in enumerate(dataloader):
        x_natural = data[0][0].to(device)
        x_aug = data[0][1].to(device)
        label = data[1][0].to(device)

        nat_output = model(x_natural)
        # calculate robust loss
        loss, nat_loss, rob_loss = CAP_loss(
            model=model,
            x_natural=x_natural,
            x_aug=x_aug,
            y=label,
            optimizer=optimizer,
            step_size=step_size,
            epsilon=epsilon,
            perturb_steps=num_steps,
            beta=args.beta,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            device=device,
            args=args,
            epoch=epoch,
            cap_adversary=cap_adversary,
            distance=args.distance,
        )

        acc1, _ = accuracy(nat_output, label, topk=(1,))
        losses.update(loss.item(), x_natural.size(0))
        nat_losses.update(nat_loss.item(), x_natural.size(0))
        rob_losses.update(rob_loss.item(), x_natural.size(0))
        top1.update(acc1[0].item(), x_natural.size(0))
        batch_time.update(time.time() - end)

        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    progress.display(i)

    return model, losses.avg, nat_losses.avg, rob_losses.avg, top1.avg,
