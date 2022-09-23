# Attacks Which Do Not Kill Training Make Adversarial Learning Stronger

import time
import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.utils.fat import earlystop

AverageMeter = importlib.import_module(f"src.utils.metrics").AverageMeter
ProgressMeter = importlib.import_module(f"src.utils.metrics").ProgressMeter
accuracy = importlib.import_module(f"src.utils.metrics").accuracy
show_tensor = importlib.import_module("src.utils.visualize").show_tensor


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
    device,
    tau,
    args,
    natural_criterion=nn.CrossEntropyLoss(),
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    batch_size = len(x_natural)

    output_adv, output_target, output_natural, count = earlystop(model, x_natural, y, step_size, epsilon, perturb_steps,
                                                                 clip_min, clip_max, device, tau,
                                                                 randominit_type="normal_distribution_randominit",
                                                                 loss_fn='kl', args=args, omega=args.omega)

    model.train()
    x_adv = Variable(torch.clamp(output_adv, clip_min, clip_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(output_natural)
    loss_natural = natural_criterion(logits, output_target)
    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(model(x_adv), dim=1), F.softmax(model(output_natural), dim=1)
    )
    loss = loss_natural + beta * loss_robust
    return loss, loss_natural, loss_robust*beta


def adjust_tau(epoch, args):
    tau = args.tau
    if args.dynamictau:
        if epoch <= 30:
            tau = 0
        elif epoch <= 50:
            tau = 1
        elif epoch <= 70:
            tau = 2
        else:
            tau = 3
    return tau


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

    tau = adjust_tau(epoch, args)

    end = time.time()
    model.train()
    for i, data in enumerate(dataloader):
        images = data[0].to(device)
        labels = data[1].to(device)

        outputs = model(images)
        # calculate robust loss
        loss, nat_loss, rob_loss = trades_loss(
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
            tau=tau,
            args=args
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

    return model, losses.avg, nat_losses.avg, rob_losses.avg, top1.avg
