
import time
import importlib
import torch
import torch.nn as nn
from torch.autograd import Variable

AverageMeter = importlib.import_module(f"src.utils.metrics").AverageMeter
ProgressMeter = importlib.import_module(f"src.utils.metrics").ProgressMeter
accuracy = importlib.import_module(f"src.utils.metrics").accuracy
show_tensor = importlib.import_module("src.utils.visualize").show_tensor


def fgsm_loss(
    model,
    x_natural,
    y,
    optimizer,
    epsilon,
    clip_min,
    clip_max,
    device,
    criterion=nn.CrossEntropyLoss(),
):
    model.eval()
    x_adv = x_natural.clone()
    x_adv.requires_grad_()

    with torch.enable_grad():
        loss = criterion(model(x_adv), y)

    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv.detach() + epsilon * torch.sgn(grad.detach())
    x_adv = torch.min(
        torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
    )
    x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)

    model.train()
    x_adv = Variable(torch.clamp(x_adv, min=clip_min, max=clip_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate loss
    logits = model(x_adv)
    loss = criterion(logits, y)
    return loss


def train(model, dataloader, optimizer, args, epoch, device, logger):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
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
        loss = fgsm_loss(
            model=model,
            x_natural=images,
            y=labels,
            optimizer=optimizer,
            epsilon=args.epsilon,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            device=device,
        )

        # measure accuracy and record loss
        acc1, _ = accuracy(outputs, labels, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    progress.display(i)

    return model, top1.avg, losses.avg, 0, 0