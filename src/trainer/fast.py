# Fast is better than free: Revisiting adversarial training

import time
import importlib
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchattacks import FFGSM
AverageMeter = importlib.import_module(f"src.utils.metrics").AverageMeter
ProgressMeter = importlib.import_module(f"src.utils.metrics").ProgressMeter
accuracy = importlib.import_module(f"src.utils.metrics").accuracy
show_tensor = importlib.import_module("src.utils.visualize").show_tensor


def pgd_loss(
    model,
    x_natural,
    y,
    optimizer,
    step_size,
    epsilon,
    perturb_steps,
    clip_min,
    clip_max,
    device,
    rand_start_mode='gaussian',
    criterion=nn.CrossEntropyLoss(),
):
    model.eval()
    if rand_start_mode == 'gaussian':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).detach().to(device)
    elif rand_start_mode == 'uniformv1':
        x_adv = x_natural.detach() + epsilon * torch.rand(x_natural.shape).detach().to(device)
    elif rand_start_mode == 'uniformv2':
        x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).to(device)
    else:
        raise NameError

    for _ in range(perturb_steps):
        x_adv.requires_grad_()

        with torch.enable_grad():
            loss = criterion(model(x_adv), y)

        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
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


def train(model, dataloader, optimizer, args, epoch, device, logger, AttackPolicy):
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

        attack = FFGSM(model, eps=args.epsilon, alpha=0.0078)
        adv_images = attack(images, labels)
        output = model(adv_images)
        loss = nn.CrossEntropyLoss()(output, labels)
        # measure accuracy and record loss
        acc1, _ = accuracy(output, labels, topk=(1, ))
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

    return model, losses.avg, 0, 0, top1.avg
