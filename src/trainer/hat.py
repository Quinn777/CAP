# Helper-based Adversarial Training: Reducing Excessive Margin to

import time
import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from src.utils.context import ctx_noparamgrad_and_eval

AverageMeter = importlib.import_module(f"src.utils.metrics").AverageMeter
ProgressMeter = importlib.import_module(f"src.utils.metrics").ProgressMeter
accuracy = importlib.import_module(f"src.utils.metrics").accuracy
show_tensor = importlib.import_module("src.utils.visualize").show_tensor


def hat_loss(model, x, y, optimizer, step_size, epsilon, perturb_steps, h, beta, gamma,
             clip_min, clip_max, device, attack='linf-pgd', hr_model=None):
    """
    TRADES + Helper-based adversarial training.
    """

    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()

    x_adv = x.detach() + 0.001 * torch.randn(x.shape).to(device).detach()
    p_natural = F.softmax(model(x), dim=1)

    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    elif attack == 'l2-pgd':
        delta = 0.001 * torch.randn(x.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        batch_size = len(x)
        optimizer_delta = torch.optim.SGD([delta], lr=step_size)

        for _ in range(perturb_steps):
            adv = x + delta
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward(retain_graph=True)

            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            delta.data.add_(x)
            delta.data.clamp_(0, 1).sub_(x)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max), requires_grad=False)
    x_hr = x + h * (x_adv - x)
    with ctx_noparamgrad_and_eval(hr_model):
        y_hr = hr_model(x_adv).argmax(dim=1)

    optimizer.zero_grad()

    out_clean, out_adv, out_help = model(x), model(x_adv), model(x_hr)
    loss_clean = F.cross_entropy(out_clean, y, reduction='mean')
    loss_adv = (1 / len(x)) * criterion_kl(F.log_softmax(out_adv, dim=1), F.softmax(out_clean, dim=1))

    loss_help = F.cross_entropy(out_help, y_hr, reduction='mean')
    loss = loss_clean + beta * loss_adv + gamma * loss_help

    return loss


def train(model, dataloader, optimizer, args, epoch, device, logger, hr_model, AttackPolicy):
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
        loss = hat_loss(
            model=model,
            x=images,
            y=labels,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            h=args.h,
            beta=args.beta,
            gamma=args.hat_gamma,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            device=device,
            hr_model=hr_model
        )

        acc1, _ = accuracy(outputs, labels, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        batch_time.update(time.time() - end)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    progress.display(i)

    return model, losses.avg, 0, 0, top1.avg
