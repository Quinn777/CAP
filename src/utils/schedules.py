import torch
import sys
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler


def get_lr_policy(lr_schedule):
    """Implement a new schduler directly in this file.
    Args should contain a single choice for learning rate scheduler."""

    d = {
        "constant": constant_schedule,
        "cosine": cosine_schedule,
        "step": step_schedule,
    }
    return d[lr_schedule]


def get_optimizer(model, choose, lr):
    if choose == "sgd":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=0.0001,
        )
    elif choose == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001, )
    elif choose == "rmsprop":
        optim = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=0.0001,
        )
    elif choose == "adamax":
        optim = torch.optim.Adamax(
            model.parameters(),
            lr=lr,
            weight_decay=0.0001,
        )
    elif choose == "adamw":
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.0001,
        )
    else:
        print(f"{choose} is not supported.")
        sys.exit(0)
    return optim


def new_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def constant_schedule(optimizer, opt):
    def set_lr(epoch, lr=opt.lr, epochs=opt.epoch):
        if epoch < opt.warmup_epochs:
            lr = opt.warmup_lr

        new_lr(optimizer, lr)

    return set_lr


def cosine_schedule(optimizer, opt):
    def set_lr(epoch, lr=opt.lr, epochs=opt.epoch):
        if epoch < opt.warmup_epochs:
            a = opt.warmup_lr
        else:
            epoch = epoch - opt.warmup_epochs
            a = lr * 0.5 * (1 + np.cos((epoch - 1) / epochs * np.pi))

        new_lr(optimizer, a)

    return set_lr


def step_schedule(optimizer, opt):
    def set_lr(epoch, lr=opt.lr, epochs=opt.epoch):
        if epoch < opt.warmup_epochs:
            a = opt.warmup_lr
        else:
            epoch = epoch - opt.warmup_epochs
            a = lr
            if epoch // opt.decay_step == 0:
                a = lr * opt.gamma
        new_lr(optimizer, a)

    return set_lr


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.decay_step, gamma=opt.gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=opt.factor, threshold=0.0001, patience=opt.patience)
    elif opt.lr_policy == 'cosineAnn':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.tmax, eta_min=1e-8)
    elif opt.lr_policy == 'cosineAnnWarm':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.tmax, T_mult=opt.tmult)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 100))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def update_learning_rate(schedulers):
    for scheduler in schedulers:
        scheduler.step()


class AttackPolicy:
    def __init__(self, args):
        self.step_max = args.num_steps
        self.step_min = 1
        self.epsilon_max = args.epsilon
        self.step_size = args.step_size

    def compute(self, epoch):
        if 0 < epoch < 20 and (epoch % 2) == 0:
            self.step_min += 1
        return self.epsilon_max, self.step_size, min(self.step_min, self.step_max)


def get_attack_amp(args, AttackPolicy, epoch):
    if args.attack_policy:
        epsilon, step_size, step = AttackPolicy.compute(epoch)
    else:
        epsilon, step_size, step = args.epsilon, args.step_size, args.num_steps
    return epsilon, step_size, step