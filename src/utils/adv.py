import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param logits: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = F.one_hot(targets, self.num_classes)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss


def pgd_whitebox(
    model,
    x,
    y,
    epsilon,
    num_steps,
    step_size,
    clip_min,
    clip_max,
    device,
    is_random=True,
):

    x_pgd = Variable(x.data, requires_grad=True)
    if is_random:
        random_noise = (
            torch.FloatTensor(x_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        )
        x_pgd = Variable(x_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(x_pgd), y)
        loss.backward()
        eta = step_size * x_pgd.grad.data.sign()
        x_pgd = Variable(x_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(x_pgd.data - x.data, -epsilon, epsilon)
        x_pgd = Variable(x.data + eta, requires_grad=True)
        x_pgd = Variable(torch.clamp(x_pgd, clip_min, clip_max), requires_grad=True)

    return x_pgd


def fgsm_whitebox(
    model,
    x,
    y,
    epsilon,
    clip_min,
    clip_max,
):
    X_fgsm = Variable(x.data, requires_grad=True)
    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X_fgsm), y)
    loss.backward()
    eta = epsilon * X_fgsm.grad.data.sign()
    X_fgsm = Variable(x.data + eta, requires_grad=True)
    X_fgsm = Variable(torch.clamp(X_fgsm, clip_min, clip_max), requires_grad=True)

    return X_fgsm


def cw_whitebox(
    model,
    x,
    y,
    epsilon,
    num_steps,
    step_size,
    clip_min,
    clip_max,
    device,
    num_classes,
    is_random=True,
):
    X_cw = Variable(x.data, requires_grad=True)
    if is_random:
        random_noise = (
            torch.FloatTensor(X_cw.shape).uniform_(-epsilon, epsilon).to(device)
        )
        X_cw = Variable(X_cw.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        with torch.enable_grad():
            loss = CWLoss(num_classes=num_classes)(model(X_cw), y)
        loss.backward()
        eta = step_size * X_cw.grad.data.sign()
        X_cw = Variable(X_cw.data + eta, requires_grad=True)
        eta = torch.clamp(X_cw.data - x.data, -epsilon, epsilon)
        X_cw = Variable(x.data + eta, requires_grad=True)
        X_cw = Variable(torch.clamp(X_cw, clip_min, clip_max), requires_grad=True)

    return X_cw


def mim_whitebox(
    model,
    x,
    y,
    epsilon,
    num_steps,
    step_size,
    clip_min,
    clip_max,
    device,
    is_random=True,
    decay_factor=1.0,
):
    previous_grad = torch.zeros_like(x.data)
    X_mim = Variable(x.data, requires_grad=True)
    if is_random:
        random_noise = (
            torch.FloatTensor(X_mim.shape).uniform_(-epsilon, epsilon).to(device)
        )
        X_mim = Variable(X_mim.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_mim), y)
        loss.backward()
        grad = X_mim.grad.data / torch.mean(torch.abs(X_mim.grad.data), [1, 2, 3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        eta = step_size * previous_grad.data.sign()
        X_mim = Variable(X_mim.data + eta, requires_grad=True)
        eta = torch.clamp(X_mim.data - x.data, -epsilon, epsilon)
        X_mim = Variable(x.data + eta, requires_grad=True)
        X_mim = Variable(torch.clamp(X_mim, clip_min, clip_max), requires_grad=True)

    return X_mim