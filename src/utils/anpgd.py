import torch
from advertorch.attacks import Attack
from advertorch.attacks import LabelMixin
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.utils import batch_multiply
from advertorch.utils import clamp
import torch.nn as nn
from advertorch.attacks import LinfPGDAttack
import torch.nn.functional as F


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


class SoftlabelCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftlabelCrossEntropyLoss, self).__init__()

    def forward(self, logits, soft_labels):
        log_softmax = F.log_softmax(logits, dim=-1)
        loss = torch.mean(torch.sum(- soft_labels * log_softmax, dim=-1))

        return loss


def get_loss_fn(name, reduction):
    if name == "xent":
        loss_fn = nn.CrossEntropyLoss(reduction=reduction)
    elif name == "slm":
        from advertorch.loss import SoftLogitMarginLoss
        loss_fn = SoftLogitMarginLoss(reduction=reduction)
    elif name == "lm":
        from advertorch.loss import LogitMarginLoss
        loss_fn = LogitMarginLoss(reduction=reduction)
    elif name == "cw":
        from advertorch.loss import CWLoss
        loss_fn = CWLoss(reduction=reduction)
    else:
        raise NotImplementedError("loss_fn={}".format(name))

    return loss_fn


def get_sum_loss_fn(name):
    return get_loss_fn(name, "sum")


def get_mean_loss_fn(name):
    return get_loss_fn(name, "elementwise_mean")


def get_none_loss_fn(name):
    return get_loss_fn(name, "none")


def bisection_search(
        cur_eps, ptb, model, data, label, fn_margin, margin_init,
        maxeps, num_steps,
        cur_min=None, clip_min=0., clip_max=1.):

    assert torch.all(cur_eps <= maxeps)

    margin = margin_init

    if cur_min is None:
        cur_min = torch.zeros_like(margin)
    cur_max = maxeps.clone().detach()

    for ii in range(num_steps):
        cur_min = torch.max((margin < 0).float() * cur_eps, cur_min)
        cur_max = torch.min(((margin < 0).float() * maxeps
                             + (margin >= 0).float() * cur_eps),
                            cur_max)

        cur_eps = (cur_min + cur_max) / 2
        margin = fn_margin(
            model(clamp(data + batch_multiply(cur_eps, ptb),
                        min=clip_min, max=clip_max)
                  ), label)

    assert torch.all(cur_eps <= maxeps)

    return cur_eps


class ANPGD(Attack, LabelMixin):

    def __init__(self, pgdadv, mineps, maxeps, num_search_steps,
                 eps_iter_scale, search_loss_fn=None):
        self.pgdadv = pgdadv
        self.predict = self.pgdadv.predict
        self.mineps = mineps  # mineps is used outside to set prev_eps
        self.maxeps = maxeps
        self.num_search_steps = num_search_steps
        self.eps_iter_scale = eps_iter_scale
        assert search_loss_fn is not None
        self.search_loss_fn = search_loss_fn


    def _get_unitptb_and_eps(self, xadv, x, y, prev_eps):
        unitptb = batch_multiply(1. / (prev_eps + 1e-12), (xadv - x))
        logit_margin = self.search_loss_fn(self.predict(xadv), y)

        maxeps = self.maxeps * torch.ones_like(y).float()

        curr_eps = bisection_search(
            prev_eps, unitptb, self.predict, x, y, self.search_loss_fn,
            logit_margin, maxeps, self.num_search_steps)
        return unitptb, curr_eps

    def perturb(self, x, y, prev_eps):

        self.pgdadv.eps = prev_eps
        self.pgdadv.eps_iter = self.scale_eps_iter(
            self.pgdadv.eps, self.pgdadv.nb_iter)
        with ctx_noparamgrad_and_eval(self.predict):
            xadv = self.pgdadv.perturb(x, y)

        unitptb, curr_eps = self._get_unitptb_and_eps(xadv, x, y, prev_eps)

        xadv = x + batch_multiply(curr_eps, unitptb)
        return xadv, curr_eps

    def scale_eps_iter(self, eps, nb_iter):
        return self.eps_iter_scale * eps / nb_iter


def get_adversaries(cfg, model):

    attack_class = LinfPGDAttack

    train_adv_loss_fn = get_sum_loss_fn(cfg.attack_loss_fn)

    pgdadv = attack_class(
        model, loss_fn=train_adv_loss_fn,
        eps=0.,  # will be set inside ANPGD
        nb_iter=cfg.nb_iter,
        eps_iter=0.,  # will be set inside ANPGD
        rand_init=True,
        clip_min=cfg.clip_min, clip_max=cfg.clip_max,
    )

    cfg.attack_maxeps = cfg.hinge_maxeps * 1.05

    train_adversary = ANPGD(
        pgdadv=pgdadv,
        mineps=cfg.attack_mineps,
        maxeps=cfg.attack_maxeps,
        num_search_steps=cfg.num_search_steps,
        eps_iter_scale=cfg.eps_iter_scale,
        search_loss_fn=get_none_loss_fn(cfg.search_loss_fn),
    )

    return train_adversary