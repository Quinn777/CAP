# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# Copied from https://github.com/facebookresearch/deit/blob/main/models.py
import torch
import torch.nn as nn
from functools import partial
from models.vit.vit import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DistilledVisionTransformer(VisionTransformer):
    """ Vision Transformer with distillation token.
    Paper: `Training data-efficient image transformers & distillation through attention` -
        https://arxiv.org/abs/2012.12877
    This impl of distilled ViT is taken from https://github.com/facebookresearch/deit
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = self.linear_layer(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


def deit_t_distilled(cl, ll, num_classes, data_name, pretrained):
    model = DistilledVisionTransformer(
        conv_layer=cl, linear_layer=ll, num_classes=num_classes,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), )
    model.default_cfg = _cfg()
    pretrained_model = ""
    if pretrained:
        if data_name == "SARS_COV_2":
            pretrained_model = "weights/best_clean_model_pretrain_sars_deit.pth.tar"
        elif data_name == "MosMed_L":
            pretrained_model = "weights/best_clean_model_pretrain_mosmed_deit.pth.tar"

    ckpt = torch.load(
        pretrained_model,
        map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k in model_dict:
            if np.shape(model_dict[k]) == np.shape(v):
                pretrained_dict[k] = v
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model
