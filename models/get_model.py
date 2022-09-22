from .vit.visformer import visformer_tiny
from .vit.deit import deit_t_distilled
import torch.nn as nn


def get_model(args):
    model = ""
    if args.model_name == "visformer_t":
        model = visformer_tiny(nn.Conv2d, nn.Linear, args.input_size, args.num_classes, args.data_name, args.pretrained)
    elif args.model_name == "deit_t_distilled":
        model = deit_t_distilled(nn.Conv2d, nn.Linear, args.num_classes, args.data_name, args.pretrained)

    return model
