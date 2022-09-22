
import torch
import os
import numpy as np
import random
import time
import json
from distutils.dir_util import copy_tree
import copy


def seed_everything(seed):
    # For the same set of parameters, ensure the network is the same
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_config(args_dict, args):
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))

    print(args.base_dir)
    if not os.path.exists(args.base_dir):
        raise Exception("Error: Cannot find base_dir to save config!")

    file = os.path.join(args.base_dir, f"config.json")
    with open(file, 'w') as f:
        f.write(json.dumps(args_dict, indent=4))


def get_new_path(dir):
    file_list = os.listdir(dir)
    file_list.sort(key=lambda fn: os.path.getmtime(dir + '\\' + fn))
    filepath = os.path.join(dir, file_list[-1])
    return filepath


def set_output_dir(args):
    args.base_dir = os.path.join(args.base_dir, f'{args.exp_name}/{args.model_name}')
    if not os.path.exists(args.base_dir):
        os.makedirs(args.base_dir, 0o700)
    if not os.listdir(args.base_dir):
        n = 1
    else:
        n = len(next(os.walk(args.base_dir))[1]) + 1
    # if args.exp_mode == "pretrain":
    args.base_dir = os.path.join(args.base_dir,
                                 f'{n}-'
                                 f'{args.train_method}-'
                                 f'{args.test_method}-'
                                 f'{args.beta}-'
                                 f'{args.optimizer}-'
                                 f'{args.lr}-'
                                 f'{args.lr_policy}')

    if not os.path.exists(args.base_dir):
        os.makedirs(args.base_dir, 0o700)
    return args


def print_state_dict(state_dict):
    for param_tensor in state_dict:
        print(param_tensor, '\t', state_dict[param_tensor].size())


def cleanup_state_dict(state_dict):
    clean_state_dict = {}
    for name, value in state_dict.items():
        if "module." in name:
            new_name = name[7:]
        else:
            new_name = name
        clean_state_dict[new_name] = value
    return clean_state_dict


def clone_results_to_latest_subdir(base_dir):
    src = copy.deepcopy(base_dir)
    dst = os.path.join("/".join(base_dir.split("/")[0:-1]), "latest_exp")
    if not os.path.exists(dst):
        os.mkdir(dst)
    copy_tree(src, dst)


