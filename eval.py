
import logging
from src.utils.metrics import *
import argparse
from models.get_model import get_model
from src.utils.utils import seed_everything
import torch.nn as nn
import dataloaders
import importlib
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from imagecorruptions import get_corruption_names
import json

def get_sub_map(model, x):
    attn_layer = list(model.children())[2][0].blocks[0].attn.qkv
    features = []

    def _hook(module, input, output):
        """hook function for getting features of input from model"""
        features.append(output)

    attn_layer.register_forward_hook(_hook)
    out = model(x)
    qkv = features[0].reshape(1536, 49, 3, 3, 96 // 3).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * 32 ** -0.5
    attn = (q @ k.transpose(-2, -1)).cpu().detach().numpy()
    return attn



def process_muti_data_test(model, criterion, args, device, logger):
    sars_data = dataloaders.__dict__["SARS_COV_2"](args.base_dir, 24, 0, 0, 0)
    sars_train, sars_test = sars_data.data_loaders()
    ood_data = dataloaders.__dict__["COV_M"](args.base_dir, batch_size=24)
    ood_test = ood_data.data_loaders()

    cn_data = dataloaders.__dict__["COV_C"](args.base_dir, batch_size=24)
    cn_test = cn_data.data_loaders()

    Russia_data = dataloaders.__dict__["COV_R"](args.base_dir, batch_size=24)
    Russia_test = Russia_data.data_loaders()
    Iran_data = dataloaders.__dict__["COV_I"](args.base_dir, batch_size=24)
    Iran_test = Iran_data.data_loaders()

    eval = importlib.import_module(f"src.eval.{args.test_method}").test

    logger.info("--------------------SARS_COV_2--------------------")
    eval(model, criterion, sars_test, args, device, logger)
    logger.info("--------------------COV_M--------------------")
    eval(model, criterion, ood_test, args, device, logger)
    logger.info("--------------------COV_R--------------------")
    eval(model, criterion, Russia_test, args, device, logger)
    logger.info("--------------------COV_I--------------------")
    eval(model, criterion, Iran_test, args, device, logger)
    logger.info("--------------------COV_C---------------------")
    eval(model, criterion, cn_test, args, device, logger)
    # logger.info("--------------------Unknown---------------------")
    # eval(model, criterion, unknown_test, args, device, logger)


def process_cov5_data_test(model, criterion, args, device, logger):
    cov5_data = dataloaders.__dict__["MosMed_L"](args.base_dir, 64, 0)
    _, cov5_test = cov5_data.data_loaders()
    med_data = dataloaders.__dict__["MosMed_M"](args.base_dir, 64, 0)
    _, med_test = med_data.data_loaders()


    eval = importlib.import_module(f"src.eval.{args.test_method}").test

    logger.info("--------------------MosMed-L--------------------")
    eval(model, criterion, cov5_test, args, device, logger)
    logger.info("--------------------MosMed-M--------------------")
    eval(model, criterion, med_test, args, device, logger)


def process_muti_attack_test(model, criterion, args, device, logger):
    sars_data = dataloaders.__dict__["SARS_COV_2"](args.base_dir, 24, 0, 0, 0)
    sars_train, sars_test = sars_data.data_loaders()
    for attack in ['fgsm', 'rfgsm', 'pgd', 'mim', 'autoattack']:
        eval = importlib.import_module(f"src.eval.{attack}").test
        logger.info(f"-------------------{attack}---------------------")
        eval(model, criterion, sars_test, args, device, logger)


def process_sars_corruption_test(model, criterion, args, device, logger):
    results = {}
    for corruption_name in get_corruption_names():
        results[corruption_name] = {"acc": [], "adv_acc": []}
        for level in range(5):
            sars_data = dataloaders.__dict__["SARS_Corruption"](args.base_dir, 24, corruption_name, level)
            sars_test = sars_data.data_loaders()
            eval = importlib.import_module(f"src.eval.{args.test_method}").test

            logger.info(f"--------------------{corruption_name}: {level}--------------------")
            _, acc, _, adv_acc, _ = eval(model, criterion, sars_test, args, device, logger)
            results[corruption_name]["acc"].append(acc)
            results[corruption_name]["adv_acc"].append(adv_acc)

    logger.info(f"-------------------------------------------------------------------------------------------")
    for corruption_name in results.keys():
        results[corruption_name]["avg"] = sum(results[corruption_name]["acc"])/len(results[corruption_name]["acc"])
        results[corruption_name]["adv_avg"] = sum(results[corruption_name]["adv_acc"])/len(results[corruption_name]["adv_acc"])
    print(results)




if __name__ == '__main__':
    seed_everything(1234)

    parser = argparse.ArgumentParser(description='Test my model')
    parser.add_argument('--base_dir', default="/share_data/dataset", type=str)
    parser.add_argument('--model_name', default="visformer_t", type=str)
    parser.add_argument('--model_path',
                        default="/share_data/xiangkun/Soraka/SARS_COV_2/visformer_t/pretrain/146-awpv2_smooth_score-pgd-3.0-1.0-adamw-0.0005-cosineAnn/best_adv_model_pretrain.pth.tar",
                        type=str)
    parser.add_argument("--evl_epsilon", default=0.03, type=float, help="perturbation")
    parser.add_argument("--evl_num_steps", default=10,  type=int, help="perturb number of steps")
    parser.add_argument("--evl_step_size", default=0.00784, type=float, help="perturb step size, 2/255 or 0.003")
    parser.add_argument("--clip_min", default=0, type=float, help="perturb step size")
    parser.add_argument("--clip_max", default=1.0, type=float, help="perturb step size")
    parser.add_argument("--distance", type=str, default="l_inf", choices=("l_inf", "l_2"),
                        help="attack distance metric", )

    parser.add_argument("--const_init", action="store_true", default=False,
                        help="use random initialization of epsilon for attacks", )
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--test_method", type=str, default="pgd")
    parser.add_argument('--pretrained', type=bool, default=False, help='use specified model')
    parser.add_argument('--input_size', type=int, default=224, help='input image size')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    model = get_model(args)
    # set_prune_rate_model(model, 1)
    device = torch.device("cuda:0")
    logger.info(f"=> loading source model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], False)
    logger.info("=> loaded checkpoint successfully")

    torch.cuda.set_device(device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    process_muti_data_test(model, criterion, args, device, logger)

    process_muti_attack_test(model, criterion, args, device, logger)