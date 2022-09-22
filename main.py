
from src.train import Trainer
from src.utils.logger import Logger
from src.utils.utils import *
from models.get_model import get_model
import dataloaders
import argparse
import copy
parser = argparse.ArgumentParser(description='PyTorch deep learning framework')
parser.add_argument('--exp_name', default='Test', type=str, help='name of experiment')
parser.add_argument('--config', default='', type=str, help='config file')
parser.add_argument("--init_type", default="kaiming_normal",
                    choices=("kaiming_normal", "kaiming_uniform", "signed_const"),
                    help="Which init to use for weight parameters: kaiming_normal | kaiming_uniform | signed_const", )
parser.add_argument("--print-freq", type=int, default=100, help="frequency for printing training logs")
# model
parser.add_argument('--pretrained', type=bool, default=True, help='use specified model')
parser.add_argument('--use_source_net', type=bool, default=False, help='use specified model')
parser.add_argument('--source_net', type=str, default="", help='specified model path')
parser.add_argument('--base_dir', type=str, default='/outputs', help='project outputs dir')
parser.add_argument('--model_name', type=str, default='visformer_t', help='visformer_t, deit_t_distilled')

# Regularization parameters
parser.add_argument('--t', default=0.003, type=float, help='t in guided filter, default 0.00')
parser.add_argument('--kernel', default=5, type=int, help='kernel size of guided filter"')

# data
parser.add_argument('--data_name', type=str, default='SARS_COV_2', help="data name")
parser.add_argument('--aug', type=str, default='')
parser.add_argument('--data_dir', type=str, default='/datasets', help='dataset dir')
parser.add_argument('--num_classes', type=int,  default=2, help='number of classes')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training ')
parser.add_argument('--epoch', type=int, default=100, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--input_size', type=int, default=224, help='input image size')

# optimizer and lr
parser.add_argument("--optimizer", type=str, default="sgd", choices=("sgd", "adam", "rmsprop", "adamw", "adamax"))
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--lr_policy', type=str, default='cosineAnn',
                    help='learning rate policy: lambda | step | plateau | cosineAnn | cosineAnnWarm')
# step
parser.add_argument('--early_stopping', type=bool, default=True, help='Early stopping')
parser.add_argument('--es_patience', type=int, default=20, help='Early stopping patience')
parser.add_argument('--decay_step', type=int, default=2, help='lr decay step')
parser.add_argument('--gamma', type=float, default=0.9, help='lr step decay rate')
# lambda
parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--epoch_count', type=int, default=1,
                    help='the starting epoch count, we save the model '
                         'by <epoch_count>,<save_latest_freq>+<epoch_count>...')
# cosine
parser.add_argument('--tmax', type=int, default=100, help=' 1/2 of cosine period')
parser.add_argument('--eta_min', type=int, default=0.000001, help='minimum of lr')
parser.add_argument('--tmult', type=int, default=2, help='rate of change in cosine')
# plateau
parser.add_argument('--patience', type=int, default=20, help='patience')
parser.add_argument('--factor', type=float, default=0.8, help='multi-factor')

# gpu
parser.add_argument("--gpu", type=str, default="0", help="Comma separated list of GPU ids")
parser.add_argument('--gpu_parallel', default=False, type=bool, help='run in parallel with multiple GPUs')
parser.add_argument('--cuda', type=bool, default=True, help='enables CUDA training')
parser.add_argument('--workers', default=4, type=int, help='cpu cores * 2')
parser.add_argument('--pin_memory', default=False, type=bool, help='use pin memory in dataloader')
parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed')

# Adversarial attacks
parser.add_argument("--epsilon", default=0.03, type=float, help="perturbation")
parser.add_argument("--num_steps", default=10, type=int, help="perturb number of steps")
parser.add_argument("--step_size", default=0.00784, type=float, help="perturb step size, 2/255 or 0.003")
parser.add_argument("--clip_min", default=0, type=float, help="perturb step size")
parser.add_argument("--clip_max", default=1.0, type=float, help="perturb step size")
parser.add_argument("--distance", type=str, default="l_inf", choices=("l_inf", "l_2"), help="attack distance metric", )
parser.add_argument("--const_init", action="store_true", default=False,
                    help="use random initialization of epsilon for attacks", )
parser.add_argument('--attack_policy', type=bool, default=False, help='attack warmup')
parser.add_argument("--beta", default=1.0, type=float, help="regularization, i.e., 1/lambda in TRADES", )

parser.add_argument("--evl_epsilon", default=0.03, type=float, help="perturbation")
parser.add_argument("--evl_num_steps", default=10, type=int, help="perturb number of steps")
parser.add_argument("--evl_step_size", default=0.00784, type=float, help="perturb step size, 2/255 or 0.003")
parser.add_argument("--evl_distance", type=str, default="l_inf", choices=("l_inf", "l_2"), help="attack distance metric", )
parser.add_argument("--clip_score", default=0.0, type=float, help="score")


#AVmixup
parser.add_argument("--avmixup_gamma", default=2, type=float, help="adversarial vertex scaling factor")
parser.add_argument('--lambda1', default=1.0, type=float, help="label smoothing parameter")
parser.add_argument('--lambda2', default=0.5, type=float, help="label smoothing parameter")

#MMA
parser.add_argument('--clean_loss_fn', default='xent', type=str, help="xent | slm | lm | cw")
parser.add_argument('--margin_loss_fn', default='xent', type=str, help="xent | slm | lm | cw")
parser.add_argument('--attack_loss_fn', default='slm', type=str, help="xent | slm | lm | cw")
parser.add_argument('--search_loss_fn', default='slm', type=str, help="xent | slm | lm | cw")
parser.add_argument('--clean_loss_coeff', default=1. / 3, type=float)
parser.add_argument('--eps_iter_scale', default=2.5, type=float)
parser.add_argument('--num_search_steps', default=10, type=int)
parser.add_argument('--nb_iter', default=10, type=int)
parser.add_argument('--attack_mineps', default=0.005, type=float)
parser.add_argument('--hinge_maxeps', default=0.1255, type=float)

#AWP
parser.add_argument('--awp_gamma', default=0.0001, type=float, help='whether or not to add parametric noise')
parser.add_argument('--awp_warmup', default=20, type=int, help='We could apply AWP after some epochs for accelerating.')

#HAT
parser.add_argument('--h', default=2.0, type=float, help='Parameter h to compute helper examples (x + h*r) for HAT.')
parser.add_argument('--hat_gamma', default=1.0, type=float, help='Weight of helper loss in HAT.')
parser.add_argument('--helper_model', type=str, default="/best_clean_model.pth.tar", help='Helper model weights file name for HAT.')

#FAT
parser.add_argument('--fat_step_size', type=float, default=0.007, help='fat step size')
parser.add_argument('--tau', type=int, default=0, help='step tau')
parser.add_argument('--omega', type=float, default=0.001, help="random sample parameter for adv data generation")
parser.add_argument('--dynamictau', type=bool, default=False, help='whether to use dynamic tau')

#co_adv
parser.add_argument('--alpha', default=10, type=float, help='Stepsize (ex.12)')
parser.add_argument('--c', default=3, type=int, help='Number of checkpoints')
parser.add_argument('--inf-batch', default=1024, type=int, help='Number of batches during checkpoints inference')

# training and eval method
parser.add_argument("--train_method", type=str, default="CAP", help="Natural (base) or adversarial training", )
parser.add_argument("--test_method", type=str, default="pgd",
                    help="base: evaluation on unmodified inputs | adv: evaluate on adversarial inputs", )

def main():
    # init
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            for k, v in json.load(f).items():
                args.__dict__[k] = v
    args_dict = args.__dict__.copy()
    args = set_output_dir(args)
    logger = Logger(args.base_dir).logger
    logger.info(args)
    save_config(args_dict, args)

    seed_everything(args.seed)

    # model
    model = get_model(args)
    # data
    train_data = dataloaders.__dict__[args.data_name](args.data_dir, args.batch_size, args.aug, args.t, args.kernel)
    train_loader, val_loader = train_data.data_loaders()
    dataloader = {
        "train": train_loader,
        "val": val_loader,}
    # train
    trainer = Trainer(args=args,
                      model=model,
                      logger=logger,
                      dataloader=dataloader, )

    trainer.training()
    logger.info(f"Output directory: {args.base_dir}")


if __name__ == '__main__':
    main()
