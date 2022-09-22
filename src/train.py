
from .utils.schedules import *
import torchvision
from .utils.utils import *
import importlib
# from .utils.anpgd import get_adversaries
from .utils.mma_trainer import MMATrainer
from .utils.utils_awp import *
import copy


class Trainer:
    """
    Trainer Class
    Here you can perform a complete training and testing process, and customize your own training parameters
    """

    def __init__(self, args, model, dataloader, logger):
        self.args = args
        self.logger = logger
        self.logger.info(f"Torch: {torch.__version__}")
        self.logger.info(f"Torchvision: {torchvision.__version__}")
        # Initialize model and gpu
        self.model = model

        self.num_steps = 1
        self.es_counter = 0
        self.num_steps_es_counter = 0

        self.dataloader = dataloader
        gpu_list = [int(i) for i in args.gpu.strip().split(",")]
        self.device = torch.device(f"cuda:{gpu_list[0]}"
                                   if torch.cuda.is_available() and self.args.cuda else "cpu")

        torch.cuda.set_device(self.device)
        # if load state dict
        self.load_state_dict()
        self.best_adv_model = {
            "path": "",
            "adv_acc": 0.0,
            "clean_acc": 0.0,
            "model": ""
        }
        self.best_clean_model = {
            "path": "",
            "adv_acc": 0.0,
            "clean_acc": 0.0,
            "model": ""
        }

        self.msg = {"train adv loss": [], "train nat loss": [], "train kl loss": [],
                    "train nat acc": [], "train adv acc": [],
                    "test nat loss": [], "test adv loss": [],
                    "test nat acc": [], "test adv acc": []}

        # load model to cuda
        if self.args.gpu_parallel and len(gpu_list) > 1 and self.args.cuda:
            logger.info(f"Use {len(gpu_list)} GPUs!")
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=gpu_list)
        self.model = self.model.to(self.device)

        # Loss function
        self.criterion = {"val": nn.CrossEntropyLoss(), "test": nn.CrossEntropyLoss()}

        # optimizer
        self.optimizer = get_optimizer(model, self.args.optimizer, self.args.lr)
        self.scheduler = get_scheduler(self.optimizer, self.args)
        logger.info([self.optimizer, self.scheduler])

        # train and val function
        self.trainer = importlib.import_module(f"src.trainer.{self.args.train_method}").train
        self.eval = importlib.import_module(f"src.eval.{self.args.test_method}").test

        self.logger.info(f"Train set: {len(self.dataloader['train']) * int(self.args.batch_size)}")
        self.logger.info(f"Validation set: {len(self.dataloader['val']) * int(self.args.batch_size)}")

        # mma training
        if self.args.train_method == 'mma':
            target = self.dataloader["train"].dataset.y
            self.dataloader["train"].dataset.y = range(len(target))
            self.dataloader["train"].target = torch.tensor(target)

            train_adversary = get_adversaries(self.args, self.model)
            clean_loss_fn = get_mean_loss_fn(args.clean_loss_fn)
            margin_loss_fn = get_none_loss_fn(args.margin_loss_fn)
            self.mma_trainer = MMATrainer(
                self.model, self.device, clean_loss_fn, self.optimizer,
                margin_loss_fn,
                hinge_maxeps=args.hinge_maxeps,
                clean_loss_coeff=args.clean_loss_coeff,
                adversary=train_adversary,
            )
        # awp method
        elif self.args.train_method == 'awp':
            self.proxy_model = copy.deepcopy(self.model)
            self.proxy_model = self.proxy_model.to(self.device)
            self.proxy_optimizer = get_optimizer(self.proxy_model, self.args.optimizer, self.args.lr)
            self.awp_adversary = TradesAWP(model=self.model, proxy=self.proxy_model,
                                           proxy_optim=self.proxy_optimizer, gamma=self.args.awp_gamma)
        elif self.args.train_method == 'cap':
            self.proxy_model = copy.deepcopy(self.model)
            self.proxy_model = self.proxy_model.to(self.device)
            self.proxy_optimizer = get_optimizer(self.proxy_model, self.args.optimizer, self.args.lr)
            self.cap_adversary = CAP(model=self.model, proxy=self.proxy_model, num_classes=args.num_classes,
                                          proxy_optim=self.proxy_optimizer, gamma=self.args.awp_gamma)
        elif self.args.train_method == 'hat':
            self.hr_model = copy.deepcopy(self.model)
            if self.args.helper_model is None:
                raise ValueError("The helper_model path is not specified!!!")
            logger.info(f"=> loading source model from {self.args.helper_model}")
            checkpoint = torch.load(self.args.helper_model, map_location='cpu')
            self.hr_model.load_state_dict(checkpoint["state_dict"], False)
            logger.info("=> loaded helper model checkpoint successfully")
            self.hr_model = self.hr_model.to(self.device)
        self.AttackPolicy = importlib.import_module("src.utils.schedules").AttackPolicy(self.args)

    def training(self):
        """
        Perform a complete training process
        """
        init_time = time.time()
        for epoch in range(self.args.epoch):
            epoch_init_time = time.time()
            # 1. Train a model
            train_acc, train_loss, val_acc, val_loss, adv_acc, adv_loss = self.train_a_epoch(epoch)

            # 2. print info
            self.logger.info(
                f"Epoch {epoch + 1} | "
                f"Val Loss:{val_loss:.4f} | "
                f"Val Acc:{val_acc:.4f} | "
                f"Adv Loss:{adv_loss:.4f} | "
                f"Adv Acc:{adv_acc:.4f} | "
                f"Best Clean Acc: {self.best_clean_model['clean_acc']} | "
                f"Best Adv Acc: {self.best_adv_model['adv_acc']} | "
                f"Time:{(time.time() - epoch_init_time):.4f} |\n")

            if self.es_counter >= self.args.es_patience and self.args.early_stopping:
                self.logger.info("Early Stopping...")
                break
            self.scheduler.step(adv_acc) if self.args.lr_policy == "plateau" else self.scheduler.step()

        self.logger.info("Test in Other Dataset:")
        # 3. move to latest dir
        clone_results_to_latest_subdir(self.args.base_dir)
        # 4. Print and finish
        time_elapsed = time.time() - init_time
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def train_a_epoch(self, epoch):
        epoch_init_time = time.time()
        self.es_counter += 1
        self.logger.info(f"Learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")

        kwargs = dict(model=self.model, dataloader=self.dataloader["train"], optimizer=self.optimizer,
                      args=self.args, epoch=epoch + 1, device=self.device, logger=self.logger,
                      AttackPolicy=self.AttackPolicy)
        #     step=self.num_steps)
        if self.args.train_method == 'mma':
            kwargs.update(dict(mma_trainer=self.mma_trainer))
        elif self.args.train_method == 'cap':
            kwargs.update(dict(cap_adversary=self.cap_adversary))
        elif self.args.train_method == 'awp':
            kwargs.update(dict(awp_adversary=self.awp_adversary))
        elif self.args.train_method == 'hat':
            kwargs.update(dict(hr_model=self.hr_model))

        self.model, train_loss, nat_loss, kl_loss, train_nat_acc = self.trainer(**kwargs)

        # Evaluate the model
        with torch.no_grad():
            self.model, val_acc, val_loss, adv_acc, adv_loss = self.eval(model=self.model,
                                                                         criterion=self.criterion["val"],
                                                                         dataloader=self.dataloader["val"],
                                                                         opt=self.args,
                                                                         device=self.device,
                                                                         logger=self.logger,
                                                                         )
        # Save the model
        # save best model for adv img
        if adv_acc > self.best_adv_model["adv_acc"]:
            self.best_adv_model["path"] = self.save_model(val_acc,
                                                          adv_acc,
                                                          f"best_adv_model.pth.tar")
            self.best_adv_model["adv_acc"] = adv_acc
            self.best_adv_model["clean_acc"] = val_acc
            self.best_adv_model["model"] = self.model
            self.logger.info(f"Save to {self.best_adv_model['path']}")
            self.es_counter = 0

        # save best model for clean img
        if val_acc > self.best_clean_model["clean_acc"]:
            self.best_clean_model["path"] = self.save_model(val_acc,
                                                            adv_acc,
                                                            f"best_clean_model.pth.tar")
            self.best_clean_model["adv_acc"] = adv_acc
            self.best_clean_model["clean_acc"] = val_acc
            self.best_clean_model["model"] = self.model
            self.logger.info(f"Save to {self.best_clean_model['path']}")
            self.es_counter = 0

        # print info
        self.logger.info(
            f"Epoch {epoch + 1} | "
            f"Val Loss:{val_loss:.4f} | "
            f"Val Acc:{val_acc:.4f} | "
            f"Adv Loss:{adv_loss:.4f} | "
            f"Adv Acc:{adv_acc:.4f} | "
            f"Best Clean Acc: {self.best_clean_model['clean_acc']} | "
            f"Best Adv Acc: {self.best_adv_model['adv_acc']} | "
            f"Time:{(time.time() - epoch_init_time):.4f} |\n")

        return train_nat_acc, train_loss, val_acc, val_loss, adv_acc, adv_loss

    def test(self, mode):
        if mode == "adv":
            model = self.best_adv_model["model"]
        elif mode == "clean":
            model = self.best_clean_model["model"]
        else:
            model = self.model

        self.model, test_acc, test_loss, adv_acc, adv_loss = self.eval(model=model,
                                                                       criterion=self.criterion["test"],
                                                                       dataloader=self.dataloader["test"],
                                                                       opt=self.args,
                                                                       device=self.device,
                                                                       logger=self.logger)
        self.logger.info(f"Val {mode} | Clean Acc:{test_acc:.4f} | Clean Loss:{test_loss:.4f}"
                         f" | Adv Acc:{adv_acc:.4f} | Adv Loss:{adv_loss:.4f}")
        return test_acc, adv_acc

    def save_model(self, val_acc, adv_acc, name):
        if self.args.gpu_parallel and torch.cuda.device_count() > 1 and self.args.cuda:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        state = {
            "state_dict": state_dict,
            "clean_acc": val_acc,
            "adv_acc": adv_acc,
            "optimizer": self.optimizer.state_dict()
        }
        # Save the best model
        path = os.path.join(self.args.base_dir, name)
        torch.save(state, path)
        return path

    def load_state_dict(self):
        if self.args.source_net != "":
            if os.path.isfile(self.args.source_net):
                self.logger.info("=> loading source model from '{}'".format(self.args.source_net))
                checkpoint = torch.load(self.args.source_net, map_location=self.device)
                self.model.load_state_dict(checkpoint["state_dict"], False)
                self.logger.info("=> loaded checkpoint successfully")
            else:
                self.logger.info("=> no checkpoint found at '{}'".format(self.args.source_net))
