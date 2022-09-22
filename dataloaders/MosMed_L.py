import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import pandas as pd
import nibabel as nib
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split
import random
from dataloaders.SARS_COV_2 import PN_Aug


class COVID19_5_Folder(Dataset):
    def __init__(self, data_dir, partition, aug):
        self.base_dir = os.path.join(data_dir, "MosMed_L")
        self.df = pd.read_csv(os.path.join(self.base_dir, "MetaInfo.csv"))
        self.partition = partition
        self.aug = aug
        x_train, x_test, y_train, y_test = train_test_split(self.df["path"],
                                                            self.df["infection"],
                                                            test_size=0.2,
                                                            stratify=self.df["infection"],
                                                            random_state=1234)
        self.cls_map = {
            "CT-0": 0,
            "CT-1": 1,
            "CT-2": 2,
            "CT-3": 3,
            "CT-4": 4
        }

        if self.partition == "train":
            self.pre_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ])

            self.post_transform = transforms.ToTensor()
            self.x = x_train.tolist()
            self.y = y_train.tolist()
        else:
            self.pre_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                ])

            self.post_transform = transforms.Compose(
                [
                    transforms.ToTensor()
                ])

            self.x = x_test.tolist()
            self.y = y_test.tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        x = Image.open(os.path.join(self.base_dir, self.x[index]).replace("\\", "/")).convert('RGB')
        y = torch.tensor(int(self.cls_map[self.y[index]]))

        if self.partition == "train":
            if self.aug == "sgf":
                if int(y) == 0:
                    while True:
                        aug_index = random.choice(range(len(self.y)))
                        if self.y[aug_index] != self.y[index]:
                            break
                else:
                    aug_index = index

                aug_x = Image.open(os.path.join(self.base_dir, self.x[aug_index]).replace("\\", "/")).convert('RGB')
                x = self.pre_transform(x)
                pn_aug = PN_Aug(x, aug_x)
                aug_x = pn_aug.get_aug()
                x, aug_x = self.post_transform(x), self.post_transform(aug_x)
                aug_y = torch.tensor(int(self.cls_map[self.y[aug_index]]))
                x = [x, aug_x]
                y = [y, aug_y]
            else:
                x = self.pre_transform(x)
                x = self.post_transform(x)
                y = torch.tensor(int(self.cls_map[self.y[index]]))
        else:
            x = self.pre_transform(x)
            x = self.post_transform(x)
            y = torch.tensor(int(self.cls_map[self.y[index]]))
        return x, y


class MosMed_L:
    def __init__(self, data_dir, batch_size, *args):
        self.batch_size = batch_size
        self.train_data = COVID19_5_Folder(data_dir, "train", args[0])
        self.val_data = COVID19_5_Folder(data_dir, "val", args[0])

    def data_loaders(self):
        train_loader = DataLoader(dataset=self.train_data,
                                  batch_size=self.batch_size,
                                  num_workers=8,
                                  pin_memory=True,
                                  shuffle=True)
        val_loader = DataLoader(dataset=self.val_data,
                                batch_size=self.batch_size,
                                num_workers=8,
                                pin_memory=True,
                                shuffle=True)
        return train_loader, val_loader


def nii2array(path):
    ct_scan = nib.load(path)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return array
