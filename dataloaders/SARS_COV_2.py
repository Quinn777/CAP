import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pandas as pd
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import random
from PIL import Image
import numpy as np


class PN_Aug:
    def __init__(self, img1, img2, t=0.003, kernel=5):
        self.img2 = img2  # contrast sample
        self.img1 = img1  # original sample
        self.t = t
        self.kernel = kernel

    @staticmethod
    def guideFilter(I, p, kernel, t, s):
        I = np.asarray(I) / 255.0
        p = np.asarray(p) / 255.0
        winSize = [kernel, kernel]
        h, w = I.shape[:2]

        size = (int(round(w * s)), int(round(h * s)))

        small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
        small_p = cv2.resize(p, size, interpolation=cv2.INTER_CUBIC)

        X = winSize[0]
        small_winSize = (int(round(X * s)), int(round(X * s)))

        mean_small_I = cv2.blur(small_I, small_winSize)

        mean_small_p = cv2.blur(small_p, small_winSize)

        mean_small_II = cv2.blur(small_I * small_I, small_winSize)

        mean_small_Ip = cv2.blur(small_I * small_p, small_winSize)

        var_small_I = mean_small_II - mean_small_I * mean_small_I
        cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p

        small_a = cov_small_Ip / (var_small_I + t)
        small_b = mean_small_p - small_a * mean_small_I

        mean_small_a = cv2.blur(small_a, small_winSize)
        mean_small_b = cv2.blur(small_b, small_winSize)

        size1 = (w, h)
        mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)

        q = mean_a * I + mean_b
        gf = q * 255
        gf[gf > 255] = 255
        gf = np.round(gf)
        gf = gf.astype(np.uint8)
        return gf

    def masked(self, img1, img2, ):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        mask = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)[1]
        mask_fg = cv2.bitwise_not(mask)
        mask_bg = mask

        img1_bg = cv2.bitwise_and(img1, img1, mask=mask_bg)
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask_fg)

        img2_fg = self.guideFilter(img1, img2_fg, self.kernel, t=self.t, s=0.5)
        dst = cv2.add(img1_bg, img2_fg)

        final_img = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
        return final_img

    def get_aug(self):
        image_dst = self.masked(np.asarray(self.img1.copy()), np.asarray(self.img1.copy()))
        return image_dst


class SARS_COV_2_Folder(Dataset):
    def __init__(self, data_dir, partition, aug="", t=0.003, kernel=5):
        self.aug = aug
        self.partition = partition
        self.t = t
        self.kernel = kernel
        self.base_dir = os.path.join(data_dir, "SARS-COV-2")
        self.df = pd.read_csv(os.path.join(self.base_dir, "MetaInfo.csv"))
        x_train, x_test, y_train, y_test = train_test_split(self.df["path"],
                                                            self.df["label"],
                                                            test_size=0.2,
                                                            stratify=self.df["label"],
                                                            random_state=1234)

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
        # replace "\" to "/" and convert to RGB
        x = Image.open(os.path.join(self.base_dir, self.x[index]).replace("\\", "/")).convert('RGB')

        if self.partition == "train":
            if self.aug == "sgf":
                if int(self.y[index]) == 0:
                    while True:
                        aug_index = random.choice(range(len(self.y)))
                        if int(self.y[aug_index]) != int(self.y[index]):
                            break
                else:
                    aug_index = index

                aug_x = Image.open(os.path.join(self.base_dir, self.x[aug_index]).replace("\\", "/")).convert('RGB')
                x, aug_x = self.pre_transform(x), self.pre_transform(aug_x)
                pn_aug = PN_Aug(x, aug_x, self.t, self.kernel)
                aug_x = pn_aug.get_aug()
                x, aug_x = self.post_transform(x), self.post_transform(aug_x)
                aug_y = torch.tensor(int(self.y[aug_index]))
                y = torch.tensor(int(self.y[index]))
                x = [x, aug_x]
                y = [y, aug_y]
            else:
                x = self.pre_transform(x)
                x = self.post_transform(x)
                y = torch.tensor(int(self.y[index]))
                x = x
                y = y

        else:
            x = self.pre_transform(x)
            x = self.post_transform(x)
            y = torch.tensor(int(self.y[index]))
        return x, y


class SARS_COV_2:
    def __init__(self, data_dir, batch_size, *args):
        self.batch_size = batch_size
        self.train_data = SARS_COV_2_Folder(data_dir, "train", args[0], args[1], args[2])
        self.val_data = SARS_COV_2_Folder(data_dir, "val", args[0], args[1], args[2])

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
