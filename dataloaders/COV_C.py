
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import pandas as pd
import nibabel as nib
from torchvision.transforms import autoaugment, transforms
from torchtoolbox.transform import Cutout


class COVID_CT_China_Folder(Dataset):
    def __init__(self, data_dir):
        self.base_dir = os.path.join(data_dir, "COV_C")
        self.df = pd.read_csv(os.path.join(self.base_dir, "MetaInfo.csv"))
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.x = self.df["path"].tolist()
        self.y = self.df["label"].tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # replace "\" to "/" and convert to RGB
        x = Image.open(os.path.join(self.base_dir, self.x[index]).replace("\\", "/")).convert('RGB')
        y = torch.tensor(int(self.y[index]))

        x = self.transform(x)
        return x, y


class COV_C:
    def __init__(self, data_dir, batch_size):
        self.batch_size = batch_size
        self.data = COVID_CT_China_Folder(data_dir)

    def data_loaders(self):
        loader = DataLoader(dataset=self.data,
                            batch_size=self.batch_size,
                            num_workers=8,
                            pin_memory=True,
                            shuffle=True)
        return loader
