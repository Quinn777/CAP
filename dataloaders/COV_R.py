import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms


class Russia_Folder(Dataset):
    def __init__(self, data_dir):
        self.base_dir = os.path.join(data_dir, "MosMed_L")
        self.df = pd.read_csv(os.path.join(self.base_dir, "Russia.csv"))

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.x = self.df["filename"].tolist()
        self.y = self.df["label"].tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # replace "\" to "/" and convert to RGB
        x = Image.open(os.path.join(self.base_dir, self.x[index]).replace("\\", "/")).convert('RGB')
        y = torch.tensor(int(self.y[index]))

        x = self.transform(x)
        return x, y


class COV_R:
    def __init__(self, data_dir, batch_size):
        self.batch_size = batch_size
        self.data = Russia_Folder(data_dir)

    def data_loaders(self):
        loader = DataLoader(dataset=self.data,
                            batch_size=self.batch_size,
                            num_workers=8,
                            pin_memory=True,
                            shuffle=True)
        return loader
