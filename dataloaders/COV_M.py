import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms


class COV_M_Folder(Dataset):
    def __init__(self, data_dir):
        self.base_dir = os.path.join(data_dir, "COV_M")
        self.df = pd.read_csv(os.path.join(self.base_dir, "MetaInfo.csv"))
        self.x = self.df["path"].tolist()
        self.y = self.df["label"].tolist()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # replace "\" to "/" and convert to RGB
        path = os.path.join(self.base_dir, self.x[index]).replace("\\", "/")
        x = Image.open(path).convert('RGB')
        y = torch.tensor(int(self.y[index]))
        x = self.transform(x)
        return x, y, path


class COV_M:
    def __init__(self, data_dir, batch_size):
        self.batch_size = batch_size
        self.test_data = COV_M_Folder(data_dir)

    def data_loaders(self):
        test_loader = DataLoader(dataset=self.test_data,
                                 batch_size=self.batch_size,
                                 num_workers=8,
                                 pin_memory=True,
                                 shuffle=True
                                 )
        return test_loader
