import os
import os.path
import torch

import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from imagecorruptions import get_corruption_names
from imagecorruptions import corrupt

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Transform_SARS_Corruption():
    def __init__(self):
        self.base_dir = "/share_data/dataset/SARS-COV-2"
        self.target_dir = "/share_data/dataset/SARS_COV_2_Corruption"
        self.df = pd.read_csv(os.path.join(self.base_dir, "MetaInfo.csv"))
        _, self.x_test, _, self.y_test = train_test_split(self.df["path"],
                                                          self.df["label"],
                                                          test_size=0.2,
                                                          stratify=self.df["label"],
                                                          random_state=1234)

    def get_corruption(self):
        for i in tqdm(range(len(self.y_test)), desc='Img'):
            x = np.array(
                Image.open(os.path.join(self.base_dir, self.x_test.tolist()[i]).replace("\\", "/")).convert('RGB'))
            for corruption_name in get_corruption_names():
                for severity in range(5):
                    x_aug = corrupt(x, corruption_name=corruption_name, severity=severity + 1)
                    output_dir = os.path.join(os.path.join(self.target_dir, corruption_name), str(severity))
                    output_dir = os.path.join(output_dir, self.x_test.tolist()[i].replace("\\", "/"))

                    if not os.path.exists("/".join(output_dir.split("/")[:-1])):
                        os.makedirs("/".join(output_dir.split("/")[:-1]))
                    x_aug = Image.fromarray(x_aug)
                    x_aug.save(output_dir)


if __name__ == '__main__':
    trans = Transform_SARS_Corruption()
    trans.get_corruption()
