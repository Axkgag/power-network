import numpy as np
from PIL import Image
import cv2

from torch.utils.data import Dataset

import os
import torch

from .aug import *
from . import functional_pil as F_pil

np.random.seed(666)

class CompDataset(Dataset):
    def __init__(self, path, config, train=True, transform=None):
        print("path:",path)
        self.annotations = np.loadtxt(path, dtype=str, encoding='utf-8', comments=None)
        self.img_names = self.annotations[:, 0]
        self.labels = self.annotations[:, 1]
        self.transform = transform
        self.data_folder=os.path.dirname(path)
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder,self.img_names[idx] )
        label = int(self.labels[idx])

        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label