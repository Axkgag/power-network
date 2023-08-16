import numpy as np
from PIL import Image
import cv2

from torch.utils.data import Dataset

import os
import torch

from .aug import *
from . import functional_pil as F_pil

np.random.seed(666)
resolution = 256

class CompDataset(Dataset):

    @property
    def get_labels(self):
        return self.labels

    @property
    def get_data(self):
        return self.img_names
    
    def __init__(self, path, config, train=True, transform=None):
        super(CompDataset, self).__init__()
        self.transform = transform
        self.train = train
        data = np.loadtxt(path, dtype=str, encoding='utf-8', comments=';')
        self.img_names = data[:, 0]
        self.labels = data[:, 1]
        self.data_folder=os.path.dirname(path)
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.img_names)


class SiameseComp(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, comp_dataset,config):
        self.comp_dataset = comp_dataset

        self.train = self.comp_dataset.train
        self.transform = self.comp_dataset.transform
        
        self.data = self.comp_dataset.get_data
        self.labels = self.comp_dataset.get_labels
    
    
    def get_img(self, img1_name):
        img1_path = os.path.join(self.comp_dataset.data_folder, img1_name)
        img1 = Image.open(img1_path).convert("RGB")
        return img1

    def __getitem__(self, index):
        img2_name = self.data[index]
        pos = img2_name.split('#')[-1].replace('.jpg', '')
        img2 = self.get_img(img2_name)
        img2 = maker_border_resize(np.asarray(img2), resolution)
        img2 = Image.fromarray(img2)

        if self.train:
            img2 = random_flip(img2)
            img2 = random_crop(img2)

            def random_aug(img):
                aug_list = ["adjust_gamma", "gaussian_blur", "adjust_sharpness"]
                aug_method = np.random.choice(aug_list)
                if aug_method == "adjust_gamma":
                    factor = np.random.choice(np.linspace(0.85, 1.15, 20))
                    img = F_pil.adjust_gamma(img, gamma=factor)
                elif aug_method == "gaussian_blur":
                    kernel_size = 3
                    sigma = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                    img = F_pil.gaussian_blur(img, kernel_size=kernel_size, sigma=(sigma, sigma))
                elif aug_method == "adjust_sharpness":
                    factor = np.random.choice(np.linspace(0.1, 2.0, 20))
                    img = F_pil.adjust_sharpness(img, sharpness_factor=0.1)
                return img
            img2 = random_aug(img2)            

        label2 = int(self.labels[index])

            
        if self.transform is not None:
            img2 = self.transform(img2)
        
        if self.train:
            return img2, label2, img2_name, pos
        
    def __len__(self):
        return len(self.comp_dataset)
