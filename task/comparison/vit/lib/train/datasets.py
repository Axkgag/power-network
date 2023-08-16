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

# 原始版本是data_old.py，区别是当前版本对点位是在线提取，方便后续正样本做数据增强。


class CompDataset(Dataset):

    @property
    def train_labels(self):
        return self.labels
    @property
    def test_labels(self):
        return self.labels
    @property
    def train_data(self):
        return self.img_names

    @property
    def test_data(self):
        return self.img_names
    
    @property
    def categories(self):
        return self.category_imgs
    
    @property
    def anchors(self):
        return self.anchor_imgs
    
    def __init__(self, txt_path, config, anchor_path=None, train=True, transform=None):
        not_keep_ratio_part = config.not_keep_ratio_part
        self.data_folder=os.path.dirname(txt_path)
        super(CompDataset, self).__init__()
        self.transform = transform
        self.train = train
        data = np.loadtxt(txt_path, dtype=str, comments=';', encoding='utf-8')
        self.img_names = data[:, 0]
        self.labels = data[:, 1] # 0 或者 1
        self.category_imgs = {} # {pos:{label:[]}}
        self.pos_ids = []
        if train:
            for i in range(len(self.img_names)):
                label = int(self.labels[i])
                pos = self.img_names[i].split('#')[-1].replace('.jpg', '')
                if not pos in self.category_imgs:
                    self.category_imgs[pos]={0:[],1:[]}
                    self.pos_ids.append(pos)
                self.category_imgs[pos][label].append(self.img_names[i]) # 按点位和类别分开图像
        if not train:
            anchor = np.loadtxt(anchor_path, dtype=str, delimiter= "\t",comments=';')
            self.anchor_imgs = {i:[] for i in pos_ids} # 只保存每个点位的正样本
            for i in range(len(anchor[:, 0])):
                pos = anchor[i, 0].split('#')[-1].replace('.jpg', '')
                label = int(anchor[i, 1]) # 获取标签，如果是ok，则添加进anchor_imgs
                if label == 0:
                    # pos = pos.replace('0020', '0017').replace('0021', '0018').replace('0022', '0019')
                    # pos = pos.replace('0003', '0002').replace('0007', '0006').replace('0011', '0010').replace('0013', '0012').replace('0005', '0004')
                    if not pos in self.anchor_imgs:
                        self.anchor_imgs[pos]=[]
                    self.anchor_imgs[pos].append(anchor[i, 0])

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
        self.category_imgs = self.comp_dataset.categories
        self.data_folder=comp_dataset.data_folder
        not_keep_ratio_part = config.not_keep_ratio_part

        self.not_keep_ratio_part=not_keep_ratio_part
        self.pos_ids=self.comp_dataset.pos_ids
        if self.train:
            self.train_labels = self.comp_dataset.train_labels # 标签
            self.train_data = self.comp_dataset.train_data # 文件名
        else:
            self.test_labels = self.comp_dataset.test_labels
            self.test_data = self.comp_dataset.test_data
            self.anchor_imgs = self.comp_dataset.anchors
    
    
    def get_img(self, img1_name, pos):
        img1_path = os.path.join(self.data_folder, img1_name)
        img1 = Image.open(img1_path).convert("RGB")
        return img1

    def __getitem__(self, index):
        if self.train:
            pos = np.random.choice(self.pos_ids)
            while len(self.category_imgs[pos][0]) == 0:
                pos = np.random.choice(self.pos_ids)
            img1_name = np.random.choice(self.category_imgs[pos][0]) # 随机选择点位的一个正样本
            img1 = self.get_img(img1_name, pos)
            target = random_pick([0, 1], [0.4, 0.6]) # 以0.4的概率选择负样本对，以0.6的概率选择正样本对
            if len(self.category_imgs[pos][1]) == 0 and target == 0: target = 1
            
            if target == 1:
                img2_name = np.random.choice(self.category_imgs[pos][0]) # 随机选择点位的一个正样本
                img2 = self.get_img(img2_name, pos)
            else:
                img2_name = np.random.choice(self.category_imgs[pos][1])
                img2 = self.get_img(img2_name, pos)
            
            if pos in self.not_keep_ratio_part:
                img1 = img1.resize((resolution, resolution))
                img2 = img2.resize((resolution, resolution))
            else:
                img1 = maker_border_resize(np.asarray(img1), resolution)
                img2 = maker_border_resize(np.asarray(img2), resolution)
                img1 = Image.fromarray(img1)
                img2 = Image.fromarray(img2)

            img1, img2 = random_flip(img1, img2)
            img1, img2 = random_crop(img1, img2)
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
            img1 = random_aug(img1)
            img2 = random_aug(img2)
        else:
            img2_name = self.test_data[index]
            pos = img2_name.split('#')[-1].replace('.jpg', '')
            img2 = Image.open(os.path.join('.', img2_name)).convert("RGB")
            if pos in self.not_keep_ratio_part:
                img2 = img2.resize((resolution, resolution))
            else:
                img2 = maker_border_resize(np.asarray(img2), resolution)
                img2 = Image.fromarray(img2)
            

            label2 = int(self.test_labels[index]) # 标签名
            target = (label2 == 0) # 是否是正样本

            anchor_list = self.anchor_imgs[pos] 
            
            if len(anchor_list) == 0:
                imgs1 = []
            else:
                np.random.permutation(anchor_list)
                imgs1_name = anchor_list[:5]
                imgs1 = []
                for name in imgs1_name:
                    img1_path = os.path.join('.', name)
                    img1 = Image.open(img1_path).convert("RGB")
                    if pos in self.not_keep_ratio_part:
                        img1 = img1.resize((resolution, resolution))
                    else:
                        img1 = maker_border_resize(np.asarray(img1), resolution)
                        img1 = Image.fromarray(img1)
                    imgs1.append(img1)
        if self.transform is not None:
            img2 = self.transform(img2)
            if self.train:
                img1 = self.transform(img1)
            else:
                for i in range(len(imgs1)):
                    imgs1[i] = self.transform(imgs1[i])
        
        if self.train:
            return (img1, img2), target
        else:
            return (imgs1, img2), target, img2_name, pos

    def __len__(self):
        return len(self.comp_dataset)
