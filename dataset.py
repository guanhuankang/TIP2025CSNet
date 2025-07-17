#!/usr/bin/python3
#coding=utf-8

import os
import logging
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as DA
from torchvision import transforms as pth_transforms
from common import *

class Data(Dataset):
    def __init__(self, cfg, mode):
        logging.info("Dataset initializing")
        
        dset = cfg.trainSet if mode=="train" else cfg.testSet
        datacfg = getattr(loadConfig(cfg.datasetCfgPath), dset)
        logging.info(datacfg)

        self.mode = mode
        self.cfg        = cfg
        self.datacfg = datacfg
        
        self.new_view = DA.Compose([
            DA.ColorJitter(p=1.0),
            DA.RandomBrightnessContrast(p=0.5),
            DA.RGBShift(),
            # DA.Defocus(p=1.0)
        ])
        
        h, w = cfg.size
        self.data_aug = DA.Compose([
            DA.HorizontalFlip(p=0.5),
            DA.RandomResizedCrop((h, w), scale=(0.64, 1.0), p=1.0),
        ], additional_targets={"image2": "image"})
        
        self.img_normalize = lambda x: pth_transforms.Compose([
            pth_transforms.Resize(cfg.size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])(Image.fromarray(x).convert("RGB"))
        
        self.mak_normalize = lambda x: pth_transforms.Compose([
            pth_transforms.Resize(cfg.size),
            pth_transforms.ToTensor()
        ])(Image.fromarray(x).convert("L"))
        
        self.samples = [ x[0:-len(datacfg.image.suffix)] for x in os.listdir(os.path.join(datacfg.root, datacfg.image.path)) if x.endswith(datacfg.image.suffix)]
        
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    def __getitem__(self, idx):
        name  = self.samples[idx]
        image = Image.open(os.path.join(self.datacfg.root, self.datacfg.image.path, name+self.datacfg.image.suffix)).convert("RGB")
        if self.cfg.use_pseudo:
            mask = Image.open(os.path.join(self.datacfg.root, self.datacfg.pseudo.path, name + self.datacfg.pseudo.suffix)).convert("L")
        else:
            mask = Image.open(os.path.join(self.datacfg.root, self.datacfg.mask.path, name + self.datacfg.mask.suffix)).convert("L")
        
        image = np.array(image).astype(np.uint8) ## RGB
        mask = np.array(mask).astype(np.uint8)  ## L

        if self.mode=='train':
            image2 = self.new_view(image=image)["image"]
            aug = self.data_aug(image=image, image2=image2, mask=mask)
            image, image2, mask = aug["image"], aug["image2"], aug["mask"]
            
            image = self.img_normalize(image)
            image2 = self.img_normalize(image2)
            mask = self.mak_normalize(mask)
            
            return image, image2, mask
        else:
            raise
            # shape = image.shape[:2]
            # uint8_img = image.copy()
            # image = self.img_normalize(image)
            # return image, uint8_img, shape, name

    def __len__(self):
        return len(self.samples)

    def collate(self, batch):
        x1, x2, y = [], [], []
        for sample in batch:
            x1.append(sample[0])
            x2.append(sample[1])
            y.append(sample[2])
        images = torch.stack(x1+x2, dim=0)
        masks = torch.stack(y+y, dim=0)
        
        # [224, 280, 336, 392, 448]
        sizes = self.cfg.train_sizes
        size = sizes[np.random.randint(0, len(sizes))]
        images = F.interpolate(images, size=size, mode="bilinear")
        masks = F.interpolate(masks, size=size, mode="bilinear")
        
        return images, masks
        