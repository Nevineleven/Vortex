import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim


class GelDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        #super(GelDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        if self.images[index].endswith('.jpg'):
            img_path = os.path.join(self.image_dir, self.images[index])
            mask_path = os.path.join(self.mask_dir, self.images[index])
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))
            mask[mask==255] = 1

            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations['image']
                mask = augmentations['mask']
            return image, mask