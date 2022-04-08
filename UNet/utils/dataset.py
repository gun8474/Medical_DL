import os
import numpy as np
import torch
import torch.nn as nn
from os.path import splitext
from os import listdir
from torch.utils.data import Dataset
from PIL import Image
from skimage.transform import resize

## CustomDataset
class UDataset(Dataset):
    def __init__(self, data_dir, mode = 'train', transform = None):
        self.data_dir = data_dir
        self.transform = transform

        if mode == 'train':
            self.images_dir = os.path.join(data_dir, 'TrainDataset', 'TrainDataset', 'image')
            self.masks_dir = os.path.join(data_dir, 'TrainDataset', 'TrainDataset', 'mask')

        else:
            self.images_dir = os.path.join(data_dir, 'TestDataset', 'TestDataset', 'CVC-ClinicDB', 'images')
            self.masks_dir = os.path.join(data_dir, 'TestDataset', 'TestDataset', 'CVC-ClinicDB', 'masks')

        self.data_name = os.listdir(self.images_dir)


    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, index):
        image = np.array(Image.open(os.path.join(self.images_dir, self.data_name[index])))
        mask = np.array(Image.open(os.path.join(self.masks_dir, self.data_name[index])))

        if np.array(mask).ndim == 2:
            mask = mask[:,:,np.newaxis]
        if np.array(image).ndim == 2:
            image = image[:,:,np.newaxis]

        if mask.shape[2] == 3:
            mask = mask[:,:,0]
            mask = mask[:, :, np.newaxis]

        data = {'image' : image, 'mask' : mask}

        if self.transform:
            data = self.transform(data)

        return data

class ReSize(object): # Resize 이후 Loss가 큰 음수에서 소수점의 양수로 바뀜 (이미지와 마스크간의 스케일 문제로 로스가 이상한것 같음)
    def __call__(self, data):
        mask, image = data['mask'], data['image']

        mask = resize(mask, (288, 384))
        image = resize(image, (288, 384))

        data = {'image' : image, 'mask' : mask}

        return data

class ToTensor(object):
    def __call__(self, data):
        mask, image = data['mask'], data['image']

        mask = mask.transpose((2,0,1)).astype(np.float32)
        image = image.transpose((2,0,1)).astype(np.float32)

        data = {'image' : torch.from_numpy(image), 'mask' : torch.from_numpy(mask)}

        return data

class Normaization(object):
    def __init__(self, mean = 0.5, std = 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image, mask = data['image'], data['mask']
        image = (image - self.mean) / self.std
        data = {'image' : image, 'mask' : mask}

        return data

class RandomFlip(object):
    def __call__(self, data):
        mask, image = data['mask'], data['image']

        if np.random.rand() > 0.5:
            mask = np.fliplr(mask)
            image = np.fliplr(image)

        if np.random.rand() > 0.5:
            mask = np.flipud(mask)
            image = np.flipud(image)

        data = {'mask': mask, 'image': image}

        return data