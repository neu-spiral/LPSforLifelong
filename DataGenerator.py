import numpy as np
import scipy.io as spio
import random
import pickle
import math
import os
import timeit
import collections
from multiprocessing import Pool, cpu_count

from torch.utils.data import Dataset
from PIL import Image

class CifarDataGenerator(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        
        
    def __len__(self):
        """Denotes the total number of examples, later index will be sampled according to this number"""
        return int(len(self.x))
    
    def __getitem__(self, index):
        
        img, target = self.x[index,:,:,:], self.y[index]

        img = img.transpose((2, 0, 1))
        target = target[0]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        return img, target

class MnistDataGenerator(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        
        
    def __len__(self):
        """Denotes the total number of examples, later index will be sampled according to this number"""
        return int(len(self.x))
    
    def __getitem__(self, index):
        
        img, target = self.x[index,:], self.y[index]

        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        return img, target

class MixtureDataGenerator(Dataset):
    def __init__(self, x, y, transform=None, scale=1, trigger=False):
        self.x = x
        self.y = y
        self.transform = transform
        self.scale = scale
        self.trigger = trigger
        
    def __len__(self):
        """Denotes the total number of examples, later index will be sampled according to this number"""
        return int(len(self.x)*self.scale)
    
    def __getitem__(self, index):
        
        img, target = self.x[index,:,:,:], self.y[index]

        img = img.transpose((2, 0, 1))
        
        if not self.trigger:
            target = target[0] # for cifar
        
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        return img, target
