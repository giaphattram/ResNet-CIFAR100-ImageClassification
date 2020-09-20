# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 01:54:34 2020

@author: Admin
"""
from config import parent_directory
import os
os.chdir(r"C:\MountedDrive\Projects\Resnet CIFAR100")
import torch
import torch.utils.data as data
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import json

class CIFAR100Dataset(data.Dataset):
    def __init__(self, transform, mode, batch_size, data_folder = './data'):
        self.transform, self.mode, self.batch_size = transform, mode, batch_size
        if self.mode == 'train':
            annotation_path = os.path.join(data_folder, "train.csv")
        else:
            annotation_path = os.path.join(data_folder, "test.csv")
            
        annotations = pd.read_csv(annotation_path)
        class_to_int = pd.read_csv(os.path.join(data_folder, "class_to_int.csv"))
        superclass_to_int = pd.read_csv(os.path.join(data_folder, "superclass_to_int.csv"))
        
        
    
    def __getitem__(self, index):
        """
          Input: index
          Output: 3-tuple (image, class, superclass)
        """
    def __len__(self):
        return len(self.images)

    def get_train_indices(self):
        return np.random.randint(low = 0, high = len(self.images), size = self.batch_size)
    
