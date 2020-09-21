# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 01:54:34 2020

@author: Admin
"""
# from config import parent_directory
import os
os.chdir(r"C:\MountedDrive\Projects\Resnet CIFAR100")
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import numpy as np
from CIFAR import CIFAR

def get_loader(transform = None, mode = 'train', batch_size = 1, data_folder = './data', num_workers = 0):
    """
        Returns the data loader
    """
    assert mode in ['train', 'test']
    
    if mode == 'test':
        assert batch_size == 1
        
    
    dataset = CIFAR100Dataset(mode = mode, batch_size = batch_size, transform = transform, data_folder = data_folder)
    if mode == 'train':
        # create a batch sampler
        initial_sampler = data.sampler.SubsetRandomSampler(indices = dataset.get_train_indices())
        # data loader for dataset
        data_loader = data.DataLoader(dataset = dataset, num_workers = num_workers, 
                                      batch_sampler = data.sampler.BatchSampler(sampler = initial_sampler,
                                                                                batch_size = dataset.batch_size,
                                                                                drop_last = False))
    else:
        data_loader = data.DataLoader(dataset = dataset, batch_size = dataset.batch_size, shuffle = True, num_workers = num_workers)
        
    return data_loader

class CIFAR100Dataset(data.Dataset):
    def __init__(self, mode, batch_size = 128, transform = None, data_folder = './data'):
        assert mode in ['train', 'test']
        if transform is None:
            if mode == 'train':
                self.transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406),      
                                                        (0.229, 0.224, 0.225))])
            else:
                self.transform = transforms.ToTensor()
        else:
            self.transform = transform
            
        self.mode = mode
        self.batch_size = batch_size
        self.data_folder = data_folder
        
        self.cifar = CIFAR(mode = self.mode, data_folder = self.data_folder)
        
    def __getitem__(self, index):
        """
          Input: index
          Output: 3-tuple (image, class index, superclass index)
        """
        # get class index, superclass index, and filename
        class_idx = self.cifar.fine_labels[index]
        superclass_idx = self.cifar.coarse_labels[index]
        filename = self.cifar.file_names[index]
        
        # Read the image and apply transformations
        image = Image.open(os.path.join(self.cifar.image_folder, filename))
        image = self.transform(image)
        
        # Return image, class index and superclass index
        return image, class_idx, superclass_idx
        
    def __len__(self):
        return len(self.cifar.images)

    def get_train_indices(self):
        return np.random.randint(low = 0, high = len(self.cifar.images), size = self.batch_size)
    
dataset = CIFAR100Dataset(mode = 'train')
loader = get_loader()