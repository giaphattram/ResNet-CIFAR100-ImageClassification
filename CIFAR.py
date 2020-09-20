# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:53:20 2020

@author: Admin
"""
parent_dictionary = "C:\MountedDrive\Projects\Resnet CIFAR100"
import os
os.chdir(parent_dictionary)
import pickle
from PIL import Image
import numpy as np
import pandas as pd

class CIFAR(object):
    def __init__(self, data_folder = './data'):
        '''
        '''
        self.data_folder = data_folder
        
        self.train_dict          = self.get_data_dict('train')
        self.train_file_names    = self.get_file_names(self.train_dict)
        self.train_fine_labels   = self.get_fine_labels(self.train_dict)
        self.train_coarse_labels = self.get_coarse_labels(self.train_dict)
        self.train_images        = self.get_images(self.train_dict)
        # self.save_images(self.train_images, self.train_file_names, mode = 'train')
        self.create_annotation_file(mode = 'train')
        
        self.test_dict          = self.get_data_dict('test')
        self.test_file_names    = self.get_file_names(self.test_dict)
        self.test_fine_labels   = self.get_fine_labels(self.test_dict)
        self.test_coarse_labels = self.get_coarse_labels(self.test_dict)
        self.test_images        = self.get_images(self.test_dict)
        # self.save_images(self.test_images, self.test_file_names, mode = 'test')
        self.create_annotation_file(mode = 'test')
        
        self.class_to_int, self.int_to_class, self.superclass_to_int, self.int_to_superclass = self.build_class_decoder_encoder()
        
    def unpickle(self, file):
        '''
            Unpickle image data file downloaded from https://www.cs.toronto.edu/~kriz/cifar.html
        '''
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding = 'bytes')
        return dict 
    
    def get_data_dict(self, mode):
        if mode == 'train':
            return self.unpickle(os.path.join(self.data_folder, 'train'))
        if mode == 'test':
            return self.unpickle(os.path.join(self.data_folder, 'test'))
    
    def get_file_names(self, data_dict):
        return [filename.decode('utf-8') for filename in data_dict[b'filenames']]
    
    def get_fine_labels(self, data_dict):
        return data_dict[b'fine_labels']
    
    def get_coarse_labels(self, data_dict):
        return data_dict[b'coarse_labels']
    
    def get_images(self, data_dict):
        images = data_dict[b'data']
        images = images.reshape(len(images), 3, 32, 32).transpose(0, 2, 3, 1)
        return images
    
    def save_one_image(self, one_image_numpy, filename, mode):
        '''
            Save one image. Helper function for save_one_image
        '''
        global temp 
        temp = one_image_numpy
        PIL_image = Image.fromarray(one_image_numpy)
        img_folder = mode + "_images"
        img_path = self.data_folder + "/" + img_folder + "/" + filename
        PIL_image.save(img_path, "PNG")
        # PIL_image.save(os.path.join(self.data_folder, 'train/', filename), "PNG")
        
    def save_images(self, images_numpy, filenames, mode):
        '''
            Save images from the input image_numpy into hard disk
        '''
        assert mode == 'train' or mode == 'test'
        folder_path = os.path.join(self.data_folder, mode + '_images')
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        for img, filename in zip(images_numpy, filenames):
            self.save_one_image(img, filename, mode)
            break
    
    def create_annotation_file(self, mode):
        '''
            Create annotation file containing image file names and labels
        '''
        assert mode == 'train' or mode == 'test'
        annotation_dict = {}
        file_path = self.data_folder + "/" + mode + ".csv"
        if mode == 'train':
            annotation_dict.update({'filename': self.train_file_names, 'fine_labels': self.train_fine_labels,
                                    'coares_labels': self.train_coarse_labels})
        else:
            annotation_dict.update({'filename': self.test_file_names, 'fine_labels': self.test_fine_labels,
                                    'coares_labels': self.test_coarse_labels})
        annotation_df = pd.DataFrame(annotation_dict)
        annotation_df.to_csv(file_path, index = False)
        
    def build_class_decoder_encoder(self):
        '''
            Generate class_to_int.csv and superclass_to_int.csv
            Return: class_to_int, int_to_class, superclass_to_int, int_to_superclass
                
        '''
        meta = self.unpickle(os.path.join(self.data_folder, 'meta'))
        
        # Buid class decoder and encoder
        fine_label_names = [each.decode('utf-8') for each in meta[b'fine_label_names']]
        class_to_int = {k: v for k,v in zip(fine_label_names, range(0, len(fine_label_names)))}
        int_to_class = {item[1]:item[0] for item in class_to_int.items()}
        
        class_to_int = pd.Series(class_to_int).reset_index()
        class_to_int.columns = ['class', 'index']
        class_to_int_path = self.data_folder + "/" + "class_to_int.csv"
        class_to_int.to_csv(class_to_int_path, index = False)
        
        # Build super class decoder and encoder
        coarse_label_names = [each.decode('utf-8') for each in meta[b'coarse_label_names']]
        superclass_to_int = {k: v for k,v in zip(coarse_label_names, range(0, len(coarse_label_names)))}
        int_to_superclass = {item[1]:item[0] for item in superclass_to_int.items()}
        
        superclass_to_int = pd.Series(superclass_to_int).reset_index()
        superclass_to_int.columns = ['superclass', 'index']
        superclass_to_int_path = self.data_folder + "/" + "superclass_to_int.csv"
        superclass_to_int.to_csv(superclass_to_int_path, index = False)
        
        return class_to_int, int_to_class, superclass_to_int, int_to_superclass
        
ci = CIFAR()