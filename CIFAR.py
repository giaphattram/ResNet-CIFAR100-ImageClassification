# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:53:20 2020

@author: Admin
"""
# parent_dictionary = "C:\MountedDrive\Projects\Resnet CIFAR100"
import os
# os.chdir(parent_dictionary)
import pickle
from PIL import Image
import pandas as pd
from tqdm import tqdm
class CIFAR(object):
    def __init__(self, mode, data_folder = './data'):
        '''
        '''
        assert mode == 'train' or mode == 'test'
        self.mode = mode
        self.data_folder = data_folder
        
        self.data_dict          = self.get_data_dict()
        self.file_names         = self.get_file_names(self.data_dict)
        self.fine_labels        = self.get_fine_labels(self.data_dict)
        self.coarse_labels      = self.get_coarse_labels(self.data_dict)
        self.images             = self.get_images(self.data_dict)
        self.image_folder       = self.save_images(self.images, self.file_names)
        self.create_annotation_file()
        
        # Create dictionaries to map indices to classes and superclasses
        self.class_to_int, self.int_to_class, self.superclass_to_int, self.int_to_superclass = self.build_class_decoder_encoder()
        
    def unpickle(self, file):
        '''
            Unpickle image data file downloaded from https://www.cs.toronto.edu/~kriz/cifar.html
        '''
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding = 'bytes')
        return dict 
    
    def get_data_dict(self):
        return self.unpickle(os.path.join(self.data_folder, self.mode))
    
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
    
    def save_one_image(self, one_image_numpy, filename):
        '''
            Save one image. Helper function for save_one_image
        '''
        PIL_image = Image.fromarray(one_image_numpy)
        img_folder = self.mode + "_images"
        img_path = self.data_folder + "/" + img_folder + "/" + filename
        if not os.path.exists(img_path):
            PIL_image.save(img_path, "PNG")
        
    def save_images(self, images_numpy, filenames):
        '''
            Save images from the input image_numpy into hard disk
        '''
        folder_path = os.path.join(self.data_folder, self.mode + '_images') 
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        for img, filename in tqdm(zip(images_numpy, filenames)):
            self.save_one_image(img, filename)
        return folder_path
    
    def create_annotation_file(self):
        '''
            Create annotation file containing image file names and labels
        '''
        annotation_dict = {}
        annotation_dict.update({'filename': self.file_names, 'fine_label': self.fine_labels,
                                    'coarse_label': self.coarse_labels})
        file_path = self.data_folder + "/" + self.mode + ".csv"
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
        
        int_to_class = pd.Series(int_to_class).reset_index()
        int_to_class.columns = ['index', 'class']
        int_to_class_path = self.data_folder + "/" + "int_to_class.csv"
        int_to_class.to_csv(int_to_class_path, index = False)
        
        
        # Build super class decoder and encoder
        coarse_label_names = [each.decode('utf-8') for each in meta[b'coarse_label_names']]
        superclass_to_int = {k: v for k,v in zip(coarse_label_names, range(0, len(coarse_label_names)))}
        int_to_superclass = {item[1]:item[0] for item in superclass_to_int.items()}
        
        int_to_superclass = pd.Series(int_to_superclass).reset_index()
        int_to_superclass.columns = ['superclass', 'index']
        int_to_superclass_path = self.data_folder + "/" + "int_to_superclass.csv"
        int_to_superclass.to_csv(int_to_superclass_path, index = False)
        
        return class_to_int, int_to_class, superclass_to_int, int_to_superclass
        
ci = CIFAR(mode = 'train')