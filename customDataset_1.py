import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import sys
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
import pandas as pd
import sklearn as sk
import os
import time 
from datetime import datetime
import pytz


class Lung_Train_Dataset_1():
    def __init__(self, train_normal,train_infected_covid,train_infected_non_covid):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infected_covid', 2:'infected_non_covid'}
        # The dataset consists only of training images
        self.groups = 'train'
        # Number of images in each part of the dataset
        self.dataset_numbers = {'train_normal': train_normal,\
                                'train_infected_covid': train_infected_covid,\
                                'train_infected_non_covid':train_infected_non_covid}
        
        # Path to images for different parts of the dataset
        self.dataset_paths = {'train_normal': 'small_proj_dataset/train/normal/',\
                              'train_infected_covid': 'small_proj_dataset/train/infected/covid',\
                              'train_infected_non_covid': 'small_proj_dataset/train/infected/non-covid'}   
        
    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        msg = "This is the training dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)
        
    
    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected_covid' or 'infected_non_covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg
        
        err_msg = "Error - class_val variable should be set to 'normal' or 'infected_covid' or 'infected_non_covid'."
        assert class_val in self.classes.values(), err_msg
        
        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        if class_val =='normal':
            path_to_file = '{}{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        else:
            path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        # print(path_to_file)
        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f))/255
        f.close()
        return im, path_to_file

    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected_covid' or 'infected_non_covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        im, path = self.open_img(group_val, class_val, index_val)
        
        # Display
        plt.imshow(im)
        
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        return sum(self.dataset_numbers.values())
    
    
    def __getitem__(self, index):
        """
        Getitem special method.
        
        Expects an integer value index, between 0 and len(self) - 1.
        
        Returns the image and its label as a single integer, both
        in torch tensor format in dataset.
        
        label 0 indicates healthy while label 1 indicates infected.
        """
        
        # Get item special method
        first_val = int(list(self.dataset_numbers.values())[0]) # healthy
        second_val = int(list(self.dataset_numbers.values())[1]) # covid
        if index < first_val:
            class_val = 'normal'
            #label = torch.Tensor([1, 0, 0])
            label = 0
        elif index < first_val + second_val:
            class_val = 'infected_covid'
            index = index - first_val
            #label = torch.Tensor([0, 1, 0])
            label = 1
        else:
            class_val = 'infected_non_covid'
            index = index - first_val - second_val
            #label = torch.Tensor([0, 0, 1])
            label = 1
    
        im, path = self.open_img(self.groups, class_val, index)
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label, path
    
class Lung_Test_Dataset_1():
    def __init__(self,test_normal,test_infected_covid,test_infected_non_covid):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """
        
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        
        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infected_covid', 2:'infected_non_covid'}
        
        # The dataset consists only of training images
        self.groups = 'test'
        
        # Number of images in each part of the dataset
        self.dataset_numbers = {'test_normal': test_normal,\
                                'test_infected_covid': test_infected_covid,\
                                'test_infected_non_covid': test_infected_non_covid}
        
        # Path to images for different parts of the dataset
        self.dataset_paths = {'test_normal': 'small_proj_dataset/test/normal/',\
                              'test_infected_covid': 'small_proj_dataset/test/infected/covid',\
                              'test_infected_non_covid': 'small_proj_dataset/test/infected/non-covid'}

        # self.class_to_idx = {'normal': 0, 'infected_covid': 1, 'infected_non_covid': 2}

    #TODO create augmentation for the images.
    def augment(self):
        train_transforms = transforms.Compose([tranforms.to_tensor()])
        return      
        
    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        msg = "This is the testing dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)
        
    
    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected_covid' or 'infected_non_covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg
        
        err_msg = "Error - class_val variable should be set to 'normal' or 'infected_covid' or 'infected_non_covid'."
        assert class_val in self.classes.values(), err_msg
        
        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        if class_val =='normal':
            path_to_file = '{}{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        else:
            path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        # print(path_to_file)
        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f))/255
        f.close()
        return im, path_to_file

    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected_covid' or 'infected_non_covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        im, path = self.open_img(group_val, class_val, index_val)
        
        # Display
        plt.imshow(im)
        
        
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        return sum(self.dataset_numbers.values())
    
    
    def __getitem__(self, index):
        """
        Getitem special method.
        
        Expects an integer value index, between 0 and len(self) - 1.
        
        Returns the image and its label as a single integer, both
        in torch tensor format in dataset.
        
        label 0 indicates healthy while label 1 indicates infected.
        """
        
        # Get item special method
        first_val = int(list(self.dataset_numbers.values())[0]) # healthy
        second_val = int(list(self.dataset_numbers.values())[1]) # covid
        if index < first_val:
            class_val = 'normal'
            #label = torch.Tensor([1, 0, 0])
            label = 0
        elif index < first_val + second_val:
            class_val = 'infected_covid'
            index = index - first_val
            #label = torch.Tensor([0, 1, 0])
            label = 1
        else:
            class_val = 'infected_non_covid'
            index = index - first_val - second_val
            #label = torch.Tensor([0, 0, 1])
            label = 1
    
        im, path = self.open_img(self.groups, class_val, index)
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label, path
    
class Lung_Valid_Dataset_1():
    def __init__(self,valid_normal, valid_infected_covid, valid_infected_non_covid):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """
        
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        
        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infected_covid', 2:'infected_non_covid'}
        
        # The dataset consists only of training images
        self.groups = 'valid'
        
        # Number of images in each part of the dataset
        self.dataset_numbers = {'valid_normal': valid_normal,\
                                'valid_infected_covid': valid_infected_covid,\
                                'valid_infected_non_covid': valid_infected_non_covid}
        
        # Path to images for different parts of the dataset
        self.dataset_paths = {'valid_normal': 'small_proj_dataset/val/normal/',\
                              'valid_infected_covid': 'small_proj_dataset/val/infected/covid',\
                              'valid_infected_non_covid': 'small_proj_dataset/val/infected/non-covid'}

        # self.class_to_idx = {'normal': 0, 'infected_covid': 1, 'infected_non_covid': 2}     
        
    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        msg = "This is the validation dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)
        
    
    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected_covid' or 'infected_non_covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg
        
        err_msg = "Error - class_val variable should be set to 'normal' or 'infected_covid' or 'infected_non_covid'."
        assert class_val in self.classes.values(), err_msg
        
        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        if class_val =='normal':
            path_to_file = '{}{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        else:
            path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        # print(path_to_file)
        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f))/255
        f.close()
        return im, path_to_file

    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected_covid' or 'infected_non_covid'
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        im, path = self.open_img(group_val, class_val, index_val)
        
        # Display
        plt.imshow(im)
        
        
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        return sum(self.dataset_numbers.values())
    
    
    def __getitem__(self, index):
        """
        Getitem special method.
        
        Expects an integer value index, between 0 and len(self) - 1.
        
        Returns the image and its label as a single integer, both
        in torch tensor format in dataset.
        
        label 0 indicates healthy while label 1 indicates infected.
        """
        
        # Get item special method
        first_val = int(list(self.dataset_numbers.values())[0]) # healthy
        second_val = int(list(self.dataset_numbers.values())[1]) # covid
        if index < first_val:
            class_val = 'normal'
            #label = torch.Tensor([1, 0, 0])
            label = 0
        elif index < first_val + second_val:
            class_val = 'infected_covid'
            index = index - first_val
            #label = torch.Tensor([0, 1, 0])
            label = 1
        else:
          class_val = 'infected_non_covid'
          index = index - first_val - second_val
          #label = torch.Tensor([0, 0, 1])
          label = 1
    
        im, path = self.open_img(self.groups, class_val, index)
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label, path    
    
class Lung_Train_Dataset_1_Aug():
    def __init__(self, train_normal,train_infected_covid,train_infected_non_covid, transform=None):
        """
        Constructor for generic Dataset class with additional data augmentation feature - simply assembles
        the important parameters in attributes.
        """
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infected_covid', 2:'infected_non_covid'}
        # The dataset consists only of training images
        self.groups = 'train'
        # Number of images in each part of the dataset
        self.dataset_numbers = {'train_normal': train_normal,\
                                'train_infected_covid': train_infected_covid,\
                                'train_infected_non_covid':train_infected_non_covid}
        
        # Path to images for different parts of the dataset
        self.dataset_paths = {'train_normal': 'small_proj_dataset/train/normal/',\
                              'train_infected_covid': 'small_proj_dataset/train/infected/covid',\
                              'train_infected_non_covid': 'small_proj_dataset/train/infected/non-covid'}
        
        self.transform = transform

    #create augmentation for the images.
    def augment(self):
        train_transforms = transforms.Compose([tranforms.to_tensor()])
        return      
        
    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        msg = "This is the training dataset of the Lung Dataset"
        msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)
        
    
    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected_covid' or 'infected_non_covid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg
        
        err_msg = "Error - class_val variable should be set to 'normal' or 'infected_covid' or 'infected_non_covid'."
        assert class_val in self.classes.values(), err_msg
        
        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        if class_val =='normal':
            path_to_file = '{}{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        else:
            path_to_file = '{}/{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)
        # print(path_to_file)
        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f))/255
        f.close()
        return im, path_to_file

    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        im, path = self.open_img(group_val, class_val, index_val)
        
        # Display
        plt.imshow(im)
        
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        return sum(self.dataset_numbers.values())
    
    
    def __getitem__(self, index):
        """
        Getitem special method.
        
        Expects an integer value index, between 0 and len(self) - 1.
        
        Returns the image and its label as single integer, both
        in torch tensor format in dataset.
        """
        
        # Get item special method
        first_val = int(list(self.dataset_numbers.values())[0]) # healthy
        second_val = int(list(self.dataset_numbers.values())[1]) # covid
        if index < first_val:
            class_val = 'normal'
            #label = torch.Tensor([1, 0, 0])
            label = 0
        elif index < first_val + second_val:
            class_val = 'infected_covid'
            index = index - first_val
            #label = torch.Tensor([0, 1, 0])
            label = 1
        else:
            class_val = 'infected_non_covid'
            index = index - first_val - second_val
            #label = torch.Tensor([0, 0, 1])
            label = 1
    
        im, path = self.open_img(self.groups, class_val, index)
        if self.transform:
            im = transforms.functional.to_tensor(np.array(im)).float()
            im = transforms.ToPILImage()(im.squeeze_(0))
            im = self.transform(im)
            im = torchvision.transforms.functional.to_tensor(im)
        else:
            im = torchvision.transforms.functional.to_tensor(np.array(im)).float()  
        return im, label, path
