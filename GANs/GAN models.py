import numpy as np
import torch
import time
import torchvision
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms as tt
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.getcwd()


# 02/24

# it finally starts to learn something, but mode collapse happened

class DC_GAN(torch.nn.Module):
        
    # latent dimension is for noise sampling
    def __init__(self, latent_dimension, image_height, image_width, color_channels): 
        super().__init__()
        self.latent_dimension = latent_dimension
        self.image_height = image_height
        self.image_width = image_width
        self.color_channels = color_channels
        
        # 3 conv layers
        self.generator = nn.Sequential(
            torch.nn.Linear(self.latent_dimension,256),
            torch.nn.Unflatten(1,(16,4,4)),
            torch.nn.ConvTranspose2d(in_channels=16,
                                     out_channels=4*self.image_height,
                                     kernel_size=(4,4),
                                     stride=(1,1),
                                     padding=0
            ),
            torch.nn.BatchNorm2d(4*self.image_height),
            # Relu or LeakyReLU remains for test
            torch.nn.LeakyReLU(),
            
            torch.nn.ConvTranspose2d(in_channels=4*self.image_height,
                                     out_channels=2*self.image_height,
                                     kernel_size=(2,2),
                                     stride=(2,2),
                                     padding=0
            ),
            torch.nn.BatchNorm2d(2*self.image_height),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels=2*self.image_height,
                                     out_channels=self.color_channels,
                                     kernel_size=(2,2),
                                     stride=(2,2),
                                     padding=0
            ),
            # torch.nn.Tanh()
            torch.nn.Sigmoid() # [N, 3, 28, 28]
        )

        # Lets use fully convolutional network
        # 5convs
        self.discriminator = nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=32,
                            kernel_size=(3,3),
                            stride=(2,2),
                            padding=1
                           ),
            # torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(0.2, inplace=True), # 
            # dropout are for test
            # torch.nn.Dropout2d(p=0.5),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=(3,3),
                            stride=(2,2),
                            padding=1
                           ),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2, inplace=True), #
            # torch.nn.Dropout2d(p=0.5),
            torch.nn.Conv2d(in_channels=64,
                            out_channels=128,
                            kernel_size=(3,3),
                            stride=(2,2),
                            padding=1
                            ),
            torch.nn.LeakyReLU(0.2, inplace=True),                
            torch.nn.AvgPool2d(4), 
            torch.nn.Flatten(),                                                
            torch.nn.Linear(128,1),
            torch.nn.Tanh(),           
        )
        
    def generator_forward(self, z):
        img = self.generator(z)
        return img
        
    def discriminator_forward(self, img):
        logits = self.discriminator(img)
        return logits
        

# Lets generate random pictures with linear connection
class Linear_GAN(torch.nn.Module):

    def __init__(self, latent_dimension, image_height, image_width, color_channels):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.image_height = image_height
        self.image_width = image_width
        self.color_channels = color_channels
        
        self.generator = nn.Sequential(
            nn.Linear(latent_dimension, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, image_height*image_width*color_channels),
            nn.Tanh() # need to normalize imags with mean 0 and std 1
        )
        
        self.discriminator = nn.Sequential(
            nn.Flatten(), # NCHW -> N, C*H*W
            nn.Linear(image_height*image_width*color_channels, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1)
        )

    def generator_forward(self, z): # z has dimension NCHW
        z = torch.flatten(z, start_dim=1) 
        img = self.generator(z)
        img = img.view(z.size(0),
                       self.color_channels,
                       self.image_height,
                       self.image_width)
        return img
    
    def discriminator_forward(self,img):
        logits = self.discriminator(img)
        return logits
