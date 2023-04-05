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


# 04/04

# v2
# no reshape in generator -> works really good though 
# 3 convolutional layers

class C_GAN(torch.nn.Module):
        
    # latent dimension is for noise sampling
    def __init__(self, latent_dimension, image_height, image_width, color_channels, number_classes, embedding_size): 
        super().__init__()
        self.latent_dimension = latent_dimension
        self.image_height = image_height
        self.image_width = image_width
        self.color_channels = color_channels
        
        # for conditional GAN
        self.number_classes = number_classes
        self.embedding_size = embedding_size
        
        # embedding layer for generator
        self.gen_embedding = nn.Embedding(self.number_classes, self.embedding_size)
        
        # embedding layer for discriminator
        self.disc_embedding = nn.Embedding(self.number_classes, self.image_height*self.image_width)
            
        # 3 conv layers
        self.generator = nn.Sequential(
            # torch.nn.Linear(self.latent_dimension,256),
            # torch.nn.Unflatten(1,(16,4,4)),
            torch.nn.ConvTranspose2d(in_channels=self.latent_dimension + self.embedding_size,
                                     out_channels=2*self.image_height,
                                     kernel_size=(3,3),
                                     stride=(2,2),
                                     padding=0
            ),
            torch.nn.BatchNorm2d(2*self.image_height),
            # Relu or LeakyReLU remains for test
            torch.nn.LeakyReLU(),
            
            torch.nn.ConvTranspose2d(in_channels=2*self.image_height, # feel free to play around with in/out channels 
                                     out_channels=4*self.image_height,
                                     kernel_size=(3,3),
                                     stride=(2,2),
                                     padding=0
            ),
            torch.nn.BatchNorm2d(4*self.image_height),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels=4*self.image_height, # feel free to play around with in/out channels 
                                     out_channels=2*self.image_height,
                                     kernel_size=(3,3),
                                     stride=(2,2),
                                     padding=0
            ),
            torch.nn.BatchNorm2d(2*self.image_height),
            torch.nn.LeakyReLU(),           

            torch.nn.ConvTranspose2d(in_channels=2*self.image_height,
                                     out_channels=self.color_channels,
                                     kernel_size=(2,2),
                                     stride=(2,2),
                                     padding=1
            ),
            # torch.nn.Tanh()
            torch.nn.Sigmoid() # [N, 3, 28, 28]
        )

        # Lets use fully convolutional network
        # 3convs
        self.discriminator = nn.Sequential(
            torch.nn.Conv2d(in_channels=3 + 1,
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
            # torch.nn.AvgPool2d(4), 
            torch.nn.MaxPool2d(4), # its okay
            torch.nn.Flatten(),                                                
            torch.nn.Linear(128,1),
            # if it is in WGAN training, no longer out put value between 0,1
            torch.nn.Tanh(),   
            # torch.nn.Sigmoid()
        )
        
    def generator_forward(self, z, labels):
        # the embedding layer of label should be the consistent with the shape of noise
        # remember the shape of noise is N*latent_dimentsion*1*1
        # thats why we have 2 unsqueeze here
        gen_embed = self.gen_embedding(labels).unsqueeze(2).unsqueeze(3)
        z = torch.cat([z, gen_embed], dim=1)
        img = self.generator(z)
        return img
        
    def discriminator_forward(self, img, labels):
        # add label embeddings layer to the discriminator
        # label embedding layer is N*1*H*W
        dis_embed = self.disc_embedding(labels).view(labels.shape[0], 1, self.image_height, self.image_width) 
        img = torch.cat([img, dis_embed], dim=1)
        logits = self.discriminator(img)
        return logits
        
# 02/26
# dcgan with reshape in generator, 5conv in both gen and disc
class W_GAN_5(torch.nn.Module):
        
    # latent dimension is for noise sampling
    def __init__(self, latent_dimension, image_height, image_width, color_channels): 
        super().__init__()
        self.latent_dimension = latent_dimension
        self.image_height = image_height
        self.image_width = image_width
        self.color_channels = color_channels
        
        # 5 conv layers
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
                                     out_channels=8*self.image_height,
                                     kernel_size=(4,4),
                                     stride=(2,2),
                                     padding=1
                                    ),
            torch.nn.BatchNorm2d(8*self.image_height),
            torch.nn.LeakyReLU(),
            
            torch.nn.ConvTranspose2d(in_channels=8*self.image_height,
                                     out_channels=4*self.image_height,
                                     kernel_size=(4,4),
                                     stride=(2,2),
                                     padding=1
                                    ),  
            torch.nn.BatchNorm2d(4*self.image_height),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels=4*self.image_height,
                                     out_channels=2*self.image_height,
                                     kernel_size=(4,4),
                                     stride=(1,1),
                                     padding=1
            ),
            torch.nn.BatchNorm2d(2*self.image_height),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels=2*self.image_height,
                                     out_channels=self.color_channels,
                                     kernel_size=(2,2),
                                     stride=(1,1),
                                     padding=1
            ),
            torch.nn.Tanh()
            # torch.nn.Sigmoid() # [N, 3, 28, 28]
        )

        # Lets use fully convolutional network
        # 5 convs
        self.discriminator = nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=32,
                            kernel_size=(3,3),
                            stride=(2,2),
                            padding=1
                           ),
            # torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(0.2, inplace=True), 
            # dropout are for test
            # torch.nn.Dropout2d(p=0.5),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=(3,3),
                            stride=(2,2),
                            padding=1
                           ),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2, inplace=True), 
            # torch.nn.Dropout2d(p=0.5),
            
            torch.nn.Conv2d(in_channels=64,
                            out_channels=128,
                            kernel_size=(3,3),
                            stride=(2,2),
                            padding=1
                            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(in_channels=128,
                            out_channels=64,
                            kernel_size=(3,3),
                            stride=(2,2),
                            padding=1
                           ),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2, inplace=True), 
            
            torch.nn.Conv2d(in_channels=64,
                            out_channels=1,
                            kernel_size=(4,4),
                            stride=(1,1),
                            padding=1
                           ),
            torch.nn.Flatten(),
            # torch.nn.Tanh(),
      
        )
        
    def generator_forward(self, z):
        img = self.generator(z)
        return img
        
    def discriminator_forward(self, img):
        logits = self.discriminator(img)
        return logits
        