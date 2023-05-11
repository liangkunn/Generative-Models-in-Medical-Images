import torch
import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :28, :28]

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, stride=1, kernel_size=3, bias=False, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                #
                nn.Conv2d(32, 64, stride=2, kernel_size=3, bias=False, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                #
                nn.Conv2d(64, 128, stride=2, kernel_size=3, bias=False, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                #
                nn.Conv2d(128, 32, stride=1, kernel_size=3, bias=False, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                #
                nn.Flatten(),
                nn.Linear(1568, 2)
        ) 
        self.decoder = nn.Sequential(
                torch.nn.Linear(2, 1568),
                Reshape(-1, 32, 7, 7),
                nn.ConvTranspose2d(32, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(64, 128, stride=(2, 2), kernel_size=(3, 3), padding=1),                
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(128, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),                
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(32, 3, stride=(1, 1), kernel_size=(3, 3), padding=0), 
                Trim(),  # 3x29x29 -> 3x28x28
                nn.Sigmoid()
                )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x