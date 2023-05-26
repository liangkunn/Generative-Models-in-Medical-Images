# Generative-Models-in-Medical-Images
Kun Liang, Brian Caffo
A repository for data augmentation in medical images, using VAE and GANs (conditional GANs, Wasserstein GAN with/without Gradient Penalty)

dermamnist.npz is one of the datasets from https://medmnist.com/

The original derma iamges:

<img width="480" alt="real images" src="https://user-images.githubusercontent.com/36016499/230223300-7052d41f-a710-40f0-9b69-355dff596c77.png">

Using non weight sampler Wgan_gp, the generated derma images are as follows:

Epoch0:

<img width="434" alt="epoch0" src="https://user-images.githubusercontent.com/36016499/230223363-6637948f-69a1-43cf-8509-4935f5b3c5b6.png">

Epoch50:

<img width="435" alt="epoch50" src="https://user-images.githubusercontent.com/36016499/230223394-65500ec8-c225-45e7-8f85-0614e3942f09.png">

Epoch300:

<img width="427" alt="epoch300" src="https://user-images.githubusercontent.com/36016499/230223417-17673687-5d39-4690-974d-2a37a82fc356.png">
