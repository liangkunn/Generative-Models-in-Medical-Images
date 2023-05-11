
import os
import numpy as np
from torchvision import transforms as tt
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import torchvision

# load the npz dataset
def load_npz_files(file_dir):
    """
    Load a npz.file
    
    Parameters
    ----------
    file_dir: the direction of the npz_files
    
    more args to be added if necessaray 
    """
    file = np.load(file_dir, allow_pickle=True)
    return file

# data check plotting
def data_check_plot_figures(figures, nrows = 1, ncols=1):
    """
    Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for index,title in enumerate(figures):
        axeslist.ravel()[index].imshow(figures[title])
        axeslist.ravel()[index].set_title(title)
        axeslist.ravel()[index].set_axis_off()
    plt.tight_layout() # optional

# data check plotting for tensor figures
def data_check_plot_tensor_figures(figures, nrows = 1, ncols=1):
    """
    Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 3), sharex=True, sharey=True,
                                 gridspec_kw={'wspace': 0.1})
    for index,title in enumerate(figures):
        # axeslist.ravel()[index].imshow(figures[title].detach().numpy().transpose(1,2,0))
        # if model is up on GPU
        axeslist.ravel()[index].imshow(figures[title].cpu().detach().numpy().transpose(1,2,0))
        axeslist.ravel()[index].set_title(title)
        axeslist.ravel()[index].set_axis_off()
    # plt.tight_layout() # optional

# use torch
def torch_plot_tensor_figures(figures, nrows=1, ncols=1, figsize=(8, 8), transform=None):
    """
    Plot a tensor of figures.

    Parameters
    ----------
    figures : tensor of figures
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    figsize : figure size
    transform: Optional torchvision transform to apply before displaying the image
    """

    # Use make_grid to create a grid of images
    grid_image = torchvision.utils.make_grid(figures, nrow=ncols, padding=2, pad_value=0)

    # Convert the tensor to a PIL image
    grid_image = (grid_image * 255).to(torch.uint8)
    pil_image = tt.ToPILImage()(grid_image)

    # Apply the optional transform
    if transform:
        pil_image = transform(pil_image)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(pil_image)
    # ax.set_axis_off()

    # plt.tight_layout()  # optional

def get_indexes_of_test(dataset):
    """
    Parameters
    ----------
    dataset : dataset that has been loaded by numpy    
    """
    index = []
    labels = len(np.unique(dataset['test_labels']))
    for i in range(labels):
        label_index = dataset['test_labels'] == i
        label_index = label_index.reshape(label_index.shape[0])
        index.append(label_index)
    return index

class Dataset_Preprocessing(Dataset):
    
    'Characterizes a dataset for PyTorch'
    def __init__(self, imgs):
        'Initialization'
        self.imgs = imgs
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs)
    def __getitem__(self, index):
        'Generates one sample of data, will be used by DataLoader'
        # Select sample
        image = self.imgs[index]
        X = self.transform(image)
        return X
        
    transform = tt.Compose([
    tt.ToTensor(), # one thing to mention: ToTensor() automatically transform images to 3,28,28 from 28,28,3
    # tt.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])   



