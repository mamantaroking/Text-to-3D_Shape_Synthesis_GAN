import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
# manualSeed = 1
# manualSeed = random.randint(1, 10000) # Use if want new results
manualSeed = random.randint(1, 5)
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results


'''Inputs Definitions'''

# Root directory for the dataset
# dataroot = '/ply_training_data/'
dataroot = '/binvox_training chairs/'

# Number of worker threads for the DataLoader
workers = 2

# Batch size during training
# batch_size = 22
# batch_size = 77
batch_size = 16

# The spatial size of shaped used in training. Default is 64x64x64, if another size is desired,
# changes in D and G must be made
shape_size = 64

# Number of color channels in the input images
# nc = 3
nc = 1

# length of latent vector (size of random noise input into generator)
# nz = 256
nz = 1024
nzs = "norm"

# depths of feature maps carried through the generator
ngf = 64

# depths of feature maps carried through the discriminator
ndf = 64

# number of training epochs
num_epochs = 200

# learning rate for optimizers 128x
# Dlr = 0.00005
# Glr = 0.0001

# learning rate for optimizers 64x simple descriptions
# Dlr = 0.000025
# Glr = 0.00013

# learning rate for optimizers 64x
Dlr = 0.00005
Glr = 0.0008
# learning rate for optimizers 64x 2800 two label data
Dlr = 0.00005
Glr = 0.0012

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPU available (0 is for cpu, 1 is 1 more gpu, 2 is 2 gpu and so forth)
ngpu = 1


def plotter(D_losses, G_losses, path):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(D_losses, label="D")
    plt.plot(G_losses, label="G")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path + '/GAN_losses.png')
    plt.show()

def accuring(accuracy, path):
    plt.figure(figsize=(10, 5))
    plt.title("Accuracy During Training")
    plt.plot(accuracy, label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(path + '/Accuracy.png')
    plt.show()

