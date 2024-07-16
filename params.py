import random
import torch.utils.data
import matplotlib.pyplot as plt
import csv

# Set random seed for reproducibility
manualSeed = 1
# manualSeed = random.randint(1, 10000) # Use if want new results
# manualSeed = random.randint(0, 1)
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results


'''Inputs Definitions'''

# Root directory for the dataset
# dataroot = '/ply_training_data/'
dataroot = '/binvox_training chairs/'

# Number of worker threads for the DataLoader
workers = 0

# Batch size during training
# batch_size = 16
# batch_size = 32
batch_size = 64

# The spatial size of shaped used in training. Default is 64x64x64, if another size is desired,
# changes in D and G must be made
shape_size = 64

# Number of color channels in the input images
# nc = 3
nc = 1

# length of latent vector (size of random noise input into generator)
# nz = 256
nz = 1024

# depths of feature maps carried through the generator
ngf = 64

# depths of feature maps carried through the discriminator
ndf = 64

# number of training epochs
num_epochs = 50

# model 1d
# Dlr = 0.00088
# Glr = 0.005

# model 2
Dlr = 0.00005
Glr = 0.02


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

def plotting_FID(fid, path):
    plt.figure(figsize=(10, 5))
    plt.title("FID Score During Training Per Epoch")
    plt.plot(fid, label="FID Score")
    plt.xlabel("Epochs")
    plt.ylabel("FID Score")
    plt.legend()
    plt.savefig(path + '/FID_Score.png')
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

def loss_Save(G_losses, D_losses, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Epoch', 'G_loss', 'D_loss'])

        # Write the data
        for epoch, (g_loss, d_loss) in enumerate(zip(G_losses, D_losses)):
            writer.writerow([epoch + 1, g_loss, d_loss])

    print(f"Losses saved to {csv_file_path}")

def fid_Save(fid, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Epoch', 'FID Score'])

        # Write the data
        for epoch, fid in enumerate(zip(fid)):
            writer.writerow([epoch + 1, fid])

    print(f"Losses saved to {csv_file_path}")
