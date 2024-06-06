import torch
from matplotlib import pyplot as plt, gridspec
import skimage.measure as sk
import numpy as np
import scipy.io as io
import scipy.ndimage as nd
from intro import nz, nzs, batch_size

'''PADDED TENSOR'''
# print("PADDED TENSORS")
shape = (3,5,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

'''print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")'''


'''PACKED TENSOR'''
# print("PACKED TENSORS")
shape = (12,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

'''print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")'''


voxel_grid = torch.rand((4, 4, 4))

# print(voxel_grid)

'''tensor = torch.rand(3,5,3)

if torch.cuda.is_available():
    tensor = tensor.to("cuda")

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")'''

'''def getVoxelFromMat(path):
        # voxels = np.load(path)
        # voxels = io.loadmat(path)['instance'] # 64x64x64
        # voxels = np.pad(voxels, (2, 2), 'constant', constant_values=(0, 0))
        # print (voxels.shape)
        voxels = io.loadmat(path)['instance'] # 30x30x30
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
        # print ('here')
        # print (voxels.shape)
        return voxels'''

def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:2].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='blue')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
    # print (path + '/{}.png'.format(str(iteration).zfill(3)))
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fixed_noise = torch.randn(batch_size, nz, 1, 1, 1, device=device)


def generateZ(batch):

    if nzs == "norm":
        # Z = torch.Tensor(batch, nz).normal_(0, 0.33).to(device)
        Z = torch.Tensor(batch).to(device)
        # Z = fixed_noise
    elif nzs == "uni":
        Z = torch.randn(batch, nz).to(device).to(device)
    else:
        print("z_dist is not normal or uniform")

    return Z

model_save_step = 1

