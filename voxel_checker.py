import sys
import binvox_rw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


root_dir = "C:/Users/Maman/PycharmProjects/fyp/textto3dgan/"
sys.path.insert(0, root_dir)
sys.path.append(root_dir + "/binvox_files/")


'''def ShowVoxelModel(voxels, path, iteration, figsize=(128, 128), axisoff=True, edgecolor="k", facecolor="green", alpha=1.0, linewidth=0.0):
    voxels = voxels.__ge__(0.5)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.05, hspace=0.05)

    x, y, x = voxels.nonzero()
    ax = plt.subplot(gs[0], projection='3d')
    ax.set_proj_type('ortho')
    ax.voxels(voxels, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)

    if axisoff:
        ax.set_axis_off()

    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.show()'''

def ShowVoxelModel(voxels, path, iteration, figsize=(32, 32), axisoff=False, edgecolor="k", facecolor="green", alpha=1.0, linewidth=0.0):
    # voxels = voxels[:8].__ge__(0.5)
    voxels = voxels[:16].__ge__(0.5)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, voxels in enumerate(voxels):
        x, y, z = voxels.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        # ax.set_proj_type('ortho')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
        # ax.voxels(voxels, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)

        if axisoff:
            ax.set_axis_off()

    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.clf()
    # plt.show()



def voxvox(voxels):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32,32))
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxels, facecolors='lightblue', edgecolor='k', linewidth=0.0)
    ax.set_axis_off()
    plt.savefig('we_did_it.png', format=None, dpi=300)
    plt.show()


with open(root_dir + 'binvox_files/chair_2.binvox', 'rb') as f:
    vox = binvox_rw.read_as_3d_array(f).data
# print(vox.shape)
# voxvox(vox)
# ShowVoxelModel(vox)
