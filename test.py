import numpy as np
import os
import torch
import vren
import open3d as o3d

G = 128
CUBIC_SIZE = 2
VOXEL_RESOLUTION = 128
VOXEL_SIZE = V = CUBIC_SIZE/ VOXEL_RESOLUTION
VOXEL_OCCUPANCY_TRESHOLD = 0.4
GRID_MIN = np.array([-1,-1,-1])
GRID_MAX = np.array([1,1,1])
DENSITY_TRESHOLD = 0.5

def voxel_to_pc(voxel):
    #voxel = voxel.sigmoid()
    out = []
    for i in range(voxel.shape[0]):
        indices = (voxel[i,:] >= DENSITY_TRESHOLD).nonzero(as_tuple=False)
        if indices.nelement() == 0:
            points = torch.zeros((1,3))
        else:
            points = VOXEL_SIZE * indices[:,:] - CUBIC_SIZE / 2
        out.append(points)
    return out, indices

def coords_to_pc(coords):
    indices = torch.Tensor(coords)
    out = []
    if indices.nelement() == 0:
        points = torch.zeros((1,3))
    else:
        points = VOXEL_SIZE * indices[:,:] - CUBIC_SIZE / 2
    out.append(points)
    return out

coords = np.load(os.path.join('/home/maturk/git/ObjectReconstructor/ngp_pl/CustomData/pepsi/', 'coords.npy')) # z, x, y
prior = torch.Tensor(np.load('/home/maturk/git/ObjectReconstructor/ngp_pl/CustomData/pepsi/pred_grid.npy')).type(torch.float32)
prior[prior < 0] = 0
#prior = prior.reshape(-1, G**3) # similar to density grid

coords = torch.nonzero(prior>DENSITY_TRESHOLD)[:,1:]
indices = vren.morton3D(coords.int().contiguous().cuda())

print(coords)
print(indices)

out, indices = voxel_to_pc(prior)
out_coords = coords_to_pc(coords)[0] 

print(torch.sort(out_coords), indices)

pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(out[0].numpy())
o3d.visualization.draw_geometries([pc])

pc_coords = o3d.geometry.PointCloud()
pc_coords.points = o3d.utility.Vector3dVector(out_coords.numpy())
o3d.visualization.draw_geometries([pc_coords])

