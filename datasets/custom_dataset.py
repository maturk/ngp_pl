from logging import root
import torch
import glob
import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation 

DEBUG = False

if DEBUG == True:
    from ngp_pl.datasets.ray_utils import get_ray_directions
    from ngp_pl.datasets.color_utils import read_image
    from ngp_pl.datasets.base import BaseDataset
else:
    from .ray_utils import get_ray_directions
    from .color_utils import read_image

    from .base import BaseDataset

CV_TO_OPENGL = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

test = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])

M = np.array([[ 0,  1,  0,  0],
                [-1,  0,  0,  0],
                [ 0,  0,  1,  0],
                [ 0,  0,  0,  1]])

class CustomDataset(BaseDataset):
    '''' Custom dataset for blender multi-view
    TODO: implement test set
    '''
    def __init__(self, root_dir='/home/maturk/git/ngp_pl/CustomData', split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)
        self.read_intrinsics()
        print('root direcotry is',root_dir)
        if kwargs.get('read_meta', True):
            xyz_min, xyz_max = \
                np.loadtxt(os.path.join(root_dir, 'bbox.txt'))[:6].reshape(2, 3)
            self.shift = (xyz_max+xyz_min)/2
            self.scale = (xyz_max-xyz_min).max()/2 #* 1.05 # enlarge a little

            # hard-code fix the bound error for some scenes...
            if 'Mic' in self.root_dir: self.scale *= 1.2
            elif 'Lego' in self.root_dir: self.scale *= 1.1

            self.read_meta(split)

    def read_intrinsics(self):
        K = np.loadtxt(os.path.join(self.root_dir, 'intrinsics.txt'),
                        dtype=np.float32)[:3, :3]
        print(K)
        w, h = int(240), int(240)
        K[:2] *= self.downsample
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'train': prefix = '1_'
        elif split == 'val': prefix = '2_'
        elif split == 'test': prefix = '[1-2]_' # test set for real scenes TODO [0-1]_
        else: raise ValueError(f'{split} split not recognized!')
        img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix+'*.png')))
        poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix+'*.txt')))
        print(f'Loading {len(img_paths)} {split} images ...')
        counter = 0
        for img_path, pose in tqdm(zip(img_paths, poses)):
            
            blender_pre_rotation = True
            if blender_pre_rotation:
                r = Rotation.from_euler('z', 90, degrees=True) #looked right had z -90
                pre_rot_T = np.eye(4)
                pre_rotation = r.as_matrix()
                pre_rot_T[:3,:3] = pre_rotation

            mine = True
            looks_correct = False
            
            if mine!= True and looks_correct!= True:
                # Apply transformations required by the NeRF convention.
                # - Flip the y and z axes.
                T_WC =  (np.loadtxt(pose)[:4])
                #T_WC = np.linalg.inv(T_WC)
                #T_WC[0:3, 2] *= -1
                T_WC[0:3, 1] *= -1
                # - Swap y and z.
                T_WC = T_WC[[1, 0, 2, 3], :]
                # - Flip the whole world upside down.
                #T_WC[2, :] *= -1
                c2w = T_WC[:3]
                c2w[:, 3] /= 2*self.scale
                self.poses += [c2w]
                
            
            #python show_gui.py --root_dir ./CustomData/can/ --exp_name Custom --dataset_name custom --ckpt_path ./ckpts/custom/Custom/epoch\=9.ckpt 
            if looks_correct:
                T_WC =   pre_rot_T@ (np.loadtxt(pose)[:4]) 
                T_WC = M @ T_WC
                c2w = T_WC[:3]
                c2w[:, 1] *= -1
                c2w[:, 2] *= -1
                
                #c2w[:, 3] /= 2*self.scale
                self.poses += [c2w]
            
            if mine == True:
                c2w = (np.loadtxt(pose)[:4]) 
                T_WC = c2w 
                #T_WC = T_WC[[1, 0, 2, 3], :]
                T_WC[0:3, 1] *= -1
                T_WC[0:3, 2] *= -1
                # - Swap y and z.
                #T_WC = T_WC[[1, 0, 2, 3], :]
                # - Flip the whole world upside down. sike
                #T_WC[2, :] *= 1
                c2w = T_WC[:3]
                #c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale#2.1
                self.poses += [c2w]
            
            save = np.eye(4)
            save[:3,:4] = c2w
            np.savetxt(f'/home/maturk/git/ObjectReconstructor/ngp_pl/datasets/debug/{counter}.txt', save)
            counter +=1
            
            img = read_image(img_path, self.img_wh)
            if 'Jade' in self.root_dir or 'Fountain' in self.root_dir:
                # these scenes have black background, changing to white
                img[torch.all(img<=0.1, dim=-1)] = 1.0
            self.rays += [img]
            
        self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)


def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose

def generate_transform_matrix(T_WC):
    xf_rot = np.eye(4)
    xf_rot[:3,:3] = T_WC[:3,:3]
    xf_pos = np.eye(4)
    xf_pos[:3,3] = T_WC[:3,3] # - average_position

    # barbershop_mirros_hd_dense:
    # - camera plane is y+z plane, meaning: constant x-values
    # - cameras look to +x
    # Don't ask me...
    extra_xf = np.array([
        [-1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1]])
    # NerF will cycle forward, so lets cycle backward.
    shift_coords = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])
    xf = shift_coords @ extra_xf @ xf_pos
    assert np.abs(np.linalg.det(xf) - 1.0) < 1e-4
    xf = xf @ xf_rot
    return xf

if __name__ == "__main__":
    dataset = CustomDataset()