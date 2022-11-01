from logging import root
import torch
import glob
import numpy as np
import os
from tqdm import tqdm

DEBUG = False

if DEBUG == True:
    from ngp_pl.datasets.ray_utils import get_ray_directions
    from ngp_pl.datasets.color_utils import read_image
    from ngp_pl.datasets.base import BaseDataset
else:
    from .ray_utils import get_ray_directions
    from .color_utils import read_image

    from .base import BaseDataset


class CustomDataset(BaseDataset):
    '''' Custom dataset for blender multi-view
    TODO: implement test set
    '''
    def __init__(self, root_dir='/home/maturk/git/ngp_pl/CustomData', split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)
        self.read_intrinsics()
        if kwargs.get('read_meta', True):
            xyz_min, xyz_max = \
                np.loadtxt(os.path.join(root_dir, 'bbox.txt'))[:6].reshape(2, 3)
            self.shift = (xyz_max+xyz_min)/2
            self.scale = (xyz_max-xyz_min).max()/2 * 1.05 # enlarge a little

            # hard-code fix the bound error for some scenes...
            if 'Mic' in self.root_dir: self.scale *= 1.2
            elif 'Lego' in self.root_dir: self.scale *= 1.1

            self.read_meta(split)

    def read_intrinsics(self):
        K = np.loadtxt(os.path.join(self.root_dir, 'intrinsics.txt'),
                        dtype=np.float32)[:3, :3]
        w, h = int(240), int(240)
        K[:2] *= self.downsample
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'train': prefix = '0_'
        elif split == 'val': prefix = '1_'
        elif split == 'test': prefix = '0_' # test set for real scenes TODO
        else: raise ValueError(f'{split} split not recognized!')
        img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix+'*.png')))
        poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix+'*.txt')))
        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path, pose in tqdm(zip(img_paths, poses)):
            c2w = np.loadtxt(pose)[:3]
            c2w[:, 3] -= self.shift
            c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
            self.poses += [c2w]

            img = read_image(img_path, self.img_wh)
            if 'Jade' in self.root_dir or 'Fountain' in self.root_dir:
                # these scenes have black background, changing to white
                img[torch.all(img<=0.1, dim=-1)] = 1.0
            self.rays += [img]
        self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
        
if __name__ == "__main__":
    dataset = CustomDataset()