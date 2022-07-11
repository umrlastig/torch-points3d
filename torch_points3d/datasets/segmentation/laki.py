import os
import numpy as np
import torch
from plyfile import PlyData
from sklearn.neighbors import KDTree
from random import randint

import logging

from torch_geometric.data import Dataset, Data, Batch

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.core.data_transform import Select, GridSampling3D
from torch_points3d.metrics.laki_tracker import LAKITracker

log = logging.getLogger(__name__)


class LAKI(Dataset):
    r"""LAKI: A Dataset for Semantic Scene Understanding.
    
    root dir should be structured as
    rootdir
        └─ inference.ply
        └─ testing.ply
        └─ training.ply
        └─ validation.ply
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            If :obj:`"inferance"`, loads the inferance dataset.
            (default: :obj:`"train"`)
        sphere_count (int, optional): Number of drawn sphere
            Required for train and val split set.
        sphere_size (int, optional): Size of drawn sphere in number of point 
            (default: 1024)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, split="train", sphere_count=None, sphere_size=1024, transform=None, pre_transform=None):
        self.split = split
        self.sphere_count = sphere_count
        self.sphere_size = sphere_size
        super().__init__(root, transform=transform, pre_transform=pre_transform)

        self.data = torch.load(self.processed_paths[0])
        if "label_index" in self.data:
            self.label_index = self.data.label_index
            self.data.label_index = None

        self.name = split

    @property
    def raw_file_names(self):
        if self.split == "train":
            return ["training.ply"]
        elif self.split == "val":
            return ["validation.ply"]
        elif self.split == "test":
            return ["testing.ply"]
        elif self.split == "inference":
            return ["inference.ply"]

    @property
    def processed_file_names(self):
        return [self.split+".pt"]

    def download(self):
        print(f"please download the LAKI dataset with the following folder structure")
        print("""
                rootdir
                    └─ inference.ply
                    └─ testing.ply
                    └─ training.ply
                    └─ validation.ply
            """)
        exit()

    def process(self):

        with open(self.raw_paths[0], "rb") as ply_file:
            plydata = PlyData.read(ply_file)
            pos = np.concatenate((plydata["vertex"].data['x'].reshape(-1,1), plydata["vertex"].data['y'].reshape(-1,1), plydata["vertex"].data['z'].reshape(-1,1)), axis=1)
            rgb = np.concatenate((plydata["vertex"].data['red'].reshape(-1,1), plydata["vertex"].data['green'].reshape(-1,1), plydata["vertex"].data['blue'].reshape(-1,1)), axis=1)
            normal = np.concatenate((plydata["vertex"].data['nx'].reshape(-1,1), plydata["vertex"].data['ny'].reshape(-1,1), plydata["vertex"].data['nz'].reshape(-1,1)), axis=1)
            intensity = plydata["vertex"].data['scalar_Intensity']
            returns = plydata["vertex"].data['scalar_NumberOfReturns']
            if 'scalar_clasa' in plydata["vertex"].data.dtype.names:
                label = plydata["vertex"].data['scalar_clasa'].astype(np.uint8)

        # Creating the data object with position, RGB color, intensity, number of returns, label and normals
        if self.split in ["train", "val", "test"]:
            data = Data(
                pos = torch.from_numpy(pos),
                rgb = torch.from_numpy(rgb),
                normal = torch.from_numpy(normal),
                intensity = torch.from_numpy(intensity.copy()),
                returns = torch.from_numpy(returns.copy()),
                y = torch.from_numpy(label.copy()).long()
                )
        elif self.split in ["inference"]:
            data = Data(
                pos = torch.from_numpy(pos),
                rgb = torch.from_numpy(rgb),
                normal = torch.from_numpy(normal),
                intensity = torch.from_numpy(intensity.copy()),
                returns = torch.from_numpy(returns.copy())
                )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Computing label index
        if self.split in ["train", "val"]:
            label_index = []
            for i in range(max(label)+1):
                label_index.append(torch.nonzero(data.y == i, as_tuple=False))
            data.label_index = label_index

        # Computing KD-Tree
        tree = KDTree(data.pos.numpy(), leaf_size=64)
        data.kd_tree = tree

        # Computing sphere
        if self.split in ["test", "inference"]:
            grid = GridSampling3D(
                size=(self.sphere_size/25/3.14)**(1/2),
                quantize_coords=False, mode='mean'
                )(data.clone())
            sphere_list = []
            index = data.kd_tree.query(grid.pos, k=self.sphere_size, return_distance=False)
            for i in range(len(grid.pos)):
                sphere_list.append(Select(indices=index[i])(data))
            data = sphere_list

        # Save pre-process file
        torch.save(data, self.processed_paths[0])

    def len(self):
        if self.split in ["train", "val"]:
            return self.sphere_count
        elif self.split in ["test", "inference"]:
            return len(self.data)

    def get(self, idx):
        if self.split in ["train", "val"]:
            label = randint(0, len(self.label_index)-1)
            point_index = randint(0, len(self.label_index[label])-1)
            point_id = self.label_index[label][point_index]
            point_pos = self.data.pos[point_id]
            index = self.data.kd_tree.query(point_pos, k=self.sphere_size, return_distance=False)
            data = Select(indices=index)(self.data)
            return data
        elif self.split in ["test", "inference"]:
            return self.data[idx].clone()

    @property
    def num_classes(self):
        return 4

class LAKIDataset(BaseDataset):
    """ Wrapper around LAKI that creates train, val and test datasets.
    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain
            - root,
            - split,
            - transform,
            - pre_transfor
    """

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self.train_dataset = LAKI(
            self._data_path,
            split="train",
            sphere_count=dataset_opt.train_sphere_count,
            sphere_size=dataset_opt.sphere_size,
            transform=self.train_transform,
            pre_transform=self.pre_transform,
        )

        self.val_dataset = LAKI(
            self._data_path,
            split="val",
            sphere_count=dataset_opt.val_sphere_count,
            sphere_size=dataset_opt.sphere_size,
            transform=self.val_transform,
            pre_transform=self.pre_transform,
        )

        self.test_dataset = [
            LAKI(
                self._data_path,
                split="test",
                sphere_size=dataset_opt.sphere_size,
                transform=self.test_transform,
                pre_transform=self.pre_transform,
            ),
            LAKI(
                self._data_path,
                split="inference",
                sphere_size=dataset_opt.sphere_size,
                transform=self.test_transform,
                pre_transform=self.pre_transform,
            )
        ]

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return LAKITracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

if __name__ == "__main__":
    DIR = os.path.dirname(os.path.realpath(__file__))
    dataroot = os.path.join(DIR, "..", "..", "data", "laki")
    LAKI(
        dataroot, split="train"
    )

