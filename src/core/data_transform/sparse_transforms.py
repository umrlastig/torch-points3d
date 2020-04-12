
from typing import List
import itertools
import numpy as np
import math
import re
import torch
import scipy
import random
from tqdm import tqdm as tq
from torch.nn import functional as F
from sklearn.neighbors import NearestNeighbors, KDTree
from functools import partial
from torch_geometric.nn import fps, radius, knn, voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_scatter import scatter_add, scatter_mean

from src.datasets.multiscale_data import MultiScaleData
from src.utils.transform_utils import SamplingStrategy
from src.utils.config import is_list
from src.utils import is_iterable
from .grid_transform import group_data, GridSampling, shuffle_data, sparse_coords_to_clusters


class RemoveDuplicateCoords(object):
    """ This transform allow sto remove duplicated coords within ``indices`` from data.
    Selects the last point within each voxel to set the features and labels.
    Parameters
    ----------
    shuffle: bool
        If True, the data will be suffled before removing the extra points
    """

    def __init__(self, shuffle=False, mode="last"):
        self._shuffle = shuffle
        self._mode = mode

    def _process(self, data):
        if self._mode == "last":
            data = shuffle_data(data)
        
        coords = data.pos
        batch = data.batch if hasattr(data, "batch") else None
        cluster, unique_pos_indices = sparse_coords_to_clusters(coords, batch)
        
        if self._mode == "last":
            delattr(data, "pos")
        data = group_data(data, cluster, unique_pos_indices, mode=self._mode)
        
        if self._mode == "last":
            data.pos = coords[unique_pos_indices]
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(itertools.chain(*data))  # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(shuffle={}, mode={})".format(self.__class__.__name__, self._shuffle, self._mode)

class ToSparseInput(object):
    """This transform allows to prepare data for sparse model as SparseConv / Minkowski Engine.
    It does the following things:

    - Puts ``pos`` on a fixed integer grid based on grid size
    - Keeps one point per grid cell. The strategy for defining the feature nad label at that point depends on the ``mode`` option

    Parameters
    ----------
    grid_size: float
        Grid voxel size
    mode : str
        Option to select how the features and labels for each voxel are computed. Can be ``last`` or ``mean``.
        ``last`` selects the last point falling in a voxel as the representent, 
        ``mean`` takes the average. 

    Returns
    -------
    data: Data
        Returns the same data object with only one point per voxel
    """

    def __init__(self, grid_size=None, mode="last"):
        self._grid_size = grid_size
        self._mode = mode
        self._transform = GridSampling(grid_size, quantize_coords=True, mode=mode)

    def _process(self, data):
        return self._transform(data)

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(itertools.chain(*data))  # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(grid_size={}, mode={})"\
            .format(self.__class__.__name__, self._grid_size, self._mode)



class ElasticDistortion:

    """Apply elastic distortion on sparse coordinate space.

    Parameters
    ----------
    granularity: float
        Size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: bool
        Noise multiplier

    Returns
    -------
    data: Data
        Returns the same data object with distorted grid
    """

    def __init__(self, apply_distorsion:bool = True, granularity: List = [0.2, 0.4]):
        self._apply_distorsion = apply_distorsion
        self._granularity = list(granularity)

    @staticmethod
    def elastic_distortion(coords, granularity):
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)[0]

        # Create Gaussian noise tensor of the size given by granularity.
        dim = coords.shape[-1]
        denom = torch.Tensor([np.random.uniform(granularity[0], granularity[1]) for _ in range(dim)])
        noise_dim = ((coords - coords_min).max(0)[0] // denom).int() + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        granularity_shift = granularity[1]
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity_shift, coords_min + granularity_shift *
                                    (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        return (coords + torch.Tensor(interp(coords))).int()

    def __call__(self, data):
        if self._apply_distorsion:
            if np.random.uniform(0, 1) < .5:
                data.pos = ElasticDistortion.elastic_distortion(data.pos, torch.Tensor(self._granularity))
        return data

    def __repr__(self):
        return "{}(apply_distorsion={}, granularity={})".format(self.__class__.__name__, self._apply_distorsion, self._granularity)