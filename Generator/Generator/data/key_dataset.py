# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from functools import lru_cache
import contextlib
import logging
import torch
import numpy as np
from scipy.linalg import svd
logger = logging.getLogger(__name__)


def random_rotation_z(coords):
    """
    Apply a random rotation around the z-axis to a set of coordinates.
    
    Parameters:
    - coords: A numpy array of shape (n_points, 3)
    
    Returns:
    - A numpy array of shape (n_points, 3) with the coordinates after rotation
    """
    theta = np.radians(np.random.uniform(0, 360))  # Random angle for rotation
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    return coords.dot(rotation_matrix)

def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def calculate_centroid(coords):
    return np.mean(coords, axis=0)

def calculate_normal_vector(coords):
    centroid = calculate_centroid(coords)
    coords_centered = coords - centroid
    _, _, vh = svd(coords_centered, full_matrices=False)
    normal = vh[2, :]
    return normal

def rotate_to_xy_plane(coords):
    normal = calculate_normal_vector(coords)
    rotation_matrix = rotation_matrix_from_vectors(normal, np.array([0, 0, 1]))
    return coords.dot(rotation_matrix)

def rotate_coordinates(coords_np):
    if not isinstance(coords_np, np.ndarray):
        raise ValueError("The input must be a numpy array.")
    rotated_coords = rotate_to_xy_plane(coords_np)
    return rotated_coords

def rotation_matrix_from_vectors(vec1, vec2):
    """Create a rotation matrix that rotates vec1 to vec2."""
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def pca_based_rotation(coords):
    """Rotate the coordinates based on PCA to align with the principal axes."""
    # Calculate the centroid
    centroid = calculate_centroid(coords)
    # Center the coordinates
    coords_centered = coords - centroid
    # Perform SVD
    _, _, vh = svd(coords_centered, full_matrices=False)
    # Create the rotation matrix using the first two principal components
    rotation_matrix = rotation_matrix_from_vectors(vh[0], np.array([1, 0, 0]))
    rotation_matrix = rotation_matrix.dot(rotation_matrix_from_vectors(vh[1], np.array([0, 1, 0])))
    # Apply the rotation matrix to the centered coordinates
    rotated_coords = coords_centered.dot(rotation_matrix.T)
    # Translate the coordinates back to the original center
    rotated_coords += centroid
    return rotated_coords



class KeyDataset(BaseWrapperDataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    # @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.dataset[idx][self.key]
    

class ListDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return list(self.dataset[idx])

class RawListDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.dataset[idx]))






@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class ConformerSampleDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        assert len(atoms) > 0
        
        # size = len(self.dataset[index][self.coordinates][0])
        size = len(self.dataset[index][self.coordinates])
        with numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        
        # coordinates = self.dataset[index][self.coordinates][0][sample_idx]
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        
        return {
            "atoms": atoms, 
            "coordinates": coordinates.astype(np.float32),
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

def collate_tokens_2d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    sz = (len(values), size, size, ) + values[0].size()[2:]
    res = values[0].new(*sz).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):, size - len(v):] if left_pad else res[i][:len(v), :len(v)])
    return res

class RightPadDataset2D(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return collate_tokens_2d(
            samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8
        )

def collate_cross_2d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 2d tensors into a padded 2d tensor."""
    size_h = max(v.size(0) for v in values)
    size_w = max(v.size(1) for v in values)
    if pad_to_multiple != 1 and size_h % pad_to_multiple != 0:
        size_h = int(((size_h - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    if pad_to_multiple != 1 and size_w % pad_to_multiple != 0:
        size_w = int(((size_w - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size_h, size_w).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][size_h - v.size(0) :, size_w - v.size(1) :]
            if left_pad
            else res[i][: v.size(0), : v.size(1)],
        )
    return res


class CoordOptimDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    # @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        coord = pca_based_rotation(self.dataset[idx].cpu().numpy())
        return coord
    

class NumpyDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.set_epoch(None)

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.atoms in self.dataset[idx]:
            atoms = np.array(self.dataset[idx][self.atoms])
            assert len(atoms) > 0
            coordinates = np.array(self.dataset[idx][self.coordinates])
        return {'atoms': atoms, 'coordinates': coordinates.astype(np.float32), }
    


class NoiseCoordDataset(BaseWrapperDataset):
    def __init__(self, dataset, noise_level=0.01):
        self.noise_level = noise_level
        super().__init__(dataset)

    def __len__(self):
        return len(self.dataset)
    
    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.dataset[idx] + np.random.normal(0, self.noise_level, self.dataset[idx].shape)
    

# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import torch
from unicore.data import Dictionary
from functools import lru_cache


class TokenizeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dictionary: Dictionary,
        max_seq_len: int=512,
    ):
        self.dataset = dataset
        self.dictionary = dictionary
        self.max_seq_len = max_seq_len

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        raw_data = self.dataset[index]
        try:
            print(f"AssertionError at index {index}: raw_data length is {len(raw_data)}, max_seq_len is {self.max_seq_len}")
            assert len(raw_data) < self.max_seq_len and len(raw_data) > 0
        except AssertionError:
            print(f"AssertionError at index {index}: raw_data length is {len(raw_data)}, max_seq_len is {self.max_seq_len}")
            raise
        return torch.from_numpy(self.dictionary.vec_index(raw_data)).long()

