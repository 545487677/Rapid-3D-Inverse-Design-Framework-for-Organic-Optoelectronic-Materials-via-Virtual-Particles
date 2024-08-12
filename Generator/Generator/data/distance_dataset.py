
import numpy as np
import torch
from scipy.spatial import distance_matrix
from functools import lru_cache

from unicore.data import BaseWrapperDataset

class DistanceDataset(BaseWrapperDataset):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    # @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        pos = self.dataset[idx].view(-1, 3).numpy()
        # add eps to avoid zero distances
        # dist  = (pos.view(-1, 1, 3) - pos.view(1, -1, 3)).norm(dim=-1) + 1e-5
        dist = distance_matrix(pos, pos).astype(np.float32)
        return torch.from_numpy(dist)

class EdgeTypeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_types: int
    ):
        self.dataset = dataset
        self.num_types = num_types

    # @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        node_input = self.dataset[index].clone()
        offset = node_input.view(-1, 1) * self.num_types + node_input.view(1, -1)
        return offset

class EdgeType2dDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        pocketdataset: torch.utils.data.Dataset,
        num_types: int
    ):
        self.dataset = dataset
        self.pocketdataset = pocketdataset
        self.num_types = num_types

    # @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        node_input = self.dataset[index].clone()
        pocket_input = self.pocketdataset[index].clone()
        # offset = pocket_input.view(-1, 1) * self.num_types + node_input.view(1, -1)
        offset = node_input.view(-1, 1) * self.num_types + pocket_input.view(1, -1)
        return offset