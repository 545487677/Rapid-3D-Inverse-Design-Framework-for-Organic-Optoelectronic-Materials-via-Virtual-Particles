from functools import lru_cache
from typing import List

import numpy as np
import torch

from unicore.data import BaseWrapperDataset


class PbcOffsetsDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        lattice_matrix: torch.utils.data,
    ):
        self.dataset = dataset
        self.lattice_matrix = lattice_matrix
        self.offsets = self.build_offsets()

    def build_offsets(self):
        from itertools import product
        candidates = [0, -1, 1]
        ret = list(product(candidates, candidates, candidates))
        assert len(ret) == 27 and ret[0] == (0, 0, 0)
        return np.array(ret, dtype=float)

    def __getitem__(self, index):
        return (self.offsets @ self.dataset[index][self.lattice_matrix]).astype(float)


class PbcDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        pbc_offsets_dataset: torch.utils.data.Dataset,
        coordinates: str,
        repeat_keys: List[str] = [],
    ):
        self.dataset = dataset
        self.pbc_offsets_dataset = pbc_offsets_dataset
        self.coordinates = coordinates
        self.repeat_keys = repeat_keys
        self.epoch = None

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        ret = self.dataset[index].copy()
        offsets = self.pbc_offsets_dataset[index]
        repeats = offsets.shape[0]

        def convert_to_torch(array):
            if array is np.ndarray:
                return torch.from_numpy(array)
            else:
                return array

        coords = convert_to_torch(ret[self.coordinates])
        natm = coords.shape[0]
        ret[self.coordinates] = (
            coords.reshape(natm, 1, 3) + offsets.reshape(1, -1, 3)
        ).reshape(-1, 3)

        for key in self.repeat_keys:
            tensor = convert_to_torch(ret[key])
            ret[key] = tensor.repeat([repeats] + [1] * (len(tensor.shape) - 1))

        return ret
