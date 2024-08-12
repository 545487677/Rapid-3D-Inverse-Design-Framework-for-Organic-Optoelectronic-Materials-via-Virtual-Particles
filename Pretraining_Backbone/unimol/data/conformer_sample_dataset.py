# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from . import data_utils


class ConformerSampleDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
    ):
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
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        assert len(atoms) > 0
        if not isinstance(dd[self.coordinates], list):
            dd[self.coordinates] = [dd[self.coordinates]]
        size = len(dd[self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = dd[self.coordinates][sample_idx].astype(np.float32)
        assert len(atoms) == len(coordinates)
        dd[self.coordinates] = torch.from_numpy(coordinates)

        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

