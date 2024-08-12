# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class NormalizeDataset(BaseWrapperDataset):
    def __init__(self, dataset, coordinates, normalize_coord=True):
        self.dataset = dataset
        self.coordinates = coordinates
        self.normalize_coord = normalize_coord  # normalize the coordinates.
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        coordinates = dd[self.coordinates]
        # normalize
        if self.normalize_coord:
            coordinates = coordinates - coordinates.mean(axis=0)
            dd[self.coordinates] = coordinates
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
    
class NormalizeAndAlignDataset(BaseWrapperDataset):
    def __init__(self, dataset, coordinates, coordinates_s0, coordinates_s1, **unused):
        self.dataset = dataset
        self.coordinates = coordinates
        self.coordinates_s0 = coordinates_s0
        self.coordinates_s1 = coordinates_s1
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        coordinates = dd[self.coordinates]
        coordinates_s0 = torch.from_numpy(dd[self.coordinates_s0]).float()
        coordinates_s1 = torch.from_numpy(dd[self.coordinates_s1]).float()
        # normalize
        _mean = coordinates.mean(axis=0)
        coordinates = coordinates - _mean
        coordinates_s0 = coordinates_s0 - _mean
        coordinates_s1 = coordinates_s1 - _mean
        dd[self.coordinates] = coordinates
        dd[self.coordinates_s0] = coordinates_s0
        dd[self.coordinates_s1] = coordinates_s1
        dd['coord_center'] = _mean
        return dd
    
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)