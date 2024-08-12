# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import lmdb
import os
import pickle
import torch
import numpy as np
from functools import lru_cache
import logging
from unicore.data import BaseWrapperDataset, data_utils

logger = logging.getLogger(__name__)

class NormalizeDataset(BaseWrapperDataset):
    def __init__(self, dataset, pocket_coordinates, coordinates, normalize_coord=True):
        self.dataset = dataset
        self.coordinates = coordinates
        self.pocket_coordinates = pocket_coordinates
        self.normalize_coord = normalize_coord # normalize the coordinates.
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        if  self.dataset[index][self.coordinates] is not None:
            dd = self.dataset[index].copy()
            coordinates = dd[self.coordinates]
            pocket_coordinates = dd[self.pocket_coordinates]
            
            # normalize
            if self.normalize_coord:
                coordinates_mean = pocket_coordinates.numpy().mean(axis=0)
                dd['coordinates_mean'] = coordinates_mean
                coordinates = coordinates - coordinates_mean
                dd[self.coordinates] = coordinates
                pocket_coordinates = pocket_coordinates - torch.from_numpy(coordinates_mean)
                dd[self.pocket_coordinates] = pocket_coordinates
                dd['all_pocket_coordinates'] = dd['all_pocket_coordinates' ] - torch.from_numpy(coordinates_mean)

                if dd['sphere_centre'] is not None:
                    # print('????')
                    # dd['space'] = dd['space'] - coordinates_mean
                    dd['sphere_centre'] = dd['sphere_centre'] - coordinates_mean
                else:
                    # dd['space'] = None
                    dd['sphere_centre'] = None
        else:

            dd = self.dataset[index].copy()
            pocket_coordinates = dd[self.pocket_coordinates]

            if self.normalize_coord:
                coordinates_mean = pocket_coordinates.numpy().mean(axis=0)
                dd['coordinates_mean'] = coordinates_mean
                pocket_coordinates = pocket_coordinates - torch.from_numpy(coordinates_mean)
                dd[self.pocket_coordinates] = pocket_coordinates
                dd['all_pocket_coordinates'] = dd['all_pocket_coordinates' ] - torch.from_numpy(coordinates_mean)
                if dd['sphere_centre'] is not None:
                    # dd['space'] = dd['space'] - coordinates_mean
                    dd['sphere_centre'] = dd['sphere_centre'] - coordinates_mean
                else:
                    # dd['space'] = None
                    dd['sphere_centre'] = None

        
        # if dd['add_num'] == 10 :
        #     with open(dd['pdbid']+'_pocket.xyz','w') as w:
        #         pocket_atoms = dd['pocket_atoms'].tolist()
        #         pocket_coordinates = pocket_coordinates.numpy().tolist()
        #         for i in range(len(pocket_atoms)):
        #             w.write(str(pocket_atoms[i])+' '+ str(pocket_coordinates[i][0]) +' '+ str(pocket_coordinates[i][1])+ ' ' + str(pocket_coordinates[i][2]) +'\n')
            
        #     with open(dd['pdbid']+'_cavity.xyz','w') as w:
        #         sphere_centre = dd['sphere_centre'].tolist()
        #         half_bin_len = 1
        #         sphere_dict = {}
        #         for i in range(len(sphere_centre)):
        #             for x in [-1, 1]:
        #                 for y in [-1, 1]:
        #                     for z in [-1, 1]:
        #                         sphere_centre_t1 = sphere_centre[i][0]+ x*half_bin_len 
        #                         sphere_centre_t2 = sphere_centre[i][1]+ y*half_bin_len 
        #                         sphere_centre_t3 = sphere_centre[i][2]+ z*half_bin_len
        #                         sphere_centre_t_str = str(sphere_centre_t1)+' '+str(sphere_centre_t2) + ' '+str(sphere_centre_t3)
        #                         if sphere_centre_t_str not in sphere_dict:
        #                             sphere_dict[sphere_centre_t_str] = 1

        #         for item in sphere_dict:
        #              w.write(str('C')+' '+ item +'\n')







        
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

class NormalizeDockingPoseDataset(BaseWrapperDataset):
    def __init__(self, dataset, holo_coordinates, holo_pocket_coordinates, normalize_coord=True):
        self.dataset = dataset
        self.holo_coordinates = holo_coordinates
        self.holo_pocket_coordinates = holo_pocket_coordinates
        self.normalize_coord = normalize_coord # normalize the coordinates.
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        holo_coordinates = dd[self.holo_coordinates]
        holo_pocket_coordinates = dd[self.holo_pocket_coordinates]
        self.holo_center_coordinates = 'holo_center_coordinates'

        # normalize coordinates and pocket coordinates for holo models
        if self.normalize_coord:
            holo_center_coordinates = holo_pocket_coordinates.mean(axis=0)
            holo_pocket_coordinates = holo_pocket_coordinates - holo_center_coordinates
            holo_coordinates = holo_coordinates - holo_center_coordinates
            dd[self.holo_pocket_coordinates] = holo_pocket_coordinates.astype(np.float32)
            dd[self.holo_coordinates] = holo_coordinates.astype(np.float32)
            dd[self.holo_center_coordinates] = holo_center_coordinates.astype(np.float32)

        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)