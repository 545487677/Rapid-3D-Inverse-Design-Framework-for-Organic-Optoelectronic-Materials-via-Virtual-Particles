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
from unicore.data import BaseWrapperDataset

logger = logging.getLogger(__name__)

class RemoveHydrogenDataset(BaseWrapperDataset):
    def __init__(self, dataset, atoms, coordinates, remove_hydrogen=False, remove_polar_hydrogen=False, set_frag=False, set_residue=False):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.remove_hydrogen = remove_hydrogen
        self.remove_polar_hydrogen = remove_polar_hydrogen
        self.set_frag = set_frag
        self.set_residue = set_residue
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):

        if  self.dataset[index][self.atoms] is not None:
            dd = self.dataset[index].copy()
            atoms = dd[self.atoms]
            coordinates = dd[self.coordinates]
            if self.set_residue:
                residue = dd['residue']

            if self.remove_hydrogen:
                mask_hydrogen = atoms != 'H'
                atoms = atoms[mask_hydrogen]
                coordinates = coordinates[mask_hydrogen]
                if self.set_residue:
                    residue = residue[mask_hydrogen]

            if not self.remove_hydrogen and self.remove_polar_hydrogen:
                end_idx = 0 
                for i, atom in enumerate(atoms[::-1]):
                    if atom != 'H':
                        break
                    else:
                        end_idx = i + 1
                if end_idx != 0:
                    atoms = atoms[:-end_idx]
                    coordinates = coordinates[:-end_idx]
                    if self.set_residue:
                        residue = residue[:-end_idx]
            dd[self.atoms] = atoms
            dd[self.coordinates] = coordinates#.astype(np.float32)
            if self.set_residue:
                dd['residue'] = residue
            if self.set_frag:
                dd['frag_atom_id'] = atoms
            dd['atom_num'] = len(atoms)
        else:
            dd = self.dataset[index].copy()
            dd['atom_num'] = 0
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

class RemoveHydrogenBondDataset(BaseWrapperDataset):
    def __init__(self, dataset, atoms, bond, remove_hydrogen=False, remove_polar_hydrogen=False):
        self.dataset = dataset
        self.atoms = atoms
        self.bond = bond
        self.remove_hydrogen = remove_hydrogen
        self.remove_polar_hydrogen = remove_polar_hydrogen
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = self.dataset[index][self.atoms]
        adj_mat = self.dataset[index][self.bond]

        if self.remove_hydrogen:
            mask = atoms != 'H'
            adj_mat = adj_mat[mask][:,mask]
        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != 'H':
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                adj_mat = adj_mat[:-end_idx][:,:-end_idx]
        return {self.atoms: atoms, self.bond: adj_mat.astype(int)}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)