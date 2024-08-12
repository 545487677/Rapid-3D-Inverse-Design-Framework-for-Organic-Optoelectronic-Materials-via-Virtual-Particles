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


class ReorderDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, pocket_dict_name=None, normalize_coord=True, rotation=False):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.set_epoch(None) # None TODO:None
        self.pocket_dict_name = pocket_dict_name
        self.normalize_coord = normalize_coord
        self.rotation = rotation

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        if self.atoms in self.dataset[index]:
            atoms = np.array(self.dataset[index][self.atoms])
            assert len(atoms) > 0
            if isinstance(self.dataset[index][self.coordinates],list):
                size = len(self.dataset[index][self.coordinates])
                with data_utils.numpy_seed(self.seed, epoch, index):
                    sample_idx = np.random.randint(size)
                coordinates = self.dataset[index][self.coordinates][sample_idx]
            else:
                coordinates = np.array(self.dataset[index][self.coordinates])
            # normalize
            coordinates_mean = None
            if self.normalize_coord: 
                coordinates_mean = coordinates.mean(axis=0)
                coordinates = coordinates - coordinates.mean(axis=0) 
            if self.rotation:
                center = coordinates.mean(axis=0)
                distance = np.linalg.norm(coordinates - center,axis=1)
                atom_1 = np.argmax(distance)
                vec1 = center - coordinates
                vec2 = coordinates[atom_1] - coordinates
                distance = np.linalg.norm(np.abs(np.cross(vec1, vec2)),  axis=1)
                norm = np.linalg.norm(center - coordinates[atom_1])
                # print('distance: ', distance.shape)
                distance = distance / norm
                atom_2 = np.argmax(distance)

                assert not(atom_2 == atom_1), (atom_1, atom_2, coordinates, len(coordinates))

                axis_x = coordinates[atom_1] - center
                axis_y = coordinates[atom_2] - center
                axis_z = np.cross(axis_x, axis_y)

                axis_x = axis_x/np.linalg.norm(axis_x).reshape(1,-1)
                axis_y = axis_y/np.linalg.norm(axis_y).reshape(1,-1)
                axis_z = axis_z/np.linalg.norm(axis_z).reshape(1,-1)

                matrix = np.concatenate((axis_x, axis_y, axis_z),axis=0)
                coordinates = coordinates.dot(matrix)
        else:
            atoms = None
            coordinates = None
            coordinates_mean = None



             
        if 'atoms_id' in self.dataset[index]:
            frag_atom = self.dataset[index]['atoms_id']          
        else:
            frag_atom = None
        if 'bond' in self.dataset[index]:
            bond_type = self.dataset[index]['bond']
        else:
            bond_type = None
        if 'pocket_atoms' in self.dataset[index]:
            if self.pocket_dict_name == 'pocket_dict_coarse.txt' or self.pocket_dict_name is None:
                pocket_atoms = np.array([a[0] for a in self.dataset[index]['pocket_atoms']])
            elif self.pocket_dict_name == 'dict_fine.txt':
                pocket_atoms = np.array([a[0] if len(a)==1 or a[0] == 'H' else a[:2] for a in self.dataset[index]['pocket_atoms']])
        else:
            pocket_atoms = None
        
        frag_coord = None
        if 'pocket_coordinates' in self.dataset[index]:
            if isinstance(self.dataset[index]['pocket_coordinates'],list):
                pocket_coordinates = np.array(self.dataset[index]['pocket_coordinates'][0])
            else:
                pocket_coordinates = self.dataset[index]['pocket_coordinates']
        else:
            pocket_coordinates = None
        
        if 'smi' in self.dataset[index]:
            smiles = self.dataset[index]['smi']
        elif 'smiles' in self.dataset[index]:
            smiles = self.dataset[index]['smiles']
        else:
            smiles = None
        
        pdbid = None
        if 'pdbid' in self.dataset[index]:
            pdbid = self.dataset[index]['pdbid'].replace('/','_')
        
        residue = None
        if 'residue' in self.dataset[index]:
            residue = np.array(self.dataset[index]['residue'])
            assert len(residue) == len(pocket_atoms)
        

        sphere_centre = None
        if 'sphere_centre' in self.dataset[index]:
            sphere_centre = self.dataset[index]['sphere_centre']
        space = None
        if 'space' in self.dataset[index]:
            space = self.dataset[index]['space']
        
        pred_num = 0
        if 'pred_num' in self.dataset[index]:
            pred_num = float(self.dataset[index]['pred_num'])
        
        add_num = 0
        if 'add_num' in self.dataset[index]:
            add_num = self.dataset[index]['add_num']
        

        if coordinates is not None:
            coordinates = coordinates.astype(np.float32)

        return {'atoms': atoms, 'coordinates': coordinates, 'frag_atom_id': frag_atom, \
        'bond': bond_type, 'smi':smiles, 'mol_idx':index,\
        'pocket_atoms': pocket_atoms, 'pocket_coordinates': pocket_coordinates, 'frag_coord': frag_coord, \
        'coordinates_mean': coordinates_mean, 'pdbid': pdbid, 'sphere_centre':sphere_centre, 'space': space, \
        'residue': residue, 'add_num': add_num, 'pred_num': pred_num}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)