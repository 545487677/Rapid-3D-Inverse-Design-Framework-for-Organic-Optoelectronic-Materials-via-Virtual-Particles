# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import pandas as pd
from unicore.data import BaseWrapperDataset
import copy


class PBCAtomFeaDataset(BaseWrapperDataset):
    def __init__(self, atom_dataset, atom_fea_dict, num_feat='atom_feat_num', cat_feat='atom_feat_cat'):
        self.dataset = atom_dataset
        self.atom_num_feat, self.atom_cat_feat, self.num_sz, self.cat_sz = self.load_atom_fea(atom_fea_dict)
        self.num_feat = num_feat
        self.cat_feat = cat_feat

    def load_atom_fea(self, atom_feat):
        atom_num_feat, atom_cat_feat = atom_feat['num'], atom_feat['cat']
        atom_num_feat_sz, atom_cat_feat_sz = len(atom_feat['meta_info']['num_cols']), len(atom_feat['meta_info']['cat_cols'])
        return atom_num_feat, atom_cat_feat, atom_num_feat_sz, atom_cat_feat_sz

    def __getitem__(self, index):
        tokens = self.dataset[index]
        num_feat = np.array([self.atom_num_feat.get(token, np.zeros(self.num_sz)) for token in tokens], dtype=np.float32)
        cat_feat = np.array([self.atom_cat_feat.get(token, np.zeros(self.cat_sz)) for token in tokens], dtype=int)
        return {self.num_feat: num_feat, self.cat_feat: cat_feat}
    
class PBCAtomCatFeaDataset(BaseWrapperDataset):
    def __init__(self, atom_cat_dataset):
        self.dataset = atom_cat_dataset
        self.max_cnt = np.array([18, 7, 2, 6, 10, 14, 32, 1, 5, 9, 13, 22]) + 1
        self.max_cnt = np.cumsum(self.max_cnt)
        self.sz = len(self.max_cnt)

    def __getitem__(self, index):
        atom_cat = self.dataset[index]
        new_atom_cat = copy.deepcopy(atom_cat)
        for i in range(self.sz):
            new_atom_cat[:,i] += self.max_cnt[i]
        return new_atom_cat
    