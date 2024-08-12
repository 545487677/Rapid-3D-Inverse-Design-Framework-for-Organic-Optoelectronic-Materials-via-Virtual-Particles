import numpy as np
import torch
from functools import lru_cache

from unicore.data import BaseWrapperDataset


class PrependAndAppendCentroid(BaseWrapperDataset):

    def __init__(self, dataset, centroid=None, token=None, addeos=True):
        super().__init__(dataset)
        self.token = token
        self.centroid = centroid
        self.addeos = addeos

    # @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.centroid is None:
            item = self.dataset[idx]#.clone()
            if self.addeos:
                item = torch.cat([torch.mean(item,dim=0).unsqueeze(0), item, torch.mean(item,dim=0).unsqueeze(0)], dim=0)
            else:
                item = torch.cat([torch.mean(item,dim=0).unsqueeze(0), item,], dim=0)
        else:
            item = self.dataset[idx]
            if self.addeos:
                item = torch.cat([self.centroid[idx].unsqueeze(0), item, self.centroid[idx].unsqueeze(0)], dim=0)
            else:
                item = torch.cat([self.centroid[idx].unsqueeze(0), item], dim=0)
        return item