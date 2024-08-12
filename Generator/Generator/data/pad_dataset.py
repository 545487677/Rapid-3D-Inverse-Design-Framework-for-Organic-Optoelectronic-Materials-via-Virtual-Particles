# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unicore.data import BaseWrapperDataset


def collate_tokens_2d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), (dst.numel(),src.numel())
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):, size - len(v):] if left_pad else res[i][:len(v), :len(v)])
    return res

class RightPadDataset2D(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx,left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
    def collater(self, samples):
        return collate_tokens_2d(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)

def collate_tokens_2d2(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size2 = max(v.size(1) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    size2 = size2 if pad_to_length is None else max(size2, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    if pad_to_multiple != 1 and size2 % pad_to_multiple != 0:
        size2 = int(((size2 - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, size2).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), (dst.numel(),src.numel())
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - v.size(0):, size2 - v.size(1):] if left_pad else res[i][:v.size(0), :v.size(1)])
    return res

class RightPadDataset2D2(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx,left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
    def collater(self, samples):
        return collate_tokens_2d2(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
    
def collate_tokens_coords(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, 3).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :,:] if left_pad else res[i][: len(v),:])
    return res

class RightPadDatasetCoord(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx,left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
    def collater(self, samples):
        return collate_tokens_coords(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)


def collate_tokens_3d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    #如果是个立方体
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)

    size2 = max(v.size(1) for v in values)
    size2 = size2 if pad_to_length is None else max(size2, pad_to_length)

    size3 = max(v.size(2) for v in values)
    size3 = size3 if pad_to_length is None else max(size3, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    if pad_to_multiple != 1 and size2 % pad_to_multiple != 0:
        size2 = int(((size2 - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    if pad_to_multiple != 1 and size3 % pad_to_multiple != 0:
        size3 = int(((size3 - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    
    res = values[0].new(len(values), size, size2, size3).fill_(pad_idx)


    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), (dst.numel(),src.numel())
        dst.copy_(src)
    
    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):, size2 - len(v[0]):, size3 - len(v[0][0]):] if left_pad else res[i][:len(v), :len(v[0]), :len(v[0][0])])

    return res

class RightPadDataset3D(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx,left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
    def collater(self, samples):
        return collate_tokens_3d(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)