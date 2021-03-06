"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random

import h5py
from torch.utils.data import Dataset
import numpy as np
class MRI_Data(Dataset):
    def __init__(self, transform, root, sample_rate=1.):

        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        self.examples = files

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = np.array(data['kspace'])
            target = np.array(data['target']) if 'target' in data else None
            norm, max_volume = data['norm']
            name = fname.name
            slice_no_str = name.split('_')[-1][:-3]
            slice_no = int(slice_no_str)
            name = name[:-len(slice_no_str)-1]+'.h5'
            return self.transform(kspace, target, norm, fname.name, slice_no)