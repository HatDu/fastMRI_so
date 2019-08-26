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
    def __init__(self, transform, root, challenge, acquisition, sample_rate=1.):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.acquisition = acquisition
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        self.instance_num = 0
        for fname in sorted(files):
            with h5py.File(fname, 'r') as data:
                kspace = data['kspace']
                acq = data.attrs['acquisition'] if 'acquisition' in data.attrs else 'None'
                # print(acq, self.acquisition)
                if acq in self.acquisition:
                    # print(acq, acquisition)
                    self.instance_num += 1
                    num_slices = kspace.shape[0]
                    self.examples += [(fname, slice) for slice in range(num_slices)]
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            norm = data.attrs['norm'] if 'norm' in data.attrs else None
            max_volume = data.attrs['norm'] if 'max' in data.attrs else None
            # acq = data.attrs['acquisition'] if 'acquisition' in data.attrs else 'None'
            # print(acq)
            return self.transform(kspace, target, norm, fname.name, slice)