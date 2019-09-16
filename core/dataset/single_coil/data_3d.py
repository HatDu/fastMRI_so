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
    def __init__(self, transform, root, challenge, acquisition, sample_num=-1, neigh=2):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.acquisition = acquisition
        self.neigh = neigh
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        files = sorted(list(pathlib.Path(root).iterdir()))
        pd_list = []
        pdfs_list = []
        for fname in sorted(files):
            with h5py.File(fname, 'r') as data:
                kspace = data['kspace']
                acq = data.attrs['acquisition'] if 'acquisition' in data.attrs else 'None'
                if acq == 'CORPD_FBK':
                    pd_list.append(fname)
                elif acq == 'CORPDFS_FBK':
                    pdfs_list.append(fname)
        if sample_num > 10:
            num_files = sample_num//2
            pd_list = pd_list[:min(num_files, len(pd_list))]
            pdfs_list = pdfs_list[:min(num_files, len(pdfs_list))]
        self.files = pd_list + pdfs_list
        self.instance_num = len(self.files)

    def __len__(self):
        return self.instance_num

    def __getitem__(self, i):
        fname = self.files[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace']
            target = data[self.recons_key] if self.recons_key in data else None
            return self.transform(np.array(kspace), np.array(target), data.attrs['norm'], fname.name)