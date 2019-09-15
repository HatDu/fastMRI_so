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
        files = pd_list + pdfs_list
        self.instance_num = 0
        for fname in sorted(files):
            with h5py.File(fname, 'r') as data:
                kspace = data['kspace']
                acq = data.attrs['acquisition'] if 'acquisition' in data.attrs else 'None'
                if acq in self.acquisition:
                    self.instance_num += 1
                    num_slices = kspace.shape[0]
                    self.examples += [(fname, slice) for slice in range(num_slices)]
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            max_slices = len(data['kspace'])
            neigh_l = max(slice-self.neigh, 0)
            neight_r = min(max_slices, slice+self.neigh+1)
            kspace = data['kspace'][neigh_l: neight_r]
            target = data[self.recons_key][neigh_l: neight_r] if self.recons_key in data else None
            center_id = slice - neigh_l
            return self.transform(kspace, center_id, target, data.attrs['norm'], fname.name, slice)