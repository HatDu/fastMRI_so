import argparse
import os
import pathlib
import h5py
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='data/multicoil_train/')
parser.add_argument('--output_dir', default='data/multicoil_splits_train/')

args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
pathlib.Path(output_dir).mkdir(exist_ok=True)

for fname in tqdm(os.listdir(input_dir)):
    fpath = os.path.join(input_dir, fname)
    name = fname.split('.')[0]
    with h5py.File(fpath, 'r') as data:
        kspace = data['kspace']
        target = data['reconstruction_rss']
        norm, max_volume = data.attrs['norm'], data.attrs['max']
        acquisition =  data.attrs['acquisition']

        for i, k in enumerate(kspace):
            t = target[i]
            f = h5py.File('%s_%d.h5'%(os.path.join(output_dir, name), i), 'w')
            f['kspace'] = k
            f['target'] = t
            f['norm'] = [norm, max_volume]
            f['acquisition'] = acquisition
            f.close()
