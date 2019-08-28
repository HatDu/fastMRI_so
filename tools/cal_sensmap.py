import glob
import bart
import h5py
import argparse
import os
import shutil
from tqdm import tqdm
import time
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('src', type=str)
parser.add_argument('dest', type=str)

args = parser.parse_args()

fnames = glob.glob(args.src+'/*')
total_time = 0
for name in fnames:
    with h5py.File(name, 'r') as data:
        n = os.path.split(name)[-1]
        kspace = data['kspace']
        start = time.time()
        sense_maps = [bart.bart(1, f'ecalib -d0 -m1', kspace[i]) for i in range(kspace.shape[0])]
        
        end = time.time()
        print(name, end-start)
        sense_maps = np.array(sense_maps)
        total_time += (end-start)/sense_maps.shape[0]
        with h5py.File(os.path.join(args.dest, n), 'w') as d:
            d['sensmaps'] = sense_maps
print('average time:', total_time/len(fnames))
# average time: 1.3110615470889857/slice