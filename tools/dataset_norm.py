import argparse
import os
import h5py
parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('folder')

args = parser.parse_args()

folder = args.folder

for name in os.listdir(folder):
    path = os.path.join(folder, name)
    with h5py.File(path, 'r') as data:
        kspace = data['kspace']


