import argparse
import os
import numpy as np
import h5py
import pathlib
def readcfl(name):
    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline() # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    return a.reshape(dims, order='F') # column-major

	
def writecfl(name, array):
    h = open(name + ".hdr", "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
            h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d) # tranpose for column-major order
    d.close()

parser = argparse.ArgumentParser()
parser.add_argument('input_dir')
parser.add_argument('output_dir')

args = parser.parse_args()

input_dir = args.input_dir
outputdir = args.output_dir
pathlib.Path(outputdir).mkdir(exist_ok=True)
for fname in os.listdir(input_dir):
    data = h5py.File(os.path.join(input_dir, fname), 'r')
    kspace = data['kspace']
    writecfl(os.path.join(outputdir, fname), np.array(kspace))
