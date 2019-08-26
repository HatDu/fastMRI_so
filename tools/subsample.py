import h5py
import argparse
import os
import shutil
parser = argparse.ArgumentParser()
parser.add_argument('src', type=str)
parser.add_argument('dest', type=str)
parser.add_argument('acq')
parser.add_argument('count', type=int)


acquisition_dict = dict(
    pd=['CORPD_FBK'],
    pdfs=['CORPDFS_FBK']
)
args = parser.parse_args()
acquisition = acquisition_dict[args.acq]
imnames = sorted(os.listdir(args.src))
count = 0

for i, name in enumerate(imnames):
    # print(i, name)
    src = os.path.join(args.src, name)
    with h5py.File(src, 'r') as data:
        dest = os.path.join(args.dest, name)
        acq = data.attrs['acquisition'] if 'acquisition' in data.attrs else 'None'
        if acq in acquisition:
            count += 1
            print(acq, dest, count)
            dest = os.path.join(args.dest, name)
            shutil.copy(src, dest)
            if count == args.count:
                break
