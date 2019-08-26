import argparse
import logging
import os
import pathlib
import random
import time
from collections import defaultdict
from tqdm import tqdm
import h5py
import numpy as np
import torch
from mmcv import Config

from core.dataset import create_infer_dataloader
from core.models import build_model
from tools.imshow import show_images


def parse_args():
    parser = argparse.ArgumentParser(description='infer')
    parser.add_argument('--cfg', help='config file path',
                        default='configs/baseline_unet.py')
    parser.add_argument('-c', '--ckpt', default='log/baseline_unet/best_model.pt')
    parser.add_argument('-m', '--mask', action='store_true', default=True)
    parser.add_argument('-a', '--acc', default='x4', type=str)
    parser.add_argument('-acq', '--acquisition', default='both')
    parser.add_argument('-i', '--input_dir', default=None)
    parser.add_argument('-o', '--out_dir', default='data/infer')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data_parallel', action='store_true', default=True)
    args = parser.parse_args()
    return args
acceleration_cfgs = dict(
    x4=dict(center_fractions=[0.08], accelerations=[4]),
    x8=dict(center_fractions=[0.04], accelerations=[8])
)
acquisition = dict(
    both=['CORPD_FBK', 'CORPDFS_FBK', 'None'],
    pd=['CORPD_FBK'],
    pdfs=['CORPDFS_FBK']
)
def run_unet(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(data_loader):
            input, target, mean, std, norm, fnames, slices = data
            input = input.unsqueeze(1).to(args.device)
            recons = model(input).to('cpu').squeeze(1)
            for i in range(recons.shape[0]):
                recons[i] = recons[i] * std[i] + mean[i]
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions

def save_reconstructions(reconstructions, out_dir):
    for fname, recons in reconstructions.items():
        with h5py.File(os.path.join(out_dir, fname), 'w') as f:
            f.create_dataset('reconstruction', data=recons)

def main():
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    cfg.device = args.device
    cfg.mask = args.mask
    cfg.data.test.mask.params = acceleration_cfgs[args.acc]
    cfg.acquisition = acquisition[args.acquisition]
    if args.input_dir is not None:
        cfg.data.test.dataset.params.root = args.input_dir
    # Log
    out_dir = args.out_dir
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Model
    model = build_model(cfg).to((args.device))
    # Data Parallel training
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['model'])

    # data
    dataloader = create_infer_dataloader(cfg)
    reconstructions = run_unet(args, model, dataloader)
    save_reconstructions(reconstructions, args.out_dir)
    
if __name__ == '__main__':
    main()
