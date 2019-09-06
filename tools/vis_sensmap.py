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
import pathlib
import matplotlib.pyplot as plt
import cv2

from core.dataset import create_infer_dataloader
from core.models import build_model
from tools.imshow import show_images
from torch.utils.data import DataLoader
from core.dataset import transforms

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
    results = []
    outdir = args.out_dir
    with torch.no_grad():
        for data in tqdm(data_loader):
            input, target, mean, std, norm, fnames, slices = data
            # input = input.unsqueeze(1).to(args.device)
            recons = model(input)
            
            x, sens_map, fusion, output = recons
            sens_map = sens_map.to('cpu').numpy()
            fusion = fusion.to('cpu').squeeze(1).numpy()
            x = x.to('cpu')
            output = output.to('cpu').squeeze(1).numpy()
            results.append([x, sens_map, fusion, output, target.numpy(), fnames, slices])

    for rst in results:
        x, sens_map, fusion, output, target, fnames, slices = rst
        for i, slce in enumerate(x):
            rss = transforms.root_sum_of_squares(slce)
            rss = rss.numpy()
            sens = sens_map[i]
            f = fusion[i]
            out = output[i]
            name = fnames[i]
            slice_no = slices[i]
            t = target[i]

            slce = slce[::4].numpy()
            sens = sens[::4]
            
            col = len(slce)
            row = 4
            num = 1
            plt.figure()
            for i, img in enumerate(slce):
                # plt.axis('off')
                plt.subplot(row, col, num)
                plt.imshow(img, cmap='gray')
                num += 1
                
            # [rss, f, out, t]+[rss-t, f-t, out-t, t-t]
            for i, img in enumerate(sens):
                # plt.axis('off')
                plt.subplot(row, col, num)
                plt.imshow(img, cmap='gray')
                num += 1
                
            for i, img in enumerate([rss, f, out, t]):
                # plt.axis('off')
                plt.subplot(row, col, num)
                plt.imshow(img, cmap='gray')
                num += 1
                
            for i, img in enumerate([rss-t, f-t, out-t, t-t]):
                # plt.axis('off')
                plt.subplot(row, col, num)
                plt.imshow(img, cmap='gray')
                num += 1
                
            fpath = os.path.join(outdir, 'sens', '%s_%d.png'%(name, slice_no))
            # spath = os.path.join(outdir, 'sens', '%s_%d_sens.png'%(name, slice_no))
            # plt.show()
            plt.savefig(fpath)
            # plt.imsave(spath, sens_img, cmap='gray')



def save_reconstructions(reconstructions, out_dir):
    out_dir = os.path.join(out_dir, 'sens')
    pathlib.Path(out_dir).mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        for i, r in enumerate(recons):
            fname = os.path.join(out_dir, fname+'_%d.png'%(i+1))
            plt.imsave(fname, r, cmap='gray')
        # with h5py.File(os.path.join(out_dir, fname), 'w') as f:
        #     f.create_dataset('reconstruction', data=recons)

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
    # sample
    dset = dataloader.dataset
    display_data = [dset[i] for i in range(0, len(dset), len(dset) // 16)]
    dataloader = DataLoader(
        dataset=display_data,
        **cfg.data.test.loader
    )

    run_unet(args, model, dataloader)
    
if __name__ == '__main__':
    main()
