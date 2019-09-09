from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
import h5py
import os
from core.dataset import transforms
def run_net(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(data_loader):
            masked_image, masked_kspace, target, targetk, mask, fnames, slices = data
            recons = model(masked_image.to(args.device), masked_kspace, mask)
            # recons = targetk
            # recons = transforms.ifft2(recons)
            recons = transforms.complex_abs(recons)
            recons = transforms.root_sum_of_squares(recons, 1)
            recons = recons.cpu()
            for i in range(recons.shape[0]):
                # recons[i] = recons[i] * std[i] + mean[i]
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