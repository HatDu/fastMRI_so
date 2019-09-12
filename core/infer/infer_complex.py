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
        for batch in tqdm(data_loader):
            data, norm, file_info = batch
            masked_image, masked_imagek, target_image, target_imagek, mask, target_rss = data
            mean, std, norm = norm
            fnames, slices = file_info
            recons = model(masked_image, masked_imagek, mask)
            # recons = masked_image
            b, c, h, w, _ = recons.shape
            mean = mean.view(b, 1, 1, 1, 1).to(recons.device)
            std = std.view(b, 1, 1, 1, 1).to(recons.device)
            recons = recons*std + mean
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