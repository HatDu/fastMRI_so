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
            subimg, subimgk, image, imagek, mask, target = data
            mean, std, norm = norm
            fnames, slices = file_info
            recons = model(subimg, subimgk, mask)
            # recons = subimg
            # recons = image
            mean = mean.view(subimg.size(0), 1, 1, 1).to(recons.device)
            std = std.view(subimg.size(0), 1, 1, 1).to(recons.device)
            recons = recons*std + mean
            recons = recons.permute(0,2,3,1)
            # recons = masked_image
            # b, c, h, w, _ = recons.shape
            # mean = mean.view(b, 1, 1, 1, 1).to(recons.device)
            # std = std.view(b, 1, 1, 1, 1).to(recons.device)
            # recons = recons*std + mean
            # recons = transforms.ifft2(recons)
            recons = transforms.complex_abs(recons)
            recons = recons.cpu()
            for i in range(recons.shape[0]):
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