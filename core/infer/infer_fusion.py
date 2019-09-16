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
            fnames, slices = file_info
            subimg, subimgk, image, imagek, mask, target = data
            mean, std, norm = norm
            output = model(subimg)
            # output = subimg     
            output = output.squeeze(1)
            mean = mean.view(subimg.size(0), 1, 1).to(output.device)
            std = std.view(subimg.size(0), 1, 1).to(output.device)
            output = output*std + mean

            recons = output.unsqueeze(1)
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