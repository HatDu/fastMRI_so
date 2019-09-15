from core.dataset import transforms
import numpy as np
import torch
class DataTransform:
    def __init__(self, mask_func, resolution, which_challenge, fusion='zero_mean', use_seed=True, crop=False, crop_size=96):
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.crop = crop
        self.crop_size = crop_size
        self.fusion = fusion
    def __call__(self, kspaces, center_id, targets, norm, fname, slice):
        kspaces = transforms.to_tensor(kspaces)
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname+str(slice)))
        images, imageks, mimages, mimageks, masks = [], [], [], [], []
        raw_images = transforms.ifft2(kspaces)
        raw_images = transforms.complex_center_crop(raw_images, (self.resolution, self.resolution))
        c_mimg, c_mimgk, c_image, c_imagek, c_mask = None, None, None, None, None
        c_target = targets[center_id]
        dist = []
        for i in range(kspaces.size(0)):
            img = raw_images[i]
            imgk = transforms.fft2(img)
            mimgk, mask = transforms.apply_mask(imgk, self.mask_func, seed)
            mimg = transforms.ifft2(mimgk)
            if i == center_id:
                c_image = img
                c_imagek = imgk
                c_mimg = mimg
                c_mimgk = mimgk
                c_mask = mask
            images.append(img.view(1, *img.size()))
            imageks.append(imgk.view(1, *imgk.size()))
            mimages.append(mimg.view(1, *mimg.size()))
            mimageks.append(mimgk.view(1, *mimgk.size()))
            masks.append(mask.view(1, *mask.size()))
            dist.append(abs(center_id-i))
        if self.fusion:
            pass
        _, mean, std = transforms.normalize_instance(transforms.complex_abs(c_mimg), eps=1e-11)
        c_mimg = transforms.normalize(c_mimg, mean, std, eps=1e-11)
        c_mimgk = transforms.normalize(c_mimgk, mean, std, eps=1e-11)
        c_image = transforms.normalize(c_image, mean, std, eps=1e-11)
        c_imagek = transforms.normalize(c_imagek, mean, std, eps=1e-11)

        
        if c_target is not None:
            c_target = transforms.to_tensor(c_target)
            # Normalize target
            c_target = transforms.normalize(c_target, mean, std, eps=1e-11)

        data = [c_mimg, c_mimgk, c_image, c_imagek, mask, c_target]
        norm = [mean, std, norm]
        file_info = [fname, slice]
        return [data, norm, file_info]