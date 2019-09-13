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
    def __call__(self, kspaces, center_id, targets, norm, fname, slices):
        kspaces = transforms.to_tensor(kspaces)
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname+str(slices[center_id])))
        images, imageks, mimages, mimageks, masks = [], [], [], [], []
        raw_images = transforms.ifft2(kspaces)
        raw_images = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        for i in range(kspaces.size(0)):
            img = raw_images[i]
            imgk = transforms.fft2(data)
            mimgk, mask = mask = transforms.apply_mask(imgk, self.mask_func, seed)
            mimg = transforms.ifft2(mimgk)

            images.append(img.view(1, *img.size()))
            imagesk.append(imgk.view(1, *imgk.size()))
            mimages.append(mimg.view(1, *mimg.size()))
            mimageks.append(mimgk.view(1, *mimgk.size()))
            masks.append(mask.view(1, *mask.size()))
        
        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        image = transforms.complex_abs(image)
        # # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)
        # # Normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)
        
        if target is not None:
            target = transforms.to_tensor(target)
            # Normalize target
            target = transforms.normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)   

        if self.crop:
            ih = np.random.randint(0, self.resolution - self.crop_size)
            iw = np.random.randint(0, self.resolution - self.crop_size)
            image = image[..., ih: ih + self.crop_size, iw: iw + self.crop_size]
            if target is not None:
                target = target[..., ih: ih + self.crop_size,iw: iw + self.crop_size]
        image = image.unsqueeze(0)
        return image, target, mean, std, norm, fname, slice