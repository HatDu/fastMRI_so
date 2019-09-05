from core.dataset import transforms
import numpy as np
class DataTransform:
    def __init__(self, mask_func, resolution, which_challenge, use_seed=True, crop=False, crop_size=96):
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.crop = crop
        self.crop_size = crop_size
    def __call__(self, kspace, target, norm, fname, slice):
        kspace = transforms.to_tensor(kspace)
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        image = transforms.complex_abs(image)
        # # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image_rss = transforms.root_sum_of_squares(image)
        # # Normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image_rss, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)
        image_rss = image_rss.clamp(-6, 6)
        
        if target is not None:
            target = transforms.to_tensor(target)
            # Normalize target
            target = transforms.normalize(target, mean, std, eps=1e-11)
            # target = target.clamp(-6, 6)   

        if self.crop:
            ih = np.random.randint(0, self.resolution - self.crop_size)
            iw = np.random.randint(0, self.resolution - self.crop_size)
            image = image[..., ih: ih + self.crop_size, iw: iw + self.crop_size]
            image_rss = image[..., ih: ih + self.crop_size, iw: iw + self.crop_size]
            if target is not None:
                target = target[..., ih: ih + self.crop_size,iw: iw + self.crop_size]
        image_rss = image_rss.unsqueeze(0)
        return image, target, image_rss, mean, std, norm, fname, slice
        # return image, np.abs(target)