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
        image = transforms.ifft2(kspace)
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        imagek = transforms.fft2(image)

        subimgk, mask = transforms.apply_mask(imagek, self.mask_func, seed)
        # Inverse Fourier Transform to get zero filled solution
        subimg = transforms.ifft2(subimgk)
        _, mean, std = transforms.normalize_instance(transforms.complex_abs(subimg), eps=1e-11)
        subimg = transforms.normalize(subimg, mean, std, eps=1e-11)
        subimgk = transforms.normalize(subimgk, mean, std, eps=1e-11)
        image = transforms.normalize(image, mean, std, eps=1e-11)
        imagek = transforms.normalize(imagek, mean, std, eps=1e-11)
        if target is not None:
            target = transforms.to_tensor(target)
            target = transforms.normalize(target, mean, std, eps=1e-11)

        if self.crop:
            ih = np.random.randint(0, self.resolution - self.crop_size)
            iw = np.random.randint(0, self.resolution - self.crop_size)
            image = image[..., ih: ih + self.crop_size, iw: iw + self.crop_size]
            if target is not None:
                target = target[..., ih: ih + self.crop_size,iw: iw + self.crop_size]
        # print(subimg.size(), subimgk.size(), image.size())
        data = [subimg.permute(2,0,1), subimgk.permute(2,0,1), image.permute(2,0,1), imagek.permute(2,0,1), mask.permute(2,0,1), target]
        norm = [mean, std, norm]
        file_info = [fname, slice]
        return [data, norm, file_info]