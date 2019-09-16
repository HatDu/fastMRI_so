from core.dataset import transforms
import numpy as np
class DataTransform:
    def __init__(self, mask_func, resolution, which_challenge, use_seed=True, crop=False, crop_size=320):
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.crop = crop
        self.crop_size = crop_size
    def __call__(self, kspace, target, norm, fname):
        kspace = transforms.to_tensor(kspace)
        # print(kspace.shape)
        image = transforms.ifft2(kspace)
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))

        if self.crop:
            ih = np.random.randint(0, self.resolution - self.crop_size)
            iw = np.random.randint(0, self.resolution - self.crop_size)
            image = image[..., ih: ih + self.crop_size, iw: iw + self.crop_size]
            if target is not None:
                target = target[..., ih: ih + self.crop_size,iw: iw + self.crop_size]

        kspace = transforms.fft2(image)
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        sub_imgk, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        # Inverse Fourier Transform to get zero filled solution
        sub_img = transforms.ifft2(sub_imgk)

        # # Normalize input
        sub_imgk, mean, std = transforms.normalize_instance(transforms.complex_abs(sub_imgk), eps=1e-11)
        sub_img = transforms.normalize(sub_img, mean, std, eps=1e-11)
        image = transforms.normalize(image, mean, std, eps=1e-11)
        kspace = transforms.normalize(kspace, mean, std, eps=1e-11)
        if target is not None:
            target = transforms.to_tensor(target)
            target = transforms.normalize(target, mean, std, eps=1e-11)  

        

        data = [sub_img, sub_imgk, image, kspace, target, mask]
        norm = [mean, std, norm]
        file_info = [fname]
        return [data, norm, file_info]