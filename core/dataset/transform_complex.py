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
        target_rss = target
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        
        image = transforms.ifft2(kspace)
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # normalize
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        target_image = image
        imagek = transforms.fft2(image)
        target_imagek = imagek
        masked_imagek, mask = transforms.apply_mask(imagek, self.mask_func, seed)
        masked_image = transforms.ifft2(masked_imagek)

        if target is not None:
            target = transforms.to_tensor(target)
        # print(norm)
        # print((image).mean((0,1,2)))
        # print(image.size(), masked_kspace.size(), targetk.size(), mask.size())
        # torch.Size([15, 320, 320, 2]) torch.Size([15, 320, 320, 2]) torch.Size([15, 320, 320, 2]) torch.Size([1, 1, 320, 1])
        data = [masked_image, masked_imagek, target_image, target_imagek, mask, target_rss]
        norm = [mean, std, norm]
        file_info = [fname, slice]
        return [data, norm, file_info]