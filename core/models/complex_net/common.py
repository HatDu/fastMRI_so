import torch
from torch import nn
from core.dataset import transforms
from torch.nn import functional as F
class complex_conv2d(nn.Module):
    def __init__(self, in_chans, out_chans, ksize=3, activation=True, norm=None):
        super().__init__()
        self.conv_real = nn.Conv2d(in_chans, out_chans, ksize, padding=ksize//2)
        self.conv_imag = nn.Conv2d(in_chans, out_chans, ksize, padding=ksize//2)
        self.activation = None
        if activation:
            self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        '''
        x: B, C, H, W, 2
        '''
        x_real = x[..., 0]
        x_imag = x[..., 1]
        a = self.conv_real(x_real)
        b = self.conv_real(x_imag)
        c = self.conv_imag(x_real)
        d = self.conv_imag(x_imag)
        # expand dimension
        B, C, H, W = a.size()
        out_real = (a - d).view((B, C, H, W, 1))
        out_imag = (b + c).view((B, C, H, W, 1))
        out = torch.cat((out_real, out_imag), -1)
        if self.activation:
            out = self.activation(out)
        return out

def data_consistency(xgen, xk0, mask, noise_lvl=None):
    xgen_fft = transforms.fft2(xgen)
    v = noise_lvl
    if v:  # noisy case
        out_fft = (1 - mask) * xgen_fft + (mask * xgen_fft + v * xk0) / (1 + v)
    else:  # noiseless case
        out_fft = (1 - mask) * xgen_fft + xk0
    out_img = transforms.ifft2(out_fft)
    return out_img

def max_pool2d(data, kernel_size):
    real = data[...,0]
    imag = data[...,1]
    
    pool_real = F.max_pool2d(real, kernel_size=2)
    pool_imag = F.max_pool2d(imag, kernel_size=2)
    B, C, H, W = pool_real.size()
    pool_real = pool_real.view(B, C, H, W, 1)
    pool_imag = pool_imag.view(B, C, H, W, 1)
    pool = torch.cat((pool_real, pool_imag), -1)
    return pool

def interpolate(data, scale_factor=2, mode='bilinear', align_corners=False):
    real = data[...,0]
    imag = data[...,1]
    
    pool_real = F.interpolate(real, scale_factor=2, mode=mode, align_corners=align_corners)
    pool_imag = F.interpolate(imag, scale_factor=2, mode=mode, align_corners=align_corners)
    B, C, H, W = pool_real.size()
    pool_real = pool_real.view(B, C, H, W, 1)
    pool_imag = pool_imag.view(B, C, H, W, 1)
    pool = torch.cat((pool_real, pool_imag), -1)
    return pool