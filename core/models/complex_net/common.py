import torch
from torch import nn
from core.dataset import transforms
class complex_conv2d(nn.Module):
    def __init__(self, in_chans, out_chans, ksize, activation=True, norm=None):
        super().__init__()
        self.conv_real = nn.Conv2d(in_chans, out_chans, ksize, padding=ksize//2)
        self.conv_imag = nn.Conv2d(in_chans, out_chans, ksize, padding=ksize//2)
        self.activation = None
        if activation:
            self.activation = nn.ReLU()
    
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
    # print(out_fft.size(), '1')
    out_img = transforms.ifft2(out_fft)
    # print(out_img.size(), '2')
    return out_img