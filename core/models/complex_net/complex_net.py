import torch
from torch import nn
from core.dataset import transforms
class complex_conv2d(nn.Module):
    def __init__(self, in_chans, out_chans, ksize=3, activation=True, norm=None):
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

class complex_conv2d_groups(nn.Module):
    def __init__(self, in_chans, inter_chans, out_chans, nb, ksize=3, activation=True, norm=None):
        super().__init__()
        assert nb >= 2
        conv_blocks = [complex_conv2d(in_chans, inter_chans, ksize=ksize, activation=activation, norm=norm)]

        for i in range(nb-2):
            conv_blocks.append(complex_conv2d(inter_chans, inter_chans, ksize=ksize, activation=activation, norm=norm))

        conv_blocks.append(complex_conv2d(inter_chans, out_chans, ksize=ksize, activation=activation, norm=norm))
        self.conv_blocks = nn.Sequential(*conv_blocks)
    
    def forward(self, x):
        return self.conv_blocks(x)

# def data_consistency(k, k0, mask, noise_lvl=None):
#     """
#     k    - input in k-space
#     k0   - initially sampled elements in k-space
#     mask - corresponding nonzero location
#     """
#     v = noise_lvl
#     if v:  # noisy case
#         out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
#     else:  # noiseless case
#         out = (1 - mask) * k + mask * k0
#     return out

def data_consistency(xgen, xk0, mask, noise_lvl=None):
    xgen_fft = transforms.fft2(xgen)
    v = noise_lvl
    if v:  # noisy case
        out_fft = (1 - mask) * xgen_fft + (mask * xgen_fft + v * xk0) / (1 + v)
    else:  # noiseless case
        out_fft = (1 - mask) * xgen_fft + xk0
    out_img = transforms.ifft2(out_fft)
    return out_img

class ComplexNet(nn.Module):
    def __init__(self, in_chans, out_chans, inter_chans, nb, nc, activation=True, norm=None, noise_lvl=None):
        super().__init__()
        self.conv_groups = nn.ModuleList([complex_conv2d_groups(in_chans, inter_chans, out_chans, nb, activation=activation) for i in range(nc)])
        self.dc = data_consistency
        self.nc = nc
        self.noise_lvl = noise_lvl
        
    def forward(self, x, xk0, mask):
        '''
        x: B,C,H,W,2
        x0: B,C,H,W,2
        mask: 1,1,1,W,1
        '''
        x_i = x
        for i in range(self.nc):
            x_i = self.conv_groups[i](x_i)
            x_i = self.dc(x_i, xk0, mask, self.noise_lvl)
        return x_i