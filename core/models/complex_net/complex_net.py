import torch
from torch import nn
from core.dataset import transforms
from core.models.complex_net.common import complex_conv2d, data_consistency

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