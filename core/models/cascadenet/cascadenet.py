from .common import *

import torch
from torch import nn
from core.dataset import transforms

class ConvBlock_groups(nn.Module):
    def __init__(self, in_chans, inter_chans, out_chans, nb, ksize=3):
        super().__init__()
        assert nb >= 2
        conv_blocks = [ConvBlock(in_chans, inter_chans, ksize=ksize)]

        for i in range(nb-2):
            conv_blocks.append(ConvBlock(inter_chans, inter_chans, ksize=ksize))

        conv_blocks.append(ConvBlock(inter_chans, out_chans, ksize=ksize))
        self.conv_blocks = nn.Sequential(*conv_blocks)
    
    def forward(self, x):
        tmp = x
        out = self.conv_blocks(x)
        out += tmp
        return out

class ComplexNet(nn.Module):
    def __init__(self, in_chans, out_chans, inter_chans, nb, nc=5):
        super().__init__()
        self.conv_groups = nn.ModuleList([ConvBlock_groups(in_chans, inter_chans, out_chans, nb) for i in range(nc)])
        self.nc = nc
        
    def forward(self, x):
        '''
        x: B,C,H,W,2
        x0: B,C,H,W,2
        mask: 1,1,1,W,1
        '''
        x_i = x
        for i in range(self.nc):
            x_i = self.conv_groups[i](x_i)
        return x_i