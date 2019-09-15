import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
import core.models.DnCn.kspace_pytorch as cl


def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)


def relu():
    return nn.ReLU(inplace=True)


def conv_block(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):

    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
        # pad = dilation
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    def conv_i():
        return conv(nf,   nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == 'relu' else lrelu

    layers = [conv_1, nll()]
    for i in range(nd-2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)


class DnCn(nn.Module):
    def __init__(self, n_channels=2, nc=5, nd=5, **kwargs):
        super(DnCn, self).__init__()
        self.nc = nc
        self.nd = nd
        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        dcs = []

        conv_layer = conv_block

        for i in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs.append(cl.DataConsistencyInKspace(norm='ortho'))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            # x = self.dcs[i].perform(x, k, m)

        return x