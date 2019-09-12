import torch
from torch import nn
from core.dataset import transforms
from torch.nn import functional as F
class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, ksize=3):
        super().__init__()
        self.convl = nn.Conv2d(in_chans, out_chans, ksize, padding=ksize//2)
        self.bn = nn.BatchNorm2d(out_chans)
        self.activation = nn.ReLU(inplace=False)
    
    def forward(self, x):
        out = self.convl(x)
        out = self.bn(out)
        out = self.activation(out)
        return out