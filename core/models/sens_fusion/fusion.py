import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        return self.layers(input)


class SigmoidFusionNet(nn.Module):
    def __init__(self, mid_chans=32, num_blocks=4, drop_prob=0.):
        super().__init__()
        assert num_blocks >= 2
        self.feature_conv = [ConvBlock(1, mid_chans, drop_prob)]
        for i in range(num_blocks-2):
            self.feature_conv.append(ConvBlock(mid_chans, mid_chans, drop_prob))
        self.feature_conv.append(ConvBlock(mid_chans, 1, drop_prob))
        self.feature_conv = nn.Sequential(*self.feature_conv)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        '''
        x: B,C,H,W
        '''
        B,C,H,W = x.size()
        tmp = x.view(B*C, -1, H, W)
        feature = self.feature_conv(tmp)
        feature = feature.view(B, C, H, W)
        sens_map = self.activation(feature)
        return sens_map