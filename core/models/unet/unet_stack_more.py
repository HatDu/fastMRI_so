import torch
from torch import nn
from torch.nn import functional as F

class BasicConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, dilation=1, drop_prob=0.):
        super().__init__()
        padding = (dilation*kernel_size -1)//2
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        return self.conv(input)

class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_cnns=3, drop_prob=0.):
        super().__init__()
        cnn_blocks = [BasicConv(in_chans, out_chans)]
        for i in range(1, num_cnns):
            cnn_blocks.append(BasicConv(out_chans, out_chans))
        self.conv = nn.Sequential(*cnn_blocks)
    
    def forward(self, x):
        return self.conv(x)


class UnetModel(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, num_cnns=4, drop_prob=0.):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, num_cnns)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, num_cnns)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, num_cnns)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, num_cnns)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, num_cnns)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        stack = []
        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        return self.conv2(output)

def get_model(name):
    return UnetModel