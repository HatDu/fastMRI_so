import torch.nn as nn

class DB_Block(nn.Module):
    def __init__(self, nf=64, nblocks=4):
        super().__init__()

    def forward(self, x):
        
        pass
class VnMriReconstructionCell(nn.Module):
    def __init__(self, inchans, out_chans, norm=None, bias=None):
        super().__init__()
        self.real_weight = nn.Parameter(torch.randn(out_chans, in_chans, 3, 3))
        self.imag_weight = nn.Parameter(torch.randn(out_chans, in_chans, 3, 3))
        self.bias_real = None
        self.bias_imag = None
        self.activation = nn.Relu()
        self.norm = norm(out_chans)
    def forward(self, x):
        x_real = x[...,0]
        x_image = x[...,1]
        u_k_real = nn.functional.conv2d(x_real, self.real_weight, stride=1, padding=1)
        u_k_imag = nn.functional.conv2d(x_image, self.imag_weight, stride=1, padding=1)
        u_k = u_k_real + u_k_imag
        if self.norm:
            u_k = self.norm(u_k)
        f_u_k = self.activation(u_k)
        u_k_T_real = nn.functional.conv_transpose2d(f_u_k, self.real_weight, stride=1, padding=1)
        u_k_T_imag = nn.functional.conv_transpose2d(f_u_k, self.real_weight, stride=1, padding=1)
        return u_k
    

        
        