from .fusion import SigmoidFusionNetv2
from .unet import UnetModel
from torch import nn
import torch
class FusionUnet(nn.Module):
    def __init__(self, rparams, fparams, residual=True, guid=False):
        super().__init__()
        self.residual = residual
        self.guid = guid
        self.fusion_net = SigmoidFusionNetv2(**fparams)
        self.recon_net = UnetModel(**rparams)
        pass
    
    def forward(self, x):
        sens_map = self.fusion_net(x)
        fusion = x * sens_map
        fusion = torch.sum(fusion, 1, True)
        recon = self.recon_net(fusion)
        output = recon
        if self.residual:
            output = recon + fusion
        return [x, sens_map, fusion, output]