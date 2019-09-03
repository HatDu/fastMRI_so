from .fusion import SigmoidFusionNet
from .unet import UnetModel
from torch import nn
import torch
class FusionUnet(nn.Module):
    def __init__(self, rparams, fparams, residual=True, guid=False):
        super().__init__()
        self.residual = residual
        self.guid = guid
        self.fusion_net = SigmoidFusionNet(**fparams)
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
        if self.guid:
            return [fusion, output]
        return [x, sens_map, fusion, output]