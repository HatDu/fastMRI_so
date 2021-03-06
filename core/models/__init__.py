import torch.nn as nn
import importlib
model_dict = dict(
    baseline_unet = 'core.models.unet',
    se_unet = 'core.models.sequeeze_excitation',
    sens_fusion = 'core.models.sens_fusion',
    complex_net = 'core.models.complex_net',
    cascade_net = 'core.models.cascadenet',
    CRNN_MRI='core.models.cascadenet_pytorch',
)

def build_model(cfg):
    model_cfg = cfg.model
    if model_cfg.name not in model_dict.keys():
        raise 'No such model type'
    module = importlib.import_module(model_dict[model_cfg.name])
    Model = module.import_model(model_cfg.id)
    model = Model(**model_cfg.params)
    print('model parameters:', sum(param.numel() for param in model.parameters()))
    return model.apply(weights_init)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()
