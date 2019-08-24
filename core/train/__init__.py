from torch import optim


def build_optimizer(cfg, model):
    optim_cfg = cfg.train.optimizer
    optimizer = None
    if optim_cfg.name == 'Adam':
        optimizer = optim.Adam
    else:
        optimizer = optim.RMSprop
    opt = optimizer(model.parameters(), **optim_cfg.params)
    return opt, build_lr_scheduler(opt, cfg)

def build_lr_scheduler(optimizer, cfg):
    scheduler_cfg = cfg.train.lr_scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_cfg.params)
    return scheduler