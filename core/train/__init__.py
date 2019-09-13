from torch import optim
import torch.nn.functional as F
import torch
import os
import shutil
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

def build_loss(cfg):
    loss_cfg = cfg.train.loss
    if loss_cfg.name == 'l1_loss':
        return F.l1_loss

def get_train_func(cfg):
    func_cfg = cfg.train.train_func
    if func_cfg.name == 'train_slice':
        from core.train.train_slice import train_epoch, evaluate, visualize
        return train_epoch, evaluate, visualize
    elif func_cfg.name == 'train_fusion_silce':
        from core.train.train_fusion_silce import train_epoch, evaluate, visualize
        return train_epoch, evaluate, visualize
    elif func_cfg.name == 'train_complex':
        from core.train.train_complex import train_epoch, evaluate, visualize
        return train_epoch, evaluate, visualize
    elif func_cfg.name == 'train_dncn':
        from core.train.train_dncn import train_epoch, evaluate, visualize
        return train_epoch, evaluate, visualize

def save_model(log_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss
        },
        f=os.path.join(log_dir, 'model.pt')
    )
    if is_new_best:
        shutil.copyfile(os.path.join(log_dir, 'model.pt'),
                        os.path.join(log_dir, 'best_model.pt'))