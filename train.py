from mmcv import Config
import argparse
from core.dataset import create_data_loader
from tools.imshow import show_images
import torch
from core.models import build_model
import pathlib
from tensorboardX import SummaryWriter
import os
import shutil
from core.train import build_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--cfg', help='train config file path',
                        default='configs/baseline_unet.py')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data_parallel', action='store_true', default=True)
    parser.add_argument('--ckpt', default=None)
    args = parser.parse_args()
    return args


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


def main():
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    train_loader, dev_loader, display_loader = create_data_loader(cfg)
    # Log
    logdir = cfg.logdir
    pathlib.Path(cfg.logdir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(os.path.join(logdir, 'log'))

    # Model
    model = build_model(cfg).to((args.device))

    # Optimizer and Scheduler
    optimizer, scheduler = build_optimizer(cfg, model)

    # Data Parallel training
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    save_model(logdir, 0, model, optimizer, 0.01, True)
    # Resume
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model'])

    
    
        # optimizer.load_state_dict(checkpoint['optimizer'])

    # for it, data in enumerate(train_loader):
    #     image, target, mean, std = data[:4]


if __name__ == '__main__':
    main()
