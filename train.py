from mmcv import Config
import argparse
from core.dataset import create_train_dataloaders
from tools.imshow import show_images
import torch
from core.models import build_model
import pathlib
from tensorboardX import SummaryWriter
import os
from core.train import build_optimizer, build_loss, get_train_func, save_model
import time
import logging
import random
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('-c', '--cfg', help='train config file path',
                        default='configs/singlecoil/baseline_unet.py')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('-d', '--data_parallel', action='store_true', default=True)
    parser.add_argument('-acq', '--acquisition', default='both')
    parser.add_argument('-l', '--logdir', default=None)
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--seed', default=6060, type=int)
    
    args = parser.parse_args()
    return args
acquisition = dict(
    both=['CORPD_FBK', 'CORPDFS_FBK'],
    pd=['CORPD_FBK'],
    pdfs=['CORPDFS_FBK']
)
def main():
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    cfg.device = args.device
    cfg.acquisition = acquisition[args.acquisition]
    if args.logdir is not None:
        cfg.logdir = args.logdir

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Log
    logdir = cfg.logdir
    pathlib.Path(cfg.logdir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(os.path.join(logdir, 'log'))

    # Model
    model = build_model(cfg).to((args.device))

    # Optimizer and Scheduler
    optimizer, scheduler = build_optimizer(cfg, model)

    # Loss function
    loss_func = build_loss(cfg)

    # Data Parallel training
    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    # Resume
    start_epoch = 0
    best_dev_loss = 1e9
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_dev_loss = checkpoint['best_dev_loss']


    # Train tools
    train_func, eval_func, visualize = get_train_func(cfg)

    # Data 
    train_loader, dev_loader, display_loader = create_train_dataloaders(cfg)

    # Start training
    train_cfg = cfg.train
    for epoch in range(start_epoch, train_cfg.num_epochs):
        print('Epoch %d'%epoch)
        scheduler.step(epoch)
        # train_loss, train_time = train_func(
        #     cfg, epoch, model, train_loader, optimizer, loss_func, writer)
        dev_loss, dev_time = eval_func(cfg, epoch, model, dev_loader, writer)
        visualize(cfg, epoch, model, display_loader, writer)

        # writer.add_scalars('loss_trainval', {'train_loss': train_loss, 'dev_loss': dev_loss}, epoch)

        # is_new_best = dev_loss < best_dev_loss
        # best_dev_loss = min(best_dev_loss, dev_loss)
        # save_model(logdir, epoch, model, optimizer, best_dev_loss, is_new_best)
        # logging.info(
        #     f'Epoch = [{epoch:4d}/{train_cfg.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
        #     f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        # )
        # if (epoch+1)%5 == 0:
        #     time.sleep(60*5)
    writer.close()


if __name__ == '__main__':
    main()
