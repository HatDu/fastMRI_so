from mmcv import Config
import argparse
from core.dataset import create_data_loader
from tools.imshow import show_images
import torch
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--cfg', help='train config file path',
                        default='configs/baseline_unet.py')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    train_loader, dev_loader, display_loader = create_data_loader(cfg)
    for it, data in enumerate(train_loader):
        image, target, mean, std = data[:4]
        print(image.size(), target.size())
        show_images(torch.cat((image, target)))

if __name__ == '__main__':
    main()