import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from core.dataset import transforms
def reconstruction_img(output):
    '''
    output: B, C, H, W, 2
    '''
    # out_img = transforms.ifft2(output)
    out_img = transforms.complex_abs(output)
    out_img = transforms.root_sum_of_squares(out_img, 1)
    return out_img

def to_device(tensors, device):
    device_tensor = []
    for t in tensors:
        tmp = t.to(device)
        device_tensor.append(tmp)
    return device_tensor

def train_epoch(cfg, epoch, model, data_loader, optimizer, loss_func, writer):
    model.train()
    avg_loss = 0.
    losses = []
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    with tqdm(total=len(data_loader), postfix=[dict(loss=0, avg_loss=0)]) as t:
        for iter, data in enumerate(data_loader):
            masked_image, masked_kspace, target, targetk, mask, fname, slice = data
            masked_image, masked_kspace, target, targetk, mask = to_device([masked_image, masked_kspace, target, targetk, mask], cfg.device)
            # output = model(input).squeeze(1)
            output = model(masked_image, masked_kspace, mask)
            loss = loss_func(output, targetk)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # calculate loss for compare with validate set
            losses.append(loss.item())
            avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
            writer.add_scalars('TrainLoss', {'avg_loss': avg_loss, 'loss': loss.item()}, global_step + iter)
            t.postfix[0]["loss"] = '%.4f' % (1000*loss.item())
            t.postfix[0]["avg_loss"] = '%.4f' % (1000*avg_loss)
            t.update()
            start_iter = time.perf_counter()
    return 1000*np.mean(losses), time.perf_counter() - start_epoch

def cal_loss(output, target, mean, std, norm, device):
    # mean = mean.unsqueeze(1).unsqueeze(2).to(device)
    # std = std.unsqueeze(1).unsqueeze(2).to(device)
    # target = target * std + mean
    # output = output * std + mean

    # norm = norm.unsqueeze(1).unsqueeze(2).to(device)
    
    # print(norm.dtype, mean.dtype, output.dtype)
    # norm = norm.float()
    # loss = F.mse_loss(output / norm, target / norm, reduction='sum')
    loss = F.mse_loss(output, target, reduction='sum')
    return loss

def evaluate(cfg, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    
    with torch.no_grad():
        with tqdm(total=len(data_loader), postfix=[dict(avg_loss=0)]) as t:
            for iter, data in enumerate(data_loader):
                masked_image, masked_kspace, target, targetk, mask, fname, slice = data
                masked_image, masked_kspace, target, targetk, mask = to_device([masked_image, masked_kspace, target, targetk, mask], cfg.device)
                output = model(masked_image, masked_kspace, mask)
                
                out_img = reconstruction_img(output)
                loss = cal_loss(out_img, target, 0, 1, 1., cfg.device)
                losses.append(1000*loss.item())
                t.postfix[0]["avg_loss"] = '%.4f' % (1000*np.mean(losses))
                t.update()
    return np.mean(losses), time.perf_counter() - start


def visualize(cfg, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            masked_image, masked_kspace, target, targetk, mask, fname, slice = data
            # to_device([masked_image, masked_kspace, mask], cfg.device)
            output = model(masked_image, masked_kspace, mask)
            # start = time.time()
            # about 1e-4 s
            out_img = reconstruction_img(output)
            # end = time.time()
            # print('iter %d, time: %.4f'%(iter, end-start))
            # print(out_img.size(), target.size())
            target = target.unsqueeze(1)
            out_img = out_img.cpu()
            out_img = out_img.unsqueeze(1)
            save_image(target, 'target')
            save_image(out_img, 'Reconstruction')
            save_image(torch.abs(target - out_img), 'Error')
            break