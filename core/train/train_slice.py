import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
def train_epoch(cfg, epoch, model, data_loader, optimizer, loss_func, writer):
    model.train()
    avg_loss = 0.
    losses = []
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    with tqdm(total=len(data_loader), postfix=[dict(loss=0, avg_loss=0)]) as t:
        for iter, data in enumerate(data_loader):
            input, target, mean, std, norm = data[:5]
            input = input.to(cfg.device)
            target = target.to(cfg.device)

            output = model(input).squeeze(1)
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # calculate loss for compare with validate set
            loss_cp_eval = cal_loss(output, target, mean, std, norm, cfg.device)
            losses.append(loss_cp_eval.item())
            avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
            writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

            t.postfix[0]["loss"] = '%.4f' % (loss.item())
            t.postfix[0]["avg_loss"] = '%.4f' % (avg_loss)
            t.update()
            start_iter = time.perf_counter()
    return np.mean(losses), time.perf_counter() - start_epoch

def cal_loss(output, target, mean, std, norm, device):
    mean = mean.unsqueeze(1).unsqueeze(2).to(device)
    std = std.unsqueeze(1).unsqueeze(2).to(device)
    target = target * std + mean
    output = output * std + mean

    norm = norm.unsqueeze(1).unsqueeze(2).to(device)
    
    # print(norm.dtype, mean.dtype, output.dtype)
    norm = norm.float()
    loss = F.mse_loss(output / norm, target / norm, reduction='sum')
    return loss

def evaluate(cfg, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    
    with torch.no_grad():
        with tqdm(total=len(data_loader), postfix=[dict(avg_loss=0)]) as t:
            for iter, data in enumerate(data_loader):
                input, target, mean, std, norm = data[:5]
                input = input.unsqueeze(1).to(cfg.device)
                target = target.to(cfg.device)
                output = model(input).squeeze(1)
                loss = cal_loss(output, target, mean, std, norm, cfg.device)
                losses.append(loss.item())
                t.postfix[0]["avg_loss"] = '%.4f' % (np.mean(losses))
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
            input, target = data[:2]
            input = input.unsqueeze(1).to(cfg.device)
            target = target.unsqueeze(1).to(cfg.device)
            output = model(input)
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target - output), 'Error')
            break