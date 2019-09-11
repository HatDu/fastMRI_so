import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
def train_epoch(cfg, epoch, model, data_loader, optimizer, loss_func, writer):
    model.train()
    avg_loss = 0.
    total_loss = 0.

    global_step = epoch * len(data_loader)
    with tqdm(total=len(data_loader), postfix=[dict(avg_loss=0)]) as t:
        for iter, data in enumerate(data_loader):
            # featch data
            input, target, mean, std, norm = data[:5]

            # forward
            output = model(input.to(cfg.device)).squeeze(1)
            loss = loss_func(output, target.to(cfg.device))
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            total_loss += loss.item()
            avg_loss = total_loss / (1.0 + iter)
            writer.add_scalar('TrainLoss', loss.item(), global_step + iter)
            t.postfix[0]["avg_loss"] = '%.4f' % (loss.item())
            t.update()
    return avg_loss

def cal_loss(output, target, mean, std, norm, device):
    mean = mean.unsqueeze(1).unsqueeze(2).to(device)
    std = std.unsqueeze(1).unsqueeze(2).to(device)
    target = target * std + mean
    output = output * std + mean

    norm = norm.unsqueeze(1).unsqueeze(2).to(device)

    norm = norm.float()
    loss = F.mse_loss(output / norm, target / norm, reduction='sum')
    return loss

def evaluate(cfg, epoch, model, data_loader, loss_func, writer):
    model.eval()
    total_loss_train = 0.
    total_loss_eval = 0.
    avg_loss_train = 0.
    avg_loss_eval = 0.
    eval_count = 0.
    with torch.no_grad():
        with tqdm(total=len(data_loader), postfix=[dict(eval_loss=0., func_loss=0.)]) as t:
            for iter, data in enumerate(data_loader):
                # featch data
                input, target, mean, std, norm = data[:5]
                input = input.to(cfg.device)
                output = model(input).squeeze(1)

                # cal loss
                target = target.to(cfg.device)
                train_loss = loss_func(output, target)
                eval_loss = cal_loss(input.squeeze(1), target, mean, std, norm, cfg.device)
                total_loss_eval += eval_loss.item()
                total_loss_train += train_loss.item()
                
                eval_count += input.size(0)
                avg_loss_train = total_loss_train/(iter+1.0)
                avg_loss_eval = total_loss_eval/eval_count
                
                # record
                t.postfix[0]["eval_loss"] = '%.4f' % avg_loss_eval
                t.postfix[0]["func_loss"] = '%.4f' % avg_loss_train
                t.update()
    return avg_loss_train, avg_loss_eval


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
            target = target.unsqueeze(1).to(cfg.device)
            output = model(input)
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target - output), 'Error')
            break