import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from core.dataset import transforms

def train_epoch(cfg, epoch, model, data_loader, optimizer, loss_func, writer):
    model.train()
    avg_loss = 0.
    total_loss = 0.
    global_step = epoch * len(data_loader)
    with tqdm(total=len(data_loader), postfix=[dict(avg_loss=0)]) as t:
        for iter, batch in enumerate(data_loader):
            # featch data
            data, norm, file_info = batch
            subimg, subimgk, image, imagek, mask, target = data

            # forward and backward
            output = model(subimg)
            loss = loss_func(output, image.to(cfg.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            total_loss += loss.item()
            avg_loss = total_loss/(iter+1.0)
            writer.add_scalar('TrainLoss', avg_loss, global_step + iter) 
            t.postfix[0]["avg_loss"] = '%.4f' % (avg_loss)
            t.update()
            start_iter = time.perf_counter()
    return avg_loss

def evaluate(cfg, epoch, model, data_loader, loss_func, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    total_eval_loss = 0.
    total_func_loss = 0. 
    avg_eval_loss = 0.
    avg_func_loss = 0.
    count = 0.
    with torch.no_grad():
        with tqdm(total=len(data_loader), postfix=[dict(eval_loss=0, func_loss=0)]) as t:
            for iter, batch in enumerate(data_loader):
                data, norm, file_info = batch
                subimg, subimgk, image, imagek, mask, target = data
                mean, std, norm = norm
                # output = model(subimg)
                output = subimg
                loss_f = loss_func(output, image.to(output.device))
                # none 0.2067, 0.4032
                mean = mean.view(subimg.size(0), 1, 1, 1).to(output.device)
                std = std.view(subimg.size(0), 1, 1, 1).to(output.device)
                output = output*std + mean
                target = target.to(output.device)*std.squeeze(1) + mean.squeeze(1)
                out_img = transforms.complex_abs(output)
                
                norm = norm.view(len(norm), 1, 1).float().to(out_img.device)
                loss_eval = F.mse_loss(out_img / norm, target.to(out_img.device) / norm, reduction='sum')
                
                total_func_loss += loss_f.item() 
                total_eval_loss += loss_eval.item()
                count += subimg.size(0)
                avg_eval_loss = (total_eval_loss/count)*100.
                avg_func_loss = total_func_loss/(iter + 1.)
                t.postfix[0]["eval_loss"] = '%.4f' % (avg_eval_loss)
                t.postfix[0]["func_loss"] = '%.4f' % (avg_func_loss)
                t.update()
    return avg_func_loss, avg_eval_loss


def visualize(cfg, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    gt_list = []
    gen_list = []
    err_list = []
    with torch.no_grad():
        for iter, batch in enumerate(data_loader):
            data, norm, file_info = batch
            subimg, subimgk, image, imagek, mask, target = data
            mean, std, norm = norm
            output = model(subimg)
            mean = mean.view(subimg.size(0), 1, 1, 1).to(output.device)
            std = std.view(subimg.size(0), 1, 1, 1).to(output.device)
            output = output*std + mean
            target = target.to(output.device)*std.squeeze(1) + mean.squeeze(1)
            out_img = transforms.complex_abs(output.permute(0,2,3,1))

            gt_list.append(target.unsqueeze(1))
            gen_list.append(out_img.unsqueeze(1))
            err_list.append(torch.abs(target.to(output.device) - out_img))
    
    gt_tensor = torch.cat(gt_list, 0)
    gen_tensor = torch.cat(gen_list, 0)
    err_tensor = torch.cat(err_list, 0)
    save_image(gt_tensor[:16], 'Target')
    save_image(gen_tensor[:16], 'Reconstruction')
    save_image(err_tensor.unsqueeze(1)[:16], 'Error')