import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from core.dataset import transforms
def reconstruction_img(output, mean, std):
    '''
    output: B, C, H, W, 2
    '''
    b = output.size(0)
    mean = mean.view(b, 1, 1, 1, 1).to(output.device)
    std = std.view(b, 1, 1, 1, 1).to(output.device)
    output = output*std + mean
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
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    with tqdm(total=len(data_loader), postfix=[dict(loss=0, avg_loss=0)]) as t:
        for iter, batch in enumerate(data_loader):
            data, norm, file_info = batch
            masked_image, masked_imagek, target_image, target_imagek, mask, target_rss = data
            masked_image, masked_imagek, mask, target_image = to_device([masked_image, masked_imagek, mask, target_image], cfg.device)

            output = model(masked_image, masked_imagek, mask)
            loss = loss_func(output, target_image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
            writer.add_scalars('TrainLoss', {'avg_loss': avg_loss, 'loss': loss.item()}, global_step + iter)
            t.postfix[0]["loss"] = '%.4f' % (loss.item())
            t.postfix[0]["avg_loss"] = '%.4f' % (avg_loss)
            t.update()
            start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch

def evaluate(cfg, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    
    with torch.no_grad():
        with tqdm(total=len(data_loader), postfix=[dict(avg_loss=0)]) as t:
            for iter, batch in enumerate(data_loader):
                data, norm, file_info = batch
                masked_image, masked_imagek, target_image, target_imagek, mask, target_rss = data
                mean, std, norm = norm
                masked_image, masked_imagek, mask, target_rss = to_device([masked_image, masked_imagek, mask, target_rss], cfg.device)
                output = model(masked_image, masked_imagek, mask)
                out_img = reconstruction_img(output, mean, std)
                
                norm = norm.view(len(norm), 1, 1, 1, 1).float().to(out_img.device)
                loss = F.mse_loss(out_img / norm, target_rss / norm, reduction='sum')
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
    gt_list = []
    gen_list = []
    err_list = []
    with torch.no_grad():
        for iter, batch in enumerate(data_loader):
            data, norm, file_info = batch
            masked_image, masked_imagek, target_image, target_imagek, mask, target_rss = data
            mean, std, norm = norm
            masked_image, masked_imagek, mask = to_device([masked_image, masked_imagek, mask], cfg.device)
            # output = model(masked_image, masked_imagek, mask)
            out_img = reconstruction_img(masked_image, mean, std)
            
            out_img = out_img.cpu()
            gt_list.append(target_rss.unsqueeze(1))
            gen_list.append(out_img.unsqueeze(1))
            err_list.append(torch.abs(target_rss - out_img))
    
    gt_tensor = torch.cat(gt_list, 0)
    gen_tensor = torch.cat(gen_list[:8], 0)
    err_tensor = torch.cat(err_list[:8], 0)
    save_image(gt_tensor[:16], 'Target')
    save_image(gen_tensor[:16], 'Reconstruction')
    save_image(err_tensor.unsqueeze(1)[:16], 'Error')