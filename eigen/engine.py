
import numpy as np
from tqdm import tqdm
import logging
import pdb

import torch
import torch.nn.functional as F

from metrics import AverageMeters, compute_errors
from losses import smooth_loss


logger = logging.getLogger('mylogger')

def train(cfg, model, loader, device, optimizer, epoch=-1):
    # setup metrics
    eval_metrics = ['sym_loss', 'smth_loss', 'depth_loss', 'depth_bincls_loss', 'loss']
    meters = AverageMeters(eval_metrics)

    model.train()
    for (step, value) in tqdm(enumerate(loader)):
        image = value[0].to(device)
        target = value[2].to(device)

        # forward the image to get the depth
        depth, depth_logits = model(image)
        depth = depth.clamp(0, cfg['data']['max_depth'])
        # inference with flipped image and flip the output
        image_flip = torch.flip(image, [3])
        depth_flip, depth_flip_logits = model(image_flip)
        depth_flip = torch.flip(depth_flip, [3])
        depth_flip_logits = torch.flip(depth_flip_logits, [3])
        depth_flip = depth_flip.clamp(0, cfg['data']['max_depth'])

        # averaging the results
        output_depth = (depth + depth_flip) / 2.0

        # resize to the original size
        if(cfg['model']['arch']=='depth_coarse' or cfg['model']['arch']=='depth_refine'):
            print('resize 1/4 for depth_eigen')
            target = F.interpolate(target, scale_factor=0.25, mode='bilinear', align_corners=True)
            image = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)
            # print(target.shape)
        output_depth = F.interpolate(output_depth, size=target.shape[-2:], mode='bilinear', align_corners=True)
        output_depth_logits = F.interpolate(depth_logits, size=target.shape[-2:], mode='bilinear', align_corners=True)
        # print('output shape:', output_depth.shape)

        ### compute loss
        # self-supervised losses
        sym_loss = F.mse_loss(depth, depth_flip.detach()) + F.mse_loss(depth_flip, depth.detach())
        smth_loss = smooth_loss(output_depth, image)
        # supervised losses
        depth_loss = F.mse_loss(output_depth, target)
        bin_tensor = torch.tensor(model.bin_list).to(device)
        bin_upper = bin_tensor[:-1] + bin_tensor.diff() / 2
        target_bin = model.n_bin - torch.le(target.unsqueeze(-1), bin_upper).sum(-1) - 1
        depth_bincls_loss = F.cross_entropy(output_depth_logits, target_bin.squeeze(1), reduction='none').mean()

        loss = sym_loss * cfg['loss'].get('sym_loss', 0)
        loss += smth_loss * cfg['loss'].get('smth_loss', 0)
        loss += depth_loss * cfg['loss'].get('depth_loss', 0)
        loss += depth_bincls_loss * cfg['loss'].get('depth_bincls_loss', 0)

        # compute errors and update meters
        errors = {'sym_loss': sym_loss.item(), 'smth_loss': smth_loss.item(), 'depth_loss': depth_loss.item(), \
            'depth_bincls_loss': depth_bincls_loss.item(), 'loss': loss.item()}
        meters.update(errors, image.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % cfg['print_freq'] == 0:
            logger.info(f"[Train] Epoch: {epoch}, Step: {step}, Sym_loss: {sym_loss.item():.4f}, " \
                f"Smth_loss: {smth_loss.item():.4f}, Depth_loss: {depth_loss.item():.4f}, Depth_bincls_loss: {depth_bincls_loss.item():.4f}, Loss: {loss.item():.4f}")

    return meters.get_values()



@torch.no_grad()
def inference(cfg, model, loader, device, epoch=-1):
    # setup metrics
    eval_metrics = ['a1', 'a2', 'a3', 'abs_rel', 'rmse', 'log_10', 'rmse_log', 'silog', 'sq_rel']
    meters = AverageMeters(eval_metrics)

    model.eval()
    for (step, value) in tqdm(enumerate(loader)):
        image = value[0].to(device)
        target = value[1].to(device)

        # forward the image to get the depth
        depth, _ = model(image)
        depth = depth.clamp(0, cfg['data']['max_depth'])
        # inference with flipped image and flip the output
        image_flip = torch.flip(image, [3])
        depth_flip, _ = model(image_flip)
        depth_flip = torch.flip(depth_flip, [3])
        depth_flip = depth_flip.clamp(0, cfg['data']['max_depth'])

        # averaging the results
        output_depth = (depth + depth_flip) / 2.0

        # resize to the original size
        output_depth = F.interpolate(output_depth, size=target.shape[-2:], mode='bilinear', align_corners=True)

        # compute errors and update meters
        errors = compute_errors(output_depth, target)
        meters.update(errors, image.size(0))

        if (step + 1) % cfg['print_freq'] == 0:
            err = np.mean([x for x in meters.get_values().values()])
            logger.info(f"[Val] Epoch: {epoch}, Step: {step}, Avg Error {err:.4f}")

    return meters.get_values()


