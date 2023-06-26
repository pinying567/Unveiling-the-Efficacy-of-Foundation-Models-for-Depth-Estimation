
import numpy as np
from tqdm import tqdm
import logging
import cv2
import pdb

import torch
import torch.nn.functional as F

from metrics import AverageMeters, compute_errors
from losses import photometric_loss, smooth_loss, L1
from utils import ViewTransform, colorize


logger = logging.getLogger('mylogger')


def train(cfg, model, loader, device, optimizer, epoch=-1):
    # setup metrics
    eval_metrics = ['pe_loss', 'sym_loss', 'smth_loss', 'depth_loss', 'depth_bincls_loss', 'loss']
    meters = AverageMeters(eval_metrics)

    vt = ViewTransform(loader.dataset.K)
    model.train()
    for (step, value) in tqdm(enumerate(loader)):
        image = value[0].to(device)
        target = value[2].to(device)
        seg = value[3].to(device)
        if cfg['data']['ssl']:
            src_img = value[4].to(device)
            T = value[5].to(device) # T -> S transform

        # forward the image to get the depth
        depth, depth_logits = model(image, seg)
        depth = depth.clamp(0, cfg['data']['max_depth'])
        # inference with flipped image and flip the output
        image_flip = torch.flip(image, [3])
        seg_flip = torch.flip(seg, [3])
        depth_flip, depth_flip_logits = model(image_flip, seg_flip)
        depth_flip = torch.flip(depth_flip, [3])
        depth_flip_logits = torch.flip(depth_flip_logits, [3])
        depth_flip = depth_flip.clamp(0, cfg['data']['max_depth'])

        # averaging the results
        output_depth = (depth + depth_flip) / 2.0

        # resize to the original size
        output_depth = F.interpolate(output_depth, size=target.shape[-2:], mode='bilinear', align_corners=True)
        output_depth_logits = F.interpolate(depth_logits, size=target.shape[-2:], mode='bilinear', align_corners=True)

        # project target view back to source view with the predicted depth
        if cfg['data']['ssl']:
            pred_tgt_img = vt(src_img, output_depth, T.detach())

        # visualize (need to comment normalize part in Transform)
        #from PIL import Image
        #src = src_img[0, ...].data.cpu().numpy().transpose((1, 2, 0)) * 255
        #src_image = Image.fromarray(src.astype(np.uint8))
        #src_image.save('src_img.png')
        #tgt = image[0, ...].data.cpu().numpy().transpose((1, 2, 0)) * 255
        #tgt_img = Image.fromarray(tgt.astype(np.uint8))
        #tgt_img.save('tgt_img.png')
        #sam = seg[0, ...].data.cpu().numpy().transpose((1, 2, 0)) * 255
        #sam_img = Image.fromarray(sam.astype(np.uint8))
        #sam_img.save('sam_img.png')
        #pred_tgt = pred_tgt_img[0, ...].data.cpu().numpy().transpose((1, 2, 0)) * 255
        #pred_img = Image.fromarray(pred_tgt.astype(np.uint8))
        #pred_img.save('pred_img.png')
        #gt = target[0, ...].squeeze().data.cpu().numpy()
        #gt = Image.fromarray(colorize(gt, vmin=None, vmax=None, cmap='magma_r'))
        #gt.save('gt_img.png')

        ### compute loss
        # self-supervised losses
        pe_loss = torch.zeros([1]).to(device)
        if cfg['data']['ssl']:
            pe_loss = photometric_loss(pred_tgt_img, image)
        sym_loss = F.mse_loss(depth, depth_flip.detach()) + F.mse_loss(depth_flip, depth.detach())
        smth_loss = smooth_loss(output_depth, image)
        # supervised losses
        #depth_loss = F.mse_loss(output_depth, target)
        depth_loss = L1(output_depth, target)
        bin_tensor = torch.tensor(model.bin_list).to(device)
        bin_upper = bin_tensor[:-1] + bin_tensor.diff() / 2
        target_bin = model.n_bin - torch.le(target.unsqueeze(-1), bin_upper).sum(-1) - 1
        depth_bincls_loss = F.cross_entropy(output_depth_logits, target_bin.squeeze(1), reduction='none').mean()

        loss = pe_loss * cfg['loss'].get('pe_loss', 0)
        loss += sym_loss * cfg['loss'].get('sym_loss', 0)
        loss += smth_loss * cfg['loss'].get('smth_loss', 0)
        loss += depth_loss * cfg['loss'].get('depth_loss', 0)
        loss += depth_bincls_loss * cfg['loss'].get('depth_bincls_loss', 0)

        # compute errors and update meters
        errors = {'pe_loss': pe_loss.item(), 'sym_loss': sym_loss.item(), 'smth_loss': smth_loss.item(), 'depth_loss': depth_loss.item(), \
            'depth_bincls_loss': depth_bincls_loss.item(), 'loss': loss.item()}
        meters.update(errors, image.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % cfg['print_freq'] == 0:
            logger.info(f"[Train] Epoch: {epoch}, Step: {step}, PE_loss: {pe_loss.item():.4f}, Sym_loss: {sym_loss.item():.4f}, " \
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
        seg = value[3].to(device)

        # forward the image to get the depth
        depth, _ = model(image, seg)
        depth = depth.clamp(0, cfg['data']['max_depth'])
        # inference with flipped image and flip the output
        image_flip = torch.flip(image, [3])
        seg_flip = torch.flip(seg, [3])
        depth_flip, _ = model(image_flip, seg_flip)
        depth_flip = torch.flip(depth_flip, [3])
        depth_flip = depth_flip.clamp(0, cfg['data']['max_depth'])

        # averaging the results
        output_depth = (depth + depth_flip) / 2.0

        # resize to the original size
        output_depth = F.interpolate(output_depth, size=target.shape[-2:], mode='bilinear', align_corners=True)

        # visualize (need to comment normalize part in Transform)
        #from PIL import Image
        #src = image[0, ...].data.cpu().numpy().transpose((1, 2, 0)) * 255
        #src_image = Image.fromarray(src.astype(np.uint8))
        #src_image.save('rgb_image.png')
        #sam = seg[0, ...].data.cpu().numpy().transpose((1, 2, 0)) * 255
        #sam_img = Image.fromarray(sam.astype(np.uint8))
        #sam_img.save('sam_img.png')
        #pred = output_depth[0, ...].squeeze().data.cpu().numpy()
        #pred = Image.fromarray(colorize(pred, vmin=None, vmax=None, cmap='magma_r'))
        #pred.save('pred_img.png')
        #gt = target[0, ...].squeeze().data.cpu().numpy()
        #gt = Image.fromarray(colorize(gt, vmin=None, vmax=None, cmap='magma_r'))
        #gt.save('gt_img.png')

        # compute errors and update meters
        errors = compute_errors(output_depth, target)
        meters.update(errors, image.size(0))

        if (step + 1) % cfg['print_freq'] == 0:
            err = np.mean([x for x in meters.get_values().values()])
            logger.info(f"[Val] Epoch: {epoch}, Step: {step}, Avg Error {err:.4f}")

    return meters.get_values()


