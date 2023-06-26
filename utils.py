
import os, sys
import logging
import datetime
import matplotlib
import matplotlib.cm
import numpy as np

import torch
import torch.nn as nn


class ViewTransform(nn.Module):
    def __init__(self, K):
        super(ViewTransform, self).__init__()
        self.K = torch.from_numpy(K).float()
        self.K_inv = torch.from_numpy(np.linalg.inv(K)).float()

    def forward(self, img, depth, T):
        """ Project source view to target view via inverse warping. """
        B, C, h, w = img.shape
        device = img.device
        grids = self._make_grids((h, w), B).to(device)
        grids = self.K_inv.to(device).matmul(grids)
        grids = depth.flatten(-2, -1) * grids
        grids = torch.cat((grids, grids.new_ones(B, 1, h * w)), dim=1)
        grids = T.matmul(grids)
        grids = self.K.to(device).matmul(grids)
        grids = grids / (grids[:, 2, :].unsqueeze(1) + 1e-16)
        nv = self._interpolate(img, grids).view(B, C, h, w)
        return nv

    @staticmethod
    def _make_grids(im_size, batch_size):
        """ Create grid sampling for the image map.
        Parameters:
            im_size: size of the image
            batch_size: batch size of the image
        """
        h, w = im_size
        u = torch.arange(w, dtype=torch.float32)
        v = torch.arange(h, dtype=torch.float32)
        v, u = torch.meshgrid([v, u])
        u = torch.flatten(u)
        v = torch.flatten(v)
        ones = torch.ones_like(u)
        grids = torch.stack([u, v, ones], 0)
        grids = grids.unsqueeze(0).repeat([batch_size, 1, 1]) # (B, 3, N)
        return grids

    @staticmethod
    def _interpolate(input_maps, sampled_grids):
        B, C, v_max, u_max = input_maps.size()
        # sampled_grids.size() = [B, n_channels, N], N = u_max * v_max
        u = torch.flatten(sampled_grids[:, 0, :])  # (BN, )
        v = torch.flatten(sampled_grids[:, 1, :])
        u0 = torch.floor(u).long()
        u1 = u0 + 1
        v0 = torch.floor(v).long()
        v1 = v0 + 1
        # clamp
        u0 = torch.clamp(u0, 0, u_max - 1)
        u1 = torch.clamp(u1, 0, u_max - 1)
        v0 = torch.clamp(v0, 0, v_max - 1)
        v1 = torch.clamp(v1, 0, v_max - 1)

        flat_output_size = sampled_grids.size(-1)
        pixels_batch = torch.arange(0, B) * v_max * u_max
        pixels_batch = pixels_batch.view(B, 1).to(input_maps.device)
        base = pixels_batch.repeat([1, flat_output_size])  # (B, N)
        base = torch.flatten(base)  # (BN, )
        base_v0 = base + v0 * u_max
        base_v1 = base + v1 * u_max
        #    u0  u1
        # v0 [a, c],
        # v1 [b, d]
        indices_a = base_v0 + u0  # (BN, )
        indices_b = base_v1 + u0
        indices_c = base_v0 + u1
        indices_d = base_v1 + u1

        flat_maps = torch.transpose(input_maps, 0, 1).reshape(C, -1)  # (C, BN)
        pixel_values_a = flat_maps[:, indices_a]  # (C, BN)
        pixel_values_b = flat_maps[:, indices_b]
        pixel_values_c = flat_maps[:, indices_c]
        pixel_values_d = flat_maps[:, indices_d]

        area_a = (v1.float() - v) * (u1.float() - u)  # (BN, )
        area_b = (v - v0.float()) * (u1.float() - u)
        area_c = (v1.float() - v) * (u - u0.float())
        area_d = (v - v0.float()) * (u - u0.float())

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        values = values_a + values_b + values_c + values_d
        values = torch.transpose(values.view(C, B, flat_output_size), 0, 1)
        return values  # (B, C, N)

def colorize(value, vmin=10, vmax=1000, cmap='magma_r'):
    invalid_mask = value == -1

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 255
    img = value[:, :, :3]

    return img

def get_logger(logdir):
    logger = logging.getLogger('mylogger')
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    fileHandler = logging.FileHandler(file_path)
    consoleHandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.INFO)
    return logger

