
import torch
import torch.nn as nn


def L1(x, y):
    return torch.mean(torch.abs(x - y))

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1).mean()

def photometric_loss(x, y, alpha=0.85):
    return alpha * (1 - SSIM(x, y)) / 2 + (1 - alpha) * L1(x, y)

def smooth_loss(depth, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    Ref: https://github.com/nianticlabs/manydepth/blob/master/manydepth/layers.py
    """
    grad_depth_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    grad_depth_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)

    return grad_depth_x.mean() + grad_depth_y.mean()


