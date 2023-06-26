
import torch


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


