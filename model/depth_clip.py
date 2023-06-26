# Ref: https://github.com/Adonis-galaxy/DepthCLIP/blob/main/DepthCLIP_code/monoclip.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from . import clip


@torch.no_grad()
def zeroshot_classifier(depth_classes, obj_classes, templates, model):
    zeroshot_weights = []
    for depth in depth_classes:
        for obj in obj_classes:
            texts = [template.format(obj,depth) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


# zero-shot depth prediction
class DepthCLIP(nn.Module):
    def __init__(self, **cfg):
        super(DepthCLIP, self).__init__()
        self.depth_classes = cfg['depth_classes']
        self.bin_list = cfg['bin_list']
        self.n_bin = len(self.depth_classes)
        self.temperature = cfg['temperature']

        obj_classes = ['object']
        depth_templates = ['This {} is {}']

        self.clip, _ = clip.load(cfg['name']) # load pretrained clip encoder
        self.text_f = zeroshot_classifier(self.depth_classes, obj_classes, depth_templates, self.clip) # init text feature

    @torch.no_grad()
    def forward(self, x, seg=None):
        H, W = x.shape[2:]
        img_f = self.clip.encode_image(x).permute(1, 0, 2)  # B, HW, C
        img_f = img_f / img_f.norm(dim=-1, keepdim=True) # normalize img_f

        # scaling images to match the dimension of text embeddings
        scale = float(self.text_f.size(0)) / img_f.size(-1)
        img_f = F.interpolate(img_f, scale_factor=scale)

        # compute similarity score between text and image embeddings
        depth_logits = img_f @ self.text_f  # B, HW, K
        H, W = H // 16, W // 16
        if H * W > depth_logits.size(1):
            H, W = H // 2, W // 2
        depth_logits = depth_logits.permute(0, 2, 1).view(-1, self.n_bin, H, W)  # B, K, H, W
        depth_logits /= self.temperature

        depth = F.softmax(depth_logits, dim=1)
        bin_tensor = torch.tensor(self.bin_list).to(depth.device)
        depth = depth * bin_tensor.reshape(1, self.n_bin).unsqueeze(-1).unsqueeze(-1)
        depth = depth.sum(1, keepdim=True)
        return depth, depth_logits


def depth_clip(**kwargs):
    model = DepthCLIP(**kwargs)
    return model


