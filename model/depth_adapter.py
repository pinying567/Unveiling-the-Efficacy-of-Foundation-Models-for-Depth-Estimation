
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from . import clip


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class DepthAdapter(nn.Module):
    def __init__(self, **cfg):
        super(DepthAdapter, self).__init__()
        self.depth_classes = cfg['depth_classes']
        self.max_depth = cfg['max_depth']
        self.bin_list = cfg['bin_list']
        self.n_bin = len(self.bin_list)
        self.temperature = cfg['temperature']
        self.obj_classes = ['object']
        self.depth_templates = ['This {} is {}']

        print("Loading CLIP (backbone:" + cfg['name'] + ")")
        self.clip, _ = clip.load(cfg['name']) # load pretrained clip encoder
        self.dtype = self.clip.dtype
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.clip.named_parameters():
            param.requires_grad_(False)
        self.clip.eval()

        self.alpha = cfg['image_adapter']['weight'] # image adapter weight
        self.beta = cfg['text_adapter']['weight']   # text adapter weight
        self.image_adapter = Adapter(cfg['image_adapter']['input_size'], cfg['image_adapter']['reduction']).to(self.dtype)
        self.text_adapter = Adapter(cfg['text_adapter']['input_size'], cfg['text_adapter']['reduction']).to(self.dtype)

        # double check
        enabled = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

    def encode_text(self):
        with torch.no_grad():
            text_features = []
            for depth in self.depth_classes:
                for obj in self.obj_classes:
                    texts = [template.format(obj, depth) for template in self.depth_templates]
                    texts = clip.tokenize(texts).cuda()  # tokenize
                    class_embeddings = self.clip.encode_text(texts)  # embed with text encoder
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    text_features.append(class_embedding)
            text_features = torch.stack(text_features, dim=1).T

        x = self.text_adapter(text_features)
        text_features = self.beta * x + (1 - self.beta) * text_features
        text_features = text_features.T

        return text_features

    def encode_image(self, x):
        H, W = x.shape[2:]
        with torch.no_grad():
            image_features = self.clip.encode_image(x.type(self.dtype)).permute(1, 0, 2)  # B, HW, C

        x = self.image_adapter(image_features)
        image_features = self.alpha * x + (1 - self.alpha) * image_features

        return image_features

    def forward(self, x, seg=None):
        H, W = x.shape[2:]
        text_f = self.encode_text()
        img_f = self.encode_image(x)

        # scaling images to match the dimension of text embeddings
        scale = float(text_f.size(0)) / img_f.size(-1)
        img_f = F.interpolate(img_f, scale_factor=scale)
        text_f = F.normalize(text_f, dim=0)
        img_f = F.normalize(img_f, dim=-1)

        # compute similarity score between text and image embeddings
        depth_logits = img_f @ text_f  # B, HW, K
        H, W = H // 16, W // 16
        if H * W > depth_logits.size(1):
            H, W = H // 2, W // 2
        depth_logits = depth_logits.permute(0, 2, 1).view(-1, self.n_bin, H, W)  # B, K, H, W
        depth_logits /= self.temperature

        depth = F.softmax(depth_logits, dim=1)
        bin_tensor = torch.tensor(self.bin_list).to(depth.device)
        depth = depth * bin_tensor.view(1, self.n_bin).unsqueeze(-1).unsqueeze(-1)
        depth = depth.sum(1, keepdim=True)
        return depth, depth_logits


def depth_adapter(**kwargs):
    model = DepthAdapter(**kwargs)
    return model


