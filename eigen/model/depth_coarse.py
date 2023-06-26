import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from . import clip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Coarse_Adapter(nn.Module):
    def __init__(self):
        super(Coarse_Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=17*13, out_features=4096),
            nn.Linear(in_features=4096, out_features=104*136)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(B, C, -1)
        x = self.fc(x)
        x = x.view(B, C, 104, 136)
        return x

class Depth_Coarse(nn.Module):
    def __init__(self, **cfg):
        super(Depth_Coarse, self).__init__()
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
        self.coarse_adapter = Coarse_Adapter()

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
                    texts = clip.tokenize(texts).to(device)  # tokenize
                    class_embeddings = self.clip.encode_text(texts)  # embed with text encoder
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    text_features.append(class_embedding)
            text_features = torch.stack(text_features, dim=1)

        return text_features

    def encode_image(self, x):
        H, W = x.shape[2:]
        with torch.no_grad():
            image_features = self.clip.encode_image(x.type(self.dtype)).permute(1, 0, 2)  # B, HW, C

        return image_features

    def forward(self, x):
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

        # Coarse prediction
        depth = F.softmax(depth_logits, dim=1)
        bin_tensor = torch.tensor(self.bin_list).to(depth.device)
        depth = depth * bin_tensor.view(1, self.n_bin).unsqueeze(-1).unsqueeze(-1)
        depth = depth.sum(1, keepdim=True)
        depth = self.coarse_adapter(depth)

        return depth, depth_logits


def depth_coarse(**kwargs):
    model = Depth_Coarse(**kwargs)
    return model