import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from . import clip
from .depth_coarse import Depth_Coarse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=63, kernel_size=9, padding=4, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding='same')
        )
        
    def forward(self, ori, coarse):
        out = self.conv1(ori)
        # print(out.shape)
        # print(coarse.shape)
        out = torch.cat((out, coarse), dim=1)
        out = self.conv2(out)
        return out

class Depth_Refine(nn.Module):
    def __init__(self, **cfg):
        super(Depth_Refine, self).__init__()
        self.depth_classes = cfg['depth_classes']
        self.max_depth = cfg['max_depth']
        self.bin_list = cfg['bin_list']
        self.n_bin = len(self.bin_list)
        self.temperature = cfg['temperature']
        self.obj_classes = ['object']
        self.depth_templates = ['This {} is {}']
        
        self.refiner = Refiner()
        
        # Create an instance of your model
        self.coarse_model = Depth_Coarse(**cfg)

        # Specify the path to the .pth file
        cur_path = os.getcwd()
        path = cur_path + '/runs/NYU/depth_coarse/RN50/temp0.1/dl1/best.pth'
        # Load the saved parameters into the model
        self.coarse_model.load_state_dict(torch.load(path))

    def forward(self, x):
        with torch.no_grad():
            coarse, depth_logit = self.coarse_model(x)
        depth = self.refiner(x, coarse)

        return depth, depth_logit


def depth_refine(**kwargs):
    model = Depth_Refine(**kwargs)
    return model