# Ref: https://github.com/Adonis-galaxy/DepthCLIP/blob/main/DepthCLIP_code/datasets/datasets_list.py

from torchvision import transforms
from torchvision.transforms import Lambda

from .utils import *


class DataTransform(object):
    def __init__(self, cfg):
        if cfg['name'] == 'KITTI':
            self.train_transform = EnhancedCompose([
                Merge(),
                RandomCropNumpy((cfg['height'], cfg['width'])),
                RandomHorizontalFlip(),
                Split([0, 3], [3, 4], [-2, -1]),
                [RandomColor(multiplier_range=(0.9, 1.1)), None, None],
                [Lambda(to_tensor), Lambda(to_tensor), Lambda(to_tensor)],
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
            self.test_transform = EnhancedCompose([
                Merge(),
                CenterCropNumpy((cfg['height'], cfg['width'])),
                Split([0, 3], [3, 4], [-2, -1]),
                [Lambda(to_tensor), Lambda(to_tensor), Lambda(to_tensor)],
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])

        elif cfg['name'] == 'NYU':
            self.train_transform = EnhancedCompose([
                Merge(),
                RandomCropNumpy((cfg['height'], cfg['width'])),
                RandomHorizontalFlip(),
                Split([0, 3], [3, 6], [6, 7], [-2, -1]),
                [RandomColor(multiplier_range=(0.8, 1.2)), RandomColor(multiplier_range=(0.8, 1.2)), None, None],
                [Lambda(to_tensor), Lambda(to_tensor), Lambda(to_tensor), Lambda(to_tensor)],
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None, None]
            ])
            self.ssl_train_transform = EnhancedCompose([
                Merge(),
                CenterCropNumpy((cfg['height'], cfg['width'])),
                Split([0, 3], [3, 6], [6, 9], [9, 10], [-2, -1]),
                [Lambda(to_tensor), Lambda(to_tensor), Lambda(to_tensor), Lambda(to_tensor), Lambda(to_tensor)],
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])

            self.test_transform = EnhancedCompose([
                [Lambda(to_tensor), Lambda(to_tensor), Lambda(to_tensor), Lambda(to_tensor)],
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None, None]
            ])

    def __call__(self, images, train=True, ssl=True):
        if train is True:
            if ssl:
                return self.ssl_train_transform(images)
            return self.train_transform(images)
        else:
            return self.test_transform(images)


