
import h5py
from PIL import Image
import numpy as np
from pathlib import Path
import urllib.request
import pandas as pd
import logging

import torch.utils.data as data
from torch.utils.data import Dataset

from .convert import convert
from .transforms import DataTransform
from .utils import _is_pil_image


logger = logging.getLogger('mylogger')

class DepthDataset(Dataset):
    def __init__(self, cfg, train=True):
        if train:
            self.data_root = Path(cfg['data_root']) / 'train'
        else:
            self.data_root = Path(cfg['data_root']) / 'test'

        self.datafile = cfg['train'] if train else cfg['test']
        # self.transform = DataTransform(cfg)
        self.dataset = cfg['name']
        self.train = train
        self.cfg = cfg

        if self.dataset == 'KITTI':
            self.angle_range = (-1, 1)
            self.depth_scale = 256.0
            self.max_depth = 80.0
        elif self.dataset == 'NYU':
            self.angle_range = (-2.5, 2.5)
            self.depth_scale = 1000.0
            self.max_depth = 10.0

        self.fileset = pd.read_csv(self.datafile, sep=' ', header=None)

    def __getitem__(self, index):
        rgb_file = self.data_root / self.fileset.iloc[index, 0]
        gt_file = self.data_root / self.fileset.iloc[index, 1]
        gt_dense_file = gt_file.parent / 'dense' / gt_file.name.replace('depth', 'depth_dense')

        rgb = Image.open(rgb_file).convert('RGB')
        gt = Image.open(gt_file)
        gt_dense = Image.open(gt_dense_file)

        if self.train:
            angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
            rgb = rgb.rotate(angle, resample=Image.BILINEAR)
            gt = gt.rotate(angle, resample=Image.NEAREST)
            gt_dense = gt_dense.rotate(angle, resample=Image.NEAREST)

        # cropping in size that can be divided by 16
        if self.dataset == 'KITTI':
            h = rgb.height
            w = rgb.width
            bound_left = (w - 1216) // 2
            bound_right = bound_left + 1216
            bound_top = h - 352
            bound_bottom = bound_top + 352

        elif self.dataset == 'NYU':
            if self.train:
                bound_left = 43
                bound_right = 608
                bound_top = 45
                bound_bottom = 472
            else:
                bound_left = 60
                bound_right = 604
                bound_top = 56
                bound_bottom = 472

        # crop and normalize 0 to 1 ==>  rgb range:(0,1),  depth range: (0, max_depth)
        rgb = rgb.crop((bound_left, bound_top, bound_right, bound_bottom))
        rgb = np.asarray(rgb, dtype=np.float32) / 255.0

        if _is_pil_image(gt):
            gt = gt.crop((bound_left, bound_top, bound_right, bound_bottom))
            gt = (np.asarray(gt, dtype=np.float32)) / self.depth_scale
            gt = np.expand_dims(gt, axis=2)
            gt = np.clip(gt, 0, self.max_depth)

        if _is_pil_image(gt_dense):
            gt_dense = gt_dense.crop((bound_left, bound_top, bound_right, bound_bottom))
            gt_dense = (np.asarray(gt_dense, dtype=np.float32)) / self.depth_scale
            gt_dense = np.expand_dims(gt_dense, axis=2)
            gt_dense = np.clip(gt_dense, 0, self.max_depth)

        rgb, gt, gt_dense = np.array(rgb), np.array(gt), np.array(gt_dense)
        # rgb, gt, gt_dense = self.transform([rgb] + [gt] + [gt_dense], self.train)
        transform = DataTransform(self.cfg)
        rgb, gt, gt_dense = transform([rgb] + [gt] + [gt_dense], self.train)

        return rgb, gt, gt_dense


    def __len__(self):
        return len(self.fileset)


def download_NYU():
    download_url = {
        'data': 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat',
        'splits': 'http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat'
    }
    data_root = Path('loader/data/NYU')
    data = data_root / 'nyu_depth_v2_labeled.mat'
    splits = data_root / 'splits.mat'
    if not data.is_file():
        url = download_url['data']
        logger.info(f'Downloading data from {url}')
        urllib.request.urlretrieve(url, data)

    if not splits.is_file():
        url = download_url['splits']
        logger.info(f'Downloading train/test splits from {url}')
        urllib.request.urlretrieve(url, splits)

    convert(data, splits, data_root)


def DepthDataLoader(cfg, splits):
    """Function to build data loader(s) for the specified splits given the parameters.
    """
    data_root = Path(cfg.get('data_root', '/path/to/dataset'))
    if not data_root.is_dir():
        raise Exception(f'{data_root} does not exist')

    train = data_root / 'train'
    if not train.is_dir():
        download_NYU()

    num_workers = cfg.get('n_workers', 4)
    batch_size = cfg.get('batch_size', 1)

    # data_loader = dict()
    train_loader = None
    test_loader = None
    for split in splits:

        data_list = Path(cfg.get(split, None))
        if not data_list.is_file():
            raise Exception(f'{data_list} not available')

        train = True if split == 'train' else False
        dataset = DepthDataset(cfg, train)

        if train:
            train_loader = data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                pin_memory=True, num_workers=num_workers
            )
        else:
            test_loader = data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                pin_memory=True, num_workers=num_workers
            )

        logger.info("{split}: {size}".format(split=split, size=len(dataset)))

    logger.info("Building data loader with {} workers".format(num_workers))

    return train_loader, test_loader

