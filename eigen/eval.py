
import argparse
import os
import numpy as np
import yaml
import shutil
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from loader import get_dataloader
from model import get_model
from utils import get_logger
from engine import inference


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup random seed
    torch.manual_seed(cfg.get('seed', 1))
    torch.cuda.manual_seed(cfg.get('seed', 1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load data
    splits = ['train', 'test']
    train_loader, test_loader = get_dataloader(cfg['data'], splits)

    # setup model
    model = get_model(cfg['model']).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # zero-shot inference of the depth
    logger.info('Running zero-shot inference')
    meters = inference(cfg, model, test_loader, device)
    logger.info(meters)
    df = pd.DataFrame([meters])
    print(df)


if __name__ == '__main__':
    global cfg, args, logger
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--config',
        nargs='?',
        type=str,
        help='Configuration file to use',
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    logdir = os.path.join('runs', cfg['data']['name'], cfg['model']['arch'], cfg['exp'])
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Start logging")

    main()


