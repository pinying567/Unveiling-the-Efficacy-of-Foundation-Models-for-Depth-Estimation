
import argparse
import os
import numpy as np
import yaml
import shutil
import pandas as pd
from tqdm import tqdm
import pdb

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from loader import get_dataloader
from model import get_model
from utils import get_logger
from optim import get_optimizer, step_scheduler
from engine import train, inference

from tensorboardX import SummaryWriter


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup random seed
    torch.manual_seed(cfg.get('seed', 1))
    torch.cuda.manual_seed(cfg.get('seed', 1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load data
    splits = ['train', 'test']
    data_loader = get_dataloader(cfg['data'], splits)

    # setup model
    model = get_model(cfg['model']).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # setup optimizer and lr scheduler
    opt_cls, opt_params = get_optimizer(cfg['optim'])
    model_params = []
    param_names = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_params.append(param)
            param_names.add(name)
    logger.info(f"Parameters to be updated: {param_names}")
    optimizer = opt_cls(model_params, **opt_params)
    scheduler = step_scheduler(optimizer, **cfg['scheduler'])

    # training and validation
    best_result = np.Inf
    n_epoch = cfg["max_epoch"]
    logger.info('Start training')
    for epoch in tqdm(range(n_epoch)):
        # training
        curr_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current lr: {curr_lr}")
        writer.add_scalar('train/lr', curr_lr, epoch + 1)
        train_meters = train(cfg, model, data_loader['train'], device, optimizer, epoch)
        for k in train_meters:
            writer.add_scalar(f"train/{k}", train_meters[k], epoch + 1)
        train_err = train_meters['depth_loss']
        logger.info(f"=== Epoch {epoch} ===")
        logger.info(f"Training error: {train_err}")
        logger.info(f"======")

        # validation
        if (epoch + 1) % cfg['eval_freq'] == 0:
            eval_meters = inference(cfg, model, data_loader['test'], device)
            for k in eval_meters:
                writer.add_scalar(f"val/{k}", eval_meters[k], epoch + 1)
            eval_err = np.mean([x for x in eval_meters.values()])
            df = pd.DataFrame([eval_meters])
            logger.info(f"=== Eval ===")
            logger.info(eval_meters)
            print(df)
            logger.info(f"======")
            if eval_err < best_result:
                best_result = eval_err
                logger.info("Saving the best model")
                # save the best model
                torch.save(model.state_dict(), os.path.join(logdir, "best.pth"))

            logger.info(f"Best result : {best_result} ; Epoch result : {eval_err}")

        scheduler.step()

    torch.save(model.state_dict(), os.path.join(logdir, "last.pth"))
    logger.info(f"======")

    # load the best model for evaluation
    if os.path.exists(os.path.join(logdir, "best.pth")):
        logger.info(f"=== Evaluating Best model  !! ===")
        model.load_state_dict(torch.load(os.path.join(logdir, "best.pth")))
    else:
        logger.info(f"=== Evaluating Last model  !! ===")
        model.load_state_dict(torch.load(os.path.join(logdir, "last.pth")))

    eval_meters = inference(cfg, model, data_loader['test'], device)
    df = pd.DataFrame([eval_meters])
    logger.info(f"=== Final result ===")
    logger.info(df)
    logger.info(f"======")


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
    print(cfg)

    logdir = os.path.join('runs', cfg['data']['name'], cfg['model']['arch'], cfg['exp'])
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Start logging")

    main()

