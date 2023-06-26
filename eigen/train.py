
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
from optim import get_optimizer, step_scheduler
from engine import train, inference


def run_coarse(cfg, logdir, logger):
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
    scheduler = step_scheduler(optimizer, cfg['scheduler']['step_size'], cfg['scheduler']['gamma'])

    # training and validation
    best_result = np.Inf
    n_epoch = cfg['scheduler']["max_epoch"]
    logger.info('Start training')
    for epoch in tqdm(range(n_epoch)):
        # training
        curr_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current lr: {curr_lr}")
        # train_meters = train(cfg, model, data_loader['train'], device, optimizer, epoch)
        train_meters = train(cfg, model, train_loader, device, optimizer, epoch)
        train_err = train_meters['depth_loss']
        logger.info(f"=== Epoch {epoch} ===")
        logger.info(f"Training error: {train_err}")
        logger.info(f"======")

        # validation
        if (epoch + 1) % cfg['eval_freq'] == 0:
            # eval_meters = inference(cfg, model, data_loader['test'], device)
            eval_meters = inference(cfg, model, test_loader, device)
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

    # eval_meters = inference(cfg, model, data_loader['test'], device)
    eval_meters = inference(cfg, model, test_loader, device)
    df = pd.DataFrame([eval_meters])
    logger.info(f"=== Final result ===")
    logger.info(df)
    logger.info(f"======")

def run_refine(cfg, logdir, logger):
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
    scheduler = step_scheduler(optimizer, cfg['scheduler']['step_size'], cfg['scheduler']['gamma'])

    # training and validation
    best_result = np.Inf
    n_epoch = cfg['scheduler']["max_epoch"]
    logger.info('Start training')
    for epoch in tqdm(range(n_epoch)):
        # training
        curr_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current lr: {curr_lr}")
        # train_meters = train(cfg, model, data_loader['train'], device, optimizer, epoch)
        train_meters = train(cfg, model, train_loader, device, optimizer, epoch)
        train_err = train_meters['depth_loss']
        logger.info(f"=== Epoch {epoch} ===")
        logger.info(f"Training error: {train_err}")
        logger.info(f"======")

        # validation
        if (epoch + 1) % cfg['eval_freq'] == 0:
            # eval_meters = inference(cfg, model, data_loader['test'], device)
            eval_meters = inference(cfg, model, test_loader, device)
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

    # eval_meters = inference(cfg, model, data_loader['test'], device)
    eval_meters = inference(cfg, model, test_loader, device)
    df = pd.DataFrame([eval_meters])
    logger.info(f"=== Final result ===")
    logger.info(df)
    logger.info(f"======")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # coarse prediction
    print('run coarse')
    run_coarse(cfg=cfg1, logdir=logdir1, logger=logger1) 

    # refine prediction
    print('run refine')
    run_refine(cfg=cfg2, logdir=logdir2, logger=logger2) 
    


if __name__ == '__main__':
    global cfg1, args, logger1, cfg2, logger2
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--config1',
        nargs='?',
        type=str,
        help='Configuration file to use',
    )

    parser.add_argument(
        '--config2',
        nargs='?',
        type=str,
        help='Configuration file to use',
    )

    args = parser.parse_args()

    # logger for config1
    with open(args.config1) as fp:
        cfg1 = yaml.load(fp, Loader=yaml.SafeLoader)

    logdir1 = os.path.join('runs', cfg1['data']['name'], cfg1['model']['arch'], cfg1['exp'])
    if not os.path.exists(logdir1):
        os.makedirs(logdir1)

    print("RUNDIR: {}".format(logdir1))
    shutil.copy(args.config1, logdir1)

    logger1 = get_logger(logdir1)
    logger1.info("Start logging")
    
    # logger for config2
    with open(args.config2) as fp:
        cfg2 = yaml.load(fp, Loader=yaml.SafeLoader)

    logdir2 = os.path.join('runs', cfg2['data']['name'], cfg2['model']['arch'], cfg2['exp'])
    if not os.path.exists(logdir2):
        os.makedirs(logdir2)

    print("RUNDIR: {}".format(logdir2))
    shutil.copy(args.config2, logdir2)

    logger2 = get_logger(logdir2)
    logger2.info("Start logging")
    
    main()

