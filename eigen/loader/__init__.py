
from .depth_data import DepthDataLoader
import os

def get_dataloader(cfg, splits):
    train_loader, test_loader = DepthDataLoader(cfg, splits)
    return train_loader, test_loader


