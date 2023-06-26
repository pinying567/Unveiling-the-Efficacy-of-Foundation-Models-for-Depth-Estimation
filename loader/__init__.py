
from .depth_data import DepthDataLoader


def get_dataloader(cfg, splits):
    loader = DepthDataLoader(cfg, splits)
    return loader


