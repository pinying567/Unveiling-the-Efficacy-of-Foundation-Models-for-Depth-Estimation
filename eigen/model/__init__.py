
import copy

from .depth_coarse import depth_coarse
from .depth_refine import depth_refine
from .depth_eigen_coarse import depth_eigen_coarse
from .depth_eigen_refine import depth_eigen_refine


def get_model(model_dict):

    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    model = model(**param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            'depth_coarse': depth_coarse,
            'depth_refine': depth_refine,
            'depth_eigen_coarse': depth_eigen_coarse,
            'depth_eigen_refine': depth_eigen_refine
        }[name]
    except:
        raise BaseException('Model {} not available'.format(name))


