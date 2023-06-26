
import copy

from .depth_clip import depth_clip
from .depth_adapter import depth_adapter
from .depth_convadapter import depth_convadapter


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
            'depth_clip': depth_clip,
            'depth_adapter': depth_adapter,
            'depth_convadapter': depth_convadapter
        }[name]
    except:
        raise BaseException('Model {} not available'.format(name))


