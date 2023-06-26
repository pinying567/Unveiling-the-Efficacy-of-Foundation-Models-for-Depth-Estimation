# Ref: https://github.com/simonmeister/pytorch-mono-depth/blob/master/dense_estimation/datasets/image_utils.py

import numpy as np
import scipy
import scipy.ndimage
import collections
from PIL import Image
from torchvision.transforms import Lambda
import numbers

__author__ = "Wei OUYANG"
__license__ = "GPL"
__version__ = "0.1.0"
__status__ = "Development"


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def to_tensor(x):
    import torch
    x = x.transpose((2, 0, 1))
    x = np.ascontiguousarray(x)
    return torch.from_numpy(x).float()


class Merge(object):
    """Merge a group of images
    """

    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, images):
        if isinstance(images, collections.Sequence) or isinstance(images, np.ndarray):
            assert all([isinstance(i, np.ndarray)
                        for i in images]), 'only numpy array is supported'
            shapes = [list(i.shape) for i in images]
            for s in shapes:
                s[self.axis] = None
            assert all([s == shapes[0] for s in shapes]
                       ), 'shapes must be the same except the merge axis'
            return np.concatenate(images, axis=self.axis)
        else:
            raise Exception("obj is not a sequence (list, tuple, etc)")


class Split(object):
    """Split images into individual arraies
    """

    def __init__(self, *slices, **kwargs):
        assert isinstance(slices, collections.Sequence)
        slices_ = []
        for s in slices:
            if isinstance(s, collections.Sequence):
                slices_.append(slice(*s))
            else:
                slices_.append(s)
        assert all([isinstance(s, slice) for s in slices_]
                   ), 'slices must be consist of slice instances'
        self.slices = slices_
        self.axis = kwargs.get('axis', -1)

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            ret = []
            for s in self.slices:
                sl = [slice(None)] * image.ndim
                sl[self.axis] = s
                ret.append(image[tuple(sl)])
            return ret
        else:
            raise Exception("obj is not an numpy array")


class MaxScaleNumpy(object):
    """scale with max and min of each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, range_min=0.0, range_max=1.0):
        self.scale = (range_min, range_max)

    def __call__(self, image):
        mn = image.min(axis=(0, 1))
        mx = image.max(axis=(0, 1))
        return self.scale[0] + (image - mn) * (self.scale[1] - self.scale[0]) / (mx - mn)


class MedianScaleNumpy(object):
    """Scale with median and mean of each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, range_min=0.0, range_max=1.0):
        self.scale = (range_min, range_max)

    def __call__(self, image):
        mn = image.min(axis=(0, 1))
        md = np.median(image, axis=(0, 1))
        return self.scale[0] + (image - mn) * (self.scale[1] - self.scale[0]) / (md - mn)


class NormalizeNumpy(object):
    """Normalize each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """

    def __call__(self, image):
        image -= image.mean(axis=(0, 1))
        s = image.std(axis=(0, 1))
        s[s == 0] = 1.0
        image /= s
        return image


class MutualExclude(object):
    """Remove elements from one channel
    """

    def __init__(self, exclude_channel, from_channel):
        self.from_channel = from_channel
        self.exclude_channel = exclude_channel

    def __call__(self, image):
        mask = image[:, :, self.exclude_channel] > 0
        image[:, :, self.from_channel][mask] = 0
        return image


class RandomCropNumpy(object):
    """Crops the given numpy array at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, random_state=np.random):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.random_state = random_state

    def __call__(self, img):
        h, w = img.shape[:2]
        th, tw = self.size
        if w == tw and h == th:
            return img
        elif w == tw:
            x1 = 0
            y1 = self.random_state.randint(0, h - th)
        elif h == th:
            x1 = self.random_state.randint(0, w - tw)
            y1 = 0
        else:
            x1 = self.random_state.randint(0, w - tw)
            y1 = self.random_state.randint(0, h - th)

        return img[y1:y1 + th, x1:x1 + tw, :]


class CenterCropNumpy(object):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[y1:y1 + th, x1:x1 + tw, :]


class RandomRotate(object):
    """Rotate a PIL.Image or numpy.ndarray (H x W x C) randomly
    """

    def __init__(self, angle_range=(0.0, 360.0), axes=(0, 1), mode='reflect', random_state=np.random):
        assert isinstance(angle_range, tuple)
        self.angle_range = angle_range
        self.random_state = random_state
        self.axes = axes
        self.mode = mode

    def __call__(self, image):
        angle = self.random_state.uniform(
            self.angle_range[0], self.angle_range[1])
        if isinstance(image, np.ndarray):
            mi, ma = image.min(), image.max()
            image = scipy.ndimage.interpolation.rotate(
                image, angle, reshape=False, axes=self.axes, mode=self.mode)
            return np.clip(image, mi, ma)
        elif isinstance(image, Image.Image):
            return image.rotate(angle)
        else:
            raise Exception('unsupported type')


class RandomHorizontalFlip(object):
    """Flip a numpy.ndarray (H x W x C) horizontally with probability 0.5
    """

    def __init__(self, random_state=np.random):
        self.random_state = random_state

    def __call__(self, image):
        val = self.random_state.uniform()
        if isinstance(image, np.ndarray):
            if val > 0.5:
                return image[:, ::-1, :]
            return image
        else:
            raise Exception('unsupported type')


class RandomColor(object):
    """Multiply numpy.ndarray (H x W x C) globally
    """

    def __init__(self, multiplier_range=(0.8, 1.2), random_state=np.random):
        assert isinstance(multiplier_range, tuple)
        self.multiplier_range = multiplier_range
        self.random_state = random_state

    def __call__(self, image):
        mult = self.random_state.uniform(self.multiplier_range[0],
                                         self.multiplier_range[1])
        if isinstance(image, np.ndarray):
            return np.clip(image * mult, 0, 255)
        else:
            raise Exception('unsupported type')


class BilinearResize(object):
    """Resize a PIL.Image or numpy.ndarray (H x W x C)
    """

    def __init__(self, zoom):
        self.zoom = [zoom, zoom, 1]

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            return scipy.ndimage.interpolation.zoom(image, self.zoom, order=1)
        elif isinstance(image, Image.Image):
            return image.resize(self.size, Image.BILINEAR)
        else:
            raise Exception('unsupported type')


class EnhancedCompose(object):
    """Composes several transforms together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, collections.Sequence):
                assert isinstance(img, collections.Sequence) and len(img) == len(
                    t), "size of image group and transform group does not fit"
                tmp_ = []
                for i, im_ in enumerate(img):
                    if callable(t[i]):
                        tmp_.append(t[i](im_))
                    else:
                        tmp_.append(im_)
                img = tmp_
            elif callable(t):
                img = t(img)
            elif t is None:
                continue
            else:
                raise Exception('unexpected type')
        return img


if __name__ == '__main__':
    input_channel = 3
    target_channel = 3

    # define a transform pipeline
    transform = EnhancedCompose([
        Merge(),
        RandomCropNumpy(size=(512, 512)),
        RandomRotate(),
        Split([0, input_channel], [input_channel, input_channel+target_channel]),
        [CenterCropNumpy(size=(256, 256)), CenterCropNumpy(size=(256, 256))],
        [NormalizeNumpy(), MaxScaleNumpy(0, 1.0)],
        # for non-pytorch usage, remove to_tensor conversion
        [Lambda(to_tensor), Lambda(to_tensor)]
    ])
    # read input data for test
    image_in = np.array(Image.open('input.jpg'))
    image_target = np.array(Image.open('target.jpg'))

    # apply the transform
    x, y = transform([image_in, image_target])
