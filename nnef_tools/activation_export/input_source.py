from __future__ import division, print_function, absolute_import

import nnef
import numpy as np


class RandomInput(object):
    def __init__(self, *args):
        assert len(args) in [1, 2, 5]

        if len(args) == 1:
            self.float_min = None
            self.float_max = None
            self.int_min = None
            self.int_max = None
            self.true_prob = args[0]
        elif len(args) == 2:
            self.float_min, self.float_max = args
            self.int_min, self.int_max = args
            self.true_prob = None
        elif len(args) == 5:
            self.float_min, self.float_max, self.int_min, self.int_max, self.true_prob = args
        else:
            assert False


class ImageInput(object):
    COLOR_FORMAT_RGB = 'RGB'
    COLOR_FORMAT_BGR = 'BGR'
    DATA_FORMAT_NCHW = 'NCHW'
    DATA_FORMAT_NHWC = 'NHWC'

    def __init__(self, filename, color_format='RGB', data_format='NCHW', sub=127.5, div=127.5):
        self.filenames = filename if isinstance(filename, (list, tuple)) else [filename]
        self.color_format = color_format
        self.data_format = data_format
        self.sub = sub
        self.div = div


class NNEFTensorInput(object):
    def __init__(self, filename):
        self.filename = filename


def create_input(input_source, np_dtype, shape):
    assert isinstance(input_source, (RandomInput, ImageInput, NNEFTensorInput))
    np_dtype = np.dtype(np_dtype)
    if isinstance(input_source, RandomInput):
        if 'float' in np_dtype.name:
            assert input_source.float_min is not None and input_source.float_max is not None, \
                "float_min or float_max is not set on the input source"
            return ((input_source.float_max - input_source.float_min)
                    * np.random.random(shape) + input_source.float_min).astype(np_dtype)
        elif 'int' in np_dtype.name:
            assert input_source.int_min is not None and input_source.int_max is not None, \
                "int_min or int_max is not set on the input source"
            return np.random.randint(low=input_source.int_min, high=input_source.int_max, size=shape, dtype=np_dtype)
        elif np_dtype.name == 'bool':
            assert input_source.true_prob is not None, "true_prob is not set on the input source"
            return np.random.random(shape) <= input_source.true_prob
        else:
            assert False, "Unsupported dtype: {}".format(np_dtype.name)
    elif isinstance(input_source, ImageInput):
        from matplotlib.image import imread
        from scipy.misc import imresize

        assert len(shape) == 4, "ImageInput can only produce tensors with rank=4"
        assert input_source.data_format.upper() in [ImageInput.DATA_FORMAT_NCHW, ImageInput.DATA_FORMAT_NHWC]
        assert input_source.color_format.upper() in [ImageInput.COLOR_FORMAT_RGB, ImageInput.COLOR_FORMAT_BGR]
        imgs = []
        for filename in input_source.filenames:
            if input_source.data_format.upper() == ImageInput.DATA_FORMAT_NCHW:
                target_size = [shape[2], shape[3], shape[1]]
            else:
                target_size = [shape[1], shape[2], shape[3]]
            img = imread(filename)
            if input_source.color_format.upper() == ImageInput.COLOR_FORMAT_RGB:
                img = img[..., (0, 1, 2)]  # remove alpha channel if present
            else:
                img = img[..., (2, 1, 0)]
            img = ((img.astype(np.float32) - np.array(input_source.sub)) / np.array(input_source.div))
            img = imresize(img, target_size).astype(np_dtype)
            if input_source.data_format.upper() == ImageInput.DATA_FORMAT_NCHW:
                img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, 0)
            imgs.append(img)

        if len(imgs) < shape[0]:
            imgs = imgs * ((shape[0] + len(imgs) - 1) / len(imgs))
            imgs = imgs[:shape[0]]
            assert len(imgs) == shape[0]

        return np.concatenate(a_tuple=tuple(imgs), axis=0)
    elif isinstance(input_source, NNEFTensorInput):
        with open(input_source.filename) as f:
            return nnef.read_tensor(f)[0]
    else:
        assert False
