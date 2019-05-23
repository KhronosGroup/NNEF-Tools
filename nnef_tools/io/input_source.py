# Copyright (c) 2017 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division, print_function, absolute_import

import glob
import math
import os

import nnef
import numpy as np
import six
import typing

from nnef_tools.core import utils


class InputSource(object):
    pass


class RandomInput(InputSource):
    def __init__(self, algo, *args):
        algo = algo.lower()
        if algo == 'uniform':
            if len(args) != 2:
                raise utils.NNEFToolsException("Random 'uniform' must have two parameters: min, max.")
        elif algo == 'normal':
            if len(args) != 2:
                raise utils.NNEFToolsException("Random 'normal' must have two parameters: mean, std.")
        elif algo == 'binomial':
            if len(args) != 2:
                raise utils.NNEFToolsException("Random 'binomial' must have two parameters: num, true_prob.")
        elif algo == 'bernoulli':
            if len(args) != 1:
                raise utils.NNEFToolsException("Random 'bernoulli' must have one parameter: true_prob.")
            if not (0.0 <= args[0] <= 1.0):
                raise utils.NNEFToolsException("Random 'bernoulli': true_prob must be between 0.0 and 1.0.")
        else:
            raise utils.NNEFToolsException("Unknown random algo: {}".format(algo))

        self.algo = algo
        self.args = args


class ImageInput(InputSource):
    COLOR_FORMAT_RGB = 'RGB'
    COLOR_FORMAT_BGR = 'BGR'
    DATA_FORMAT_NCHW = 'NCHW'
    DATA_FORMAT_NHWC = 'NHWC'

    def __init__(self, filename, color_format='RGB', data_format='NCHW', range=None, norm=None):
        self.filenames = filename if isinstance(filename, (list, tuple)) else [filename]
        self.color_format = color_format
        self.data_format = data_format
        self.range = range
        self.norm = norm


class NNEFTensorInput(InputSource):
    def __init__(self, filename):
        self.filename = filename


def create_input(input_source, np_dtype, shape, allow_bigger_batch=False):
    assert input_source is None or isinstance(input_source, (RandomInput, ImageInput, NNEFTensorInput))
    np_dtype = np.dtype(np_dtype)
    if input_source is None:
        if 'float' in np_dtype.name:
            input_source = RandomInput('normal', 0.0, 1.0)
        elif 'int' in np_dtype.name:
            input_source = RandomInput('binomial', 255, 0.5)
        elif 'bool' == np_dtype.name:
            input_source = RandomInput('bernoulli', 0.5)
        else:
            raise utils.NNEFToolsException("Random does not support this dtype: {}".format(np_dtype.name))
    if isinstance(input_source, RandomInput):
        if input_source.algo == 'uniform':
            if 'float' in np_dtype.name:
                return np.random.uniform(input_source.args[0], input_source.args[1], shape).astype(np_dtype)
            elif 'int' in np_dtype.name:
                return np.random.randint(int(math.ceil(input_source.args[0])),
                                         int(math.floor(input_source.args[1])) + 1,
                                         shape, np_dtype)
            else:
                raise Exception("Random 'uniform' can not be applied to: {}".format(np_dtype.name))
        elif input_source.algo == 'normal':
            if 'float' in np_dtype.name:
                return np.random.normal(input_source.args[0], input_source.args[1], shape).astype(np_dtype)
            else:
                raise Exception("Random 'normal' can not be applied to: {}".format(np_dtype.name))
        elif input_source.algo == 'binomial':
            if 'int' in np_dtype.name:
                return np.random.binomial(input_source.args[0], input_source.args[1], shape).astype(np_dtype)
            else:
                raise Exception("Random 'normal' can not be applied to: {}".format(np_dtype.name))
        elif input_source.algo == 'bernoulli':
            if 'bool' == np_dtype.name:
                return np.random.uniform(0.0, 1.0, shape) <= input_source.args[0]
            else:
                raise Exception("Random 'bernoulli' can not be applied to: {}".format(np_dtype.name))
        else:
            assert False
    elif isinstance(input_source, ImageInput):
        import skimage
        import skimage.io
        import skimage.transform

        assert len(shape) == 4, "ImageInput can only produce tensors with rank=4"
        assert input_source.data_format.upper() in [ImageInput.DATA_FORMAT_NCHW, ImageInput.DATA_FORMAT_NHWC]
        assert input_source.color_format.upper() in [ImageInput.COLOR_FORMAT_RGB, ImageInput.COLOR_FORMAT_BGR]
        imgs = []
        for pattern in input_source.filenames:
            filenames = sorted(glob.glob(os.path.expanduser(pattern)))
            assert filenames, "No files found for path: {}".format(pattern)
            for filename in filenames:
                if input_source.data_format.upper() == ImageInput.DATA_FORMAT_NCHW:
                    if shape[1] != 3:
                        raise utils.NNEFToolsException(
                            'NCHW image is specified as input, but channel dimension of input tensor is not 3.')
                    target_size = [shape[2], shape[3]]
                else:
                    if shape[3] != 3:
                        raise utils.NNEFToolsException(
                            'NHWC image is specified as input, but channel dimension of input tensor is not 3.')
                    target_size = [shape[1], shape[2]]

                img = skimage.img_as_ubyte(skimage.io.imread(filename))
                img = img.astype(np.float32)

                if input_source.color_format.upper() == ImageInput.COLOR_FORMAT_RGB:
                    img = img[..., (0, 1, 2)]  # remove alpha channel if present
                else:
                    img = img[..., (2, 1, 0)]

                if input_source.range:
                    min_ = np.array(input_source.range[0], dtype=np.float32)
                    max_ = np.array(input_source.range[1], dtype=np.float32)
                    scale = (max_ - min_) / 255.0
                    bias = min_
                    img = img / scale + bias

                if input_source.norm:
                    mean = np.array(input_source.norm[0], dtype=np.float32)
                    std = np.array(input_source.norm[1], dtype=np.float32)
                    img = (img - mean) / std

                img = skimage.transform.resize(img, target_size,
                                               preserve_range=True,
                                               anti_aliasing=True,
                                               mode='reflect')

                img = img.astype(np_dtype)

                if input_source.data_format.upper() == ImageInput.DATA_FORMAT_NCHW:
                    img = img.transpose((2, 0, 1))

                img = np.expand_dims(img, 0)

                imgs.append(img)
        if len(imgs) < shape[0]:
            print("Info: Network batch size bigger than supplied data, repeating it")
            imgs = imgs * ((shape[0] + len(imgs) - 1) // len(imgs))
            imgs = imgs[:shape[0]]
            assert len(imgs) == shape[0]
        assert len(imgs) == shape[0] or allow_bigger_batch
        return np.concatenate(tuple(imgs), 0)
    elif isinstance(input_source, NNEFTensorInput):
        with open(input_source.filename) as f:
            return nnef.read_tensor(f)[0]
    else:
        assert False


def create_feed_dict(input_sources,  # type: typing.Union[None, InputSource, typing.Dict[str, InputSource]]
                     input_shapes,  # type: typing.Dict[str, typing.Tuple[np.dtype, typing.List[int]]]
                     ):
    # type: (...)->typing.Dict[str, np.ndarray]
    if not isinstance(input_sources, dict):
        input_sources = {k: input_sources for k in six.iterkeys(input_shapes)}

    feed_dict = {}
    for name, (dtype, shape) in six.iteritems(input_shapes):
        assert name in input_sources
        feed_dict[name] = create_input(input_source=input_sources[name], np_dtype=dtype, shape=shape)

    return feed_dict
