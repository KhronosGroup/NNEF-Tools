# Copyright (c) 2017-2025 The Khronos Group Inc.
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
from .converter import ConverterToSkriptND as _Converter, Transform
from ..model import Tensor, Operation
from ..model.utils import generate_missing_constant_and_variable_names
import numpy as np


class Converter(_Converter):

    def __init__(self, custom_transforms=None, custom_functions=None, mirror_unsupported=False):
        _Converter.__init__(self, transforms=self.merge_transforms(_Transforms, custom_transforms),
                            functions=custom_functions, mirror_unsupported=mirror_unsupported)

    def __call__(self, model):
        model = _Converter.__call__(self, model)
        generate_missing_constant_and_variable_names(model)
        return model

    def convert_padding(self, padding):
        return [p for p, q in padding] + [q for p, q in padding] if len(padding) != 0 else None

    def squeeze_vector(self, tensor):
        original = self._tensor_map[tensor]
        if tensor.data is not None and len(original.consumers) == 1:
            self._transform_constant(tensor, lambda data: np.squeeze(data, 0))
            return tensor
        else:
            return self.squeeze_input(tensor, axes=[0])

    def tile_vector(self, tensor, size):
        original = self._tensor_map[tensor]
        if tensor.data is not None and len(original.consumers) == 1:
            self._transform_constant(tensor, lambda data: np.tile(data, size))
            return tensor
        else:
            return self.tile_input(tensor, shape=[size])

    def norm_axes(self, size):
        return [i for i, s in enumerate(size) if s != 1]

    def norm_size(self, size):
        return [s for s in size if s != 1]


_Transforms = Converter.unpack_transforms({
    ('external', 'constant', 'variable'):
        Transform(type=None),
    ('conv', 'deconv'):
        Transform(
            type=('nn.conv', 'nn.deconv'),
            cond={
                '!border == "constant"': '!f"border must be constant for {_type_} operation"'
            },
            using={
                'transposed': '!_type_ == "deconv"',
            },
            inputs=(
                '!I[0]',
                '!I[1]',
                '!squeeze_vector(I[2]) if len(I[2].shape) == 2 else'
                ' None if len(I[2].shape) == 0 and I[2].data == 0 else'
                ' tile_vector(I[2], I[1].shape[0]) if len(I[2].shape) == 0 else I[2]',
            ),
            outputs='!O[0]',
            attribs={
                'padding': '!convert_padding(padding)',
                'stride': '!stride if len(stride) != 0 else None',
                'dilation': '!dilation if len(dilation) != 0 else None',
                'groups': '!groups',
                'output_size': '!output_shape[2:] if transposed and len(output_shape) != 0 else None',
            },
        ),
    'box':
        Transform(
            type='!"nn.avg_pool" if normalize else "nn.sum_pool"',
            cond={
                '!border == "constant" or border == "ignore"':
                    '!f"border must be constant or ignored for {_type_} operation"'
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'size': '!size',
                'padding': '!convert_padding(padding)',
                'stride': '!stride if len(stride) != 0 else None',
                'dilation': '!dilation if len(dilation) != 0 else None',
                'axes': '!list(range(I[0].rank))',
                'ignore_border': '!border == "ignore" if normalize else None'
            },
        ),
    ('max_pool', 'avg_pool', 'rms_pool'):
        Transform(
            type=('nn.max_pool', 'nn.avg_pool', 'nn.rms_pool'),
            cond={
                '!border == "constant" or border == "ignore"':
                    '!f"border must be constant or ignored for {_type_} operation"'
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'size': '!size',
                'padding': '!convert_padding(padding)',
                'stride': '!stride if len(stride) != 0 else None',
                'dilation': '!dilation if len(dilation) != 0 else None',
                'axes': '!list(range(I[0].rank))',
                'ignore_border': '!border == "ignore" if _type_ == "avg_pool" or _type_ == "rms_pool" else None'
            },
        ),
    ('copy', 'neg', 'rcp', 'sqr', 'sqrt', 'rsqr', 'rsqrt',
     'exp', 'log', 'log2', 'abs', 'sign',
     'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
     'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
     'floor', 'ceil', 'round', 'not',
     'sigmoid', 'relu', 'gelu', 'silu', 'softplus'):
        Transform(
            type=('math.iden', 'math.neg', 'math.rcp', 'math.sqr', 'math.sqrt', 'math.rsqr', 'math.rsqrt',
                  'math.exp', 'math.log', 'math.log2', 'math.abs', 'math.sign',
                  'math.sin', 'math.cos', 'math.tan', 'math.asin', 'math.acos', 'math.atan',
                  'math.sinh', 'math.cosh', 'math.tanh', 'math.asinh', 'math.acosh', 'math.atanh',
                  'math.floor', 'math.ceil', 'math.round', 'math.not',
                  'nn.sigmoid', 'nn.relu', 'nn.gelu', 'nn.silu', 'nn.softplus'),
            inputs='!I[0]',
            outputs='!O[0]',
        ),
    ('add', 'sub', 'mul', 'div', 'pow', 'min', 'max',
     'lt', 'gt', 'le', 'ge', 'eq', 'ne', 'and', 'or'):
        Transform(
            type=('math.add', 'math.sub', 'math.mul', 'math.div', 'math.pow', 'math.min', 'math.max',
                  'math.lt', 'math.gt', 'math.le', 'math.ge', 'math.eq', 'math.ne', 'math.and', 'math.or'),
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            attribs={
                'lhs_align': '!I[0].rank - O[0].rank if I[0].rank != O[0].rank else None',
                'rhs_align': '!I[1].rank - O[0].rank if I[1].rank != O[0].rank else None',
            },
        ),
    ('select', 'clamp'):
        Transform(
            type=('math.select', 'math.clamp'),
            inputs=('!I[0]', '!I[1]', '!I[2]'),
            outputs='!O[0]',
        ),
    ('sum_reduce', 'mean_reduce', 'min_reduce', 'max_reduce', 'any_reduce', 'all_reduce'):
        Transform(
            type=('!"math.sum_reduce" if not normalize else "math.mean_reduce"',
                  'math.mean_reduce', 'math.min_reduce', 'math.max_reduce', 'math.any_reduce', 'math.all_reduce'),
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
            },
        ),
    ('argmin_reduce', 'argmax_reduce'):
        Transform(
            type=('math.argmin', 'math.argmax'),
            cond={
                '!len(axes) == 1': 'length of axes must be 1',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axis': '!axes[0]',
            },
        ),
    'prelu':
        Transform(
            type='nn.prelu',
            inputs=(
                '!I[0]',
                '!squeeze_vector(I[1]) if len(I[1].shape) == 2 else'
                ' tile_vector(I[1], I[0].shape[1]) if len(I[1].shape) == 0 else I[1]',
            ),
            outputs='!O[0]',
        ),
    'leaky_relu':
        Transform(
            type='nn.relu',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'alpha': '!alpha',
            },
        ),
    ('elu', 'selu'):
        Transform(
            type=('nn.elu', 'nn.selu'),
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'alpha': '!alpha',
                'lambda': '!locals().get("lambda") if _type_ == "selu" else None',
            },
        ),
    'softmax':
        Transform(
            type='nn.softmax',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
            },
        ),
    'matmul':
        Transform(
            type='linalg.matmul',
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            attribs={
                'transA': '!transposeA',
                'transB': '!transposeB',
            },
        ),
    'linear':
        Transform(
            type='nn.linear',
            inputs=(
                '!I[0]',
                '!I[1]',
                '!squeeze_vector(I[2]) if len(I[2].shape) == 2 else'
                ' None if len(I[2].shape) == 0 and I[2].data == 0 else'
                ' tile_vector(I[2], I[1].shape[0]) if len(I[2].shape) == 0 else I[2]',
            ),
            outputs='!O[0]',
        ),
    'local_response_normalization':
        Transform(
            type='nn.local_response_norm',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!norm_axes(size)',
                'size': '!norm_size(size)',
                'alpha': '!alpha',
                'beta': '!beta',
                'bias': '!bias',
            },
        ),
    'batch_normalization':
        Transform(
            type='nn.batch_norm',
            inputs=(
                '!I[0]',
                '!squeeze_vector(I[1])',
                '!squeeze_vector(I[2])',
                '!squeeze_vector(I[3])',
                '!squeeze_vector(I[4])',
            ),
            outputs='!O[0]',
            attribs={
                'epsilon': '!epsilon',
            },
        ),
    ('l1_normalization', 'l2_normalization'):
        Transform(
            type=('nn.l1_norm', 'nn.l2_norm'),
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
                'bias': '!bias',
                'epsilon': '!epsilon',
            },
        ),
    ('nearest_downsample', 'nearest_upsample', 'area_downsample'):
        Transform(
            type=('image.nearest_downsample', 'image.nearest_upsample', 'image.area_downsample'),
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!list(range(2, I[0].rank))',
                'factor': '!factor',
            },
        ),
    'multilinear_upsample':
        Transform(
            type='!"image.linear_resize" if as_resize else "image.linear_upsample"',
            cond={
                '!border == "constant" || border == "replicate"': 'border must be "constant" or "replicate"',
                '!method != "aligned" || border == "replicate"': 'border must be "replicate" if method is "aligned"'
            },
            using={
                'as_resize': '!method == "aligned" or border == "replicate"',
                'as_upsample': '!not as_resize',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!list(range(2, I[0].rank))',
                'size': '![s * f for s, f in zip(I[0].shape[2:], factor)] if as_resize else None',
                'factor': '!factor if as_upsample else None',
                'symmetric': '!method == "symmetric" if as_upsample else None',
                'replicate_border': '!border == "replicate" if as_upsample else None',
                'coordinate_transform': '!method.upper() if as_resize else None',
            },
        ),
})
