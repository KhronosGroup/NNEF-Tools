# Copyright (c) 2020 The Khronos Group Inc.
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
from .converter import ConverterToNNEF as _Converter, Transform
from ..model.utils import generate_tensor_names_from_op_type
from ..model import Tensor
from collections import OrderedDict
import numpy as np
import copy


_LP_POOL_FRAGMENT = """
fragment lp_pool( 
    input: tensor<scalar>,
    size: integer[],
    border: string = 'constant',
    padding: (integer, integer)[] = [],
    stride: integer[] = [],
    dilation: integer[] = [],
    p: scalar = 2.0 ) 
-> ( output: tensor<scalar> )
{
    powered = pow(abs(input), p);
    summed = box(powered, size = size, border = border, padding = padding, stride = stride, dilation = dilation);
    output = pow(summed, 1.0 / p);
}
"""

_LP_REDUCE_FRAGMENT = """
fragment lp_reduce( 
    input: tensor<scalar>,
    axes: integer[],
    p: scalar = 2.0 ) 
-> ( output: tensor<scalar> )
{
    powered = pow(abs(input), p);
    summed = sum_reduce(powered, axes = axes);
    output = pow(summed, 1.0 / p);
}
"""

_MEAN_VARIANCE_NORMALIZATION_FRAGMENT = """
fragment mean_variance_normalization( 
    input: tensor<scalar>,
    scale: tensor<scalar>,
    offset: tensor<scalar>,
    axes: integer[],
    epsilon: scalar = 1e-5 ) 
-> ( output: tensor<scalar> )
{
    mean, variance = moments(input, axes = axes);
    output = scale * (input - mean) / sqrt(variance + epsilon) + offset;
}
"""


class Converter(_Converter):

    @staticmethod
    def defined_operations():
        return {
            'lp_pool': _LP_POOL_FRAGMENT,
            'lp_reduce': _LP_REDUCE_FRAGMENT,
            'mean_variance_normalization': _MEAN_VARIANCE_NORMALIZATION_FRAGMENT,
        }

    def __init__(self, custom_transforms=None, custom_functions=None, mirror_unsupported=False, keep_io_names=False):
        _Converter.__init__(self, transforms=self.merge_transforms(_Transforms, custom_transforms),
                            functions=custom_functions, mirror_unsupported=mirror_unsupported)
        self._keep_io_names = keep_io_names

    def __call__(self, graph):
        graph = _Converter.__call__(self, graph)
        self.remove_unused_constants(graph)
        self.inline_scalar_constants(graph)
        self.convert_constants_to_variables(graph)
        self._ensure_valid_ids(graph)
        generate_tensor_names_from_op_type(graph, keep_io_names=self._keep_io_names)
        return graph

    def _prepare(self, graph):
        self._insert_externals_and_constants(graph)

    @staticmethod
    def _interleave(items):
        return [item[0] for item in items] + [item[1] for item in items]

    @staticmethod
    def _uninterleave(items):
        count = len(items) // 2
        return list(zip(items[:count], items[count:]))

    def convert_padding(self, pads, auto_pad, rank):
        if auto_pad == "NOTSET" or auto_pad == "SAME_LOWER":
            padding = self._uninterleave(pads)
            return [(0, 0)] * (rank - len(padding)) + padding
        elif auto_pad == "VALID":
            return [(0, 0,)] * rank
        elif auto_pad == "SAME_UPPER":
            return []
        else:
            assert False

    def convert_pads(self, pads):
        return self._uninterleave(pads)

    def squeeze_input(self, tensor, axes, keep_dims=False):
        return self._pre_squeeze(tensor, axes=axes) if not keep_dims and len(axes) else tensor

    def squeeze_output(self, tensor, axes, keep_dims=False):
        return self._post_squeeze(tensor, axes=axes) if not keep_dims and len(axes) else tensor

    def unsqueeze_input(self, tensor, axes, keep_dims=False):
        return self._pre_unsqueeze(tensor, axes=axes) if not keep_dims and len(axes) else tensor

    def unsqueeze_output(self, tensor, axes, keep_dims=False):
        return self._post_unsqueeze(tensor, axes=axes) if not keep_dims and len(axes) else tensor

    def unsqueeze_vector(self, tensor):
        if self._is_constant(tensor) and len(self._tensor_map[tensor].consumers) == 1:
            self._transform_constant(tensor, lambda data: np.expand_dims(data, 0))
            return tensor
        else:
            return self.unsqueeze_input(tensor, axes=[0])

    def bias_add(self, output, bias):
        if bias.rank == 0 and bias.data == 0:
            return output

        input = Tensor(output.graph, dtype=output.dtype, shape=output.shape, quant=copy.deepcopy(output.quant))
        self._bias_operation(input, output, bias)
        return input

    def lower_pads(self, input_size, filter_size, output_size, stride, dilation):
        rank = len(input_size)
        total = [None] * rank
        for i in range(rank):
            dilated_size = (filter_size[i] - 1) * dilation[i] + 1
            total[i] = max((input_size[i] // stride[i] - 1) * stride[i] + dilated_size - output_size[i], 0)
        pads = [(t // 2, t - t // 2) for t in total]
        return self._interleave(pads)

    def broadcast(self, tensor, rank):
        return self.unsqueeze_input(tensor, axes=list(range(rank - tensor.rank)))

    def ensure_list(self, arg):
        return [arg] if not isinstance(arg, list) else arg

    def ensure_scalar(self, arg):
        return arg[0] if isinstance(arg, list) and len(arg) == 1 else arg


_Transforms = Converter.unpack_transforms({
    ('Conv', 'ConvTranspose'):
        Transform(
            type=('conv', 'deconv'),
            defaults={
                'strides': '![1] * (I[0].rank - 2)',
                'dilations': '![1] * (I[0].rank - 2)',
                'pads': '![0, 0] * (I[0].rank - 2)',
                'auto_pad': "NOTSET",
                'group': 1,
                'output_shape': None,
            },
            using={
                '_pads': '!lower_pads(I[0].shape[2:], I[1].shape[2:], O[0].shape[2:], strides, dilations)'
                         ' if auto_pad == "SAME_LOWER" else pads',
            },
            inputs=(
                '!I[0]',
                '!I[1]',
                '!unsqueeze_vector(I[2]) if len(I) > 2 else None',
            ),
            outputs='!O[0]',
            attribs={
                'stride': '!strides',
                'dilation': '!dilations',
                'padding': '!convert_padding(_pads, auto_pad, I[0].rank - 2)',
                'groups': '!group',
                'output_shape': '!output_shape',
            }
        ),
    ('MaxPool', 'AveragePool', 'LpPool'):
        Transform(
            type=('max_pool', 'avg_pool', 'lp_pool'),
            defaults={
                'strides': '![1] * (I[0].rank - 2)',
                'dilations': '![1] * (I[0].rank - 2)',
                'pads': '![0, 0] * (I[0].rank - 2)',
                'auto_pad': "NOTSET",
                'ceil_mode': 0,
                'storage_order': 0,
                'count_include_pad': 0,
            },
            cond='!ceil_mode == 0 and storage_order == 0',
            using={
                '_pads': '!lower_pads(I[0].shape[2:], kernel_shape, O[0].shape[2:], strides, dilations)'
                         ' if auto_pad == "SAME_LOWER" else pads',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'size': '![1, 1] + kernel_shape',
                'stride': '![1, 1] + strides',
                'dilation': '![1, 1] + dilations',
                'padding': '!convert_padding(_pads, auto_pad, I[0].rank)',
                'border': '!"constant" if count_include_pad else "ignore"',
            }
        ),
    ('GlobalMaxPool', 'GlobalAveragePool', 'GlobalLpPool'):
        Transform(
            type=('max_reduce', 'mean_reduce', 'lp_reduce'),
            defaults={
                'p': 2,
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!list(range(2, I[0].rank))',
                'p': '!float(p) if _type_ == "GlobalLpPool" else None',
            }
        ),
    ('ReduceMin', 'ReduceMax', 'ReduceMean', 'ReduceSum', 'ReduceL1', 'ReduceL2'):
        Transform(
            type=('min_reduce', 'max_reduce', 'mean_reduce', 'sum_reduce', 'lp_reduce', 'lp_reduce'),
            defaults={
                'keepdims': 1,
            },
            inputs='!I[0]',
            outputs='!squeeze_output(O[0], axes, keepdims)',
            attribs={
                'axes': '!ensure_positive(axes, I[0].rank)',
                'p': '!1.0 if _type_ == "ReduceL1" else 2.0 if _type_ == "ReduceL2" else None',
            }
        ),
    ('ArgMin', 'ArgMax'):
        Transform(
            type=('argmin_reduce', 'argmax_reduce'),
            defaults={
                'axis': 0,
                'keepdims': 1,
                'select_last_index': 0,
            },
            cond='!not select_last_index',
            inputs='!I[0]',
            outputs='!squeeze_output(O[0], [axis], keepdims)',
            attribs={
                'axes': '![axis]',
            }
        ),
    'BatchNormalization':
        Transform(
            type='batch_normalization',
            defaults={
                'epsilon': 1e-5,
                'spatial': 1,
            },
            inputs=(
                '!I[0]',
                '!unsqueeze_vector(I[3])',
                '!unsqueeze_vector(I[4])',
                '!unsqueeze_vector(I[2])',
                '!unsqueeze_vector(I[1])',
            ),
            outputs='!O[0]',
            attribs={
                'epsilon': '!epsilon',
            }
        ),
    ('Relu', 'Sigmoid', 'Tanh', 'Softplus', 'Not', 'Identity', 'Elu', 'Abs', 'Sign',
     'Cos', 'Sin', 'Exp', 'Log', 'Neg', 'Sqrt', 'Ceil', 'Floor', 'Round'):
        Transform(
            type=('relu', 'sigmoid', 'tanh', 'softplus', 'not', 'copy', 'elu', 'abs', 'sign',
                  'cos', 'sin', 'exp', 'log', 'neg', 'sqrt', 'ceil', 'floor', 'round'),
            inputs='!I[0]',
            outputs='!O[0]',
        ),
    ('Add', 'Sub', 'Mul', 'Div', 'Pow', 'Min', 'Max', 'And', 'Or',
     'Equal', 'Less', 'Greater', 'LessOrEqual', 'GreaterOrEqual'):
        Transform(
            type=('add', 'sub', 'mul', 'div', 'pow', 'min', 'max', 'and', 'or', 'eq', 'lt', 'gt', 'le', 'ge'),
            inputs=(
                '!broadcast(I[0], O[0].rank)',
                '!broadcast(I[1], O[0].rank)',
            ),
            outputs='!O[0]',
        ),
    'LeakyRelu':
        Transform(
            type='leaky_relu',
            defaults={
                'alpha': 0.01,
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'alpha': '!alpha',
            }
        ),
    'PRelu':
        Transform(
            type='prelu',
            inputs=(
                '!I[0]',
                '!broadcast(I[1], I[0].rank)',
            ),
            outputs='!O[0]',
        ),
    'Transpose':
        Transform(
            type='transpose',
            defaults={
                'perm': '!list(reversed(range(I[0].rank)))',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!perm',
            }
        ),
    'Reshape':
        Transform(
            type='reshape',
            defaults={
                'shape': '!as_const(I[1])',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'shape': '!flexible_batch(shape, I[0].shape[0])',
            }
        ),
    'Flatten':
        Transform(
            type='reshape',
            defaults={
                'axis': 1,
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'shape': '![0] * axis + [-1]',
            }
        ),
    'Squeeze':
        Transform(
            type='squeeze',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
            }
        ),
    'Unsqueeze':
        Transform(
            type='unsqueeze',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
            }
        ),
    'MatMul':
        Transform(
            type='matmul',
            inputs=(
                '!broadcast(I[0], O[0].rank)',
                '!broadcast(I[1], O[0].rank)',
            ),
            outputs=(
                '!O[0]',
            ),
        ),
    'Gemm':
        Transform(
            type='!"linear" if is_linear else "matmul"',
            defaults={
                'alpha': 1.0,
                'beta': 1.0,
                'transA': 0,
                'transB': 0,
            },
            cond='!alpha == 1.0 and (beta == 1.0 or len(I) == 2)',
            using={
                'is_linear': '!len(I) > 2 and I[2].rank == 1 and transB',
                'bias': '!broadcast(I[2], O[0].rank) if len(I) > 2 and not is_linear else None',
            },
            inputs=(
                '!I[0]',
                '!I[1]',
                '!unsqueeze_vector(I[2]) if is_linear else None',
            ),
            outputs='!O[0] if is_linear or bias is None else bias_add(O[0], bias)',
            attribs={
                'transposeA': '!bool(transA) if not is_linear else None',
                'transposeB': '!bool(transB) if not is_linear else None',
            }
        ),
    'LRN':
        Transform(
            type='local_response_normalization',
            defaults={
                'alpha': 0.0001,
                'beta': 0.75,
                'bias': 1.0,
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'alpha': '!alpha',
                'beta': '!beta',
                'bias': '!bias',
                'size': '![1, size] + [1] * (I[0].rank - 2)',
            }
        ),
    'Concat':
        Transform(
            type='concat',
            defaults={
                'axis': 1,
            },
            inputs=['!I[:]'],
            outputs='!O[0]',
            attribs={
                'axis': '!axis',
            }
        ),
    'Split':
        Transform(
            type='split',
            defaults={
                'axis': 0,
            },
            inputs='!I[0]',
            outputs=['!O[:]'],
            attribs={
                'axis': '!axis',
                'ratios': '!split',
            }
        ),
    'Dropout':
        Transform(
            type='copy',
            inputs='!I[0]',
            outputs='!O[0]',
        ),
    'Softmax':
        Transform(
            type='softmax',
            defaults={
                'axis': 1,
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '![axis]',
            }
        ),
    'Sum':
        Transform(
            type='add_n',
            inputs=['!I[:]'],
            outputs='!O[0]',
        ),
    'Where':
        Transform(
            type='select',
            inputs=(
                '!broadcast(I[0], O[0].rank)',
                '!broadcast(I[1], O[0].rank)',
                '!broadcast(I[2], O[0].rank)',
            ),
            outputs='!O[0]',
        ),
    'Clip':
        Transform(
            type='clamp',
            inputs=(
                '!I[0]',
                '!I[1]',
                '!I[2]',
            ),
            outputs='!O[0]',
        ),
    'Pad':
        Transform(
            type='pad',
            defaults={
                'mode': "constant",
                'value': 0.0,
            },
            using={
                'constant_value': '!ensure_scalar(as_const(I[2])) if len(I) > 2 else value',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'padding': '!convert_pads(as_const(I[1]) if len(I) > 1 else pads)',
                'value': '!constant_value',
                'border': '!"replicate" if mode == "edge" else mode',
            }
        ),
    'Tile':
        Transform(
            type='tile',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'repeats': '!as_const(I[1])',
            }
        ),
    'Slice':
        Transform(
            type='slice',
            defaults={
                'axes': '!as_const(I[3]) if len(I) > 3 else list(range(I[0].rank))',
                'starts': '!as_const(I[1])',
                'ends': '!as_const(I[2])',
                'steps': '!as_const(I[4]) if len(I) > 4 else None',
            },
            cond='steps is None or all(s == 1 for s in steps)',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!ensure_positive(axes, I[0].rank)',
                'begin': '!starts',
                'end': '!ends',
            }
        ),
    'LpNormalization':
        Transform(
            type='!"l1_normalization" if p == 1 else "l2_normalization"',
            defaults={
                'axis': -1,
                'p': 2,
            },
            cond='!p == 1 or p == 2',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '![ensure_positive(axis, I[0].rank)]',
            }
        ),
    'MeanVarianceNormalization':
        Transform(
            type='mean_variance_normalization',
            defaults={
                'axes': [0, 2, 3],
            },
            inputs=(
                '!I[0]',
                '!as_tensor(1.0, np.float32, inline=True)',
                '!as_tensor(0.0, np.float32, inline=True)',
            ),
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
                'epsilon': 0.0,
            }
        ),
    'InstanceNormalization':
        Transform(
            type='mean_variance_normalization',
            defaults={
                'epsilon': 1e-5,
            },
            inputs=(
                '!I[0]',
                '!unsqueeze_vector(I[1])',
                '!unsqueeze_vector(I[2])',
            ),
            outputs='!O[0]',
            attribs={
                'axes': '!list(range(2, I[0].rank))',
                'epsilon': '!epsilon',
            }
        ),
    'Upsample':
        Transform(
            type='!"nearest_upsample" if mode == "nearest" else "multilinear_upsample"',
            defaults={
                'mode': "nearest",
                'scales': '!as_const(I[1])',
            },
            cond='!scales[0] == 1 and scales[1] == 1 and all(int(s) == s for s in scales[2:])',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'factor': '![int(s) for s in scales[2:]]',
                'method': '!"asymmetric" if mode == "linear" else None',
            }
        ),
    'Resize':
        Transform(
            type='!("nearest_downsample" if downsample else "nearest_upsample") if mode == "nearest" else'
                 ' "multilinear_upsample"',
            defaults={
                'mode': "nearest",
                'coordinate_transformation_mode': "half_pixel",
            },
            using=OrderedDict([
                ('scales', '!as_const(I[1 if len(I) == 2 else 2])'),
                ('sizes', '!as_const(I[3]) if len(I) > 3 else'
                          ' [int(I[0].shape[i] * scales[i]) for i in range(I[0].rank)]'),
                ('upsample', '!is_integer_upsample(I[0].shape, sizes)'),
                ('downsample', '!is_integer_downsample(I[0].shape, sizes)'),
            ]),
            cond='!((mode == "nearest" and (upsample or downsample)) or (mode == "linear" and upsample)) and'
                 ' sizes[0] == I[0].shape[0] and sizes[1] == I[0].shape[1] and'
                 ' (coordinate_transformation_mode == "half_pixel" or coordinate_transformation_mode == "asymmetric" or'
                 ' coordinate_transformation_mode == "align_corners")',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'factor': '!upsample_factor(I[0].shape[2:], sizes[2:]) if upsample else'
                          ' downsample_factor(I[0].shape[2:], sizes[2:])',
                'method': '!("aligned" if coordinate_transformation_mode == "align_corners" else'
                          ' "symmetric" if coordinate_transformation_mode == "half_pixel" else "asymmetric")'
                          ' if mode == "linear" else None',
            }
        ),
    'Constant':
        Transform(
            type='constant',
            outputs='!O[0]',
            attribs={
                'value': '!ensure_list(from_numpy(value)) if len(value.shape) <= 1 and int(np.prod(value.shape)) <= 10 '
                         'else value',
                'shape': '!list(value.shape)',
                'dtype': '!value.dtype',
            }
        ),
    'Gather':
        Transform(
            using={
                'index': '!ensure_scalar(as_const(I[1]))',
                'axes': '![ensure_positive(axis, I[0].rank)]',
            },
            cond='!index is not None and len(I[1].shape) == 0',
            type='slice',
            inputs='!I[0]',
            outputs='!squeeze_output(O[0], axes)',
            attribs={
                'axes': '!axes',
                'begin': '![index]',
                'end': '![index + 1]',
            },
        ),
    'Cast':
        Transform(
            cond='!O[0].dtype == I[0].dtype',
            type='copy',
            inputs='!I[0]',
            outputs='!O[0]',
        ),
})
