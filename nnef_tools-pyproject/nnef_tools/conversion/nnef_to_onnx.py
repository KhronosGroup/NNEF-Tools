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
from .converter import ConverterFromNNEF as _Converter, Transform
from ..model import Tensor, Operation
from ..utils import types
import numpy as np
from nnef.shapes import pool_shape, reduce_shape, deconv_shape


class Converter(_Converter):

    @staticmethod
    def defined_shapes():
        return {
            'lp_pool': pool_shape,
            'lp_reduce': reduce_shape,
            'mean_variance_normalization': lambda input, scale, offset, **kwargs: input,
            'lstm_step': lambda x, h, c, W, R, B: (h, c),
            'lstm_loop': lambda X, W, R, B, h, c, **kwargs: (h, c),
            'erf': lambda x: x,
            'mish': lambda x: x,
            'depth_to_space': lambda x, block_size, **kwargs: [x[0], x[1] // block_size ** 2, x[2] * block_size, x[3] * block_size],
            'space_to_depth': lambda x, block_size, **kwargs: [x[0], x[1] * block_size ** 2, x[2] // block_size, x[3] // block_size],
        }

    @staticmethod
    def decomposed_operations():
        return ['lstm_step', 'lstm_loop']

    def __init__(self, custom_transforms=None, custom_functions=None, mirror_unsupported=False):
        _Converter.__init__(self, transforms=self.merge_transforms(_Transforms, custom_transforms),
                            functions=custom_functions, mirror_unsupported=mirror_unsupported)

    def __call__(self, graph):
        self.fill_data_in_constants(graph)
        self.convert_variables_to_constants(graph)
        graph = _Converter.__call__(self, graph)
        self._fix_inline_constants(graph)
        return graph

    def _fix_inline_constants(self, graph):
        constants = 0
        for tensor in graph.tensors:
            if tensor.name is None:
                constants += 1
                tensor.name = '$' + str(constants)

    def _make_constant(self, graph, dtype, value, inline):
        return Tensor(graph, dtype=dtype, shape=self._shape_of(value), data=types.to_numpy(value, dtype=dtype))

    def _const_operation(self, output, value):
        Operation(output.graph, type='Constant', inputs=(), outputs=output,
                  attribs={'value': types.to_numpy(value, dtype=output.dtype)})

    def _transform_constant(self, tensor, func):
        if tensor.producer:
            data = func(tensor.producer.attribs['value'] if tensor.producer else tensor.data)
            tensor.shape = data.shape
            tensor.producer.attribs['value'] = data
        else:
            tensor.data = func(tensor.data)
            tensor.shape = tensor.data.shape

    def _squeeze_operation(self, input, output, axes):
        Operation(input.graph, type='Squeeze', inputs=input, outputs=output, attribs={'axes': axes})

    def _unsqueeze_operation(self, input, output, axes):
        Operation(input.graph, type='Unsqueeze', inputs=input, outputs=output, attribs={'axes': axes})

    def _interleave(self, items):
        return [item[0] for item in items] + [item[1] for item in items]

    def squeeze_input(self, tensor, axes):
        return self._pre_squeeze(tensor, axes=axes) if len(axes) else tensor

    def squeeze_output(self, tensor, axes):
        return self._post_squeeze(tensor, axes=axes) if len(axes) else tensor

    def unsqueeze_input(self, tensor, axes):
        return self._pre_unsqueeze(tensor, axes=axes) if len(axes) else tensor

    def unsqueeze_output(self, tensor, axes):
        return self._post_unsqueeze(tensor, axes=axes) if len(axes) else tensor

    def squeeze_vector(self, tensor):
        if self._is_constant(tensor) and len(self._tensor_map[tensor].consumers) == 1:
            self._transform_constant(tensor, lambda data: np.squeeze(data, 0))
            return tensor
        else:
            return self.squeeze_input(tensor, axes=[0])

    def convert_pads(self, padding, truncate=False):
        return self._interleave(padding[2:] if truncate else padding) if padding != [] else None

    def convert_auto_pad(self, padding):
        return "SAME_UPPER" if padding == [] else "NOTSET"

    def convert_output_padding(self, input_shape, filter_shape, output_shape, padding, stride, dilation, groups):
        calculated_shape = deconv_shape(input_shape, filter_shape, padding=padding, stride=stride, dilation=dilation, groups=groups)
        output_padding = [o - c for c, o in zip(calculated_shape[2:], output_shape[2:])]
        return output_padding

    def is_const(self, tensor, value=None):
        return self._is_constant(self._tensor_map[tensor]) and value is None or self.as_const(tensor) == value

    def broadcast(self, tensor, rank):
        return self.unsqueeze_input(tensor, axes=list(range(tensor.rank, rank)))


_Transforms = Converter.unpack_transforms({
    ('external', 'constant'):
        Transform(type=None),
    ('conv', 'deconv'):
        Transform(
            type=('Conv', 'ConvTranspose'),
            defaults={
                'output_shape': None,
            },
            using={
                'transposed': '!_type_ == "deconv"',
                'group': '!groups if groups != 0 else O[0].shape[1] if transposed else I[0].shape[1]',
            },
            cond={
                '!I[2].rank != 0 or (is_const(I[2]) and as_const(I[2]) == 0)': 'bias must be constant 0 or of rank 1',
            },
            inputs=(
                '!I[0]',
                '!I[1]',
                '!squeeze_vector(I[2]) if I[2].rank != 0 else None',
            ),
            outputs='!O[0]',
            attribs={
                'auto_pad': '!convert_auto_pad(padding)',
                'pads': '!convert_pads(padding)',
                'strides': '!stride',
                'dilations': '!dilation',
                'group': '!group',
                'output_shape': '!output_shape if _type_ == "deconv" and output_shape != [] and padding == [] else None',
                'output_padding': '!convert_output_padding(I[0].shape, I[1].shape, output_shape, padding=padding, '
                                  'stride=stride, dilation=dilation, groups=group) '
                                  'if _type_ == "deconv" and output_shape != [] and padding != [] else None',
                'kernel_shape': '!I[1].shape[2:]',
            }
        ),
    ('max_pool', 'avg_pool', 'lp_pool'):
        Transform(
            type=('MaxPool', 'AveragePool', 'LpPool'),
            cond={
                '!size[:2] == [1,1]': 'size must be 1 in batch and channel dimensions',
                '!stride[:2] == [1,1]': 'stride must be 1 in batch and channel dimensions',
                '!dilation[:2] == [1,1]': 'dilation must be 1 in batch and channel dimensions',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'kernel_shape': '!size[2:]',
                'auto_pad': '!convert_auto_pad(padding)',
                'pads': '!convert_pads(padding, truncate=True)',
                'strides': '!stride[2:]',
                'dilations': '!dilation[2:] if _type_ == "max_pool" and not all(d == 1 for d in dilation[2:]) else None',
                'count_include_pad': '!(1 if border == "constant" else 0) if _type_ == "avg_pool" else None',
            }
        ),
    ('min_reduce', 'max_reduce', 'mean_reduce', 'sum_reduce'):
        Transform(
            type=('ReduceMin', 'ReduceMax', 'ReduceMean', 'ReduceSum'),
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
                'keepdims': 1,
            }
        ),
    'lp_reduce':
        Transform(
            type='!"ReduceL1" if p == 1 else "ReduceL2"',
            cond={
                '!p == 1 or p == 2': 'p must be 1 or 2',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
                'keepdims': 1,
            }
        ),
    ('argmin_reduce', 'argmax_reduce'):
        Transform(
            type=('ArgMin', 'ArgMax'),
            cond={
                '!len(axes) == 1': 'axes must be of length 1',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axis': '!axes[0]',
                'keepdims': True,
            }
        ),
    'batch_normalization':
        Transform(
            type='BatchNormalization',
            inputs=(
                '!I[0]',
                '!squeeze_vector(I[4])',
                '!squeeze_vector(I[3])',
                '!squeeze_vector(I[1])',
                '!squeeze_vector(I[2])',
            ),
            outputs='!O[0]',
            attribs={
                'epsilon': '!epsilon',
                'spatial': '!0 if I[1].rank == I[0].rank else None',
            }
        ),
    ('relu', 'sigmoid', 'tanh', 'softplus', 'selu', 'not', 'copy', 'elu', 'erf', 'mish', 'abs', 'sign',
     'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
     'exp', 'log', 'neg', 'sqrt', 'ceil', 'floor', 'round'):
        Transform(
            type=('Relu', 'Sigmoid', 'Tanh', 'Softplus', 'Selu', 'Not', 'Identity', 'Elu', 'Erf', 'Mish', 'Abs', 'Sign',
                  'Sin', 'Cos', 'Tan', 'Asin', 'Acos', 'Atan', 'Sinh', 'Cosh', 'Tanh', 'Asinh', 'Acosh', 'Atanh',
                  'Exp', 'Log', 'Neg', 'Sqrt', 'Ceil', 'Floor', 'Round'),
            inputs='!I[0]',
            outputs='!O[0]',
        ),
    ('add', 'sub', 'mul', 'div', 'pow', 'min', 'max', 'and', 'or', 'eq', 'lt', 'gt', 'le', 'ge'):
        Transform(
            type=('Add', 'Sub', 'Mul', 'Div', 'Pow', 'Min', 'Max', 'And', 'Or',
                  'Equal', 'Less', 'Greater', 'LessOrEqual', 'GreaterOrEqual'),
            inputs=(
                '!broadcast(I[0], O[0].rank)',
                '!broadcast(I[1], O[0].rank)',
            ),
            outputs='!O[0]',
        ),
    'sqr':
        Transform(
            type='Mul',
            inputs=('!I[0]', '!I[0]'),
            outputs='!O[0]',
        ),
    'leaky_relu':
        Transform(
            type='LeakyRelu',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'alpha': '!alpha',
            }
        ),
    'prelu':
        Transform(
            type='PRelu',
            inputs=(
                '!I[0]',
                '!broadcast(I[1], I[0].rank)',
            ),
            outputs='!O[0]',
        ),
    'transpose':
        Transform(
            type='Transpose',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'perm': '!axes',
            }
        ),
    'reshape':
        Transform(
            type='Reshape',
            inputs=(
                '!I[0]',
                '!as_tensor(shape, np.int64)',
            ),
            outputs='!O[0]',
        ),
    'squeeze':
        Transform(
            type='Squeeze',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
            }
        ),
    'unsqueeze':
        Transform(
            type='Unsqueeze',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
            }
        ),
    'matmul':
        Transform(
            using={
                'transposed': '!transposeA or transposeB',
            },
            type="!'Gemm' if transposed else 'MatMul'",
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            attribs={
                'transA': '!int(transposeA) if transposed else None',
                'transB': '!int(transposeB) if transposed else None',
            }
        ),
    'linear':
        Transform(
            type='Gemm',
            inputs=(
                '!I[0]',
                '!I[1]',
                '!squeeze_vector(I[2])',
            ),
            outputs='!O[0]',
            attribs={
                'transA': 0,
                'transB': 1,
            }
        ),
    'local_response_normalization':
        Transform(
            type='LRN',
            cond={
                '!size[0] == 1 and all(s == 1 for s in size[2:])': 'size must be 1 in all non-channel dimensions',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'alpha': '!alpha',
                'beta': '!beta',
                'bias': '!bias',
                'size': '!size[1]',
            }
        ),
    'concat':
        Transform(
            type='Concat',
            inputs=['!I[:]'],
            outputs='!O[0]',
            attribs={
                'axis': '!axis',
            }
        ),
    'split':
        Transform(
            type='Split',
            using={
                'factor': '!I[0].shape[axis] // sum(ratios)',
            },
            inputs='!I[0]',
            outputs=['!O[:]'],
            attribs={
                'axis': '!axis',
                'split': '![r * factor for r in ratios]',
            }
        ),
    'softmax':
        Transform(
            type='Softmax',
            cond={
                '!len(axes) == 1': 'axes must be of length 1',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axis': '!axes[0]',
            }
        ),
    'add_n':
        Transform(
            type='Sum',
            inputs=['!I[:]'],
            outputs='!O[0]',
        ),
    'select':
        Transform(
            type='Where',
            inputs=(
                '!broadcast(I[0], O[0].rank)',
                '!broadcast(I[1], O[0].rank)',
                '!broadcast(I[2], O[0].rank)',
            ),
            outputs='!O[0]',
        ),
    'clamp':
        Transform(
            type='Clip',
            cond={
                '!I[1].rank == 0': 'input a must be of rank 0',
                '!I[2].rank == 0': 'input b must be of rank 0',
            },
            inputs=(
                '!I[0]',
                '!I[1]',
                '!I[2]',
            ),
            outputs='!O[0]',
        ),
    'pad':
        Transform(
            type='Pad',
            inputs=(
                '!I[0]',
                '!as_tensor(convert_pads(padding), np.int64)',
                '!as_tensor(value, np.float32)',
            ),
            outputs='!O[0]',
            attribs={
                'mode': '!"edge" if border == "replicate" else border',
            }
        ),
    'tile':
        Transform(
            type='Tile',
            inputs=(
                '!I[0]',
                '!as_tensor(repeats, np.int64)',
            ),
            outputs='!O[0]',
        ),
    'slice':
        Transform(
            type='Slice',
            inputs=(
                '!I[0]',
                '!as_tensor(begin, np.int64)',
                '!as_tensor(end, np.int64)',
                '!as_tensor(axes, np.int64)',
                '!as_tensor(stride, np.int64)',
            ),
            outputs='!O[0]',
        ),
    ('l1_normalization', 'l2_normalization'):
        Transform(
            type='LpNormalization',
            cond={
                '!len(axes) == 1': 'axes must be of length 1',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axis': '!axes[0]',
                'p': '!1 if _type_ == "l1_normalization" else 2',
            }
        ),
    'mean_variance_normalization':
        Transform(
            type='!"InstanceNormalization" if instance else "MeanVarianceNormalization"',
            using={
                'instance': '!axes == list(range(2, I[0].rank))'
                            ' and I[1].rank == 2 and I[1].shape[0] == 1'
                            ' and I[2].rank == 2 and I[2].shape[0] == 1',
            },
            cond={
                '!is_const(scale, 1.0) if not instance else True':
                    'scale must be 1 if operation does not denote instance normalization',
                '!is_const(offset, 0.0) if not instance else True':
                    'offset must be 0 if operation does not denote instance normalization',
            },
            inputs=(
                '!I[0]',
                '!squeeze_vector(I[1]) if instance else None',
                '!squeeze_vector(I[2]) if instance else None',
            ),
            outputs='!O[0]',
            attribs={
                'axes': '!axes if not instance else None',
                'epsilon': '!epsilon if instance else None',
            }
        ),
    ('nearest_upsample', 'multilinear_upsample'):
        Transform(
            type='Resize',
            using={
                'linear': '!_type_ == "multilinear_upsample"',
            },
            inputs=(
                '!I[0]',
                '!as_tensor([], np.float32)',
                '!as_tensor([1.0, 1.0] + [float(f) for f in factor], np.float32)',
            ),
            outputs='!O[0]',
            attribs={
                'mode': '!"linear" if linear else "nearest"',
                'coordinate_transformation_mode': '!("half_pixel" if method == "symmetric" else'
                                                  ' "asymmetric" if method == "asymmetric" else'
                                                  ' "align_corners") if linear else None',
            }
        ),
    'nearest_downsample':
        Transform(
            type='Resize',
            inputs=(
                '!I[0]',
                '!as_tensor([], np.float32)',
                '!as_tensor([1.0, 1.0] + [1.0 / f for f in factor], np.float32)',
            ),
            outputs='!O[0]',
            attribs={
                'mode': 'nearest',
            }
        ),
    'gather':
        Transform(
            type='Gather',
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            attribs={
                'axis': '!axis',
            },
        ),
    'cast':
        Transform(
            type='Cast',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'to': '!O[0].dtype',
            }
        ),
    'depth_to_space':
        Transform(
            type="DepthToSpace",
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'blocksize': '!block_size',
                'mode': '!"DCR" if blocks_first else "CRD"',
            },
        ),
    'space_to_depth':
        Transform(
            type="SpaceToDepth",
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'blocksize': '!block_size',
            },
        ),
})
