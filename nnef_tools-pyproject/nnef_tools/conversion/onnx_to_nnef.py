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
from .converter import ConverterToNNEF as _Converter, Transform, ConversionError
from ..model.utils import generate_tensor_names_from_op_type
from ..model import Tensor
from ..utils import types
from collections import OrderedDict
import numpy as np
import copy
from nnef.shapes import pool_shape, reduce_shape


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

_LSTM_STEP_FRAGMENT = """
fragment lstm_step(
    x: tensor<scalar>,
    h: tensor<scalar>,
    c: tensor<scalar>,
    W: tensor<scalar>,
    R: tensor<scalar>,
    B: tensor<scalar> )
-> ( h_out: tensor<scalar>,
    c_out: tensor<scalar> )
{
    [Wb, Rb] = split(B, axis = 1, ratios = [1, 1]);
    z = linear(x, W, Wb) + linear(h, R, Rb);
    [i, f, g, o] = split(z, axis = 1, ratios=[1, 1, 1, 1]);
    c_out = sigmoid(f) * c + sigmoid(i) * tanh(g);
    h_out = sigmoid(o) * tanh(c_out);
}
"""

_LSTM_LOOP_FRAGMENT = """
fragment lstm_loop(
    X: tensor<scalar>,
    W: tensor<scalar>,
    R: tensor<scalar>,
    B: tensor<scalar>,
    h0: tensor<scalar>,
    c0: tensor<scalar>,
    steps: integer,
    index: integer = 0,
    axis: integer = 0 )
-> ( hn: tensor<scalar>, cn: tensor<scalar> )
{
    x0 = squeeze(slice(X, axes = [axis], begin = [index], end = [index + 1]), axes = [axis]);
    h1, c1 = lstm_step(x0, h0, c0, W, R, B);
    hn, cn = lstm_loop(X, W, R, B, h1, c1, index = index + 1, steps=steps) if index + 1 < steps else (h1, c1);
}
"""

_ERF_FRAGMENT = """
fragment erf( x: tensor<scalar> ) -> ( y: tensor<scalar> )
{
    t = 1.0 / (1.0 + 0.3275911 * abs(x));
    z = 1.0 - (((((1.061405429 * t + -1.453152027) * t) + 1.421413741) * t + -0.284496736) * t + 0.254829592) * t * exp(-x * x);
    y = sign(x) * z;
}
"""

_MISH_FRAGMENT = """
fragment mish( x: tensor<scalar> ) -> ( y: tensor<scalar> )
{
    y = x * tanh(log(1.0 + exp(x)));
}
"""

_DEPTH_TO_SPACE_FRAGMENT = """
fragment depth_to_space( x: tensor<scalar>, block_size: integer, blocks_first: logical ) -> ( y: tensor<scalar> )
{
    r = reshape(x, axis_start=1, axis_count=1, shape=[block_size, block_size, -1] 
                                if blocks_first else [-1, block_size, block_size]);
    t = transpose(r, axes=[0, 3, 4, 1, 5, 2] if blocks_first else [0, 1, 4, 2, 5, 3]);
    q = reshape(t, axis_start=4, axis_count=2, shape=[-1]);
    y = reshape(q, axis_start=2, axis_count=2, shape=[-1]);
}
"""

_SPACE_TO_DEPTH_FRAGMENT = """
fragment space_to_depth( x: tensor<scalar>, block_size: integer, blocks_first: logical ) -> ( y: tensor<scalar> )
{
    p = reshape(x, axis_start=3, axis_count=1, shape=[-1, block_size]);
    r = reshape(p, axis_start=2, axis_count=1, shape=[-1, block_size]);
    t = transpose(r, axes=[0, 3, 5, 1, 2, 4] if blocks_first else [0, 1, 3, 5, 2, 4]);
    y = reshape(t, axis_start=1, axis_count=1, shape=[-1]);
}
"""

_INT_MAX = 2 ** 31 - 1


class Converter(_Converter):

    @staticmethod
    def defined_operations():
        return {
            'lp_pool': _LP_POOL_FRAGMENT,
            'lp_reduce': _LP_REDUCE_FRAGMENT,
            'mean_variance_normalization': _MEAN_VARIANCE_NORMALIZATION_FRAGMENT,
            'lstm_step': _LSTM_STEP_FRAGMENT,
            'lstm_loop': _LSTM_LOOP_FRAGMENT,
            'erf': _ERF_FRAGMENT,
            'mish': _MISH_FRAGMENT,
            'depth_to_space': _DEPTH_TO_SPACE_FRAGMENT,
            'space_to_depth': _SPACE_TO_DEPTH_FRAGMENT,
        }

    @staticmethod
    def defined_operation_dependencies():
        return {
            'lstm_loop': ['lstm_step'],
        }

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

    def __init__(self, custom_transforms=None, custom_functions=None, mirror_unsupported=False, keep_io_names=False,
                 infer_shapes=False, custom_shapes=None, io_transpose=False):
        _Converter.__init__(self, transforms=self.merge_transforms(_Transforms, custom_transforms),
                            functions=custom_functions,
                            mirror_unsupported=mirror_unsupported,
                            infer_shapes=infer_shapes,
                            custom_shapes=dict(**self.defined_shapes(), **custom_shapes or {}))
        self._keep_io_names = keep_io_names
        self._io_transpose = io_transpose

    def __call__(self, graph):
        graph = _Converter.__call__(self, graph)
        self.remove_unused_constants(graph)
        self.inline_scalar_constants(graph)
        self.convert_constants_to_variables(graph)
        self._ensure_valid_ids(graph)
        if self._io_transpose is not False:
            self._transpose_inputs(graph)
            self._transpose_outputs(graph)
            graph.sort()
        generate_tensor_names_from_op_type(graph, keep_io_names=self._keep_io_names)
        return graph

    def _prepare(self, graph):
        self._insert_externals_and_constants(graph)

    def _is_constant(self, tensor):
        if tensor.producer:
            return tensor.producer.type == 'Constant'
        else:
            return tensor.data is not None

    def _read_constant(self, tensor, type=None):
        if tensor.producer and tensor.producer.type == 'Constant':
            value = tensor.producer.attribs['value']
        elif not tensor.producer:
            value = tensor.data
        else:
            raise ConversionError('trying to evaluate non-constant tensor')

        return types.from_numpy(value, type=type) if isinstance(value, np.ndarray) else types.cast(value, type=type)

    def _needs_io_transpose(self, tensor):
        if tensor.rank <= 2:
            return False
        if isinstance(self._io_transpose, bool):
            return self._io_transpose
        else:
            return tensor.name in self._io_transpose

    def _transpose_inputs(self, graph):
        inputs = [self._transpose_input(tensor) if self._needs_io_transpose(tensor) else tensor
                  for tensor in graph.inputs]

        if self._keep_io_names:
            for i in range(len(inputs)):
                if inputs[i] is not graph.inputs[i]:
                    inputs[i].name = graph.inputs[i].name

        graph.inputs = inputs

    def _transpose_outputs(self, graph):
        outputs = [self._transpose_output(tensor) if self._needs_io_transpose(tensor) else tensor
                   for tensor in graph.outputs]

        if self._keep_io_names:
            for i in range(len(outputs)):
                if outputs[i] is not graph.outputs[i]:
                    outputs[i].name = graph.outputs[i].name

        graph.outputs = outputs

    def _transpose_input(self, tensor):
        external = tensor.producer
        external.outputs = self._post_transpose(tensor, self.ncx_to_nxc_perm(tensor.rank))
        external.attribs['shape'] = list(self.nxc_to_ncx(tensor.shape))
        return external.output

    def _transpose_output(self, tensor):
        return self._pre_transpose(tensor, self.nxc_to_ncx_perm(tensor.rank))

    @staticmethod
    def _interleave(items):
        return [item[0] for item in items] + [item[1] for item in items]

    @staticmethod
    def _uninterleave(items):
        count = len(items) // 2
        return list(zip(items[:count], items[count:]))

    def convert_padding(self, pads, auto_pad, output_padding, rank, ceil_stride=None):
        if auto_pad == "NOTSET" or auto_pad == "SAME_LOWER":
            padding = self._uninterleave(pads)
            if output_padding is not None:
                for i in range(len(padding)):
                    padding[i] = (padding[i][0], padding[i][1] - output_padding[i])
            padding = [(0, 0)] * (rank - len(padding)) + padding
            return self.ceil_pads(padding, ceil_stride) if ceil_stride else padding
        elif auto_pad == "VALID":
            padding = [(0, 0,)] * rank
            if output_padding is not None:
                offs = rank - len(output_padding)
                for i in range(len(output_padding)):
                    padding[i + offs] = (padding[i + offs][0], padding[i + offs][1] - output_padding[i])
            return self.ceil_pads(padding, ceil_stride) if ceil_stride else padding
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
        original = self._tensor_map[tensor]
        if self._is_constant(original) and len(original.consumers) == 1:
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

    def ceil_pads(self, pads, stride):
        return [(p, q + s - 1) for (p, q), s in zip(pads, stride)]

    def broadcast(self, tensor, rank):
        return self.unsqueeze_input(tensor, axes=list(range(rank - tensor.rank))) if tensor.rank > 0 else tensor

    def ensure_list(self, arg):
        return [arg] if not isinstance(arg, list) else arg

    def ensure_scalar(self, arg):
        return arg[0] if isinstance(arg, list) and len(arg) == 1 else arg

    def limit_range(self, x):
        return _INT_MAX if x > _INT_MAX else -_INT_MAX if x < -_INT_MAX else x

    def is_unused(self, tensor):
        if len(tensor.name) == 0:
            return True
        original = self._tensor_map[tensor]
        return len(original.consumers) == 0


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
                'output_padding': None,
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
                'padding': '!convert_padding(_pads, auto_pad, output_padding, I[0].rank - 2)',
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
            cond={
                '!storage_order == 0': 'storage_order must be 0',
            },
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
                'padding': '!convert_padding(_pads, auto_pad, None, I[0].rank, [1, 1] + strides if ceil_mode == 1 else None)',
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
            using={
                'axes': '![ensure_positive(axis, I[0].rank)]',
            },
            cond={
                '!select_last_index == 0': 'select_last_index must be 0',
            },
            inputs='!I[0]',
            outputs='!squeeze_output(O[0], axes, keepdims)',
            attribs={
                'axes': '!axes',
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
    ('Relu', 'Sigmoid', 'Tanh', 'Softplus', 'Selu', 'Not', 'Identity', 'Elu', 'Erf', 'Mish', 'Abs', 'Sign',
     'Sin', 'Cos', 'Tan', 'Asin', 'Acos', 'Atan', 'Sinh', 'Cosh', 'Tanh', 'Asinh', 'Acosh', 'Atanh',
     'Exp', 'Log', 'Neg', 'Sqrt', 'Ceil', 'Floor', 'Round'):
        Transform(
            type=('relu', 'sigmoid', 'tanh', 'softplus', 'selu', 'not', 'copy', 'elu', 'erf', 'mish', 'abs', 'sign',
                  'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
                  'exp', 'log', 'neg', 'sqrt', 'ceil', 'floor', 'round'),
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
                'axes': '!ensure_positive(perm, I[0].rank)',
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
            defaults={
                'axes': '!as_const(I[1])',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!ensure_positive(axes, I[0].rank)',
            }
        ),
    'Unsqueeze':
        Transform(
            type='unsqueeze',
            defaults={
                'axes': '!as_const(I[1])',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!ensure_positive(axes, O[0].rank)',
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
            cond={
                '!alpha == 1.0': 'alpha must be 1',
                '!beta == 1.0 or len(I) == 2': 'beta must be 1',
            },
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
                'axis': '!ensure_positive(axis, O[0].rank)',
            }
        ),
    'Split':
        Transform(
            type='split',
            defaults={
                'axis': 0,
                'split': '!as_const(I[1])',
            },
            inputs='!I[0]',
            outputs=['!O[:]'],
            attribs={
                'axis': '!ensure_positive(axis, I[0].rank)',
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
                'axes': '![ensure_positive(axis, I[0].rank)]',
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
            type='!"max" if I[2].name == "" else "min" if I[1].name == "" else "clamp"',
            inputs=(
                '!I[0]',
                '!I[1] if I[1].name != "" else None',
                '!I[2] if I[2].name != "" else None',
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
    'Expand':
        Transform(
            type='tile',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'repeats': '![O[0].shape[i] // I[0].shape[i] for i in range(I[0].rank)]',
            }
        ),
    'Slice':
        Transform(
            type='slice',
            using={
                'axes': '!as_const(I[3]) if len(I) > 3 else list(range(I[0].rank))',
                'starts': '![limit_range(x) for x in as_const(I[1])]',
                'ends': '![limit_range(x) for x in as_const(I[2])]',
                'steps': '!as_const(I[4]) if len(I) > 4 else None',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!ensure_positive(axes, I[0].rank)',
                'begin': '!starts',
                'end': '!ends',
                'stride': '!steps',
            }
        ),
    'LpNormalization':
        Transform(
            type='!"l1_normalization" if p == 1 else "l2_normalization"',
            defaults={
                'axis': -1,
                'p': 2,
            },
            cond={
                '!p == 1 or p == 2': 'p must be 1 or 2',
            },
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
            cond={
                '!scales[0] == 1 and scales[1] == 1': 'scales must be 1 in batch and channel dimensions',
                '!all(int(s) == s for s in scales[2:])': 'scales must be integers in all dimensions',
            },
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
            cond={
                '!mode == "nearest" or mode == "linear"':
                    'mode must be one of "nearest", "linear"',
                '!upsample or downsample if mode == "nearest" else True':
                    "nearest resize must be integer up-sample or down-sample",
                '!upsample if mode == "linear" else True':
                    'linear resize must be integer up-sample',
                '!sizes[0] == I[0].shape[0] and sizes[1] == I[0].shape[1]':
                    'batch and channel dimensions must be preserved',
                '!coordinate_transformation_mode == "half_pixel" or'
                ' coordinate_transformation_mode == "pytorch_half_pixel" or'
                ' coordinate_transformation_mode == "asymmetric" or'
                ' coordinate_transformation_mode == "align_corners"':
                    'coordinate_transformation_mode must be one of'
                    ' "half_pixel", "pytorch_half_pixel", "asymmetric", "align_corners"',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'factor': '!upsample_factor(I[0].shape[2:], sizes[2:]) if upsample else'
                          ' downsample_factor(I[0].shape[2:], sizes[2:])',
                'method': '!("aligned" if coordinate_transformation_mode == "align_corners" else'
                          ' "symmetric" if coordinate_transformation_mode == "half_pixel" or '
                          ' coordinate_transformation_mode == "pytorch_half_pixel" else'
                          ' "asymmetric") if mode == "linear" else None',
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
            type='gather',
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            defaults={
                'axis': 0,
            },
            attribs={
                'axis': '!ensure_positive(axis, I[0].rank)',
            },
        ),
    'Cast':
        Transform(
            using={
                'same_type': '!nnef_dtype(O[0].dtype) == nnef_dtype(I[0].dtype)',
            },
            type='!"copy" if same_type else "cast"',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'dtype': '!O[0].dtype if not same_type else None',
            },
        ),
    'LSTM':
        Transform(
            cond={
                '!direction == "forward"': 'direction must be "forward"',
                '!is_unused(O[0])': 'first output must not have consumer operations',
                '!len(I[4].name) == 0': 'sequence_lens must not be defined',
            },
            defaults={
                'layout': 0,
            },
            using={
                'seq_axis': '!0 if layout == 0 else 1',
                'dir_axis': '!0 if layout == 0 else 2',
            },
            type='lstm_loop',
            inputs=(
                '!I[0]',                                    # X
                '!squeeze_input(I[1], axes=[0])',           # W
                '!squeeze_input(I[2], axes=[0])',           # R
                '!I[3]',                                    # B
                '!squeeze_input(I[5], axes=[dir_axis])',    # h_0
                '!squeeze_input(I[6], axes=[dir_axis])',    # c_0
            ),
            outputs=(
                '!unsqueeze_output(O[1], axes=[dir_axis])',    # h_n
                '!unsqueeze_output(O[2], axes=[dir_axis])',    # c_n
            ),
            attribs={
                'steps': '!I[0].shape[seq_axis]',
                'axis': '!seq_axis',
            },
        ),
    'DepthToSpace':
        Transform(
            type="depth_to_space",
            defaults={
                'mode': "DCR",
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'block_size': '!blocksize',
                'blocks_first': '!mode == "DCR"',
            },
        ),
    'SpaceToDepth':
        Transform(
            type="space_to_depth",
            defaults={
                'mode': "DCR",
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'block_size': '!blocksize',
                'blocks_first': '!mode == "DCR"',
            },
        ),
})
