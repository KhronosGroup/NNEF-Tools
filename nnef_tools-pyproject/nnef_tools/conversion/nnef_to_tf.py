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
from ..model.utils import generate_op_names_from_op_type
from ..utils import types
import numpy as np
import copy


class Converter(_Converter):

    @staticmethod
    def defined_shapes():
        return {
            'relu6': lambda shape: shape,
        }

    @staticmethod
    def decomposed_operations():
        return _Converter.decomposed_operations() + ['linear']

    def __init__(self, data_format='NXC', io_transpose=False, custom_transforms=None, custom_functions=None,
                 mirror_unsupported=False):
        _Converter.__init__(self, transforms=self.merge_transforms(_Transforms, custom_transforms),
                            functions=custom_functions, mirror_unsupported=mirror_unsupported)
        self._data_format = data_format
        self._io_transpose = io_transpose

    def __call__(self, graph):
        self.convert_variables_to_constants(graph)
        graph = _Converter.__call__(self, graph)
        self._fix_output_transposes(graph)
        self._remove_unused_constants(graph)
        generate_op_names_from_op_type(graph)
        return graph

    def _global_attribs(self):
        return {'_lite_': False}

    def _prepare(self, graph):
        self._fix_inline_constants(graph)

    def _fix_inline_constants(self, graph):
        for tensor in graph.tensors:
            mapped = self._tensor_map[tensor]
            if not mapped.producer and mapped.data is not None:
                self._const_operation(tensor, tensor.data)

    def _remove_unused_constants(self, graph):
        ops = [op for op in graph.operations if op.type == 'Const' and not op.output.has_consumer]
        tensors = [op.output for op in ops]
        graph.outputs = [tensor for tensor in graph.outputs if tensor not in tensors]
        graph.remove_operations(ops, unlink=True)
        graph.remove_tensors(tensors)

    def _fix_output_transposes(self, graph):
        graph.outputs = [self.transpose_input(tensor) if self.needs_io_transpose(tensor) else
                         self.undo_transpose(tensor) for tensor in graph.outputs]

    def _const_operation(self, output, value):
        Operation(output.graph, type='Const', inputs=(), outputs=output,
                  attribs={'value': types.to_numpy(value, dtype=output.dtype), 'dtype': output.dtype})

    def _transpose_operation(self, input, output, perm):
        Operation(input.graph, type='Transpose', inputs=(input, self.as_tensor(perm, np.int32)),
                  outputs=output, attribs={'T': input.dtype})

    def _reshape_operation(self, input, output, shape):
        Operation(input.graph, type='Reshape', inputs=(input, self.as_tensor(shape, np.int32)), outputs=output,
                  attribs={'T': input.dtype})

    def _squeeze_operation(self, input, output, axes):
        Operation(input.graph, type='Squeeze', inputs=input, outputs=output,
                  attribs={'squeeze_dims': axes, 'T': input.dtype})

    def _unsqueeze_operation(self, input, output, axes):
        if len(axes) == 1:
            Operation(input.graph, type='ExpandDims', inputs=(input, self.as_tensor(axes[0], np.int32)),
                      outputs=output, attribs={'T': input.dtype})
        else:
            Operation(input.graph, type='Reshape', inputs=(input, self.as_tensor(output.shape, np.int32)),
                      outputs=output, attribs={'T': input.dtype})

    def _scale_operation(self, input, output, scalar):
        if not isinstance(scalar, Tensor):
            scalar = self.as_tensor(scalar, np.float32)

        Operation(input.graph, type='Mul', inputs=(input, scalar), outputs=output,
                  attribs={'T': input.dtype})

    def _bias_operation(self, input, output, bias):
        if not isinstance(bias, Tensor):
            bias = self.as_tensor(bias, np.float32)

        if bias.rank == 1:
            Operation(output.graph, type='BiasAdd', inputs=(input, bias), outputs=output, attribs={'T': output.dtype})
        else:
            Operation(output.graph, type='Add', inputs=(input, bias), outputs=output, attribs={'T': output.dtype})

    def _make_constant(self, graph, dtype, value, inline):
        tensor = Tensor(graph, dtype=dtype, shape=self._shape_of(value))
        self._const_operation(tensor, value)
        return tensor

    def _transform_constant(self, tensor, func):
        data = func(tensor.producer.attribs['value'])
        tensor.shape = data.shape
        tensor.producer.attribs['value'] = data

    def _is_conv_filter(self, tensor, groups):
        tensor = self._tensor_map.get(tensor)
        return tensor and len(tensor.consumers) > 0 and \
               all(op.type == 'conv' and op.inputs[1] is tensor and op.attribs['groups'] == groups
                   for op in tensor.consumers)

    def _ensure_constant_producer(self, tensor):
        if tensor.is_constant and tensor.producer is None:
            Operation(tensor.graph, type='Const', inputs=(), outputs=tensor,
                      attribs={'value': tensor.data, 'dtype': tensor.data.dtype.type})

    def _is_nxc(self, format):
        return format[0] == 'N' and format[-1] == 'C' and len(format) > 2

    def _is_xcn(self, format):
        return format[-2] == 'C' and format[-1] == 'N' and len(format) > 2

    def _is_cxn(self, format):
        return format[0] == 'C' and format[-1] == 'N' and len(format) > 2

    def needs_io_transpose(self, tensor):
        if tensor.rank <= 2:
            return False
        if isinstance(self._io_transpose, bool):
            return self._io_transpose
        else:
            return tensor.name in self._io_transpose

    def is_nxc(self):
        return self._is_nxc(self._data_format)

    def data_format(self, rank):
        X = 'W' if rank == 1 else 'HW' if rank == 2 else 'DHW' if rank == 3 else None
        return self._data_format.replace('X', X) if X else self._data_format

    def convert_padding(self, value):
        return 'SAME' if value == [] else 'VALID' if all(item == (0, 0) for item in value) else 'EXPLICIT'

    def convert_explicit_paddings(self, value):
        if value == [] or all(item == (0, 0) for item in value):
            return None
        else:
            paddings = [item for pair in value for item in pair]
            return [0, 0] + paddings + [0, 0] if self.is_nxc() else [0, 0, 0, 0] + paddings

    def convert_size(self, value):
        if isinstance(value, tuple):
            return (1,) + value + (1,) if self.is_nxc() else (1, 1) + value[2:]
        else:
            return [1] + value + [1] if self.is_nxc() else [1, 1] + value[2:]

    def transpose_input(self, tensor):
        if self.is_nxc():
            return self._pre_transpose(tensor, self.ncx_to_nxc_perm(tensor.rank)) \
                if not self.transposing(tensor) and tensor.rank > 2 else tensor
        else:
            assert not self.transposing(tensor)
            return tensor

    def transpose_output(self, tensor):
        if self.is_nxc():
            self._transposes[tensor] = self.ncx_to_nxc(tensor.shape)
        return tensor

    def transpose_filter(self, tensor, format='XCN'):
        if self._is_xcn(format):
            perm = self.ncx_to_xcn_perm(tensor.rank)
        elif self._is_nxc(format):
            perm = self.ncx_to_nxc_perm(tensor.rank)
        elif self._is_cxn(format):
            perm = self.ncx_to_cxn_perm(tensor.rank)
        else:
            assert False

        return self._pre_transpose(tensor, perm)

    def transpose_depthwise_filter(self, tensor, channels, format='XCN'):
        if self._is_xcn(format):
            perm = self.ncx_to_xcn_perm(tensor.rank)
        elif self._is_nxc(format):
            perm = self.ncx_to_nxc_perm(tensor.rank)
        elif self._is_cxn(format):
            perm = self.ncx_to_cxn_perm(tensor.rank)
        else:
            assert False

        shape = tensor.shape[2:] + (channels, tensor.shape[0] // channels)
        return self._reshape(self._pre_transpose(tensor, perm), shape)

    def transpose_like(self, tensor, reference):
        if self.transposing(reference):
            self.transpose_output(tensor)
        return tensor

    def transpose_list_like(self, items, ref):
        return self.ncx_to_nxc(items) if self.transposing(ref) else items

    def transpose_axis_like(self, axis, ref, rank=None):
        return self.axis_ncx_to_nxc(axis, rank or ref.rank) if self.transposing(ref) else axis

    def undo_transpose(self, tensor):
        perm = self.nxc_to_ncx_perm(tensor.rank)
        if perm == list(range(tensor.rank)):
            return tensor
        return self._pre_transpose(tensor, perm) if self.transposing(tensor) else tensor

    def squeeze_input(self, tensor, axes):
        return self._pre_squeeze(tensor, axes=axes)

    def squeeze_output(self, tensor, axes):
        return self._post_squeeze(tensor, axes=axes)

    def unsqueeze_input(self, tensor, axes):
        return self._pre_unsqueeze(tensor, axes=axes)

    def unsqueeze_output(self, tensor, axes):
        return self._post_unsqueeze(tensor, axes=axes)

    def squeeze_vector(self, tensor):
        if self._is_constant(tensor) and len(self._tensor_map[tensor].consumers) == 1:
            self._transform_constant(tensor, lambda data: np.squeeze(data, 0))
            return tensor
        else:
            return self.squeeze_input(tensor, axes=[0])

    def scale_output(self, output, scalar):
        input = Tensor(output.graph, dtype=output.dtype, shape=self._working_shape(output), quant=copy.deepcopy(output.quant))
        self._scale_operation(input, output, scalar)
        return input

    def bias_add(self, output, bias):
        if bias.rank == 0 and np.all(bias.data == 0):
            return output

        input = Tensor(output.graph, dtype=output.dtype, shape=self._working_shape(output), quant=copy.deepcopy(output.quant))
        self._bias_operation(input, output, bias)
        return input

    def split_sizes(self, ratios, size):
        p = size / sum(ratios)
        return [p * r for r in ratios]

    def convert_binarg(self, tensor, other):
        self._ensure_constant_producer(tensor)
        if tensor.rank == 0:
            return tensor
        needs_transpose = self.transposing(other) and not self.transposing(tensor)
        if other.rank > tensor.rank:
            if tensor.rank == 2 and tensor.shape[0] == 1 and needs_transpose:
                return self.squeeze_vector(tensor)
            tensor = self._pre_unsqueeze(tensor, axes=list(range(tensor.rank, other.rank)))
        return self.transpose_input(tensor) if needs_transpose else tensor

    def as_numpy(self, value, dtype=None):
        return types.to_numpy(value, dtype)

    def as_bits(self, items):
        bits = 0
        for idx, val in enumerate(items):
            if val:
                bits |= (1 << idx)
        return bits

    def out_of_range(self, x, limit):
        return x >= limit or x <= -limit


_Transforms = Converter.unpack_transforms({
    'external':
        Transform(
            type='Placeholder',
            using={'needs_transpose': '!needs_io_transpose(O[0])'},
            outputs='!transpose_output(O[0]) if needs_transpose else O[0]',
            attribs={
                'shape': '!tuple(ncx_to_nxc(shape) if needs_transpose else shape)',
                'dtype': '!dtype',
            }
        ),
    'constant':
        Transform(
            type='Const',
            outputs='!O[0]',
            attribs={
                'dtype': '!O[0].dtype',
                'value': '!value if isinstance(value, np.ndarray) else as_numpy(value[0] if shape == [] else value)',
            }
        ),
    'conv':
        Transform(
            type='!"Conv{n}D".format(n=I[0].rank - 2) if groups != 0 else "DepthwiseConv2dNative"',
            cond={
                '!I[0].rank == 4 or I[0].rank == 5': 'rank must be 4 or 5',
            },
            using={
                'channels': '!I[0].shape[1]',
            },
            inputs=(
                '!transpose_input(I[0])',
                '!transpose_filter(I[1]) if groups != 0 else transpose_depthwise_filter(I[1], channels)',
            ),
            outputs=(
                '!bias_add(transpose_output(O[0]), squeeze_vector(I[2]) if I[2].rank == 2 else I[2])',
            ),
            attribs={
                'padding': '!convert_padding(padding)',
                'explicit_paddings': '!convert_explicit_paddings(padding)',
                'strides': '!convert_size(stride)',
                'dilations': '!convert_size(dilation)',
                'data_format': '!data_format(I[0].rank - 2)',
                'T': '!I[0].dtype',
            }
        ),
    'deconv':
        Transform(
            type='!"Conv{n}DBackpropInput".format(n=I[0].rank - 2) if groups != 0 else "DepthwiseConv2dNativeBackpropInput"',
            cond={
                '!I[0].rank == 4 or I[0].rank == 5': 'rank must be 4 or 5',
            },
            using={
                'channels': '!O[0].shape[1]',
            },
            inputs=(
                '!as_tensor(ncx_to_nxc(output_shape) if is_nxc() else output_shape, np.int32)',
                '!transpose_filter(I[1]) if groups != 0 else transpose_depthwise_filter(I[1], channels)',
                '!transpose_input(I[0])',
            ),
            outputs=(
                '!bias_add(transpose_output(O[0]), squeeze_vector(I[2]) if I[2].rank == 2 else I[2])',
            ),
            attribs={
                'padding': '!convert_padding(padding)',
                'explicit_paddings': '!convert_explicit_paddings(padding)',
                'strides': '!convert_size(stride)',
                'dilations': '!convert_size(dilation)',
                'data_format': '!data_format(I[0].rank - 2)',
                'T': '!I[0].dtype',
            }
        ),
    ('max_pool', 'avg_pool', 'max_pool_with_index'):
        Transform(
            type=('MaxPool', 'AvgPool', 'MaxPoolWithArgmax'),
            cond={
                '!border == "ignore"': 'border must be "ignore"',
            },
            inputs=(
                '!transpose_input(I[0])',
            ),
            outputs=(
                '!transpose_output(O[0])',
                '!transpose_output(O[1]) if len(O) > 1 else None',
            ),
            attribs={
                'ksize': '!ncx_to_nxc(size) if is_nxc() else size',
                'strides': '!ncx_to_nxc(stride) if is_nxc() else stride',
                'padding': '!convert_padding(padding)',
                'explicit_paddings': '!convert_explicit_paddings(padding)',
                'data_format': '!data_format(I[0].rank - 2) if _type_ != "max_pool_with_index" else None',
                'T': '!I[0].dtype',
            }
        ),
    'box':
        Transform(
            type='AvgPool',
            using={'volume': '!int(np.prod(size))'},
            inputs=(
                '!transpose_input(I[0])',
            ),
            outputs=(
                '!scale_output(transpose_output(O[0]), volume) if not normalize else transpose_output(O[0])',
            ),
            attribs={
                'ksize': '!ncx_to_nxc(size) if is_nxc() else size',
                'strides': '!ncx_to_nxc(stride) if is_nxc() else stride',
                'padding': '!convert_padding(padding)',
                'explicit_paddings': '!convert_explicit_paddings(padding)',
                'data_format': '!data_format(I[0].rank - 2)',
                'T': '!I[0].dtype',
            }
        ),
    'reshape':
        Transform(
            type='Reshape',
            inputs=(
                '!undo_transpose(I[0])',
                '!as_tensor(fixed_batch(shape, I[0].shape[0]), np.int32)',
            ),
            outputs='!O[0]',
            attribs={
                'T': '!dtype if not _lite_ else None',
            }
        ),
    'transpose':
        Transform(
            type='Transpose',
            inputs=(
                '!I[0]',
                '!as_tensor(transpose_axis_like(axes, I[0]), np.int32)',
            ),
            outputs='!O[0]',
            attribs={
                'T': '!dtype if not _lite_ else None',
            }
        ),
    'squeeze':
        Transform(
            type='Squeeze',
            inputs='!undo_transpose(I[0])',
            outputs='!O[0]',
            attribs={
                'squeeze_dims': '!axes',
                'T': '!dtype if not _lite_ else None',
            }
        ),
    'unsqueeze':
        Transform(
            type='!"ExpandDims" if len(axes) == 1 else "Reshape"',
            using={
                'new_shape': '!unsqueeze_shape(I[0].shape, axes)',
            },
            inputs=(
                '!undo_transpose(I[0])',
                '!as_tensor(axes if len(axes) == 1 else new_shape, np.int32)',
            ),
            outputs='!O[0]',
            attribs={
                'T': '!dtype if not _lite_ else None',
                'new_shape': '!new_shape if _lite_ else None',
            }
        ),
    'stack':
        Transform(
            type='Pack',
            inputs=['![undo_transpose(t) for t in I]'],
            outputs='!O[0]',
            attribs={
                'axis': '!axis',
                'N': '!len(I) if not _lite_ else None',
                'values_count': '!len(I) if _lite_ else None',
                'T': '!dtype if not _lite_ else None',
            }
        ),
    'unstack':
        Transform(
            type='Unpack',
            inputs='!undo_transpose(I[0])',
            outputs=['!O[:]'],
            attribs={
                'axis': '!axis',
                'num': '!len(O)',
                'T': '!dtype if not _lite_ else None',
            }
        ),
    ('min_reduce', 'max_reduce', 'mean_reduce', 'sum_reduce', 'any_reduce', 'all_reduce'):
        Transform(
            type=('Min', 'Max', 'Mean', 'Sum', 'Any', 'All'),
            using={'dims': '!transpose_axis_like(axes, I[0])'},
            inputs=(
                '!I[0]',
                '!as_tensor(dims, np.int32)',
            ),
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'keep_dims': True,
                'T': '!I[0].dtype if I[0].dtype != bool and not _lite_ else None',
            }
        ),
    'concat':
        Transform(
            type='Concat',
            using={
                'dim': '!transpose_axis_like(axis, I[0])'
            },
            inputs=['!as_tensor(dim, np.int32)', '!I[:]'],
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'N': '!len(I)',
                'T': '!O[0].dtype if not _lite_ else None',
            }
        ),
    'split':
        Transform(
            type='SplitV',
            using={
                'dim': '!transpose_axis_like(axis, I[0])'
            },
            inputs=(
                '!I[0]',
                '!as_tensor(split_sizes(ratios, I[0].shape[axis]), np.int64)',
                '!as_tensor(dim, np.int32)',
            ),
            outputs=['![transpose_like(O[i], I[0]) for i in range(len(O))]'],
            attribs={
                'num_split': '!len(ratios) if not _lite_ else None',
                'num_splits': '!len(ratios) if _lite_ else None',
                'T': '!I[0].dtype if not _lite_ else None',
            }
        ),
    ('add', 'sub', 'mul', 'div', 'pow', 'lt', 'gt', 'le', 'ge', 'eq', 'ne', 'min', 'max', 'and', 'or'):
        Transform(
            type=('Add', 'Sub', 'Mul', 'RealDiv', 'Pow', 'Less', 'Greater', 'LessEqual', 'GreaterEqual',
                  'Equal', 'NotEqual', 'Minimum', 'Maximum', 'LogicalAnd', 'LogicalOr'),
            inputs=(
                '!convert_binarg(I[0], I[1])',
                '!convert_binarg(I[1], I[0])',
            ),
            outputs='!transpose_output(O[0]) if transposing(I[0]) or transposing(I[1]) else O[0]',
            attribs={
                'T': '!I[0].dtype if I[0].dtype != bool and not _lite_ else None',
            }
        ),
    ('copy', 'relu', 'relu6', 'elu', 'selu', 'gelu', 'silu', 'sigmoid', 'softplus', 'exp', 'log',
     'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
     'neg', 'rcp', 'sign', 'abs', 'floor', 'ceil', 'round', 'sqr', 'sqrt', 'rsqrt', 'not'):
        Transform(
            type=('Identity', 'Relu', 'Relu6', 'Elu', 'Selu', 'Gelu', 'Silu', 'Sigmoid', 'Softplus', 'Exp', 'Log',
                  'Sin', 'Cos', 'Tan', 'Asin', 'Acos', 'Atan', 'Sinh', 'Cosh', 'Tanh', 'Asinh', 'Acosh', 'Atanh',
                  'Neg', 'Reciprocal', 'Sign', 'Abs', 'Floor', 'Ceil', 'Round', 'Square', 'Sqrt', 'Rsqrt', 'LogicalNot'),
            inputs='!I[0]',
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'T': '!I[0].dtype if I[0].dtype != bool and not _lite_ else None',
            }
        ),
    'leaky_relu':
        Transform(
            type='LeakyRelu',
            inputs='!I[0]',
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'alpha': '!alpha',
                'T': '!I[0].dtype if not _lite_ else None',
            }
        ),
    'batch_normalization':
        Transform(
            type='FusedBatchNorm',
            using={
                'channels': '!O[0].shape[1]'
            },
            inputs=(
                '!transpose_input(I[0])',
                '!squeeze_vector(I[4])',
                '!squeeze_vector(I[3])',
                '!squeeze_vector(I[1])',
                '!squeeze_vector(I[2])',
            ),
            outputs=(
                '!transpose_output(O[0])',
                '!new_tensor(shape=(channels,), dtype=O[0].dtype)',
                '!new_tensor(shape=(channels,), dtype=O[0].dtype)',
                '!new_tensor(shape=(channels,), dtype=O[0].dtype)',
                '!new_tensor(shape=(channels,), dtype=O[0].dtype)',
            ),
            attribs={
                'epsilon': '!epsilon',
                'data_format': '!data_format(I[0].rank - 2)',
                'T': '!I[0].dtype if not _lite_ else None',
                'is_training': False,
            }
        ),
    'softmax':
        Transform(
            type='Softmax',
            cond={
                '!axes == [1])': 'axes must equal channel dimension (1)',
            },
            inputs='!transpose_input(I[0])',
            outputs='!transpose_output(O[0])',
            attribs={
                'T': '!I[0].dtype if not _lite_ else None',
                'beta': '!1.0 if _lite_ else None',
            }
        ),
    'matmul':
        Transform(
            type='MatMul',
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            attribs={
                'transpose_a': '!transposeA',
                'transpose_b': '!transposeB',
                'T': '!I[0].dtype if not _lite_ else None',
            },
        ),
    'clamp':
        Transform(
            type='ClipByValue',
            inputs=('!I[0]', '!I[1]', '!I[2]'),
            outputs='!transpose_like(O[0], I[0])',
            attribs={'T': '!I[0].dtype if not _lite_ else None'},
        ),
    'pad':
        Transform(
            type='!"Pad" if border == "constant" else "MirrorPad"',
            cond={
                '!border in ["constant", "reflect", "reflect-even"]':
                    'border must be one of "constant", "reflect", "reflect-even"',
            },
            using={'paddings': '![list(item) for item in padding]'},
            inputs=(
                '!I[0]',
                '!as_tensor(ncx_to_nxc(paddings, cond=transposing(I[0])), np.int32)',
            ),
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'T': '!I[0].dtype if not _lite_ else None',
                'mode': '!"REFLECT" if border == "reflect" else "SYMMETRIC" if border == "reflect-even" else None',
            },
        ),
    'tile':
        Transform(
            type='Tile',
            inputs=(
                '!I[0]',
                '!as_tensor(transpose_list_like(repeats, I[0]), np.int32)',
            ),
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'T': '!I[0].dtype if not _lite_ else None',
            },
        ),
    'slice':
        Transform(
            type='StridedSlice',
            using={
                'dims': '!ncx_to_nxc(list(range(I[0].rank)), cond=transposing(I[0]))',
                'axis': '!ncx_to_nxc(axes, cond=transposing(I[0]))',
                'begs': '!ncx_to_nxc(begin, cond=transposing(I[0]))',
                'ends': '!ncx_to_nxc(end, cond=transposing(I[0]))',
                'strs': '!ncx_to_nxc(stride, cond=transposing(I[0]))',
            },
            inputs=(
                '!I[0]',
                '!as_tensor([begs[axis.index(i)] if i in axis else 0 for i in dims], np.int32)',
                '!as_tensor([ends[axis.index(i)] if i in axis else 0 for i in dims], np.int32)',
                '!as_tensor([strs[axis.index(i)] if i in axis else 1 for i in dims], np.int32)',
            ),
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'T': '!I[0].dtype if not _lite_ else None',
                'Index': '!np.int32',
                'begin_mask': '!as_bits([1 if i not in axis or out_of_range(begs[axis.index(i)], I[0].shape[i]) else 0'
                              ' for i in dims])',
                'end_mask': '!as_bits([1 if i not in axis or (ends[axis.index(i)] == 0 and strs[axis.index(i)] == 1)'
                            ' or out_of_range(ends[axis.index(i)], I[0].shape[i]) else 0 for i in dims])',
                'ellipsis_mask': 0,
                'new_axis_mask': 0,
                'shrink_axis_mask': 0,
            },
        ),
    ('argmin_reduce', 'argmax_reduce'):
        Transform(
            type=('ArgMin', 'ArgMax'),
            cond={
                '!len(axes) == 1': 'axes must be of length 1',
            },
            using={'axis': '!transpose_axis_like(axes[0], ref=I[0])'},
            inputs=(
                '!I[0]',
                '!as_tensor(axis, np.int32)',
            ),
            outputs='!transpose_like(unsqueeze_output(O[0], axes) if not _lite_ else O[0], I[0])',
            attribs={
                'T': '!I[0].dtype if not _lite_ else None',
                'output_type': '!O[0].dtype',
            }
        ),
    'select':
        Transform(
            type='Select',
            inputs=(
                '!I[0]',
                '!convert_binarg(I[1], I[0])',
                '!convert_binarg(I[2], I[0])',
            ),
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'T': '!I[1].dtype if not _lite_ else None',
            }
        ),
    'nearest_upsample':
        Transform(
            type='ResizeNearestNeighbor',
            using={
                'size': '!I[0].shape[2:]'
            },
            inputs=(
                '!transpose_input(I[0])',
                '!as_tensor([s * f for s, f in zip(size, factor)], np.int32)',
            ),
            outputs='!transpose_output(O[0])',
            attribs={
                'T': '!I[0].dtype if not _lite_ else None',
                'align_corners': False,
                'half_pixel_centers': False,
            }
        ),
    'multilinear_upsample':
        Transform(
            type='ResizeBilinear',
            using={
                'size': '!I[0].shape[2:]'
            },
            inputs=(
                '!transpose_input(I[0])',
                '!as_tensor([s * f for s, f in zip(size, factor)], np.int32)',
            ),
            outputs='!transpose_output(O[0])',
            attribs={
                'T': '!I[0].dtype if not _lite_ else None',
                'align_corners': '!method == "aligned"',
                'half_pixel_centers': '!method == "symmetric"',
            }
        ),
    ('nearest_downsample', 'area_downsample'):
        Transform(
            type=('ResizeNearestNeighbor', 'ResizeArea'),
            using={'size': '!I[0].shape[2:]'},
            inputs=(
                '!transpose_input(I[0])',
                '!as_tensor([s // f for s, f in zip(size, factor)], np.int32)',
            ),
            outputs='!transpose_output(O[0])',
            attribs={
                'T': '!I[0].dtype if not _lite_ else None',
                'align_corners': False,
                'half_pixel_centers': '!False if _type_ == "nearest_downsample" else None',
            }
        ),
    'local_response_normalization':
        Transform(
            type='LRN',
            cond={
                '!all(s == 1 or i == 1 for i, s in enumerate(size))':
                    'size must be singular for all non-channel dimensions',
            },
            inputs='!I[0]',
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'depth_radius': '!size[1] // 2 if not _lite_ else None',
                'radius': '!size[1] // 2 if _lite_ else None',
                'alpha': '!alpha / size[1]',
                'beta': '!beta',
                'bias': '!bias',
            }
        ),
    'add_n':
        Transform(
            type='AddN',
            inputs=['!I[:]'],
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'T': '!O[0].dtype',
                'N': '!len(I)',
            },
        ),
    'cast':
        Transform(
            type='Cast',
            inputs='!I[0]',
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'SrcT': '!I[0].dtype',
                'DstT': '!O[0].dtype',
            },
        ),
    'gather':
        Transform(
            type='GatherV2',
            inputs=(
                '!I[0]',
                '!I[1]',
                '!as_tensor(transpose_axis_like(axis, I[0]), np.int32)',
            ),
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'Tparams': '!I[0].dtype',
                'Tindices': '!I[1].dtype',
                'Taxis': np.int32,
            },
        ),
})
