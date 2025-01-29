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
from ..utils import types
from collections import OrderedDict
import numpy as np


_RELU6_FRAGMENT = """
fragment relu6( input: tensor<scalar> ) -> ( output: tensor<scalar> )
{
    output = clamp(input, 0.0, 6.0);
}
"""

_INT_MAX = 2 ** 31 - 1


class Converter(_Converter):

    _ConvOpTypes = ['Conv1D', 'Conv2D', 'Conv3D', 'Conv2DBackpropInput', 'Conv1DBackpropInput', 'Conv3DBackpropInput']
    _DepthwiseConvOpTypes = ['DepthwiseConv2dNative', 'DepthwiseConv2dNativeBackpropInput']

    @staticmethod
    def defined_operations():
        return {
            'relu6': _RELU6_FRAGMENT,
        }

    def __init__(self, io_transpose=False, custom_transforms=None, custom_functions=None,
                 mirror_unsupported=False, keep_io_names=False):
        _Converter.__init__(self, transforms=self.merge_transforms(_Transforms, custom_transforms),
                            functions=custom_functions, mirror_unsupported=mirror_unsupported)
        self._io_transpose = io_transpose
        self._keep_io_names = keep_io_names

    def __call__(self, graph):
        graph = _Converter.__call__(self, graph)
        self.remove_unused_constants(graph)
        self.inline_scalar_constants(graph)
        self.convert_constants_to_variables(graph)
        self._fix_output_transposes(graph)
        self._ensure_valid_ids(graph)
        generate_tensor_names_from_op_type(graph, keep_io_names=self._keep_io_names)
        return graph

    def _global_attribs(self):
        return {'_lite_': False}

    def _fix_output_transposes(self, graph):
        outputs = [self.transpose_input(tensor) if self.needs_io_transpose(tensor) else
                   self.undo_transpose(tensor) for tensor in graph.outputs]

        if self._keep_io_names:
            for i in range(len(outputs)):
                if outputs[i] is not graph.outputs[i]:
                    outputs[i].name = graph.outputs[i].name

        graph.outputs = outputs

    def _is_conv_filter(self, tensor):
        tensor = self._tensor_map.get(tensor)
        return tensor and len(tensor.consumers) > 0 and \
               all(op.type in Converter._ConvOpTypes and op.inputs[1] is tensor for op in tensor.consumers)

    def _is_depthwise_conv_filter(self, tensor):
        tensor = self._tensor_map.get(tensor)
        return tensor and len(tensor.consumers) > 0 and \
               all(op.type in Converter._DepthwiseConvOpTypes and op.inputs[1] is tensor for op in tensor.consumers)

    def _is_constant(self, tensor):
        if tensor.producer:
            return tensor.producer.type == 'Const'
        else:
            return tensor.data is not None

    def _read_constant(self, tensor, type=None):
        if tensor.producer is None:
            return types.from_numpy(tensor.data, type=type)
        elif tensor.producer.type == 'Const':
            value = tensor.producer.attribs['value']
            return types.from_numpy(value, type=type) if isinstance(value, np.ndarray) else types.cast(value, type=type)
        else:
            raise ConversionError('trying to evaluate non-constant tensor')

    def needs_io_transpose(self, tensor):
        if tensor.rank <= 2:
            return False
        if isinstance(self._io_transpose, bool):
            return self._io_transpose
        else:
            return tensor.name in self._io_transpose

    def is_nxc(self, format):
        return format[0] == 'N' and format[-1] == 'C' and len(format) > 2

    def is_cxn(self, format):
        return format[0] == 'C' and format[-1] == 'N' and len(format) > 2

    def is_xcn(self, format):
        return format[-2] == 'C' and format[-1] == 'N' and len(format) > 2

    def transpose_input(self, tensor, format='NXC'):
        if self.is_nxc(format):
            return self._pre_transpose(tensor, self.nxc_to_ncx_perm(tensor.rank)) \
                if not self.transposing(tensor) and tensor.rank > 2 else tensor
        else:
            assert not self.transposing(tensor)
            return tensor

    def transpose_output(self, tensor, format='NXC'):
        if self.is_nxc(format):
            self._transposes[tensor] = self.nxc_to_ncx(tensor.shape)
        return tensor

    def transpose_filter(self, tensor, format='XCN'):
        if self.is_xcn(format):
            perm = self.xcn_to_ncx_perm(tensor.rank)
        elif self.is_nxc(format):
            perm = self.nxc_to_ncx_perm(tensor.rank)
        elif self.is_cxn(format):
            perm = self.cxn_to_ncx_perm(tensor.rank)
        else:
            assert False

        return self._pre_transpose(tensor, perm)

    def transpose_depthwise_filter(self, tensor, format='XCN'):
        if self.is_xcn(format):
            perm = self.xcn_to_ncx_perm(tensor.rank)
        elif self.is_nxc(format):
            perm = self.nxc_to_ncx_perm(tensor.rank)
        elif self.is_cxn(format):
            perm = self.cxn_to_ncx_perm(tensor.rank)
        else:
            assert False

        shape = tensor.shape[:-2] + (1, -1)
        return self._pre_transpose(self._reshape(tensor, shape), perm)

    def transpose_like(self, tensor, ref):
        if ref is not None and self.transposing(ref):
            self.transpose_output(tensor)
        return tensor

    def undo_transpose(self, tensor):
        perm = self.ncx_to_nxc_perm(tensor.rank)
        if perm == list(range(tensor.rank)):
            return tensor
        return self._pre_transpose(tensor, perm) if self.transposing(tensor) else tensor

    def convert_size(self, value, format):
        return value[1:-1] if self.is_nxc(format) else value[2:]

    def convert_padding(self, padding, rank, explicit_paddings=None, format=None):
        padding = padding.upper()
        if padding == 'SAME':
            return []
        elif padding == 'VALID':
            return [(0, 0)] * rank
        elif padding == 'EXPLICIT':
            assert explicit_paddings is not None and format is not None
            explicit_paddings = list(zip(explicit_paddings[0::2], explicit_paddings[1::2]))
            return explicit_paddings[1:-1] if self.is_nxc(format) else explicit_paddings[2:]
        else:
            assert False, "unknown padding type '{}'".format(padding)

    def transpose_list_like(self, items, ref):
        return self.nxc_to_ncx(items) if ref is not None and self.transposing(ref) else items

    def transpose_axis_like(self, axis, ref, rank=None):
        return self.axis_nxc_to_ncx(axis, rank or ref.rank) if ref is not None and self.transposing(ref) else \
            self.ensure_positive(axis, rank or ref.rank)

    def squeeze_input(self, tensor, axes, keep_dims=False):
        return self._pre_squeeze(tensor, axes=axes) if not keep_dims else tensor

    def squeeze_output(self, tensor, axes, keep_dims=False):
        return self._post_squeeze(tensor, axes=axes) if not keep_dims else tensor

    def unsqueeze_input(self, tensor, axes, keep_dims=False):
        return self._pre_unsqueeze(tensor, axes=axes) if not keep_dims else tensor

    def unsqueeze_output(self, tensor, axes, keep_dims=False):
        return self._post_unsqueeze(tensor, axes=axes) if not keep_dims else tensor

    def unsqueeze_vector(self, tensor):
        original = self._tensor_map[tensor]
        if self._is_constant(original) and len(original.consumers) == 1:
            self._transform_constant(tensor, lambda data: np.expand_dims(data, 0))
            return tensor
        else:
            return self.unsqueeze_input(tensor, axes=[0])

    def convert_binarg(self, tensor, other):
        if tensor.rank == 0:
            return tensor
        needs_transpose = self.transposing(other) and not self.transposing(tensor)
        if other.rank > tensor.rank:
            if tensor.rank == 1 and needs_transpose:
                return self.unsqueeze_vector(tensor)
            tensor = self._pre_unsqueeze(tensor, axes=list(range(other.rank - tensor.rank)))
        return self.transpose_input(tensor) if needs_transpose else tensor

    def ensure_list(self, value):
        return value if isinstance(value, list) else list(value) if isinstance(value, tuple) else [value]

    def is_bit_set(self, mask, idx):
        return mask & (1 << idx) != 0

    def bit_count(self, mask):
        count = 0
        for i in range(mask.bit_length()):
            if self.is_bit_set(mask, i):
                count += 1
        return count

    def replace_item_with(self, items, index, count, value):
        return items[:index] + [value] * count + items[index+1:]

    def replace_bit_with(self, mask, index, count, value):
        value_bits = (((1 << count) - 1) << index) if value else 0
        low_bits = mask & ((1 << index) - 1)
        high_bits = (mask & ~((1 << (index + 1)) - 1)) << (count - 1)
        return low_bits | value_bits | high_bits

    def beg_index(self, stride):
        return _INT_MAX if stride < 0 else 0

    def end_index(self, stride):
        return _INT_MAX if stride > 0 else -_INT_MAX


_Transforms = Converter.unpack_transforms({
    'Placeholder':
        Transform(
            type='external',
            using={'needs_transpose': '!needs_io_transpose(O[0])'},
            outputs='!transpose_output(O[0]) if needs_transpose else O[0]',
            attribs={
                'shape': '!list(nxc_to_ncx(shape) if needs_transpose else shape)',
                'dtype': '!dtype',
            }
        ),
    'Const':
        Transform(
            type='constant',
            outputs='!O[0]',
            attribs={
                'shape': '!list(value.shape)',
                'dtype': '!dtype',
                'value': '!value',
            }
        ),
    ('Conv2D', 'Conv3D', 'DepthwiseConv2dNative'):
        Transform(
            type='conv',
            using={
                'depthwise': '!_type_ == "DepthwiseConv2dNative"',
            },
            defaults={
                'explicit_paddings': [],
            },
            inputs=(
                '!transpose_input(I[0], data_format)',
                '!transpose_filter(I[1]) if not depthwise else transpose_depthwise_filter(I[1])',
            ),
            outputs=(
                '!transpose_output(O[0], data_format)',
            ),
            attribs={
                'stride': '!convert_size(strides, data_format)',
                'dilation': '!convert_size(dilations, data_format)',
                'padding': '!convert_padding(padding, I[0].rank - 2, explicit_paddings, data_format)',
                'groups': '!1 if not depthwise else 0',
            }
        ),
    ('Conv2DBackpropInput', 'Conv3DBackpropInput', 'DepthwiseConv2dNativeBackpropInput'):
        Transform(
            type='deconv',
            using={
                'depthwise': '!_type_ == "DepthwiseConv2dNativeBackpropInput"',
            },
            defaults={
                'explicit_paddings': [],
            },
            inputs=(
                '!transpose_input(I[2], data_format)',
                '!transpose_filter(I[1]) if not depthwise else transpose_depthwise_filter(I[1])',
            ),
            outputs=(
                '!transpose_output(O[0], data_format)',
            ),
            attribs={
                'stride': '!convert_size(strides, data_format)',
                'dilation': '!convert_size(dilations, data_format)',
                'padding': '!convert_padding(padding, I[2].rank - 2, explicit_paddings, data_format)',
                'output_shape': '!nxc_to_ncx(as_const(I[0])) if is_nxc(data_format) else as_const(I[0])',
                'groups': '!1 if not depthwise else 0',
            }
        ),
    ('MaxPool', 'AvgPool', 'MaxPoolWithArgmax'):
        Transform(
            type=('max_pool', 'avg_pool', 'max_pool_with_index'),
            defaults={
                'explicit_paddings': [],
                'data_format': 'NHWC',
            },
            inputs=(
                '!transpose_input(I[0], data_format)',
            ),
            outputs=(
                '!transpose_output(O[0], data_format)',
                '!transpose_output(O[1], data_format) if len(O) > 1 else None',
            ),
            attribs={
                'size': '!nxc_to_ncx(ksize) if is_nxc(data_format) else ksize',
                'stride': '!nxc_to_ncx(strides) if is_nxc(data_format) else strides',
                'padding': '!convert_padding(padding, I[0].rank, explicit_paddings, data_format)',
                'border': '!"ignore"',
            }
        ),
    'Concat':
        Transform(
            type='concat',
            inputs=['!I[1:]'],
            outputs='!transpose_like(O[0], I[1])',
            attribs={
                'axis': '!transpose_axis_like(as_const(I[0]), I[1], O[0].rank)',
            }
        ),
    'ConcatV2':
        Transform(
            type='concat',
            inputs=['!I[:-1]'],
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'axis': '!transpose_axis_like(as_const(I[-1]), I[0], O[0].rank)',
            }
        ),
    'Split':
        Transform(
            type='split',
            inputs='!I[1]',
            outputs=['![transpose_like(O[i], I[1]) for i in range(len(O))]'],
            attribs={
                'axis': '!transpose_axis_like(as_const(I[0]), I[1])',
                'ratios': '![1] * (num_split if not _lite_ else num_splits)',
            }
        ),
    'SplitV':
        Transform(
            type='split',
            inputs='!I[0]',
            outputs=['![transpose_like(O[i], I[0]) for i in range(len(O))]'],
            attribs={
                'axis': '!transpose_axis_like(as_const(I[2]), I[0])',
                'ratios': '!as_const(I[1])',
            }
        ),
    'Reshape':
        Transform(
            type='reshape',
            inputs='!undo_transpose(I[0])',
            outputs='!O[0]',
            attribs={
                'shape': '!flexible_batch(as_const(I[1]), I[0].shape[0])',
                'dtype': '!I[0].dtype',
            }
        ),
    'Transpose':
        Transform(
            type='transpose',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!transpose_axis_like(as_const(I[1]), I[0])',
                'dtype': '!I[0].dtype',
            }
        ),
    'Squeeze':
        Transform(
            type='squeeze',
            inputs='!undo_transpose(I[0])',
            outputs='!O[0]',
            attribs={
                'axes': '!ensure_list(ensure_positive(squeeze_dims, I[0].rank)) if len(squeeze_dims) != 0 else'
                        ' [i for i, x in enumerate(I[0].shape) if x == 1]',
                'dtype': '!I[0].dtype',
            }
        ),
    'ExpandDims':
        Transform(
            type='unsqueeze',
            inputs='!undo_transpose(I[0])',
            outputs='!O[0]',
            attribs={
                'axes': '!ensure_list(ensure_positive(as_const(I[1]), O[0].rank))',
                'dtype': '!I[0].dtype',
            }
        ),
    'Pack':
        Transform(
            type='stack',
            inputs=['![undo_transpose(t) for t in I]'],
            outputs='!O[0]',
            attribs={
                'axis': '!ensure_positive(axis, O[0].rank)',
            }
        ),
    'Unpack':
        Transform(
            type='unstack',
            inputs='!undo_transpose(I[0])',
            outputs=['!O[:]'],
            attribs={
                'axis': '!ensure_positive(axis, I[0].rank)',
            }
        ),
    ('Min', 'Max', 'Mean', 'Sum', 'Any', 'All'):
        Transform(
            type=('min_reduce', 'max_reduce', 'mean_reduce', 'sum_reduce', 'any_reduce', 'all_reduce'),
            using={
                'axes': '!ensure_list(transpose_axis_like(as_const(I[1]), I[0]))'
            },
            inputs='!I[0]',
            outputs=(
                '!transpose_like(O[0], I[0]) if keep_dims else squeeze_output(O[0], axes)',
            ),
            attribs={
                'axes': '!axes',
            }
        ),
    ('Add', 'AddV2', 'Sub', 'Mul', 'RealDiv', 'Pow', 'LogicalAnd', 'LogicalOr',
     'Less', 'Greater', 'LessEqual', 'GreaterEqual', 'Equal', 'NotEqual', 'Minimum', 'Maximum'):
        Transform(
            type=('add', 'add', 'sub', 'mul', 'div', 'pow', 'and', 'or',
                  'lt', 'gt', 'le', 'ge', 'eq', 'ne', 'min', 'max'),
            inputs=(
                '!convert_binarg(I[0], I[1])',
                '!convert_binarg(I[1], I[0])',
            ),
            outputs='!transpose_output(O[0]) if transposing(I[0]) or transposing(I[1]) else O[0]',
        ),
    ('Identity', 'Relu', 'Relu6', 'Elu', 'Selu', 'Gelu', 'Silu', 'Swish', 'Sigmoid', 'Softplus', 'Exp', 'Log',
     'Sin', 'Cos', 'Tan', 'Asin', 'Acos', 'Atan', 'Sinh', 'Cosh', 'Tanh', 'Asinh', 'Acosh', 'Atanh',
     'Neg', 'Reciprocal', 'Sign', 'Abs', 'Floor', 'Ceil', 'Round', 'Square', 'Sqrt', 'Rsqrt', 'LogicalNot'):
        Transform(
            type=('copy', 'relu', 'relu6', 'elu', 'selu', 'gelu', 'silu', 'silu', 'sigmoid', 'softplus', 'exp', 'log',
                  'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
                  'neg', 'rcp', 'sign', 'abs', 'floor', 'ceil', 'round', 'sqr', 'sqrt', 'rsqrt', 'not'),
            inputs='!I[0]',
            outputs='!transpose_like(O[0], I[0])',
        ),
    'LeakyRelu':
        Transform(
            type='leaky_relu',
            inputs='!I[0]',
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'alpha': '!alpha',
            }
        ),
    ('FusedBatchNorm', 'FusedBatchNormV3'):
        Transform(
            type='batch_normalization',
            inputs=(
                '!transpose_input(I[0], data_format)',
                '!unsqueeze_vector(I[3])',
                '!unsqueeze_vector(I[4])',
                '!unsqueeze_vector(I[2])',
                '!unsqueeze_vector(I[1])',
            ),
            outputs=(
                '!transpose_output(O[0], data_format)',
            ),
            attribs={
                'epsilon': '!epsilon',
            }
        ),
    'BiasAdd':
        Transform(
            type='add',
            inputs=(
                '!transpose_input(I[0], data_format)',
                '!unsqueeze_vector(I[1])',
            ),
            outputs='!transpose_output(O[0], data_format)',
        ),
    'Softmax':
        Transform(
            type='softmax',
            cond={
                '!beta == 1 if _lite_ else True': 'beta must be 1',
            },
            inputs='!transpose_input(I[0])',
            outputs='!transpose_output(O[0])',
            attribs={
                'axes': [1],
            }
        ),
    'MatMul':
        Transform(
            type='matmul',
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            attribs={
                'transposeA': '!transpose_a',
                'transposeB': '!transpose_b',
            },
        ),
    'ClipByValue':
        Transform(
            type='clamp',
            inputs=('!I[0]', '!I[1]', '!I[2]'),
            outputs='!transpose_like(O[0], I[0])',
        ),
    ('Pad', 'MirrorPad'):
        Transform(
            type='pad',
            cond={
                '!mode in ["CONSTANT", "REFLECT", "SYMMETRIC"]':
                    'mode must be one of "CONSTANT", "REFLECT" or SYMMETRIC',
            },
            defaults={
                'mode': 'CONSTANT',
            },
            using={
                'paddings': '!transpose_list_like(as_const(I[1]), ref=I[0])',
            },
            inputs='!I[0]',
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'padding': '![tuple(item) for item in paddings]',
                'border': '!"reflect" if mode == "REFLECT" else "reflect-even" if mode == "SYMMETRIC" else "constant"',
            }
        ),
    'Slice':
        Transform(
            type='slice',
            using={
                'beg': '!as_const(I[1])',
                'end': '![0 if s == -1 else b + s for b, s in zip(as_const(I[1]), as_const(I[2]))]',
            },
            inputs='!I[0]',
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'axes': '!list(range(I[0].rank))',
                'begin': '!transpose_list_like(beg, ref=I[0])',
                'end': '!transpose_list_like(end, ref=I[0])',
            }
        ),
    'StridedSlice':
        Transform(
            type='slice',
            using=OrderedDict([
                ('ref', '!I[0] if new_axis_mask == 0 else None'),
                ('rank', '!I[0].rank + bit_count(new_axis_mask)'),
                ('stride', '!as_const(I[3])'),
                ('ellipsis_index', '!int(math.log2(ellipsis_mask)) if ellipsis_mask != 0 else None'),
                ('ellipsis_count', '!rank - (len(stride) - 1) if ellipsis_mask != 0 else None'),
                ('beg', '!replace_item_with(as_const(I[1]), ellipsis_index, ellipsis_count, 0) '
                        'if ellipsis_index is not None else as_const(I[1])'),
                ('end', '!replace_item_with(as_const(I[2]), ellipsis_index, ellipsis_count, 0) '
                        'if ellipsis_index is not None else as_const(I[2])'),
                ('begin_mask', '!replace_bit_with(begin_mask, ellipsis_index, ellipsis_count, 1) '
                               'if ellipsis_index is not None else begin_mask'),
                ('end_mask', '!replace_bit_with(end_mask, ellipsis_index, ellipsis_count, 1) '
                             'if ellipsis_index is not None else end_mask'),
                ('new_axis_mask', '!replace_bit_with(new_axis_mask, ellipsis_index, ellipsis_count, 0) '
                                  'if ellipsis_index is not None else new_axis_mask'),
                ('shrink_axis_mask', '!replace_bit_with(shrink_axis_mask, ellipsis_index, ellipsis_count, 0) '
                                     'if ellipsis_index is not None else shrink_axis_mask'),
                ('masked_beg', '![beg_index(s) if is_bit_set(begin_mask,i) else b '
                               'for i, (b, s) in enumerate(zip(beg,stride))]'),
                ('masked_end', '![end_index(s) if is_bit_set(end_mask,i) else b + 1 '
                               'if is_bit_set(shrink_axis_mask,i) else e '
                               'for i, (b, e, s) in enumerate(zip(beg,end,stride))]'),
                ('axes', '!transpose_axis_like([i for i in range(rank) '
                         'if not (is_bit_set(begin_mask,i) and is_bit_set(end_mask,i)) '
                         'and not is_bit_set(new_axis_mask,i)], ref, rank)'),
                ('new_axes', '!transpose_axis_like([i for i in range(rank) '
                             'if is_bit_set(new_axis_mask,i)], ref, rank)'),
                ('del_axes', '!transpose_axis_like([i for i in range(rank) '
                             'if is_bit_set(shrink_axis_mask,i)], ref, rank)'),
            ]),
            inputs='!unsqueeze_input(undo_transpose(I[0]), new_axes) if len(new_axes) else I[0]',
            outputs='!transpose_like(squeeze_output(O[0], del_axes) if len(del_axes) else O[0], ref)',
            attribs={
                'axes': '![i for i in range(rank) if i in axes]',
                'begin': '![b for i, b in enumerate(transpose_list_like(masked_beg, ref)) if i in axes]',
                'end': '![e for i, e in enumerate(transpose_list_like(masked_end, ref)) if i in axes]',
                'stride': '![s for i, s in enumerate(transpose_list_like(stride, ref)) if i in axes]',
            }
        ),
    ('ArgMin', 'ArgMax'):
        Transform(
            type=('argmin_reduce', 'argmax_reduce'),
            using={
                'axis': '!transpose_axis_like(as_const(I[1]), ref=I[0])'
            },
            inputs='!I[0]',
            outputs='!transpose_like(O[0], ref=I[0]) if _lite_ else squeeze_output(O[0], [axis])',
            attribs={
                'axes': '!ensure_list(axis)',
            }
        ),
    'Select':
        Transform(
            type='select',
            inputs=(
                '!I[0]',
                '!convert_binarg(I[1], I[0])',
                '!convert_binarg(I[2], I[0])',
            ),
            outputs='!transpose_like(O[0], ref=I[0])',
        ),
    'Tile':
        Transform(
            type='tile',
            inputs='!I[0]',
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'repeats': '!transpose_list_like(as_const(I[1]), I[0])',
            }
        ),
    'ResizeNearestNeighbor':
        Transform(
            type='!"nearest_upsample" if upsample else "nearest_downsample"',
            using=OrderedDict([
                ('old_size', '!I[0].shape[1:-1]'),
                ('new_size', '!as_const(I[1])'),
                ('upsample', '!is_integer_upsample(old_size, new_size)'),
                ('downsample', '!is_integer_downsample(old_size, new_size)'),
            ]),
            cond={
                '!upsample or downsample': 'nearest resize must be integer up-sample or down-sample',
                '!not align_corners': 'align_corners is not supported',
            },
            inputs='!transpose_input(I[0])',
            outputs='!transpose_output(O[0])',
            attribs={
                'factor': '!upsample_factor(old_size, new_size) if upsample else downsample_factor(old_size, new_size)',
            }
        ),
    'ResizeArea':
        Transform(
            type='area_downsample',
            cond={
                '!is_integer_downsample(I[0].shape[1:-1], O[0].shape[1:-1])': 'area resize must be integer down-sample',
                '!not align_corners': 'align_corners is not supported',
            },
            using={
                'size': '!I[0].shape[1:-1]'
            },
            inputs='!transpose_input(I[0])',
            outputs='!transpose_output(O[0])',
            attribs={
                'factor': '!downsample_factor(size, as_const(I[1]))',
            }
        ),
    'ResizeBilinear':
        Transform(
            type='multilinear_upsample',
            cond={
                '!is_integer_upsample(I[0].shape[1:-1], O[0].shape[1:-1])': 'bilinear resize must be integer up-sample',
            },
            using={
                'size': '!I[0].shape[1:-1]'
            },
            inputs='!transpose_input(I[0])',
            outputs='!transpose_output(O[0])',
            attribs={
                'factor': '!upsample_factor(size, as_const(I[1]))',
                'method': '!"aligned" if align_corners else "symmetric" if half_pixel_centers else "asymmetric"',
            }
        ),
    'LRN':
        Transform(
            type='local_response_normalization',
            using={
                'size': '!(radius if _lite_ else depth_radius) * 2 + 1'
            },
            inputs='!I[0]',
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'size': '![1, size] + [1] * (I[0].rank - 2)',
                'alpha': '!alpha * size',
                'beta': '!beta',
                'bias': '!bias',
            }
        ),
    'Cast':
        Transform(
            using={
                'same_type': '!nnef_dtype(O[0].dtype) == nnef_dtype(I[0].dtype)',
            },
            type='!"copy" if same_type else "cast"',
            inputs='!I[0]',
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'dtype': '!O[0].dtype if not same_type else None',
            },
        ),
    ('Gather', 'GatherV2'):
        Transform(
            type='gather',
            using={
                'axis': '!transpose_axis_like(as_const(I[2]), ref=I[0])'
            },
            inputs=('!I[0]', '!I[1]'),
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'axis': '!axis',
            },
        ),
    'AddN':
        Transform(
            type='add_n',
            inputs=['!I[:]'],
            outputs='!transpose_like(O[0], I[0])'
        ),
})
