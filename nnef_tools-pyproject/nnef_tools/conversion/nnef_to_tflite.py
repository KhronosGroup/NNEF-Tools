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
from .converter import ConverterFromNNEF as _Converter, Transform, ConversionError
from .nnef_to_tf import Converter as _TFConverter, _Transforms as _TFTransforms
from ..model import Tensor, Operation
from ..model.utils import generate_tensor_names_from_op_type
from ..io.tf.lite import CustomOptionsKey
import numpy as np
import copy


def tflite_detection_postprocess_shape(input, scores, anchors, **kwargs):
    return [], [], [], []


class Converter(_TFConverter):

    @staticmethod
    def defined_shapes():
        return {
            'relu6': lambda shape: shape,
            'TFLite_Detection_PostProcess': tflite_detection_postprocess_shape,
        }

    @staticmethod
    def decomposed_operations():
        return _Converter.decomposed_operations()

    def __init__(self, io_transpose=False, custom_transforms=None, custom_functions=None, mirror_unsupported=False):
        _Converter.__init__(self, transforms=self.merge_transforms(_Transforms, custom_transforms),
                            functions=custom_functions, mirror_unsupported=mirror_unsupported)
        self._data_format = 'NXC'
        self._io_transpose = io_transpose

    def __call__(self, graph):
        graph = _TFConverter.__call__(self, graph)
        self._generate_tensor_names(graph)
        self._fix_custom_options(graph)
        return graph

    def _global_attribs(self):
        return {'_lite_': True}

    def _prepare(self, graph):
        self._fix_quantized_dtypes(graph)
        self._fix_quantization_attribs(graph)
        self._transpose_externals(graph)

    def _transpose_externals(self, graph):
        for tensor in graph.tensors:
            mapped = self._tensor_map[tensor]
            if mapped.producer and mapped.producer.type == 'external' and self.needs_io_transpose(tensor):
                self._transposes[tensor] = self.ncx_to_nxc(tensor.shape)

    def _generate_tensor_names(self, graph):
        generate_tensor_names_from_op_type(graph)

        placeholders = 0
        constants = 0
        for tensor in graph.tensors:
            if tensor.name is None:
                if tensor.data is None:
                    placeholders += 1
                    tensor.name = 'PLACEHOLDER' + str(placeholders)
                else:
                    constants += 1
                    tensor.name = 'CONSTANT' + str(constants)

    def _fix_quantized_dtypes(self, graph):
        for tensor in graph.tensors:
            if tensor.quant and tensor.dtype == np.float32:
                bits = tensor.quant['bits']
                signed = tensor.quant['signed']
                assert bits == 8 or bits == 32
                tensor.dtype = (np.int8 if signed else np.uint8) if bits == 8 else (np.int32 if signed else np.uint32)

    def _fix_quantization_attribs(self, graph):
        for tensor in graph.tensors:
            if tensor.quant:
                opname = tensor.quant['op-name']
                if opname != 'zero_point_linear_quantize':
                    raise ConversionError("Quantization operation '{}' cannot be converted to TFLite")

                del tensor.quant['op-name']
                del tensor.quant['bits']
                if 'signed' in tensor.quant:
                    del tensor.quant['signed']
                if 'symmetric' in tensor.quant:
                    del tensor.quant['symmetric']

    def _fix_custom_options(self, graph):
        for op in graph.operations:
            if op.custom:
                options = op.attribs.get(CustomOptionsKey)
                if options is not None:
                    op.attribs[CustomOptionsKey] = bytes.fromhex(options)

    def _make_constant(self, graph, dtype, value, inline):
        return Tensor(graph, dtype=dtype, shape=self._shape_of(value), data=value)

    def _ensure_constant_producer(self, tensor):
        pass

    def _transform_constant(self, tensor, func):
        data = func(tensor.data)
        tensor.shape = data.shape
        tensor.data = data

    def _squeeze_operation(self, input, output, axes):
        Operation(input.graph, type='SQUEEZE', inputs=input, outputs=output, attribs={'squeeze_dims': axes})

    def _unsqueeze_operation(self, input, output, axes):
        if len(axes) == 1:
            Operation(input.graph, type='EXPAND_DIMS', inputs=(input, self.as_tensor(axes[0], np.int32)),
                      outputs=output)
        else:
            Operation(input.graph, type='RESHAPE', inputs=(input, self.as_tensor(output.shape, np.int32)),
                      outputs=output, attribs={'new_shape': output.shape})

    def _transpose_operation(self, input, output, perm):
        Operation(input.graph, type='TRANSPOSE', inputs=(input, self.as_tensor(perm, np.int32)),
                  outputs=output)

    def _reshape_operation(self, input, output, shape):
        Operation(input.graph, type='RESHAPE', inputs=(input, self.as_tensor(shape, np.int32)), outputs=output,
                  attribs={'new_shape': shape})

    def _bias_operation(self, input, output, bias):
        if not isinstance(bias, Tensor):
            bias = self.as_tensor(bias, np.float32)

        Operation(input.graph, type='ADD', inputs=(input, bias), outputs=output)

    def _scale_operation(self, input, output, scalar):
        if not isinstance(scalar, Tensor):
            scalar = self.as_tensor(scalar, np.float32)

        Operation(input.graph, type='MUL', inputs=(input, scalar), outputs=output)

    def _pad_operation(self, input, output, paddings):
        if not isinstance(paddings, Tensor):
            paddings = self.as_tensor(paddings, np.int64)

        Operation(input.graph, type='PAD', inputs=(input, paddings), outputs=output, attribs={})

    def is_same_padding(self, input_size, output_size, stride):
        return all(o == i // s for i, o, s in zip(input_size, output_size, stride))

    def is_valid_padding(self, padding):
        return len(padding) != 0 and all(p == (0, 0) for p in padding)

    def pad_input(self, input, paddings):
        if all(item == (0, 0) for item in paddings):
            return input

        shape = tuple(p + x + q for x, (p, q) in zip(self._working_shape(input), paddings))
        output = Tensor(input.graph, dtype=input.dtype, shape=shape, quant=copy.deepcopy(input.quant))
        self._pad_operation(input, output, paddings)
        return output


_Transforms = Converter.unpack_transforms({
    ('external', 'constant'):
        Transform(type=None),
    'conv':
        Transform(
            type='!"CONV_2D" if not depthwise else "DEPTHWISE_CONV_2D"',
            cond={
                '!I[0].rank == 4': 'rank must be 4',
            },
            using={
                'depthwise': '!groups == 0',
                'channels': '!I[0].shape[1]',
                'valid_pad': '!is_valid_padding(padding)',
                'same_pad': '!is_same_padding(I[0].shape[2:], O[0].shape[2:], stride)',
                'pads': '![(0, 0)] + padding + [(0, 0)]',
            },
            inputs=(
                '!transpose_input(I[0]) if same_pad or valid_pad else pad_input(transpose_input(I[0]), pads)',
                '!transpose_filter(I[1], format="NXC" if not depthwise else "CXN")',
                '!squeeze_vector(I[2])',
            ),
            outputs='!transpose_output(O[0])',
            attribs={
                'stride_h': '!stride[0]',
                'stride_w': '!stride[1]',
                'dilation_h_factor': '!dilation[0]',
                'dilation_w_factor': '!dilation[1]',
                'padding': '!"VALID" if valid_pad else "SAME"',
                'depth_multiplier': '!O[0].shape[1] // channels if depthwise else None',
            }
        ),
    'deconv':
        Transform(
            type='TRANSPOSE_CONV',
            cond={
                '!I[0].rank == 4': 'rank must be 4',
                '!groups == 1': 'groups must be 1',
            },
            using={
                'depthwise': '!groups == 0',
                'channels': '!O[0].shape[1]',
                'valid_pad': '!is_valid_padding(padding)',
                'same_pad': '!is_same_padding(I[0].shape[2:], O[0].shape[2:], stride)',
                'pads': '![(0, 0)] + padding + [(0, 0)]',
            },
            inputs=(
                '!as_tensor(ncx_to_nxc(output_shape), np.int32)',
                '!transpose_filter(I[1], format="CXN" if not depthwise else "NXC")',
                '!transpose_input(I[0]) if same_pad or valid_pad else pad_input(transpose_input(I[0]), pads)',
            ),
            outputs='!bias_add(transpose_output(O[0]), squeeze_vector(I[2]) if I[2].rank == 2 else I[2])',
            attribs={
                'stride_h': '!stride[0]',
                'stride_w': '!stride[1]',
                'padding': '!"VALID" if valid_pad else "SAME"',
                'depth_multiplier': '!I[1].shape[0] // channels if depthwise else None',
            }
        ),
    ('max_pool', 'avg_pool'):
        Transform(
            cond={
                '!size[0] == 1 and size[1] == 1 and ': 'size must be 1 in batch and channel dimensions',
                '!stride[0] == 1 and stride[1] == 1': 'stride must be 1 in batch and channel dimensions',
                '!border == "ignore"': 'border must be "ignore"',
            },
            type=('MAX_POOL_2D', 'AVERAGE_POOL_2D'),
            using={
                'valid_pad': '!is_valid_padding(padding)',
                'same_pad': '!is_same_padding(I[0].shape[2:], O[0].shape[2:], stride[2:])',
            },
            inputs=(
                '!transpose_input(I[0]) if same_pad or valid_pad else pad_input(transpose_input(I[0]), padding)',
            ),
            outputs=(
                '!transpose_output(O[0])',
            ),
            attribs={
                'filter_height': '!size[2]',
                'filter_width': '!size[3]',
                'stride_h': '!stride[2]',
                'stride_w': '!stride[3]',
                'padding': '!"VALID" if valid_pad else "SAME"',
            }
        ),
    'reshape':
        Transform(
            type='RESHAPE',
            using={
                'new_shape': '!fixed_batch(shape, I[0].shape[0])',
            },
            inputs=(
                '!undo_transpose(I[0])',
                '!as_tensor(new_shape, np.int32)',
            ),
            outputs='!O[0]',
            attribs={
                'new_shape': '!new_shape',
            }
        ),
    'concat':
        Transform(
            type='CONCATENATION',
            inputs=['!I[:]'],
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'axis': '!transpose_axis_like(axis, I[0])',
            }
        ),
    'copy':
        Transform(
            type='RESHAPE',
            using={
                'shape': '!transpose_list_like(I[0].shape, I[0])',
            },
            inputs=(
                '!I[0]',
                '!as_tensor(shape, np.int32)',
            ),
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'new_shape': '!shape',
            }
        ),
    'linear':
        Transform(
            type='FULLY_CONNECTED',
            inputs=(
                '!I[0]',
                '!I[1]',
                '!squeeze_vector(I[2]) if not is_zero(I[2]) else None',
            ),
            outputs='!O[0]',
            attribs={
                'fused_activation_function': "NONE",
                'weights_format': "DEFAULT",
                'keep_num_dims': True,
                'asymmetric_quantize_inputs': False,
            }
        ),
    'matmul':
        Transform(
            type='BATCH_MATMUL',
            inputs=(
                '!I[0]',
                '!I[1]',
            ),
            outputs='!O[0]',
            attribs={
                'adj_x': '!transposeA',
                'adj_y': '!transposeB',
                'asymmetric_quantize_inputs': False,
            }
        ),
    'batch_normalization':
        Transform(
            type='MUL',
            cond={
                '!I[1].data is not None and I[2].data is not None and'
                ' (len(I) == 3 or I[3].data is not None) and (len(I) == 4 or I[4].data is not None)':
                    'all parameters must be constants',
                '!not any(t.quant for t in I)': 'quantized inputs or parameters are not supported',
            },
            using={
                'mean': '!np.squeeze(I[1].data, axis=0) if I[1].data is not None else None',
                'std': '!np.squeeze(np.sqrt(I[2].data + epsilon), axis=0) if I[2].data is not None else None',
                'offset': '!np.squeeze(I[3].data, axis=0) if I[3].data is not None else None if len(I) > 3 else 0',
                'scale': '!np.squeeze(I[4].data, axis=0) if I[4].data is not None else None if len(I) > 4 else 1',
            },
            inputs=(
                '!transpose_input(I[0])',
                '!as_tensor(scale / std, np.float32)',
            ),
            outputs='!bias_add(transpose_like(O[0], I[0]), as_tensor(offset - scale * mean / std, np.float32))',
        ),
    'l2_normalization':
        Transform(
            type='L2_NORMALIZATION',
            cond={
                '!axes == list(range(I[0].rank))': 'axes must denote all dimensions',
            },
            inputs='!I[0]',
            outputs='!transpose_like(O[0], I[0])',
        ),
    'prelu':
        Transform(
            type='PRELU',
            inputs=('!I[0]', '!I[1]'),
            outputs='!transpose_like(O[0], I[0])',
        ),
    'pad':
        Transform(
            type='!"PAD" if border == "constant" else "MIRROR_PAD"',
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
                'mode': '!0 if border == "reflect" else 1 if border == "reflect-even" else None',
            },
        ),
    'gather':
        Transform(
            type='GATHER',
            inputs=('!I[0]', '!I[1]'),
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'axis': '!transpose_axis_like(axis, I[0])',
            },
        ),
    'cast':
        Transform(
            type='CAST',
            inputs='!I[0]',
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'in_data_type': '!I[0].dtype',
                'out_data_type': '!O[0].dtype',
            },
        ),
    # 'copy': _TFTransforms['copy'].with_type('IDENTITY'),  # only works in TF 2.3
    'transpose': _TFTransforms['transpose'].with_type('TRANSPOSE'),
    'split': _TFTransforms['split'].with_type('SPLIT_V'),
    'squeeze': _TFTransforms['squeeze'].with_type('SQUEEZE'),
    'unsqueeze': _TFTransforms['unsqueeze'].with_type('!"EXPAND_DIMS" if len(axes) == 1 else "RESHAPE"'),
    'relu': _TFTransforms['relu'].with_type('RELU'),
    'relu6': _TFTransforms['relu6'].with_type('RELU6'),
    'elu': _TFTransforms['elu'].with_type('ELU'),
    'leaky_relu': _TFTransforms['leaky_relu'].with_type('LEAKY_RELU'),
    'sigmoid': _TFTransforms['sigmoid'].with_type('LOGISTIC'),
    'sin': _TFTransforms['sin'].with_type('SIN'),
    'cos': _TFTransforms['cos'].with_type('COS'),
    'tan': _TFTransforms['tan'].with_type('TAN'),
    'asin': _TFTransforms['asin'].with_type('ASIN'),
    'acos': _TFTransforms['acos'].with_type('ACOS'),
    'atan': _TFTransforms['atan'].with_type('ATAN'),
    'sinh': _TFTransforms['sinh'].with_type('SINH'),
    'cosh': _TFTransforms['cosh'].with_type('COSH'),
    'tanh': _TFTransforms['tanh'].with_type('TANH'),
    'asinh': _TFTransforms['asinh'].with_type('ASINH'),
    'acosh': _TFTransforms['acosh'].with_type('ACOSH'),
    'atanh': _TFTransforms['atanh'].with_type('ATANH'),
    'exp': _TFTransforms['exp'].with_type('EXP'),
    'log': _TFTransforms['log'].with_type('LOG'),
    'abs': _TFTransforms['abs'].with_type('ABS'),
    'neg': _TFTransforms['neg'].with_type('NEG'),
    'not': _TFTransforms['not'].with_type('LOGICAL_NOT'),
    'floor': _TFTransforms['floor'].with_type('FLOOR'),
    'ceil': _TFTransforms['ceil'].with_type('CEIL'),
    'round': _TFTransforms['round'].with_type('ROUND'),
    'sqr': _TFTransforms['sqr'].with_type('SQUARE'),
    'sqrt': _TFTransforms['sqrt'].with_type('SQRT'),
    'rsqrt': _TFTransforms['rsqrt'].with_type('RSQRT'),
    'add': _TFTransforms['add'].with_type('ADD'),
    'sub': _TFTransforms['sub'].with_type('SUB'),
    'mul': _TFTransforms['mul'].with_type('MUL'),
    'div': _TFTransforms['div'].with_type('DIV'),
    'pow': _TFTransforms['pow'].with_type('POW'),
    'min': _TFTransforms['min'].with_type('MINIMUM'),
    'max': _TFTransforms['max'].with_type('MAXIMUM'),
    'and': _TFTransforms['and'].with_type('LOGICAL_AND'),
    'or': _TFTransforms['or'].with_type('LOGICAL_OR'),
    'lt': _TFTransforms['lt'].with_type('LESS'),
    'le': _TFTransforms['le'].with_type('LESS_EQUAL'),
    'gt': _TFTransforms['gt'].with_type('GREATER'),
    'ge': _TFTransforms['ge'].with_type('GREATER_EQUAL'),
    'eq': _TFTransforms['eq'].with_type('EQUAL'),
    'ne': _TFTransforms['ne'].with_type('NOT_EQUAL'),
    'select': _TFTransforms['select'].with_type('SELECT'),
    'min_reduce': _TFTransforms['min_reduce'].with_type('REDUCE_MIN'),
    'max_reduce': _TFTransforms['max_reduce'].with_type('REDUCE_MAX'),
    'mean_reduce': _TFTransforms['mean_reduce'].with_type('MEAN'),
    'sum_reduce': _TFTransforms['sum_reduce'].with_type('SUM'),
    'any_reduce': _TFTransforms['any_reduce'].with_type('REDUCE_ANY'),
    'all_reduce': _TFTransforms['all_reduce'].with_type('REDUCE_ALL'),
    'argmin_reduce': _TFTransforms['argmin_reduce'].with_type('ARG_MIN'),
    'argmax_reduce': _TFTransforms['argmax_reduce'].with_type('ARG_MAX'),
    'stack': _TFTransforms['stack'].with_type('PACK'),
    'unstack': _TFTransforms['unstack'].with_type('UNPACK'),
    'tile': _TFTransforms['tile'].with_type('TILE'),
    'slice': _TFTransforms['slice'].with_type('STRIDED_SLICE'),
    'softmax': _TFTransforms['softmax'].with_type('SOFTMAX'),
    'local_response_normalization': _TFTransforms['local_response_normalization'].with_type('LOCAL_RESPONSE_NORMALIZATION'),
    'nearest_upsample': _TFTransforms['nearest_upsample'].with_type('RESIZE_NEAREST_NEIGHBOR'),
    'nearest_downsample': _TFTransforms['nearest_downsample'].with_type('RESIZE_NEAREST_NEIGHBOR'),
    'multilinear_upsample': _TFTransforms['multilinear_upsample'].with_type('RESIZE_BILINEAR'),
    'add_n': _TFTransforms['add_n'].with_type('ADD_N'),
})
