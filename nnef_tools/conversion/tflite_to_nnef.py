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
from .converter import Converter as _Converter, Transform, ConversionError
from .tf_to_nnef import Converter as _TFConverter, _Transforms as _TFTransforms, _RELU6_FRAGMENT
from ..model import Tensor, Operation
from ..io.tf.lite import CustomOptionsKey
import numpy as np
import copy


_DETECTION_POSTPROCESS_FRAGMENT = """
fragment TFLite_Detection_PostProcess( 
    boxes: tensor<scalar>, 
    scores: tensor<scalar>, 
    anchors: tensor<scalar>, 
    detections_per_class: integer,
    max_classes_per_detection: integer,
    max_detections: integer, 
    nms_iou_threshold: scalar, 
    nms_score_threshold: scalar, 
    num_classes: integer, 
    use_regular_nms: logical, 
    h_scale: scalar,
    w_scale: scalar, 
    x_scale: scalar, 
    y_scale: scalar ) 
-> ( 
    detection_boxes: tensor<scalar>, 
    detection_classes: tensor<scalar>, 
    detection_scores: tensor<scalar>, 
    num_detections: tensor<scalar> );
"""


class Converter(_TFConverter):

    _ConvOpTypes = ['CONV_1D', 'CONV_2D', 'CONV_3D', 'TRANSPOSE_CONV', 'DEPTHWISE_CONV_2D']

    _ActivationOpTypes = {
        'ELU': 'elu',
        'RELU': 'relu',
        'RELU6': 'relu6',
        'LOGISTIC': 'sigmoid',
        'TANH': 'tanh',
    }

    @staticmethod
    def defined_operations():
        return {
            'relu6': _RELU6_FRAGMENT,
            'TFLite_Detection_PostProcess': _DETECTION_POSTPROCESS_FRAGMENT,
        }

    def __init__(self, io_transpose=False, custom_transforms=None, custom_functions=None,
                 mirror_unsupported=False, keep_io_names=False):
        _Converter.__init__(self, transforms=self.merge_transforms(_Transforms, custom_transforms),
                            functions=custom_functions, mirror_unsupported=mirror_unsupported)
        self._io_transpose = io_transpose
        self._transposed = set()
        self._keep_io_names = keep_io_names

    def __call__(self, graph):
        graph = _TFConverter.__call__(self, graph)
        self._fix_custom_options(graph)
        return graph

    def _global_attribs(self):
        return {'_lite_': True}

    def _prepare(self, graph):
        self._fix_quantization_attribs(graph)
        self._fix_quantized_dtypes(graph)
        self._insert_externals_and_constants(graph)
        self._transpose_externals(graph)

    def _transpose_externals(self, graph):
        for op in graph.operations:
            if op.type == 'external':
                if self.needs_io_transpose(op.output):
                    op.output.shape = self.nxc_to_ncx(op.output.shape)
                    op.attribs['shape'] = list(op.output.shape)
                    self._transposed.add(op.output)

    def _fix_quantized_dtypes(self, graph):
        for tensor in graph.tensors:
            if tensor.quant:
                scale = tensor.quant.get('scale')
                if scale:
                    tensor.dtype = np.float32
                else:
                    tensor.quant = None

    def _fix_quantization_attribs(self, graph):
        for tensor in graph.tensors:
            if tensor.quant and tensor.quant.get('scale'):
                if 'min' in tensor.quant:
                    del tensor.quant['min']
                if 'max' in tensor.quant:
                    del tensor.quant['max']
                tensor.quant['op-name'] = 'zero_point_linear_quantize'
                tensor.quant['bits'] = 32 if self._is_conv_bias(tensor) else 8
                assert tensor.dtype == np.uint8 or tensor.dtype == np.int8 or \
                       tensor.dtype == np.uint32 or tensor.dtype == np.int32, \
                    "unknown quantized dtype '{}'".format(tensor.dtype)
                tensor.quant['signed'] = tensor.dtype == np.int8 or tensor.dtype == np.int32
                tensor.quant['symmetric'] = self._is_conv_filter(tensor)

    def _fix_custom_options(self, graph):
        for op in graph.operations:
            if op.custom:
                options = op.attribs.get(CustomOptionsKey)
                if options is not None:
                    op.attribs[CustomOptionsKey] = options.hex()

    def _is_conv_filter(self, tensor):
        tensor = self._tensor_map.get(tensor)
        return tensor and len(tensor.consumers) > 0 and \
               all(op.type in Converter._ConvOpTypes and op.inputs[1] is tensor for op in tensor.consumers)

    def _is_conv_bias(self, tensor):
        tensor = self._tensor_map.get(tensor)
        return tensor and len(tensor.consumers) > 0 and \
               all(op.type in Converter._ConvOpTypes and op.inputs[2] is tensor for op in tensor.consumers)

    def activation(self, output, func):
        if func is None or func == 'NONE':
            return output

        if func not in self._ActivationOpTypes:
            raise ConversionError("Unsupported fused activation function '{}'".format(func))

        input = Tensor(output.graph, dtype=output.dtype, shape=output.shape, quant=copy.deepcopy(output.quant))
        Operation(output.graph, type=self._ActivationOpTypes[func], inputs=input, outputs=output)
        return input

    def flat_list(self, array):
        return [item for items in array for item in items] if len(array) and isinstance(array[0], (list, tuple)) else array

    def flatten(self, input):
        shape = (input.shape[0], int(np.prod(input.shape[1:])))
        output = Tensor(input.graph, dtype=input.dtype, shape=shape, quant=copy.deepcopy(input.quant))
        self._reshape_operation(input, output, shape)
        return output

    def same_shape(self, input, output):
        return self._tensor_map[input].shape == self._tensor_map[output].shape


_Transforms = Converter.unpack_transforms({
    ('CONV_1D', 'CONV_2D', 'CONV_3D', 'DEPTHWISE_CONV_2D'):
        Transform(
            type='conv',
            using={
                'depthwise': '!_type_ == "DEPTHWISE_CONV_2D"',
            },
            inputs=(
                '!transpose_input(I[0])',
                '!transpose_filter(I[1], format="NXC" if not depthwise else "CXN")',
                '!unsqueeze_vector(I[2])',
            ),
            outputs='!activation(transpose_output(O[0]), fused_activation_function)',
            attribs={
                'stride': '![stride_h, stride_w]',
                'dilation': '![dilation_h_factor, dilation_w_factor]',
                'padding': '!convert_padding(padding, I[0].rank - 2)',
                'groups': '!1 if not depthwise else 0',
            }
        ),
    'TRANSPOSE_CONV':
        Transform(
            type='deconv',
            using={
                'depthwise': False,
            },
            inputs=(
                '!transpose_input(I[2])',
                '!transpose_filter(I[1], format="CXN" if not depthwise else "NXC")',
            ),
            outputs='!transpose_output(O[0])',
            attribs={
                'stride': '![stride_h, stride_w]',
                'padding': '!convert_padding(padding, I[0].rank - 2)',
                'output_shape': '!nxc_to_ncx(as_const(I[0]))',
                'groups': '!1 if not depthwise else 0',
            }
        ),
    ('MAX_POOL_2D', 'AVERAGE_POOL_2D'):
        Transform(
            type=('max_pool', 'avg_pool'),
            inputs=(
                '!transpose_input(I[0])',
            ),
            outputs=(
                '!transpose_output(O[0])',
            ),
            attribs={
                'size': '![1, 1, filter_height, filter_width]',
                'stride': '![1, 1, stride_h, stride_w]',
                'padding': '!convert_padding(padding, I[0].rank)',
            }
        ),
    'RESHAPE':
        Transform(
            type='reshape',
            inputs='!undo_transpose(I[0])',
            outputs='!O[0]',
            attribs={
                'shape': '!flexible_batch(flat_list(as_const(I[1])) if len(I) > 1 else new_shape, I[0].shape[0])',
                'dtype': '!I[0].dtype',
            }
        ),
    'CONCATENATION':
        Transform(
            type='concat',
            inputs=['!I[:]'],
            outputs='!activation(transpose_like(O[0], I[0]), fused_activation_function)',
            attribs={
                'axis': '!transpose_axis_like(axis, I[0], O[0].rank)',
            }
        ),
    'FULLY_CONNECTED':
        Transform(
            type='linear',
            cond='!weights_format == "DEFAULT"',
            inputs=(
                '!I[0] if keep_num_dims else flatten(I[0])',
                '!I[1]',
                '!I[2] if len(I) > 2 else None',
            ),
            outputs='!activation(O[0], fused_activation_function)',
        ),
    'L2_NORMALIZATION':
        Transform(
            type='l2_normalization',
            inputs='!I[0]',
            outputs='!transpose_like(O[0], I[0])',
            attribs={
                'axes': '!list(range(I[0].rank))',
            }
        ),
    'PRELU':
        Transform(
            type='prelu',
            inputs=('!I[0]', '!I[1]'),
            outputs='!transpose_like(O[0], I[0])',
        ),
    'IDENTITY': _TFTransforms['Identity'],
    'TRANSPOSE': _TFTransforms['Transpose'],
    'SPLIT': _TFTransforms['Split'],
    'SPLIT_V': _TFTransforms['SplitV'],
    'PACK': _TFTransforms['Pack'],
    'UNPACK': _TFTransforms['Unpack'],
    'PAD': _TFTransforms['Pad'],
    'TILE': _TFTransforms['Tile'],
    'SQUEEZE': _TFTransforms['Squeeze'],
    'EXPAND_DIMS': _TFTransforms['ExpandDims'],
    'SLICE': _TFTransforms['Slice'],
    'STRIDED_SLICE': _TFTransforms['StridedSlice'],
    'RELU': _TFTransforms['Relu'],
    'RELU6': _TFTransforms['Relu6'],
    'ELU': _TFTransforms['Elu'],
    'LEAKY_RELU': _TFTransforms['LeakyRelu'],
    'LOGISTIC': _TFTransforms['Sigmoid'],
    'TANH': _TFTransforms['Tanh'],
    'SIN': _TFTransforms['Sin'],
    'COS': _TFTransforms['Cos'],
    'EXP': _TFTransforms['Exp'],
    'LOG': _TFTransforms['Log'],
    'ABS': _TFTransforms['Abs'],
    'NEG': _TFTransforms['Neg'],
    'LOGICAL_NOT': _TFTransforms['LogicalNot'],
    'FLOOR': _TFTransforms['Floor'],
    'CEIL': _TFTransforms['Ceil'],
    'ROUND': _TFTransforms['Round'],
    'SQUARE': _TFTransforms['Square'],
    'SQRT': _TFTransforms['Sqrt'],
    'RSQRT': _TFTransforms['Rsqrt'],
    'ADD': _TFTransforms['Add'],
    'SUB': _TFTransforms['Sub'],
    'MUL': _TFTransforms['Mul'],
    'DIV': _TFTransforms['RealDiv'],
    'POW': _TFTransforms['Pow'],
    'MINIMUM': _TFTransforms['Minimum'],
    'MAXIMUM': _TFTransforms['Maximum'],
    'LOGICAL_AND': _TFTransforms['LogicalAnd'],
    'LOGICAL_OR': _TFTransforms['LogicalOr'],
    'LESS': _TFTransforms['Less'],
    'LESS_EQUAL': _TFTransforms['LessEqual'],
    'GREATER': _TFTransforms['Greater'],
    'GREATER_EQUAL': _TFTransforms['GreaterEqual'],
    'EQUAL': _TFTransforms['Equal'],
    'NOT_EQUAL': _TFTransforms['NotEqual'],
    'SELECT': _TFTransforms['Select'],
    'REDUCE_MIN': _TFTransforms['Min'],
    'REDUCE_MAX': _TFTransforms['Max'],
    'MEAN': _TFTransforms['Mean'],
    'SUM': _TFTransforms['Sum'],
    'REDUCE_ANY': _TFTransforms['Any'],
    'REDUCE_ALL': _TFTransforms['All'],
    'ARG_MIN': _TFTransforms['ArgMin'],
    'ARG_MAX': _TFTransforms['ArgMax'],
    'SOFTMAX': _TFTransforms['Softmax'],
    'LOCAL_RESPONSE_NORMALIZATION': _TFTransforms['LRN'],
    'RESIZE_NEAREST_NEIGHBOR': _TFTransforms['ResizeNearestNeighbor'],
    'RESIZE_BILINEAR': _TFTransforms['ResizeBilinear'],
})
