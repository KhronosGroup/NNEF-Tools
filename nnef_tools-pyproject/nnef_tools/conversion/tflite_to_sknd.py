from __future__ import division, print_function, absolute_import
from .converter import Converter as _Converter, Transform, ConversionError
from .tf_to_sknd import Converter as _TFConverter, _Transforms as _TFTransforms
from ..model import Tensor, Operation
from ..utils import types
from ..io.tf.lite import CustomOptionsKey
import numpy as np
import copy


_TFLITE_DETECTION_POSTPROCESS = """
operator tflite_detection_postprocess {
    @attrib {
        detections_per_class: int;
        max_classes_per_detection: int;
        max_detections: int;
        nms_iou_threshold: real; 
        nms_score_threshold: real; 
        num_classes: int; 
        use_regular_nms: bool; 
        h_scale: real;
        w_scale: real; 
        x_scale: real; 
        y_scale: real;
    }
    @input {
        boxes: real[];
        scores: real[];
        anchors: real[];
    }
    @output {
        detection_boxes: real[];
        detection_classes: real[]; 
        detection_scores: real[]; 
        num_detections: real[];
    }
}
"""

_TFLITE_RELU6 = """
operator relu6 {
    @input {
        x: real[s..];
    }
    @output {
        y: real[s..];
    }
    @compose {
        y = nn.relu{max=6.0}(x);
    }
}
"""


class Converter(_TFConverter):

    _ConvOpTypes = ['CONV_1D', 'CONV_2D', 'CONV_3D', 'TRANSPOSE_CONV', 'DEPTHWISE_CONV_2D']

    _ActivationOpTypes = {
        'ELU': 'nn.elu',
        'RELU': 'nn.relu',
        'RELU6': 'relu6',
        'LOGISTIC': 'nn.sigmoid',
        'TANH': 'math.tanh',
    }

    @staticmethod
    def defined_operations():
        return {
            'RELU6': _TFLITE_RELU6,
            'TFLite_Detection_PostProcess': _TFLITE_DETECTION_POSTPROCESS,
        }

    def __init__(self, custom_transforms=None, custom_functions=None, mirror_unsupported=False):
        _Converter.__init__(self, transforms=self.merge_transforms(_Transforms, custom_transforms),
                            functions=custom_functions, mirror_unsupported=mirror_unsupported)

    def __call__(self, model):
        model = _TFConverter.__call__(self, model)
        self._fix_custom_options(model)
        return model

    def _global_attribs(self):
        return {'_lite_': True}

    def _prepare(self, model):
        self._fix_quantization_attribs(model)
        self._fix_quantized_dtypes(model)

    def _is_constant(self, tensor):
        return tensor.producer is None and tensor.data is not None

    def _read_constant(self, tensor, type=None):
        if tensor.producer is None:
            return types.from_numpy(tensor.data, type=type)
        else:
            raise ConversionError('trying to evaluate non-constant tensor')

    @staticmethod
    def _is_zero(value):
        return np.all(value == 0) if isinstance(value, np.ndarray) else value == 0

    def _fix_quantized_dtypes(self, model):
        for graph in model.graphs:
            for tensor in graph.tensors:
                if tensor.quant:
                    scale = tensor.quant.get('scale')
                    if scale is not None and not self._is_zero(scale):
                        tensor.dtype = np.float32
                    else:
                        tensor.quant = None

    def _fix_quantization_attribs(self, model):
        dtype_bits = {
            np.int8: 8,
            np.uint8: 8,
            np.int16: 16,
            np.uint16: 16,
            np.int32: 32,
            np.uint32: 32,
            np.int64: 64,
            np.uint64: 64,
        }

        for graph in model.graphs:
            for tensor in graph.tensors:
                if tensor.quant:
                    scale = tensor.quant.get('scale')
                    zero_point = tensor.quant.get('zero_point')
                    if scale is not None and not self._is_zero(scale):
                        if 'min' in tensor.quant:
                            del tensor.quant['min']
                        if 'max' in tensor.quant:
                            del tensor.quant['max']
                        assert tensor.dtype == np.uint8 or tensor.dtype == np.int8 or \
                               tensor.dtype == np.uint16 or tensor.dtype == np.int16 or \
                               tensor.dtype == np.uint32 or tensor.dtype == np.int32, \
                            "unknown quantized dtype '{}'".format(tensor.dtype)
                        tensor.quant['op-type'] = 'quant.zero_point_linear_quantize'
                        tensor.quant['bits'] = 32 if self._is_conv_bias(tensor) else dtype_bits[tensor.dtype]
                        tensor.quant['signed'] = tensor.dtype == np.int8 or tensor.dtype == np.int16 or tensor.dtype == np.int32
                        tensor.quant['symmetric'] = self._is_conv_filter(tensor)

    def _fix_custom_options(self, model):
        for graph in model.graphs:
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

        input = Tensor(output.graph, dtype=output.dtype, shape=self._working_shape(output), quant=copy.deepcopy(output.quant))
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
            type='nn.conv',
            using={
                'depthwise': '!_type_ == "DEPTHWISE_CONV_2D"',
            },
            inputs=(
                '!I[0]',
                '!I[1]',
                '!I[2]',
            ),
            outputs='!activation(O[0], fused_activation_function)',
            attribs={
                'stride': '![stride_h, stride_w]',
                'dilation': '![dilation_h_factor, dilation_w_factor]',
                'padding': '!convert_padding(padding, I[0].rank - 2)',
                'groups': '!0 if depthwise else None',
                'data_format': 'NXC',
                'filter_format': '!"CXN" if depthwise else "NXC"',
                'ceil_mode': '!padding == "SAME"',
            }
        ),
    'TRANSPOSE_CONV':
        Transform(
            type='nn.deconv',
            using={
                'depthwise': False,
                'output_shape': '!arg_as_attrib(I[0])',
            },
            inputs=(
                '!I[2]',
                '!I[1]',
            ),
            outputs='!O[0]',
            attribs={
                'stride': '![stride_h, stride_w]',
                'padding': '!convert_padding(padding, I[0].rank - 2)',
                'output_size': '!output_shape[1:-1]',
                'groups': '!0 if depthwise else None',
                'data_format': 'NXC',
                'filter_format': '!"NXC" if depthwise else "CXN"',
            }
        ),
    ('MAX_POOL_2D', 'AVERAGE_POOL_2D'):
        Transform(
            type=('nn.max_pool', 'nn.avg_pool'),
            inputs=(
                '!I[0]',
            ),
            outputs=(
                '!O[0]',
            ),
            attribs={
                'axes': [1, 2],
                'size': '![filter_height, filter_width]',
                'stride': '![stride_h, stride_w]',
                'padding': '!convert_padding(padding, I[0].rank)',
                'ceil_mode': '!padding == "SAME"',
                'ignore_border': '!True if _type_ == "AVERAGE_POOL_2D" else None',
            }
        ),
    'RESHAPE':
        Transform(
            type='layout.reshape',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'shape': '!arg_as_attrib(I[1]) if len(I) > 1 else new_shape',
            }
        ),
    'CONCATENATION':
        Transform(
            type='layout.concat',
            inputs='!list(I)',
            outputs='!activation(O[0], fused_activation_function)',
            attribs={
                'axis': '!axis',
            }
        ),
    'FULLY_CONNECTED':
        Transform(
            type='nn.linear',
            cond={
                '!weights_format == "DEFAULT"': 'wights_format must be "DEFAULT"',
            },
            inputs=(
                '!I[0] if keep_num_dims else flatten(I[0])',
                '!I[1]',
                '!I[2] if len(I) > 2 else None',
            ),
            outputs='!activation(O[0], fused_activation_function)',
        ),
    'BATCH_MATMUL':
        Transform(
            type='linalg.matmul',
            cond={
                '!asymmetric_quantize_inputs == False': 'asymmetric_quantize_inputs must be False',
            },
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            attribs={
                'transA': '!adj_x',
                'transB': '!adj_y',
            },
        ),
    'L2_NORMALIZATION':
        Transform(
            type='nn.l2_norm',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': [-1],
            }
        ),
    'PRELU':
        Transform(
            type='nn.prelu',
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            attribs={
                'axis': -1,
            },
        ),
    ('PAD', 'MIRROR_PAD'):
        Transform(
            type='layout.pad',
            cond={
                '!mode < 2': 'mode must be 0 or 1',
            },
            defaults={
                'mode': None,
            },
            using={
                'paddings': '!arg_as_attrib(I[1])',
                'before': '![p for p, q in paddings]',
                'after': '![q for p, q in paddings]',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'padding': '!before + after',
                'method': '!"REFLECT" if mode == 0 else "SYMMETRIC" if mode == 1 else "CONSTANT"',
            }
        ),
    'GATHER':
        Transform(
            type='layout.gather',
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            attribs={
                'axis': '!axis',
            },
        ),
    'IDENTITY': _TFTransforms['Identity'],
    'QUANTIZE': _TFTransforms['Identity'],
    'TRANSPOSE': _TFTransforms['Transpose'],
    'SPLIT': _TFTransforms['Split'],
    'SPLIT_V': _TFTransforms['SplitV'],
    'PACK': _TFTransforms['Pack'],
    'UNPACK': _TFTransforms['Unpack'],
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
    'SIN': _TFTransforms['Sin'],
    'COS': _TFTransforms['Cos'],
    'TAN': _TFTransforms['Tan'],
    'ASIN': _TFTransforms['Asin'],
    'ACOS': _TFTransforms['Acos'],
    'ATAN': _TFTransforms['Atan'],
    'SINH': _TFTransforms['Sinh'],
    'COSH': _TFTransforms['Cosh'],
    'TANH': _TFTransforms['Tanh'],
    'ASINH': _TFTransforms['Asinh'],
    'ACOSH': _TFTransforms['Acosh'],
    'ATANH': _TFTransforms['Atanh'],
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
    'ADD_N': _TFTransforms['AddN'],
    'CAST': _TFTransforms['Cast'],
})
