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

import re
import sys

import flatbuffers
import numpy as np

import nnef_tools.io.tensorflow.tflite_fb as tflite_fb
from nnef_tools.conversion.tensorflow import tflite_to_tf_py, tf_py_to_tflite
from nnef_tools.core import utils
from nnef_tools.io.tensorflow.tf_graph import *

# See this: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs

OUTPUT_FILE_IDENTIFIER = "TFL3"
OUTPUT_SCHEMA_VERSION = 3

_BuiltinOptionsClasses = [
    None,
    tflite_fb.Conv2DOptions,
    tflite_fb.DepthwiseConv2DOptions,
    tflite_fb.ConcatEmbeddingsOptions,
    tflite_fb.LSHProjectionOptions,
    tflite_fb.Pool2DOptions,
    tflite_fb.SVDFOptions,
    tflite_fb.RNNOptions,
    tflite_fb.FullyConnectedOptions,
    tflite_fb.SoftmaxOptions,
    tflite_fb.ConcatenationOptions,
    tflite_fb.AddOptions,
    tflite_fb.L2NormOptions,
    tflite_fb.LocalResponseNormalizationOptions,
    tflite_fb.LSTMOptions,
    tflite_fb.ResizeBilinearOptions,
    tflite_fb.CallOptions,
    tflite_fb.ReshapeOptions,
    tflite_fb.SkipGramOptions,
    tflite_fb.SpaceToDepthOptions,
    tflite_fb.EmbeddingLookupSparseOptions,
    tflite_fb.MulOptions,
    tflite_fb.PadOptions,
    tflite_fb.GatherOptions,
    tflite_fb.BatchToSpaceNDOptions,
    tflite_fb.SpaceToBatchNDOptions,
    tflite_fb.TransposeOptions,
    tflite_fb.ReducerOptions,
    tflite_fb.SubOptions,
    tflite_fb.DivOptions,
    tflite_fb.SqueezeOptions,
    tflite_fb.SequenceRNNOptions,
    tflite_fb.StridedSliceOptions,
    tflite_fb.ExpOptions,
    tflite_fb.TopKV2Options,
    tflite_fb.SplitOptions,
    tflite_fb.LogSoftmaxOptions,
    tflite_fb.CastOptions,
    tflite_fb.DequantizeOptions,
    tflite_fb.MaximumMinimumOptions,
    tflite_fb.ArgMaxOptions,
    tflite_fb.LessOptions,
    tflite_fb.NegOptions,
    tflite_fb.PadV2Options,
    tflite_fb.GreaterOptions,
    tflite_fb.GreaterEqualOptions,
    tflite_fb.LessEqualOptions,
    tflite_fb.SelectOptions,
    tflite_fb.SliceOptions,
    tflite_fb.TransposeConvOptions,
    tflite_fb.SparseToDenseOptions,
    tflite_fb.TileOptions,
    tflite_fb.ExpandDimsOptions,
    tflite_fb.EqualOptions,
    tflite_fb.NotEqualOptions,
    tflite_fb.ShapeOptions,
    tflite_fb.PowOptions,
    tflite_fb.ArgMinOptions,
    tflite_fb.FakeQuantOptions,
    tflite_fb.PackOptions,
    tflite_fb.LogicalOrOptions,
    tflite_fb.OneHotOptions,
    tflite_fb.LogicalAndOptions,
    tflite_fb.LogicalNotOptions,
    tflite_fb.UnpackOptions,
    tflite_fb.FloorDivOptions,
    tflite_fb.SquareOptions,
    tflite_fb.ZerosLikeOptions,
    tflite_fb.FillOptions,
    tflite_fb.BidirectionalSequenceLSTMOptions,
    tflite_fb.BidirectionalSequenceRNNOptions,
    tflite_fb.UnidirectionalSequenceLSTMOptions,
    tflite_fb.FloorModOptions,
    tflite_fb.RangeOptions,
    tflite_fb.ResizeNearestNeighborOptions,
    tflite_fb.LeakyReluOptions,
    tflite_fb.SquaredDifferenceOptions,
    tflite_fb.MirrorPadOptions,
    tflite_fb.AbsOptions,
    tflite_fb.SplitVOptions,
    tflite_fb.UniqueOptions,
    tflite_fb.ReverseV2Options,
    tflite_fb.AddNOptions,
    tflite_fb.GatherNdOptions,
    tflite_fb.CosOptions,
    tflite_fb.WhereOptions,
    tflite_fb.RankOptions,
    tflite_fb.ReverseSequenceOptions,
    tflite_fb.MatrixDiagOptions,
    tflite_fb.QuantizeOptions,
    tflite_fb.MatrixSetDiagOptions,
]

_BuiltinOptionsByOperator = {
    tflite_fb.BuiltinOperator.ADD: tflite_fb.BuiltinOptions.AddOptions,
    tflite_fb.BuiltinOperator.AVERAGE_POOL_2D: tflite_fb.BuiltinOptions.Pool2DOptions,
    tflite_fb.BuiltinOperator.CONCATENATION: tflite_fb.BuiltinOptions.ConcatenationOptions,
    tflite_fb.BuiltinOperator.CONV_2D: tflite_fb.BuiltinOptions.Conv2DOptions,
    tflite_fb.BuiltinOperator.DEPTHWISE_CONV_2D: tflite_fb.BuiltinOptions.DepthwiseConv2DOptions,
    tflite_fb.BuiltinOperator.DEQUANTIZE: tflite_fb.BuiltinOptions.DequantizeOptions,
    tflite_fb.BuiltinOperator.EMBEDDING_LOOKUP: None,
    tflite_fb.BuiltinOperator.FLOOR: None,
    tflite_fb.BuiltinOperator.FULLY_CONNECTED: tflite_fb.BuiltinOptions.FullyConnectedOptions,
    tflite_fb.BuiltinOperator.HASHTABLE_LOOKUP: None,
    tflite_fb.BuiltinOperator.L2_NORMALIZATION: tflite_fb.BuiltinOptions.L2NormOptions,
    tflite_fb.BuiltinOperator.L2_POOL_2D: tflite_fb.BuiltinOptions.Pool2DOptions,
    tflite_fb.BuiltinOperator.LOCAL_RESPONSE_NORMALIZATION: tflite_fb.BuiltinOptions.LocalResponseNormalizationOptions,
    tflite_fb.BuiltinOperator.LOGISTIC: None,
    tflite_fb.BuiltinOperator.LSH_PROJECTION: None,
    tflite_fb.BuiltinOperator.LSTM: tflite_fb.BuiltinOptions.LSTMOptions,
    tflite_fb.BuiltinOperator.MAX_POOL_2D: tflite_fb.BuiltinOptions.Pool2DOptions,
    tflite_fb.BuiltinOperator.MUL: tflite_fb.BuiltinOptions.MulOptions,
    tflite_fb.BuiltinOperator.RELU: None,
    tflite_fb.BuiltinOperator.RELU_N1_TO_1: None,
    tflite_fb.BuiltinOperator.RELU6: None,
    tflite_fb.BuiltinOperator.RESHAPE: tflite_fb.BuiltinOptions.ReshapeOptions,
    tflite_fb.BuiltinOperator.RESIZE_BILINEAR: tflite_fb.BuiltinOptions.ResizeBilinearOptions,
    tflite_fb.BuiltinOperator.RNN: tflite_fb.BuiltinOptions.RNNOptions,
    tflite_fb.BuiltinOperator.SOFTMAX: tflite_fb.BuiltinOptions.SoftmaxOptions,
    tflite_fb.BuiltinOperator.SPACE_TO_DEPTH: tflite_fb.BuiltinOptions.SpaceToDepthOptions,
    tflite_fb.BuiltinOperator.SVDF: tflite_fb.BuiltinOptions.SVDFOptions,
    tflite_fb.BuiltinOperator.TANH: None,
    tflite_fb.BuiltinOperator.CONCAT_EMBEDDINGS: tflite_fb.BuiltinOptions.ConcatEmbeddingsOptions,
    tflite_fb.BuiltinOperator.SKIP_GRAM: tflite_fb.BuiltinOptions.SkipGramOptions,
    tflite_fb.BuiltinOperator.CALL: tflite_fb.BuiltinOptions.CallOptions,
    tflite_fb.BuiltinOperator.CUSTOM: None,
    tflite_fb.BuiltinOperator.EMBEDDING_LOOKUP_SPARSE: tflite_fb.BuiltinOptions.EmbeddingLookupSparseOptions,
    tflite_fb.BuiltinOperator.PAD: tflite_fb.BuiltinOptions.PadOptions,
    tflite_fb.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_RNN: None,
    tflite_fb.BuiltinOperator.GATHER: tflite_fb.BuiltinOptions.GatherOptions,
    tflite_fb.BuiltinOperator.BATCH_TO_SPACE_ND: tflite_fb.BuiltinOptions.BatchToSpaceNDOptions,
    tflite_fb.BuiltinOperator.SPACE_TO_BATCH_ND: tflite_fb.BuiltinOptions.SpaceToBatchNDOptions,
    tflite_fb.BuiltinOperator.TRANSPOSE: tflite_fb.BuiltinOptions.TransposeOptions,
    tflite_fb.BuiltinOperator.MEAN: tflite_fb.BuiltinOptions.ReducerOptions,
    tflite_fb.BuiltinOperator.SUB: tflite_fb.BuiltinOptions.SubOptions,
    tflite_fb.BuiltinOperator.DIV: tflite_fb.BuiltinOptions.DivOptions,
    tflite_fb.BuiltinOperator.SQUEEZE: tflite_fb.BuiltinOptions.SqueezeOptions,
    tflite_fb.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM: tflite_fb.BuiltinOptions.UnidirectionalSequenceLSTMOptions,
    tflite_fb.BuiltinOperator.STRIDED_SLICE: tflite_fb.BuiltinOptions.StridedSliceOptions,
    tflite_fb.BuiltinOperator.BIDIRECTIONAL_SEQUENCE_RNN: tflite_fb.BuiltinOptions.BidirectionalSequenceRNNOptions,
    tflite_fb.BuiltinOperator.EXP: tflite_fb.BuiltinOptions.ExpOptions,
    tflite_fb.BuiltinOperator.TOPK_V2: tflite_fb.BuiltinOptions.TopKV2Options,
    tflite_fb.BuiltinOperator.SPLIT: tflite_fb.BuiltinOptions.SplitOptions,
    tflite_fb.BuiltinOperator.LOG_SOFTMAX: tflite_fb.BuiltinOptions.LogSoftmaxOptions,
    tflite_fb.BuiltinOperator.DELEGATE: None,
    tflite_fb.BuiltinOperator.BIDIRECTIONAL_SEQUENCE_LSTM: tflite_fb.BuiltinOptions.BidirectionalSequenceLSTMOptions,
    tflite_fb.BuiltinOperator.CAST: tflite_fb.BuiltinOptions.CastOptions,
    tflite_fb.BuiltinOperator.PRELU: None,
    tflite_fb.BuiltinOperator.MAXIMUM: tflite_fb.BuiltinOptions.MaximumMinimumOptions,
    tflite_fb.BuiltinOperator.ARG_MAX: tflite_fb.BuiltinOptions.ArgMaxOptions,
    tflite_fb.BuiltinOperator.MINIMUM: tflite_fb.BuiltinOptions.MaximumMinimumOptions,
    tflite_fb.BuiltinOperator.LESS: tflite_fb.BuiltinOptions.LessOptions,
    tflite_fb.BuiltinOperator.NEG: tflite_fb.BuiltinOptions.NegOptions,
    tflite_fb.BuiltinOperator.PADV2: tflite_fb.BuiltinOptions.PadV2Options,
    tflite_fb.BuiltinOperator.GREATER: tflite_fb.BuiltinOptions.GreaterOptions,
    tflite_fb.BuiltinOperator.GREATER_EQUAL: tflite_fb.BuiltinOptions.GreaterEqualOptions,
    tflite_fb.BuiltinOperator.LESS_EQUAL: tflite_fb.BuiltinOptions.LessEqualOptions,
    tflite_fb.BuiltinOperator.SELECT: tflite_fb.BuiltinOptions.SelectOptions,
    tflite_fb.BuiltinOperator.SLICE: tflite_fb.BuiltinOptions.SliceOptions,
    tflite_fb.BuiltinOperator.SIN: None,
    tflite_fb.BuiltinOperator.TRANSPOSE_CONV: tflite_fb.BuiltinOptions.TransposeConvOptions,
    tflite_fb.BuiltinOperator.SPARSE_TO_DENSE: tflite_fb.BuiltinOptions.SparseToDenseOptions,
    tflite_fb.BuiltinOperator.TILE: tflite_fb.BuiltinOptions.TileOptions,
    tflite_fb.BuiltinOperator.EXPAND_DIMS: tflite_fb.BuiltinOptions.ExpandDimsOptions,
    tflite_fb.BuiltinOperator.EQUAL: tflite_fb.BuiltinOptions.EqualOptions,
    tflite_fb.BuiltinOperator.NOT_EQUAL: tflite_fb.BuiltinOptions.NotEqualOptions,
    tflite_fb.BuiltinOperator.LOG: None,
    tflite_fb.BuiltinOperator.SUM: tflite_fb.BuiltinOptions.ReducerOptions,
    tflite_fb.BuiltinOperator.SQRT: None,
    tflite_fb.BuiltinOperator.RSQRT: None,
    tflite_fb.BuiltinOperator.SHAPE: tflite_fb.BuiltinOptions.ShapeOptions,
    tflite_fb.BuiltinOperator.POW: tflite_fb.BuiltinOptions.PowOptions,
    tflite_fb.BuiltinOperator.ARG_MIN: tflite_fb.BuiltinOptions.ArgMinOptions,
    tflite_fb.BuiltinOperator.FAKE_QUANT: tflite_fb.BuiltinOptions.FakeQuantOptions,
    tflite_fb.BuiltinOperator.REDUCE_PROD: tflite_fb.BuiltinOptions.ReducerOptions,
    tflite_fb.BuiltinOperator.REDUCE_MAX: tflite_fb.BuiltinOptions.ReducerOptions,
    tflite_fb.BuiltinOperator.PACK: tflite_fb.BuiltinOptions.PackOptions,
    tflite_fb.BuiltinOperator.LOGICAL_OR: tflite_fb.BuiltinOptions.LogicalOrOptions,
    tflite_fb.BuiltinOperator.ONE_HOT: tflite_fb.BuiltinOptions.OneHotOptions,
    tflite_fb.BuiltinOperator.LOGICAL_AND: tflite_fb.BuiltinOptions.LogicalAndOptions,
    tflite_fb.BuiltinOperator.LOGICAL_NOT: tflite_fb.BuiltinOptions.LogicalNotOptions,
    tflite_fb.BuiltinOperator.UNPACK: tflite_fb.BuiltinOptions.UnpackOptions,
    tflite_fb.BuiltinOperator.REDUCE_MIN: tflite_fb.BuiltinOptions.ReducerOptions,
    tflite_fb.BuiltinOperator.FLOOR_DIV: tflite_fb.BuiltinOptions.FloorDivOptions,
    tflite_fb.BuiltinOperator.REDUCE_ANY: tflite_fb.BuiltinOptions.ReducerOptions,
    tflite_fb.BuiltinOperator.SQUARE: tflite_fb.BuiltinOptions.SquareOptions,
    tflite_fb.BuiltinOperator.ZEROS_LIKE: tflite_fb.BuiltinOptions.ZerosLikeOptions,
    tflite_fb.BuiltinOperator.FILL: tflite_fb.BuiltinOptions.FillOptions,
    tflite_fb.BuiltinOperator.FLOOR_MOD: tflite_fb.BuiltinOptions.FloorModOptions,
    tflite_fb.BuiltinOperator.RANGE: tflite_fb.BuiltinOptions.RangeOptions,
    tflite_fb.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR: tflite_fb.BuiltinOptions.ResizeNearestNeighborOptions,
    tflite_fb.BuiltinOperator.LEAKY_RELU: tflite_fb.BuiltinOptions.LeakyReluOptions,
    tflite_fb.BuiltinOperator.SQUARED_DIFFERENCE: tflite_fb.BuiltinOptions.SquaredDifferenceOptions,
    tflite_fb.BuiltinOperator.MIRROR_PAD: tflite_fb.BuiltinOptions.MirrorPadOptions,
    tflite_fb.BuiltinOperator.ABS: tflite_fb.BuiltinOptions.AbsOptions,
    tflite_fb.BuiltinOperator.SPLIT_V: tflite_fb.BuiltinOptions.SplitVOptions,
    tflite_fb.BuiltinOperator.UNIQUE: tflite_fb.BuiltinOptions.UniqueOptions,
    tflite_fb.BuiltinOperator.CEIL: None,
    tflite_fb.BuiltinOperator.REVERSE_V2: tflite_fb.BuiltinOptions.ReverseV2Options,
    tflite_fb.BuiltinOperator.ADD_N: tflite_fb.BuiltinOptions.AddNOptions,
    tflite_fb.BuiltinOperator.GATHER_ND: tflite_fb.BuiltinOptions.GatherNdOptions,
    tflite_fb.BuiltinOperator.COS: tflite_fb.BuiltinOptions.CosOptions,
    tflite_fb.BuiltinOperator.WHERE: tflite_fb.BuiltinOptions.WhereOptions,
    tflite_fb.BuiltinOperator.RANK: tflite_fb.BuiltinOptions.RankOptions,
    tflite_fb.BuiltinOperator.ELU: None,
    tflite_fb.BuiltinOperator.REVERSE_SEQUENCE: tflite_fb.BuiltinOptions.ReverseSequenceOptions,
    tflite_fb.BuiltinOperator.MATRIX_DIAG: tflite_fb.BuiltinOptions.MatrixDiagOptions,
    tflite_fb.BuiltinOperator.QUANTIZE: tflite_fb.BuiltinOptions.QuantizeOptions,
    tflite_fb.BuiltinOperator.MATRIX_SET_DIAG: tflite_fb.BuiltinOptions.MatrixSetDiagOptions,
}


def _enumerate_options_getters(optionsClass):
    return {_camel_to_snake(name): func for name, func in optionsClass.__dict__.items()
            if not name.startswith('_')
            and name != 'Init' and not name.startswith('GetRootAs')
            and not name.endswith('AsNumpy') and not name.endswith('Length')}


def _enumerate_options_length_getters(optionsClass):
    return {_camel_to_snake(name[:-6]): func for name, func in optionsClass.__dict__.items()
            if not name.startswith('_')
            and not name.startswith('GetRootAs') and name.endswith('Length')}


def _enumerate_options_adders(optionsClass):
    className = optionsClass.__name__
    prefix = className + 'Add'
    optionsModule = sys.modules[optionsClass.__module__]
    return {_camel_to_snake(name[len(prefix):]): func for name, func in optionsModule.__dict__.items()
            if name.startswith(prefix)}


def _enumerate_options_vector_starters(optionsClass):
    className = optionsClass.__name__
    prefix, suffix = className + 'Start', 'Vector'
    optionsModule = sys.modules[optionsClass.__module__]
    return {_camel_to_snake(name[len(prefix):-len(suffix)]): func for name, func in optionsModule.__dict__.items()
            if name.startswith(prefix) and name.endswith(suffix)}


def _get_options_starter_ender(optionsClass):
    className = optionsClass.__name__
    optionsModule = sys.modules[optionsClass.__module__]
    moduleDict = optionsModule.__dict__
    return moduleDict[className + 'Start'], moduleDict[className + 'End']


def _enumerate_attributes(optionsClass, optionsObject):
    getters = _enumerate_options_getters(optionsClass)
    length_getters = _enumerate_options_length_getters(optionsClass)

    attribs = {}
    for name, getter in getters.items():
        length_getter = length_getters.get(name)

        value = getter(optionsObject) if length_getter is None else \
            [getter(optionsObject, i) for i in range(length_getter(optionsObject))]

        attribs[name] = _substitute_enum_value_with_name(name, value, optionsClass)

    return attribs


def _substitute_enum_value_with_name(key, value, optionsClass):
    cls, map = _OptionEnumNameByValueMaps.get(key, (None, None))
    return map[value] if map is not None and (cls is None or cls == optionsClass) else value


def _substitute_enum_name_with_value(key, name, optionsClass):
    cls, map = _OptionEnumValueByNameMaps.get(key, (None, None))
    return map[name] if map is not None and (cls is None or cls == optionsClass) else name


def _generate_enum_value_by_name(enumClass):
    return {name: value for name, value in enumClass.__dict__.items() if not name.startswith('_')}


def _generate_enum_name_by_value(enumClass):
    return {value: name for name, value in enumClass.__dict__.items() if not name.startswith('_')}


_OptionEnumNameByValueMaps = {
    'padding': (None, _generate_enum_name_by_value(tflite_fb.Padding)),
    'fused_activation_function': (None, _generate_enum_name_by_value(tflite_fb.ActivationFunctionType)),
    'weights_format': (
        tflite_fb.FullyConnectedOptions, _generate_enum_name_by_value(tflite_fb.FullyConnectedOptionsWeightsFormat)),
    'type': (tflite_fb.LSHProjectionOptions, _generate_enum_name_by_value(tflite_fb.LSHProjectionType)),
    'kernel_type': (tflite_fb.LSTMOptions, _generate_enum_name_by_value(tflite_fb.LSTMKernelType)),
    'combiner': (tflite_fb.EmbeddingLookupSparseOptions, _generate_enum_name_by_value(tflite_fb.CombinerType)),
}

_OptionEnumValueByNameMaps = {
    'padding': (None, _generate_enum_value_by_name(tflite_fb.Padding)),
    'fused_activation_function': (None, _generate_enum_value_by_name(tflite_fb.ActivationFunctionType)),
    'weights_format': (
        tflite_fb.FullyConnectedOptions, _generate_enum_value_by_name(tflite_fb.FullyConnectedOptionsWeightsFormat)),
    'type': (tflite_fb.LSHProjectionOptions, _generate_enum_value_by_name(tflite_fb.LSHProjectionType)),
    'kernel_type': (tflite_fb.LSTMOptions, _generate_enum_value_by_name(tflite_fb.LSTMKernelType)),
    'combiner': (tflite_fb.EmbeddingLookupSparseOptions, _generate_enum_value_by_name(tflite_fb.CombinerType)),
}

_TensorTypeNameByValue = _generate_enum_name_by_value(tflite_fb.TensorType)
_TensorTypeValueByName = _generate_enum_value_by_name(tflite_fb.TensorType)

_BuiltinOperatorNameByValue = _generate_enum_name_by_value(tflite_fb.BuiltinOperator)
_BuiltinOperatorValueByName = _generate_enum_value_by_name(tflite_fb.BuiltinOperator)

_regex1 = re.compile('(.)([A-Z][a-z]+)')
_regex2 = re.compile('([a-z0-9])([A-Z])')


def _camel_to_snake(s):
    subbed = _regex1.sub(r'\1_\2', s)
    return _regex2.sub(r'\1_\2', subbed).lower()


def _snake_to_camel(s):
    return ''.join(c for c in s.title() if c != '_')


def _get_quantization(tensor):
    quant = tensor.Quantization()

    if quant.MinLength() == 0:
        min = None
    elif quant.MinLength() == 1:
        min = float(quant.Min(0))
    else:
        min = quant.MinAsNumpy()

    if quant.MaxLength() == 0:
        max = None
    elif quant.MaxLength() == 1:
        max = float(quant.Max(0))
    else:
        max = quant.MaxAsNumpy()

    if quant.ScaleLength() == 0:
        scale = None
    elif quant.ScaleLength() == 1:
        scale = float(quant.Scale(0))
    else:
        scale = quant.ScaleAsNumpy()

    if quant.ZeroPointLength() == 0:
        zero_point = None
    elif quant.ZeroPointLength() == 1:
        zero_point = int(quant.ZeroPoint(0))
    else:
        zero_point = quant.ZeroPointAsNumpy()

    if all(x is None for x in [min, max, scale, zero_point]):
        return None
    else:
        return TFTensor.Quantization(min, max, scale, zero_point)


def _get_data_as_ndarray(buffer, dtype, shape):
    return buffer.DataAsNumpy().view(dtype).reshape(shape) if buffer.DataLength() != 0 else None


_TensorDtypeAsNumpy = [
    np.float32,
    np.float16,
    np.int32,
    np.uint8,
    np.int64,
    np.str,
    np.bool,
    np.int16,
    np.complex64,
]

_NumpyDtypeAsTFLite = {
    np.float32: tflite_fb.TensorType.FLOAT32,
    np.float16: tflite_fb.TensorType.FLOAT16,
    np.int32: tflite_fb.TensorType.INT32,
    np.uint8: tflite_fb.TensorType.UINT8,
    np.int64: tflite_fb.TensorType.INT64,
    np.str: tflite_fb.TensorType.STRING,
    np.bool: tflite_fb.TensorType.BOOL,
    np.int16: tflite_fb.TensorType.INT16,
    np.complex64: tflite_fb.TensorType.COMPLEX64,
}


def _CreateNumpyVector(builder, x):
    if not isinstance(x, np.ndarray):
        raise TypeError("non-numpy-ndarray passed to CreateNumpyVector")

    if x.dtype.kind not in ['b', 'i', 'u', 'f']:
        raise TypeError("numpy-ndarray holds elements of unsupported datatype")

    if x.ndim > 1:
        raise TypeError("multidimensional-ndarray passed to CreateNumpyVector")

    builder.StartVector(x.itemsize, x.size, x.dtype.alignment)

    # Ensure little endian byte ordering
    if x.dtype.str[0] != "<":
        x = x.byteswap()

    length = x.itemsize * x.size
    builder.head -= length
    builder.Bytes[builder.head: builder.head + length] = x.tobytes()

    return builder.EndVector(x.size)


def _build_buffer(builder, bytes):
    data = _CreateNumpyVector(builder, bytes)
    tflite_fb.BufferStart(builder)
    tflite_fb.BufferAddData(builder, data)
    return tflite_fb.BufferEnd(builder)


def _build_tensor(builder, tensor, buffer_index):
    name = builder.CreateString(tensor.name)
    type = _TensorTypeValueByName[tensor.dtype]

    tflite_fb.TensorStartShapeVector(builder, len(tensor.shape))
    for s in reversed(tensor.shape):
        builder.PrependInt32(s)
    shape = builder.EndVector(len(tensor.shape))

    buffer = buffer_index if tensor.data is not None else 0

    quant = _build_quantization(builder, tensor.quantization, tensor.dtype)

    tflite_fb.TensorStart(builder)
    tflite_fb.TensorAddName(builder, name)
    tflite_fb.TensorAddShape(builder, shape)
    tflite_fb.TensorAddType(builder, type)
    tflite_fb.TensorAddBuffer(builder, buffer)
    if quant is not None:
        tflite_fb.TensorAddQuantization(builder, quant)
    return tflite_fb.TensorEnd(builder)


def _ensure_numpy_array(x, dtype):
    if isinstance(x, np.ndarray):
        assert x.dtype == dtype
        return x
    else:
        return np.array(x, dtype=dtype)


def _build_quantization(builder, quant, dtype):
    if quant is None or quant.all_zero():
        return None

    min = _CreateNumpyVector(builder, _ensure_numpy_array(quant.min, dtype=np.float32))
    max = _CreateNumpyVector(builder, _ensure_numpy_array(quant.max, dtype=np.float32))
    scale = _CreateNumpyVector(builder, _ensure_numpy_array(quant.scale, dtype=np.float32))
    zero_point = _CreateNumpyVector(builder, _ensure_numpy_array(quant.zero_point, dtype=np.int64))

    if dtype == "INT32":
        tflite_fb.QuantizationParametersStart(builder)
        tflite_fb.QuantizationParametersAddScale(builder, scale)
        return tflite_fb.QuantizationParametersEnd(builder)
    else:
        tflite_fb.QuantizationParametersStart(builder)
        tflite_fb.QuantizationParametersAddMin(builder, min)
        tflite_fb.QuantizationParametersAddMax(builder, max)
        tflite_fb.QuantizationParametersAddScale(builder, scale)
        tflite_fb.QuantizationParametersAddZeroPoint(builder, zero_point)
        return tflite_fb.QuantizationParametersEnd(builder)


def _build_operator_code(builder, op_name):
    tflite_fb.OperatorCodeStart(builder)
    tflite_fb.OperatorCodeAddBuiltinCode(builder, _BuiltinOperatorValueByName[op_name])
    return tflite_fb.OperatorCodeEnd(builder)


def _build_operator_options(builder, attribs, optionsClass):
    starter, ender = _get_options_starter_ender(optionsClass)
    adders = _enumerate_options_adders(optionsClass)
    vector_starters = _enumerate_options_vector_starters(optionsClass)

    vector_values = {}
    for name, vector_starter in vector_starters.items():
        value = attribs[name]
        assert isinstance(value, list) and (len(value) == 0 or isinstance(value[0], int))
        vector_starter(builder, len(value))
        for i in reversed(value):
            builder.PrependInt32(i)
        vector_values[name] = builder.EndVector(len(value))

    starter(builder)
    for name, adder in adders.items():
        if name == 'fused_activation_function' and name not in attribs:
            value = 'NONE'
        else:
            value = attribs[name]
        value = vector_values.get(name, value)
        value = _substitute_enum_name_with_value(name, value, optionsClass)

        adder(builder, value)

    return ender(builder)


def _build_operator(builder, operation, op_code_index, tensor_index):
    inputs = [tensor_index[tensor] for tensor in operation.inputs]
    tflite_fb.OperatorStartInputsVector(builder, len(inputs))
    for input in reversed(inputs):
        builder.PrependInt32(input)
    inputs = builder.EndVector(len(inputs))

    outputs = [tensor_index[tensor] for tensor in operation.outputs]
    tflite_fb.OperatorStartOutputsVector(builder, len(outputs))
    for output in reversed(outputs):
        builder.PrependInt32(output)
    outputs = builder.EndVector(len(outputs))

    attribs = {name: value for name, value in operation.attribs.items()}

    optionsType = _BuiltinOptionsByOperator[_BuiltinOperatorValueByName[operation.name]]

    if optionsType is None:
        optionsType = 0

    optionsClass = _BuiltinOptionsClasses[optionsType]

    if optionsClass is not None:
        options = _build_operator_options(builder, attribs, optionsClass)
    else:
        options = None

    tflite_fb.OperatorStart(builder)
    tflite_fb.OperatorAddOpcodeIndex(builder, op_code_index[operation.name])
    tflite_fb.OperatorAddInputs(builder, inputs)
    tflite_fb.OperatorAddOutputs(builder, outputs)
    tflite_fb.OperatorAddBuiltinOptionsType(builder, optionsType)

    if options:
        tflite_fb.OperatorAddBuiltinOptions(builder, options)

    return tflite_fb.OperatorEnd(builder)


def read_tflite_graph_from_flatbuffers(filename):
    with open(filename, 'rb') as file:
        bytes = bytearray(file.read())

    model = tflite_fb.Model.GetRootAsModel(bytes, 0)

    if model.SubgraphsLength() != 1:
        raise NotImplementedError('graphs with multiple sub-graphs are not supported')

    subgraph = model.Subgraphs(0)
    name = subgraph.Name()

    graph = TFGraph(name.decode() if name is not None else None)

    tensors = []
    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        name = tensor.Name().decode()
        shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]
        dtype = _TensorTypeNameByValue[tensor.Type()]
        buffer = model.Buffers(tensor.Buffer())
        data = _get_data_as_ndarray(buffer, _TensorDtypeAsNumpy[tensor.Type()], shape)
        quant = _get_quantization(tensor)
        label = name if data is not None else None
        tensors.append(TFTensor(graph,
                                utils.anystr_to_str(name),
                                shape,
                                dtype,
                                data,
                                utils.anystr_to_str(label) if label is not None else None,
                                quant))

    for i in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(i)
        operatorCode = model.OperatorCodes(operator.OpcodeIndex())
        name = _BuiltinOperatorNameByValue[operatorCode.BuiltinCode()]

        options = operator.BuiltinOptions()
        optionsClass = _BuiltinOptionsClasses[operator.BuiltinOptionsType()]

        inputs = [tensors[operator.Inputs(i)] for i in range(operator.InputsLength()) if operator.Inputs(i) != -1]
        outputs = [tensors[operator.Outputs(i)] for i in range(operator.OutputsLength()) if operator.Outputs(i) != -1]

        if optionsClass is not None:
            optionsObject = optionsClass()
            optionsObject.Init(options.Bytes, options.Pos)
            attribs = _enumerate_attributes(optionsClass, optionsObject)
        else:
            attribs = {}

        if operatorCode.BuiltinCode() == tflite_fb.BuiltinOperator.CUSTOM:
            assert tflite_to_tf_py._custom_op_type_key not in attribs, \
                "'{}' shall not be set as an attribute".format(tflite_to_tf_py._custom_op_type_key)
            attribs[tflite_to_tf_py._custom_op_type_key] = operatorCode.CustomCode().decode('ascii')
        TFOperation(graph, name, inputs, outputs, attribs)

    inputs = []
    for i in range(subgraph.InputsLength()):
        tensor_index = subgraph.Inputs(i)
        inputs.append(tensors[tensor_index])

    outputs = []
    for i in range(subgraph.OutputsLength()):
        tensor_index = subgraph.Outputs(i)
        outputs.append(tensors[tensor_index])

    graph.inputs = inputs
    graph.outputs = outputs

    return graph


# https://github.com/google/flatbuffers/issues/4814
def FinishWithFileIdentifier(builder, rootTable, fid):
    from flatbuffers import number_types as N
    from flatbuffers import encode

    if fid is None or len(fid) != 4:
        raise Exception('fid must be 4 chars')

    flags = N.Uint8Flags
    prepSize = 4
    builder.Prep(builder.minalign, prepSize + len(fid))
    for i in range(3, -1, -1):
        builder.head = builder.head - flags.bytewidth
        encode.Write(flags.packer_type, builder.Bytes, builder.Head(), ord(fid[i]))

    return builder.Finish(rootTable)


def write_tflite_graph_to_flatbuffers(graph, filename):
    graph.sort()
    builder = flatbuffers.Builder(0)

    tflite_fb.BufferStartDataVector(builder, 0)
    data = builder.EndVector(0)
    tflite_fb.BufferStart(builder)
    tflite_fb.BufferAddData(builder, data)
    buffer = tflite_fb.BufferEnd(builder)

    buffers = [buffer]
    for tensor in graph.tensors:
        if tensor.data is not None:
            tensor_data = tensor.data
            if isinstance(tensor_data, (list, tuple)):
                tensor_data = np.array(tensor_data, dtype=_TensorDtypeAsNumpy[_TensorTypeValueByName[tensor.dtype]])
            bytes = tensor_data.reshape([-1]).view(np.uint8)
            buffers.append(_build_buffer(builder, bytes))

    tflite_fb.ModelStartBuffersVector(builder, len(buffers))
    for buffer in reversed(buffers):
        builder.PrependUOffsetTRelative(buffer)
    buffers = builder.EndVector(len(buffers))

    buffer_index = 1

    tensors = []
    tensor_index = {}
    for tensor in graph.tensors:
        tensor_index[tensor] = len(tensors)
        tensors.append(_build_tensor(builder, tensor, buffer_index))
        if tensor.data is not None:
            buffer_index += 1

    tflite_fb.SubGraphStartTensorsVector(builder, len(tensors))
    for tensor in reversed(tensors):
        builder.PrependUOffsetTRelative(tensor)
    tensors = builder.EndVector(len(tensors))

    op_codes = []
    op_code_index = {}
    for operation in graph.operations:
        if operation.name not in op_code_index:
            op_code_index[operation.name] = len(op_codes)
            op_codes.append(_build_operator_code(builder, operation.name))

    tflite_fb.ModelStartOperatorCodesVector(builder, len(op_codes))
    for op_code in reversed(op_codes):
        builder.PrependUOffsetTRelative(op_code)
    op_codes = builder.EndVector(len(op_codes))

    operators = []
    for operation in graph.operations:
        operators.append(_build_operator(builder, operation, op_code_index, tensor_index))

    tflite_fb.SubGraphStartOperatorsVector(builder, len(operators))
    for operator in reversed(operators):
        builder.PrependUOffsetTRelative(operator)
    operators = builder.EndVector(len(operators))

    name = builder.CreateString(graph.name) if graph.name is not None else None

    inputs = graph.inputs
    tflite_fb.SubGraphStartInputsVector(builder, len(inputs))
    for input in reversed(inputs):
        builder.PrependInt32(tensor_index[input])
    inputs = builder.EndVector(len(inputs))

    outputs = graph.outputs
    tflite_fb.SubGraphStartInputsVector(builder, len(outputs))
    for output in reversed(outputs):
        builder.PrependInt32(tensor_index[output])
    outputs = builder.EndVector(len(outputs))

    tflite_fb.SubGraphStart(builder)
    if name is not None:
        tflite_fb.SubGraphAddName(builder, name)
    tflite_fb.SubGraphAddTensors(builder, tensors)
    tflite_fb.SubGraphAddOperators(builder, operators)
    tflite_fb.SubGraphAddInputs(builder, inputs)
    tflite_fb.SubGraphAddOutputs(builder, outputs)
    subgraph = tflite_fb.SubGraphEnd(builder)

    tflite_fb.ModelStartSubgraphsVector(builder, 1)
    builder.PrependUOffsetTRelative(subgraph)
    subgraphs = builder.EndVector(1)

    tflite_fb.ModelStart(builder)
    tflite_fb.ModelAddVersion(builder, OUTPUT_SCHEMA_VERSION)
    tflite_fb.ModelAddBuffers(builder, buffers)
    tflite_fb.ModelAddOperatorCodes(builder, op_codes)
    tflite_fb.ModelAddSubgraphs(builder, subgraphs)
    model = tflite_fb.ModelEnd(builder)

    FinishWithFileIdentifier(builder, model, OUTPUT_FILE_IDENTIFIER)

    bytes = builder.Output()

    with open(filename, 'wb') as file:
        file.write(bytes)


class Reader(object):

    def __init__(self, convert_to_tf_py=False):
        self._convert_to_tf_py = convert_to_tf_py

    def __call__(self, filename):
        g = read_tflite_graph_from_flatbuffers(filename)

        if self._convert_to_tf_py:
            tflite_to_tf_py.convert(g)
        return g


class Writer(object):

    def __init__(self, convert_from_tf_py=False):
        self._convert_from_tf_py = convert_from_tf_py

    def __call__(self, graph, filename):
        if self._convert_from_tf_py:
            tf_py_to_tflite.convert(graph)

        return write_tflite_graph_to_flatbuffers(graph, filename)
