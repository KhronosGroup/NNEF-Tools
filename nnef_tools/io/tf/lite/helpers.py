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

from . import flatbuffers as fb
import numpy as np
import sys
import re

# See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs


OUTPUT_FILE_IDENTIFIER = "TFL3"
OUTPUT_SCHEMA_VERSION = 3

BuiltinOptionsClasses = [
    None,
    fb.Conv2DOptions,
    fb.DepthwiseConv2DOptions,
    fb.ConcatEmbeddingsOptions,
    fb.LSHProjectionOptions,
    fb.Pool2DOptions,
    fb.SVDFOptions,
    fb.RNNOptions,
    fb.FullyConnectedOptions,
    fb.SoftmaxOptions,
    fb.ConcatenationOptions,
    fb.AddOptions,
    fb.L2NormOptions,
    fb.LocalResponseNormalizationOptions,
    fb.LSTMOptions,
    fb.ResizeBilinearOptions,
    fb.CallOptions,
    fb.ReshapeOptions,
    fb.SkipGramOptions,
    fb.SpaceToDepthOptions,
    fb.EmbeddingLookupSparseOptions,
    fb.MulOptions,
    fb.PadOptions,
    fb.GatherOptions,
    fb.BatchToSpaceNDOptions,
    fb.SpaceToBatchNDOptions,
    fb.TransposeOptions,
    fb.ReducerOptions,
    fb.SubOptions,
    fb.DivOptions,
    fb.SqueezeOptions,
    fb.SequenceRNNOptions,
    fb.StridedSliceOptions,
    fb.ExpOptions,
    fb.TopKV2Options,
    fb.SplitOptions,
    fb.LogSoftmaxOptions,
    fb.CastOptions,
    fb.DequantizeOptions,
    fb.MaximumMinimumOptions,
    fb.ArgMaxOptions,
    fb.LessOptions,
    fb.NegOptions,
    fb.PadV2Options,
    fb.GreaterOptions,
    fb.GreaterEqualOptions,
    fb.LessEqualOptions,
    fb.SelectOptions,
    fb.SliceOptions,
    fb.TransposeConvOptions,
    fb.SparseToDenseOptions,
    fb.TileOptions,
    fb.ExpandDimsOptions,
    fb.EqualOptions,
    fb.NotEqualOptions,
    fb.ShapeOptions,
    fb.PowOptions,
    fb.ArgMinOptions,
    fb.FakeQuantOptions,
    fb.PackOptions,
    fb.LogicalOrOptions,
    fb.OneHotOptions,
    fb.LogicalAndOptions,
    fb.LogicalNotOptions,
    fb.UnpackOptions,
    fb.FloorDivOptions,
    fb.SquareOptions,
    fb.ZerosLikeOptions,
    fb.FillOptions,
    fb.BidirectionalSequenceLSTMOptions,
    fb.BidirectionalSequenceRNNOptions,
    fb.UnidirectionalSequenceLSTMOptions,
    fb.FloorModOptions,
    fb.RangeOptions,
    fb.ResizeNearestNeighborOptions,
    fb.LeakyReluOptions,
    fb.SquaredDifferenceOptions,
    fb.MirrorPadOptions,
    fb.AbsOptions,
    fb.SplitVOptions,
    fb.UniqueOptions,
    fb.ReverseV2Options,
    fb.AddNOptions,
    fb.GatherNdOptions,
    fb.CosOptions,
    fb.WhereOptions,
    fb.RankOptions,
    fb.ReverseSequenceOptions,
    fb.MatrixDiagOptions,
    fb.QuantizeOptions,
    fb.MatrixSetDiagOptions,
    fb.HardSwishOptions,
    fb.IfOptions,
    fb.WhileOptions,
    fb.DepthToSpaceOptions,
    fb.NonMaxSuppressionV4Options,
    fb.NonMaxSuppressionV5Options,
    fb.ScatterNdOptions,
    fb.SelectV2Options,
    fb.DensifyOptions,
    fb.SegmentSumOptions,
    fb.BatchMatMulOptions,
]

BuiltinOptionsByOperator = {
    fb.BuiltinOperator.ADD: fb.BuiltinOptions.AddOptions,
    fb.BuiltinOperator.AVERAGE_POOL_2D: fb.BuiltinOptions.Pool2DOptions,
    fb.BuiltinOperator.CONCATENATION: fb.BuiltinOptions.ConcatenationOptions,
    fb.BuiltinOperator.CONV_2D: fb.BuiltinOptions.Conv2DOptions,
    fb.BuiltinOperator.DEPTHWISE_CONV_2D: fb.BuiltinOptions.DepthwiseConv2DOptions,
    fb.BuiltinOperator.DEQUANTIZE: fb.BuiltinOptions.DequantizeOptions,
    fb.BuiltinOperator.EMBEDDING_LOOKUP: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.FLOOR: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.FULLY_CONNECTED: fb.BuiltinOptions.FullyConnectedOptions,
    fb.BuiltinOperator.HASHTABLE_LOOKUP: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.L2_NORMALIZATION: fb.BuiltinOptions.L2NormOptions,
    fb.BuiltinOperator.L2_POOL_2D: fb.BuiltinOptions.Pool2DOptions,
    fb.BuiltinOperator.LOCAL_RESPONSE_NORMALIZATION: fb.BuiltinOptions.LocalResponseNormalizationOptions,
    fb.BuiltinOperator.LOGISTIC: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.LSH_PROJECTION: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.LSTM: fb.BuiltinOptions.LSTMOptions,
    fb.BuiltinOperator.MAX_POOL_2D: fb.BuiltinOptions.Pool2DOptions,
    fb.BuiltinOperator.MUL: fb.BuiltinOptions.MulOptions,
    fb.BuiltinOperator.RELU: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.RELU_N1_TO_1: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.RELU6: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.RESHAPE: fb.BuiltinOptions.ReshapeOptions,
    fb.BuiltinOperator.RESIZE_BILINEAR: fb.BuiltinOptions.ResizeBilinearOptions,
    fb.BuiltinOperator.RNN: fb.BuiltinOptions.RNNOptions,
    fb.BuiltinOperator.SOFTMAX: fb.BuiltinOptions.SoftmaxOptions,
    fb.BuiltinOperator.SPACE_TO_DEPTH: fb.BuiltinOptions.SpaceToDepthOptions,
    fb.BuiltinOperator.SVDF: fb.BuiltinOptions.SVDFOptions,
    fb.BuiltinOperator.TANH: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.CONCAT_EMBEDDINGS: fb.BuiltinOptions.ConcatEmbeddingsOptions,
    fb.BuiltinOperator.SKIP_GRAM: fb.BuiltinOptions.SkipGramOptions,
    fb.BuiltinOperator.CALL: fb.BuiltinOptions.CallOptions,
    fb.BuiltinOperator.CUSTOM: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.EMBEDDING_LOOKUP_SPARSE: fb.BuiltinOptions.EmbeddingLookupSparseOptions,
    fb.BuiltinOperator.PAD: fb.BuiltinOptions.PadOptions,
    fb.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_RNN: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.GATHER: fb.BuiltinOptions.GatherOptions,
    fb.BuiltinOperator.BATCH_TO_SPACE_ND: fb.BuiltinOptions.BatchToSpaceNDOptions,
    fb.BuiltinOperator.SPACE_TO_BATCH_ND: fb.BuiltinOptions.SpaceToBatchNDOptions,
    fb.BuiltinOperator.TRANSPOSE: fb.BuiltinOptions.TransposeOptions,
    fb.BuiltinOperator.MEAN: fb.BuiltinOptions.ReducerOptions,
    fb.BuiltinOperator.SUB: fb.BuiltinOptions.SubOptions,
    fb.BuiltinOperator.DIV: fb.BuiltinOptions.DivOptions,
    fb.BuiltinOperator.SQUEEZE: fb.BuiltinOptions.SqueezeOptions,
    fb.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM: fb.BuiltinOptions.UnidirectionalSequenceLSTMOptions,
    fb.BuiltinOperator.STRIDED_SLICE: fb.BuiltinOptions.StridedSliceOptions,
    fb.BuiltinOperator.BIDIRECTIONAL_SEQUENCE_RNN: fb.BuiltinOptions.BidirectionalSequenceRNNOptions,
    fb.BuiltinOperator.EXP: fb.BuiltinOptions.ExpOptions,
    fb.BuiltinOperator.TOPK_V2: fb.BuiltinOptions.TopKV2Options,
    fb.BuiltinOperator.SPLIT: fb.BuiltinOptions.SplitOptions,
    fb.BuiltinOperator.LOG_SOFTMAX: fb.BuiltinOptions.LogSoftmaxOptions,
    fb.BuiltinOperator.DELEGATE: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.BIDIRECTIONAL_SEQUENCE_LSTM: fb.BuiltinOptions.BidirectionalSequenceLSTMOptions,
    fb.BuiltinOperator.CAST: fb.BuiltinOptions.CastOptions,
    fb.BuiltinOperator.PRELU: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.MAXIMUM: fb.BuiltinOptions.MaximumMinimumOptions,
    fb.BuiltinOperator.ARG_MAX: fb.BuiltinOptions.ArgMaxOptions,
    fb.BuiltinOperator.MINIMUM: fb.BuiltinOptions.MaximumMinimumOptions,
    fb.BuiltinOperator.LESS: fb.BuiltinOptions.LessOptions,
    fb.BuiltinOperator.NEG: fb.BuiltinOptions.NegOptions,
    fb.BuiltinOperator.PADV2: fb.BuiltinOptions.PadV2Options,
    fb.BuiltinOperator.GREATER: fb.BuiltinOptions.GreaterOptions,
    fb.BuiltinOperator.GREATER_EQUAL: fb.BuiltinOptions.GreaterEqualOptions,
    fb.BuiltinOperator.LESS_EQUAL: fb.BuiltinOptions.LessEqualOptions,
    fb.BuiltinOperator.SELECT: fb.BuiltinOptions.SelectOptions,
    fb.BuiltinOperator.SLICE: fb.BuiltinOptions.SliceOptions,
    fb.BuiltinOperator.SIN: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.TRANSPOSE_CONV: fb.BuiltinOptions.TransposeConvOptions,
    fb.BuiltinOperator.SPARSE_TO_DENSE: fb.BuiltinOptions.SparseToDenseOptions,
    fb.BuiltinOperator.TILE: fb.BuiltinOptions.TileOptions,
    fb.BuiltinOperator.EXPAND_DIMS: fb.BuiltinOptions.ExpandDimsOptions,
    fb.BuiltinOperator.EQUAL: fb.BuiltinOptions.EqualOptions,
    fb.BuiltinOperator.NOT_EQUAL: fb.BuiltinOptions.NotEqualOptions,
    fb.BuiltinOperator.LOG: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.SUM: fb.BuiltinOptions.ReducerOptions,
    fb.BuiltinOperator.SQRT: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.RSQRT: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.SHAPE: fb.BuiltinOptions.ShapeOptions,
    fb.BuiltinOperator.POW: fb.BuiltinOptions.PowOptions,
    fb.BuiltinOperator.ARG_MIN: fb.BuiltinOptions.ArgMinOptions,
    fb.BuiltinOperator.FAKE_QUANT: fb.BuiltinOptions.FakeQuantOptions,
    fb.BuiltinOperator.REDUCE_PROD: fb.BuiltinOptions.ReducerOptions,
    fb.BuiltinOperator.REDUCE_MAX: fb.BuiltinOptions.ReducerOptions,
    fb.BuiltinOperator.PACK: fb.BuiltinOptions.PackOptions,
    fb.BuiltinOperator.LOGICAL_OR: fb.BuiltinOptions.LogicalOrOptions,
    fb.BuiltinOperator.ONE_HOT: fb.BuiltinOptions.OneHotOptions,
    fb.BuiltinOperator.LOGICAL_AND: fb.BuiltinOptions.LogicalAndOptions,
    fb.BuiltinOperator.LOGICAL_NOT: fb.BuiltinOptions.LogicalNotOptions,
    fb.BuiltinOperator.UNPACK: fb.BuiltinOptions.UnpackOptions,
    fb.BuiltinOperator.REDUCE_MIN: fb.BuiltinOptions.ReducerOptions,
    fb.BuiltinOperator.FLOOR_DIV: fb.BuiltinOptions.FloorDivOptions,
    fb.BuiltinOperator.REDUCE_ANY: fb.BuiltinOptions.ReducerOptions,
    fb.BuiltinOperator.SQUARE: fb.BuiltinOptions.SquareOptions,
    fb.BuiltinOperator.ZEROS_LIKE: fb.BuiltinOptions.ZerosLikeOptions,
    fb.BuiltinOperator.FILL: fb.BuiltinOptions.FillOptions,
    fb.BuiltinOperator.FLOOR_MOD: fb.BuiltinOptions.FloorModOptions,
    fb.BuiltinOperator.RANGE: fb.BuiltinOptions.RangeOptions,
    fb.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR: fb.BuiltinOptions.ResizeNearestNeighborOptions,
    fb.BuiltinOperator.LEAKY_RELU: fb.BuiltinOptions.LeakyReluOptions,
    fb.BuiltinOperator.SQUARED_DIFFERENCE: fb.BuiltinOptions.SquaredDifferenceOptions,
    fb.BuiltinOperator.MIRROR_PAD: fb.BuiltinOptions.MirrorPadOptions,
    fb.BuiltinOperator.ABS: fb.BuiltinOptions.AbsOptions,
    fb.BuiltinOperator.SPLIT_V: fb.BuiltinOptions.SplitVOptions,
    fb.BuiltinOperator.UNIQUE: fb.BuiltinOptions.UniqueOptions,
    fb.BuiltinOperator.CEIL: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.REVERSE_V2: fb.BuiltinOptions.ReverseV2Options,
    fb.BuiltinOperator.ADD_N: fb.BuiltinOptions.AddNOptions,
    fb.BuiltinOperator.GATHER_ND: fb.BuiltinOptions.GatherNdOptions,
    fb.BuiltinOperator.COS: fb.BuiltinOptions.CosOptions,
    fb.BuiltinOperator.WHERE: fb.BuiltinOptions.WhereOptions,
    fb.BuiltinOperator.RANK: fb.BuiltinOptions.RankOptions,
    fb.BuiltinOperator.ELU: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.REVERSE_SEQUENCE: fb.BuiltinOptions.ReverseSequenceOptions,
    fb.BuiltinOperator.MATRIX_DIAG: fb.BuiltinOptions.MatrixDiagOptions,
    fb.BuiltinOperator.QUANTIZE: fb.BuiltinOptions.QuantizeOptions,
    fb.BuiltinOperator.MATRIX_SET_DIAG: fb.BuiltinOptions.MatrixSetDiagOptions,
    fb.BuiltinOperator.HARD_SWISH: fb.HardSwishOptions,
    fb.BuiltinOperator.IF: fb.IfOptions,
    fb.BuiltinOperator.WHILE: fb.WhileOptions,
    fb.BuiltinOperator.DEPTH_TO_SPACE: fb.DepthToSpaceOptions,
    fb.BuiltinOperator.NON_MAX_SUPPRESSION_V4: fb.NonMaxSuppressionV4Options,
    fb.BuiltinOperator.NON_MAX_SUPPRESSION_V5: fb.NonMaxSuppressionV5Options,
    fb.BuiltinOperator.SCATTER_ND: fb.ScatterNdOptions,
    fb.BuiltinOperator.ROUND: fb.BuiltinOptions.NONE,
    fb.BuiltinOperator.SELECT_V2: fb.BuiltinOptions.SelectV2Options,
    fb.BuiltinOperator.DENSIFY: fb.BuiltinOptions.DensifyOptions,
    fb.BuiltinOperator.SEGMENT_SUM: fb.BuiltinOptions.SegmentSumOptions,
    fb.BuiltinOperator.BATCH_MATMUL: fb.BuiltinOptions.BatchMatMulOptions,
}

CustomOptionsKey = 'custom_options'

DtypeToNumpy = {
    fb.TensorType.FLOAT16: np.float16,
    fb.TensorType.FLOAT32: np.float32,
    fb.TensorType.INT8: np.int8,
    fb.TensorType.INT16: np.int16,
    fb.TensorType.INT32: np.int32,
    fb.TensorType.INT64: np.int64,
    fb.TensorType.UINT8: np.uint8,
    fb.TensorType.STRING: np.str_,
    fb.TensorType.BOOL: np.bool_,
    fb.TensorType.COMPLEX64: np.complex64,
}

DtypeFromNumpy = {
    np.float16: fb.TensorType.FLOAT16,
    np.float32: fb.TensorType.FLOAT32,
    np.int8: fb.TensorType.INT8,
    np.int16: fb.TensorType.INT16,
    np.int32: fb.TensorType.INT32,
    np.int64: fb.TensorType.INT64,
    np.uint8: fb.TensorType.UINT8,
    np.str_: fb.TensorType.STRING,
    np.bool_: fb.TensorType.BOOL,
    np.complex64: fb.TensorType.COMPLEX64,
}


_regex1 = re.compile('(.)([A-Z][a-z]+)')
_regex2 = re.compile('([a-z0-9])([A-Z])')


def camel_to_snake(s):
    subbed = _regex1.sub(r'\1_\2', s)
    return _regex2.sub(r'\1_\2', subbed).lower()


def snake_to_camel(s):
    return ''.join(c for c in s.title() if c != '_')


def substitute_enum_value_with_name(key, value, optionsClass):
    cls, map = _OptionEnumNameByValueMaps.get(key, (None, None))
    return map[value] if map is not None and (cls is None or cls == optionsClass) else value


def substitute_enum_name_with_value(key, name, optionsClass):
    cls, map = _OptionEnumValueByNameMaps.get(key, (None, None))
    return map[name] if map is not None and (cls is None or cls == optionsClass) else name


def _generate_enum_value_by_name(enumClass):
    return {name: value for name, value in enumClass.__dict__.items() if not name.startswith('_')}


def _generate_enum_name_by_value(enumClass):
    return {value: name for name, value in enumClass.__dict__.items() if not name.startswith('_')}


def enumerate_options_getters(optionsClass):
    return {camel_to_snake(name): func for name, func in optionsClass.__dict__.items()
            if not name.startswith('_')
            and name != 'Init' and not name.startswith('GetRootAs')
            and not name.endswith('AsNumpy') and not name.endswith('Length')
            and not isinstance(func, classmethod)}


def enumerate_options_length_getters(optionsClass):
    return {camel_to_snake(name[:-6]): func for name, func in optionsClass.__dict__.items()
            if not name.startswith('_')
            and not name.startswith('GetRootAs') and name.endswith('Length')}


def enumerate_options_adders(optionsClass):
    className = optionsClass.__name__
    prefix = className + 'Add'
    optionsModule = sys.modules[optionsClass.__module__]
    return {camel_to_snake(name[len(prefix):]): func for name, func in optionsModule.__dict__.items()
            if name.startswith(prefix)}


def enumerate_options_vector_starters(optionsClass):
    className = optionsClass.__name__
    prefix, suffix = className + 'Start', 'Vector'
    optionsModule = sys.modules[optionsClass.__module__]
    return {camel_to_snake(name[len(prefix):-len(suffix)]): func for name, func in optionsModule.__dict__.items()
            if name.startswith(prefix) and name.endswith(suffix)}


def get_options_starter_ender(optionsClass):
    className = optionsClass.__name__
    optionsModule = sys.modules[optionsClass.__module__]
    moduleDict = optionsModule.__dict__
    return moduleDict[className + 'Start'], moduleDict[className + 'End']


_OptionEnumNameByValueMaps = {
    'padding': (None, _generate_enum_name_by_value(fb.Padding)),
    'fused_activation_function': (None, _generate_enum_name_by_value(fb.ActivationFunctionType)),
    'weights_format': (
        fb.FullyConnectedOptions, _generate_enum_name_by_value(fb.FullyConnectedOptionsWeightsFormat)),
    'type': (fb.LSHProjectionOptions, _generate_enum_name_by_value(fb.LSHProjectionType)),
    'kernel_type': (fb.LSTMOptions, _generate_enum_name_by_value(fb.LSTMKernelType)),
    'combiner': (fb.EmbeddingLookupSparseOptions, _generate_enum_name_by_value(fb.CombinerType)),
}

_OptionEnumValueByNameMaps = {
    'padding': (None, _generate_enum_value_by_name(fb.Padding)),
    'fused_activation_function': (None, _generate_enum_value_by_name(fb.ActivationFunctionType)),
    'weights_format': (
        fb.FullyConnectedOptions, _generate_enum_value_by_name(fb.FullyConnectedOptionsWeightsFormat)),
    'type': (fb.LSHProjectionOptions, _generate_enum_value_by_name(fb.LSHProjectionType)),
    'kernel_type': (fb.LSTMOptions, _generate_enum_value_by_name(fb.LSTMKernelType)),
    'combiner': (fb.EmbeddingLookupSparseOptions, _generate_enum_value_by_name(fb.CombinerType)),
}

BuiltinOperatorTypeByValue = _generate_enum_name_by_value(fb.BuiltinOperator)
BuiltinOperatorValueByType = _generate_enum_value_by_name(fb.BuiltinOperator)
