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

import typing
from functools import partial

import numpy as np

from nnef_tools.io.tensorflow.tf_graph import *
from nnef_tools.shape_inference import shape_inference as infer

_ConstValueByTensorT = typing.Dict[TFTensor, np.ndarray]


def get_op_t(op):
    return op.attribs['T'] if 'T' in op.attribs else op.inputs[0].dtype


def propagate_conv(op, const_value_by_tensor, depthwise):
    # type: (TFOperation, _ConstValueByTensorT, bool)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    input, filter = op.inputs
    format = infer.Format.NCHW if op.attribs["data_format"][1].upper() == "C" else infer.Format.NHWC
    return [infer.conv(
        input=input.shape,
        filter=filter.shape[:-2],
        padding=infer.Padding.SAME_UPPER if op.attribs["padding"].upper() == 'SAME' else infer.Padding.VALID,
        stride=infer.spatial(op.attribs["strides"], format),
        dilation=(infer.spatial(op.attribs["dilations"], format)
                  if 'dilations' in op.attribs
                  else infer.singleton(input.rank - 2)),
        groups=0 if depthwise else 1,
        output_channels=filter.shape[-2] * filter.shape[-1] if depthwise else filter.shape[-1],
        format=format,
    )], [op.attribs['T']]


def propagate_deconv(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    input_sizes, filter, out_backprop = op.inputs

    return [list(const_value_by_tensor[input_sizes].tolist())], [op.attribs['T']]


def propagate_broadcast(op, const_value_by_tensor, dtype=''):
    # type: (TFOperation, _ConstValueByTensorT, str)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    return ([infer.elementwise(inputs=[input.shape for input in op.inputs],
                               broadcast=infer.Broadcast.FROM_RIGHT)],
            [dtype if dtype else get_op_t(op)])


def propagate_first(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    return [infer.copy(op.inputs[0].shape)], [get_op_t(op)]


def propagate_pool(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    return [infer.sliding_window(
        input=op.input.shape,
        filter=op.attribs['ksize'],
        padding=infer.Padding.SAME_UPPER if op.attribs["padding"].upper() == 'SAME' else infer.Padding.VALID,
        stride=op.attribs['strides'],
        dilation=[1] * len(op.attribs["strides"]),
    )], [op.attribs['T']]


def propagate_max_pool_with_argmax(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    output_shape = infer.sliding_window(
        input=op.input.shape,
        filter=op.attribs['ksize'],
        padding=infer.Padding.SAME_UPPER if op.attribs["padding"].upper() == 'SAME' else infer.Padding.VALID,
        stride=op.attribs['strides'],
        dilation=[1] * len(op.attribs["strides"]),
    )

    return [output_shape, infer.copy(output_shape)], [op.attribs['T'], op.attribs['Targmax']]


def propagate_fused_batch_norm(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    format = (infer.Format.NCHW if "data_format" in op.attribs and op.attribs["data_format"][1].upper() == "C"
              else infer.Format.NHWC)
    input_shape = op.inputs[0].shape
    channel_shape = [input_shape[infer.channel_axis(format)]]
    return [infer.copy(input_shape),
            infer.copy(channel_shape),
            infer.copy(channel_shape),
            infer.copy(channel_shape),
            infer.copy(channel_shape)], [op.attribs['T']] * 5


def propagate_matmul(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    a, b = op.inputs
    return (
        [infer.matmul(a.shape, b.shape, transpose_a=op.attribs["transpose_a"], transpose_b=op.attribs["transpose_b"])],
        [op.attribs['T']]
    )


def propagate_pack(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    return [infer.stack(inputs=[input.shape for input in op.inputs], axis=op.attribs['axis'])], [op.attribs['T']]


def propagate_shape(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    return [[op.input.rank]], [op.attribs['T']]


def propagate_squeeze(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    axes = op.attribs['squeeze_dims']
    return [infer.squeeze(input=op.input.shape, axes=axes if axes else None)], [op.attribs['T']]


def propagate_reduce(op, const_value_by_tensor, dtype=None):
    # type: (TFOperation, _ConstValueByTensorT, str)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    input, axis = op.inputs
    axis = const_value_by_tensor[axis].tolist()  # type: typing.List[int]
    if not isinstance(axis, list):
        axis = [axis]
    return [infer.reduce(input=input.shape, axes=axis, squeeze=not op.attribs["keep_dims"])], \
           [op.attribs['T'] if not dtype else dtype]


def propagate_argminmax(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    input, axis = op.inputs
    axis = const_value_by_tensor[axis].tolist()  # type: int
    return [infer.reduce(input=input.shape, axes=[axis], squeeze=True)], [op.attribs['T']]


def propagate_expand_dims(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    input, axis = op.inputs
    axis = const_value_by_tensor[axis].item()
    return [infer.unsqueeze(input=input.shape, axes=[axis])], [op.attribs['T']]


def propagate_concat(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    inputs = op.inputs[:-1]
    axis = op.inputs[-1]
    axis = const_value_by_tensor[axis].item()
    return [infer.concat(inputs=[i.shape for i in inputs], axis=axis)], [op.attribs['T']]


def propagate_pad(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    input, paddings = op.inputs
    paddings = const_value_by_tensor[paddings].tolist()  # type: typing.List[typing.Tuple[int, int]]
    return [infer.pad(input=input.shape, padding=paddings)], [op.attribs['T']]


def propagate_reshape(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    input, shape = op.inputs
    shape = const_value_by_tensor[shape].tolist()  # type: typing.List[int]
    return [infer.reshape(input=input.shape, shape=shape)], [op.attribs['T']]


def propagate_resize(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    input, size = op.inputs
    size = const_value_by_tensor[size].tolist()  # type: typing.List[int]
    return [infer.resize(input=input.shape, size=size, format=infer.Format.NHWC)], [op.attribs['T']]


def propagate_transpose(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    input, perm = op.inputs
    perm = const_value_by_tensor[perm].tolist()  # type: typing.List[int]
    return [infer.transpose(input=input.shape, axes=perm)], [op.attribs['T']]


def propagate_split(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    axis, input = op.inputs
    axis = const_value_by_tensor[axis].item()
    return (infer.split(input=input.shape, axis=axis, num=op.attribs['num_split']),
            [op.attribs['T']] * op.attribs['num_split'])


def propagate_splitv(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    input, size_splits, axis = op.inputs
    axis = const_value_by_tensor[axis].item()
    return (infer.split(input=input.shape, axis=axis, num=op.attribs['num_split']),
            [op.attribs['T']] * op.attribs['num_split'])


def propagate_slice(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    input, begin, size = op.inputs
    begin = const_value_by_tensor[begin].tolist()  # type: typing.List[int]
    size = const_value_by_tensor[size].tolist()  # type: typing.List[int]
    return [infer.slice(input=input.shape, begin=begin, size=size, zero_means_all=True)], [op.attribs['T']]


def propagate_strided_slice(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    input, begin, end, strides = op.inputs
    begin = const_value_by_tensor[begin].tolist()  # type: typing.List[int]
    end = const_value_by_tensor[end].tolist()  # type: typing.List[int]
    strides = const_value_by_tensor[strides].tolist()  # type: typing.List[int]
    return [infer.strided_slice(input=input.shape,
                                begin=begin,
                                end=end,
                                stride=strides,
                                ellipsis_mask=op.attribs['ellipsis_mask'],
                                new_axis_mask=op.attribs['new_axis_mask'],
                                shrink_axis_mask=op.attribs['shrink_axis_mask'],
                                begin_mask=op.attribs['begin_mask'],
                                end_mask=op.attribs['end_mask'])], [op.attribs['T']]


def propagate_tile(op, const_value_by_tensor):
    # type: (TFOperation, _ConstValueByTensorT)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    input, multiples = op.inputs
    multiples = const_value_by_tensor[multiples].tolist()  # type: typing.List[int]

    return [infer.tile(input=input.shape,
                       repeat=multiples)], [op.attribs['T']]


_DefaultPropagators = {
    "Abs": propagate_first,
    "Add": propagate_broadcast,
    "BiasAdd": propagate_first,
    "Ceil": propagate_first,
    "Elu": propagate_first,
    "Equal": partial(propagate_broadcast, dtype='DT_BOOL'),
    "Exp": propagate_first,
    "Floor": propagate_first,
    "Greater": partial(propagate_broadcast, dtype='DT_BOOL'),
    "GreaterEqual": partial(propagate_broadcast, dtype='DT_BOOL'),
    "Identity": propagate_first,
    "LeakyRelu": propagate_first,
    "Less": partial(propagate_broadcast, dtype='DT_BOOL'),
    "LessEqual": partial(propagate_broadcast, dtype='DT_BOOL'),
    "Log": propagate_first,
    "LogicalAnd": propagate_broadcast,
    "LogicalNot": propagate_first,
    "LogicalOr": propagate_broadcast,
    "Maximum": propagate_broadcast,
    "Minimum": propagate_broadcast,
    "Mul": propagate_broadcast,
    "Neg": propagate_first,
    "NotEqual": partial(propagate_broadcast, dtype='DT_BOOL'),
    "Pow": propagate_first,
    "RealDiv": propagate_broadcast,
    "Relu": propagate_first,
    "Relu6": propagate_first,
    "Round": propagate_first,
    "Rsqrt": propagate_first,
    "Sigmoid": propagate_first,
    "Sign": propagate_first,
    "Softmax": propagate_first,
    "Softplus": propagate_first,
    "Softsign": propagate_first,
    "Sqrt": propagate_first,
    "Square": propagate_first,
    "Sub": propagate_broadcast,
    "Tanh": propagate_first,
    "Select": propagate_broadcast,
    "ClipByValue": propagate_first,
    "Sin": propagate_first,
    "Cos": propagate_first,

    # more complex:
    "AvgPool": propagate_pool,
    "Conv2D": partial(propagate_conv, depthwise=False),
    "Conv3D": partial(propagate_conv, depthwise=False),
    "Conv2DBackpropInput": propagate_deconv,
    "Conv3DBackpropInputV2": propagate_deconv,
    # "CudnnRNN": None,
    "DepthwiseConv2dNative": partial(propagate_conv, depthwise=True),
    "FusedBatchNorm": propagate_fused_batch_norm,
    "LRN": propagate_first,
    "MatMul": propagate_matmul,
    "MaxPool": propagate_pool,
    "MaxPoolWithArgmax": propagate_max_pool_with_argmax,
    "ArgMin": propagate_argminmax,
    "ArgMax": propagate_argminmax,
    "Pack": propagate_pack,
    "Shape": propagate_shape,
    "Squeeze": propagate_squeeze,

    # even more complex:
    "Min": propagate_reduce,
    "Max": propagate_reduce,
    "Sum": propagate_reduce,
    "Mean": propagate_reduce,
    "Any": partial(propagate_reduce, dtype="DT_BOOL"),
    "All": partial(propagate_reduce, dtype="DT_BOOL"),

    "ExpandDims": propagate_expand_dims,
    "ConcatV2": propagate_concat,
    "Pad": propagate_pad,
    "MirrorPad": propagate_pad,
    "Reshape": propagate_reshape,
    "ResizeArea": propagate_resize,
    "ResizeBilinear": propagate_resize,
    "ResizeNearestNeighbor": propagate_resize,
    "Transpose": propagate_transpose,
    "Split": propagate_split,
    "SplitV": propagate_splitv,

    "Slice": propagate_slice,
    "StridedSlice": propagate_strided_slice,
    "Tile": propagate_tile,
    "Cast": propagate_first,
}
