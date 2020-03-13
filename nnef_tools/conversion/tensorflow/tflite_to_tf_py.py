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

from functools import partial

from nnef_tools.conversion.transforms import squeezed_shape
from nnef_tools.core import graph_utils, utils
from nnef_tools.io.tensorflow.tf_graph import *


def _to_tf_py_dtype(tflite_dtype):
    # type: (str)->str
    return tflite_dtype.lower()


def _to_tf_py_activation_function(tflite_activation_function):
    return {'RELU': 'tf.nn.relu',
            'RELU6': 'tf.nn.relu6'}[tflite_activation_function]


def convert(tf_graph, enable_default_conversion=False):
    # type: (TFGraph, bool)->None

    tf_graph.sort()

    for tensor in tf_graph.tensors:
        if tensor.name is not None and ':' not in tensor.name:
            # The tf-to-nnef converter distinguishes original and generated tensors based on the presence of the ':'
            tensor.name = tensor.name + ":0"
        tensor.dtype = _to_tf_py_dtype(tensor.dtype)

    for op in list(tf_graph.operations):
        # Conversion
        assert enable_default_conversion or op.name in _DefaultConverters, \
            "No tflite_to_tf_py converter for {}".format(op.name)

        act_fun = op.attribs.get('fused_activation_function', None)
        if act_fun == 'NONE':
            act_fun = None
        output = op.output if act_fun else None

        if op.name in _DefaultConverters:
            _DefaultConverters[op.name](op)

        if act_fun:
            assert act_fun in ["RELU", "RELU6"]
            last_new_op = output.producer
            last_new_op.outputs = (TFTensor(graph=op.graph, name=None, shape=list(output.shape), dtype=output.dtype),)
            TFOperation(graph=last_new_op.graph,
                        name=_to_tf_py_activation_function(act_fun),
                        inputs=last_new_op.output,
                        outputs=output)

    graph_utils.remove_unreachable(tf_graph)


def convert_conv2d(op):
    # type: (TFOperation)->None

    op.name = "tf.nn.conv2d"
    op.attribs = dict(strides=[1, op.attribs['stride_h'], op.attribs['stride_w'], 1],
                      padding=op.attribs['padding'],
                      dilations=[1, op.attribs['dilation_h_factor'], op.attribs['dilation_w_factor'], 1],
                      data_format="NHWC")
    filter = op.inputs[1]
    filter.shape = filter.shape[1:] + [filter.shape[0]]
    axes = list(range(1, filter.rank)) + [0]
    filter.data = filter.data.transpose(tuple(axes))
    if len(op.inputs) == 3:
        bias = op.inputs[2]
        output = op.output
        op.inputs = (op.inputs[0], op.inputs[1])
        op.outputs = (TFTensor(graph=op.graph, name=None, shape=list(output.shape), dtype=output.dtype),)
        TFOperation(graph=op.graph,
                    name='tf.nn.bias_add',
                    attribs=dict(data_format="NHWC"),
                    inputs=(op.output, bias),
                    outputs=output)


def convert_conv2d_transpose(op):
    # type: (TFOperation)->None

    op.name = "tf.nn.conv2d_transpose"
    assert op.inputs[0].data is not None, "TRANSPOSE_CONV is only supported with constant output_shape"
    output_shape = op.inputs[0].data.tolist()
    filter = op.inputs[1]
    filter.shape = filter.shape[1:-1] + [filter.shape[0], filter.shape[-1]]
    axes = list(range(1, filter.rank - 1)) + [0, -1]
    filter.data = filter.data.transpose(tuple(axes))
    input = op.inputs[2]

    op.attribs = dict(strides=[1, op.attribs['stride_h'], op.attribs['stride_w'], 1],
                      padding=op.attribs['padding'],
                      output_shape=output_shape,
                      data_format="NHWC")
    op.inputs = (input, filter)


def convert_depthwise_conv2d(op):
    # type: (TFOperation)->None
    assert op.attribs['dilation_h_factor'] == op.attribs['dilation_w_factor']
    op.name = "tf.nn.depthwise_conv2d"
    op.attribs = dict(strides=[1, op.attribs['stride_h'], op.attribs['stride_w'], 1],
                      padding=op.attribs['padding'],
                      rate=op.attribs['dilation_h_factor'],
                      data_format="NHWC")
    input = op.inputs[0]
    filter = op.inputs[1]
    assert filter.shape[-1] % input.shape[-1] == 0
    filter.shape = filter.shape[1:-1] + [input.shape[-1], filter.shape[-1] // input.shape[-1]]
    filter.data = filter.data.reshape(filter.shape)
    if len(op.inputs) == 3:
        bias = op.inputs[2]
        output = op.output
        op.inputs = (op.inputs[0], op.inputs[1])
        op.outputs = (TFTensor(graph=op.graph, name=None, shape=list(output.shape), dtype=output.dtype),)
        TFOperation(graph=op.graph,
                    name='tf.nn.bias_add',
                    attribs=dict(data_format="NHWC"),
                    inputs=(op.output, bias),
                    outputs=output)


def generic_convert_pool_2d(op, target_name):
    # type: (TFOperation, str)->None
    op.name = target_name
    op.attribs = dict(strides=[1, op.attribs['stride_h'], op.attribs['stride_w'], 1],
                      padding=op.attribs['padding'],
                      ksize=[1, op.attribs['filter_height'], op.attribs['filter_width'], 1],
                      data_format="NHWC")


def convert_reshape(op):
    # type: (TFOperation)->None
    op.name = "tf.reshape"
    op.inputs = (op.inputs[0],)
    op.attribs = dict(shape=op.attribs['new_shape'])


def convert_softmax(op):
    # type: (TFOperation)->None
    assert op.attribs["beta"] == 1.0

    op.name = "tf.nn.softmax"
    op.attribs = dict(axis=None)  # last axis


def convert_squeeze(op):
    # type: (TFOperation)->None

    op.name = "tf.squeeze"
    op.attribs = dict(axis=op.attribs['squeeze_dims'])


def generic_convert_reduce(op, target_name):
    # type: (TFOperation, str)->None

    op.name = target_name
    assert op.inputs[1].data is not None, "{} is only supported with constant axes".format(op.name)
    axes = utils.listify(op.inputs[1].data.tolist())
    op.inputs = (op.inputs[0],)
    op.attribs = dict(axis=axes, keepdims=op.attribs["keep_dims"])


def convert_fully_connected(op):
    # type: (TFOperation)->None
    assert op.attribs["weights_format"] == "DEFAULT"
    op.name = "tf.matmul"
    op.attribs = dict(transpose_a=False, transpose_b=True, adjoint_a=False, adjoint_b=False)
    if len(op.inputs) == 3:
        bias = op.inputs[2]
        output = op.output
        op.inputs = (op.inputs[0], op.inputs[1])
        if op.inputs[0].rank == 4:
            a = op.inputs[0]
            squeezed_a = TFTensor(graph=op.graph, name=None, shape=list([a.shape[0], a.shape[-1]]), dtype=a.dtype)
            TFOperation(graph=op.graph, name="tf.squeeze", inputs=a, attribs=dict(axis=[1, 2]), outputs=squeezed_a)
            op.inputs = (squeezed_a, op.inputs[1])
        op.outputs = (TFTensor(graph=op.graph, name=None, shape=list(output.shape), dtype=output.dtype),)
        TFOperation(graph=op.graph,
                    name='tf.nn.bias_add',
                    attribs=dict(data_format="NHWC"),
                    inputs=(op.output, bias),
                    outputs=output)


def convert_pad(op):
    # type: (TFOperation)->None
    op.name = "tf.pad"
    assert op.inputs[1].data is not None, "PAD is only supported with constant padding"
    paddings = op.inputs[1].data.tolist()
    op.inputs = (op.inputs[0],)
    op.attribs = dict(paddings=paddings, mode="CONSTANT", constant_values=0)


def convert_split(op):
    # type: (TFOperation)->None
    op.name = "tf.split"
    assert op.inputs[0].data is not None, "SPLIT is only supported with constant split_dim (inputs[0])"
    op.attribs = dict(num_or_size_splits=op.attribs['num_splits'], axis=op.inputs[0].data.tolist(), num=None)
    op.inputs = (op.inputs[1],)


def convert_split_v(op):
    # type: (TFOperation)->None
    op.name = "tf.split"
    input, size_splits, axis = op.inputs
    assert size_splits.data is not None, "SPLIT_V is only supported with constant size_splits (inputs[1])"
    assert axis.data is not None, "SPLIT_V is only supported with constant size_splits (inputs[2])"
    op.attribs = dict(num_or_size_splits=size_splits.data.tolist(), axis=axis.data.tolist(), num=None)
    op.inputs = (input,)


def convert_strided_slice(op):
    # type: (TFOperation)->None
    op.name = "tf.strided_slice"

    input, begin, end, strides = op.inputs
    assert all(t.data is not None for t in [begin, end, strides]), \
        "STRIDED_SLICE is only supported for constant begin, end, strides"

    op.inputs = (input,)
    op.attribs = dict(begin=begin.data.tolist(),
                      end=end.data.tolist(),
                      strides=strides.data.tolist(),
                      begin_mask=op.attribs['begin_mask'],
                      end_mask=op.attribs['end_mask'],
                      ellipsis_mask=op.attribs['ellipsis_mask'],
                      new_axis_mask=op.attribs['new_axis_mask'],
                      shrink_axis_mask=op.attribs['shrink_axis_mask'])


def convert_slice(op):
    # type: (TFOperation)->None
    op.name = "tf.slice"

    input, begin, size = op.inputs
    assert all(t.data is not None for t in [begin, size]), \
        "SLICE is only supported for constant begin, size"

    op.inputs = (input,)
    op.attribs = dict(begin=begin.data.tolist(),
                      size=size.data.tolist())


def generic_convert_argminmax(op, target_name):
    # type: (TFOperation, str)->None
    tflite_to_tf_dtype = {
        4: 9,
        2: 3
    }
    op.name = target_name
    assert op.inputs[1].data is not None, "ARG_MIN/ARG_MAX is only supported with constant axis (inputs[1])"
    axis = op.inputs[1].data.tolist()
    op.attribs = dict(axis=axis,
                      output_type=tflite_to_tf_dtype[op.attribs["output_type"]])
    op.inputs = (op.inputs[0],)

    output_tensor = op.output
    op.outputs = (TFTensor(graph=op.graph,
                           shape=squeezed_shape(shape=output_tensor.shape, axes=[axis]),
                           dtype=output_tensor.dtype),)
    TFOperation(graph=op.graph,
                name="tf.expand_dims",
                inputs=op.output,
                outputs=output_tensor,
                attribs=dict(axis=axis))


def convert_fill(op):
    # type: (TFOperation)->None
    op.name = "tf.split"
    assert op.inputs[0].data is not None, "FILL is only supported with constant dims (inputs[0])"
    assert op.inputs[1].data is not None, "FILL is only supported with constant value (inputs[1])"
    op.attribs = dict(dims=op.inputs[0].data.tolist(), value=op.inputs[1].data.tolist())
    op.inputs = tuple()


def convert_l2_normalize(op):
    # type: (TFOperation)->None
    op.name = "tf.nn.l2_normalize"
    op.attribs = dict(axis=[-1], epsilon=1e-12)


def convert_lrn(op):
    # type: (TFOperation)->None
    op.name = "tf.nn.lrn"
    op.attribs = dict(depth_radius=op.attribs["radius"],
                      bias=op.attribs["bias"],
                      alpha=op.attribs["alpha"],
                      beta=op.attribs["beta"])


def convert_stack(op):
    # type: (TFOperation)->None
    op.name = "tf.stack"
    op.attribs = dict(axis=op.attribs["axis"])


def convert_resize_nearest_neighbor(op):
    # type: (TFOperation)->None
    op.name = "tf.image.resize_nearest_neighbor"
    assert op.inputs[1].data is not None, "RESIZE_NEAREST_NEIGHBOR is only supported with constant size"
    op.attribs = dict(size=op.inputs[1].data.tolist(), align_corners=op.attribs['align_corners'])
    op.inputs = (op.inputs[0],)


def convert_resize_bilinear(op):
    # type: (TFOperation)->None
    op.name = "tf.image.resize_bilinear"
    assert op.inputs[1].data is not None, "RESIZE_BILINEAR is only supported with constant size"
    op.attribs = dict(size=op.inputs[1].data.tolist(), align_corners=op.attribs['align_corners'])
    op.inputs = (op.inputs[0],)


def convert_transpose(op):
    # type: (TFOperation)->None
    op.name = "tf.transpose"
    assert op.inputs[1].data is not None, "TRANSPOSE is only supported with constant perm (inputs[1])"
    op.attribs = dict(perm=op.inputs[1].data.tolist())
    op.inputs = (op.inputs[0],)


def convert_tile(op):
    # type: (TFOperation)->None
    op.name = "tf.tile"
    assert op.inputs[1].data is not None, "TILE is only supported with constant repeats (inputs[1])"
    op.attribs = dict(multiples=op.inputs[1].data.tolist())
    op.inputs = (op.inputs[0],)


def rename(op, target_name):
    # type: (TFOperation, str)->None
    op.name = target_name


def UNSUPPORTED(op):
    # print(op)
    # for i in op.inputs:
    #     print(i)
    raise utils.NNEFToolsException('TFLITE to TF_PY: Unsupported op: {}'.format(op.name))


_DefaultConverters = {
    "ABS": partial(rename, target_name="tf.abs"),
    "ADD_N": partial(rename, target_name="tf.add_n"),
    "ADD": partial(rename, target_name="tf.add"),
    "ARG_MAX": partial(generic_convert_argminmax, target_name="tf.argmax"),
    "ARG_MIN": partial(generic_convert_argminmax, target_name="tf.argmin"),
    "AVERAGE_POOL_2D": partial(generic_convert_pool_2d, target_name="tf.nn.avg_pool"),
    "BATCH_TO_SPACE_ND": UNSUPPORTED,
    "BIDIRECTIONAL_SEQUENCE_LSTM": UNSUPPORTED,
    "BIDIRECTIONAL_SEQUENCE_RNN": UNSUPPORTED,
    "CALL": UNSUPPORTED,
    "CAST": UNSUPPORTED,
    "CEIL": partial(rename, target_name="tf.ceil"),
    "CONCAT_EMBEDDINGS": UNSUPPORTED,
    "CONCATENATION": partial(rename, target_name="tf.concat"),
    "CONV_2D": convert_conv2d,
    "COS": partial(rename, target_name="tf.cos"),
    "CUSTOM": UNSUPPORTED,
    "DELEGATE": UNSUPPORTED,
    "DEPTH_TO_SPACE": UNSUPPORTED,
    "DEPTHWISE_CONV_2D": convert_depthwise_conv2d,
    "DEQUANTIZE": UNSUPPORTED,
    "DIV": partial(rename, target_name="tf.divide"),
    "ELU": partial(rename, target_name="tf.nn.elu"),
    "EMBEDDING_LOOKUP_SPARSE": UNSUPPORTED,
    "EMBEDDING_LOOKUP": UNSUPPORTED,
    "EQUAL": partial(rename, target_name="tf.equal"),
    "EXPAND_DIMS": UNSUPPORTED,  # Not in tflite docs? (not generated)
    "EXP": partial(rename, target_name="tf.exp"),
    "FAKE_QUANT": UNSUPPORTED,
    "FILL": convert_fill,
    "FLOOR_DIV": UNSUPPORTED,
    "FLOOR_MOD": UNSUPPORTED,
    "FLOOR": partial(rename, target_name="tf.floor"),
    "FULLY_CONNECTED": convert_fully_connected,
    "GATHER_ND": UNSUPPORTED,
    "GATHER": UNSUPPORTED,
    "GREATER_EQUAL": partial(rename, target_name="tf.greater_equal"),
    "GREATER": partial(rename, target_name="tf.greater"),
    "HASHTABLE_LOOKUP": UNSUPPORTED,
    "L2_NORMALIZATION": convert_l2_normalize,
    "L2_POOL_2D": UNSUPPORTED,
    "LEAKY_RELU": partial(rename, target_name="tf.nn.leaky_relu"),
    "LESS_EQUAL": partial(rename, target_name="tf.less_equal"),
    "LESS": partial(rename, target_name="tf.less"),
    "LOCAL_RESPONSE_NORMALIZATION": convert_lrn,
    "LOGICAL_AND": partial(rename, target_name="tf.logical_and"),
    "LOGICAL_NOT": partial(rename, target_name="tf.logical_not"),
    "LOGICAL_OR": partial(rename, target_name="tf.logical_or"),
    "LOGISTIC": partial(rename, target_name="tf.nn.sigmoid"),
    "LOG": partial(rename, target_name="tf.log"),
    "LOG_SOFTMAX": UNSUPPORTED,
    "LSH_PROJECTION": UNSUPPORTED,
    "LSTM": UNSUPPORTED,
    "MATRIX_DIAG": UNSUPPORTED,
    "MATRIX_SET_DIAG": UNSUPPORTED,
    "MAXIMUM": partial(rename, target_name="tf.maximum"),
    "MAX_POOL_2D": partial(generic_convert_pool_2d, target_name="tf.nn.max_pool"),
    "MEAN": partial(generic_convert_reduce, target_name="tf.reduce_mean"),
    "MINIMUM": partial(rename, target_name="tf.minimum"),
    "MIRROR_PAD": UNSUPPORTED,
    "MUL": partial(rename, target_name="tf.multiply"),
    "NEG": partial(rename, target_name="tf.negative"),
    "NOT_EQUAL": partial(rename, target_name="tf.not_equal"),
    "ONE_HOT": UNSUPPORTED,
    "PACK": convert_stack,
    "PAD": convert_pad,
    "PADV2": UNSUPPORTED,
    "POW": partial(rename, target_name="tf.pow"),
    "PRELU": UNSUPPORTED,  # Not in tflite docs? (not generated)
    "QUANTIZE": UNSUPPORTED,
    "RANGE": UNSUPPORTED,
    "RANK": UNSUPPORTED,
    "REDUCE_ANY": UNSUPPORTED,
    "REDUCE_MAX": partial(generic_convert_reduce, target_name="tf.reduce_max"),
    "REDUCE_MIN": partial(generic_convert_reduce, target_name="tf.reduce_min"),
    "REDUCE_PROD": UNSUPPORTED,
    "RELU6": partial(rename, target_name="tf.nn.relu6"),
    "RELU_N1_TO_1": UNSUPPORTED,
    "RELU": partial(rename, target_name="tf.nn.relu"),
    "RESHAPE": convert_reshape,
    "RESIZE_BILINEAR": convert_resize_bilinear,
    "RESIZE_NEAREST_NEIGHBOR": convert_resize_nearest_neighbor,
    "REVERSE_SEQUENCE": UNSUPPORTED,
    "REVERSE_V2": UNSUPPORTED,
    "RNN": UNSUPPORTED,
    "RSQRT": UNSUPPORTED,
    "SELECT": partial(rename, target_name="tf.where"),
    "SHAPE": UNSUPPORTED,
    "SIN": partial(rename, target_name="tf.sin"),
    "SKIP_GRAM": UNSUPPORTED,
    "SLICE": convert_slice,
    "SOFTMAX": convert_softmax,
    "SPACE_TO_BATCH_ND": UNSUPPORTED,
    "SPACE_TO_DEPTH": UNSUPPORTED,
    "SPARSE_TO_DENSE": UNSUPPORTED,
    "SPLIT": convert_split,
    "SPLIT_V": convert_split_v,
    "SQRT": partial(rename, target_name="tf.sqrt"),
    "SQUARED_DIFFERENCE": UNSUPPORTED,
    "SQUARE": partial(rename, target_name="tf.square"),
    "SQUEEZE": convert_squeeze,
    "STRIDED_SLICE": convert_strided_slice,
    "SUB": partial(rename, target_name="tf.subtract"),
    "SUM": partial(generic_convert_reduce, target_name="tf.reduce_sum"),
    "TANH": UNSUPPORTED,  # Not ready for use
    "TILE": convert_tile,
    "TOPK_V2": UNSUPPORTED,
    "TRANSPOSE_CONV": convert_conv2d_transpose,
    "TRANSPOSE": convert_transpose,
    "UNIDIRECTIONAL_SEQUENCE_LSTM": UNSUPPORTED,
    "UNIDIRECTIONAL_SEQUENCE_RNN": UNSUPPORTED,
    "UNIQUE": UNSUPPORTED,
    "UNPACK": partial(rename, target_name="tf.unstack"),
    "VDF": UNSUPPORTED,
    "WHERE": UNSUPPORTED,
    "ZEROS_LIKE": UNSUPPORTED,
}
