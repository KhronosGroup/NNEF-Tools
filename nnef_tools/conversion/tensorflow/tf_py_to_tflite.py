from __future__ import division, print_function, absolute_import

from functools import partial

import numpy as np

from nnef_tools.conversion.transforms import unsqueezed_shape
from nnef_tools.core import graph_utils
from nnef_tools.core import matcher
from nnef_tools.core import utils
from nnef_tools.io.tensorflow.tf_graph import *


def _to_tflite_dtype(tf_py_dtype):
    # type: (str)->str
    return tf_py_dtype.upper()


def convert(tf_graph, enable_default_conversion=False):
    # type: (TFGraph, bool)->None

    tf_graph.sort()

    transform_fuse_bias_add_to_conv(tf_graph)
    transform_fuse_add_to_matmul(tf_graph)
    transform_fuse_activations(tf_graph)

    for tensor in tf_graph.tensors:
        tensor.dtype = _to_tflite_dtype(tensor.dtype)

    for op in list(tf_graph.operations):
        # Conversion
        assert enable_default_conversion or op.name in _DefaultConverters, \
            "No tf_py_to_tflite converter for {}".format(op.name)

        if op.name in _DefaultConverters:
            _DefaultConverters[op.name](op)

    graph_utils.remove_unreachable(tf_graph)
    tf_graph.generate_missing_names()


def transform_fuse_bias_add_to_conv(tf_graph):
    # type: (TFGraph)->None

    conv_output = matcher.Tensor()
    conv = matcher.Operation(name=["tf.nn.conv2d", "tf.nn.depthwise_conv2d"], outputs=conv_output)
    add = matcher.Operation(name="tf.nn.bias_add", inputs={0: conv_output})

    matcher.replace(tf_graph,
                    add,
                    lambda m: TFOperation(graph=tf_graph,
                                          name=m[conv].name,
                                          attribs=m[conv].attribs,
                                          inputs=tuple(m[conv].inputs) + (m[add].inputs[1],),
                                          outputs=m[add].outputs),
                    lambda m: len(m[conv].inputs) == 2)


def transform_fuse_add_to_matmul(tf_graph):
    # type: (TFGraph)->None

    matmul_output, bias = matcher.tensors(2)
    matmul = matcher.Operation(name="tf.matmul", outputs=matmul_output)
    add = matcher.Operation(name="tf.add", inputs={matmul_output, bias})

    matcher.replace(tf_graph,
                    add,
                    lambda m: TFOperation(graph=tf_graph,
                                          name=m[matmul].name,
                                          attribs=m[matmul].attribs,
                                          inputs=tuple(m[matmul].inputs) + (m[bias],),
                                          outputs=m[add].outputs),
                    lambda m: len(m[matmul].inputs) == 2)

    # Seems not needed:
    # def replace(m):
    #     assert isinstance(m[bias].data, np.ndarray)
    #     assert m[bias].shape[0] == 1
    #     assert len(m[bias].consumers) == 1
    #     m[bias].shape = m[bias].shape[1:]
    #     m[bias].data = np.squeeze(m[bias].data, axis=0)
    #
    #     TFOperation(graph=tf_graph,
    #                 name=m[matmul].name,
    #                 attribs=m[matmul].attribs,
    #                 inputs=tuple(m[matmul].inputs) + (m[bias],),
    #                 outputs=m[add].outputs)
    #
    # matcher.replace(tf_graph, add, replace)


def transform_fuse_activations(tf_graph):
    # type: (TFGraph)->None

    fuse_to = [
        "tf.add",
        "tf.subtract",
        "tf.multiply",
        "tf.divide",
        "tf.nn.conv2d",
        "tf.nn.depthwise_conv2d",
        "tf.nn.max_pool",
        "tf.nn.avg_pool",
        # "tf.nn.conv2d_transpose", (not working yet)
        "tf.matmul",
        "tf.nn.l2_normalize",
        # "tf.concat" (not working yet)
    ]

    conv_output = matcher.Tensor()
    convlike = matcher.Operation(name=fuse_to, outputs=conv_output)
    activation = matcher.Operation(name="tf.nn.relu", inputs={0: conv_output})

    matcher.replace(tf_graph,
                    activation,
                    lambda m: TFOperation(graph=tf_graph,
                                          name=m[convlike].name,
                                          attribs=utils.dict_union(m[convlike].attribs,
                                                                   dict(fused_activation_function='RELU')),
                                          inputs=m[convlike].inputs,
                                          outputs=m[activation].outputs),
                    lambda m: not m[convlike].attribs.get('fused_activation_function'))

    conv_output = matcher.Tensor()
    convlike = matcher.Operation(name=fuse_to, outputs=conv_output)
    activation = matcher.Operation(name="tf.clip_by_value", inputs={0: conv_output})

    matcher.replace(graph=tf_graph,
                    pattern=activation,
                    replacement=lambda m: TFOperation(graph=tf_graph,
                                                      name=m[convlike].name,
                                                      attribs=utils.dict_union(m[convlike].attribs,
                                                                               dict(fused_activation_function='RELU6')),
                                                      inputs=m[convlike].inputs,
                                                      outputs=m[activation].outputs),
                    condition=lambda m: (m[activation].inputs[1].data == [0]
                                         and m[activation].inputs[2].data == [6]
                                         and not m[convlike].attribs.get('fused_activation_function')))


def convert_conv2d(op):
    # type: (TFOperation)->None

    op.name = "CONV_2D"
    assert len(op.attribs['strides']) == 4 and op.attribs['strides'][0] == op.attribs['strides'][3] == 1
    assert len(op.attribs['dilations']) == 4 and op.attribs['dilations'][0] == op.attribs['dilations'][3] == 1
    op.attribs = dict(stride_h=op.attribs['strides'][1],
                      stride_w=op.attribs['strides'][2],
                      padding=op.attribs['padding'],
                      dilation_h_factor=op.attribs['dilations'][1],
                      dilation_w_factor=op.attribs['dilations'][2],
                      fused_activation_function=op.attribs.get('fused_activation_function', 'NONE'))
    filter = op.inputs[1]
    filter.shape = [filter.shape[-1]] + filter.shape[:-1]
    axes = [filter.rank - 1] + list(range(0, filter.rank - 1))
    filter.data = filter.data.transpose(tuple(axes))


def convert_conv2d_transpose(op):
    # type: (TFOperation)->None

    op.name = "TRANSPOSE_CONV"

    filter = op.inputs[1]
    filter.shape = [filter.shape[-2]] + filter.shape[:-2] + [filter.shape[-1]]
    axes = [filter.rank - 2] + list(range(0, filter.rank - 2)) + [filter.rank - 1]
    filter.data = filter.data.transpose(tuple(axes))
    op.inputs = (TFTensor(graph=op.graph,
                          name=None,
                          shape=[len(op.attribs['output_shape'])],
                          data=np.array(op.attribs['output_shape'], dtype=np.int32),
                          dtype="INT32"), filter, op.inputs[0])

    assert len(op.attribs['strides']) == 4 and op.attribs['strides'][0] == op.attribs['strides'][3] == 1
    op.attribs = dict(stride_h=op.attribs['strides'][1],
                      stride_w=op.attribs['strides'][2],
                      padding=op.attribs['padding'],
                      fused_activation_function=op.attribs.get('fused_activation_function', 'NONE'))


def convert_depthwise_conv2d(op):
    # type: (TFOperation)->None
    op.name = "DEPTHWISE_CONV_2D"
    assert op.outputs[0].shape[3] % op.inputs[0].shape[3] == 0
    assert len(op.attribs['strides']) == 4 and op.attribs['strides'][0] == op.attribs['strides'][3] == 1
    rate = op.attribs.get('rate')
    if not rate:
        rate = [1, 1]
    assert len(rate) == 2
    op.attribs = dict(stride_h=op.attribs['strides'][1],
                      stride_w=op.attribs['strides'][2],
                      padding=op.attribs['padding'],
                      dilation_h_factor=rate[0],
                      dilation_w_factor=rate[1],
                      depth_multiplier=op.outputs[0].shape[3] // op.inputs[0].shape[3],
                      fused_activation_function=op.attribs.get('fused_activation_function', 'NONE'))
    filter = op.inputs[1]
    filter.shape = [1] + filter.shape[:-2] + [filter.shape[-2] * filter.shape[-1]]
    filter.data = filter.data.reshape(filter.shape)


def generic_convert_pool_2d(op, target_name):
    # type: (TFOperation, str)->None
    op.name = target_name
    assert len(op.attribs['strides']) == 4 and op.attribs['strides'][0] == op.attribs['strides'][3] == 1
    assert len(op.attribs['ksize']) == 4 and op.attribs['ksize'][0] == op.attribs['ksize'][3] == 1
    op.attribs = dict(stride_h=op.attribs['strides'][1],
                      stride_w=op.attribs['strides'][2],
                      padding=op.attribs['padding'],
                      filter_height=op.attribs['ksize'][1],
                      filter_width=op.attribs['ksize'][2])


def convert_flatten(op):
    # type: (TFOperation)->None
    op.name = "RESHAPE"
    shape = [op.inputs[0].shape[0], -1]
    shape_tensor = TFTensor(graph=op.graph, name=None, shape=[2], data=np.array(shape, dtype=np.int64), dtype="INT64")
    op.inputs = (op.inputs[0], shape_tensor)
    op.attribs = dict(new_shape=shape)


def convert_reshape(op):
    # type: (TFOperation)->None
    op.name = "RESHAPE"
    op.inputs = (op.inputs[0], TFTensor(graph=op.graph,
                                        name=None,
                                        shape=[len(op.attribs['shape'])],
                                        data=np.array(op.attribs['shape'], dtype=np.int32),
                                        dtype="INT32"))
    op.attribs = dict(new_shape=op.attribs['shape'])


def convert_softmax(op):
    # type: (TFOperation)->None
    assert op.attribs['axis'] in [None, -1, op.input.rank - 1]

    op.name = "SOFTMAX"
    op.attribs = dict(beta=1)


def convert_squeeze(op):
    # type: (TFOperation)->None

    op.name = "SQUEEZE"
    op.attribs = dict(squeeze_dims=op.attribs['axis'])


def generic_convert_reduce(op, target_name):
    # type: (TFOperation, str)->None

    op.name = target_name
    axes = op.attribs["axis"]
    op.inputs = (op.input, TFTensor(graph=op.graph, shape=[len(axes)], dtype='INT32', data=list(axes)))
    op.attribs = dict(keep_dims=op.attribs["keepdims"])


def convert_fully_connected(op):
    # type: (TFOperation)->None
    op.name = "FULLY_CONNECTED"
    op.attribs["weights_format"] = "DEFAULT"
    assert (not op.attribs["transpose_a"] and op.attribs["transpose_b"]
            and not op.attribs.get("adjoint_a") and not op.attribs.get("adjoint_b"))


def convert_pad(op):
    # type: (TFOperation)->None
    op.name = "PAD"
    pads = list(np.array(op.attribs['paddings']).flatten().tolist())
    op.inputs = (op.input, TFTensor(graph=op.graph,
                                    shape=[len(pads) // 2, 2],
                                    dtype='INT32',
                                    data=pads))
    op.attribs = dict()


def convert_split(op):
    # type: (TFOperation)->None
    num_or_size_splits = op.attribs['num_or_size_splits']

    if not isinstance(num_or_size_splits, (list, tuple)) or len(utils.unique(num_or_size_splits)) == 1:
        op.name = "SPLIT"
        num_splits = len(num_or_size_splits) if isinstance(num_or_size_splits, (list, tuple)) else num_or_size_splits
        op.inputs = (TFTensor(graph=op.graph, shape=[], dtype='INT32', data=[op.attribs["axis"]]), op.input)
        op.attribs = dict(num_splits=num_splits)
    else:
        op.name = "SPLIT_V"
        size_splits = list(num_or_size_splits)
        op.inputs = (op.input,
                     TFTensor(graph=op.graph, shape=[len(size_splits)], dtype='INT32', data=size_splits),
                     TFTensor(graph=op.graph, shape=[], dtype='INT32', data=[op.attribs["axis"]]))
        op.attribs = dict(num_splits=len(size_splits))


def convert_slice(op):
    # type: (TFOperation)->None
    op.name = "SLICE"

    op.inputs = (op.input,
                 TFTensor(graph=op.graph, shape=[4], dtype='INT32', data=list(op.attribs["begin"])),
                 TFTensor(graph=op.graph, shape=[4], dtype='INT32', data=list(op.attribs["size"])))
    op.attribs = dict()


def generic_convert_argminmax(op, target_name):
    # type: (TFOperation, str)->None
    kTfLiteInt64 = 4

    axis = op.attribs["axis"]

    op.name = target_name
    op.inputs = (op.input, TFTensor(graph=op.graph, shape=[], dtype='INT32', data=[axis]))
    op.attribs = dict(output_type=kTfLiteInt64)

    output_tensor = op.output
    op.outputs = (TFTensor(graph=op.graph,
                           shape=unsqueezed_shape(shape=output_tensor.shape, axes=[axis]),
                           dtype=output_tensor.dtype),)
    TFOperation(graph=op.graph,
                name="SQUEEZE",
                inputs=op.output,
                outputs=output_tensor,
                attribs=dict(squeeze_dims=[axis]))


def rename(op, target_name):
    # type: (TFOperation, str)->None
    op.name = target_name


def convert_expand_dims(op):
    # type: (TFOperation)->None
    shape = op.output.shape
    op.name = "RESHAPE"
    op.inputs = (op.inputs[0], TFTensor(graph=op.graph,
                                        name=None,
                                        shape=[len(shape)],
                                        data=np.array(shape, dtype=np.int32),
                                        dtype="INT32"))
    op.attribs = dict(new_shape=list(shape))


def convert_fill(op):
    # type: (TFOperation)->None
    op.name = "FILL"
    op.inputs = (TFTensor(graph=op.graph,
                          name=None,
                          shape=[len(op.attribs['dims'])],
                          data=np.array(op.attribs['dims'], dtype=np.int32),
                          dtype="INT32"),
                 TFTensor(graph=op.graph,
                          name=None,
                          shape=[],
                          data=np.array(op.attribs['value'], dtype=np.int32),
                          dtype="INT32"))
    op.attribs = dict()


def convert_l2_normalize(op):
    # type: (TFOperation)->None
    assert op.attribs["axis"] in [[-1], [op.input.rank - 1]]
    op.name = "L2_NORMALIZATION"
    op.attribs = dict()


def convert_leaky_relu(op):
    # type: (TFOperation)->None
    op.name = "LEAKY_RELU"
    assert op.inputs[1].data is not None
    if isinstance(op.inputs[1].data, list):
        assert len(op.inputs[1].data) == 1
        alpha = op.inputs[1].data[0]
    else:
        assert isinstance(op.inputs[1].data, np.ndarray) and op.inputs[1].data.shape == tuple()
        alpha = float(op.inputs[1].data)
    op.inputs = (op.inputs[0],)
    op.attribs = dict(alpha=alpha)


def convert_lrn(op):
    # type: (TFOperation)->None
    op.name = "LOCAL_RESPONSE_NORMALIZATION"
    op.attribs = dict(radius=op.attribs["depth_radius"],
                      bias=op.attribs["bias"],
                      alpha=op.attribs["alpha"],
                      beta=op.attribs["beta"])


def convert_stack(op):
    # type: (TFOperation)->None
    op.name = "PACK"
    op.attribs = dict(axis=op.attribs["axis"],
                      values_count=len(op.inputs))


def convert_clip_by_value(op):
    # type: (TFOperation)->None
    assert op.inputs[1].data == [0] and op.inputs[2].data == [6]
    op.name = "RELU6"
    op.inputs = (op.inputs[0],)


def convert_resize_nearest_neighbor(op):
    # type: (TFOperation)->None
    op.name = "RESIZE_NEAREST_NEIGHBOR"
    op.inputs = (op.inputs[0], TFTensor(graph=op.graph,
                                        name=None,
                                        shape=[len(op.attribs['size'])],
                                        data=np.array(op.attribs['size'], dtype=np.int32),
                                        dtype="INT32"))
    op.attribs = dict(align_corners=op.attribs['align_corners'])


def convert_transpose(op):
    # type: (TFOperation)->None
    op.name = "TRANSPOSE"
    op.inputs = (op.inputs[0], TFTensor(graph=op.graph,
                                        name=None,
                                        shape=[len(op.attribs['perm'])],
                                        data=np.array(op.attribs['perm'], dtype=np.int32),
                                        dtype="INT32"))
    op.attribs = dict()


# TODO remove unsqueeze/squeeze that cancel out each other
_DefaultConverters = {
    "tf.nn.conv2d": convert_conv2d,
    "tf.nn.conv2d_transpose": convert_conv2d_transpose,
    "tf.nn.bias_add": partial(rename, target_name="ADD"),
    "tf.nn.depthwise_conv2d": convert_depthwise_conv2d,
    "tf.nn.avg_pool": partial(generic_convert_pool_2d, target_name="AVERAGE_POOL_2D"),
    "tf.layers.flatten": convert_flatten,
    "tf.reshape": convert_reshape,
    "tf.nn.softmax": convert_softmax,
    "tf.squeeze": convert_squeeze,
    "tf.multiply": partial(rename, target_name="MUL"),
    "tf.concat": partial(rename, target_name="CONCATENATION"),
    "tf.nn.relu": partial(rename, target_name="RELU"),
    "tf.split": convert_split,
    "tf.reduce_mean": partial(generic_convert_reduce, target_name="MEAN"),
    "tf.reduce_min": partial(generic_convert_reduce, target_name="REDUCE_MIN"),
    "tf.reduce_max": partial(generic_convert_reduce, target_name="REDUCE_MAX"),
    "tf.reduce_sum": partial(generic_convert_reduce, target_name="SUM"),
    "tf.matmul": convert_fully_connected,
    "tf.pad": convert_pad,
    "tf.slice": convert_slice,
    "tf.abs": partial(rename, target_name="ABS"),
    "tf.add_n": partial(rename, target_name="ADD_N"),
    "tf.add": partial(rename, target_name="ADD"),
    "tf.divide": partial(rename, target_name="DIV"),
    "tf.argmax": partial(generic_convert_argminmax, target_name="ARG_MAX"),
    "tf.argmin": partial(generic_convert_argminmax, target_name="ARG_MIN"),
    "tf.expand_dims": convert_expand_dims,
    "tf.nn.elu": partial(rename, target_name="ELU"),
    "tf.equal": partial(rename, target_name="EQUAL"),
    "tf.exp": partial(rename, target_name="EXP"),
    "tf.fill": convert_fill,
    "tf.floor": partial(rename, target_name="FLOOR"),
    "tf.ceil": partial(rename, target_name="CEIL"),
    "tf.greater_equal": partial(rename, target_name="GREATER_EQUAL"),
    "tf.greater": partial(rename, target_name="GREATER"),
    "tf.nn.l2_normalize": convert_l2_normalize,
    "tf.maximum": partial(rename, target_name="MAXIMUM"),
    "tf.minimum": partial(rename, target_name="MINIMUM"),
    "tf.nn.leaky_relu": convert_leaky_relu,
    "tf.less": partial(rename, target_name="LESS"),
    "tf.less_equal": partial(rename, target_name="LESS_EQUAL"),
    "tf.nn.lrn": convert_lrn,
    "tf.logical_or": partial(rename, target_name="LOGICAL_OR"),
    "tf.logical_and": partial(rename, target_name="LOGICAL_AND"),
    "tf.logical_not": partial(rename, target_name="LOGICAL_NOT"),
    "tf.log": partial(rename, target_name="LOG"),
    "tf.nn.max_pool": partial(generic_convert_pool_2d, target_name="MAX_POOL_2D"),
    "tf.negative": partial(rename, target_name="NEG"),
    "tf.not_equal": partial(rename, target_name="NOT_EQUAL"),
    "tf.stack": convert_stack,
    "tf.pow": partial(rename, target_name="POW"),
    "tf.nn.relu6": partial(rename, target_name="RELU6"),
    "tf.clip_by_value": convert_clip_by_value,
    "tf.image.resize_nearest_neighbor": convert_resize_nearest_neighbor,
    "tf.where": partial(rename, target_name="SELECT"),
    "tf.sqrt": partial(rename, target_name="SQRT"),
    "tf.square": partial(rename, target_name="SQUARE"),
    "tf.subtract": partial(rename, target_name="SUB"),
    "tf.transpose": convert_transpose,
    "tf.unstack": partial(rename, target_name="UNPACK"),
}
