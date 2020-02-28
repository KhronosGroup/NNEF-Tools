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

import copy
from collections import OrderedDict

import numpy as np

from nnef_tools.core import graph_utils, utils
from nnef_tools.core import matcher
from nnef_tools.core.graph import Tensor, Graph
from nnef_tools.io.tensorflow.tf_graph import TFOperation, TFTensor, TFGraph
from nnef_tools.io.tensorflow.tf_py import tf_py_unify
from nnef_tools.shape_inference import shape_inference


def pre_conversion_pass(g):
    # type: (TFGraph)->None

    transform_by_op_name = {
        "tf.cast": _transform_cast,
        "tf.fill": _transform_fill,
        "tf.zeros": _transform_zeros_ones_like,
        "tf.ones": _transform_zeros_ones_like,
        "tf.zeros_like": _transform_zeros_ones_like,
        "tf.ones_like": _transform_zeros_ones_like,
        "tf.range": _transform_range,
        "tf.strided_slice": _transform_strided_slice,
        "tf.nn.fused_batch_norm": _transform_fused_batch_norm,

        "_tf.TransposeGrad": _transform_transpose_grad,
        "_tf.strided_slice_grad": _transform_strided_slice_grad,
        "_tf.sqrt_grad": _transform_sqrt_grad,
        "_tf.elu_grad": _transform_elu_grad,
        "_tf.relu_grad": _transform_relu_grad,
        "_tf.leaky_relu_grad": _transform_leaky_relu_grad,
        "_tf.relu6_grad": _transform_relu6_grad,
        "_tf.softplus_grad": _transform_softplus_grad,
        "_tf.rsqrt_grad": _transform_rsqrt_grad,
        "_tf.sigmoid_grad": _transform_sigmoid_grad,
        "_tf.tanh_grad": _transform_tanh_grad,
        "_tf.reciprocal_grad": _transform_reciprocal_grad,
        "_tf.bias_add_grad": _transform_bias_add_grad,
        "_tf.MinOrMaxGrad": _transform_min_or_max_grad,
        "_tf.lrn_grad": _transform_lrn_grad,
    }

    # tf.identity is now not a passthrough, we convert it to copy and the optimizer can remove it
    passthroughs = ["tf.stop_gradient", "tf.nn.dropout"]

    tf_py_unify.unify_ops(g)

    for op in list(g.operations):
        transform = transform_by_op_name.get(op.name)
        if transform:
            transform(g, op)

    _transform_cgf_stb(g)
    _transform_bts_conv_stb(g)
    _transform_pad(g)
    _transform_add_conv(g)
    _transform_bias_add_conv(g)
    _transform_space_to_batch_depthwise_conv(g)

    graph_utils.remove_passthroughs(g, is_passthrough=lambda op_: op_.name in passthroughs)
    graph_utils.remove_unreachable(g)

    _transform_separate_inputs_and_outputs(g)
    _transform_separate_duplicated_outputs(g)

    g.generate_missing_names()


def _is_nhwc(tf_op, default=True):
    # type: (TFOperation, bool)->bool
    if not tf_op.attribs.get("data_format"):
        return default
    return not tf_op.attribs["data_format"].upper().startswith("NC")


def _reduced_shape(shape, axes):
    return [1 if i in axes else s for i, s in enumerate(shape)]


def _nonneg_axis(axis, rank):
    while axis < 0:
        axis += rank
    return axis


def _nonneg_axes(axes, rank, none_means_all=False):
    if axes is None:
        if none_means_all:
            return list(range(rank))
        assert False, "Axes is None, use none_means_all if applicable"
    return [_nonneg_axis(a, rank) for a in axes]


# One to many:

def _transform_cast(g, op):
    # type: (TFGraph, TFOperation)->None
    from_ = op.input.dtype  # type: str
    to_ = op.attribs["dtype"]  # type: str

    if (from_ == to_
            or (from_.startswith('float') and to_.startswith('float'))
            or (from_.startswith('int') and to_.startswith('int'))):
        TFOperation(graph=g, name="tf.identity", inputs=op.input, outputs=op.outputs)
    elif from_ == "bool" and to_.startswith("float"):
        zeros = TFTensor(graph=g, shape=list(op.input.shape), dtype=to_, data=0.0)
        ones = TFTensor(graph=g, shape=list(op.input.shape), dtype=to_, data=1.0)
        TFOperation(graph=g, name="tf.where", inputs=(op.input, ones, zeros), outputs=op.outputs)
    elif from_.startswith("float") and to_ == "bool":
        zeros = TFTensor(graph=g, shape=list(op.input.shape), dtype=from_, data=0.0)
        TFOperation(graph=g, name="tf.not_equal", inputs=(op.input, zeros), outputs=op.outputs)
    else:
        print("Possibly unsupported tf.cast: {} -> {}".format(from_, to_))
        return
    g.remove_operation(op, unlink=True)


def _transform_fused_batch_norm(g, op):
    # type: (TFGraph, TFOperation)->None
    VARIANCE_CORRECTION_ENABLED = True

    in_input = op.inputs[0]
    in_scale = op.inputs[1]
    in_offset = op.inputs[2]

    epsilon = op.attribs["epsilon"]

    out_y = op.outputs[0]
    out_batch_mean = op.outputs[1]
    out_batch_var = op.outputs[2]

    data_format = op.attribs["data_format"].upper() if op.attribs["data_format"] else "NHWC"
    channel_dim = 1 if data_format == "NCHW" else in_input.rank - 1
    rest_count = int(op.inputs[0].count / channel_dim)
    tensors_to_remove = []

    if op.attribs["is_training"]:
        if VARIANCE_CORRECTION_ENABLED:
            biased_batch_var = TFTensor(graph=g, shape=list(out_batch_var.shape), dtype=out_batch_var.dtype)
            const = TFTensor(graph=g, shape=[], dtype=in_input.dtype, data=float(rest_count) / max(rest_count - 1, 1))
            TFOperation(graph=g,
                        name="tf.nn.moments",
                        inputs=in_input,
                        attribs=dict(axes=utils.without(range(in_input.rank), channel_dim), keep_dims=False),
                        outputs=(out_batch_mean, biased_batch_var))
            TFOperation(graph=g,
                        name="tf.multiply",
                        inputs=(biased_batch_var, const),
                        outputs=out_batch_var)
            TFOperation(graph=g,
                        name="tf.nn.batch_normalization",
                        inputs=(in_input, out_batch_mean, out_batch_var, in_offset, in_scale),
                        attribs=dict(variance_epsilon=epsilon, _data_format=data_format),
                        outputs=out_y)
            if len(op.outputs) > 3:  # This can happen in gradients
                out_saved_mean = op.outputs[3]
                out_saved_var = op.outputs[4]
                graph_utils.replace_tensor_in_consumers(g, out_saved_mean, out_batch_mean)
                graph_utils.replace_tensor_in_consumers(g, out_saved_var, out_batch_var)
                tensors_to_remove += [out_saved_mean, out_saved_var]
        else:  # not VARIANCE_CORRECTION_ENABLED
            TFOperation(graph=g,
                        name="tf.nn.moments",
                        inputs=in_input,
                        attribs=dict(axes=utils.without(range(in_input.rank), channel_dim), keep_dims=False),
                        outputs=(out_batch_mean, out_batch_var))
            TFOperation(graph=g,
                        name="tf.nn.batch_normalization",
                        inputs=(in_input, out_batch_mean, out_batch_var, in_offset, in_scale),
                        attribs=dict(variance_epsilon=epsilon, _data_format=data_format),
                        outputs=out_y)
            if len(op.outputs) > 3:  # This can happen in gradients
                out_saved_mean = op.outputs[3]
                out_saved_var = op.outputs[4]
                graph_utils.replace_tensor_in_consumers(g, out_saved_mean, out_batch_mean)
                graph_utils.replace_tensor_in_consumers(g, out_saved_var, out_batch_var)
                tensors_to_remove += [out_saved_mean, out_saved_var]
    else:  # not training
        in_mean = op.inputs[3]
        in_variance = op.inputs[4]
        graph_utils.replace_tensor_in_consumers(g, out_batch_mean, in_mean)
        graph_utils.replace_tensor_in_consumers(g, out_batch_var, in_variance)
        tensors_to_remove += [out_batch_mean, out_batch_var]
        if len(op.outputs) > 3:  # This can happen in gradients
            out_saved_mean = op.outputs[3]
            out_saved_var = op.outputs[4]
            graph_utils.replace_tensor_in_consumers(g, out_saved_mean, in_mean)
            graph_utils.replace_tensor_in_consumers(g, out_saved_var, in_variance)
            tensors_to_remove += [out_saved_mean, out_saved_var]
        TFOperation(graph=g,
                    name="tf.nn.batch_normalization",
                    inputs=(in_input, in_mean, in_variance, in_offset, in_scale),
                    attribs=dict(variance_epsilon=epsilon, _data_format=data_format),
                    outputs=out_y)
    g.remove_operation(op, unlink=True)
    for t in tensors_to_remove:
        g.remove_tensor(t)


# One to one:

def _transform_fill(g, op):  # to constant
    # type: (TFGraph, TFOperation)->None

    op.output.data = op.attribs["value"]
    g.remove_operation(op, unlink=True)


def _transform_zeros_ones_like(g, op):
    # type: (TFGraph, TFOperation)->None

    assert op.name in ["tf.zeros", "tf.ones", "tf.zeros_like", "tf.ones_like"]

    op.output.data = np.array(0 if op.name.startswith('tf.zeros') else 1, dtype=np.dtype(op.output.dtype)).tolist()
    g.remove_operation(op, unlink=True)


def _transform_range(g, op):
    # type: (TFGraph, TFOperation)->None

    op.output.data = np.arange(start=op.attribs["start"], stop=op.attribs["limit"], step=op.attribs["delta"],
                               dtype=np.dtype(op.output.dtype)).tolist()
    g.remove_operation(op, unlink=True)


def _transform_strided_slice(g, op):
    # type: (TFGraph, TFOperation)->None

    if op.attribs["strides"] is not None and not all(s == 1 for s in op.attribs["strides"]):
        return

    ssl_begin, ssl_end, ssl_stride, ssl_shape, reshape_shape = shape_inference.decompose_strided_slice(
        input=op.input.shape,
        begin=op.attribs['begin'],
        end=op.attribs['end'],
        stride=op.attribs["strides"] if op.attribs["strides"] is not None else [1] * len(op.attribs["begin"]),
        ellipsis_mask=op.attribs['ellipsis_mask'],
        new_axis_mask=op.attribs['new_axis_mask'],
        shrink_axis_mask=op.attribs['shrink_axis_mask'],
        begin_mask=op.attribs['begin_mask'],
        end_mask=op.attribs['end_mask']
    )

    assert all(stride == 1 for stride in ssl_stride)

    slice_size = [e - b for b, e in zip(ssl_begin, ssl_end)]

    if reshape_shape != ssl_shape:
        slice_output = TFTensor(graph=g, shape=ssl_shape, dtype=op.output.dtype)
        TFOperation(graph=g,
                    name="tf.slice",
                    inputs=op.input,
                    attribs=dict(begin=ssl_begin, size=slice_size),
                    outputs=slice_output)
        TFOperation(graph=g,
                    name="tf.reshape",
                    inputs=slice_output,
                    attribs=dict(shape=reshape_shape),
                    outputs=op.outputs)
    else:
        TFOperation(graph=g,
                    name="tf.slice",
                    inputs=op.input,
                    attribs=dict(begin=ssl_begin, size=slice_size),
                    outputs=op.outputs)

    g.remove_operation(op, unlink=True)


# One to one grad:

def _transform_transpose_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    grad = op.inputs[2]

    TFOperation(graph=g,
                name="tf.transpose",
                inputs=grad,
                attribs=dict(perm=utils.inverse_permutation(op.attribs["orig_perm"])),
                outputs=op.outputs)

    g.remove_operation(op, unlink=True)


def _transform_strided_slice_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    def is_compatible(s1, s2):
        s1 = list(s1)
        s2 = list(s2)
        if (s1 == [] and s2 == [1]) or (s2 == [] and s1 == [1]):
            return True
        for a, b in zip(s1, s2):
            if a != b:
                return False
        return True

    assert op.attribs["strides"] is None or all(s == 1 for s in op.attribs["strides"]), \
        "Only strides=1 is supported for tf.strided_slice, got: {}".format(op.attribs["strides"])

    input_shape = op.attribs["shape"]

    ssl_begin, ssl_end, ssl_stride, ssl_shape, reshape_shape = shape_inference.decompose_strided_slice(
        input=input_shape,
        begin=op.attribs['begin'],
        end=op.attribs['end'],
        stride=op.attribs["strides"] if op.attribs["strides"] is not None else [1] * len(op.attribs["begin"]),
        ellipsis_mask=op.attribs['ellipsis_mask'],
        new_axis_mask=op.attribs['new_axis_mask'],
        shrink_axis_mask=op.attribs['shrink_axis_mask'],
        begin_mask=op.attribs['begin_mask'],
        end_mask=op.attribs['end_mask']
    )

    assert all(stride == 1 for stride in ssl_stride)

    if reshape_shape != ssl_shape:
        assert is_compatible(reshape_shape, op.input.shape), \
            "Shape mismatch in strided_slice_grad {} {}".format(reshape_shape, op.input.shape)

        reshape = TFOperation(graph=g,
                              name="tf.reshape",
                              inputs=op.input,
                              attribs=dict(shape=list(ssl_shape)),
                              outputs=TFTensor(graph=g, shape=list(ssl_shape), dtype=op.output.dtype))
        TFOperation(graph=g,
                    name="tf.pad",
                    inputs=reshape.output,
                    attribs=dict(
                        paddings=[[b, s - e] for b, e, s in zip(ssl_begin, ssl_end, input_shape)],
                        mode="CONSTANT",
                        constant_values=0),
                    outputs=op.outputs)
    else:
        TFOperation(graph=g,
                    name="tf.pad",
                    inputs=op.input,
                    attribs=dict(paddings=[[b, s - e] for b, e, s in zip(ssl_begin, ssl_end, input_shape)],
                                 mode="CONSTANT",
                                 constant_values=0),
                    outputs=op.outputs)

    g.remove_operation(op, unlink=True)


# One to many grad:

def _transform_sqrt_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    # def sqrt_grad(y, dy):
    #     return dy * 0.5 / y

    y, dy = op.inputs

    const_half = TFTensor(graph=g, shape=[], dtype=op.output.dtype, data=0.5)
    mul = TFOperation(graph=g,
                      name="tf.multiply",
                      inputs=(dy, const_half),
                      outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype))
    TFOperation(graph=g, name="tf.divide", inputs=(mul.output, y), outputs=op.outputs)
    g.remove_operation(op, unlink=True)


def _transform_elu_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    # def elu_grad(gradients, outputs):
    #     return tf.where(outputs > 0, gradients, gradients * (outputs + 1))

    gradients, outputs = op.inputs

    const0 = TFTensor(graph=g, shape=[], dtype=op.output.dtype, data=0.0)
    const1 = TFTensor(graph=g, shape=[], dtype=op.output.dtype, data=1.0)
    greater = TFOperation(graph=g,
                          name="tf.greater",
                          inputs=(outputs, const0),
                          outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype="bool"))
    add = TFOperation(graph=g,
                      name="tf.add",
                      inputs=(outputs, const1),
                      outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype))
    mul = TFOperation(graph=g,
                      name="tf.multiply",
                      inputs=(gradients, add.output),
                      outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype))
    TFOperation(graph=g, name="tf.where", inputs=(greater.output, gradients, mul.output), outputs=op.outputs)
    g.remove_operation(op, unlink=True)


def _transform_relu_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    # def relu_grad(gradients, features):
    #     return tf.where(features > 0, gradients, 0.0)

    gradients, features = op.inputs

    const0 = TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype, data=0.0)
    greater = TFOperation(graph=g,
                          name="tf.greater",
                          inputs=(features, const0),
                          outputs=TFTensor(graph=g, name=None, shape=list(op.output.shape), dtype="bool"))
    TFOperation(graph=g, name="tf.where", inputs=(greater.output, gradients, const0), outputs=op.outputs)
    g.remove_operation(op, unlink=True)


def _transform_leaky_relu_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    # def leaky_relu_grad(gradients, features):
    #     return tf.where(features > 0, gradients, gradients * alpha)

    gradients, features = op.inputs

    const0 = TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype, data=0.0)
    const_alpha = TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype, data=op.attribs['alpha'])
    greater = TFOperation(graph=g,
                          name="tf.greater",
                          inputs=(features, const0),
                          outputs=TFTensor(graph=g, name=None, shape=list(op.output.shape), dtype="bool"))
    multiply = TFOperation(graph=g,
                           name="tf.multiply",
                           inputs=(gradients, const_alpha),
                           outputs=TFTensor(graph=g, name=None, shape=list(op.output.shape), dtype=op.output.dtype))
    TFOperation(graph=g, name="tf.where", inputs=(greater.output, gradients, multiply.output), outputs=op.outputs)
    g.remove_operation(op, unlink=True)


def _transform_relu6_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    # def relu6_grad(gradients, features):
    #     return tf.where(features > 0 and features < 6, gradients, 0.0)

    gradients, features = op.inputs

    const0 = TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype, data=0.0)
    const6 = TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype, data=6.0)
    greater = TFOperation(graph=g,
                          name="tf.greater",
                          inputs=(features, const0),
                          outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype="bool"))
    less = TFOperation(graph=g,
                       name="tf.less",
                       inputs=(features, const6),
                       outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype="bool"))
    and_ = TFOperation(graph=g,
                       name="tf.logical_and",
                       inputs=(greater.output, less.output),
                       outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype="bool"))
    TFOperation(graph=g, name="tf.where", inputs=(and_.output, gradients, const0), outputs=op.outputs)
    g.remove_operation(op, unlink=True)


def _transform_softplus_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    # def softplus_grad(gradients, features):
    #     return gradients * (tf.exp(features) / (tf.exp(features) + 1))

    gradients, features = op.inputs

    const1 = TFTensor(graph=g, shape=[], dtype=op.output.dtype, data=1.0)
    exp = TFOperation(graph=g,
                      name="tf.exp",
                      inputs=features,
                      outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype))
    add = TFOperation(graph=g,
                      name="tf.add",
                      inputs=(exp.output, const1),
                      outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype))
    div = TFOperation(graph=g,
                      name="tf.divide",
                      inputs=(exp.output, add.output),
                      outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype))
    TFOperation(graph=g, name="tf.multiply", inputs=(gradients, div.output), outputs=op.outputs)
    g.remove_operation(op, unlink=True)


def _transform_rsqrt_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    # def rsqrt_grad(y, dy):
    #     return  (-0.5 * dy) *  y ** 3

    y, dy = op.inputs

    const_neg_half = TFTensor(graph=g, shape=[], dtype=op.output.dtype, data=-0.5)
    const_3 = TFTensor(graph=g, shape=[], dtype=op.output.dtype, data=3.0)

    mul1 = TFOperation(graph=g,
                       name="tf.multiply",
                       inputs=(const_neg_half, dy),
                       outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype))
    pow = TFOperation(graph=g,
                      name="tf.pow",
                      inputs=(y, const_3),
                      outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype))
    TFOperation(graph=g, name="tf.multiply", inputs=(mul1.output, pow.output), outputs=op.outputs)
    g.remove_operation(op, unlink=True)


def _transform_sigmoid_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    # def sigmoid_grad(y, dy):
    #     return dy * y * (1 - y)

    y, dy = op.inputs

    const1 = TFTensor(graph=g, shape=[], dtype=op.output.dtype, data=1.0)
    mul1 = TFOperation(graph=g,
                       name="tf.multiply",
                       inputs=(dy, y),
                       outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype))
    sub = TFOperation(graph=g,
                      name="tf.subtract",
                      inputs=(const1, y),
                      outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype))
    TFOperation(graph=g, name="tf.multiply", inputs=(mul1.output, sub.output), outputs=op.outputs)
    g.remove_operation(op, unlink=True)


def _transform_tanh_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    # def tanh_grad(y, dy):
    #     return dy * (1 - y ** 2)

    y, dy = op.inputs

    const1 = TFTensor(graph=g, shape=[], dtype=op.output.dtype, data=1.0)
    square = TFOperation(graph=g,
                         name="tf.square",
                         inputs=y,
                         outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype))
    sub = TFOperation(graph=g,
                      name="tf.subtract",
                      inputs=(const1, square.output),
                      outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype))
    TFOperation(graph=g, name="tf.multiply", inputs=(dy, sub.output), outputs=op.outputs)
    g.remove_operation(op, unlink=True)


def _transform_reciprocal_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    # def reciprocal_grad(y, dy):
    #     return -dy * y ** 2

    y, dy = op.inputs

    neg = TFOperation(graph=g,
                      name="tf.negative",
                      inputs=dy,
                      outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype))
    square = TFOperation(graph=g,
                         name="tf.square",
                         inputs=y,
                         outputs=TFTensor(graph=g, shape=list(op.output.shape), dtype=op.output.dtype))
    TFOperation(graph=g, name="tf.multiply", inputs=(neg.output, square.output), outputs=op.outputs)
    g.remove_operation(op, unlink=True)


def _transform_bias_add_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    # def bias_add_grad(out_backprop, data_format="NHWC"):
    #     return tf.reduce_sum(out_backprop, [axes except c])

    out_backprop = op.input

    if _is_nhwc(op):
        axes = list(range(out_backprop.rank - 1))
    else:
        axes = [0] + list(range(2, out_backprop.rank))

    TFOperation(graph=g,
                name="tf.reduce_sum",
                inputs=out_backprop,
                attribs=dict(axis=axes, keepdims=False),
                outputs=op.outputs)
    g.remove_operation(op, unlink=True)


def _transform_min_or_max_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    # def _MinOrMaxGrad(input, axes, y, grad):
    #     output_shape_kept_dims = reduced_shape(input.shape, axes)
    #     y = reshape(y, output_shape_kept_dims) # needed?
    #     grad = reshape(grad, output_shape_kept_dims) # needed?
    #     equal = math_ops.equal(y, input)
    #     indicators = cast(equal, tf.float32)
    #     num_selected = reshape(reduce_sum(indicators, axes), output_shape_kept_dims) # needed?
    #     return indicators / num_selected * grad

    input, y, grad = op.inputs

    axes = _nonneg_axes(op.attribs["orig_axis"], input.rank, none_means_all=True)
    output_shape_kept_dims = _reduced_shape(input.shape, axes)

    reshape0 = TFOperation(graph=g,
                           name="tf.reshape",
                           inputs=y,
                           attribs=dict(shape=list(output_shape_kept_dims)),
                           outputs=TFTensor(graph=g, shape=list(output_shape_kept_dims), dtype=op.output.dtype))
    reshape1 = TFOperation(graph=g,
                           name="tf.reshape",
                           inputs=grad,
                           attribs=dict(shape=list(output_shape_kept_dims)),
                           outputs=TFTensor(graph=g, shape=list(output_shape_kept_dims), dtype=op.output.dtype))
    equal = TFOperation(graph=g,
                        name="tf.equal",
                        inputs=(reshape0.output, input),
                        outputs=TFTensor(graph=g, shape=input.shape, dtype="bool"))
    const0 = TFTensor(graph=g, shape=list(equal.output.shape), dtype=op.output.dtype, data=0.0)
    const1 = TFTensor(graph=g, shape=list(equal.output.shape), dtype=op.output.dtype, data=1.0)
    where = TFOperation(graph=g,
                        name="tf.where",
                        inputs=(equal.output, const1, const0),
                        outputs=TFTensor(graph=g, shape=list(equal.output.shape), dtype=op.output.dtype))
    reduce = TFOperation(graph=g,
                         name="tf.reduce_sum",
                         inputs=where.output,
                         attribs=dict(axis=axes, keepdims=False),
                         outputs=TFTensor(graph=g, shape=list(output_shape_kept_dims), dtype=op.output.dtype))
    reshape2 = TFOperation(graph=g,
                           name="tf.reshape",
                           inputs=reduce.output,
                           attribs=dict(shape=list(output_shape_kept_dims)),
                           outputs=TFTensor(graph=g, shape=list(output_shape_kept_dims), dtype=op.output.dtype))
    div = TFOperation(graph=g,
                      name="tf.divide",
                      inputs=(where.output, reshape2.output),
                      outputs=TFTensor(graph=g, shape=list(output_shape_kept_dims), dtype=op.output.dtype))
    TFOperation(graph=g, name="tf.multiply", inputs=(div.output, reshape1.output), outputs=op.outputs)
    g.remove_operation(op, unlink=True)


def _transform_lrn_grad(g, op):
    # type: (TFGraph, TFOperation)->None

    # def lrn_grad(input_grads, input_image, output_image, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None)

    input_grads, input_image, output_image = op.inputs

    depth_radius = int(op.attribs["depth_radius"])
    bias = op.attribs["bias"]
    alpha = op.attribs["alpha"]
    beta = op.attribs["beta"]

    input_shape = input_image.shape
    input_shape_transposed = input_shape[:-2] + [input_shape[-1], input_shape[-2]]
    input_shape_transposed_padded = input_shape[:-2] + [input_shape[-1] + 2 * depth_radius, input_shape[-2]]
    input_dtype = input_image.dtype

    t_depth_size = TFTensor(graph=g, shape=[], dtype=input_dtype, data=2.0 * depth_radius + 1.0)
    t_alpha = TFTensor(graph=g, shape=[], dtype=input_dtype, data=alpha)
    t_bias = TFTensor(graph=g, shape=[], dtype=input_dtype, data=bias)
    t_beta = TFTensor(graph=g, shape=[], dtype=input_dtype, data=beta)
    t_beta_minus_1 = TFTensor(graph=g, shape=[], dtype=input_dtype, data=beta - 1.0)
    const2 = TFTensor(graph=g, shape=[], dtype=input_dtype, data=2.0)

    tensor0 = input_image
    op1 = TFOperation(graph=g,
                      name="tf.square",
                      inputs=tensor0,
                      outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    op2 = TFOperation(graph=g,
                      name="tf.transpose",
                      inputs=op1.output,
                      attribs=dict(perm=[0, 1, 3, 2], conjugate=False),
                      outputs=TFTensor(graph=g, shape=list(input_shape_transposed), dtype=input_dtype))
    op3 = TFOperation(graph=g,
                      name="tf.pad",
                      inputs=op2.output,
                      attribs=dict(mode="CONSTANT",
                                   paddings=[(0, 0), (0, 0), (depth_radius, depth_radius), (0, 0)],
                                   constant_values=0),
                      outputs=TFTensor(graph=g,
                                       name=None,
                                       shape=list(input_shape_transposed_padded),
                                       dtype=input_dtype))
    op4 = TFOperation(graph=g,
                      name="_avg_pool",
                      inputs=op3.output,
                      attribs=dict(padding="VALID",
                                   size=[1, 1, 2 * depth_radius + 1, 1],
                                   stride=[1, 1, 1, 1],
                                   data_format="NHWC"),
                      outputs=TFTensor(graph=g, shape=list(input_shape_transposed), dtype=input_dtype))
    op5 = TFOperation(graph=g,
                      name="tf.multiply",
                      inputs=(t_depth_size, op4.output),
                      outputs=TFTensor(graph=g, shape=list(input_shape_transposed), dtype=input_dtype))
    op6 = TFOperation(graph=g,
                      name="tf.transpose",
                      inputs=op5.output,
                      attribs=dict(perm=[0, 1, 3, 2], conjugate=False),
                      outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    op7 = TFOperation(graph=g,
                      name="tf.multiply",
                      inputs=(t_alpha, op6.output),
                      outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    op8 = TFOperation(graph=g,
                      name="tf.add",
                      inputs=(t_bias, op7.output),
                      outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    op9 = TFOperation(graph=g,
                      name="tf.pow",
                      inputs=(op8.output, t_beta),
                      outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    tensor10 = input_grads
    op11 = TFOperation(graph=g,
                       name="tf.divide",
                       inputs=(tensor10, op9.output),
                       outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    op12 = TFOperation(graph=g,
                       name="tf.negative",
                       inputs=tensor0,
                       outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    op13 = TFOperation(graph=g,
                       name="tf.divide",
                       inputs=(op12.output, op9.output),
                       outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    op14 = TFOperation(graph=g,
                       name="tf.divide",
                       inputs=(op13.output, op9.output),
                       outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    op15 = TFOperation(graph=g,
                       name="tf.multiply",
                       inputs=(tensor10, op14.output),
                       outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    op16 = TFOperation(graph=g,
                       name="tf.multiply",
                       inputs=(op15.output, t_beta),
                       outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    tensor17 = t_beta_minus_1
    op18 = TFOperation(graph=g,
                       name="tf.pow",
                       inputs=(op8.output, tensor17),
                       outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    op19 = TFOperation(graph=g,
                       name="tf.multiply",
                       inputs=(op16.output, op18.output),
                       outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    op20 = TFOperation(graph=g,
                       name="tf.multiply",
                       inputs=(t_alpha, op19.output),
                       outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    op21 = TFOperation(graph=g,
                       name="tf.transpose",
                       attribs=dict(conjugate=False, perm=[0, 1, 3, 2]),
                       inputs=op20.output,
                       outputs=TFTensor(graph=g, shape=list(input_shape_transposed), dtype=input_dtype))
    op22 = TFOperation(graph=g,
                       name="tf.multiply",
                       inputs=(t_depth_size, op21.output),
                       outputs=TFTensor(graph=g, shape=list(input_shape_transposed), dtype=input_dtype))
    op23 = TFOperation(graph=g,
                       name="_avg_pool_grad",
                       inputs=op22.output,
                       attribs=dict(padding="VALID",
                                    size=[1, 1, int(2 * depth_radius + 1), 1],
                                    stride=[1, 1, 1, 1],
                                    orig_input_shape=list(input_shape_transposed_padded),
                                    data_format="NHWC"),
                       outputs=TFTensor(graph=g, shape=list(input_shape_transposed_padded), dtype=input_dtype))
    op24 = TFOperation(graph=g,
                       name="tf.slice",
                       inputs=op23.output,
                       attribs=dict(begin=[0, 0, depth_radius, 0], size=[-1, -1, input_shape[-1], -1]),
                       outputs=TFTensor(graph=g, shape=list(input_shape_transposed), dtype=input_dtype))
    op25 = TFOperation(graph=g,
                       name="tf.transpose",
                       inputs=op24.output,
                       attribs=dict(conjugate=False, perm=[0, 1, 3, 2]),
                       outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    op26 = TFOperation(graph=g,
                       name="tf.multiply",
                       inputs=(tensor0, const2),
                       outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    op27 = TFOperation(graph=g,
                       name="tf.multiply",
                       inputs=(op25.output, op26.output),
                       outputs=TFTensor(graph=g, shape=list(input_shape), dtype=input_dtype))
    TFOperation(graph=g,
                name="tf.add",
                inputs=(op11.output, op27.output),
                outputs=op.outputs)
    g.remove_operation(op, unlink=True)


# Many to one

def _transform_bias_add_conv(g):
    # type: (TFGraph)->None

    conv_output = matcher.Tensor()
    conv = matcher.Operation(name=["_conv", "_planewise_conv", "_separable_conv", "_deconv", "_planewise_deconv"],
                             inputs={2: None},
                             outputs={0: conv_output})
    add = matcher.Operation(name="tf.nn.bias_add", inputs={0: conv_output})

    matcher.replace(g, add, lambda m: TFOperation(graph=g,
                                                  name=m[conv].name,
                                                  inputs=(m[conv].inputs[0], m[conv].inputs[1], m[add].inputs[1]),
                                                  attribs=m[conv].attribs,
                                                  outputs=m[add].outputs))


def _transform_add_conv(g):
    # type: (TFGraph)->None

    def is_bias(conv_op, bias_tensor):
        if _is_nhwc(conv_op):
            return bias_tensor.rank == 1
        else:
            return (bias_tensor.rank == conv_op.output.rank
                    and all(i == 1 or s == 1 for i, s in enumerate(bias_tensor.shape)))

    input, filter, bias, conv_output = matcher.tensors(4)
    conv = matcher.Operation(name=["_conv", "_planewise_conv", "_separable_conv", "_deconv", "_planewise_deconv"],
                             inputs=(input, filter),
                             outputs=conv_output)
    add = matcher.Operation(name="tf.add", inputs={conv_output, bias})

    matcher.replace(g, add,
                    lambda m: TFOperation(graph=g,
                                          name=m[conv].name,
                                          inputs=(m[input], m[filter], m[bias]),
                                          attribs=m[conv].attribs,
                                          outputs=m[add].outputs),
                    lambda m: is_bias(m[conv], m[bias]))


def _transform_space_to_batch_depthwise_conv(g):
    # type: (TFGraph)->None

    input, block_shape1, block_shape2, paddings, crops, filter, batched, filtered, output = matcher.tensors(9)
    space_to_batch = matcher.Operation(name=["tf.space_to_batch", "tf.space_to_batch_nd"],
                                       inputs=(input, block_shape1, paddings), outputs=batched)
    conv = matcher.Operation(name="_planewise_conv",
                             inputs=(batched, filter), outputs=filtered)
    batch_to_space = matcher.Operation(name=["tf.batch_to_space", "tf.batch_to_space_nd"],
                                       inputs=(filtered, block_shape2, crops), outputs=output)

    matcher.replace(g, batch_to_space,
                    lambda m: TFOperation(graph=g,
                                          name=m[conv].name,
                                          inputs=(m[input], m[filter]),
                                          attribs=dict(m[conv].attribs, dilation=m[block_shape1].data.tolist(), padding='SAME'),
                                          outputs=m[batch_to_space].outputs))


def _transform_pad(g):
    # type: (TFGraph)->None
    input, pad_output = matcher.tensors(2)
    pad = matcher.Operation(name="tf.pad", inputs=input, outputs=pad_output)
    conv = matcher.Operation(name=["_conv", "_planewise_conv", "_separable_conv",
                                   "_max_pool", "_max_pool_with_index", "_avg_pool"],
                             inputs={0: pad_output})
    matcher.replace(
        g, conv,
        lambda m: TFOperation(graph=g,
                              name=m[conv].name,
                              inputs=(m[input],) + tuple(m[conv].inputs[1:]),
                              attribs=utils.updated_dict(m[conv].attribs,
                                                         padding=[tuple(p) for p in m[pad].attribs["paddings"]],
                                                         _border=m[pad].attribs["mode"]),
                              outputs=m[conv].outputs),
        lambda m: m[conv].attribs["padding"].upper() == 'VALID'
    )


def _apply_block_shape(shape, block_shape, data_format, crops):
    if data_format is None:
        data_format = "NHWC"
    if crops is None:
        crops = [[0, 0] for _ in range(len(block_shape))]

    shape = list(shape)
    rank = len(shape)

    assert shape[0] % int(np.prod(block_shape)) == 0, "batch size is not divisable by prod(block_shape)"
    shape[0] //= int(np.prod(block_shape))

    spatial_begin = (2 if data_format.upper() == "NCHW" else 1)
    for i in range(0, rank - 2):
        shape[spatial_begin + i] *= block_shape[i]
        shape[spatial_begin + i] -= (crops[i][0] + crops[i][1])
    return shape


def _transform_cgf_stb(g):
    # type: (TFGraph)->None
    orig_input, output_grad, stb1_output, stb2_output = matcher.tensors(4)
    stb1 = matcher.Operation(name=["tf.space_to_batch", "tf.space_to_batch_nd"],
                             inputs=orig_input,
                             outputs=stb1_output)
    _stb2 = matcher.Operation(name=["tf.space_to_batch", "tf.space_to_batch_nd"],
                              inputs=output_grad,
                              outputs=stb2_output)
    cgf = matcher.Operation(name="_conv_grad_filter", inputs=(stb1_output, stb2_output))
    pattern = matcher.SetParams(cgf, allow_multi_consumer_inside=True)

    def action(m):
        # type: (matcher.Match)->None
        block_shape = (m[stb1].attribs["block_shape"] if m[stb1].name.endswith("_nd")
                       else [m[stb1].attribs["block_size"]] * len(m[stb1].attribs["paddings"]))
        padding = "SAME" if utils.recursive_any(m[stb1].attribs["paddings"], lambda x: x > 0) else "VALID"

        TFOperation(graph=g,
                    name=m[cgf].name,
                    inputs=(m[orig_input], m[output_grad]),
                    attribs=utils.updated_dict(m[cgf].attribs,
                                               dilation=block_shape,
                                               padding=padding),
                    outputs=m[cgf].outputs)
        g.remove_operation(m[cgf], unlink=True)

    matcher.for_each(g, pattern, action)


def _transform_bts_conv_stb(g):
    # type: (TFGraph)->None

    input, filter, stb_output, conv_output = matcher.tensors(4)
    stb = matcher.Operation(name=["tf.space_to_batch", "tf.space_to_batch_nd"], inputs=input, outputs=stb_output)
    conv = matcher.Operation(name=["_conv", "_deconv"], inputs=(stb_output, filter), outputs=conv_output)
    bts = matcher.Operation(name=["tf.batch_to_space", "tf.batch_to_space_nd"], inputs=conv_output)

    def replacement(m):
        # type: (matcher.Match)->TFOperation
        block_shape = (m[stb].attribs["block_shape"] if m[stb].name.endswith("_nd")
                       else [m[stb].attribs["block_size"]] * len(m[stb].attribs["paddings"]))
        if len(block_shape) > m[conv].output.rank - 2:  # In the NCHW case there might be a leading 1
            block_shape = block_shape[-(m[conv].output.rank - 2):]
        if m[conv].name == "_conv":
            padding = "SAME" if utils.recursive_any(m[stb].attribs["paddings"], lambda x: x > 0) else "VALID"

            return TFOperation(graph=g,
                               name=m[conv].name,
                               inputs=(m[input], m[filter]),
                               attribs=utils.updated_dict(m[conv].attribs, dilation=block_shape, padding=padding),
                               outputs=m[bts].outputs)
        else:
            padding = "SAME" if utils.recursive_any(m[bts].attribs["crops"], lambda x: x > 0) else "VALID"
            output_shape = _apply_block_shape(shape=m[conv].attribs["output_shape"],
                                              block_shape=block_shape,
                                              data_format=m[conv].attribs["data_format"],
                                              crops=m[bts].attribs["crops"])

            return TFOperation(graph=g,
                               name=m[conv].name,
                               inputs=(m[input], m[filter]),
                               attribs=utils.updated_dict(m[conv].attribs,
                                                          dilation=block_shape,
                                                          padding=padding,
                                                          output_shape=output_shape),
                               outputs=m[bts].outputs)

    matcher.replace(g, bts, replacement)


def _replace_tensor_in_outputs(graph, old_tensor, new_tensor):
    # type: (Graph, Tensor, Tensor)->None

    if graph.output_ids is not None:
        graph.outputs = OrderedDict((name, new_tensor if t is old_tensor else t)
                                    for name, t in zip(graph.output_ids, graph.outputs))
    else:
        graph.outputs = [new_tensor if t is old_tensor else t for t in graph.outputs]


def _transform_separate_inputs_and_outputs(tf_graph):
    # type: (TFGraph)->None

    for tensor in list(tf_graph.tensors):
        if tensor in tf_graph.inputs and tensor in tf_graph.outputs:
            output_tensor = TFTensor(graph=tf_graph,
                                     name=None,
                                     shape=list(tensor.shape),
                                     dtype=tensor.dtype,
                                     data=copy.copy(tensor.data))
            TFOperation(graph=tf_graph, name="tf.identity", inputs=tensor, outputs=output_tensor)
            _replace_tensor_in_outputs(tf_graph, tensor, output_tensor)


def _transform_separate_duplicated_outputs(tf_graph):
    # type: (TFGraph)->None

    new_outputs = []
    seen = set()

    for tensor in tf_graph.outputs:
        if tensor in seen:
            new_outputs.append(
                TFOperation(
                    graph=tf_graph,
                    name='tf.identity',
                    inputs=tensor,
                    outputs=TFTensor(
                        graph=tf_graph,
                        shape=list(tensor.shape),
                        dtype=tensor.dtype,
                        data=copy.copy(tensor.data))).output)
        else:
            seen.add(tensor)
            new_outputs.append(tensor)

    if tf_graph.output_ids:
        tf_graph.outputs = OrderedDict([(name, tensor) for name, tensor in zip(tf_graph.output_ids, new_outputs)])
    else:
        tf_graph.outputs = new_outputs
