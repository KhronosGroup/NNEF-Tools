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

from __future__ import division, print_function

from collections import deque

import nnef
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_grad as tf_array_grad
from tensorflow.python.ops import gen_array_ops as tf_array_ops
from tensorflow.python.ops import gen_math_ops as tf_math_ops
from tensorflow.python.ops import gen_nn_ops as tf_nn_ops
from tensorflow.python.ops import math_grad as tf_math_grad

from .. import tf_compat
from ...common import dog
from ...common import nnef_shape_optimizer
from ...common import utils
from ...common.matchers import Op, OpAndArg, OneOf, match_arg_chain
from ...common.utils import get_qualified_name, get_qualified_names


# HELPERS

def unlink_op(op):
    for dn in op.get_arg_nodes():
        if op in dn.consumers:
            dn.consumers.remove(op)
    for dn in op.get_result_nodes():
        if dn.producer is op:
            utils.print_warning("Unlinking op which is still the producer of something")
            dn.producer = None


def unlink_and_push_down_arg(op, converter, is_nnef, arg_name="input"):
    dn_input = op.args[arg_name]
    for dn_arg in op.get_arg_nodes():
        dn_arg.consumers.remove(op)
    replace_in_consumers(op.result, dn_input, converter, is_nnef)


def replace_in_consumers(dn_old, dn_new, converter, is_nnef):
    if is_nnef:
        if dn_old.source_name and dn_old.source_name in converter.output_name_by_tfname:
            dn_new.source_name = dn_old.source_name
    else:
        if dn_old.name in converter.output_name_by_tfname:
            dn_new.name = dn_old.name

    def replace(x):
        if x == dn_old:
            return dn_new
        return x

    for consumer in dn_old.consumers:
        dn_new.consumers.append(consumer)
        consumer.args = utils.recursive_transform(consumer.args, replace)
        consumer.results = utils.recursive_transform(consumer.results, replace)


def merge_up_op(op, arg_name, converter, is_nnef):
    # TODO extra update?
    if is_nnef:
        op.args[arg_name].source_name = op.result.source_name
    else:
        op.args[arg_name].name = op.result.name

    unlink_and_push_down_arg(op, converter, is_nnef, arg_name)


def is_var_or_const_or_placeholder(tfdn):
    return tfdn.producer.name in get_qualified_names([
        tf.Variable,
        tf.get_variable,
        tf.placeholder,
        tf.constant,
        tf.zeros,
        tf.zeros_like,
        tf.ones,
        tf.ones_like
    ])


def normalize_data_format(data_format):
    return utils.normalize_str_upper(data_format)


def with_block_size_applied(shape, block_shape, data_format, crops=None):
    if crops is None:
        crops = [[0, 0] for _ in range(len(block_shape))]

    shape = list(shape)
    input_rank = len(shape)
    if shape[0] % int(np.prod(block_shape)) != 0:
        utils.print_error("batch size is not dividable by prod(block_shape)")
    shape[0] //= int(np.prod(block_shape))
    spatial_begin = (1 if data_format == "NHWC" else 2)
    for i in range(0, input_rank - 2):
        shape[spatial_begin + i] *= block_shape[i]
        shape[spatial_begin + i] -= (crops[i][0] + crops[i][1])
    return shape


def get_tf_reduction_axes(tfop_reduce):
    tfdn_input = tfop_reduce.args.get("input_tensor")
    if tfdn_input is None:
        tfdn_input = tfop_reduce.args["input"]

    rank = len(dog.get_shape_safe(tfdn_input))
    tfaxes = tfop_reduce.args.get('axis')
    if tfaxes is None:
        tfaxes = tfop_reduce.args.get('reduction_indices')
    if tfaxes is None:
        tfaxes = list(range(rank))
    if isinstance(tfaxes, tuple):
        tfaxes = list(tfaxes)
    if not isinstance(tfaxes, list):
        tfaxes = [tfaxes]
    tfaxes = [rank + axe if axe < 0 else axe for axe in tfaxes]
    return tfaxes


def reduced_shape(input_shape, axes):
    input_shape = np.array(input_shape)
    input_shape[np.array(axes)] = 1
    return input_shape.tolist()


def handle_minus_one_in_reshape_shape(input_shape, reshape_shape):
    if isinstance(reshape_shape, (list, tuple)) and -1 in reshape_shape:
        idx = reshape_shape.index(-1)
        reshape_shape2 = list(reshape_shape)
        reshape_shape2[idx] = 1
        rem = int(np.prod(input_shape)) % int(np.prod(reshape_shape2))
        if rem != 0:
            utils.print_error("reshape -1 calculation failed, shapes not dividable")
        div = int(int(np.prod(input_shape)) / int(np.prod(reshape_shape2)))
        reshape_shape2[idx] = div
        return reshape_shape2
    return list(reshape_shape)


# TRANSFORMS

def transform_casts(tfops, converter):
    tfops2 = []
    for tfop in tfops:
        if tfop.name == get_qualified_name(tf.cast):
            if tfop.args["x"].dtype.startswith('float') and tfop.args["dtype"] == tf.bool:
                tfops2.append(tfop)
            elif tfop.args["x"].dtype == 'bool' and tfop.args["dtype"].startswith('float'):
                x = tfop.args["x"]
                res = tfop.result

                zeros = dog.OperationNode(get_qualified_name(tf.constant))
                zeros.add_arg("value", 0.0)
                zeros.add_arg("dtype", tfop.args["dtype"])
                zeros.add_arg("shape", dog.get_shape_safe(x))
                zeros_res = dog.DataNode()
                zeros_res.shape = dog.get_shape_safe(x)
                zeros_res.dtype = tfop.args["dtype"]
                zeros.add_result("result", zeros_res)

                ones = dog.OperationNode(get_qualified_name(tf.constant))
                ones.add_arg("value", 1.0)
                ones.add_arg("dtype", tfop.args["dtype"])
                ones.add_arg("shape", dog.get_shape_safe(x))
                ones_res = dog.DataNode()
                ones_res.shape = dog.get_shape_safe(x)
                ones_res.dtype = tfop.args["dtype"]
                ones.add_result("result", ones_res)

                where = dog.OperationNode(get_qualified_name(tf.where))
                where.add_arg("condition", x)
                where.add_arg("x", ones.result)
                where.add_arg("y", zeros.result)
                where.add_result("result", res)

                unlink_op(tfop)

                tfops2.append(zeros)
                tfops2.append(ones)
                tfops2.append(where)
            else:
                tfops2.append(tfop)
        else:
            tfops2.append(tfop)
    return tfops2


def transform_eliminate_tf_passthroughs(tfops, converter):
    input_arg_name_by_passthrough_name = {
        k: v.input_arg_name
        for k, v in converter.converter_by_name.items()
        if v.is_passthrough()
    }

    return transform_eliminate_passthroughs(tfops, input_arg_name_by_passthrough_name, converter, is_nnef=False)


def transform_eliminate_nnef_passthroughs(nnefops, converter):
    input_arg_name_by_fun_name = {
        "_nnef_passthrough": "input"
    }

    return transform_eliminate_passthroughs(nnefops, input_arg_name_by_fun_name, converter, is_nnef=True)


def transform_eliminate_passthroughs(ops, input_arg_name_by_fun_name, converter, is_nnef):
    passthrough_fun_names = list(input_arg_name_by_fun_name.keys())
    ops2 = []
    for op in ops:
        if op.name in passthrough_fun_names:
            unlink_and_push_down_arg(op, converter, arg_name=input_arg_name_by_fun_name[op.name], is_nnef=is_nnef)
        else:
            ops2.append(op)
    return ops2


def transform_bts_conv_stb(tfops, converter):
    to_remove = set()
    to_unlink_ex = set()  # set of (tfop, link_arg_name)

    matches = []

    for tfop in tfops:
        # eg: tf.batch_to_space_nd(input=tf.nn.conv2d(input=tf.space_to_batch_nd(...), ...), ...)
        match = match_arg_chain(tfop, [
            OpAndArg(tf.batch_to_space_nd, "input"),
            OneOf(OpAndArg(tf.nn.conv2d, "input"),
                  OpAndArg(tf.nn.conv2d_backprop_input, "out_backprop")),
            Op(OneOf(tf.space_to_batch_nd, tf.constant))
        ], only_single_consumer=False, tf_hacks=True)

        if match:
            matches.append(match)
            to_remove.add(match[0])
            to_remove.add(match[4])

    for bts, _, conv, _, stb_or_constant in matches:
        if stb_or_constant.name == get_qualified_name(tf.space_to_batch_nd):
            to_unlink_ex.add((stb_or_constant, "input"))
            block_shape = utils.to_list(stb_or_constant.args["block_shape"])
            conv.set_arg('dilations', block_shape)
            crops = None

            if "backprop" in conv.name:
                conv.set_arg('padding', 'SAME' if utils.has_greater_than_0(bts.args['crops']) else 'VALID')
                crops = bts.args['crops']
            else:
                conv.set_arg(
                    'padding', 'SAME' if utils.has_greater_than_0(stb_or_constant.args['paddings']) else 'VALID')

            if conv.name == get_qualified_name(tf.nn.conv2d_backprop_input):
                conv.args["input_sizes"] = with_block_size_applied(
                    conv.args["input_sizes"], block_shape, conv.args["data_format"], crops=crops)

        merge_up_op(bts, "input", converter, is_nnef=False)

    matches = []

    for tfop in tfops:
        # eg: tf.nn.conv2d_backprop_filter(input=tf.space_to_batch_nd(...), ...)

        match = match_arg_chain(tfop, [
            OpAndArg(tf.nn.conv2d_backprop_filter, "input"),
            Op(tf.space_to_batch_nd)
        ], only_single_consumer=False, tf_hacks=True)

        if match:
            matches.append(match)
            to_remove.add(match[2])

        match = match_arg_chain(tfop, [
            OpAndArg(tf.nn.conv2d_backprop_filter, "out_backprop"),
            Op(tf.space_to_batch)
        ], only_single_consumer=False, tf_hacks=True)

        if match:
            matches.append(match)
            to_remove.add(match[2])

    for conv2d_backprop_filter, conv2d_backprop_filter_arg_name, stb in matches:
        to_unlink_ex.add((stb, "input"))

        if conv2d_backprop_filter_arg_name == "input":
            if "block_shape" in stb.args:
                block_shape = utils.to_list(stb.args["block_shape"])
            else:
                block_shape = [stb.args["block_size"]] * len(stb.args['paddings'])
            conv2d_backprop_filter.set_arg('dilations', block_shape)
            conv2d_backprop_filter.set_arg(
                'padding', 'SAME' if utils.has_greater_than_0(stb.args['paddings']) else 'VALID')

    for tfop, link_arg_name in to_unlink_ex:
        unlink_and_push_down_arg(tfop, converter, arg_name=link_arg_name, is_nnef=False)

    return [tfop for tfop in tfops if tfop not in to_remove]


bias_add_propagation_sinks = [
    tf.nn.conv1d,
    tf.nn.conv2d,
    tf.nn.conv3d,
    tf.nn.convolution,
    tf.nn.conv2d_transpose,
    tf.nn.conv3d_transpose,
    tf.nn.depthwise_conv2d,
    tf.nn.depthwise_conv2d_native
]


def transform_add_conv(tfops, converter):
    to_remove = set()
    matches = []

    other_arg_name = {
        "value": "bias",
        "x": "y",
        "y": "x"
    }

    for tfop in tfops:
        # eg: tf.nn.bias_add(value=tf.nn.conv2d(...), ...)
        match = match_arg_chain(tfop, [
            OneOf(OpAndArg(tf.nn.bias_add, "value"),
                  OpAndArg(OneOf(tf.add, tf_math_ops.add), OneOf("x", "y"))),
            Op(OneOf(*bias_add_propagation_sinks))
        ], tf_hacks=True)
        if match:
            add, add_arg, conv = match
            tfdn_bias = add.args[other_arg_name[add_arg]]
            if not len(tfdn_bias.shape) == 1:  # TODO check if correct
                continue
            to_remove.add(add)

            matches.append(match)

    for add, add_arg, conv in matches:
        tfdn_bias = add.args[other_arg_name[add_arg]]
        merge_up_op(add, add_arg, converter, is_nnef=False)
        conv.add_arg("_bias", tfdn_bias)

    def transform(elem):
        for add, _, conv in matches:
            if elem is add:
                return conv
            if elem is conv:
                return utils.REMOVE
        return elem

    return utils.recursive_transform(tfops, transform)


padding_propagation_sinks = [
    tf.nn.conv1d,
    tf.nn.conv2d,
    tf.nn.conv3d,
    tf.nn.convolution,
    tf.nn.depthwise_conv2d,
    tf.nn.depthwise_conv2d_native,
    tf.nn.separable_conv2d,
    tf.nn.max_pool,
    tf.nn.max_pool_with_argmax,
    tf.nn.avg_pool
]


def transform_pad(tfops, converter):
    to_remove = set()
    matches = []

    for tfop in tfops:
        # eg: tf.nn.conv2d(input=tf.pad(...), ...)
        match = match_arg_chain(tfop, [
            OpAndArg(OneOf(*padding_propagation_sinks),
                     OneOf("input", "value")),
            Op(tf.pad)
        ], tf_hacks=True)

        if match:
            matches.append(match)
            to_remove.add(match[2])

    for conv_or_pool, _, pad in matches:
        unlink_and_push_down_arg(pad, converter, arg_name="tensor", is_nnef=False)
        if conv_or_pool.args["padding"].upper() != 'VALID':
            utils.print_error(
                "only 'VALID' padding is accepted after an explicit 'pad' operation in {}".format(conv_or_pool.name))
        conv_or_pool.set_arg("padding", [tuple(p) for p in pad.args["paddings"]])
        conv_or_pool.add_arg("_border", pad.args["mode"])

    return [tfop for tfop in tfops if tfop not in to_remove]


FUSED_BATCH_NORM_VARIANCE_CORRECTION_ENABLED = True


def transform_fused_batch_norm(tfops, converter):
    tfops2 = []
    for tfop in tfops:
        if tfop.name in [get_qualified_name(tf_nn_ops._fused_batch_norm),
                         get_qualified_name(tf_compat.gen_nn_ops_fused_batch_norm_v2)]:
            tfdn_x = tfop.args['x']
            data_format = normalize_data_format(tfop.args['data_format'])
            input_shape = dog.get_shape_safe(tfdn_x)
            rank = len(input_shape)
            size = int(np.prod(input_shape))
            depth_idx = 1 if data_format == "NCHW" else rank - 1
            depth = input_shape[depth_idx]
            rest_size = int(size / depth)

            res_y = tfop.results["result0"]
            res_batch_mean = tfop.results["result1"]
            res_batch_var = tfop.results["result2"]
            res_saved_mean = tfop.results["result3"]
            res_saved_var = tfop.results["result4"]

            if tfop.args['is_training']:
                tfop_moments = dog.OperationNode(get_qualified_name(tf.nn.moments))
                tfop_moments.add_arg('x', tfdn_x)
                tfop_moments.add_arg('axes', utils.without(range(rank), depth_idx))
                tfop_moments.add_arg('keep_dims', False)
                if FUSED_BATCH_NORM_VARIANCE_CORRECTION_ENABLED:
                    tfdn_biased_var = dog.DataNode()
                    tfdn_biased_var.shape = dog.get_shape_safe(res_batch_var)
                else:
                    tfdn_biased_var = res_batch_var
                tfop_moments.add_result('result0', res_batch_mean)
                tfop_moments.add_result('result1', tfdn_biased_var)
                tfops2.append(tfop_moments)

                if FUSED_BATCH_NORM_VARIANCE_CORRECTION_ENABLED:
                    tfop_mul = dog.OperationNode(get_qualified_name(tf.multiply))
                    tfop_mul.add_arg('x', tfdn_biased_var)
                    tfop_mul.add_arg('y', float(rest_size) / max(rest_size - 1, 1))
                    tfop_mul.add_result('result0', res_batch_var)
                    tfops2.append(tfop_mul)

                replace_in_consumers(res_saved_mean, res_batch_mean, converter, is_nnef=False)
                replace_in_consumers(res_saved_var, res_batch_var, converter, is_nnef=False)

                tfop_batch_norm = dog.OperationNode(get_qualified_name(tf.nn.batch_normalization))
                tfop_batch_norm.add_arg('x', tfdn_x)
                tfop_batch_norm.add_arg('mean', res_batch_mean)
                tfop_batch_norm.add_arg('variance', tfdn_biased_var)
                tfop_batch_norm.add_arg('offset', tfop.args["offset"])
                tfop_batch_norm.add_arg('scale', tfop.args["scale"])
                tfop_batch_norm.add_arg('variance_epsilon', tfop.args["epsilon"])
                # just for internal processing,
                # originally tf.nn.batch_normalization does not have a data_format argument
                tfop_batch_norm.add_arg('_data_format', tfop.args["data_format"])
                tfop_batch_norm.add_result('result0', res_y)
                tfops2.append(tfop_batch_norm)

                tfop.remove_result("result0")
                tfop.remove_result("result1")
                tfop.remove_result("result2")
                tfop.remove_result("result3")
                tfop.remove_result("result4")
                unlink_op(tfop)
            else:
                tfop_batch_norm = dog.OperationNode(get_qualified_name(tf.nn.batch_normalization))
                tfop_batch_norm.add_arg('x', tfdn_x)
                tfop_batch_norm.add_arg('mean', tfop.args["mean"])
                tfop_batch_norm.add_arg('variance', tfop.args["variance"])
                tfop_batch_norm.add_arg('offset', tfop.args["offset"])
                tfop_batch_norm.add_arg('scale', tfop.args["scale"])
                tfop_batch_norm.add_arg('variance_epsilon', tfop.args["epsilon"])
                # just for internal processing,
                # originally tf.nn.batch_normalization does not have a data_format argument
                tfop_batch_norm.add_arg('data_format', tfop.args["data_format"])
                tfop_batch_norm.add_result('result0', tfop.results["result0"])

                replace_in_consumers(res_batch_mean, tfop.args['mean'], converter, is_nnef=False)
                replace_in_consumers(res_batch_var, tfop.args['variance'], converter, is_nnef=False)
                replace_in_consumers(res_saved_mean, tfop.args['mean'], converter, is_nnef=False)
                replace_in_consumers(res_saved_var, tfop.args['variance'], converter, is_nnef=False)

                tfop.remove_result("result0")
                tfop.remove_result("result1")
                tfop.remove_result("result2")
                tfop.remove_result("result3")
                tfop.remove_result("result4")
                unlink_op(tfop)

                tfops2.append(tfop_batch_norm)
        else:
            tfops2.append(tfop)
    return tfops2


def transform_fill_to_constant(ops, converter):
    for op in ops:
        if op.name == get_qualified_name(tf.fill):
            op.add_arg("shape", op.args["dims"])
            del op.args["dims"]
            op.name = get_qualified_name(tf.constant)
    return ops


def transform_evaluate_shapes(ops, converter):
    ops2 = list(ops)
    # Perm is not really good
    shapelike_args = ["shape", "output_shape", "axis", "axes", "reduction_indices", "dims", "size", "begin", "end",
                      "strides", "perm", "filter_sizes", "input_sizes", "dilations", "block_shape", "orig_input_shape",
                      "crops", "max_output_size"]
    for op in ops:
        arg_names = [arg for arg in shapelike_args if arg in op.args]
        for arg_name in arg_names:
            arg = op.args[arg_name]

            def evaluate(arg):
                if isinstance(arg, dog.DataNode):
                    evaluated_arg = utils.tf_constant_value(converter.get_tftensor_by_tfdn(arg))
                    if evaluated_arg is not None:
                        evaluated_arg = evaluated_arg.tolist()
                    if isinstance(evaluated_arg, (list, tuple)) and None in evaluated_arg:
                        evaluated_arg = None
                    if evaluated_arg is None:
                        utils.print_error("Could not evaluate {}={} for {}".format(arg_name, arg.name, op.name))
                    if arg.producer in ops2:
                        ops2.remove(arg.producer)  # TODO consider removing this line
                    return evaluated_arg
                else:
                    return arg

            op.set_arg(arg_name, utils.recursive_transform(arg, evaluate))
    return ops2


# TODO consider removing this transformation
def transform_evaluate_multiples(ops, converter):
    ops2 = list(ops)
    for op in ops:
        if op.name == get_qualified_name(tf.tile):
            arg = op.args["multiples"]
            if isinstance(arg, dog.DataNode):
                evaluated_arg = utils.tf_constant_value(converter.get_tftensor_by_tfdn(arg))
                if evaluated_arg is not None:
                    evaluated_arg = evaluated_arg.tolist()
                if arg.producer in ops2:
                    ops2.remove(arg.producer)  # TODO consider removing this line
                op.set_arg("multiples", evaluated_arg)
    return ops2


def transform_evaluate_other_constants(ops, converter):
    for op in ops:
        if op.name == get_qualified_name(tf.constant):
            value = op.args['value']
            if isinstance(value, dog.DataNode):
                value = utils.tf_constant_value(converter.get_tftensor_by_tfdn(value))
                if value is not None:
                    value = value.tolist()
                else:
                    utils.print_error("Could not evaluate constant: {}, resetting to 0.".format(op.result.name))
                    value = 0.0
                op.set_arg('value', value)
        elif op.name == get_qualified_name(tf.pad) or op.name == get_qualified_name(tf.space_to_batch_nd):
            if op.name == get_qualified_name(tf.pad):
                dn_tensor = op.args["tensor"]
            else:
                dn_tensor = op.args["input"]
            paddings = op.args["paddings"]
            rank = len(dog.get_shape_safe(dn_tensor))

            if isinstance(paddings, dog.DataNode):
                paddings = utils.tf_constant_value(converter.get_tftensor_by_tfdn(paddings))
                if paddings is not None:
                    paddings = paddings.tolist()
                else:
                    utils.print_error("Paddings parameter cannot be evaluated, resetting to 0.")
                    paddings = [[0, 0] for _ in range(rank)]
                op.set_arg("paddings", paddings)
            elif isinstance(paddings, np.ndarray):
                op.set_arg("paddings", paddings.tolist())
    return ops


def transform_remove_unreachables(ops, output_tfdn_names, converter):
    visited = set()

    q = deque()

    output_tfops = []
    for tfop in ops:
        if any([result.name and result.name in output_tfdn_names for result in tfop.get_result_nodes()]):
            output_tfops.append(tfop)

    for tfop in utils.unique(output_tfops):
        visited.add(tfop)
        q.append(tfop)

    while len(q) > 0:
        tfop = q.popleft()

        for tfdn in tfop.get_arg_nodes():
            if tfdn.producer and tfdn.producer not in visited:
                visited.add(tfdn.producer)
                q.append(tfdn.producer)

    return [tfop for tfop in ops if tfop in visited]


def mask_to_array(mask, rank):
    arr = list(reversed([int(d) for d in bin(mask)[2:]]))
    if len(arr) > rank:
        utils.print_error("Invalid mask: {}, rank: {}, resetting to empty mask".format(mask, rank))
        arr = []
    return arr + [0] * (rank - len(arr))


def mask_to_int(mask, rank):
    res = utils.int_log2(mask)
    if res == -2 or res >= rank:
        utils.print_error("Invalid mask: {}, resetting to empty mask".format(mask))
        res = -1
    return res


def _decompose_strided_slice(tf_input_shape, tfbegin, tfend, tf_ellipsis_mask, tf_new_axis_mask, tf_shrink_axis_mask,
                             tf_begin_mask, tf_end_mask):
    tf_input_rank = len(tf_input_shape)
    tf_mask_rank = len(tfbegin)
    tf_ellipsis_mask = mask_to_int(tf_ellipsis_mask, tf_mask_rank)
    tf_new_axis_mask = mask_to_array(tf_new_axis_mask, tf_mask_rank)
    tf_shrink_axis_mask = mask_to_array(tf_shrink_axis_mask, tf_mask_rank)
    tf_begin_mask = mask_to_array(tf_begin_mask, tf_mask_rank)
    tf_end_mask = mask_to_array(tf_end_mask, tf_mask_rank)
    tf_output_rank_before_shrink = tf_input_rank + sum(tf_new_axis_mask)

    arrays = [tfbegin, tfend, tf_new_axis_mask, tf_shrink_axis_mask, tf_begin_mask, tf_end_mask]
    if tf_ellipsis_mask >= 0:
        if tf_output_rank_before_shrink == tf_mask_rank - 1:
            for arr in arrays:
                del arr[tf_ellipsis_mask]
        elif tf_output_rank_before_shrink == tf_mask_rank:
            for arr in arrays:
                arr[tf_ellipsis_mask] = 0
            tf_begin_mask[tf_ellipsis_mask] = 1
            tf_end_mask[tf_ellipsis_mask] = 1
        elif tf_output_rank_before_shrink > tf_mask_rank:
            d = tf_output_rank_before_shrink - tf_mask_rank
            for arr in arrays:
                arr[tf_ellipsis_mask] = 0
                for _ in range(d):
                    arr.insert(tf_ellipsis_mask, 0)
            for i in range(tf_ellipsis_mask, tf_ellipsis_mask + d + 1):
                tf_begin_mask[i] = 1
                tf_end_mask[i] = 1
    elif tf_mask_rank < tf_output_rank_before_shrink:
        for i in range(tf_mask_rank, tf_output_rank_before_shrink):
            for arr in arrays:
                arr.append(0)
            tf_begin_mask[i] = 1
            tf_end_mask[i] = 1

    for arr in arrays:
        assert len(arr) == tf_output_rank_before_shrink

    strided_slice_result_shape = [0] * tf_input_rank
    tf_shape_before_shrink = [0] * tf_output_rank_before_shrink

    shapeIdx = 0
    ssrsIdx = 0
    for i in range(tf_output_rank_before_shrink):
        if tfbegin[i] < 0:
            tfbegin[i] += tf_input_shape[shapeIdx]
        if tfend[i] < 0:
            tfend[i] += tf_input_shape[shapeIdx]

        if tf_begin_mask[i] == 1:
            tfbegin[i] = 0
        if tf_end_mask[i] == 1:
            tfend[i] = tf_input_shape[shapeIdx]

        if tf_new_axis_mask[i] == 1:
            tf_shape_before_shrink[i] = 1
        else:
            tf_shape_before_shrink[i] = tfend[i] - tfbegin[i]
            strided_slice_result_shape[ssrsIdx] = tfend[i] - tfbegin[i]
            ssrsIdx += 1

        if tf_new_axis_mask[i] == 0:
            shapeIdx += 1

    if utils.has_greater_than_0(tf_new_axis_mask) or utils.has_greater_than_0(tf_shrink_axis_mask):
        opt_reshape_shape = []
        for i in range(tf_output_rank_before_shrink):
            if tf_shrink_axis_mask[i] == 0:
                opt_reshape_shape.append(tf_shape_before_shrink[i])
    else:
        opt_reshape_shape = None

    strided_slice_begin, strided_slice_end = utils.zip_inverse(2, [
        (b, e)
        for b, e, nm in zip(tfbegin, tfend, tf_new_axis_mask)
        if not nm
    ])

    return strided_slice_begin, strided_slice_end, strided_slice_result_shape, opt_reshape_shape


def transform_strided_slice(ops, converter):
    ops2 = []
    for tfop in ops:
        if tfop.name == get_qualified_name(tf.strided_slice):
            tfstrides = tfop.args['strides']
            if utils.has_not_equal_1(tfstrides):
                utils.print_error("Only strides=1 is supported for tf.strided_slice, got: {}".format(tfstrides))

            tfdn = tfop.args['input_']

            strided_slice_begin, strided_slice_end, strided_slice_result_shape, opt_reshape_shape = _decompose_strided_slice(
                tf_input_shape=dog.get_shape_safe(tfdn),
                tfbegin=tfop.args['begin'],
                tfend=tfop.args['end'],
                tf_ellipsis_mask=tfop.args['ellipsis_mask'],
                tf_new_axis_mask=tfop.args['new_axis_mask'],
                tf_shrink_axis_mask=tfop.args['shrink_axis_mask'],
                tf_begin_mask=tfop.args['begin_mask'],
                tf_end_mask=tfop.args['end_mask']
            )

            tfop.args['begin'], tfop.args['end'] = strided_slice_begin, strided_slice_end

            del tfop.args['ellipsis_mask']
            del tfop.args['begin_mask']
            del tfop.args['end_mask']
            del tfop.args['new_axis_mask']
            del tfop.args['shrink_axis_mask']

            if opt_reshape_shape is not None:
                final_result = tfop.result

                result_name = list(tfop.results.keys())[0]

                tfdn_intermediate = dog.DataNode()
                tfdn_intermediate.shape = strided_slice_result_shape
                tfop.set_result(result_name, tfdn_intermediate)

                tfop_reshape = dog.OperationNode(get_qualified_name(tf.reshape))
                tfop_reshape.add_arg("tensor", tfop.result)
                tfop_reshape.add_arg("shape", opt_reshape_shape)
                tfop_reshape.add_result("result", final_result)

                ops2.append(tfop)
                ops2.append(tfop_reshape)
            else:
                ops2.append(tfop)
        else:
            ops2.append(tfop)

    return ops2


def transform_strided_slice_grad(ops, converter):
    # def strided_slice_grad(shape, begin, end, strides, dy, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
    #   return pad(reshape(dy, ...), ...)

    ops2 = []
    for op in ops:
        if op.name == get_qualified_name(tf_array_ops.strided_slice_grad):
            dn_dy = op.args["dy"]

            strides = op.args['strides']
            if utils.has_not_equal_1(strides):
                utils.print_error("Only strides=1 is supported for strided_slice_grad, got: {}".format(strides))

            input_shape = op.args["shape"]

            strided_slice_begin, strided_slice_end, strided_slice_result_shape, opt_reshape_shape = _decompose_strided_slice(
                tf_input_shape=input_shape,
                tfbegin=op.args['begin'],
                tfend=op.args['end'],
                tf_ellipsis_mask=op.args['ellipsis_mask'],
                tf_new_axis_mask=op.args['new_axis_mask'],
                tf_shrink_axis_mask=op.args['shrink_axis_mask'],
                tf_begin_mask=op.args['begin_mask'],
                tf_end_mask=op.args['end_mask']
            )

            if opt_reshape_shape is not None:
                if not _is_compatible(opt_reshape_shape, dn_dy.shape):
                    utils.print_error(
                        "Shape mismatch in strided_slice_grad {} {}".format(opt_reshape_shape, dn_dy.shape))
                op_reshape = dog.OperationNode(get_qualified_name(tf.reshape))
                op_reshape.add_arg("tensor", dn_dy)
                op_reshape.add_arg("shape", strided_slice_result_shape)

                dn_intermediate = dog.DataNode()
                dn_intermediate.shape = strided_slice_result_shape

                op_reshape.add_result("result", dn_intermediate)

                ops2.append(op_reshape)

                dn_dy = op_reshape.result

            op_pad = dog.OperationNode(get_qualified_name(tf.pad))
            op_pad.add_arg("tensor", dn_dy)
            op_pad.add_arg("paddings",
                           [[b, s - e] for b, e, s in zip(strided_slice_begin, strided_slice_end, input_shape)])
            op_pad.add_arg("mode", "CONSTANT")
            op_pad.add_arg("constant_values", 0)
            op_pad.add_result("result", op.result)

            unlink_op(op)

            ops2.append(op_pad)
        else:
            ops2.append(op)
    return ops2


def transform_sqrt_grad(ops, converter):
    # def sqrt_grad(y, dy):
    #     return dy * 0.5 / y

    ops2 = []
    for op in ops:
        if op.name == get_qualified_name(tf_compat.gen_math_ops_sqrt_grad):
            y = op.args["y"]
            dy = op.args["dy"]
            res = op.result
            mul = dog.OperationNode(get_qualified_name(tf.multiply))
            mul.add_arg("x", dy)
            mul.add_arg("y", 0.5)
            mul_res = dog.DataNode()
            mul_res.shape = dog.get_shape_safe(res)
            mul.add_result("result", mul_res)
            div = dog.OperationNode(get_qualified_name(tf.divide))
            div.add_arg("x", mul.result)
            div.add_arg("y", y)
            div.add_result("result", res)

            unlink_op(op)

            ops2.append(mul)
            ops2.append(div)
        else:
            ops2.append(op)
    return ops2


def transform_elu_grad(ops, converter):
    # def elu_grad(gradients, outputs):
    #     return tf.where(outputs > 0, gradients, gradients * (outputs + 1))

    ops2 = []
    for op in ops:
        if op.name == get_qualified_name(tf_compat.gen_nn_ops_elu_grad):
            gradients = op.args["gradients"]
            outputs = op.args["outputs"]
            res = op.result

            greater = dog.OperationNode(get_qualified_name(tf.greater))
            greater.add_arg("x", outputs)
            greater.add_arg("y", 0.0)
            greater_res = dog.DataNode()
            greater_res.shape = dog.get_shape_safe(res)
            greater.add_result("result", greater_res)

            add = dog.OperationNode(get_qualified_name(tf.add))
            add.add_arg("x", outputs)
            add.add_arg("y", 1.0)
            add_res = dog.DataNode()
            add_res.shape = dog.get_shape_safe(res)
            add.add_result("result", add_res)

            mul = dog.OperationNode(get_qualified_name(tf.multiply))
            mul.add_arg("x", gradients)
            mul.add_arg("y", add.result)
            mul_res = dog.DataNode()
            mul_res.shape = dog.get_shape_safe(res)
            mul.add_result("result", mul_res)

            where = dog.OperationNode(get_qualified_name(tf.where))
            where.add_arg("condition", greater.result)
            where.add_arg("x", gradients)
            where.add_arg("y", mul.result)
            where.add_result("result", res)

            unlink_op(op)

            ops2.append(greater)
            ops2.append(add)
            ops2.append(mul)
            ops2.append(where)
        else:
            ops2.append(op)
    return ops2


def transform_relu_grad(ops, converter):
    # def relu_grad(gradients, features):
    #     return tf.where(features > 0, gradients, 0.0)

    ops2 = []
    for op in ops:
        if op.name == get_qualified_name(tf_compat.gen_nn_ops_relu_grad):
            gradients = op.args["gradients"]
            features = op.args["features"]
            res = op.result

            greater = dog.OperationNode(get_qualified_name(tf.greater))
            greater.add_arg("x", features)
            greater.add_arg("y", 0.0)
            greater_res = dog.DataNode()
            greater_res.shape = dog.get_shape_safe(res)
            greater.add_result("result", greater_res)

            constant = dog.OperationNode(get_qualified_name(tf.constant))
            constant.add_arg("value", 0.0)
            constant.add_arg("dtype", features.dtype)
            constant.add_arg("shape", dog.get_shape_safe(gradients))
            constant_res = dog.DataNode()
            constant_res.shape = dog.get_shape_safe(gradients)
            constant_res.dtype = features.dtype
            constant.add_result("result", constant_res)

            where = dog.OperationNode(get_qualified_name(tf.where))
            where.add_arg("condition", greater.result)
            where.add_arg("x", gradients)
            where.add_arg("y", constant.result)
            where.add_result("result", res)

            unlink_op(op)

            ops2.append(greater)
            ops2.append(constant)
            ops2.append(where)
        else:
            ops2.append(op)
    return ops2


def transform_softplus_grad(ops, converter):
    # def softplus_grad(gradients, features):
    #     return gradients * (tf.exp(features) / (tf.exp(features) + 1))

    ops2 = []
    for op in ops:
        if op.name == get_qualified_name(tf_compat.gen_nn_ops_softplus_grad):
            gradients = op.args["gradients"]
            features = op.args["features"]
            res = op.result

            exp = dog.OperationNode(get_qualified_name(tf.exp))
            exp.add_arg("x", features)
            exp_res = dog.DataNode()
            exp_res.shape = dog.get_shape_safe(res)
            exp.add_result("result", exp_res)

            add = dog.OperationNode(get_qualified_name(tf.add))
            add.add_arg("x", exp.result)
            add.add_arg("y", 1.0)
            add_res = dog.DataNode()
            add_res.shape = dog.get_shape_safe(res)
            add.add_result("result", add_res)

            div = dog.OperationNode(get_qualified_name(tf.divide))
            div.add_arg("x", exp.result)
            div.add_arg("y", add.result)
            div_res = dog.DataNode()
            div_res.shape = dog.get_shape_safe(res)
            div.add_result("result", div_res)

            mul = dog.OperationNode(get_qualified_name(tf.multiply))
            mul.add_arg("x", gradients)
            mul.add_arg("y", div.result)
            mul.add_result("result", res)

            unlink_op(op)

            ops2.append(exp)
            ops2.append(add)
            ops2.append(div)
            ops2.append(mul)
        else:
            ops2.append(op)
    return ops2


def transform_rsqrt_grad(ops, converter):
    # def rsqrt_grad(y, dy):
    #     return  (-0.5 * dy) *  y ** 3

    ops2 = []
    for op in ops:
        if op.name == get_qualified_name(tf_compat.gen_math_ops_rsqrt_grad):
            y = op.args["y"]
            dy = op.args["dy"]
            res = op.result

            mul1 = dog.OperationNode(get_qualified_name(tf.multiply))
            mul1.add_arg("x", -0.5)
            mul1.add_arg("y", dy)
            mul1_res = dog.DataNode()
            mul1_res.shape = dog.get_shape_safe(res)
            mul1.add_result("result", mul1_res)

            pow = dog.OperationNode(get_qualified_name(tf.pow))
            pow.add_arg("x", y)
            pow.add_arg("y", 3.0)
            pow_res = dog.DataNode()
            pow_res.shape = dog.get_shape_safe(res)
            pow.add_result("result", pow_res)

            mul2 = dog.OperationNode(get_qualified_name(tf.multiply))
            mul2.add_arg("x", mul1.result)
            mul2.add_arg("y", pow.result)
            mul2.add_result("result", res)

            unlink_op(op)

            ops2.append(mul1)
            ops2.append(pow)
            ops2.append(mul2)
        else:
            ops2.append(op)
    return ops2


def transform_sigmoid_grad(ops, converter):
    # def sigmoid_grad(y, dy):
    #     return dy * y * (1 - y)

    ops2 = []
    for op in ops:
        if op.name == get_qualified_name(tf_compat.gen_math_ops_sigmoid_grad):
            y = op.args["y"]
            dy = op.args["dy"]
            res = op.result

            mul1 = dog.OperationNode(get_qualified_name(tf.multiply))
            mul1.add_arg("x", dy)
            mul1.add_arg("y", y)
            mul1_res = dog.DataNode()
            mul1_res.shape = dog.get_shape_safe(res)
            mul1.add_result("result", mul1_res)

            sub = dog.OperationNode(get_qualified_name(tf.subtract))
            sub.add_arg("x", 1.0)
            sub.add_arg("y", y)
            sub_res = dog.DataNode()
            sub_res.shape = dog.get_shape_safe(res)
            sub.add_result("result", sub_res)

            mul2 = dog.OperationNode(get_qualified_name(tf.multiply))
            mul2.add_arg("x", mul1.result)
            mul2.add_arg("y", sub.result)
            mul2.add_result("result", res)

            unlink_op(op)

            ops2.append(mul1)
            ops2.append(sub)
            ops2.append(mul2)
        else:
            ops2.append(op)
    return ops2


def transform_tanh_grad(ops, converter):
    # def tanh_grad(y, dy):
    #     return dy * (1 - y ** 2)

    ops2 = []
    for op in ops:
        if op.name == get_qualified_name(tf_compat.gen_math_ops_tanh_grad):
            y = op.args["y"]
            dy = op.args["dy"]
            res = op.result

            square = dog.OperationNode(get_qualified_name(tf.square))
            square.add_arg("x", y)
            square_res = dog.DataNode()
            square_res.shape = dog.get_shape_safe(res)
            square.add_result("result", square_res)

            sub = dog.OperationNode(get_qualified_name(tf.subtract))
            sub.add_arg("x", 1.0)
            sub.add_arg("y", square.result)
            sub_res = dog.DataNode()
            sub_res.shape = dog.get_shape_safe(res)
            sub.add_result("result", sub_res)

            mul = dog.OperationNode(get_qualified_name(tf.multiply))
            mul.add_arg("x", dy)
            mul.add_arg("y", sub.result)
            mul.add_result("result", res)

            unlink_op(op)

            ops2.append(square)
            ops2.append(sub)
            ops2.append(mul)
        else:
            ops2.append(op)
    return ops2


def transform_reciprocal_grad(ops, converter):
    # def reciprocal_grad(y, dy):
    #     return -dy * y ** 2

    ops2 = []
    for op in ops:
        if op.name == get_qualified_name(tf_compat.gen_math_ops_reciprocal_grad):
            y = op.args["y"]
            dy = op.args["dy"]
            res = op.result

            neg = dog.OperationNode(get_qualified_name(tf.negative))
            neg.add_arg("x", dy)
            neg_res = dog.DataNode()
            neg_res.shape = dog.get_shape_safe(res)
            neg.add_result("result", neg_res)

            square = dog.OperationNode(get_qualified_name(tf.square))
            square.add_arg("x", y)
            square_res = dog.DataNode()
            square_res.shape = dog.get_shape_safe(res)
            square.add_result("result", square_res)

            mul = dog.OperationNode(get_qualified_name(tf.multiply))
            mul.add_arg("x", neg.result)
            mul.add_arg("y", square.result)
            mul.add_result("result", res)

            unlink_op(op)

            ops2.append(neg)
            ops2.append(square)
            ops2.append(mul)
        else:
            ops2.append(op)
    return ops2


def transform_bias_add_grad(ops, converter):
    # def bias_add_grad(out_backprop, data_format="NHWC"):
    #     return tf.reduce_sum(out_backprop, [axes except c])

    ops2 = []
    for op in ops:
        if op.name == get_qualified_name(tf_nn_ops.bias_add_grad):
            dy = op.args["out_backprop"]
            data_format = normalize_data_format(op.args["data_format"])
            res = op.result
            dy_shape = dog.get_shape_safe(dy)
            dy_rank = len(dy_shape)

            if data_format == "NHWC":
                axes = list(range(dy_rank - 1))
            else:
                axes = [0] + list(range(2, dy_rank))

            sum = dog.OperationNode(get_qualified_name(tf.reduce_sum))
            sum.add_arg("input_tensor", dy)
            sum.add_arg("axis", axes)
            sum.add_result("result", res)

            unlink_op(op)

            ops2.append(sum)
        else:
            ops2.append(op)
    return ops2


def transform_transpose_grad(ops, converter):
    # def _TransposeGrad(op, grad):
    #     p = op.inputs[1]
    #     return [array_ops.transpose(grad, array_ops.invert_permutation(p)), None]

    ops2 = []
    for op in ops:
        if op.name == get_qualified_name(tf_array_grad._TransposeGrad):
            orig_inputs = op.args["orig_inputs"]
            res = op.result[0]
            output_grad = op.args["output_grad"]

            input_ = orig_inputs[0]
            p_orig = orig_inputs[1]

            p = utils.tf_constant_value(converter.get_tftensor_by_tfdn(p_orig))
            if p is None:
                # TODO remove HACK
                if (p_orig.producer.name == get_qualified_name(tf_compat.gen_math_ops_sub)
                        and "/Range:0" in p_orig.producer.args["y"].name):
                    input_rank = len(dog.get_shape_safe(input_))
                    p = [input_rank - 1 - i for i in range(input_rank)]
                else:
                    utils.print_error("Permutation cannot be evaluated")
                    p = None

            if p is None:
                inv_p = None
            else:
                inv_p = utils.get_inverse_permutation(p)

            transpose = dog.OperationNode(get_qualified_name(tf.transpose))
            transpose.add_arg("a", output_grad)
            transpose.add_arg("perm", inv_p)
            transpose.add_result("result", res)

            unlink_op(op)

            ops2.append(transpose)
        else:
            ops2.append(op)
    return ops2


def transform_min_or_max_grad(ops, converter):
    # def _MinOrMaxGrad(input, axes, y, grad):
    #     output_shape_kept_dims = reduced_shape(input.shape, axes)
    #     y = reshape(y, output_shape_kept_dims)
    #     grad = reshape(grad, output_shape_kept_dims)
    #     equal = math_ops.equal(y, input)
    #     indicators = cast(equal, tf.float32)
    #     num_selected = reshape(reduce_sum(indicators, axes), output_shape_kept_dims)
    #     return indicators / num_selected * grad

    ops2 = []
    for op in ops:
        if op.name == get_qualified_name(tf_math_grad._MinOrMaxGrad):
            orig_inputs = op.args["orig_inputs"]
            orig_outputs = op.args["orig_outputs"]
            dn_output_grad = op.args["output_grad"]
            res = op.result[0]

            dn_orig_input = orig_inputs[0]
            axes = orig_inputs[1]
            dn_orig_output = orig_outputs[0]

            input_shape = dog.get_shape_safe(dn_orig_input)
            axes = utils.tf_constant_value(converter.get_tftensor_by_tfdn(axes)).tolist()
            output_shape_kept_dims = reduced_shape(input_shape, axes)

            # maybe not needed
            reshape0 = dog.OperationNode(get_qualified_name(tf.reshape))
            reshape0.add_arg("tensor", dn_orig_output)
            reshape0.add_arg("shape", output_shape_kept_dims)
            reshape0_res = dog.DataNode()
            reshape0_res.shape = output_shape_kept_dims
            reshape0.add_result("result", reshape0_res)

            # maybe not needed
            reshape1 = dog.OperationNode(get_qualified_name(tf.reshape))
            reshape1.add_arg("tensor", dn_output_grad)
            reshape1.add_arg("shape", output_shape_kept_dims)
            reshape1_res = dog.DataNode()
            reshape1_res.shape = output_shape_kept_dims
            reshape1.add_result("result", reshape1_res)

            equal = dog.OperationNode(get_qualified_name(tf.equal))
            equal.add_arg("x", reshape0.result)
            equal.add_arg("y", dn_orig_input)
            equal_res = dog.DataNode()
            equal_res.shape = input_shape
            equal_res.dtype = tf.bool.name
            equal.add_result("result", equal_res)

            cast = dog.OperationNode(get_qualified_name(tf.cast))
            cast.add_arg("x", equal.result)
            cast.add_arg("dtype", tf.float32.name)
            cast_res = dog.DataNode()
            cast_res.shape = input_shape
            cast_res.dtype = tf.float32.name
            cast.add_result("result", cast_res)

            reduce = dog.OperationNode(get_qualified_name(tf.reduce_sum))
            reduce.add_arg("input_tensor", cast.result)
            reduce.add_arg("axis", axes)
            reduce.add_arg("keepdims", False)
            reduce_res = dog.DataNode()
            reduce_res.shape = output_shape_kept_dims
            reduce.add_result("result", reduce_res)

            # maybe not needed
            reshape2 = dog.OperationNode(get_qualified_name(tf.reshape))
            reshape2.add_arg("tensor", reduce.result)
            reshape2.add_arg("shape", output_shape_kept_dims)
            reshape2_res = dog.DataNode()
            reshape2_res.shape = output_shape_kept_dims
            reshape2.add_result("result", reshape2_res)

            div = dog.OperationNode(get_qualified_name(tf.divide))
            div.add_arg("x", cast.result)
            div.add_arg("y", reshape2.result)
            div_res = dog.DataNode()
            div_res.shape = output_shape_kept_dims
            div.add_result("result", div_res)

            mul = dog.OperationNode(get_qualified_name(tf.multiply))
            mul.add_arg("x", div.result)
            mul.add_arg("y", reshape1.result)
            mul.add_result("result", res)

            unlink_op(op)

            ops2.append(reshape0)
            ops2.append(reshape1)
            ops2.append(equal)
            ops2.append(cast)
            ops2.append(reduce)
            ops2.append(reshape2)
            ops2.append(div)
            ops2.append(mul)

        else:
            ops2.append(op)
    return ops2


def _is_compatible(s1, s2):
    s1 = list(s1)
    s2 = list(s2)
    if (s1 == [] and s2 == [1]) or (s2 == [] and s1 == [1]):
        return True
    for a, b in zip(s1, s2):
        if a != b:
            return False
    return True


def _is_compatible_or_none(s1, s2):
    if s1 is None or s2 is None:
        return True
    s1 = list(s1)
    s2 = list(s2)
    if (s1 == [] and s2 == [1]) or (s2 == [] and s1 == [1]):
        return True
    if len(s1) != len(s2):
        return False
    for a, b in zip(s1, s2):
        if not (a is None or b is None or a == b):
            return False
    return True


def transform_tile_if_const(ops, converter):
    ops2 = []
    for op in ops:
        if op.name == get_qualified_name(tf.tile):
            if isinstance(op.args["input"], dog.DataNode) and isinstance(op.args["multiples"], dog.DataNode):
                input_ = utils.tf_constant_value(converter.get_tftensor_by_tfdn(op.args["input"]))
                multiples = utils.tf_constant_value(converter.get_tftensor_by_tfdn(op.args["multiples"]))
                if input_ is not None and multiples is not None and np.all(input_ == input_.flat[0]):
                    shape = (input_.shape * multiples).tolist()
                    const = dog.OperationNode(get_qualified_name(tf.constant))
                    const.add_arg("value", float(input_.flat[0]))
                    const.add_arg("shape", shape)
                    if not _is_compatible_or_none(op.result.shape, shape):
                        utils.print_error("Incompatible shapes {} {}".format(op.result.shape, shape))
                    op.result.shape = shape
                    const.add_result("result", op.result)
                    unlink_op(op)
                    ops2.append(const)
                    continue
        ops2.append(op)
    return ops2


def transform_zeros_ones_like(ops, converter):
    value_and_like_by_name = {
        get_qualified_name(tf.zeros_like): (0.0, True),
        get_qualified_name(tf.ones_like): (1.0, True),
        get_qualified_name(tf.zeros): (0.0, False),
        get_qualified_name(tf.ones): (1.0, False),
    }

    ops2 = []
    for op in ops:
        if op.name in value_and_like_by_name:
            value, like = value_and_like_by_name[op.name]
            if like:
                tfdn_tensor = op.args["tensor"]
                op.remove_arg("tensor")

                op.name = get_qualified_name(tf.constant)
                op.add_arg("value", value)
                op.add_arg("shape", dog.get_shape_safe(tfdn_tensor))
                ops2.append(op)
            else:
                op.name = get_qualified_name(tf.constant)
                op.add_arg("value", value)
                ops2.append(op)
        else:
            ops2.append(op)
    return ops2


def transform_range(ops, converter):
    for op in ops:
        if op.name == get_qualified_name(tf.range):
            start = op.args["start"]
            limit = op.args["limit"]
            delta = op.args["delta"]

            if isinstance(start, dog.DataNode) or isinstance(limit, dog.DataNode) or isinstance(delta, dog.DataNode):
                utils.print_error("Range is only supported with constant parameters")
                start = 0
                limit = 10
                delta = 1

            if limit is None:
                start, limit = 0, start

            op.remove_arg("start")
            op.remove_arg("limit")
            op.remove_arg("delta")

            op.name = get_qualified_name(tf.constant)

            dtype = utils.try_tf_dtype_to_np(op.result.dtype)
            if dtype is None:
                utils.print_error("Unsupported dtype {}".format(op.result.dtype))
                dtype = np.float32
            op.set_arg("value", np.arange(start=start, stop=limit, step=delta, dtype=dtype).tolist())

    return ops


def transform_lrn_grad(ops, converter):
    # def lrn_grad(input_grads, input_image, output_image, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None)

    ops2 = []
    for op in ops:
        if op.name == get_qualified_name(tf_compat.gen_nn_ops_lrn_grad):
            dn_output_grad = op.args["input_grads"]
            dn_input = op.args["input_image"]
            dn_result = op.result

            depth_radius = int(op.args["depth_radius"])
            bias = op.args["bias"]
            alpha = op.args["alpha"]
            beta = op.args["beta"]

            input_shape = dog.get_shape_safe(dn_input)
            input_shape_transposed = input_shape[:-2] + [input_shape[-1], input_shape[-2]]
            input_shape_transposed_padded = input_shape[:-2] + [input_shape[-1] + 2 * depth_radius, input_shape[-2]]

            dn0 = dn_input
            dn10 = dn_output_grad
            dn17 = beta - 1.0

            op1 = dog.OperationNode(get_qualified_name(tf.square))
            op1.add_arg("x", dn0)
            dn1 = dog.DataNode()
            dn1.shape = input_shape
            op1.add_result("result0", dn1)

            op2 = dog.OperationNode(get_qualified_name(tf.transpose))
            op2.add_arg("conjugate", False)
            op2.add_arg("perm", [0, 1, 3, 2])
            op2.add_arg("a", dn1)
            dn2 = dog.DataNode()
            dn2.shape = input_shape_transposed
            op2.add_result("result0", dn2)

            op3 = dog.OperationNode(get_qualified_name(tf.pad))
            op3.add_arg("mode", 'CONSTANT')
            op3.add_arg("paddings", [(0, 0), (0, 0), (depth_radius, depth_radius), (0, 0)])
            op3.add_arg("constant_values", 0)
            op3.add_arg("tensor", dn2)
            dn3 = dog.DataNode()
            dn3.shape = input_shape_transposed_padded
            op3.add_result("result0", dn3)

            op4 = dog.OperationNode(get_qualified_name(tf.nn.avg_pool))
            op4.add_arg("padding", 'VALID')
            op4.add_arg("value", dn3)
            op4.add_arg("ksize", [1, 1, 2 * depth_radius + 1, 1])
            op4.add_arg("strides", [1, 1, 1, 1])
            op4.add_arg("data_format", 'NHWC')
            dn4 = dog.DataNode()
            dn4.shape = input_shape_transposed
            op4.add_result("result0", dn4)

            op5 = dog.OperationNode(get_qualified_name(tf.multiply))
            op5.add_arg("y", dn4)
            op5.add_arg("x", float(2 * depth_radius + 1))
            dn5 = dog.DataNode()
            dn5.shape = input_shape_transposed
            op5.add_result("result0", dn5)

            op6 = dog.OperationNode(get_qualified_name(tf.transpose))
            op6.add_arg("conjugate", False)
            op6.add_arg("perm", [0, 1, 3, 2])
            op6.add_arg("a", dn5)
            dn6 = dog.DataNode()
            dn6.shape = input_shape
            op6.add_result("result0", dn6)

            op7 = dog.OperationNode(get_qualified_name(tf.multiply))
            op7.add_arg("y", dn6)
            op7.add_arg("x", alpha)
            dn7 = dog.DataNode()
            dn7.shape = input_shape
            op7.add_result("result0", dn7)

            op8 = dog.OperationNode(get_qualified_name(tf.add))
            op8.add_arg("y", dn7)
            op8.add_arg("x", bias)
            dn8 = dog.DataNode()
            dn8.shape = input_shape
            op8.add_result("result0", dn8)

            op9 = dog.OperationNode(get_qualified_name(tf.pow))
            op9.add_arg("y", beta)
            op9.add_arg("x", dn8)
            dn9 = dog.DataNode()
            dn9.shape = input_shape
            op9.add_result("result0", dn9)

            op11 = dog.OperationNode(get_qualified_name(tf.divide))
            op11.add_arg("y", dn9)
            op11.add_arg("x", dn10)
            dn11 = dog.DataNode()
            dn11.shape = input_shape
            op11.add_result("result0", dn11)

            op12 = dog.OperationNode(get_qualified_name(tf.negative))
            op12.add_arg("x", dn0)
            dn12 = dog.DataNode()
            dn12.shape = input_shape
            op12.add_result("result0", dn12)

            op13 = dog.OperationNode(get_qualified_name(tf.divide))
            op13.add_arg("y", dn9)
            op13.add_arg("x", dn12)
            dn13 = dog.DataNode()
            dn13.shape = input_shape
            op13.add_result("result0", dn13)

            op14 = dog.OperationNode(get_qualified_name(tf.divide))
            op14.add_arg("y", dn9)
            op14.add_arg("x", dn13)
            dn14 = dog.DataNode()
            dn14.shape = input_shape
            op14.add_result("result0", dn14)

            op15 = dog.OperationNode(get_qualified_name(tf.multiply))
            op15.add_arg("y", dn14)
            op15.add_arg("x", dn10)
            dn15 = dog.DataNode()
            dn15.shape = input_shape
            op15.add_result("result0", dn15)

            op16 = dog.OperationNode(get_qualified_name(tf.multiply))
            op16.add_arg("y", beta)
            op16.add_arg("x", dn15)
            dn16 = dog.DataNode()
            dn16.shape = input_shape
            op16.add_result("result0", dn16)

            op18 = dog.OperationNode(get_qualified_name(tf.pow))
            op18.add_arg("y", dn17)
            op18.add_arg("x", dn8)
            dn18 = dog.DataNode()
            dn18.shape = input_shape
            op18.add_result("result0", dn18)

            op19 = dog.OperationNode(get_qualified_name(tf.multiply))
            op19.add_arg("y", dn18)
            op19.add_arg("x", dn16)
            dn19 = dog.DataNode()
            dn19.shape = input_shape
            op19.add_result("result0", dn19)

            op20 = dog.OperationNode(get_qualified_name(tf.multiply))
            op20.add_arg("y", dn19)
            op20.add_arg("x", alpha)
            dn20 = dog.DataNode()
            dn20.shape = input_shape
            op20.add_result("result0", dn20)

            op21 = dog.OperationNode(get_qualified_name(tf.transpose))
            op21.add_arg("conjugate", False)
            op21.add_arg("perm", [0, 1, 3, 2])
            op21.add_arg("a", dn20)
            dn21 = dog.DataNode()
            dn21.shape = input_shape_transposed
            op21.add_result("result0", dn21)

            op22 = dog.OperationNode(get_qualified_name(tf.multiply))
            op22.add_arg("y", dn21)
            op22.add_arg("x", float(2 * depth_radius + 1))
            dn22 = dog.DataNode()
            dn22.shape = input_shape_transposed
            op22.add_result("result0", dn22)

            op23 = dog.OperationNode(get_qualified_name(tf_compat.gen_nn_ops_avg_pool_grad))
            op23.add_arg("grad", dn22)
            op23.add_arg("padding", 'VALID')
            op23.add_arg("ksize", [1, 1, int(2 * depth_radius + 1), 1])
            op23.add_arg("strides", [1, 1, 1, 1])
            op23.add_arg("orig_input_shape", input_shape_transposed_padded)
            op23.add_arg("data_format", 'NHWC')
            dn23 = dog.DataNode()
            dn23.shape = input_shape_transposed_padded
            op23.add_result("result0", dn23)

            op24 = dog.OperationNode(get_qualified_name(tf.slice))
            op24.add_arg("begin", [0, 0, depth_radius, 0])
            op24.add_arg("size", [-1, -1, input_shape[-1], -1])
            op24.add_arg("input_", dn23)
            dn24 = dog.DataNode()
            dn24.shape = input_shape_transposed
            op24.add_result("result0", dn24)

            op25 = dog.OperationNode(get_qualified_name(tf.transpose))
            op25.add_arg("conjugate", False)
            op25.add_arg("perm", [0, 1, 3, 2])
            op25.add_arg("a", dn24)
            dn25 = dog.DataNode()
            dn25.shape = input_shape
            op25.add_result("result0", dn25)

            op26 = dog.OperationNode(get_qualified_name(tf.multiply))
            op26.add_arg("y", 2.0)
            op26.add_arg("x", dn0)
            dn26 = dog.DataNode()
            dn26.shape = input_shape
            op26.add_result("result0", dn26)

            op27 = dog.OperationNode(get_qualified_name(tf.multiply))
            op27.add_arg("y", dn26)
            op27.add_arg("x", dn25)
            dn27 = dog.DataNode()
            dn27.shape = input_shape
            op27.add_result("result0", dn27)

            op28 = dog.OperationNode(get_qualified_name(tf.add_n))
            op28.add_arg("inputs", [dn11, dn27])
            op28.add_result("result0", dn_result)

            unlink_op(op)

            ops2.append(op1)
            ops2.append(op2)
            ops2.append(op3)
            ops2.append(op4)
            ops2.append(op5)
            ops2.append(op6)
            ops2.append(op7)
            ops2.append(op8)
            ops2.append(op9)
            ops2.append(op11)
            ops2.append(op12)
            ops2.append(op13)
            ops2.append(op14)
            ops2.append(op15)
            ops2.append(op16)
            ops2.append(op18)
            ops2.append(op19)
            ops2.append(op20)
            ops2.append(op21)
            ops2.append(op22)
            ops2.append(op23)
            ops2.append(op24)
            ops2.append(op25)
            ops2.append(op26)
            ops2.append(op27)
            ops2.append(op28)
        else:
            ops2.append(op)

    return ops2
