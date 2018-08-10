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

import math

from ...common import dog
from ...common import nnef_shape_optimizer
from ...common import utils


# HELPERS

def calculate_padding_elem(upscaled_size, downscaled_size, filter_size, stride, dilation):
    dilated_filter_size = (filter_size - 1) * dilation + 1
    t = (downscaled_size - 1) * stride + dilated_filter_size - upscaled_size
    return math.floor(t / 2), math.ceil(t / 2)


def calculate_padding(upscaled_shape, downscaled_shape, filter_shape, strides, dilations):
    return [
        calculate_padding_elem(i, o, f, s, d)
        for i, o, f, s, d in zip(upscaled_shape, downscaled_shape, filter_shape, strides, dilations)
    ]


def get_paddings(nnefop):
    """
    Returns:
        nnefpadding_separate: int[]|None
        tfpadding_in_op: "SAME"|"VALID"
    """
    are_args_spatial = "conv" in nnefop.name

    nnefpadding = nnefop.args["padding"]
    nnefborder = nnefop.args["border"].lower()

    if nnefpadding == [] and (nnefborder == "constant" or nnefborder == "ignore"):
        return None, "SAME"
    elif not utils.has_greater_than_0(nnefpadding):
        return None, "VALID"
    else:
        if nnefpadding == []:
            if are_args_spatial:
                nnefpadding = calculate_padding(
                    upscaled_shape=dog.get_shape_safe(nnefop.args["input"])[2:],
                    downscaled_shape=dog.get_shape_safe(nnefop.result)[2:],
                    filter_shape=dog.get_shape_safe(nnefop.args["filter"])[2:],
                    strides=nnefop.args["stride"],
                    dilations=nnefop.args["dilation"]
                )
            else:
                nnefpadding = calculate_padding(
                    upscaled_shape=dog.get_shape_safe(nnefop.args["input"]),
                    downscaled_shape=dog.get_shape_safe(nnefop.result),
                    filter_shape=nnefop.args["size"],
                    strides=nnefop.args["stride"],
                    dilations=nnefop.args["dilation"]
                )

        if utils.has_greater_than_0(nnefpadding):
            if are_args_spatial:
                return [(0, 0), (0, 0)] + nnefpadding, "VALID"
            else:
                return nnefpadding, "VALID"
        else:
            return None, "VALID"


def is_varlike(nnefdn):
    return nnefdn.producer.name in ["external", "variable", "constant"]


# TRANSFORMS

def transform_extract_padding(nnefdog):
    nnefopnames = {"conv", "planewise_conv", "separable_conv", "argmax_pool", "max_pool_with_index", "max_pool",
                   "avg_pool", "rms_pool"}

    new_nnefops = []

    for nnefop in nnefdog.ops:
        if nnefop.name in nnefopnames:
            nnefpadding_separate, tfpadding_in_op = get_paddings(nnefop)
            nnefop.add_arg("_tf_padding", tfpadding_in_op)
            if nnefpadding_separate is not None:
                nnefdn_input = nnefop.args["input"]
                nnefdn_result = nnefop.results["index" if nnefop.name == "argmax_pool" else "output"]

                nnefdn_new_result = dog.DataNode(nnefdn_result.name + "_pad__")
                nnefdn_new_result.shape = [s + p + q for s, (p, q) in
                                           zip(dog.get_shape_safe(nnefdn_input), nnefpadding_separate)]
                nnefdog.dn_by_name[nnefdn_new_result.name] = nnefdn_new_result

                nnefop_pad = dog.OperationNode("_nnef_pad")
                nnefop_pad.add_arg("input", nnefdn_input)
                nnefop_pad.add_arg("padding", nnefpadding_separate)
                nnefop_pad.add_arg("border", nnefop.args["border"])
                nnefop_pad.add_result("result", nnefdn_new_result)
                nnefop_pad.extra["original_op"] = nnefop

                nnefop.set_arg("input", nnefop_pad.result)
                new_nnefops.append(nnefop_pad)

        new_nnefops.append(nnefop)

    nnefdog.ops = new_nnefops


def transform_extract_padding_for_grads(nnefdog):
    # These are handled separately with calculate_tfpadding_for_deconv
    # "conv_grad_input", "conv_grad_filter"

    nnefopnames = {"max_pool_grad_with_index", "avg_pool_grad", "max_pool_grad"}

    new_nnefops = []

    for nnefop in nnefdog.ops:
        if nnefop.name in nnefopnames:
            nnefpadding_separate, tfpadding_in_op = get_paddings(nnefop)
            nnefop.add_arg("_tf_padding", tfpadding_in_op)
            if nnefpadding_separate is not None and "filter" not in nnefop.name:
                utils.print_error("Separate padding not supported in grads, {}".format(nnefpadding_separate))
        new_nnefops.append(nnefop)

    nnefdog.ops = new_nnefops


def transform_extract_bias_add(nnefdog):
    nnefopnames = {"conv", "planewise_conv", "separable_conv", "deconv", "planewise_deconv", "separable_deconv"}

    new_nnefops = []

    for nnefop in nnefdog.ops:
        new_nnefops.append(nnefop)
        if nnefop.name in nnefopnames and nnefop.args["bias"] != 0.0:
            nnefdn_bias = nnefop.args["bias"]
            nnefop.remove_arg("bias")

            nnefdn_old_result = nnefop.result
            nnefdn_new_result = dog.DataNode(nnefdn_old_result.name + "_conv__")
            nnefdn_new_result.shape = list(nnefdn_old_result.shape)
            nnefdog.dn_by_name[nnefdn_new_result.name] = nnefdn_new_result
            nnefop.set_result("output", nnefdn_new_result)

            nnefop_bias_add = dog.OperationNode("_nnef_bias_add")
            nnefop_bias_add.add_arg("x", nnefdn_new_result)
            nnefop_bias_add.add_arg("y", nnefdn_bias)
            nnefop_bias_add.add_result("z", nnefdn_old_result)
            nnefop_bias_add.extra["original_op"] = nnefop

            new_nnefops.append(nnefop_bias_add)

    nnefdog.ops = new_nnefops


def transform_extract_bias_add_for_grads(nnefdog):
    nnefopnames = {"conv_grad_input", "conv_grad_filter"}

    new_nnefops = []

    for nnefop in nnefdog.ops:
        new_nnefops.append(nnefop)
        if nnefop.name in nnefopnames and nnefop.args["bias"] != 0.0:
            utils.print_error("Bias not supported in grads")
            #
            # nnefdn_bias = nnefop.args["bias"]
            # nnefop.remove_arg("bias")
            #
            # nnefdn_old_result = nnefop.result
            # nnefdn_new_result = dog.DataNode(nnefdn_old_result.name + "_conv__")
            # nnefdn_new_result.shape = list(nnefdn_old_result.shape)
            # nnefdog.dn_by_name[nnefdn_new_result.name] = nnefdn_new_result
            # nnefop.set_result("output", nnefdn_new_result)
            #
            # nnefop_bias_add = dog.OperationNode("_nnef_bias_add")
            # nnefop_bias_add.add_arg("x", nnefdn_new_result)
            # nnefop_bias_add.add_arg("y", nnefdn_bias)
            # nnefop_bias_add.add_result("z", nnefdn_old_result)
            # nnefop_bias_add.extra["original_op"] = nnefop
            #
            # new_nnefops.append(nnefop_bias_add)

    nnefdog.ops = new_nnefops


def transform_transpose_to_target_lang(nnefdog):
    global ctr
    ctr = 0

    # Tensorflow does broadcasting from right, nnef does it from left

    # btw we handle rms_pool as non_atomic
    conv_ops = ['conv', 'deconv']
    pooling_ops = ['max_pool_with_index', 'max_pool', 'avg_pool', 'rms_pool']
    up_down_sampling_ops = ['nearest_downsample', 'area_downsample', 'nearest_upsample', 'multilinear_upsample']

    binary_ops = ['add', 'sub', 'mul', 'div', 'pow', 'lt', 'gt', 'le', 'ge', 'eq', 'ne', 'and', 'or',
                  'min', 'max']

    new_ops = []
    for op in nnefdog.ops:
        if op.name in conv_ops:
            input_channels = op.args['input'].shape[1]

            add_transpose_to_arg(op, 'input', new_ops)
            if op.args["groups"] == 1:
                add_transpose_to_filter(op, 'filter', new_ops)
            else:
                if op.args["groups"] not in [0, op.result.shape[1]]:
                    utils.print_error("Unsupported groups value for {}: {}"
                                      .format(op.result.name, op.args["groups"]))
                add_transpose_reshape_to_planewise_filter(op, 'filter', new_ops, input_channels)

            if op.args.get("output_shape"):
                nchw_arg_to_nhwc(op, 'output_shape')

            new_ops.append(op)
            add_transpose_to_result(op, 'output', new_ops, nnefdog)
        elif op.name == "conv_grad_input":
            input_channels = op.args['orig_input_shape'][1]

            if op.args["groups"] == 1:
                add_transpose_to_filter(op, 'orig_filter', new_ops)
            else:
                if op.args["groups"] not in [0, op.result.shape[1]]:
                    utils.print_error("Unsupported groups value for {}: {}"
                                      .format(op.result.name, op.args["groups"]))

                add_transpose_reshape_to_planewise_filter(op, 'orig_filter', new_ops, input_channels)

            add_transpose_to_arg(op, 'output_grad', new_ops)

            nchw_arg_to_nhwc(op, 'orig_input_shape')

            new_ops.append(op)
            add_transpose_to_result(op, 'input_grad', new_ops, nnefdog)
        elif op.name == "conv_grad_filter":
            input_channels = op.args['orig_input'].shape[1]

            add_transpose_to_arg(op, 'orig_input', new_ops)
            add_transpose_to_arg(op, 'output_grad', new_ops)

            if op.args["groups"] == 1:
                nchw_arg_to_hwcn(op, 'orig_filter_shape')
            else:
                if op.args["groups"] not in [0, op.result.shape[1]]:
                    utils.print_error("Unsupported groups value for {}: {}"
                                      .format(op.result.name, op.args["groups"]))
                nchw_arg_to_hwcm(op, 'orig_filter_shape', input_channels)

            new_ops.append(op)

            if op.args["groups"] == 1:
                add_transpose_to_result_filter(op, 'input_grad', new_ops, nnefdog)
            else:
                if op.args["groups"] not in [0, op.result.shape[1]]:
                    utils.print_error("Unsupported groups value for {}: {}"
                                      .format(op.result.name, op.args["groups"]))
                add_reshape_transpose_to_result_planewise_filter(op, 'input_grad', new_ops, nnefdog, input_channels)
        elif op.name in pooling_ops:
            add_transpose_to_arg(op, 'input', new_ops)
            nchw_arg_to_nhwc(op, 'size')
            nchw_arg_to_nhwc(op, 'padding')
            nchw_arg_to_nhwc(op, 'stride')
            nchw_arg_to_nhwc(op, 'dilation')
            new_ops.append(op)
            add_transpose_to_result(op, 'output', new_ops, nnefdog)
            if 'index' in op.results.keys():
                add_transpose_to_result(op, 'index', new_ops, nnefdog)
        elif op.name in ["max_pool_grad", "max_pool_grad_with_index"]:
            add_transpose_to_arg(op, 'orig_input', new_ops)
            if 'index' in op.name:
                add_transpose_to_arg(op, 'orig_index', new_ops)
            else:
                add_transpose_to_arg(op, 'orig_output', new_ops)
            add_transpose_to_arg(op, 'output_grad', new_ops)
            nchw_arg_to_nhwc(op, 'size')
            nchw_arg_to_nhwc(op, 'padding')
            nchw_arg_to_nhwc(op, 'stride')
            nchw_arg_to_nhwc(op, 'dilation')
            new_ops.append(op)
            add_transpose_to_result(op, 'input_grad', new_ops, nnefdog)
        elif op.name == "avg_pool_grad":
            add_transpose_to_arg(op, 'output_grad', new_ops)
            nchw_arg_to_nhwc(op, 'orig_input_shape')
            nchw_arg_to_nhwc(op, 'size')
            nchw_arg_to_nhwc(op, 'padding')
            nchw_arg_to_nhwc(op, 'stride')
            nchw_arg_to_nhwc(op, 'dilation')
            new_ops.append(op)
            add_transpose_to_result(op, 'input_grad', new_ops, nnefdog)
        elif op.name == "_nnef_bias_add":
            add_transpose_to_arg(op, 'x', new_ops)
            add_squeeze_to_arg(op, 'y', new_ops)
            new_ops.append(op)
            add_transpose_to_result(op, 'z', new_ops, nnefdog)
        elif op.name == "local_response_normalization":
            if len(op.args["size"]) > 2 and op.args["size"][1] > 1:
                add_transpose_to_arg(op, 'input', new_ops)
                nchw_arg_to_nhwc(op, 'size')
                new_ops.append(op)
                add_transpose_to_result(op, 'output', new_ops, nnefdog)
            else:
                new_ops.append(op)
        elif op.name == "batch_normalization":
            add_transpose_to_arg(op, 'input', new_ops)
            add_squeeze_or_transpose_to_arg(op, 'mean', new_ops)
            add_squeeze_or_transpose_to_arg(op, 'variance', new_ops)
            add_squeeze_or_transpose_to_arg(op, 'offset', new_ops)
            add_squeeze_or_transpose_to_arg(op, 'scale', new_ops)
            new_ops.append(op)
            add_transpose_to_result(op, 'output', new_ops, nnefdog)
        elif op.name in up_down_sampling_ops:
            add_transpose_to_arg(op, 'input', new_ops)
            new_ops.append(op)
            add_transpose_to_result(op, 'output', new_ops, nnefdog)
        elif op.name in binary_ops:
            add_unsqueeze_to_arg_if_broadcast(op, 'x', 'y', new_ops)
            add_unsqueeze_to_arg_if_broadcast(op, 'y', 'x', new_ops)
            new_ops.append(op)
        elif op.name == "clamp":
            add_unsqueeze_to_arg_if_broadcast(op, 'a', 'x', new_ops)
            add_unsqueeze_to_arg_if_broadcast(op, 'b', 'x', new_ops)
            new_ops.append(op)
        else:
            new_ops.append(op)

    nnefdog.ops = new_ops


ctr = 0  # TODO


def add_transpose_to_arg(op, arg_name, new_ops):
    global ctr
    dn = op.args[arg_name]

    input_rank = dog.get_rank_safe(dn)

    op_transpose = dog.OperationNode("transpose")
    op_transpose.add_arg("input", dn)
    op_transpose.add_arg("axes", utils.transpose_axes_nchw_to_nhwc(input_rank))
    op_transpose.add_result("output", dog.DataNode("_nnef_nhwc_" + str(ctr)))
    ctr += 1
    op_transpose.result.shape = utils.shape_nchw_to_nhwc(dog.get_shape_safe(dn))
    op_transpose.extra[nnef_shape_optimizer.EXTRA_GENERATED_TRANSPOSE] = True

    op.set_arg(arg_name, op_transpose.result)

    new_ops.append(op_transpose)


def add_squeeze_to_arg(op, arg_name, new_ops):
    global ctr
    dn = op.args[arg_name]

    op_squeeze = dog.OperationNode("squeeze")
    op_squeeze.add_arg("input", dn)
    op_squeeze.add_arg("axes", [0])
    op_squeeze.add_result("output", dog.DataNode("_nnef_squeeze_" + str(ctr)))
    op_squeeze.extra[nnef_shape_optimizer.EXTRA_GENERATED_SQUEEZE] = True

    ctr += 1
    op_squeeze.result.shape = utils.apply_squeeze_shape(dog.get_shape_safe(dn), [0])

    op.set_arg(arg_name, op_squeeze.result)

    new_ops.append(op_squeeze)


def add_transpose_to_filter(op, arg_name, new_ops):
    global ctr
    dn = op.args[arg_name]

    input_rank = dog.get_rank_safe(dn)

    op_transpose = dog.OperationNode("transpose")
    op_transpose.add_arg("input", dn)
    op_transpose.add_arg("axes", utils.transpose_axes_nchw_to_hwcn(input_rank))
    op_transpose.add_result("output", dog.DataNode("_nnef_hwcn_" + str(ctr)))
    ctr += 1
    op_transpose.result.shape = utils.shape_nchw_to_hwcn(dog.get_shape_safe(dn))
    op_transpose.extra[nnef_shape_optimizer.EXTRA_GENERATED_TRANSPOSE] = True

    op.set_arg(arg_name, op_transpose.result)

    new_ops.append(op_transpose)


def add_transpose_reshape_to_planewise_filter(op, arg_name, new_ops, input_channels):
    global ctr
    dn = op.args[arg_name]

    input_rank = dog.get_rank_safe(dn)

    op_transpose = dog.OperationNode("transpose")
    op_transpose.add_arg("input", dn)
    op_transpose.add_arg("axes", utils.transpose_axes_nchw_to_hwcn(input_rank))
    op_transpose.add_result("output", dog.DataNode("_nnef_hwcn_" + str(ctr)))
    ctr += 1
    op_transpose.result.shape = utils.shape_nchw_to_hwcn(dog.get_shape_safe(dn))
    op_transpose.extra[nnef_shape_optimizer.EXTRA_GENERATED_TRANSPOSE] = True
    new_ops.append(op_transpose)

    reshape_shape = list(op_transpose.result.shape)
    reshape_shape = reshape_shape[:-2] + [input_channels, reshape_shape[-1] // input_channels]

    op_reshape = dog.OperationNode("reshape")
    op_reshape.add_arg("input", op_transpose.result)
    op_reshape.add_arg("shape", reshape_shape)
    op_reshape.add_result("output", dog.DataNode("_nnef_reshape_" + str(ctr)))
    ctr += 1
    op_reshape.result.shape = reshape_shape
    op_reshape.extra[nnef_shape_optimizer.EXTRA_GENERATED_RESHAPE] = True
    new_ops.append(op_reshape)

    op.set_arg(arg_name, op_reshape.result)


def swap_names(dn1, dn2):
    n = dn1.name
    dn1.name = dn2.name
    dn2.name = n


def add_transpose_to_result(op, result_name, new_ops, dog_graph):
    global ctr
    dn = op.results[result_name]
    orig_dn_shape = dn.shape

    input_rank = dog.get_rank_safe(dn)
    perm = utils.transpose_axes_nhwc_to_nchw(input_rank)
    perm_inverse = utils.get_inverse_permutation(perm)
    dn.extra[nnef_shape_optimizer.EXTRA_APPLIED_TRANSFORMATIONS] = [("transpose", perm_inverse)]

    op_transpose = dog.OperationNode("transpose")
    op_transpose.add_arg("input", dn)
    op_transpose.add_arg("axes", perm)
    op_transpose.add_result("output", dog.DataNode("_nnef_nchw_" + str(ctr)))
    op_transpose.result.shape = orig_dn_shape
    op_transpose.extra[nnef_shape_optimizer.EXTRA_GENERATED_TRANSPOSE] = True

    dn.shape = utils.shape_nchw_to_nhwc(orig_dn_shape)
    ctr += 1

    new_ops.append(op_transpose)

    replace_in_consumers(dn, op_transpose.result, dog_graph)


def add_transpose_to_result_filter(op, result_name, new_ops, dog_graph):
    global ctr
    dn = op.results[result_name]
    orig_dn_shape = dn.shape

    input_rank = dog.get_rank_safe(dn)
    perm = utils.transpose_axes_hwcn_to_nchw(input_rank)
    perm_inverse = utils.get_inverse_permutation(perm)
    dn.extra[nnef_shape_optimizer.EXTRA_APPLIED_TRANSFORMATIONS] = [("transpose", perm_inverse)]

    op_transpose = dog.OperationNode("transpose")
    op_transpose.add_arg("input", dn)
    op_transpose.add_arg("axes", perm)
    op_transpose.add_result("output", dog.DataNode("_nnef_nchw" + str(ctr)))
    op_transpose.result.shape = orig_dn_shape
    op_transpose.extra[nnef_shape_optimizer.EXTRA_GENERATED_TRANSPOSE] = True

    dn.shape = utils.shape_nchw_to_hwcn(orig_dn_shape)
    ctr += 1

    new_ops.append(op_transpose)

    replace_in_consumers(dn, op_transpose.result, dog_graph)


def add_reshape_transpose_to_result_planewise_filter(op, result_name, new_ops, dog_graph, input_channels):
    global ctr
    dn = op.results[result_name]
    orig_dn_shape = dn.shape
    shape_hwcm = utils.shape_nchw_to_hwcm(orig_dn_shape, input_channels)
    shape_hw1x = shape_hwcm[:-2] + [1, shape_hwcm[-2] * shape_hwcm[-1]]

    input_rank = dog.get_rank_safe(dn)
    perm = utils.transpose_axes_hwcn_to_nchw(input_rank)
    perm_inverse = utils.get_inverse_permutation(perm)
    dn.extra[nnef_shape_optimizer.EXTRA_APPLIED_TRANSFORMATIONS] = [
        ("transpose", perm_inverse),
        ("reshape", shape_hwcm)
    ]

    op_reshape = dog.OperationNode("reshape")
    op_reshape.add_arg("input", dn)
    op_reshape.add_arg("shape", shape_hw1x)
    op_reshape.add_result("output", dog.DataNode("_nnef_reshape_" + str(ctr)))
    ctr += 1
    op_reshape.result.shape = shape_hw1x
    op_reshape.extra[nnef_shape_optimizer.EXTRA_GENERATED_RESHAPE] = True
    new_ops.append(op_reshape)

    op_transpose = dog.OperationNode("transpose")
    op_transpose.add_arg("input", op_reshape.result)
    op_transpose.add_arg("axes", perm)
    op_transpose.add_result("output", dog.DataNode("_nnef_nchw_" + str(ctr)))
    op_transpose.result.shape = orig_dn_shape
    op_transpose.extra[nnef_shape_optimizer.EXTRA_GENERATED_TRANSPOSE] = True

    dn.shape = shape_hwcm
    ctr += 1

    new_ops.append(op_transpose)

    replace_in_consumers(dn, op_transpose.result, dog_graph)


def add_squeeze_or_transpose_to_arg(op, arg_name, new_ops):
    if dog.get_rank_safe(op.args[arg_name]) == 2 and dog.get_shape_safe(op.args[arg_name])[0] == 1:
        add_squeeze_to_arg(op, arg_name, new_ops)
    else:
        add_transpose_to_arg(op, arg_name, new_ops)


def add_unsqueeze_to_arg_if_broadcast(op, arg_name, other_arg_name, new_ops):
    global ctr
    dn = op.args[arg_name]
    dn_other = op.args[other_arg_name]

    if is_broadcast(dn, dn_other):
        dn_rank = dog.get_rank_safe(dn)
        rank_diff = dog.get_rank_safe(dn_other) - dog.get_rank_safe(dn)
        axes = list(range(dn_rank, dn_rank + rank_diff))
        op_unsqueeze = dog.OperationNode("unsqueeze")
        op_unsqueeze.add_arg("input", dn)
        op_unsqueeze.add_arg("axes", axes)
        op_unsqueeze.add_result("output", dog.DataNode("_nnef_unsqueeze_" + str(ctr)))
        ctr += 1
        op_unsqueeze.extra[nnef_shape_optimizer.EXTRA_GENERATED_UNSQUEEZE] = True
        op_unsqueeze.result.shape = utils.apply_unsqueeze_shape(dog.get_shape_safe(dn), axes)

        op.set_arg(arg_name, op_unsqueeze.result)

        new_ops.append(op_unsqueeze)


def is_ancestor_of2(dn, ancestor):  # TODO nicer
    if dn.producer is ancestor:
        return True
    for dn2 in dn.producer.get_arg_nodes():
        if dn2.producer is ancestor:
            return True
    return False


def replace_in_consumers(dn_old, dn_new, dog_graph):
    def replace(x):
        if x == dn_old:
            return dn_new
        return x

    for consumer in dn_old.consumers:
        if not is_ancestor_of2(dn_new, consumer):
            dn_new.consumers.append(consumer)
            consumer.args = utils.recursive_transform(consumer.args, replace)
            consumer.results = utils.recursive_transform(consumer.results, replace)


def nchw_arg_to_nhwc(op, arg_name):
    arg = op.args[arg_name]
    if arg:
        arg = utils.shape_nchw_to_nhwc(arg)
    op.set_arg(arg_name, arg)


def nchw_arg_to_hwcn(op, arg_name):
    arg = op.args[arg_name]
    if arg:
        arg = utils.shape_nchw_to_hwcn(arg)
    op.set_arg(arg_name, arg)


def nchw_arg_to_hwcm(op, arg_name, input_channels):
    arg = op.args[arg_name]
    if arg:
        arg = utils.shape_nchw_to_hwcm(arg, input_channels)
    op.set_arg(arg_name, arg)


# TODO rename
def is_broadcast(nnefdn_broadcasted, nnefdn_other):
    return (isinstance(nnefdn_broadcasted, dog.DataNode)
            and isinstance(nnefdn_other, dog.DataNode)
            and nnefdn_broadcasted.shape is not None
            and nnefdn_broadcasted.shape != []
            and nnefdn_broadcasted.shape != [1]
            and nnefdn_other.shape is not None
            and len(nnefdn_other.shape) > len(nnefdn_broadcasted.shape)
            and utils.can_broadcast_from_left(nnefdn_other.shape, nnefdn_broadcasted.shape))
