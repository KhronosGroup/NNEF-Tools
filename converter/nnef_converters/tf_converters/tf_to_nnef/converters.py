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

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers as tf_layers
from tensorflow.python.layers import utils as tf_utils
from tensorflow.python.ops import array_grad as tf_array_grad
from tensorflow.python.ops import gen_array_ops as tf_array_ops
from tensorflow.python.ops import gen_math_ops as tf_math_ops
from tensorflow.python.ops import gen_nn_ops as tf_nn_ops
from tensorflow.python.ops import math_grad as tf_math_grad

from . import transformations as trafos
from .. import tf_compat
from ...common import dog
from ...common import utils
from ...common.utils import get_qualified_name


class ConversionRule(object):
    def __init__(self, converter_fun, op_name=None, input_arg_name=None):
        self.converter_fun = converter_fun
        self.op_name = op_name
        self.input_arg_name = input_arg_name

    def is_passthrough(self):
        return self.converter_fun is convert_passthrough

    @staticmethod
    def from_old_style(old):
        if old is None:
            return None
        elif isinstance(old, tuple):
            if old[0] is convert_passthrough:
                return ConversionRule(converter_fun=old[0], input_arg_name=old[1])
            else:
                return ConversionRule(converter_fun=old[0], op_name=old[1])
        else:
            return ConversionRule(converter_fun=old)


# Converters:
def convert_skip_with_error(tfop, converter):
    utils.print_error("This op must not appear in the final tf graph: {}".format(tfop.name))


def convert_passthrough(tfop, converter):
    convert_skip_with_error(tfop, converter)


def convert_placeholder(tfop, converter):
    nnefop = dog.OperationNode("external")
    nnefop.add_arg("shape", tfop.result.shape)

    if tfop.args['name'] is not None:
        nnefdn = converter.make_nnefdn(tfop.result, utils.to_id_without_number(tfop.result.name), indexed=False)
    else:
        utils.print_warning("Cannot test activations for network if there is a placeholder without name argument")
        nnefdn = converter.make_nnefdn(tfop.result, 'input')

    nnefop.add_result('output', nnefdn)
    nnefop.result.dtype = converter.nnef_dtype(tfop.result.dtype)

    converter.add_nnefop(nnefop, tfop)


def convert_variable(tfop, converter):
    name = tfop.args['name']
    if name is None or name == '':
        utils.print_error("non-empty 'name' argument must be provided for {}".format(tfop.name))
        return

    nnefop = dog.OperationNode('variable')

    nnefop.add_arg("shape", tfop.result.shape)
    nnefop.add_arg("label", name)
    nnefop.add_result("output", converter.make_nnefdn(tfop.result, utils.get_short_name(name)))
    nnefop.result.dtype = converter.nnef_dtype(tfop.result.dtype)

    if converter.checkpoint_reader:
        if tfop.result.name is None:
            utils.print_error("Variable does not have name")
        key = tfop.result.name[:-2]
        converter.vars_names_labels_to_export.append((nnefop, key, name))
    converter.add_nnefop(nnefop, tfop)


def convert_constant(tfop, converter):
    value = tfop.args['value']

    if not isinstance(value, np.ndarray):
        value = np.array(value, dtype=converter.numpy_dtype(tfop.result.dtype))

    # TODO do inline only for unary and binary
    inlining_enabled = False
    # inlining_enabled = (all([consumer.name != get_qualified_name(tf.where) for consumer in tfop.result.consumers])
    #                     and tfop.result.name not in converter.output_name_by_tfname)

    if inlining_enabled and len(tfop.result.shape) == 0:
        converter.make_constant(tfop.result, float(value.item()))
        return

    nnefop = dog.OperationNode('constant')
    nnefop.add_arg("shape", tfop.result.shape)
    nnefop.add_arg("value", value.flatten().tolist())
    nnefop.add_result('output', converter.make_nnefdn(tfop.result, 'const'))
    nnefop.result.dtype = converter.nnef_dtype(tfop.result.dtype)

    converter.add_nnefop(nnefop, tfop)


def convert_tile_to_add(tfop, converter):
    input_ = tfop.args["input"]
    input_shape = dog.get_shape_safe(input_)
    multiples = tfop.args["multiples"]
    assert len(input_shape) == len(multiples)

    broadcasts = [m if s == 1 and m != 1 else 1 for m, s in zip(multiples, input_shape)]
    concats = [m if s != 1 and m != 1 else 1 for m, s in zip(multiples, input_shape)]

    assert sum(c != 1 for c in concats) == 0, "Tile emulation with concat is not yet implemented"

    nnefop_const = dog.OperationNode('constant')
    nnefop_const.add_arg('shape', broadcasts)
    nnefop_const.add_arg('value', [0.0])
    tfdn_const_result = dog.DataNode()
    nnefop_const.add_result('z', converter.make_nnefdn(tfdn_const_result, nnefop_const.name))

    converter.add_nnefop(nnefop_const, tfop)

    nnefop = dog.OperationNode("add")
    nnefop.add_arg('x', converter.get_nnefdn(input_))
    nnefop.add_arg('y', nnefop_const.result)
    nnefop.add_result('z', converter.make_nnefdn(tfop.result, nnefop.name))

    converter.add_nnefop(nnefop, tfop)


def convert_conv(tfop, converter):
    is_nhwc = utils.is_nhwc(tfop)

    tfdn_value = tfop.args.get('input')
    if tfdn_value is None:
        tfdn_value = tfop.args['value']

    tfdn_kernel = tfop.args.get('filter')
    if tfdn_kernel is None:
        tfdn_kernel = tfop.args['filters']

    tfdn_bias = tfop.args.get('_bias', 0.0)

    tfdn_result = tfop.result

    input_rank = len(converter.get_shape_safe(tfdn_value))
    size = converter.spatial_size_of_tf_filter(converter.get_shape_safe(tfdn_kernel))

    if 'strides' in tfop.args:
        strides = converter.spatial_size(tfop.args['strides'], is_nhwc=is_nhwc)
    elif 'stride' in tfop.args:
        strides = [tfop.args['stride']] * len(size)
    else:
        strides = [1] * len(size)

    dilations = tfop.args.get('rate', tfop.args.get('dilation_rate', tfop.args.get('dilations')))
    if dilations is not None:
        if isinstance(dilations, (list, tuple)):
            dilations = list(dilations)
            if len(dilations) == len(tfdn_value.shape):
                dilations = converter.spatial_size(dilations, is_nhwc=is_nhwc)
        else:
            dilations = [dilations] * (len(tfdn_value.shape) - 2)
    else:
        dilations = [1] * (len(tfdn_value.shape) - 2)

    padding = tfop.args['padding']

    filter_sizes = converter.dilated_size(size, dilations)
    border = converter.nnef_border(tfop.args.get('_border', 'constant'))

    if 'output_shape' in tfop.args:
        tf_output_shape = tfop.args['output_shape']
        if tf_output_shape is None:
            tf_output_shape = tfdn_result.shape
        if tf_output_shape is None:
            utils.print_warning("dynamic 'output_shape' cannot be evaluated, reverting to default")

        value_shape = converter.spatial_size(tfdn_value.shape, is_nhwc=is_nhwc)

        if tf_output_shape is not None:
            input_shape = converter.spatial_size(tf_output_shape, is_nhwc=is_nhwc)
        else:
            input_shape = [
                tf_utils.deconv_output_length(value_shape[i], filter_sizes[i], padding.lower(), strides[i])
                for i in range(len(value_shape))
            ]
        padding = converter.nnef_padding_ex(padding, input_shape, filter_sizes, strides)
    else:
        tf_output_shape = None

        padding = converter.spatial_size(converter.nnef_padding(padding, input_rank), is_nhwc=is_nhwc)

    nnefdn_input = converter.add_transpose_to_input_if_nhwc(tfop, tfdn_value)
    if "depthwise" in tfop.name:
        nnefdn_filter = converter.add_reshape_transpose_to_filter_hwcm(tfop, tfdn_kernel)
    else:
        nnefdn_filter = converter.add_transpose_to_filter_hwcn(tfop, tfdn_kernel)
    nnefdn_bias = converter.add_unsqueeze_to_arg_if_rank_1(tfop, tfdn_bias)

    nnefop = dog.OperationNode(converter.nnef_op(tfop.name))
    nnefop.add_arg('input', nnefdn_input)
    nnefop.add_arg('filter', nnefdn_filter)
    nnefop.add_arg('bias', nnefdn_bias)
    if border != 'constant':
        nnefop.add_arg('border', border)
    nnefop.add_arg('padding', padding)
    if utils.has_not_equal_1(strides):
        nnefop.add_arg('stride', strides)
    if utils.has_not_equal_1(dilations):
        nnefop.add_arg('dilation', dilations)
    if "depthwise" in tfop.name:
        nnefop.add_arg('groups', 0)

    if tf_output_shape:
        nnefop.add_arg("output_shape", utils.shape_nhwc_to_nchw(tf_output_shape) if is_nhwc else tf_output_shape)

    result_name = nnefop.name

    converter.add_nnefop_with_result_transposed_if_nhwc(tfop, nnefop, result_name="output", result_var_name=result_name)


def convert_convolution(tfop, converter):
    # This changes the original graph
    tf_strides = tfop.args['strides']
    if tf_strides is None:
        tfop.args['strides'] = [1] * len(tfop.args['input'].shape)
    else:
        tfop.args['strides'] = converter.full_shape(list(tf_strides), is_nhwc=utils.is_nhwc(tfop))
    convert_conv(tfop, converter)


def convert_conv_backprop_input(tfop, converter):
    tfdn_output_grad = tfop.args["out_backprop"]
    tfdn_filter = tfop.args["filter"]
    tf_orig_input_shape = tfop.args["input_sizes"]

    is_nhwc = utils.is_nhwc(tfop)

    spatial_strides = converter.spatial_size(tfop.args['strides'], is_nhwc=is_nhwc)

    if len(tfop.args["dilations"]) <= 3:  # from batch_to_space_nd
        spatial_dilations = tfop.args["dilations"]
    else:
        spatial_dilations = converter.spatial_size(tfop.args["dilations"], is_nhwc=is_nhwc)

    dilated_filter_sizes = converter.spatial_size_of_tf_filter(dog.get_shape_safe(tfdn_filter))
    dilated_filter_sizes = converter.dilated_size(dilated_filter_sizes, spatial_dilations)

    spatial_input_shape = converter.spatial_size(tf_orig_input_shape, is_nhwc=is_nhwc)
    nnefpadding = converter.nnef_padding_ex(
        tfop.args['padding'], spatial_input_shape, dilated_filter_sizes, spatial_strides)

    if "depthwise" in tfop.name:
        nnefdn_filter = converter.add_reshape_transpose_to_filter_hwcm(tfop, tfdn_filter)
    else:
        nnefdn_filter = converter.add_transpose_to_filter_hwcn(tfop, tfdn_filter)

    use_conv_grad_input = False
    if use_conv_grad_input:
        nnefop = dog.OperationNode(converter.nnef_op("conv_grad_input"))
        nnefop.add_arg('orig_filter', nnefdn_filter)
        nnefop.add_arg('output_grad', converter.add_transpose_to_input_if_nhwc(tfop, tfdn_output_grad))
        nnefop.add_arg('orig_input_shape',
                       utils.shape_nhwc_to_nchw(tf_orig_input_shape) if is_nhwc else tf_orig_input_shape)
        nnefop.add_arg('padding', nnefpadding)
        if utils.has_not_equal_1(spatial_strides):
            nnefop.add_arg('stride', spatial_strides)
        if utils.has_not_equal_1(spatial_dilations):
            nnefop.add_arg('dilation', spatial_dilations)
        if "depthwise" in tfop.name:
            nnefop.add_arg('groups', 0)
        converter.add_nnefop_with_result_transposed_if_nhwc(tfop, nnefop, 'input_grad', nnefop.name)
    else:
        nnefop = dog.OperationNode("deconv")
        nnefop.add_arg('input', converter.add_transpose_to_input_if_nhwc(tfop, tfdn_output_grad))
        nnefop.add_arg('filter', nnefdn_filter)
        nnefop.add_arg('bias', 0.0)
        nnefop.add_arg('border', 'constant')
        nnefop.add_arg('padding', nnefpadding)
        nnefop.add_arg('stride', spatial_strides)
        nnefop.add_arg('dilation', spatial_dilations)
        nnefop.add_arg('output_shape', (utils.shape_nhwc_to_nchw(tf_orig_input_shape)
                                        if is_nhwc else tf_orig_input_shape))
        nnefop.add_arg('groups', 0 if "depthwise" in tfop.name else 1)
        converter.add_nnefop_with_result_transposed_if_nhwc(tfop, nnefop, 'output', nnefop.name)


def convert_conv_backprop_filter(tfop, converter):
    tfdn_input = tfop.args["input"]
    tfdn_output_grad = tfop.args["out_backprop"]

    tf_filter_shape = tfop.args["filter_sizes"]

    is_nhwc = utils.is_nhwc(tfop)
    spatial_strides = converter.spatial_size(tfop.args['strides'], is_nhwc=is_nhwc)

    if len(tfop.args["dilations"]) <= 3:  # from batch_to_space_nd
        spatial_dilations = tfop.args["dilations"]
    else:
        spatial_dilations = converter.spatial_size(tfop.args["dilations"], is_nhwc=is_nhwc)

    dilated_filter_sizes = converter.spatial_size_of_tf_filter(tf_filter_shape)
    dilated_filter_sizes = converter.dilated_size(dilated_filter_sizes, spatial_dilations)

    spatial_input_shape = converter.spatial_size(dog.get_shape_safe(tfdn_input), is_nhwc=is_nhwc)

    nnefpadding = converter.nnef_padding_ex(
        tfop.args['padding'], spatial_input_shape, dilated_filter_sizes, spatial_strides)

    nnefop = dog.OperationNode(converter.nnef_op(tfop.name))
    nnefop.add_arg('orig_input', converter.add_transpose_to_input_if_nhwc(tfop, tfdn_input))
    nnefop.add_arg('output_grad', converter.add_transpose_to_input_if_nhwc(tfop, tfdn_output_grad))

    if "depthwise" in tfop.name:
        nnefop.add_arg('orig_filter_shape', utils.shape_hwcm_to_nchw(tf_filter_shape))
    else:
        nnefop.add_arg('orig_filter_shape', utils.shape_hwcn_to_nchw(tf_filter_shape))

    nnefop.add_arg('padding', nnefpadding)
    if utils.has_not_equal_1(spatial_strides):
        nnefop.add_arg('stride', spatial_strides)
    if utils.has_not_equal_1(spatial_dilations):
        nnefop.add_arg('dilation', spatial_dilations)

    if "depthwise" in tfop.name:
        nnefop.add_arg('groups', 0)

    if "depthwise" in tfop.name:
        converter.add_nnefop_with_result_transposed_reshaped_hwcm(tfop, nnefop, 'filter_grad', nnefop.name,
                                                                  tf_output_shape=tf_filter_shape)
    else:
        converter.add_nnefop_with_result_transposed_hwcn(tfop, nnefop, 'filter_grad', nnefop.name)


def convert_pool_grad(tfop, converter):
    is_nhwc = utils.is_nhwc(tfop)

    size = list(tfop.args['ksize'])
    input_rank = len(size)
    stride = list(tfop.args['strides'])

    padding = converter.nnef_padding(tfop.args['padding'], input_rank)
    border = converter.nnef_border(tfop.args.get('_border', 'constant'))

    nnefop = dog.OperationNode(converter.nnef_op(tfop.name))
    if tfop.name == get_qualified_name(tf_compat.gen_nn_ops_max_pool_grad_with_argmax):
        nnefop.add_arg('orig_input', converter.add_transpose_to_input_if_nhwc(tfop, tfop.args["input"]))
        nnefop.add_arg('orig_index', converter.add_transpose_to_input_if_nhwc(tfop, tfop.args["argmax"]))
    elif tfop.name == get_qualified_name(tf_compat.gen_nn_ops_max_pool_grad):
        nnefop.add_arg('orig_input', converter.add_transpose_to_input_if_nhwc(tfop, tfop.args["orig_input"]))
        nnefop.add_arg('orig_output', converter.add_transpose_to_input_if_nhwc(tfop, tfop.args["orig_output"]))
    elif tfop.name == get_qualified_name(tf_compat.gen_nn_ops_avg_pool_grad):
        tf_orig_input_shape = tfop.args["orig_input_shape"]
        nnefop.add_arg('orig_input_shape',
                       utils.shape_nhwc_to_nchw(tf_orig_input_shape) if is_nhwc else tf_orig_input_shape)
    else:
        assert False

    nnefop.add_arg('output_grad', converter.add_transpose_to_input_if_nhwc(tfop, tfop.args["grad"]))

    nnefop.add_arg('size', utils.shape_nhwc_to_nchw(size) if is_nhwc else size)
    if border != 'constant':
        nnefop.add_arg('border', border)
    nnefop.add_arg('padding', padding)
    if utils.has_not_equal_1(stride):
        nnefop.add_arg('stride', utils.shape_nhwc_to_nchw(stride) if is_nhwc else stride)

    converter.add_nnefop_with_result_transposed_if_nhwc(tfop, nnefop, 'input_grad', nnefop.name)


def convert_separable_conv(tfop, converter):
    is_nhwc = utils.is_nhwc(tfop)

    tf_value = tfop.args['input']
    input_rank = dog.get_rank_safe(tf_value)
    tf_kernel = tfop.args['depthwise_filter']
    tf_point_kernel = tfop.args['pointwise_filter']

    strides = converter.spatial_size(tfop.args['strides'], is_nhwc=is_nhwc)

    tf_rate = tfop.args['rate']
    if tf_rate is None:
        tf_rate = 1
    rate = converter.nnef_array(tf_rate, 2)

    padding = converter.spatial_size(converter.nnef_padding(tfop.args['padding'], input_rank), is_nhwc=is_nhwc)
    border = converter.nnef_border(tfop.args.get('_border', 'constant'))

    nnefop = dog.OperationNode('separable_conv')
    nnefop.add_arg('input', converter.add_transpose_to_input_if_nhwc(tfop, tf_value))
    nnefop.add_arg('plane_filter', converter.add_reshape_transpose_to_filter_hwcm(tfop, tf_kernel))
    nnefop.add_arg('point_filter', converter.add_transpose_to_filter_hwcn(tfop, tf_point_kernel))
    nnefop.add_arg('padding', padding)
    nnefop.add_arg('border', border)
    nnefop.add_arg('stride', strides)
    nnefop.add_arg('dilation', rate)
    converter.add_nnefop_with_result_transposed_if_nhwc(tfop, nnefop, result_name="output", result_var_name="conv")


def convert_pool(tfop, converter):
    tfdn_value = tfop.args.get('value')
    if tfdn_value is None:
        tfdn_value = tfop.args['input']
    input_rank = len(converter.get_shape_safe(tfdn_value))

    size = list(tfop.args['ksize'])
    strides = list(tfop.args['strides'])
    padding = converter.nnef_padding(tfop.args['padding'], input_rank)
    border = converter.nnef_border(tfop.args.get('_border', 'ignore'))

    nnefdn_input = converter.add_transpose_to_input_if_nhwc(tfop, tfdn_value)

    if utils.is_nhwc(tfop):
        size = utils.shape_nhwc_to_nchw(size)
        strides = utils.shape_nhwc_to_nchw(strides)
        padding = utils.shape_nhwc_to_nchw(padding)

    nnefop = dog.OperationNode(converter.nnef_op(tfop.name))
    nnefop.add_arg('input', nnefdn_input)
    nnefop.add_arg("size", size)
    nnefop.add_arg("padding", padding)
    nnefop.add_arg("border", border)
    if utils.has_not_equal_1(strides):
        nnefop.add_arg("stride", strides)

    # indices should be recalculated, but we don't do it
    # in the index consumer function they won't be recalculated either
    # so it cancels out, and it will be good
    if nnefop.name == "max_pool_with_index":
        converter.add_nnefop_with_result_transposed_if_nhwc(tfop, nnefop,
                                                            result_name=["output", "index"],
                                                            result_var_name=["pool", "index"])
    else:
        converter.add_nnefop_with_result_transposed_if_nhwc(tfop, nnefop,
                                                            result_name="output",
                                                            result_var_name="pool")


def convert_activation(tfop, converter):
    nnefop = dog.OperationNode(converter.nnef_op(tfop.name))
    nnefop.add_arg('x', converter.get_nnefdn(tfop.args['features']))
    nnefop.add_result("output", converter.make_nnefdn(tfop.result, nnefop.name))

    converter.add_nnefop(nnefop, tfop)


def convert_leaky_relu(tfop, converter):
    tfdn_input = tfop.args['features']
    tfdn_alpha = tfop.args["alpha"]

    nnefop = dog.OperationNode("prelu" if isinstance(tfdn_alpha, dog.DataNode) else "leaky_relu")
    nnefop.add_arg('x', converter.get_nnefdn(tfdn_input))
    nnefop.add_arg('alpha', converter.add_unsqueeze_to_arg_if_broadcast(tfop, tfdn_alpha, tfdn_input))
    nnefop.add_result("y", converter.make_nnefdn(tfop.result, nnefop.name))

    converter.add_nnefop(nnefop, tfop)


def convert_unary(tfop, converter):
    nnefop = dog.OperationNode(converter.nnef_op(tfop.name))
    nnefop.add_arg('x', converter.get_nnefdn(tfop.args['x']))
    nnefop.add_result('y', converter.make_nnefdn(tfop.result, nnefop.name))

    converter.add_nnefop(nnefop, tfop)


def convert_binary(tfop, converter):
    nnef_op_name = converter.nnef_op(tfop.name)

    if nnef_op_name in ["eq", "ne"]:
        assert isinstance(tfop.args["x"], float) or tfop.args["x"].dtype.startswith("float"), \
            "Eq/ne is only supported for floats in nnef."

    nnefop = dog.OperationNode(nnef_op_name)

    nnefop.add_arg('x', converter.add_unsqueeze_to_arg_if_broadcast(tfop, tfop.args['x'], tfop.args['y']))
    nnefop.add_arg('y', converter.add_unsqueeze_to_arg_if_broadcast(tfop, tfop.args['y'], tfop.args['x']))
    nnefop.add_result('z', converter.make_nnefdn(tfop.result, nnefop.name))

    converter.add_nnefop(nnefop, tfop)


def convert_squared_diff(tfop, converter):
    nnefop_diff = dog.OperationNode('sub')
    nnefop_diff.add_arg('x', converter.get_nnefdn(tfop.args['x']))
    nnefop_diff.add_arg('y', converter.get_nnefdn(tfop.args['y']))
    tfdn_sub_result = dog.DataNode()
    nnefop_diff.add_result('z', converter.make_nnefdn(tfdn_sub_result, nnefop_diff.name))

    nnefop_sqr = dog.OperationNode('sqr')
    nnefop_sqr.add_arg('x', nnefop_diff.result)
    nnefop_sqr.add_result('y', converter.make_nnefdn(tfop.result, nnefop_sqr.name))

    converter.add_nnefop(nnefop_diff, tfop)
    converter.add_nnefop(nnefop_sqr, tfop)


def convert_where(tfop, converter):
    x = tfop.args['x']
    y = tfop.args['y']

    if x is None or y is None:
        utils.print_error("arguments must not be None in tf.where() operation")
        return

    nnefop = dog.OperationNode('select')
    nnefop.add_arg('condition', converter.get_nnefdn(tfop.args['condition']))
    nnefop.add_arg('true_value', converter.get_nnefdn(x))
    nnefop.add_arg('false_value', converter.get_nnefdn(y))
    nnefop.add_result('output', converter.make_nnefdn(tfop.result, 'select'))

    converter.add_nnefop(nnefop, tfop)


def convert_reduce(tfop, converter):
    tfdn_input = tfop.args.get('input_tensor')
    if tfdn_input is None:
        tfdn_input = tfop.args['input']

    input_rank = converter.get_rank_safe(tfdn_input)

    keep_dims = tfop.args.get("keepdims")
    if keep_dims is None:
        keep_dims = tfop.args.get("keep_dims")

    axes = trafos.get_tf_reduction_axes(tfop)

    if keep_dims:
        nnefop = dog.OperationNode(converter.nnef_op(tfop.name))
        nnefop.add_arg('input', converter.get_nnefdn(tfdn_input))
        nnefop.add_arg('axes', sorted(converter.nnef_axes(axes, input_rank)))
        nnefop.add_result('output', converter.make_nnefdn(tfop.result, 'reduce'))
        converter.add_nnefop(nnefop, tfop)
        return
    else:
        nnefop_reduce = dog.OperationNode(converter.nnef_op(tfop.name))
        nnefop_reduce.add_arg('input', converter.get_nnefdn(tfdn_input))
        nnefop_reduce.add_arg('axes', sorted(converter.nnef_axes(axes, input_rank)))
        nnefop_reduce.add_result('output', converter.make_nnefdn(dog.DataNode(), 'reduce'))

        nnefop_squeeze = dog.OperationNode('squeeze')
        nnefop_squeeze.add_arg('input', nnefop_reduce.result)
        nnefop_squeeze.add_arg('axes', sorted(converter.nnef_axes(axes, input_rank)))
        nnefop_squeeze.add_result('output', converter.make_nnefdn(tfop.result, 'squeeze'))

        converter.add_nnefop(nnefop_reduce, tfop)
        converter.add_nnefop(nnefop_squeeze, tfop)
        return


def convert_lrn(tfop, converter):
    # tf <= 1.3 compat:
    if tfop.args['depth_radius'] is None:
        tfop.set_arg('depth_radius', 5)
    if tfop.args['bias'] is None:
        tfop.set_arg('bias', 1.0)
    if tfop.args['alpha'] is None:
        tfop.set_arg('alpha', 1.0)
    if tfop.args['beta'] is None:
        tfop.set_arg('beta', 0.5)

    depth_size = 2 * tfop.args['depth_radius'] + 1

    nnefop = dog.OperationNode('local_response_normalization')
    nnefop.add_arg('input', converter.add_transpose_to_input_if_nhwc(tfop, tfop.args['input'], is_nhwc=True))
    nnefop.add_arg('size', [1, int(depth_size), 1, 1])
    nnefop.add_arg('alpha', float(tfop.args['alpha'] * depth_size))
    nnefop.add_arg('beta', float(tfop.args['beta']))
    nnefop.add_arg('bias', float(tfop.args['bias']))
    converter.add_nnefop_with_result_transposed_if_nhwc(tfop, nnefop,
                                                        result_name="output",
                                                        result_var_name="norm",
                                                        is_nhwc=True)


def convert_batch_normalization(tfop, converter):
    mean_rank = converter.get_rank_safe(tfop.args['mean'])
    mean_shape = converter.get_shape_safe(tfop.args['mean'])
    is_nhwc = (not (mean_rank >= 2 and mean_shape[1] > 1)
               and not tfop.args.get("_data_format", "").upper().startswith("NC"))

    nnefdn_input = converter.add_transpose_to_input_if_nhwc(tfop, tfop.args['x'], is_nhwc=is_nhwc)
    nnefdn_mean = converter.add_unsqueeze_or_transpose_to_arg_if_needed(tfop, tfop.args['mean'], is_nhwc=is_nhwc)
    nnefdn_var = converter.add_unsqueeze_or_transpose_to_arg_if_needed(tfop, tfop.args['variance'], is_nhwc=is_nhwc)
    nnefdn_offset = converter.add_unsqueeze_or_transpose_to_arg_if_needed(tfop, tfop.args['offset'], is_nhwc=is_nhwc)
    nnefdn_scale = converter.add_unsqueeze_or_transpose_to_arg_if_needed(tfop, tfop.args['scale'], is_nhwc=is_nhwc)

    nnefop = dog.OperationNode('batch_normalization')
    nnefop.add_arg('input', nnefdn_input)
    nnefop.add_arg('mean', nnefdn_mean)
    nnefop.add_arg('variance', nnefdn_var)
    nnefop.add_arg('offset', nnefdn_offset if nnefdn_offset is not None else 0.0)
    nnefop.add_arg('scale', nnefdn_scale if nnefdn_scale is not None else 1.0)
    nnefop.add_arg('epsilon', float(tfop.args['variance_epsilon']))

    converter.add_nnefop_with_result_transposed_if_nhwc(tfop, nnefop,
                                                        result_name="output",
                                                        result_var_name="norm",
                                                        is_nhwc=is_nhwc)


def convert_l2_normalization(tfop, converter):
    tfdn_x = tfop.args['x']
    input_rank = converter.get_rank_safe(tfdn_x)

    axes = tfop.args.get('axis')
    if axes is None:
        axes = tfop.args.get('dim')

    nnefop = dog.OperationNode('l2_normalization')
    nnefop.add_arg('input', converter.get_nnefdn(tfdn_x))
    nnefop.add_arg('axes', sorted(converter.nnef_axes(axes, input_rank)))
    nnefop.add_arg('bias', float(tfop.args['epsilon']))
    nnefop.add_result('output', converter.make_nnefdn(tfop.result, 'norm'))

    converter.add_nnefop(nnefop, tfop)


def convert_matmul(tfop, converter):
    nnefop = dog.OperationNode('matmul')
    nnefop.add_arg('A', converter.get_nnefdn(tfop.args['a']))
    nnefop.add_arg('B', converter.get_nnefdn(tfop.args['b']))
    nnefop.add_arg('transposeA', (converter.nnef_bool(tfop.args['transpose_a'])
                                  or converter.nnef_bool(tfop.args.get('adjoint_a', False))))
    nnefop.add_arg('transposeB', (converter.nnef_bool(tfop.args['transpose_b'])
                                  or converter.nnef_bool(tfop.args.get('adjoint_b', False))))

    nnefop.add_result('C', converter.make_nnefdn(tfop.result, 'matmul'))

    converter.add_nnefop(nnefop, tfop)


def convert_assign(tfop, converter):
    nnefop = dog.OperationNode('update')
    nnefop.add_arg('variable', converter.get_nnefdn(tfop.args['ref']))
    nnefop.add_arg('value', converter.get_nnefdn(tfop.args['value']))
    nnefop.add_result('result', converter.make_nnefdn(tfop.result, 'assign'))

    converter.add_nnefop(nnefop, tfop)


def convert_add_n(tfop, converter):
    nnefop = dog.OperationNode('add_n')
    nnefop.add_arg('x', [converter.get_nnefdn(tfdn) for tfdn in tfop.args['inputs']])
    nnefop.add_result('y', converter.make_nnefdn(tfop.result, 'add'))

    converter.add_nnefop(nnefop, tfop)


def convert_bias_add(tfop, converter):
    # bias must be rank 1 in tensorflow
    nnefop = dog.OperationNode('add')
    nnefop.add_arg('x', converter.add_transpose_to_input_if_nhwc(tfop, tfop.args['value']))
    nnefop.add_arg('y', converter.add_unsqueeze_to_arg_if_rank_1(tfop, tfop.args['bias']))
    converter.add_nnefop_with_result_transposed_if_nhwc(tfop, nnefop, result_name='z', result_var_name='add')


def convert_concat(tfop, converter):
    values = tfop.args['values']
    input_rank = len(dog.get_shape_safe(values[0]))

    nnefop = dog.OperationNode('concat')
    nnefop.add_arg('values', [converter.get_nnefdn(value) for value in values])
    nnefop.add_arg('axis', converter.nnef_axis(tfop.args['axis'], input_rank))
    nnefop.add_result('value', converter.make_nnefdn(tfop.result, 'concat'))

    converter.add_nnefop(nnefop, tfop)


def convert_split(tfop, converter):
    value = tfop.args['value']
    num_or_sizes = tfop.args['num_or_size_splits']

    nnefop = dog.OperationNode('split')
    nnefop.add_arg('value', converter.get_nnefdn(value))
    nnefop.add_arg('axis', converter.nnef_axis(tfop.args['axis'], converter.get_rank_safe(value)))
    nnefop.add_arg('ratios', num_or_sizes if isinstance(num_or_sizes, list) else [1] * num_or_sizes)
    nnefop.add_result('values', [converter.make_nnefdn(tfdn, "split") for tfdn in tfop.result])

    converter.add_nnefop(nnefop, tfop)


def convert_softmax(tfop, converter):
    logits = tfop.args['logits']

    axis = tfop.args.get('dim')
    if axis is None:
        axis = tfop.args.get('axis')
    if axis is None:
        axis = -1

    nnefop = dog.OperationNode('softmax')
    nnefop.add_arg('x', converter.get_nnefdn(logits))
    nnefop.add_arg('axes', [converter.nnef_axis(axis, converter.get_rank_safe(logits))])
    nnefop.add_result('y', converter.make_nnefdn(tfop.result, 'softmax'))

    converter.add_nnefop(nnefop, tfop)


def convert_moments(tfop, converter):
    value = tfop.args['x']
    nnef_axes = sorted(converter.nnef_axes(tfop.args['axes'], converter.get_rank_safe(value)))

    keep_dims = tfop.args.get("keepdims")
    if keep_dims is None:
        keep_dims = tfop.args.get("keep_dims")

    if keep_dims:
        nnefop = dog.OperationNode('moments')
        nnefop.add_arg('input', converter.get_nnefdn(value))
        nnefop.add_arg('axes', nnef_axes)
        nnefop.add_result('mean', converter.make_nnefdn(tfop.results['result0'], 'mean'))
        nnefop.add_result('variance', converter.make_nnefdn(tfop.results['result1'], 'variance'))
        converter.add_nnefop(nnefop, tfop)
    else:
        nnefop = dog.OperationNode('moments')
        nnefop.add_arg('input', converter.get_nnefdn(value))
        nnefop.add_arg('axes', nnef_axes)
        nnefop.add_result('mean', converter.make_nnefdn(dog.DataNode(), 'mean'))
        nnefop.add_result('variance', converter.make_nnefdn(dog.DataNode(), 'variance'))

        nnefop_squeeze_mean = dog.OperationNode('squeeze')
        nnefop_squeeze_mean.add_arg('input', nnefop.results['mean'])
        nnefop_squeeze_mean.add_arg('axes', nnef_axes)
        nnefop_squeeze_mean.add_result('output', converter.make_nnefdn(tfop.results['result0'], 'squeeze'))

        nnefop_squeeze_variance = dog.OperationNode('squeeze')
        nnefop_squeeze_variance.add_arg('input', nnefop.results['variance'])
        nnefop_squeeze_variance.add_arg('axes', nnef_axes)
        nnefop_squeeze_variance.add_result('output', converter.make_nnefdn(tfop.results['result1'], 'squeeze'))

        converter.add_nnefop(nnefop, tfop)
        converter.add_nnefop(nnefop_squeeze_mean, tfop)
        converter.add_nnefop(nnefop_squeeze_variance, tfop)


def convert_reshape(tfop, converter):
    tfdn_tensor = tfop.args['tensor']
    nnefop = dog.OperationNode('reshape')
    nnefop.add_arg('input', converter.get_nnefdn(tfdn_tensor))

    nnefshape = list(tfop.args['shape'])
    nnefop.add_arg('shape', nnefshape)
    nnefop.add_result('output', converter.make_nnefdn(tfop.result, 'reshape'))

    converter.add_nnefop(nnefop, tfop)


def convert_flatten(tfop, converter):
    nnefop = dog.OperationNode('reshape')
    nnefop.add_arg('input', converter.get_nnefdn(tfop.args['inputs']))
    nnefop.add_arg('shape', [0, -1])
    nnefop.add_result('output', converter.make_nnefdn(tfop.result, 'reshape'))

    converter.add_nnefop(nnefop, tfop)


def convert_expand_dims(tfop, converter):
    tfdn_value = tfop.args['input']

    axis = tfop.args.get('axis')
    if axis is None:
        axis = tfop.args['dim']

    nnefop = dog.OperationNode('unsqueeze')
    nnefop.add_arg('input', converter.get_nnefdn(tfdn_value))
    nnefop.add_arg('axes', [converter.nnef_axis(axis, converter.get_rank_safe(tfdn_value) + 1)])
    nnefop.add_result('output', converter.make_nnefdn(tfop.result, 'reshape'))
    converter.add_nnefop(nnefop, tfop)


def convert_squeeze(tfop, converter):
    value = tfop.args['input']
    nnef_value = converter.get_nnefdn(value)

    axes = tfop.args['axis']

    if axes is not None:
        axes = list(sorted(axes))
    else:
        axes = [i for i in range(len(value.shape)) if value.shape[i] == 1]

    nnefop = dog.OperationNode('squeeze')
    nnefop.add_arg('input', nnef_value)
    nnefop.add_arg('axes', converter.nnef_axes(axes, converter.get_rank_safe(value)))
    nnefop.add_result('output', converter.make_nnefdn(tfop.result, 'squeeze'))

    converter.add_nnefop(nnefop, tfop)


def convert_transpose(tfop, converter):
    value = tfop.args['a']
    rank = len(value.shape)
    perm = tfop.args['perm']
    if perm is None:
        perm = list(reversed(range(rank)))

    nnefop = dog.OperationNode('transpose')
    nnefop.add_arg('input', converter.get_nnefdn(value))
    nnefop.add_arg('axes', converter.nnef_axes(perm, rank))
    nnefop.add_result('output', converter.make_nnefdn(tfop.result, 'trans'))

    converter.add_nnefop(nnefop, tfop)


def convert_resize_images(tfop, converter):
    tfdn_images = tfop.args['images']
    nnefdn_images = converter.get_nnefdn(tfdn_images)
    size = tfop.args['size']
    method = tfop.args['method']
    aligned = tfop.args['align_corners']

    if isinstance(size, dog.DataNode):
        utils.print_error('cannot handle dynamic target size in tf.image.resize()')
        return

    input_size = tfdn_images.shape[1:-1]
    size = [s.value if isinstance(s, tf.Dimension) else int(s) for s in size]

    if size[0] == input_size[0] and size[1] == input_size[1]:
        nnefop = dog.OperationNode("_nnef_passthrough")
        nnefop.add_arg('input', nnefdn_images)
        nnefop.add_result('output', converter.make_nnefdn(tfop.result, '_nnef_passthrough'))
        converter.add_nnefop(nnefop, tfop)
        return

    nnefdn_images = converter.add_transpose_to_input_if_nhwc(tfop, tfdn_images, is_nhwc=True)

    if (size[0] > input_size[0] and size[1] < input_size[1]) or (size[0] < input_size[0] and size[1] > input_size[1]):
        utils.print_error("resize must be up or down-sampling")
        return

    if size[0] > input_size[0]:
        if size[0] % input_size[0] or size[1] % input_size[1]:
            utils.print_error('only integer factor resize allowed')
            return

        factor = [size[0] // input_size[0], size[1] // input_size[1]]

        result_var_name = 'upsample'

        if method == tf.image.ResizeMethod.BILINEAR:
            nnefop = dog.OperationNode('multilinear_upsample')
            nnefop.add_arg('input', nnefdn_images)
            nnefop.add_arg('factor', factor)
            nnefop.add_arg('method', 'aligned' if aligned else 'asymmetric')
            nnefop.add_arg('border', 'replicate')
        elif method == tf.image.ResizeMethod.NEAREST_NEIGHBOR:
            nnefop = dog.OperationNode('nearest_upsample')
            nnefop.add_arg('input', nnefdn_images)
            nnefop.add_arg('factor', factor)
        else:
            utils.print_error("unsupported upsample method '{}'".format(method))
            return
    else:
        if input_size[0] % size[0] or input_size[1] % size[1]:
            utils.print_error('only integer factor resize allowed')
            return

        factor = [input_size[0] // size[0], input_size[1] // size[1]]

        result_var_name = 'downsample'

        if method == tf.image.ResizeMethod.AREA:
            nnefop = dog.OperationNode('area_downsample')
            nnefop.add_arg('input', nnefdn_images)
            nnefop.add_arg('factor', factor)
        elif method == tf.image.ResizeMethod.NEAREST_NEIGHBOR:
            nnefop = dog.OperationNode('nearest_downsample')
            nnefop.add_arg('input', nnefdn_images)
            nnefop.add_arg('factor', factor)
        else:
            utils.print_error("unsupported downsample method '{}'".format(method))
            return

    converter.add_nnefop_with_result_transposed_if_nhwc(
        tfop, nnefop, result_name="output", result_var_name=result_var_name, is_nhwc=True)


def convert_resize_bilinear(tfop, converter):
    tfop.args['method'] = tf.image.ResizeMethod.BILINEAR
    return convert_resize_images(tfop, converter)


def convert_resize_bicubic(tfop, converter):
    tfop.args['method'] = tf.image.ResizeMethod.BICUBIC
    return convert_resize_images(tfop, converter)


def convert_resize_nearest(tfop, converter):
    tfop.args['method'] = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    return convert_resize_images(tfop, converter)


def convert_resize_area(tfop, converter):
    tfop.args['method'] = tf.image.ResizeMethod.AREA
    return convert_resize_images(tfop, converter)


def convert_clip_by_value(tfop, converter):
    nnefop = dog.OperationNode('clamp')
    nnefop.add_arg('x', converter.get_nnefdn(tfop.args['t']))
    nnefop.add_arg('a', converter.add_unsqueeze_to_arg_if_broadcast(tfop, tfop.args['clip_value_min'], tfop.args['t']))
    nnefop.add_arg('b', converter.add_unsqueeze_to_arg_if_broadcast(tfop, tfop.args['clip_value_max'], tfop.args['t']))
    nnefop.add_result('y', converter.make_nnefdn(tfop.result, 'clamp'))

    converter.add_nnefop(nnefop, tfop)


def convert_relu6(tfop, converter):
    nnefop = dog.OperationNode('clamp')
    nnefop.add_arg('x', converter.get_nnefdn(tfop.args['features']))
    nnefop.add_arg('a', 0.0)
    nnefop.add_arg('b', 6.0)
    nnefop.add_result('y', converter.make_nnefdn(tfop.result, 'clamp'))

    converter.add_nnefop(nnefop, tfop)


def convert_slice(tfop, converter):
    tfdn = tfop.args['input_']
    nnef_axes = list(range(len(tfdn.shape)))
    nnef_begin = tfop.args['begin']
    nnef_size = tfop.args['size']
    nnef_shape = tfdn.shape

    nnef_axes, nnef_begin, nnef_end = utils.zip_inverse(3, [
        (axis, begin, shape if size == -1 else begin + size)
        for axis, begin, size, shape in zip(nnef_axes, nnef_begin, nnef_size, nnef_shape)
        if not (begin == 0 and size == -1) and not (begin == 0 and size != -1 and begin + size == shape)
    ])

    nnefop = dog.OperationNode('slice')
    nnefop.add_arg('input', converter.get_nnefdn(tfdn))
    nnefop.add_arg('axes', converter.nnef_axes(nnef_axes, len(nnef_shape)))
    nnefop.add_arg('begin', nnef_begin)
    nnefop.add_arg('end', nnef_end)
    nnefop.add_result('output', converter.make_nnefdn(tfop.result, 'slice'))

    converter.add_nnefop(nnefop, tfop)


def convert_strided_slice(tfop, converter):
    tfdn = tfop.args['input_']
    begin = tfop.args['begin']
    end = tfop.args['end']

    shape = dog.get_shape_safe(tfdn)
    # tfformat = trafos.get_format_safe(tfdn)

    axes = list(range(len(shape)))

    axes, begin, end = utils.zip_inverse(3, [
        (a, b, e)
        for a, b, e, s in zip(axes, begin, end, shape)
        if not (b == 0 and e == s)
    ])

    nnefop = dog.OperationNode('slice')
    nnefop.add_arg('input', converter.get_nnefdn(tfdn))
    nnefop.add_arg('axes', converter.nnef_axes(axes, len(shape)))
    nnefop.add_arg('begin', begin)
    nnefop.add_arg('end', end)
    nnefop.add_result('output', converter.make_nnefdn(tfop.result, 'slice'))

    converter.add_nnefop(nnefop, tfop)


def convert_stack(tfop, converter):
    values = tfop.args['values']

    nnefop = dog.OperationNode('stack')
    nnefop.add_arg('values', [converter.get_nnefdn(value) for value in values])
    nnefop.add_arg('axis', converter.nnef_axis(tfop.args['axis'], converter.get_rank_safe(values[0]) + 1))
    nnefop.add_result('value', converter.make_nnefdn(tfop.result, 'stack'))

    converter.add_nnefop(nnefop, tfop)


def convert_unstack(tfop, converter):
    value = tfop.args['value']
    rank = len(value.shape)

    nnefop = dog.OperationNode('unstack')
    nnefop.add_arg('value', converter.get_nnefdn(value))
    nnefop.add_arg('axis', converter.nnef_axis(tfop.args['axis'], rank))

    num = tfop.args.get('num')
    if num is not None and num != len(tfop.result):
        utils.print_error("Num for unstack is {} but it has {} results".format(num, len(tfop.results)))

    nnefop.add_result('values', [converter.make_nnefdn(tfdn, "unstack") for tfdn in tfop.result])

    converter.add_nnefop(nnefop, tfop)


def convert_pad(tfop, converter):
    tfdn_tensor = tfop.args["tensor"]
    paddings = tfop.args["paddings"]
    rank = len(dog.get_shape_safe(tfdn_tensor))

    if isinstance(paddings, np.ndarray):
        paddings = paddings.tolist()
    elif isinstance(paddings, tuple):
        paddings = np.array(paddings).tolist()

    if tfop.args["constant_values"] != 0:
        utils.print_error(
            "Only tf.pad constant_values=0 is supported, got: {}, resetting to 0.".format(tfop.args["constant_values"]))

    nnefop = dog.OperationNode("box")  # TODO replace with pad when implemented
    nnefop.add_arg("input", converter.get_nnefdn(tfdn_tensor))
    nnefop.add_arg("size", [1] * rank)
    nnefop.add_arg("border", converter.nnef_border(tfop.args["mode"]))
    nnefop.add_arg("padding", converter.nnef_padding(paddings, rank))
    nnefop.add_result("output", converter.make_nnefdn(tfop.result, 'pad'))

    converter.add_nnefop(nnefop, tfop)


DefaultConverters = {
    tf.Variable: (convert_variable, 'variable'),
    tf.get_variable: (convert_variable, 'variable'),
    tf.placeholder: (convert_placeholder, 'external'),
    tf.constant: (convert_constant, 'constant'),
    tf.concat: (convert_concat, 'concat'),
    tf.split: (convert_split, 'split'),
    tf.reshape: (convert_reshape, 'reshape'),
    tf.squeeze: (convert_squeeze, 'squeeze'),
    tf.expand_dims: (convert_expand_dims, 'reshape'),
    tf.transpose: (convert_transpose, 'transpose'),
    tf.pad: (convert_pad, 'box'),
    tf.add: (convert_binary, 'add'),
    tf.subtract: (convert_binary, 'sub'),
    tf.multiply: (convert_binary, 'mul'),
    tf.divide: (convert_binary, 'div'),
    tf.pow: (convert_binary, 'pow'),
    tf.squared_difference: (convert_squared_diff, 'sqr'),
    tf.logical_and: (convert_binary, 'and'),
    tf.logical_or: (convert_binary, 'or'),
    tf.negative: (convert_unary, 'neg'),
    tf.logical_not: (convert_unary, 'not'),
    tf.abs: (convert_unary, 'abs'),
    tf.sign: (convert_unary, 'sign'),
    tf.exp: (convert_unary, 'exp'),
    tf.log: (convert_unary, 'log'),
    tf.sqrt: (convert_unary, 'sqrt'),
    tf.rsqrt: (convert_unary, 'rsqrt'),
    tf.square: (convert_unary, 'sqr'),
    tf.floor: (convert_unary, 'floor'),
    tf.ceil: (convert_unary, 'ceil'),
    tf.round: (convert_unary, 'round'),
    tf.where: (convert_where, 'select'),
    tf.greater: (convert_binary, 'gt'),
    tf.greater_equal: (convert_binary, 'ge'),
    tf.less: (convert_binary, 'lt'),
    tf.less_equal: (convert_binary, 'le'),
    tf.equal: (convert_binary, 'eq'),
    tf.not_equal: (convert_binary, 'ne'),
    tf.minimum: (convert_binary, 'min'),
    tf.maximum: (convert_binary, 'max'),
    tf.assign: (convert_assign, 'update'),
    tf_math_ops.add: (convert_binary, 'add'),
    tf_math_ops.div: (convert_binary, 'div'),
    tf_math_ops._pow: (convert_binary, 'pow'),
    tf_math_ops.logical_and: (convert_binary, 'and'),
    tf_math_ops.logical_or: (convert_binary, 'or'),
    tf_math_ops.reciprocal: (convert_unary, 'rcp'),
    tf_math_ops.logical_not: (convert_unary, 'not'),
    tf_math_ops._abs: (convert_unary, 'abs'),
    tf_math_ops.sign: (convert_unary, 'sign'),
    tf_math_ops.exp: (convert_unary, 'exp'),
    tf_math_ops.log: (convert_unary, 'log'),
    tf_math_ops.square: (convert_unary, 'sqr'),
    tf_math_ops.floor: (convert_unary, 'floor'),
    tf_math_ops.ceil: (convert_unary, 'ceil'),
    tf_math_ops.round: (convert_unary, 'round'),
    tf_math_ops.greater: (convert_binary, 'gt'),
    tf_math_ops.greater_equal: (convert_binary, 'ge'),
    tf_math_ops.less: (convert_binary, 'lt'),
    tf_math_ops.less_equal: (convert_binary, 'le'),
    tf_math_ops.equal: (convert_binary, 'eq'),
    tf_math_ops.not_equal: (convert_binary, 'ne'),
    tf_math_ops.sqrt: (convert_unary, 'sqrt'),
    tf_math_ops.rsqrt: (convert_unary, 'rsqrt'),
    tf.sigmoid: (convert_unary, 'sigmoid'),
    tf.tanh: (convert_unary, 'tanh'),
    tf.reduce_sum: (convert_reduce, 'sum_reduce'),
    tf.reduce_mean: (convert_reduce, 'mean_reduce'),
    tf.reduce_max: (convert_reduce, 'max_reduce'),
    tf.reduce_min: (convert_reduce, 'min_reduce'),
    tf.argmax: (convert_reduce, 'argmax_reduce'),
    tf.argmin: (convert_reduce, 'argmin_reduce'),
    tf.matmul: (convert_matmul, 'matmul'),
    tf.add_n: (convert_add_n, 'add_n'),
    tf.nn.sigmoid: (convert_unary, 'sigmoid'),
    tf.nn.tanh: (convert_unary, 'tanh'),
    tf.nn.elu: (convert_activation, 'elu'),
    tf.nn.relu: (convert_activation, 'relu'),
    tf.nn.relu6: (convert_relu6, 'relu6'),
    tf.nn.softsign: (convert_activation, 'softsign'),
    tf.nn.softplus: (convert_activation, 'softplus'),
    tf.nn.conv1d: (convert_conv, 'conv'),
    tf.nn.conv2d: (convert_conv, 'conv'),
    tf.nn.conv3d: (convert_conv, 'conv'),
    tf.nn.convolution: (convert_convolution, 'conv'),
    tf.nn.conv2d_transpose: (convert_conv, 'deconv'),
    tf.nn.conv3d_transpose: (convert_conv, 'deconv'),
    tf.nn.atrous_conv2d: (convert_conv, 'conv'),
    tf.nn.atrous_conv2d_transpose: (convert_conv, 'deconv'),
    tf.nn.depthwise_conv2d: (convert_conv, 'conv'),
    tf.nn.depthwise_conv2d_native: (convert_conv, 'conv'),
    tf.nn.separable_conv2d: (convert_separable_conv, 'conv'),
    tf.nn.max_pool: (convert_pool, 'max_pool'),
    tf.nn.max_pool_with_argmax: (convert_pool, 'max_pool_with_index'),
    tf.nn.avg_pool: (convert_pool, 'avg_pool'),
    tf.nn.bias_add: (convert_bias_add, 'add'),
    tf.nn.lrn: (convert_lrn, 'local_response_normalization'),
    tf.nn.local_response_normalization: (convert_lrn, 'local_response_normalization'),
    tf.nn.batch_normalization: (convert_batch_normalization, 'batch_normalization'),
    tf.nn.l2_normalize: (convert_l2_normalization, 'l2_normalization'),
    tf.nn.softmax: (convert_softmax, 'softmax'),
    tf.nn.moments: (convert_moments, 'moments'),
    tf.image.resize_images: convert_resize_images,
    tf.image.resize_bilinear: convert_resize_bilinear,
    tf.image.resize_nearest_neighbor: convert_resize_nearest,
    tf.image.resize_bicubic: convert_resize_bicubic,
    tf.image.resize_area: convert_resize_area,
    tf_layers.softmax: (convert_softmax, 'softmax'),
    tf_layers.flatten: (convert_flatten, 'reshape'),
    tf.clip_by_value: (convert_clip_by_value, 'clamp'),
    tf.slice: (convert_slice, 'slice'),
    tf.strided_slice: (convert_strided_slice, 'slice'),
    tf.stack: (convert_stack, 'stack'),
    tf.unstack: (convert_unstack, 'unstack'),

    tf.identity: (convert_passthrough, "input"),
    tf.stop_gradient: (convert_passthrough, "input"),
    tf.cast: (convert_passthrough, "x"),
    tf.nn.dropout: (convert_passthrough, "x"),
    tf.space_to_batch: convert_skip_with_error,
    tf.space_to_batch_nd: convert_skip_with_error,
    tf.batch_to_space: convert_skip_with_error,
    tf.batch_to_space_nd: convert_skip_with_error,
    tf.fill: convert_skip_with_error,
    tf.tile: convert_tile_to_add,
    tf.invert_permutation: convert_skip_with_error,
    tf.floor_div: convert_skip_with_error,
    tf.dynamic_stitch: convert_skip_with_error,
    tf.range: convert_skip_with_error,
    tf.mod: convert_skip_with_error,
    tf.zeros: convert_skip_with_error,
    tf.ones: convert_skip_with_error,
    tf.zeros_like: convert_skip_with_error,
    tf.ones_like: convert_skip_with_error,

    tf_array_ops.strided_slice_grad: convert_skip_with_error,
    tf_math_ops._range: convert_skip_with_error,
    tf_array_ops.rank: convert_skip_with_error,

    tf.nn.conv2d_backprop_input: (convert_conv_backprop_input, 'conv_grad_input'),
    tf_nn_ops.conv3d_backprop_input_v2: (convert_conv_backprop_input, 'conv_grad_input'),
    tf.nn.depthwise_conv2d_native_backprop_input: (convert_conv_backprop_input, 'conv_grad_input'),

    tf.nn.conv2d_backprop_filter: (convert_conv_backprop_filter, 'conv_grad_filter'),
    tf.nn.conv3d_backprop_filter_v2: (convert_conv_backprop_filter, 'conv_grad_filter'),
    tf.nn.depthwise_conv2d_native_backprop_filter: (convert_conv_backprop_filter, 'conv_grad_filter'),

    tf_nn_ops.bias_add_grad: convert_skip_with_error,
    tf_nn_ops.fused_batch_norm_grad: convert_skip_with_error,
    tf_compat.gen_nn_ops_fused_batch_norm_grad_v2: convert_skip_with_error,
    tf_nn_ops._fused_batch_norm: convert_skip_with_error,

    tf_array_grad._TransposeGrad: convert_skip_with_error,  # see tf2dog._fix_special_grad_invocations
    tf_math_grad._MinOrMaxGrad: convert_skip_with_error,

    tf.shape: convert_skip_with_error,
    tf.shape_n: convert_skip_with_error,
}

DefaultConverters.update({
    tf_compat.sinh: (convert_unary, 'sinh'),
    tf_compat.cosh: (convert_unary, 'cosh'),

    tf_compat.gen_math_ops_sub: (convert_binary, 'sub'),
    tf_compat.gen_math_ops_mul: (convert_binary, 'mul'),
    tf_compat.gen_math_ops_real_div: (convert_binary, 'div'),
    tf_compat.gen_math_ops_neg: (convert_unary, 'neg'),
    tf_compat.gen_math_ops_mat_mul: (convert_matmul, 'matmul'),
    tf_compat.gen_nn_ops_softmax: (convert_softmax, 'softmax'),
    tf_compat.gen_array_ops_concat_offset: convert_skip_with_error,
    tf_compat.gen_nn_ops_fused_batch_norm_v2: convert_skip_with_error,
    tf_compat.nn_leaky_relu: (convert_leaky_relu, 'leaky_relu'),

    tf_compat.gen_nn_ops_max_pool_grad: (convert_pool_grad, 'max_pool_grad'),
    tf_compat.gen_nn_ops_max_pool_grad_with_argmax: (convert_pool_grad, 'max_pool_grad_with_index'),
    tf_compat.gen_nn_ops_avg_pool_grad: (convert_pool_grad, 'avg_pool_grad'),
    tf_compat.gen_math_ops_sqrt_grad: convert_skip_with_error,
    tf_compat.gen_nn_ops_elu_grad: convert_skip_with_error,
    tf_compat.gen_nn_ops_relu_grad: convert_skip_with_error,
    tf_compat.gen_nn_ops_softplus_grad: convert_skip_with_error,
    tf_compat.gen_math_ops_rsqrt_grad: convert_skip_with_error,
    tf_compat.gen_math_ops_sigmoid_grad: convert_skip_with_error,
    tf_compat.gen_math_ops_tanh_grad: convert_skip_with_error,
    tf_compat.gen_math_ops_reciprocal_grad: convert_skip_with_error,
    tf_compat.gen_nn_ops_lrn_grad: convert_skip_with_error,

    tf_compat.gen_array_ops_broadcast_gradient_args: convert_skip_with_error
})
