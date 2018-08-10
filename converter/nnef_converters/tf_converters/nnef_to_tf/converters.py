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

from collections import OrderedDict

import nnef
import numpy as np

from .. import tf_compat
from ...common import dog
from ...common import utils


class Converter(object):
    def __init__(self, nnefdog):
        # TODO maybe we would need tfdn_by_nnefdn_name
        self.tfdn_by_name = OrderedDict()  # doesn't have to be ordered
        self.nnefop_by_tfop = OrderedDict()
        self.nnefdn_by_tfdn = OrderedDict()
        self.nnefdog = nnefdog

    def make_tfdn(self, nnefdn=None, name=None, discriminator=None):
        assert (nnefdn is None) != (name is None)
        if name is None:
            name = nnefdn.name
        if discriminator is not None:
            counter = 1
            name_candidate = name + "_{}{}".format(discriminator, counter)
            while name_candidate in self.tfdn_by_name:  # TODO optimize
                counter += 1
                name_candidate = name + "_{}{}".format(discriminator, counter)
            name = name_candidate
        if name in self.tfdn_by_name:
            utils.print_error("Duplicate dn definition: {}".format(name))
            return self.tfdn_by_name[name]
        tfdn = dog.DataNode(name)
        self.tfdn_by_name[name] = tfdn
        self.nnefdn_by_tfdn[tfdn] = nnefdn
        return tfdn

    def get_tfdn_safe(self, nnefdn=None, name=None):
        if isinstance(nnefdn, (bool, int, float)):
            return nnefdn
        assert (nnefdn is None) != (name is None)
        orig_name = name
        if name is None:
            name = nnefdn.name
        if name not in self.tfdn_by_name:
            utils.print_error("Dn not defined: {}".format(name))
            return self.make_tfdn(nnefdn=nnefdn, name=orig_name)
        return self.tfdn_by_name[name]

    def add_tfop(self, tfop, nnefop=None):
        assert tfop not in self.nnefop_by_tfop
        self.nnefop_by_tfop[tfop] = nnefop

    def nnef_shapeof_to_tf(self, nnef_shapeof):
        name_of_nnefdn = str(nnef_shapeof)
        nnefdn = self.nnefdog.dn_by_name[name_of_nnefdn]
        return nnef.ShapeOf(self.get_tfdn_safe(nnefdn).name)

    @staticmethod
    def get_shape_safe(tfdn_or_other):
        return dog.get_shape_safe(tfdn_or_other)

    @staticmethod
    def get_rank_safe(tfdn_or_other):
        return len(dog.get_shape_safe(tfdn_or_other))

    @staticmethod
    def is_binary_nnefop_name(nnefop_name):
        return DefaultConverters[nnefop_name] == generic_convert_binary


# HELPERS:

# TODO revise these methods, is negative pad_total possible?

def calculate_same_padding_element_ex(input_size, filter_size, stride, dilation):
    dilated_filter_size = (filter_size - 1) * dilation + 1
    output_size = int(np.ceil(float(input_size) / float(stride)))
    pad_total = (output_size - 1) * stride + dilated_filter_size - input_size
    if pad_total >= 0:
        pad_front = pad_total // 2
        pad_back = pad_total - pad_front
        return pad_front, pad_back
    else:
        return 0, pad_total


def calculate_same_padding_ex(input_size, filter_size, stride, dilation):
    return [calculate_same_padding_element_ex(i, f, s, d) for i, f, s, d in
            zip(input_size, filter_size, stride, dilation)]


def calculate_valid_padding_element_ex(input_size, filter_size, stride, dilation):
    dilated_filter_size = (filter_size - 1) * dilation + 1
    output_size = int(np.ceil(float(input_size - dilated_filter_size + 1) / float(stride)))
    pad_total = (output_size - 1) * stride + dilated_filter_size - input_size
    return 0, pad_total


def calculate_valid_padding_ex(input_size, filter_size, stride, dilation):
    return [calculate_valid_padding_element_ex(i, f, s, d) for i, f, s, d in
            zip(input_size, filter_size, stride, dilation)]


# TODO unify this with trafos.get_paddings

def calculate_tfpadding_for_deconv(nnefop):
    nnefborder = nnefop.args["border"].lower()
    nnefpadding = nnefop.args["padding"]

    if "input" in nnefop.args:
        input_shape = dog.get_shape_safe(nnefop.args["input"])
    elif "orig_input" in nnefop.args:
        input_shape = dog.get_shape_safe(nnefop.args["orig_input"])
    else:
        input_shape = nnefop.args["orig_input_shape"]

    if "filter" in nnefop.args:
        filter_shape = dog.get_shape_safe(nnefop.args["filter"])
    elif "orig_filter" in nnefop.args:
        filter_shape = dog.get_shape_safe(nnefop.args["orig_filter"])
    else:
        filter_shape = nnefop.args["orig_filter_shape"]
    # TODO Check this
    same_padding = calculate_same_padding_ex(input_size=input_shape[1:-1],  # tf order
                                             filter_size=filter_shape[:-2],
                                             stride=nnefop.args["stride"],
                                             dilation=nnefop.args["dilation"])
    valid_padding = calculate_valid_padding_ex(input_size=input_shape[1:-1],
                                               filter_size=filter_shape[:-2],
                                               stride=nnefop.args["stride"],
                                               dilation=nnefop.args["dilation"])
    if nnefborder == "constant" and (nnefpadding == [] or nnefpadding == same_padding):
        return "SAME"
    elif nnefborder == "constant" and (not utils.has_greater_than_0(nnefpadding) or nnefpadding == valid_padding):
        return "VALID"
    else:
        utils.print_error("Special paddings or borders are not supported for deconv.")
        return "SAME"


def calculate_nnef_reshape_shape(old_shape, new_shape):
    new_shape = list(new_shape)
    minus_one_idx = None
    prod = 1
    for i in range(len(new_shape)):
        if new_shape[i] == 0:
            if i >= len(old_shape):
                utils.print_error("New shape has 0 at index {}, but old shape is only {} long, setting dimension to 1"
                                  .format(i, len(old_shape)))
                new_shape[i] = 1
            else:
                new_shape[i] = old_shape[i]
        if new_shape[i] == -1:
            if minus_one_idx is None:
                minus_one_idx = i
            else:
                utils.print_error("Shape had more than one -1's, setting it to 1")
                new_shape[i] = 1
        if new_shape[i] != -1:
            prod *= new_shape[i]

    if minus_one_idx is not None:
        if int(np.prod(old_shape)) % prod != 0:
            utils.print_error("-1 is used in shape but the product of other dimensions does not divide the old size")
        new_shape[minus_one_idx] = int(np.prod(old_shape)) // prod

    return new_shape


def nnefarray_to_tf(nnefarray, nnefshape):
    if not isinstance(nnefarray, np.ndarray):
        nnefarray = np.array(nnefarray)

    if nnefarray.size == 1:
        return nnefarray

    return np.reshape(nnefarray, nnefshape)


def nnefstride_to_tf_nhwc(stride, in_spatial, out_spatial, empty_value=None):
    if not stride:
        return empty_value

    if in_spatial:
        stride = [1, 1] + stride

    nhwc_stride = utils.shape_nchw_to_nhwc(stride)

    if out_spatial:
        nhwc_stride = nhwc_stride[1:-1]

    return nhwc_stride


def spatial_nhwc(stride_like, in_spatial, out_spatial, empty_value, unit_elem=1):
    if not stride_like:
        return empty_value

    if bool(in_spatial) == bool(out_spatial):
        return list(stride_like)

    if in_spatial:
        stride_like = [unit_elem] + stride_like + [unit_elem]

    if out_spatial:
        stride_like = stride_like[1:-1]

    return stride_like


def nnefborder_to_tf(border):
    to_tf = {
        'ignore': None,
        'constant': 'CONSTANT',
        'replicate': None,
        'reflect': 'REFLECT',
        'reflect-even': 'SYMMETRIC'
    }

    if to_tf[border] is None:
        utils.print_error("Border mode {} is not supported in TensorFlow.".format(border))
        return 'CONSTANT'
    else:
        return to_tf[border]


# CONVERTERS:

def convert_external(nnefop, converter):
    tfop = dog.OperationNode("tf.placeholder")
    tfop.add_arg("dtype", utils.nnef_dtype_to_tf(nnefop.result.dtype))
    tfop.add_arg("shape", nnefop.args["shape"])
    tfop.add_arg("name", nnefop.result.name)
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_variable(nnefop, converter):
    tfop = dog.OperationNode("tf.get_variable")
    tfop.add_arg("name", nnefop.args["label"])
    tfop.add_arg("shape", nnefop.args["shape"])
    tfop.add_arg("dtype", utils.nnef_dtype_to_tf(nnefop.result.dtype))
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_constant(nnefop, converter):
    array = nnefarray_to_tf(nnefop.args["value"], nnefop.args["shape"])

    shape = nnefop.args["shape"]
    if list(array.shape) != list(shape) and list(array.shape) != [1]:
        utils.print_error("Internal error, shape mismatch: {} {} {} {}"
                          .format(nnefop.result.name, list(array.shape), list(shape), nnefop.result.extra))

    tfop = dog.OperationNode("tf.constant")
    tfop.add_arg("value", array)
    tfop.add_arg("dtype", utils.nnef_dtype_to_tf(nnefop.result.dtype))
    tfop.add_arg("shape", shape)
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_update(nnefop, converter):
    tfop = dog.OperationNode("tf.assign")
    tfop.add_arg("ref", converter.get_tfdn_safe(nnefop.args["variable"]))
    tfop.add_arg("value", converter.get_tfdn_safe(nnefop.args["value"]))
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_reshape(nnefop, converter):
    if nnefop.args["shape"] == [0, -1]:
        tfop = dog.OperationNode("tf.contrib.layers.flatten")
        tfop.add_arg("inputs", converter.get_tfdn_safe(nnefop.args["input"]))
        tfop.add_result("result", converter.make_tfdn(nnefop.result))
        converter.add_tfop(tfop, nnefop)
    else:
        nnefdn_input = nnefop.args["input"]
        tfop = dog.OperationNode("tf.reshape")
        tfop.add_arg("tensor", converter.get_tfdn_safe(nnefdn_input))

        if isinstance(nnefop.args["shape"], nnef.ShapeOf):
            tfop.add_arg('shape', converter.nnef_shapeof_to_tf(nnefop.args["shape"]))
        else:
            nnef_shape = calculate_nnef_reshape_shape(dog.get_shape_safe(nnefdn_input),
                                                      nnefop.args["shape"])  # TODO needed?
            tfop.add_arg("shape", nnef_shape)

        tfop.add_result("result", converter.make_tfdn(nnefop.result))
        converter.add_tfop(tfop, nnefop)


def convert_squeeze(nnefop, converter):
    nnef_axes = nnefop.args["axes"]
    nnefdn_input = nnefop.args["input"]

    tfop = dog.OperationNode("tf.squeeze")
    tfop.add_arg("input", converter.get_tfdn_safe(nnefdn_input))
    tfop.add_arg("axis", nnef_axes)
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def partial_convert_unsqueeze_to_reshape(nnefop, converter):
    nnef_axes = nnefop.args["axes"]
    nnefdn_input = nnefop.args["input"]
    nnef_input_shape = dog.get_shape_safe(nnefdn_input)

    shape = utils.apply_unsqueeze_shape(list(nnef_input_shape), nnef_axes)

    tfop = dog.OperationNode("tf.reshape")
    tfop.add_arg("tensor", converter.get_tfdn_safe(nnefdn_input))
    tfop.add_arg("shape", shape)
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def partial_convert_unsqueeze_to_expand_dims(nnefop, converter):
    assert len(nnefop.args["axes"]) == 1

    axis = nnefop.args["axes"][0]
    nnefdn_input = nnefop.args["input"]

    tfop = dog.OperationNode("tf.expand_dims")
    tfop.add_arg("input", converter.get_tfdn_safe(nnefdn_input))
    tfop.add_arg("axis", axis)
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_unsqueeze(nnefop, converter):
    if len(nnefop.args["axes"]) == 1:
        partial_convert_unsqueeze_to_expand_dims(nnefop, converter)
    else:
        partial_convert_unsqueeze_to_reshape(nnefop, converter)


def convert_transpose(nnefop, converter):
    tfop = dog.OperationNode("tf.transpose")
    tfop.add_arg("a", nnefop.args["input"])
    # TODO something might be needed like adding axes which are not written out
    tfop.add_arg("perm", nnefop.args["axes"])
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_concat(nnefop, converter):
    nnefdns = nnefop.args["values"]
    tfop = dog.OperationNode("tf.concat")
    tfop.add_arg("values", [converter.get_tfdn_safe(nnefdn) for nnefdn in nnefdns])
    tfop.add_arg("axis", nnefop.args['axis'])
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_split(nnefop, converter):
    nnefaxis = nnefop.args["axis"]
    nnefdn_value = nnefop.args["value"]
    nnefratios = nnefop.args["ratios"]
    nnefshape = dog.get_shape_safe(nnefdn_value)
    if nnefshape[nnefaxis] == 0:
        if len(np.unique(nnefratios)) == 1:
            tf_num_or_size_splits = len(nnefratios)
        else:
            utils.print_error("Split with different ratios on an axis ({}) with unspecified size.".format(nnefaxis))
            tf_num_or_size_splits = len(nnefratios)
    else:
        part_size = nnefshape[nnefaxis] // int(np.sum(nnefratios))
        tf_num_or_size_splits = [ratio * part_size for ratio in nnefratios]

    tfop = dog.OperationNode("tf.split")
    tfop.add_arg("value", converter.get_tfdn_safe(nnefdn_value))
    tfop.add_arg("num_or_size_splits", tf_num_or_size_splits)
    tfop.add_arg("axis", nnefaxis)
    for i in range(len(nnefratios)):
        tfop.add_result("result{}".format(i), converter.make_tfdn(nnefop.result[i]))
    converter.add_tfop(tfop, nnefop)


def generic_convert_binary(nnefop, converter):
    tfop = dog.OperationNode(nnefopname_to_tf(nnefop.name))
    tfop.add_arg("x", converter.get_tfdn_safe(nnefop.args["x"]))
    tfop.add_arg("y", converter.get_tfdn_safe(nnefop.args["y"]))
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def generic_convert_unary(nnefop, converter):
    tfop = dog.OperationNode(nnefopname_to_tf(nnefop.name))
    tfop.add_arg("x", converter.get_tfdn_safe(nnefop.args["x"]))
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_rsqr(nnefop, converter):
    tfop = dog.OperationNode("tf.pow")
    tfop.add_arg("x", converter.get_tfdn_safe(nnefop.args["x"]))
    tfop.add_arg("y", -2.0)
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_select(nnefop, converter):
    x = converter.get_tfdn_safe(nnefop.args["true_value"])
    y = converter.get_tfdn_safe(nnefop.args["false_value"])
    cond_shape = dog.get_shape_safe(nnefop.args["condition"])

    if isinstance(x, (float, int, bool)):
        tfop_constant = dog.OperationNode("tf.constant")
        tfop_constant.add_arg("value", x)
        tfop_constant.add_arg("dtype", utils.tf_type_of_python_scalar(x))
        tfop_constant.add_arg("shape", cond_shape)
        tfdn_constant = dog.DataNode(nnefop.result.name + "_x__")
        tfdn_constant.shape = cond_shape
        tfop_constant.add_result("result", converter.make_tfdn(tfdn_constant))
        converter.add_tfop(tfop_constant, nnefop)
        x = tfdn_constant

    if isinstance(y, (float, int, bool)):
        tfop_constant = dog.OperationNode("tf.constant")
        tfop_constant.add_arg("value", y)
        tfop_constant.add_arg("dtype", utils.tf_type_of_python_scalar(y))
        tfop_constant.add_arg("shape", cond_shape)
        tfdn_constant = dog.DataNode(nnefop.result.name + "_y__")
        tfdn_constant.shape = cond_shape
        tfop_constant.add_result("result", converter.make_tfdn(tfdn_constant))
        converter.add_tfop(tfop_constant, nnefop)
        y = tfdn_constant

    tfop = dog.OperationNode("tf.where")
    tfop.add_arg("condition", converter.get_tfdn_safe(nnefop.args["condition"]))
    tfop.add_arg("x", x)
    tfop.add_arg("y", y)
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_clamp(nnefop, converter):
    tfop = dog.OperationNode("tf.clip_by_value")
    tfop.add_arg("t", converter.get_tfdn_safe(nnefop.args["x"]))
    tfop.add_arg("clip_value_min", converter.get_tfdn_safe(nnefop.args["a"]))
    tfop.add_arg("clip_value_max", converter.get_tfdn_safe(nnefop.args["b"]))
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_matmul(nnefop, converter):
    tfop = dog.OperationNode("tf.matmul")
    tfop.add_arg("a", converter.get_tfdn_safe(nnefop.args["A"]))
    tfop.add_arg("b", converter.get_tfdn_safe(nnefop.args["B"]))
    tfop.add_arg("transpose_a", nnefop.args["transposeA"])
    tfop.add_arg("transpose_b", nnefop.args["transposeB"])
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def partial_convert_conv_to_conv2d_or_conv3d(nnefop, converter):
    d = len(dog.get_shape_safe(nnefop.args["input"])) - 2

    tfop = dog.OperationNode("tf.nn.conv2d" if d == 2 else "tf.nn.conv3d")

    tfop.add_arg("input", converter.get_tfdn_safe(nnefop.args["input"]))
    tfop.add_arg("filter", converter.get_tfdn_safe(nnefop.args["filter"]))
    tfop.add_arg("strides", spatial_nhwc(nnefop.args["stride"], in_spatial=True,
                                         out_spatial=False, empty_value=[1] * (d + 2)))
    tfop.add_arg("padding", nnefop.args["_tf_padding"])

    if utils.has_greater_than_1(nnefop.args["dilation"]):
        tfop.add_arg("dilations", spatial_nhwc(nnefop.args["dilation"], in_spatial=True,
                                               out_spatial=False, empty_value=[1] * (d + 2)))

    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop)


def partial_convert_conv_to_convolution(nnefop, converter):
    d = len(dog.get_shape_safe(nnefop.args["input"])) - 2

    tfop = dog.OperationNode("tf.nn.convolution")
    tfop.add_arg("input", converter.get_tfdn_safe(nnefop.args["input"]))
    tfop.add_arg("filter", converter.get_tfdn_safe(nnefop.args["filter"]))
    tfop.add_arg("padding", nnefop.args["_tf_padding"])
    if utils.has_greater_than_1(nnefop.args["stride"]):
        tfop.add_arg("strides", nnefop.args["stride"])
    if utils.has_greater_than_1(nnefop.args["dilation"]):
        tfop.add_arg("dilation_rate", nnefop.args["dilation"])
    if utils.has_greater_than_1(nnefop.args["stride"]) and utils.has_greater_than_1(nnefop.args["dilation"]):
        utils.print_error("Custom stride AND dilation is not supported by tf.nn.convolution!")
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop)


def partial_convert_conv_to_deptwise_conv2d(nnefop, converter):
    tfop = dog.OperationNode("tf.nn.depthwise_conv2d")
    tfop.add_arg("input", converter.get_tfdn_safe(nnefop.args["input"]))
    tfop.add_arg("filter", converter.get_tfdn_safe(nnefop.args["filter"]))
    tfop.add_arg("strides", spatial_nhwc(nnefop.args["stride"],
                                         in_spatial=True, out_spatial=False, empty_value=[1] * 4))
    tfop.add_arg("padding", nnefop.args["_tf_padding"])
    if utils.has_greater_than_1(nnefop.args["dilation"]):
        tfop.add_arg("rate", spatial_nhwc(nnefop.args["dilation"], in_spatial=True, out_spatial=True, empty_value=None))
    if utils.has_greater_than_1(nnefop.args["stride"]) and utils.has_greater_than_1(nnefop.args["dilation"]):
        utils.print_error("Custom stride AND dilation is not supported by tf.nn.depthwise_conv2d!")
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop)


def convert_conv(nnefop, converter):
    groups = nnefop.args["groups"]
    nnef_input_shape = dog.get_shape_safe(nnefop.args["input"])
    d = len(nnef_input_shape) - 2

    if not nnefop.args["stride"]:
        nnefop.args["stride"] = [1 for _ in range(d)]
    if not nnefop.args["dilation"]:
        nnefop.args["dilation"] = [1 for _ in range(d)]

    dilated = utils.has_greater_than_1(nnefop.args["dilation"])

    if groups == 1:
        if d in [2, 3] and not dilated:
            partial_convert_conv_to_conv2d_or_conv3d(nnefop, converter)
        else:
            partial_convert_conv_to_convolution(nnefop, converter)
    else:
        if groups in [0, nnef_input_shape[-1]] and d == 2:
            partial_convert_conv_to_deptwise_conv2d(nnefop, converter)
        else:
            # TODO consider implementing this with split and conv
            utils.print_error(
                "Grouped convolutions are only supported if they can be converted to tf.nn.depthwise_conv2d.")
            return


def partial_convert_conv_grad_input_normal_2d(nnefop, converter):
    tfop = dog.OperationNode("tf.nn.conv2d_backprop_input")
    tfop.add_arg("input_sizes", nnefop.args["orig_input_shape"])
    tfop.add_arg("filter", converter.get_tfdn_safe(nnefop.args["orig_filter"]))
    tfop.add_arg("out_backprop", converter.get_tfdn_safe(nnefop.args["output_grad"]))
    tfop.add_arg("strides", spatial_nhwc(nnefop.args["stride"],
                                         in_spatial=True, out_spatial=False, empty_value=[1] * 4))
    tfop.add_arg("padding", calculate_tfpadding_for_deconv(nnefop))
    if utils.has_greater_than_1(nnefop.args["dilation"]):
        tfop.add_arg("dilations", spatial_nhwc(nnefop.args["dilation"],
                                               in_spatial=True, out_spatial=False, empty_value=[1] * 4))
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop)


def partial_convert_conv_grad_input_normal_3d(nnefop, converter):
    tfop = dog.OperationNode("tf.nn.conv3d_transpose")
    tfop.add_arg("value", converter.get_tfdn_safe(nnefop.args["output_grad"]))
    tfop.add_arg("filter", converter.get_tfdn_safe(nnefop.args["orig_filter"]))
    tfop.add_arg("output_shape", nnefop.args["orig_input_shape"])
    tfop.add_arg("strides", spatial_nhwc(nnefop.args["stride"],
                                         in_spatial=True, out_spatial=False, empty_value=[1] * 5))
    tfop.add_arg("padding", calculate_tfpadding_for_deconv(nnefop))
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop)


def partial_convert_conv_grad_input_planewise(nnefop, converter):
    tfop = dog.OperationNode("tf.nn.depthwise_conv2d_native_backprop_input")
    tfop.add_arg("input_sizes", nnefop.args["orig_input_shape"])
    tfop.add_arg("filter", converter.get_tfdn_safe(nnefop.args["orig_filter"]))
    tfop.add_arg("out_backprop", converter.get_tfdn_safe(nnefop.args["output_grad"]))
    tfop.add_arg("strides", spatial_nhwc(nnefop.args["stride"],
                                         in_spatial=True, out_spatial=False, empty_value=[1] * 4))
    tfop.add_arg("padding", calculate_tfpadding_for_deconv(nnefop))
    if utils.has_greater_than_1(nnefop.args["dilation"]):
        tfop.add_arg("rate",
                     spatial_nhwc(nnefop.args["dilation"],
                                  in_spatial=True, out_spatial=True, empty_value=None))
    if utils.has_greater_than_1(nnefop.args["stride"]) and utils.has_greater_than_1(nnefop.args["dilation"]):
        utils.print_error(
            "Custom stride AND dilation is not supported by tf.nn.depthwise_conv2d_native_backprop_input!")
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop)


def convert_conv_grad_input(nnefop, converter):
    groups = nnefop.args["groups"]
    nnef_input_shape = nnefop.args["orig_input_shape"]
    d = len(nnef_input_shape) - 2

    if not nnefop.args["stride"]:
        nnefop.args["stride"] = [1 for _ in range(d)]
    if not nnefop.args["dilation"]:
        nnefop.args["dilation"] = [1 for _ in range(d)]

    dilated = utils.has_greater_than_1(nnefop.args["dilation"])

    if groups == 1:
        if d == 2:
            partial_convert_conv_grad_input_normal_2d(nnefop, converter)
        elif d == 3 and not dilated:
            partial_convert_conv_grad_input_normal_3d(nnefop, converter)
        else:
            utils.print_error("Unsupported conv_grad_input: {}d, {}".format(d, "dilated" if dilated else "not dilated"))
    else:
        if groups in [0, nnef_input_shape[-1]] and d == 2:
            partial_convert_conv_grad_input_planewise(nnefop, converter)
        else:
            # TODO consider implementing this
            utils.print_error(
                "Grouped deconvolutions are only supported if they can be converted to "
                + "tf.nn.depthwise_conv2d_native_backprop_input.")
            return


def partial_convert_conv_grad_filter_normal(nnefop, converter):
    d = len(dog.get_shape_safe(nnefop.args["orig_input"])) - 2

    tfop = dog.OperationNode("tf.nn.conv2d_backprop_filter" if d == 2 else "tf.nn.conv3d_backprop_filter_v2")
    tfop.add_arg("input", converter.get_tfdn_safe(nnefop.args["orig_input"]))
    tfop.add_arg("filter_sizes", nnefop.args["orig_filter_shape"])
    tfop.add_arg("out_backprop", converter.get_tfdn_safe(nnefop.args["output_grad"]))
    tfop.add_arg("strides", spatial_nhwc(nnefop.args["stride"],
                                         in_spatial=True, out_spatial=False, empty_value=[1] * (d + 2)))
    tfop.add_arg("padding", calculate_tfpadding_for_deconv(nnefop))
    if utils.has_greater_than_1(nnefop.args["dilation"]):
        tfop.add_arg("dilations", spatial_nhwc(nnefop.args["dilation"],
                                               in_spatial=True, out_spatial=False, empty_value=[1] * (d + 2)))
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop)


def partial_convert_conv_grad_filter_planewise(nnefop, converter):
    tfop = dog.OperationNode("tf.nn.depthwise_conv2d_native_backprop_filter")
    tfop.add_arg("input", converter.get_tfdn_safe(nnefop.args["orig_input"]))
    tfop.add_arg("filter_sizes", nnefop.args["orig_filter_shape"])
    tfop.add_arg("out_backprop", converter.get_tfdn_safe(nnefop.args["output_grad"]))
    tfop.add_arg("strides", spatial_nhwc(nnefop.args["stride"],
                                         in_spatial=True, out_spatial=False, empty_value=[1] * 4))
    tfop.add_arg("padding", calculate_tfpadding_for_deconv(nnefop))
    if utils.has_greater_than_1(nnefop.args["dilation"]):
        tfop.add_arg("rate",
                     spatial_nhwc(nnefop.args["dilation"], in_spatial=True, out_spatial=True, empty_value=None))
    if utils.has_greater_than_1(nnefop.args["stride"]) and utils.has_greater_than_1(nnefop.args["dilation"]):
        utils.print_error(
            "Custom stride AND dilation is not supported by tf.nn.depthwise_conv2d_native_backprop_filter!")
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop)


def convert_conv_grad_filter(nnefop, converter):
    groups = nnefop.args["groups"]
    nnef_input_shape = dog.get_shape_safe(nnefop.args["orig_input"])
    d = len(nnef_input_shape) - 2

    if not nnefop.args["stride"]:
        nnefop.args["stride"] = [1 for _ in range(d)]
    if not nnefop.args["dilation"]:
        nnefop.args["dilation"] = [1 for _ in range(d)]

    if groups == 1:
        if d in [2, 3]:
            partial_convert_conv_grad_filter_normal(nnefop, converter)
        else:
            utils.print_error("Unsupported conv_grad_filter: {}d".format(d))
    else:
        if groups in [0, nnef_input_shape[-1]] and d == 2:
            partial_convert_conv_grad_filter_planewise(nnefop, converter)
        else:
            # TODO consider implementing this
            utils.print_error(
                "Grouped convolutions are only supported if they can be converted to "
                + "tf.nn.depthwise_conv2d_native_backprop_filter.")
            return


def partial_convert_deconv_to_conv2d_transpose_or_conv3d_transpose(nnefop, converter):
    d = len(dog.get_shape_safe(nnefop.args["input"])) - 2

    tfop = dog.OperationNode("tf.nn.conv2d_transpose" if d == 2 else "tf.nn.conv3d_transpose")
    tfop.add_arg("value", converter.get_tfdn_safe(nnefop.args["input"]))
    tfop.add_arg("filter", converter.get_tfdn_safe(nnefop.args["filter"]))
    # if isinstance(nnefop.args["output_shape"], nnef.ShapeOf):
    #     tfop.add_arg('output_shape', converter.nnef_shapeof_to_tf(nnefop.args["output_shape"]))
    # else:
    if nnefop.args["output_shape"]:
        tfop.add_arg("output_shape", nnefop.args["output_shape"])
    tfop.add_arg("strides", spatial_nhwc(nnefop.args["stride"],
                                         in_spatial=True, out_spatial=False, empty_value=[1] * (d + 2)))
    tfop.add_arg("padding", calculate_tfpadding_for_deconv(nnefop))
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop)


def partial_convert_deconv_to_atrous_conv2d_transpose(nnefop, converter):
    tfdilation = spatial_nhwc(nnefop.args["dilation"], in_spatial=True, out_spatial=True, empty_value=[1] * 2)

    if len(np.unique(tfdilation)) > 1:
        utils.print_error("Cannot specify different x and y dilation in tf.nn.atrous_conv2d_transpose.")

    tfop = dog.OperationNode("tf.nn.atrous_conv2d_transpose")
    tfop.add_arg("value", converter.get_tfdn_safe(nnefop.args["input"]))
    tfop.add_arg("filters", converter.get_tfdn_safe(nnefop.args["filter"]))
    # if isinstance(nnefop.args["output_shape"], nnef.ShapeOf):
    #     tfop.add_arg('output_shape', converter.nnef_shapeof_to_tf(nnefop.args["output_shape"]))
    # else:
    tfop.add_arg("output_shape", nnefop.args["output_shape"])
    tfop.add_arg("rate", tfdilation[0])
    tfop.add_arg("padding", calculate_tfpadding_for_deconv(nnefop))

    if utils.has_greater_than_1(nnefop.args["stride"]):
        utils.print_error("Cannot use stride>1 in tf.nn.atrous_conv2d_transpose.")

    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop)


def convert_deconv(nnefop, converter):
    groups = nnefop.args["groups"]
    nnef_input_shape = dog.get_shape_safe(nnefop.args["input"])
    d = len(nnef_input_shape) - 2

    if not nnefop.args["stride"]:
        nnefop.args["stride"] = [1 for _ in range(d)]
    if not nnefop.args["dilation"]:
        nnefop.args["dilation"] = [1 for _ in range(d)]

    is_dilated = utils.has_greater_than_1(nnefop.args["dilation"])

    if groups == 1:
        if not is_dilated and d in [2, 3]:
            partial_convert_deconv_to_conv2d_transpose_or_conv3d_transpose(nnefop, converter)
        elif is_dilated and d == 2:
            partial_convert_deconv_to_atrous_conv2d_transpose(nnefop, converter)
        else:
            utils.print_error(
                "{} dimensional{} deconv is not supported by TensorFlow.".format(d, " dilated" if is_dilated else ""))
    else:
        utils.print_error("Deconv with groups>1 is not supported by TensorFlow.")


def generic_convert_pooling(nnefop, converter):
    rank = len(dog.get_shape_safe(nnefop.args["input"]))

    if not nnefop.args["stride"]:
        nnefop.args["stride"] = [1 for _ in range(rank)]
    # if not nnefop.args["dilation"]:
    #     nnefop.args["dilation"] = [1 for _ in range(rank)]
    if nnefop.args["dilation"]:
        utils.print_error("Dilated pool is not supported for now")

    tfargname_input = "input" if nnefop.name in ["max_pool_with_index", "argmax_pool"] else "value"

    tfop = dog.OperationNode(nnefopname_to_tf(nnefop.name))
    tfop.add_arg(tfargname_input, converter.get_tfdn_safe(nnefop.args["input"]))
    tfop.add_arg("ksize", nnefop.args["size"])
    tfop.add_arg("strides", nnefop.args["stride"])
    tfop.add_arg("padding", nnefop.args["_tf_padding"])  # TODO padding?

    if nnefop.name == "argmax_pool":
        tfop.add_result("output", converter.make_tfdn(nnefop.result, discriminator="output"))
        tfop.add_result("argmax", converter.make_tfdn(nnefop.result))
    elif nnefop.name == "max_pool_with_index":
        tfop.add_result("output", converter.make_tfdn(nnefop.results["output"]))
        tfop.add_result("argmax", converter.make_tfdn(nnefop.results["index"]))
    else:
        tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop)


def generic_convert_pool_grad(nnefop, converter):
    if nnefop.name == "avg_pool_grad":
        rank = len(nnefop.args["orig_input_shape"])
    else:
        rank = len(dog.get_shape_safe(nnefop.args["orig_input"]))

    if not nnefop.args["stride"]:
        nnefop.args["stride"] = [1 for _ in range(rank)]
    if not nnefop.args["dilation"]:
        nnefop.args["dilation"] = [1 for _ in range(rank)]

    tfop = dog.OperationNode(nnefopname_to_tf(nnefop.name))
    if nnefop.name == "max_pool_grad_with_index":
        tfop.add_arg("input", converter.get_tfdn_safe(nnefop.args["orig_input"]))
        tfop.add_arg("grad", converter.get_tfdn_safe(nnefop.args["output_grad"]))
        tfop.add_arg("argmax", converter.get_tfdn_safe(nnefop.args["orig_index"]))
    elif nnefop.name == "max_pool_grad":
        tfop.add_arg("orig_input", converter.get_tfdn_safe(nnefop.args["orig_input"]))
        tfop.add_arg("orig_output", converter.get_tfdn_safe(nnefop.args["orig_output"]))
        tfop.add_arg("grad", converter.get_tfdn_safe(nnefop.args["output_grad"]))
    elif nnefop.name == "avg_pool_grad":
        tfop.add_arg("orig_input_shape", nnefop.args["orig_input_shape"])
        tfop.add_arg("grad", converter.get_tfdn_safe(nnefop.args["output_grad"]))
    else:
        assert False

    tfop.add_arg("ksize", nnefop.args["size"])
    tfop.add_arg("strides", spatial_nhwc(nnefop.args["stride"],
                                         in_spatial=False, out_spatial=False, empty_value=[1] * rank))
    tfop.add_arg("padding", nnefop.args["_tf_padding"])
    tfop.add_result("result", converter.make_tfdn(nnefop.results["input_grad"]))
    converter.add_tfop(tfop)


def generic_convert_upsample_downsample(nnefop, converter):
    nnef_input_shape = dog.get_shape_safe(nnefop.args["input"])
    nnef_factors = nnefop.args["factor"]
    is_downsample = "down" in nnefop.name
    output_size = [(int(i / f) if is_downsample else int(i * f)) for i, f in zip(nnef_input_shape[1:-1], nnef_factors)]

    tfop = dog.OperationNode(nnefopname_to_tf(nnefop.name))
    tfop.add_arg("images", converter.get_tfdn_safe(nnefop.args["input"]))
    tfop.add_arg("size", output_size)
    if nnefop.name == "multilinear_upsample":
        # TODO implement full border and method support
        if nnefop.args["border"].lower() != "replicate":
            utils.print_error("border={} is unsupported in multilinear_upsample.".format(nnefop.args["border"]))
        if nnefop.args["method"].lower() not in ["aligned", "asymmetric"]:
            utils.print_error("method={} is unsupported in multilinear_upsample.".format(nnefop.args["method"]))
        tfop.add_arg("align_corners", nnefop.args["method"].lower() == 'aligned')
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def generic_convert_reduce(nnefop, converter):
    tfopname = nnefopname_to_tf(nnefop.name)
    if nnefop.name == "sum_reduce" and nnefop.args["normalize"]:
        tfopname = "tf.mean_reduce"

    tfinput = converter.get_tfdn_safe(nnefop.args["input"])
    if isinstance(tfinput, float):  # TF can't reduce a single value
        tfinput = [tfinput]

    tfop = dog.OperationNode(tfopname)
    tfop.add_arg("input_tensor", tfinput)
    tfop.add_arg("axis", nnefop.args["axes"])
    tfop.add_arg(tf_compat.reduce_arg_keepdims, True)
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_moments(nnefop, converter):
    tfop = dog.OperationNode("tf.nn.moments")
    tfop.add_arg("x", converter.get_tfdn_safe(nnefop.args["input"]))
    tfop.add_arg("axes", nnefop.args["axes"])
    tfop.add_arg("keep_dims", True)
    tfop.add_result("mean", converter.make_tfdn(nnefop.results["mean"]))
    tfop.add_result("variance", converter.make_tfdn(nnefop.results["variance"]))
    converter.add_tfop(tfop, nnefop)


def generic_convert_activation(nnefop, converter):
    tfop = dog.OperationNode(nnefopname_to_tf(nnefop.name))
    tfop.add_arg("features", converter.get_tfdn_safe(nnefop.args["x"]))
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_softmax(nnefop, converter):
    tfop = dog.OperationNode("tf.nn.softmax")
    tfop.add_arg("logits", converter.get_tfdn_safe(nnefop.args["x"]))
    rank = len(dog.get_shape_safe(nnefop.args["x"]))
    nnefaxes = nnefop.args["axes"]
    if len(nnefaxes) != 1:  # btw using rank-1 causes error on tf 1.0.0, while -1 is ok
        utils.print_error("In softmax 'axes' parameter is only supported if it has exactly one value in it.")
    if nnefaxes[0] != rank - 1:
        tfop.add_arg(tf_compat.softmax_arg_axis, nnefaxes[0])
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_leaky_relu(nnefop, converter):
    tfop = dog.OperationNode("tf.nn.leaky_relu")
    tfop.add_arg("features", converter.get_tfdn_safe(nnefop.args["x"]))
    tfop.add_arg("alpha", converter.get_tfdn_safe(nnefop.args["alpha"]))
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_prelu(nnefop, converter):
    convert_leaky_relu(nnefop, converter)


def convert_local_response_normalization(nnefop, converter):
    nnefsize = list(nnefop.args["size"])
    if len(nnefsize) < 2:
        utils.print_error("arg 'size' of local_response_normalization must have at least 2 elements")
        nnefsize = [1, 11]
    depth_size = nnefsize[-1]
    nnefsize[-1] = 1
    if utils.has_greater_than_1(nnefsize):
        utils.print_error("local_response_normalization only supported when only the last element of 'size' is > 1")

    depth_radius = (depth_size - 1) / 2
    alpha = nnefop.args["alpha"] / depth_size

    tfop = dog.OperationNode("tf.nn.local_response_normalization")
    tfop.add_arg("input", converter.get_tfdn_safe(nnefop.args["input"]))
    tfop.add_arg("depth_radius", depth_radius)
    tfop.add_arg("bias", nnefop.args["bias"])
    tfop.add_arg("alpha", alpha)
    tfop.add_arg("beta", nnefop.args["beta"])
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_l2_normalization(nnefop, converter):
    # TODO epsilon and bias are not really the same, is this a problem?
    tfop = dog.OperationNode("tf.nn.l2_normalize")
    tfop.add_arg("x", converter.get_tfdn_safe(nnefop.args["input"]))
    tfop.add_arg(tf_compat.l2_normalize_arg_axis, nnefop.args["axes"])
    tfop.add_arg("epsilon", nnefop.args["bias"])
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_batch_normalization(nnefop, converter):
    tfop = dog.OperationNode("tf.nn.batch_normalization ")
    tfop.add_arg("x", converter.get_tfdn_safe(nnefop.args["input"]))
    tfop.add_arg("mean", converter.get_tfdn_safe(nnefop.args["mean"]))
    tfop.add_arg("variance", converter.get_tfdn_safe(nnefop.args["variance"]))
    tfop.add_arg("offset", nnefop.args["offset"])
    tfop.add_arg("scale", nnefop.args["scale"])
    tfop.add_arg("variance_epsilon", nnefop.args["epsilon"])
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_copy_n(nnefop, converter):
    if len(nnefop.result) != nnefop.args["times"]:
        utils.print_error("copy_n must have 'times' outputs")

    for nnefdn_result in nnefop.result:
        tfop = dog.OperationNode("tf.identity")
        tfop.add_arg("input", converter.get_tfdn_safe(nnefop.args["x"]))
        tfop.add_result("result", converter.make_tfdn(nnefdn_result))
        converter.add_tfop(tfop, nnefop)


def convert_add_n(nnefop, converter):
    tfop = dog.OperationNode("tf.add_n")
    tfop.add_arg("inputs", [converter.get_tfdn_safe(nnefdn) for nnefdn in nnefop.args["x"]])
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert__nnef_pad(nnefop, converter):
    tfop = dog.OperationNode("tf.pad")
    tfop.add_arg("tensor", converter.get_tfdn_safe(nnefop.args["input"]))
    tfop.add_arg("paddings", nnefop.args["padding"])
    tf_pad_mode = nnefborder_to_tf(nnefop.args["border"])
    if tf_pad_mode != "CONSTANT":
        tfop.add_arg("mode", tf_pad_mode)
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop.extra.get("original_op", nnefop))


def convert__nnef_bias_add(nnefop, converter):
    tfop = dog.OperationNode("tf.nn.bias_add")
    tfop.add_arg("value", converter.get_tfdn_safe(nnefop.args["x"]))
    tfop.add_arg("bias", converter.get_tfdn_safe(nnefop.args["y"]))
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop.extra.get("original_op", nnefop))


def convert_zeros(nnefop, converter):
    tfshape = nnefop.args["shape"]
    tfop = dog.OperationNode("tf.zeros")
    tfop.add_arg("shape", tfshape)
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_ones(nnefop, converter):
    tfshape = nnefop.args["shape"]
    tfop = dog.OperationNode("tf.ones")
    tfop.add_arg("shape", tfshape)
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_zeros_like(nnefop, converter):
    tfop = dog.OperationNode("tf.zeros_like")
    tfop.add_arg("tensor", converter.get_tfdn_safe(nnefop.args["reference"]))
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_ones_like(nnefop, converter):
    tfop = dog.OperationNode("tf.ones_like")
    tfop.add_arg("tensor", converter.get_tfdn_safe(nnefop.args["reference"]))
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_slice(nnefop, converter):
    rank = len(dog.get_shape_safe(nnefop.args["input"]))
    axes = nnefop.args["axes"]
    begin = nnefop.args["begin"]
    end = nnefop.args["end"]

    size = [
        -1 if e == -1 else e - b
        for b, e in zip(begin, end)
    ]

    tfbegin, tfsize = utils.zip_inverse(2, [
        (begin[axes.index(i)], size[axes.index(i)]) if i in axes else (0, -1)
        for i in range(rank)
    ])

    tfop = dog.OperationNode("tf.slice")
    tfop.add_arg("input_", converter.get_tfdn_safe(nnefop.args["input"]))
    tfop.add_arg("begin", tfbegin)
    tfop.add_arg("size", tfsize)
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_stack(nnefop, converter):
    nnefdns = nnefop.args["values"]
    tfop = dog.OperationNode("tf.stack")
    tfop.add_arg("values", [converter.get_tfdn_safe(nnefdn) for nnefdn in nnefdns])
    tfaxis = nnefop.args['axis']
    tfop.add_arg("axis", tfaxis)
    tfop.add_result("result", converter.make_tfdn(nnefop.result))
    converter.add_tfop(tfop, nnefop)


def convert_unstack(nnefop, converter):
    nnefaxis = nnefop.args["axis"]
    nnefdn_value = nnefop.args["value"]
    nnefshape = dog.get_shape_safe(nnefdn_value)

    tfop = dog.OperationNode("tf.unstack")
    tfop.add_arg("value", converter.get_tfdn_safe(nnefdn_value))
    tfop.add_arg("num", len(nnefop.result))
    tfop.add_arg("axis", nnefaxis)

    for i in range(len(nnefop.result)):
        tfop.add_result("result{}".format(i), converter.make_tfdn(nnefop.result[i]))

    converter.add_tfop(tfop, nnefop)


def convert_box(nnefop, converter):
    # only padding
    if (not utils.has_not_equal_1(nnefop.args["size"])
            and nnefop.args["border"] in ["constant", "reflect", "reflect-even"]
            and not utils.has_not_equal_1(nnefop.args["stride"])
            and not utils.has_not_equal_1(nnefop.args["dilation"])):
        tfop = dog.OperationNode("tf.pad")
        tfop.add_arg("tensor", converter.get_tfdn_safe(nnefop.args["input"]))
        tfop.add_arg("paddings", nnefop.args["padding"])
        tfmode = nnefborder_to_tf(nnefop.args["border"])
        if tfmode != 'CONSTANT':
            tfop.add_arg("mode", tfmode)
        tfop.add_result("result", converter.make_tfdn(nnefop.result))
        converter.add_tfop(tfop, nnefop)
    else:
        utils.print_error("Box is not yet fully supported, your use is invalid.")


def generic_convert_unsupported(nnefop, converter):
    utils.print_error("{} is not supported".format(nnefop.name))


generic_converters = {
    "add": (generic_convert_binary, "tf.add"),
    "sub": (generic_convert_binary, "tf.subtract"),
    "mul": (generic_convert_binary, "tf.multiply"),
    "div": (generic_convert_binary, "tf.divide"),
    "pow": (generic_convert_binary, "tf.pow"),
    "lt": (generic_convert_binary, "tf.less"),
    "gt": (generic_convert_binary, "tf.greater"),
    "le": (generic_convert_binary, "tf.less_equal"),
    "ge": (generic_convert_binary, "tf.greater_equal"),
    "eq": (generic_convert_binary, "tf.equal"),
    "ne": (generic_convert_binary, "tf.not_equal"),
    "and": (generic_convert_binary, "tf.logical_and"),
    "or": (generic_convert_binary, "tf.logical_or"),
    "min": (generic_convert_binary, "tf.minimum"),
    "max": (generic_convert_binary, "tf.maximum"),
    "exp": (generic_convert_unary, "tf.exp"),
    "log": (generic_convert_unary, "tf.log"),
    "abs": (generic_convert_unary, "tf.abs"),
    "sign": (generic_convert_unary, "tf.sign"),
    "rcp": (generic_convert_unary, "tf.reciprocal"),
    "neg": (generic_convert_unary, "tf.negative"),
    "floor": (generic_convert_unary, "tf.floor"),
    "ceil": (generic_convert_unary, "tf.ceil"),
    "round": (generic_convert_unary, "tf.round"),
    "sqr": (generic_convert_unary, "tf.square"),
    "sqrt": (generic_convert_unary, "tf.sqrt"),
    "rsqrt": (generic_convert_unary, "tf.rsqrt"),
    "not": (generic_convert_unary, "tf.logical_not"),
    "sigmoid": (generic_convert_unary, "tf.sigmoid"),
    "sinh": (generic_convert_unary, "tf.sinh"),
    "cosh": (generic_convert_unary, "tf.cosh"),
    "tanh": (generic_convert_unary, "tf.tanh"),
    "nearest_downsample": (generic_convert_upsample_downsample, "tf.image.resize_nearest_neighbor"),
    "area_downsample": (generic_convert_upsample_downsample, "tf.image.resize_area"),
    "nearest_upsample": (generic_convert_upsample_downsample, "tf.image.resize_nearest_neighbor"),
    "multilinear_upsample": (generic_convert_upsample_downsample, "tf.image.resize_bilinear"),
    "sum_reduce": (generic_convert_reduce, "tf.reduce_sum"),
    "min_reduce": (generic_convert_reduce, "tf.reduce_min"),
    "max_reduce": (generic_convert_reduce, "tf.reduce_max"),
    "mean_reduce": (generic_convert_reduce, "tf.reduce_mean"),
    "elu": (generic_convert_activation, "tf.nn.elu"),
    "relu": (generic_convert_activation, "tf.nn.relu"),
    "softplus": (generic_convert_activation, "tf.nn.softplus"),
    "argmax_pool": (generic_convert_pooling, "tf.nn.max_pool_with_argmax"),
    "max_pool_with_index": (generic_convert_pooling, "tf.nn.max_pool_with_argmax"),
    "max_pool": (generic_convert_pooling, "tf.nn.max_pool"),
    "avg_pool": (generic_convert_pooling, "tf.nn.avg_pool"),
    "max_pool_grad": (generic_convert_pool_grad, tf_compat.gen_nn_ops_max_pool_grad_name),
    "max_pool_grad_with_index": (generic_convert_pool_grad, tf_compat.gen_nn_ops_max_pool_grad_with_argmax_name),
    "avg_pool_grad": (generic_convert_pool_grad, tf_compat.gen_nn_ops_avg_pool_grad_name),
    "debox": (generic_convert_unsupported, ""),
    "sample": (generic_convert_unsupported, ""),
    "desample": (generic_convert_unsupported, ""),
    "local_mean_normalization": (generic_convert_unsupported, ""),
    "local_variance_normalization": (generic_convert_unsupported, ""),
    "local_contrast_normalization": (generic_convert_unsupported, ""),
    "linear_quantize": (generic_convert_unsupported, ""),
    "logarithmic_quantize": (generic_convert_unsupported, ""),
    "binary_quantize": (generic_convert_unsupported, ""),
    "ternary_quantize": (generic_convert_unsupported, "")
}


def nnefopname_to_tf(nnefopname):
    return generic_converters[nnefopname][1]


DefaultConverters = {fun.__name__[len("convert_"):]: fun for fun in utils.get_functions(prefix="convert_")}
for k, v in generic_converters.items():
    DefaultConverters[k] = v[0]
