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

import caffe
import google.protobuf.message as protobuf_message
from caffe.proto.caffe_pb2 import LayerParameter

from ...common import CaffeOp
from ...common import EXTRA_WEIGHTS
from ....common import dog
from ....common import utils
from ....common.types import *


class Converter:
    def __init__(self, net):
        # type: (caffe.Net)->None
        self.input_names = []  # type: List[str]
        self.net = net  # type: caffe.Net

    def add_input_name(self, input_name):
        # type: (str)->None
        self.input_names.append(input_name)

    def add_variables(self, layer, op, *var_names):
        # type: (LayerParameter, CaffeOp, List[str])->None
        vars = self.net.params[layer.name]
        weights = {}
        for var, name in zip(vars, var_names):  # maybe skipping some, that are not set
            weights[name] = var.data.copy()
        op.extra[EXTRA_WEIGHTS] = weights

    @staticmethod
    def get_size(
            message,  # type: protobuf_message.Message
            field_name_prefix,  # type: str
            size_rank,  # type: int
            h_suffix="_h",  # type: str
            w_suffix="_w",  # type: str
            size_suffix="_size",  # type: str
            can_have_hw=True,  # type: bool
            size_is_list=True,  # type: bool
            default_value=None  # type: Optional[int]
    ):
        # type: (...)->List[int]

        h_name = field_name_prefix + h_suffix
        w_name = field_name_prefix + w_suffix
        size_name = field_name_prefix + size_suffix

        if can_have_hw and message.HasField(h_name):
            size = [int(getattr(message, h_name)), int(getattr(message, w_name))]
        elif size_is_list and (getattr(message, size_name) or default_value is None):
            size = [int(s) for s in getattr(message, size_name)]
            if len(size) == 1:
                size = size_rank * size
        elif not size_is_list and (message.HasField(size_name) or default_value is None):
            size = size_rank * [getattr(message, size_name)]
        else:
            size = size_rank * [default_value]
        return size


def generic_convert_empty(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    pass


def generic_convert_input(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    op.add_arg("shapes", [dn.shape for dn in op.get_result_nodes()])
    for top in layer.top:
        converter.add_input_name(top)


def convert_scale(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.scale_param
    op.add_arg("axis", params.axis)
    op.add_arg("num_axes", params.num_axes)
    op.add_arg("bias_term", params.bias_term)
    converter.add_variables(layer, op, "weight", "bias")


def convert_power(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.power_param
    op.add_arg("power", params.power)
    op.add_arg("scale", params.scale)
    op.add_arg("shift", params.shift)


def generic_convert_convolution(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.convolution_param
    spatial_rank = len(op.args[dog.gen_arg_name(0)].shape) - 2
    op.add_arg("num_output", params.num_output)
    op.add_arg("bias_term", params.bias_term)
    op.add_arg("pad", converter.get_size(params, "pad", spatial_rank,
                                         size_suffix="", default_value=0))
    op.add_arg("kernel_size", converter.get_size(params, "kernel", spatial_rank))
    op.add_arg("stride", converter.get_size(params, "stride", spatial_rank,
                                            size_suffix="", default_value=1))
    op.add_arg("dilation", converter.get_size(params, "dilation", spatial_rank,
                                              size_suffix="", default_value=1, can_have_hw=False))
    op.add_arg("group", params.group)
    op.add_arg("axis", params.axis)

    op.add_arg("weight_filler_type", utils.ensure_not_unicode_in_python2(params.weight_filler.type))

    converter.add_variables(layer, op, "weight", "bias")


def convert_prelu(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.prelu_param
    op.add_arg("channel_shared", bool(params.channel_shared))
    converter.add_variables(layer, op, "alpha")


def convert_pooling(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.pooling_param
    spatial_rank = len(op.args[dog.gen_arg_name(0)].shape) - 2

    if params.global_pooling:
        kernel_size = op.args[dog.gen_arg_name(0)].shape[2:]
    else:
        kernel_size = converter.get_size(params, "kernel", spatial_rank, size_is_list=False)

    op.add_arg("pool", params.pool)
    op.add_arg("pad", converter.get_size(params, "pad", spatial_rank,
                                         size_suffix="", default_value=0, size_is_list=False))
    op.add_arg("kernel_size", kernel_size)
    op.add_arg("stride", converter.get_size(params, "stride", spatial_rank,
                                            size_suffix="", default_value=1, size_is_list=False))
    op.add_arg("global_pooling", params.global_pooling)


def convert_softmax(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.softmax_param
    op.add_arg("axis", params.axis)


def convert_reshape(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.reshape_param
    op.add_arg("shape", [int(d) for d in params.shape.dim])
    op.add_arg("axis", params.axis)
    op.add_arg("num_axes", params.num_axes)


def convert_flatten(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.flatten_param
    op.add_arg("axis", params.axis)
    op.add_arg("end_axis", params.end_axis)


def convert_argmax(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.argmax_param
    op.add_arg("out_max_val", params.out_max_val)
    op.add_arg("top_k", params.top_k)
    op.add_arg("axis", params.axis if params.HasField("axis") else None)


def convert_elu(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.elu_param
    op.add_arg("alpha", params.alpha)


def convert_relu(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.relu_param
    op.add_arg("negative_slope", params.negative_slope)


def convert_concat(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.concat_param

    if params.HasField("concat_dim"):
        op.add_arg("axis", params.concat_dim)
    else:
        op.add_arg("axis", params.axis)


def convert_eltwise(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.eltwise_param
    op.add_arg("operation", params.operation)
    op.add_arg("coeff", [float(c) for c in params.coeff])


def convert_dropout(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.dropout_param
    op.add_arg("dropout_ratio", params.dropout_ratio)


def convert_batch_norm(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.batch_norm_param
    op.add_arg("eps", params.eps)
    converter.add_variables(layer, op, "mean", "variance", "scale_factor")


def convert_inner_product(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.inner_product_param
    op.add_arg("num_output", params.num_output)
    op.add_arg("bias_term", params.bias_term)
    op.add_arg("axis", params.axis)
    op.add_arg("transpose", params.transpose)
    converter.add_variables(layer, op, "weight", "bias")


def convert_lrn(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.lrn_param
    op.add_arg("local_size", params.local_size)
    op.add_arg("alpha", params.alpha)
    op.add_arg("beta", params.beta)
    op.add_arg("norm_region", params.norm_region)
    op.add_arg("k", params.k)


def convert_crop(layer, op, converter):
    # type: (LayerParameter, CaffeOp, Converter)->None
    params = layer.crop_param
    op.add_arg("axis", params.axis)
    op.add_arg("offset", [int(o) for o in params.offset])


DefaultConverters = {
    "BatchNorm": convert_batch_norm,
    "BNLL": generic_convert_empty,
    "TanH": generic_convert_empty,
    "AbsVal": generic_convert_empty,
    "Sigmoid": generic_convert_empty,
    "Convolution": generic_convert_convolution,
    "Deconvolution": generic_convert_convolution,
    "Input": generic_convert_input,
    "Python": generic_convert_input,
    "Data": generic_convert_input,
    "Scale": convert_scale,
    "Power": convert_power,
    "Softmax": convert_softmax,
    "Reshape": convert_reshape,
    "Flatten": convert_flatten,
    "ArgMax": convert_argmax,
    "Pooling": convert_pooling,
    "PReLU": convert_prelu,
    "ELU": convert_elu,
    "ReLU": convert_relu,
    "Concat": convert_concat,
    "Eltwise": convert_eltwise,
    "Dropout": convert_dropout,
    "InnerProduct": convert_inner_product,
    "LRN": convert_lrn,
    "Crop": convert_crop
}  # type: Dict[str, Callable[(LayerParameter, CaffeOp, Converter), None]]
