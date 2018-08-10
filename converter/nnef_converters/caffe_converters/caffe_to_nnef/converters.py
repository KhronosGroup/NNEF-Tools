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
from collections import OrderedDict
from functools import partial

import numpy as np

from ..common import CaffeGraph, CaffeOp, CaffeDN, caffe_factory
from ..common import EXTRA_WEIGHTS
from ...common import dog
from ...common.converter_base import ConverterBase
from ...common.nnef_dog_types import NnefOp, NnefDN, nnef_factory
from ...common.types import *


class Converter(ConverterBase):
    def __init__(self, caffedog, custom_converters=None):
        # type: (CaffeGraph, Optional[Dict[str, Callable[(CaffeOp, Converter), None]]])->None

        converters = dict(DefaultConverters)
        if custom_converters:
            converters.update(custom_converters)

        super(Converter, self).__init__(sourcedog=caffedog,
                                        source_factory=caffe_factory,
                                        target_factory=nnef_factory,
                                        converters=converters)

    def make_variable(self, caffeop, discriminator, value):
        # type: (CaffeOp, str, np.ndarray)->NnefDN

        nnefop = NnefOp("variable")
        nnefop.add_arg("shape", list(value.shape))
        nnefop.add_arg("label", "{}/{}".format(caffeop.args["name"], discriminator))
        nnefop.add_result("output", self.make_targetdn(name=caffeop.args["name"], discriminator=discriminator))
        self.add_targetop(nnefop)
        nnefop.extra[EXTRA_WEIGHTS] = value
        return nnefop.result

    @staticmethod
    def nnef_padding(pad):
        # type: (List[int])->List[Tuple[int, int]]
        return list(zip(pad, pad))

    @staticmethod
    def nnef_axis(axis, rank):
        # type: (int, int)->int
        return axis if axis >= 0 else rank + axis

    @staticmethod
    def get_pooling_right_padding(h, k, p, q, s):
        # type: (int, int, int, int, int)->int
        a = int(math.ceil(float(h + p + q - k)/s))
        return s * a + k - h - p


def generic_convert_input(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    if len(caffeop.args["shapes"]) != 1:
        converter.print_error(caffeop, "Input operation only supported with 1 result")

    converter.add_targetop_ex(caffeop, "external",
                              OrderedDict([
                                  ("shape", caffeop.args["shapes"][0])
                              ]),
                              OrderedDict([
                                  ("output", caffeop.result)
                              ]))


def convert_scaled_batch_norm(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    weights = caffeop.extra[EXTRA_WEIGHTS]

    if weights["scale_factor"].shape != (1,):
        converter.print_error(caffeop, "scale_factor.shape must be [1]")

    scale_factor = weights["scale_factor"][0]
    norm = 0.0 if scale_factor == 0.0 else 1.0 / scale_factor

    converter.add_targetop_ex(caffeop, "batch_normalization",
                              OrderedDict([
                                  ("input", caffeop.args[dog.gen_arg_name(0)]),
                                  ("mean", converter.make_variable(caffeop, "mean",
                                                                   np.expand_dims(weights["mean"] * norm, 0))),
                                  ("variance", converter.make_variable(caffeop, "variance",
                                                                       np.expand_dims(weights["variance"] * norm, 0))),
                                  ("offset", converter.make_variable(caffeop, "offset",
                                                                     np.expand_dims(weights["bias"], 0))),
                                  ("scale", converter.make_variable(caffeop, "scale",
                                                                    np.expand_dims(weights["weight"], 0))),
                                  ("epsilon", caffeop.args["eps"]),
                              ]),
                              OrderedDict([
                                  ("output", caffeop.result)
                              ]))


def convert_scale(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    weights = caffeop.extra[EXTRA_WEIGHTS]

    mul = NnefOp("mul")
    mul.add_arg("x", converter.get_targetdn(caffeop.args[dog.gen_arg_name(0)]))
    mul.add_arg("y", converter.make_variable(caffeop, "weight", np.expand_dims(weights["weight"], 0)))
    if caffeop.args["bias_term"]:
        mul.add_result("z", converter.make_targetdn(name=caffeop.args["name"], discriminator="scale"))
    else:
        mul.add_result("z", converter.make_targetdn(sourcedn=caffeop.result_node))
    converter.add_targetop(mul, caffeop)

    if caffeop.args["bias_term"]:
        add = NnefOp("add")
        add.add_arg("x", mul.result_node)
        add.add_arg("y", converter.make_variable(caffeop, "bias", np.expand_dims(weights["bias"], 0)))
        add.add_result("z", converter.make_targetdn(sourcedn=caffeop.result_node))
        converter.add_targetop(add, caffeop)


def generic_convert_unary(caffeop, converter, target_name):
    # type: (CaffeOp, Converter)->None

    converter.add_targetop_ex(caffeop, target_name,
                              OrderedDict([
                                  ("x", caffeop.args[dog.gen_arg_name(0)]),
                              ]),
                              OrderedDict([
                                  ("y", caffeop.result)
                              ]))


def convert_power(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    input_ = converter.get_targetdn(caffeop.args[dog.gen_arg_name(0)])  # type: NnefDN

    ops = []
    if caffeop.args["scale"] != 1:
        mul = NnefOp("mul")
        mul.add_arg("x", input_)
        mul.add_arg("y", caffeop.args["scale"])
        output_ = (converter.make_targetdn(sourcedn=caffeop.result)
                   if caffeop.args["shift"] == 0 and caffeop.args["power"] == 1
                   else converter.make_targetdn(name=caffeop.args["name"], discriminator="scale"))
        mul.add_result("z", output_)
        ops.append(mul)
        input_ = output_
    if caffeop.args["shift"] != 0:
        add = NnefOp("add")
        add.add_arg("x", input_)
        add.add_arg("y", caffeop.args["shift"])
        output_ = (converter.make_targetdn(sourcedn=caffeop.result)
                   if caffeop.args["power"] == 1
                   else converter.make_targetdn(name=caffeop.args["name"], discriminator="shift"))
        add.add_result("z", output_)
        ops.append(add)
        input_ = output_
    if caffeop.args["power"] != 1 or not ops:
        pow_ = NnefOp("pow")
        pow_.add_arg("x", input_)
        pow_.add_arg("y", caffeop.args["power"])
        pow_.add_result("z", converter.make_targetdn(sourcedn=caffeop.result))
        ops.append(pow_)

    for op in ops:
        converter.add_targetop(op, caffeop)


def convert_relu(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    if caffeop.args["negative_slope"] == 0:
        converter.add_targetop_ex(caffeop, "relu",
                                  OrderedDict([
                                      ("x", caffeop.args[dog.gen_arg_name(0)]),
                                  ]),
                                  OrderedDict([
                                      ("y", caffeop.result)
                                  ]))
    else:
        converter.add_targetop_ex(caffeop, "leaky_relu",
                                  OrderedDict([
                                      ("x", caffeop.args[dog.gen_arg_name(0)]),
                                      ("alpha", caffeop.args["negative_slope"]),
                                  ]),
                                  OrderedDict([
                                      ("y", caffeop.result)
                                  ]))


def convert_elu(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    if caffeop.args["alpha"] != 1:
        converter.print_error(caffeop, "only alpha=1 is supported")

    converter.add_targetop_ex(caffeop, "elu",
                              OrderedDict([
                                  ("x", caffeop.args[dog.gen_arg_name(0)]),
                              ]),
                              OrderedDict([
                                  ("y", caffeop.result)
                              ]))


def generic_convert_convolution(caffeop, converter, target_name):
    # type: (CaffeOp, Converter)->None

    factor = caffeop.args["stride"][0]
    if (caffeop.name == "Deconvolution"
            and caffeop.args["weight_filler_type"] == "bilinear"
            and caffeop.args["num_output"] == caffeop.args[dog.gen_arg_name(0)].shape[1]
            and not caffeop.args["bias_term"]
            and caffeop.args["kernel_size"] == 2 * [2 * factor - factor % 2]
            and caffeop.args["stride"] == 2 * [factor]
            and caffeop.args["pad"] == 2 * [int(math.ceil((factor - 1) / 2.0))]):
        converter.add_targetop_ex(caffeop, "multilinear_upsample",
                                  OrderedDict([
                                      ("input", caffeop.args[dog.gen_arg_name(0)]),
                                      ("factor", 2 * [factor]),
                                      ("method", "symmetric"),
                                      ("border", "constant")
                                  ]),
                                  OrderedDict([
                                      ("output", caffeop.result)
                                  ]))
    else:
        weights = caffeop.extra[EXTRA_WEIGHTS]
        converter.add_targetop_ex(caffeop, target_name,
                                  OrderedDict([
                                      ("input", caffeop.args[dog.gen_arg_name(0)]),
                                      ("filter", converter.make_variable(caffeop, "filter", weights["weight"])),
                                      ("bias",
                                       (converter.make_variable(caffeop, "bias", np.expand_dims(weights["bias"], 0))
                                        if caffeop.args["bias_term"]
                                        else 0.0)),
                                      ("border", "constant"),
                                      ("padding", converter.nnef_padding(caffeop.args["pad"])),
                                      ("stride", caffeop.args["stride"]),
                                      ("dilation", caffeop.args["dilation"]),
                                      ("groups", caffeop.args["group"]),
                                  ]),
                                  OrderedDict([
                                      ("output", caffeop.result)
                                  ]))


def convert_pooling(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    rank = len(caffeop.args[dog.gen_arg_name(0)].shape)

    pool_name_by_pool = {
        0: "max_pool",
        1: "avg_pool"
    }

    reduce_name_by_pool = {
        0: "max_reduce",
        1: "mean_reduce"
    }

    if caffeop.args["pool"] not in reduce_name_by_pool:
        converter.print_error(caffeop, "unsupported pool method {}".format(caffeop.args["pool"]))
        caffeop.set_arg("pool", 0)

    if caffeop.args["global_pooling"]:
        converter.add_targetop_ex(caffeop, reduce_name_by_pool[caffeop.args["pool"]],
                                  OrderedDict([
                                      ("input", caffeop.args[dog.gen_arg_name(0)]),
                                      ("axes", list(range(2, rank)))
                                  ]),
                                  OrderedDict([
                                      ("output", caffeop.result)
                                  ]))
    else:
        input_size = caffeop.args[dog.gen_arg_name(0)].shape
        padding = converter.nnef_padding([0, 0] + caffeop.args["pad"])
        stride = [1, 1] + caffeop.args["stride"]
        kernel_size = [1, 1] + caffeop.args["kernel_size"]

        # compensate for caffe's pooling output size calculation
        # https://github.com/BVLC/caffe/issues/1318#issuecomment-59594323
        old_padding = padding
        padding = [(p, converter.get_pooling_right_padding(h, k, p, q, s))
                   for h, k, (p, q), s in zip(input_size, kernel_size, old_padding, stride)]

        converter.add_targetop_ex(caffeop, pool_name_by_pool[caffeop.args["pool"]],
                                  OrderedDict([
                                      ("input", caffeop.args[dog.gen_arg_name(0)]),
                                      ("size", kernel_size),
                                      ("border", "constant"),
                                      ("padding", padding),
                                      ("stride", stride)
                                  ]),
                                  OrderedDict([
                                      ("output", caffeop.result)
                                  ]))


def convert_softmax(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    rank = len(caffeop.args[dog.gen_arg_name(0)].shape)

    converter.add_targetop_ex(caffeop, "softmax",
                              OrderedDict([
                                  ("x", caffeop.args[dog.gen_arg_name(0)]),
                                  ("axes", [converter.nnef_axis(caffeop.args["axis"], rank)])
                              ]),
                              OrderedDict([
                                  ("y", caffeop.result)
                              ]))


def convert_inner_product(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    weights = caffeop.extra[EXTRA_WEIGHTS]

    input_ = caffeop.args[dog.gen_arg_name(0)]
    rank = len(input_.shape)

    axis = converter.nnef_axis(caffeop.args["axis"], rank)
    if axis != rank - 1:
        reshape = converter.add_targetop_ex(caffeop, "reshape",
                                            OrderedDict([
                                                ("input", input_),
                                                ("shape", input_.shape[:axis] + [-1])
                                            ]),
                                            OrderedDict([
                                                ("output", converter.make_targetdn(name=caffeop.args["name"],
                                                                                   discriminator="flatten"))
                                            ]))
        input_ = reshape.result_node

    matmul = converter.add_targetop_ex(caffeop, "matmul",
                                       OrderedDict([
                                           ("A", input_),
                                           ("B",
                                            converter.make_variable(caffeop,
                                                                    discriminator="weight",
                                                                    value=weights["weight"])),
                                           ("transposeA", False),
                                           ("transposeB", not caffeop.args["transpose"])
                                       ]),
                                       OrderedDict([
                                           ("C",
                                            (converter.make_targetdn(name=caffeop.args["name"], discriminator="matmul")
                                             if caffeop.args["bias_term"]
                                             else caffeop.result))
                                       ]))

    input_ = matmul.result_node
    if caffeop.args["bias_term"]:
        converter.add_targetop_ex(caffeop, "add",
                                  OrderedDict([
                                      ("x", input_),
                                      ("y", converter.make_variable(caffeop,
                                                                    discriminator="bias",
                                                                    value=np.expand_dims(weights["bias"], 0)))
                                  ]),
                                  OrderedDict([
                                      ("z", caffeop.result)
                                  ]))


def convert_reshape(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    input_ = caffeop.args[dog.gen_arg_name(0)]
    shape = input_.shape
    rank = len(shape)

    axis = converter.nnef_axis(caffeop.args["axis"], rank)
    num_axes = rank if caffeop.args["num_axes"] == -1 else caffeop.args["num_axes"]
    new_shape = shape[:axis] + caffeop.args["shape"] + shape[axis + num_axes:]

    converter.add_targetop_ex(caffeop, "reshape",
                              OrderedDict([
                                  ("input", input_),
                                  ("shape", new_shape)
                              ]),
                              OrderedDict([
                                  ("output", caffeop.result)
                              ]))


def convert_flatten(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    input_ = caffeop.args[dog.gen_arg_name(0)]
    shape = input_.shape
    rank = len(shape)

    axis = converter.nnef_axis(caffeop.args["axis"], rank)
    end_axis = converter.nnef_axis(caffeop.args["end_axis"], rank)
    new_shape = shape[:axis] + [-1] + shape[end_axis + 1:]

    converter.add_targetop_ex(caffeop, "reshape",
                              OrderedDict([
                                  ("input", input_),
                                  ("shape", new_shape)
                              ]),
                              OrderedDict([
                                  ("output", caffeop.result)
                              ]))


def convert_eltwise(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    target_name_by_operation = {
        0: "mul",
        1: "add",
        2: "max"
    }

    if caffeop.num_gen_args() != 2:
        converter.print_error(caffeop, "Eltwise only supported with 2 inputs")

    converter.add_targetop_ex(caffeop, target_name_by_operation[caffeop.args["operation"]],
                              OrderedDict([
                                  ("x", caffeop.args[dog.gen_arg_name(0)]),
                                  ("y", caffeop.args[dog.gen_arg_name(1)])
                              ]),
                              OrderedDict([
                                  ("z", caffeop.result_node),
                              ]))


def convert_lrn(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    rank = len(caffeop.args[dog.gen_arg_name(0)].shape)

    if caffeop.args["norm_region"] != 0:
        converter.print_error(caffeop, "Only ACROSS_CHANNELS is supported")

    size = [caffeop.args["local_size"] if i == 1 else 1 for i in range(rank)]

    converter.add_targetop_ex(caffeop, "local_response_normalization",
                              OrderedDict([
                                  ("input", caffeop.args[dog.gen_arg_name(0)]),
                                  ("size", size),
                                  ("alpha", caffeop.args["alpha"]),
                                  ("beta", caffeop.args["beta"]),
                              ]),
                              OrderedDict([
                                  ("output", caffeop.result_node),
                              ]))


def convert_concat(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    rank = len(caffeop.args[dog.gen_arg_name(0)].shape)

    converter.add_targetop_ex(caffeop, "concat",
                              OrderedDict([
                                  ("values", caffeop.get_arg_nodes()),
                                  ("axis", converter.nnef_axis(caffeop.args["axis"], rank)),
                              ]),
                              OrderedDict([
                                  ("value", caffeop.result_node),
                              ]))


def convert_argmax(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    rank = len(caffeop.args[dog.gen_arg_name(0)].shape)

    if caffeop.args["top_k"] != 1 or caffeop.args["out_max_val"]:
        converter.print_error(caffeop, "Argmax params not supported yet")

    if caffeop.args["axis"] is None:
        axes = list(range(1, rank))
    else:
        axes = [converter.nnef_axis(caffeop.args["axis"], rank)]

    converter.add_targetop_ex(caffeop, "argmax_reduce",
                              OrderedDict([
                                  ("input", caffeop.args[dog.gen_arg_name(0)]),
                                  ("axes", axes),
                              ]),
                              OrderedDict([
                                  ("output", caffeop.result_node),
                              ]))


def convert_crop(caffeop, converter):
    # type: (CaffeOp, Converter)->None

    caffedn_input = caffeop.args[dog.gen_arg_name(0)]  # type: CaffeDN
    caffedn_reference = caffeop.args[dog.gen_arg_name(1)]  # type: CaffeDN

    input_shape = caffedn_input.shape
    reference_shape = caffedn_reference.shape

    axis = caffeop.args["axis"]
    offset = caffeop.args["offset"]
    if len(offset) == 1:
        offset = (len(input_shape) - axis) * offset

    axes = list(range(axis, len(input_shape)))
    begin = list(offset)
    end = [o + s for o, s in zip(offset, reference_shape[axis:])]

    converter.add_targetop_ex(caffeop, "slice",
                              OrderedDict([
                                  ("input", caffeop.args[dog.gen_arg_name(0)]),
                                  ("axes", axes),
                                  ("begin", begin),
                                  ("end", end),
                              ]),
                              OrderedDict([
                                  ("output", caffeop.result_node),
                              ]))


DefaultConverters = {
    "BNLL": partial(generic_convert_unary, target_name="softplus"),
    "TanH": partial(generic_convert_unary, target_name="tanh"),
    "AbsVal": partial(generic_convert_unary, target_name="abs"),
    "Sigmoid": partial(generic_convert_unary, target_name="sigmoid"),
    "Convolution": partial(generic_convert_convolution, target_name="conv"),
    "Deconvolution": partial(generic_convert_convolution, target_name="deconv"),
    "Input": generic_convert_input,
    "Python": generic_convert_input,
    "Data": generic_convert_input,
    "_ScaledBatchNorm": convert_scaled_batch_norm,
    "Scale": convert_scale,
    "Power": convert_power,
    "Softmax": convert_softmax,
    "Reshape": convert_reshape,
    "Flatten": convert_flatten,
    "ArgMax": convert_argmax,
    "Pooling": convert_pooling,
    "ELU": convert_elu,
    "ReLU": convert_relu,
    "Concat": convert_concat,
    "Eltwise": convert_eltwise,
    "InnerProduct": convert_inner_product,
    "LRN": convert_lrn,
    "Crop": convert_crop
}  # type: Dict[str, Callable[(CaffeOp, Converter), None]]
