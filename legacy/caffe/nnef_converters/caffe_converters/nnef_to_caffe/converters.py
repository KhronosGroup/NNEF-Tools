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

from .dog_to_caffe import EXTRA_CAFFE_PARAM_NAME
from ..common import CaffeOp, CaffeDN
from ..common import EXTRA_VARIABLE_LABELS, VARIABLE_LABEL_SKIP
from ...common import dog
from ...common import utils
from ...common.nnef_dog_types import NnefOp, NnefDN, NnefDNLike
from ...common.types import *

if has_typing:
    LogAnchor = Union[dog.OperationNode, dog.DataNode, str]
    ShapeLike = List[Union[int, Tuple[int, int]]]


class Converter(object):
    ordered_dict_maker = utils.ordered_dict_maker

    def __init__(self, nnefdog):
        # type: (dog.Graph)->None

        self.nnefdog = nnefdog  # type: dog.Graph
        self.used_names = set()  # type: Set[str]
        self.caffedn_by_source_name = OrderedDict()  # type: OrderedDict[str, CaffeDN]
        self.nnefdn_by_caffedn = OrderedDict()  # type: OrderedDict[CaffeDN, Optional[NnefDN]]
        self.nnefop_by_caffeop = OrderedDict()  # type: OrderedDict[CaffeOp, Optional[NnefOp]]

    def get_new_name(self, name, discriminator=None):
        # type: (str, Optional[str]) -> str

        if discriminator:
            name = "{}_{}".format(name, discriminator)

        name_candidate = name
        counter = 1
        while name_candidate in self.used_names:
            name_candidate = "{}_{}".format(name, counter)
            counter += 1

        self.used_names.add(name_candidate)
        return name_candidate

    def make_caffedn(self, nnefdn=None, name=None, discriminator=None):
        # type: (Optional[NnefDN], Optional[str], Optional[str]) -> CaffeDN

        assert (nnefdn is None) != (name is None)

        if name is None:
            name = nnefdn.name
        name = self.get_new_name(name, discriminator)

        caffedn = CaffeDN(name)
        if nnefdn:
            caffedn.source_name = nnefdn.name
            self.caffedn_by_source_name[nnefdn.name] = caffedn

        self.nnefdn_by_caffedn[caffedn] = nnefdn

        return caffedn

    def get_caffedn(self, nnefdnlike):
        # type: (NnefDNLike)->CaffeDNLike

        if not isinstance(nnefdnlike, NnefDN):
            return nnefdnlike

        if nnefdnlike.name not in self.caffedn_by_source_name:
            self.print_error(nnefdnlike, "Corresponding caffedn not defined")
            return self.make_caffedn(nnefdn=nnefdnlike)

        return self.caffedn_by_source_name[nnefdnlike.name]

    def add_caffeop(self, caffeop, nnefop=None):
        # type: (CaffeOp, Optional[NnefOp])->None

        assert caffeop not in self.nnefop_by_caffeop

        self.nnefop_by_caffeop[caffeop] = nnefop

    def add_caffeop_ex(
            self,
            nnefop,  # type: NnefOp
            op_name_caffe,  # type: str
            tensor_args,  # type: List[NnefDNLike]
            args=None,  # type: Optional[OrderedDict[str, Any]]
            results=None,  # type: Optional[OrderedDict[str, Any]]
            extra=None  # type: Dict[str, Any]
    ):
        # type: (...)->None

        if args is None:
            args = OrderedDict()
        if results is None:
            results = OrderedDict([(dog.gen_result_name(i), r) for i, r in enumerate(nnefop.results.values())])
        assert all(isinstance(r, NnefDN) for r in results.values())

        caffeop = CaffeOp(op_name_caffe)

        def input_value(val):
            # type: (Any)->Any
            if isinstance(val, NnefDN):
                val = self.get_caffedn(val)
            return val

        def output_value(val):
            # type: (NnefDN)->CaffeDN
            return self.make_caffedn(nnefdn=val)

        for i, tensor_arg in enumerate(tensor_args):
            caffeop.add_arg(dog.gen_arg_name(i), input_value(tensor_arg))

        for k, v in args.items():
            caffeop.add_arg(k, utils.recursive_transform(v, input_value))

        for k, v in results.items():
            caffeop.add_result(k, utils.recursive_transform(v, output_value))

        if extra:
            caffeop.extra.update(extra)

        self.add_caffeop(caffeop=caffeop, nnefop=nnefop)

    @staticmethod
    def get_shape_safe(dnlike):
        # type: (dog.DataNodeLike)->List[int]

        return dog.get_shape_safe(dnlike)

    @staticmethod
    def get_rank_safe(dnlike):
        # type: (dog.DataNodeLike)->int

        return dog.get_rank_safe(dnlike)

    @staticmethod
    def nnef_batch_size(shape):
        # type: (List[int], int)->int
        return shape[0]

    @staticmethod
    def nnef_channels(shape):
        # type: (List[int], int)->int
        return shape[1]

    @staticmethod
    def nnef_spatial_part(shape):
        # type: (List[Any])->List[Any]
        if not shape:
            return shape

        return shape[2:]

    @staticmethod
    def nnef_nonspatial_part(shape):
        # type: (List[Any])->List[Any]
        if not shape:
            return shape

        return shape[:2]

    @staticmethod
    def nnef_filter_downscaled_channels(filter_shape):
        # type: (List[int])->int
        return filter_shape[0]

    @staticmethod
    def nnef_filter_upscaled_channels(filter_shape, groups):
        # type: (List[int], int)->int
        return filter_shape[1] * groups

    @staticmethod
    def caffe_stride(nnef_stride):
        # type: (List[int])->Optional[List[int]]
        return nnef_stride if nnef_stride and utils.has_not_equal_1(nnef_stride) else None

    @staticmethod
    def caffe_dilation(nnef_dilation):
        # type: (List[int])->Optional[List[int]]
        return nnef_dilation if nnef_dilation and utils.has_not_equal_1(nnef_dilation) else None

    @staticmethod
    def caffe_pad_stride_dilation(
        spatial_upscaled_size,  # type: List[int]
        spatial_downscaled_size,  # type: List[int]
        spatial_filter_size,  # type: List[int]
        spatial_padding,  # type: List[Tuple[int, int]]
        spatial_stride,  # type: List[int]
        spatial_dilation,  # type: List[int]
        is_pool,  # type: bool
        log_anchor  # type: LogAnchor
    ):
        # type: (...)->Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]

        if not spatial_stride:
            spatial_stride = [1] * len(spatial_upscaled_size)
        if not spatial_dilation:
            spatial_dilation = [1] * len(spatial_upscaled_size)

        if not spatial_padding:
            spatial_padding = utils.nnef_auto_padding(
                spatial_upscaled_size,
                spatial_downscaled_size,
                spatial_filter_size,
                spatial_stride,
                spatial_dilation
            )

        if is_pool:
            nnef_output_shape = []
            caffe_output_shape = []

            for i, (h, k, (p, q), s) in enumerate(zip(spatial_upscaled_size,
                                                      spatial_filter_size,
                                                      spatial_padding,
                                                      spatial_stride)):
                nnef_output_size = Converter.get_nnef_pool_output_size(h, k, p, q, s)
                caffe_output_size = Converter.get_caffe_pool_output_size(h, k, p, q, s)
                nnef_output_shape.append(nnef_output_size)
                caffe_output_shape.append(caffe_output_size)

            if nnef_output_shape != caffe_output_shape:
                Converter.print_error(log_anchor,
                                      "Output shape of pooling is not the same for NNEF and Caffe: {} -> {}.\n"
                                      "This case is not yet supported.".format(nnef_output_shape, caffe_output_shape))

            # compensate for caffe's pooling output size calculation
            # https://github.com/BVLC/caffe/issues/1318#issuecomment-59594323
            spatial_padding_old = list(spatial_padding)
            for i, (h, k, (p, q), s) in enumerate(zip(spatial_upscaled_size,
                                                      spatial_filter_size,
                                                      spatial_padding_old,
                                                      spatial_stride)):
                asymmetric_size = Converter.get_caffe_pool_output_size(h, k, p, q, s)
                symmetric_size = Converter.get_caffe_pool_output_size(h, k, p, p, s)

                if asymmetric_size == symmetric_size:
                    spatial_padding[i] = (p, p)

        if any(left != right for left, right in spatial_padding):
            Converter.print_error(log_anchor, "Asymmetric pad is not supported, padding={}".format(spatial_padding))

        caffe_pad = [max(left, right) for left, right in spatial_padding]

        return (caffe_pad if utils.has_not_equal_0(caffe_pad) else None,
                spatial_stride if utils.has_not_equal_1(spatial_stride) else None,
                spatial_dilation if utils.has_not_equal_1(spatial_dilation) else None)

    @staticmethod
    def is_scalar(nnefdnlike):
        # type: (NnefDNLike)->bool

        return isinstance(nnefdnlike, (bool, int, float))

    @staticmethod
    def is_variable(nnefdnlike):
        # type: (NnefDNLike)->bool

        return (isinstance(nnefdnlike, NnefDN)
                and nnefdnlike.producer.name == "variable")

    @staticmethod
    def get_label_safe(nnefdnlike):
        # type: (NnefDNLike)->Optional[str]
        return nnefdnlike.producer.args["label"] if Converter.is_variable(nnefdnlike) else None

    @staticmethod
    def is_unique_variable(nnefdnlike):
        # type: (NnefDNLike)->bool

        return (isinstance(nnefdnlike, NnefDN)
                and nnefdnlike.producer.name == "variable"
                and len(nnefdnlike.consumers) == 1)

    @staticmethod
    def is_nonvar_nonconst_dn(nnefdnlike):
        # type: (NnefDNLike)->bool

        return isinstance(nnefdnlike, NnefDN) and nnefdnlike.producer.name not in ["constant", "variable"]

    @staticmethod
    def assert_unique_variable(nnefop, arg_name):
        # type: (NnefOp, str)->None

        nnefdnlike = nnefop.args[arg_name]
        if not isinstance(nnefdnlike, NnefDN) or nnefdnlike.producer.name != "variable":
            Converter.print_error(nnefop, "Arg {} is not a variable".format(arg_name))

        if isinstance(nnefdnlike, NnefDN) and len(nnefdnlike.consumers) > 1:
            Converter.print_error(nnefop, "Arg {} is a variable used by multiple operations".format(arg_name))

    @staticmethod
    def assert_unique_variable_or_zero(nnefop, arg_name):
        # type: (NnefOp, str)->None

        nnefdnlike = nnefop.args[arg_name]
        if isinstance(nnefdnlike, (int, float)):
            if nnefdnlike != 0:
                Converter.print_error(nnefop, "Arg {} is neither zero nor variable".format(arg_name))
        else:
            Converter.assert_unique_variable(nnefop, arg_name)

    @staticmethod
    def assert_scalar_value(nnefop, arg_name, expected_value):
        # type: (NnefOp, str, Union[bool, int, float])->None

        nnefdnlike = nnefop.args[arg_name]  # type: NnefDNLike
        if not isinstance(nnefdnlike, (bool, int, float)):
            Converter.print_error(nnefop, "Arg {} has bad type".format(arg_name))
        elif nnefdnlike != expected_value:
            Converter.print_error(nnefop, "Arg {} != {}".format(arg_name, expected_value))

    @staticmethod
    def assert_nonspatial_items(nnefop, arg_name, expected_item_value):
        # type: (NnefOp, str, Any)->None

        shapelike = nnefop.args[arg_name]  # type: ShapeLike
        if not all(size == expected_item_value for size in Converter.nnef_nonspatial_part(shapelike)):
            Converter.print_error(nnefop, "The non-spatial part of arg {} contains an item which is not {}"
                                  .format(arg_name, expected_item_value))

    @staticmethod
    def assert_all_items(nnefop, arg_name, expected_item_value):
        # type: (NnefOp, str, Any)->None

        shapelike = nnefop.args[arg_name]  # type: ShapeLike
        if not all(size == expected_item_value for size in shapelike):
            Converter.print_error(nnefop, "Arg {} contains an item which is not {}"
                                  .format(arg_name, expected_item_value))

    @staticmethod
    def default_to_none(value, default):
        # type: (Any, Any)->Any
        return None if value == default else value

    @staticmethod
    def print_error(anchor, message):
        # type: (LogAnchor, str)->None

        if isinstance(anchor, dog.OperationNode):
            utils.print_error("{}={}(): {}".format(
                ", ".join([dn.name for dn in anchor.get_result_nodes()]),
                anchor.name,
                message
            ))
        elif isinstance(anchor, dog.DataNode):
            utils.print_error("{}: {}".format(anchor.name, message))
        else:
            utils.print_error("{}: {}".format(anchor, message))

    @staticmethod
    def get_nnef_pool_output_size(h, k, p, q, s):
        return int(math.floor(float(h + p + q - k) / s)) + 1

    @staticmethod
    def get_caffe_pool_output_size(h, k, p, q, s):
        return int(math.ceil(float(h + p + q - k) / s)) + 1


# noinspection PyUnusedLocal
def convert_ignore(nnefop, converter):
    # type: (NnefOp, Converter)->None
    pass


def convert_external(nnefop, converter):
    # type: (NnefOp, Converter)->None

    converter.add_caffeop_ex(nnefop, "Input", [],
                             converter.ordered_dict_maker[
                             "shape": nnefop.args["shape"]
                             ])


def generic_convert_conv(nnefop, converter, target_name, is_upscale=False):
    # type: (NnefOp, Converter, str, bool)->None

    converter.assert_unique_variable(nnefop, "filter")
    converter.assert_unique_variable_or_zero(nnefop, "bias")

    filter_shape = converter.get_shape_safe(nnefop.args["filter"])

    if target_name == "Deconvolution":
        spatial_upscaled_size = converter.nnef_spatial_part(converter.get_shape_safe(nnefop.result))
        spatial_downscaled_size = converter.nnef_spatial_part(converter.get_shape_safe(nnefop.args["input"]))
    else:
        spatial_upscaled_size = converter.nnef_spatial_part(converter.get_shape_safe(nnefop.args["input"]))
        spatial_downscaled_size = converter.nnef_spatial_part(converter.get_shape_safe(nnefop.result))

    pad, stride, dilation = converter.caffe_pad_stride_dilation(
        spatial_upscaled_size=spatial_upscaled_size,
        spatial_downscaled_size=spatial_downscaled_size,
        spatial_filter_size=converter.nnef_spatial_part(converter.get_shape_safe(nnefop.args["filter"])),
        spatial_padding=nnefop.args["padding"],
        spatial_stride=nnefop.args["stride"],
        spatial_dilation=nnefop.args["dilation"],
        is_pool=False,
        log_anchor=nnefop
    )

    if is_upscale:
        num_output = converter.nnef_filter_upscaled_channels(filter_shape, nnefop.args["groups"])
    else:
        num_output = converter.nnef_filter_downscaled_channels(filter_shape)

    converter.add_caffeop_ex(nnefop, target_name,
                             [nnefop.args["input"]],
                             converter.ordered_dict_maker[
                                 "kernel_size": converter.nnef_spatial_part(filter_shape),
                                 "num_output": num_output,
                                 "bias_term": converter.default_to_none(bool(nnefop.args["bias"]), True),
                                 "pad": pad,
                                 "stride": stride,
                                 "dilation": dilation,
                                 "group": nnefop.args["groups"],
                                 "engine": "CAFFE" if nnefop.args["groups"] != 1 else None
                             ],
                             extra={
                                 EXTRA_CAFFE_PARAM_NAME: "convolution_param",
                                 EXTRA_VARIABLE_LABELS: [
                                     converter.get_label_safe(nnefop.args["filter"]),
                                     converter.get_label_safe(nnefop.args["bias"])
                                 ]
                             })


def generic_convert_unary(nnefop, converter, target_name):
    # type: (NnefOp, Converter, str)->None

    converter.add_caffeop_ex(nnefop, target_name,
                             [nnefop.args["x"]])


def convert_relu(nnefop, converter):
    # type: (NnefOp, Converter)->None

    converter.add_caffeop_ex(nnefop, "ReLU",
                             [nnefop.args["x"]])


def convert_leaky_relu(nnefop, converter):
    # type: (NnefOp, Converter)->None

    converter.add_caffeop_ex(nnefop, "ReLU",
                             [nnefop.args["x"]],
                             converter.ordered_dict_maker[
                                 "negative_slope": nnefop.args["alpha"]
                             ])


def convert_softmax(nnefop, converter):
    # type: (NnefOp, Converter)->None

    if len(nnefop.args["axes"]) != 1:
        converter.print_error(nnefop, "len(axes) != 1")

    converter.add_caffeop_ex(nnefop, "Softmax",
                             [nnefop.args["x"]],
                             converter.ordered_dict_maker[
                                 "axis": converter.default_to_none(nnefop.args["axes"][0], 1)
                             ])


def convert_prelu(nnefop, converter):
    # type: (NnefOp, Converter)->None

    converter.assert_unique_variable(nnefop, "alpha")
    converter.add_caffeop_ex(nnefop, "PReLU",
                             [nnefop.args["x"]],
                             converter.ordered_dict_maker[
                                "channel_shared": (len(converter.get_shape_safe(nnefop.args["alpha"])) == 0)
                             ],
                             extra={
                                 EXTRA_VARIABLE_LABELS: [
                                     converter.get_label_safe(nnefop.args["alpha"]),
                                 ]
                             })


def generic_convert_binary(
        nnefop,  # type: NnefOp
        converter,  # type: Converter
        is_commutative=False,  # type: bool
        eltwise_operation=None,  # type: Optional[int]
        eltwise_coeff=None,  # type: Optional[List[float]]
        learnable_name=None,  # type: Optional[str]
        scalar_name=None,  # type: Optional[str]
        scalar_arg_name=None  # type: Optional[str]
):
    # type: (...)->None

    def make_learnable(input_, variable):
        # type: (NnefDN, NnefDN)->None

        input_shape = dog.get_shape_safe(input_)
        variable_shape = dog.get_shape_safe(variable)

        axis = -1
        for j, (i, v) in enumerate(zip(input_shape, variable_shape)):
            if i == v and v != 1:
                axis = j
                break

        if axis == -1:
            if not utils.has_greater_than_1(variable_shape):
                if len(variable_shape) >= 2 and len(input_shape) >= 2:
                    axis = 1
                else:
                    axis = 0
            else:
                converter.print_error(nnefop, "Variable and input shapes are incompatible")

        num_axes = dog.get_rank_safe(variable) - axis
        converter.add_caffeop_ex(nnefop, learnable_name,
                                 [input_],
                                 converter.ordered_dict_maker[
                                     "axis": converter.default_to_none(axis, 1),
                                     "num_axes": converter.default_to_none(num_axes, 1)
                                 ],
                                 extra={
                                     EXTRA_VARIABLE_LABELS: [
                                         converter.get_label_safe(variable)
                                     ]
                                 })

    def make_eltwise(input1, input2):
        # type: (NnefDN, NnefDN)->None
        converter.add_caffeop_ex(nnefop, "Eltwise",
                                 [input1, input2],
                                 converter.ordered_dict_maker[
                                     "operation": eltwise_operation,
                                     "coeff": eltwise_coeff
                                 ])

    def make_scalar(input_, scalar):
        # type: (NnefDN, Union[bool, int, float])->None
        converter.add_caffeop_ex(nnefop, scalar_name,
                                 [input_],
                                 converter.ordered_dict_maker[
                                     scalar_arg_name: scalar
                                 ])

    if (learnable_name
            and converter.is_nonvar_nonconst_dn(nnefop.args["x"])
            and converter.is_unique_variable(nnefop.args["y"])):
        make_learnable(input_=nnefop.args["x"], variable=nnefop.args["y"])
    elif (learnable_name
          and is_commutative
          and converter.is_nonvar_nonconst_dn(nnefop.args["y"])
          and converter.is_unique_variable(nnefop.args["x"])):
        make_learnable(input_=nnefop.args["y"], variable=nnefop.args["x"])
    elif (scalar_name
          and converter.is_nonvar_nonconst_dn(nnefop.args["x"])
          and converter.is_scalar(nnefop.args["y"])):
        make_scalar(input_=nnefop.args["x"], scalar=nnefop.args["y"])
    elif (scalar_name
          and is_commutative
          and converter.is_nonvar_nonconst_dn(nnefop.args["y"])
          and converter.is_scalar(nnefop.args["x"])):
        make_scalar(input_=nnefop.args["y"], scalar=nnefop.args["x"])
    elif (eltwise_operation is not None
          and converter.is_nonvar_nonconst_dn(nnefop.args["x"])
          and converter.is_nonvar_nonconst_dn(nnefop.args["y"])):
        make_eltwise(input1=nnefop.args["x"], input2=nnefop.args["y"])
    else:
        converter.print_error(nnefop, "Unsupported binary operation config")


def convert_local_response_normalization(nnefop, converter):
    # type: (NnefOp, Converter)->None
    # todo handle within channel

    converter.assert_scalar_value(nnefop, "bias", 1.0)

    converter.add_caffeop_ex(nnefop, "LRN",
                             [nnefop.args["input"]],
                             converter.ordered_dict_maker[
                                 "alpha": converter.default_to_none(nnefop.args["alpha"], 1.),
                                 "beta": converter.default_to_none(nnefop.args["beta"], 0.75)
                             ])


def generic_convert_pool(nnefop, converter, pool):
    # type: (NnefOp, Converter)->None
    if converter.get_rank_safe(nnefop.args["input"]) != 4:
        converter.print_error(nnefop, "Input rank != 4")

    converter.assert_nonspatial_items(nnefop, "padding", (0, 0))
    converter.assert_nonspatial_items(nnefop, "stride", 1)
    converter.assert_all_items(nnefop, "dilation", 1)

    kernel_size = converter.nnef_spatial_part(nnefop.args["size"])

    pad, stride, _dilation = converter.caffe_pad_stride_dilation(
        spatial_upscaled_size=converter.nnef_spatial_part(converter.get_shape_safe(nnefop.args["input"])),
        spatial_downscaled_size=converter.nnef_spatial_part(converter.get_shape_safe(nnefop.result)),
        spatial_filter_size=converter.nnef_spatial_part(nnefop.args["size"]),
        spatial_padding=converter.nnef_spatial_part(nnefop.args["padding"]),
        spatial_stride=converter.nnef_spatial_part(nnefop.args["stride"]),
        spatial_dilation=[],
        is_pool=True,
        log_anchor=nnefop
    )

    converter.add_caffeop_ex(nnefop, "Pooling",
                             [nnefop.args["input"]],
                             converter.ordered_dict_maker[
                                 "kernel_h": kernel_size[0],
                                 "kernel_w": kernel_size[1],
                                 "pool": pool,
                                 "pad_h": pad[0] if pad else None,
                                 "pad_w": pad[1] if pad else None,
                                 "stride_h": stride[0] if stride else None,
                                 "stride_w": stride[1] if stride else None
                             ])


def generic_convert_reduce(nnefop, converter, pool):
    # type: (NnefOp, Converter)->None
    if converter.get_rank_safe(nnefop.args["input"]) != 4:
        converter.print_error(nnefop, "Input rank != 4")

    if sorted(nnefop.args["axes"]) != [2, 3]:
        converter.print_error(nnefop, "Not convertible to global pooling.")

    converter.add_caffeop_ex(nnefop, "Pooling",
                             [nnefop.args["input"]],
                             converter.ordered_dict_maker[
                                 "pool": pool,
                                 "global_pooling": True
                             ])


def convert_concat(nnefop, converter):
    # type: (NnefOp, Converter)->None

    converter.add_caffeop_ex(nnefop, "Concat",
                             nnefop.args["values"],
                             converter.ordered_dict_maker[
                                 "axis": converter.default_to_none(nnefop.args["axis"], 1)
                             ])


def convert_argmax_reduce(nnefop, converter):
    # type: (NnefOp, Converter)->None

    if set(nnefop.args["axes"]) == set(range(1, converter.get_rank_safe(nnefop.args["input"]))):
        axis = None
    elif len(nnefop.args["axes"]) == 1:
        axis = nnefop.args["axes"]
    else:
        converter.print_error(nnefop, "Unsupported axes arg: {}".format(nnefop.args["axes"]))
        axis = None

    converter.add_caffeop_ex(nnefop, "ArgMax",
                             [nnefop.args["input"]],
                             converter.ordered_dict_maker[
                                 "axis": axis
                             ])


def convert_reshape(nnefop, converter):
    # type: (NnefOp, Converter)->None

    if utils.count(s == -1 for s in nnefop.args["shape"]) == 1:

        input_rank = converter.get_rank_safe(nnefop.args["input"])
        shape = nnefop.args["shape"]  # type: List[int]
        axis = shape.index(-1)
        end_axis = axis + (input_rank - len(shape))
        if end_axis == input_rank - 1:
            end_axis = -1

        tmp_shape = list(shape)
        for _ in range(input_rank - len(shape)):
            tmp_shape.insert(axis, -1)

        if all(s == t or t in [0, -1] for s, t in zip(shape, tmp_shape)):
            converter.add_caffeop_ex(nnefop, "Flatten",
                                     [nnefop.args["input"]],
                                     converter.ordered_dict_maker[
                                         "axis": converter.default_to_none(axis, 1),
                                         "end_axis": converter.default_to_none(end_axis, -1)
                                     ])
            return

    converter.add_caffeop_ex(nnefop, "Reshape",
                             [nnefop.args["input"]],
                             converter.ordered_dict_maker[
                                 "shape": nnefop.args["shape"]
                             ])


def generic_convert_squeeze_unsqueeze(nnefop, converter):
    # type: (NnefOp, Converter)->None

    converter.add_caffeop_ex(nnefop, "Reshape",
                             [nnefop.args["input"]],
                             converter.ordered_dict_maker[
                                 "shape": nnefop.result_node.shape
                             ])


def convert_batch_normalization(nnefop, converter):
    # type: (NnefOp, Converter)->None

    converter.assert_unique_variable(nnefop, "mean")
    converter.assert_unique_variable(nnefop, "variance")
    converter.assert_unique_variable(nnefop, "offset")
    converter.assert_unique_variable(nnefop, "scale")

    caffeop_batch_norm = CaffeOp("BatchNorm")
    caffeop_batch_norm.add_arg(dog.gen_arg_name(0), converter.get_caffedn(nnefop.args["input"]))
    caffeop_batch_norm.add_arg("eps", converter.default_to_none(nnefop.args["epsilon"], 1e-5))
    caffeop_batch_norm.add_result(dog.gen_result_name(0),
                                  converter.make_caffedn(name=nnefop.result.name, discriminator="norm"))
    caffeop_batch_norm.extra[EXTRA_CAFFE_PARAM_NAME] = "batch_norm_param"
    caffeop_batch_norm.extra[EXTRA_VARIABLE_LABELS] = [
        converter.get_label_safe(nnefop.args["mean"]),
        converter.get_label_safe(nnefop.args["variance"]),
        [1.0]
    ]
    converter.add_caffeop(caffeop_batch_norm, nnefop)

    caffeop_scale = CaffeOp("Scale")
    caffeop_scale.add_arg(dog.gen_arg_name(0), caffeop_batch_norm.result)
    caffeop_scale.add_arg("bias_term", True)
    caffeop_scale.add_result(dog.gen_result_name(0), converter.make_caffedn(nnefop.result))
    caffeop_scale.extra[EXTRA_VARIABLE_LABELS] = [
        converter.get_label_safe(nnefop.args["scale"]),
        converter.get_label_safe(nnefop.args["offset"])
    ]
    converter.add_caffeop(caffeop_scale, nnefop)


def convert_multilinear_upsample(nnefop, converter):
    # type: (NnefOp, Converter)->None

    if nnefop.args["method"] != "symmetric":
        converter.print_error(nnefop, "method != symmetric")
    if nnefop.args["border"] != "constant":
        converter.print_error(nnefop, "border != constant")
    if nnefop.args["factor"][0] != nnefop.args["factor"][1]:
        converter.print_error(nnefop, "factor not symmetric")

    factor = nnefop.args["factor"][0]
    kernel_size = 2 * factor - factor % 2

    converter.add_caffeop_ex(nnefop, "Deconvolution",
                             [nnefop.args["input"]],
                             converter.ordered_dict_maker[
                                 "kernel_size": [kernel_size, kernel_size],
                                 "num_output": converter.nnef_channels(dog.get_shape_safe(nnefop.result)),
                                 "bias_term": False,
                                 "pad": int(math.ceil((factor - 1) / 2.0)),
                                 "stride": factor,
                                 "weight_filler": dict(type="bilinear"),
                                 "group": dog.get_shape_safe(nnefop.args["input"])[1],
                             ],
                             extra={
                                 EXTRA_CAFFE_PARAM_NAME: "convolution_param",
                                 EXTRA_VARIABLE_LABELS: [VARIABLE_LABEL_SKIP]
                             })


def convert_matmul(nnefop, converter):
    # type: (NnefOp, Converter)->None

    converter.assert_unique_variable(nnefop, "B")

    if nnefop.args["transposeA"]:
        utils.print_error("transposeA not yet supported")

    converter.add_caffeop_ex(nnefop, "InnerProduct",
                             [nnefop.args["A"]],
                             converter.ordered_dict_maker[
                                 "num_output": nnefop.result_node.shape[-1],
                                 "bias_term": False,
                                 "axis": -1,
                                 "transpose": not nnefop.args["transposeB"]
                             ],
                             extra={
                                 EXTRA_CAFFE_PARAM_NAME: "inner_product_param",
                                 EXTRA_VARIABLE_LABELS: [converter.get_label_safe(nnefop.args["B"])]
                             })


def convert_slice(nnefop, converter):
    # type: (NnefOp, Converter)->None

    nnefdn_input = nnefop.args["input"]
    input_shape = converter.get_shape_safe(nnefdn_input)
    input_rank = converter.get_rank_safe(nnefdn_input)
    axes = nnefop.args["axes"]  # type: List[int]
    begin = nnefop.args["begin"]  # type: List[int]
    end = nnefop.args["end"]  # type: List[int]
    axis = min(axes)
    offset = [begin[axes.index(a)] if a in axes else 0 for a in range(axis, input_rank)]
    new_shape = [end[axes.index(a)] - begin[axes.index(a)]
                 if a in axes
                 else (input_shape[a] if a >= axis else 1)
                 for a in range(input_rank)]

    caffeop_reference = CaffeOp("Input")
    caffeop_reference.add_arg("shape", new_shape)
    caffeop_reference.add_result(dog.gen_result_name(0),
                                 converter.make_caffedn(name=nnefop.result.name, discriminator="reference"))
    converter.add_caffeop(caffeop_reference, nnefop)

    caffeop_crop = CaffeOp("Crop")
    caffeop_crop.add_arg(dog.gen_arg_name(0), converter.get_caffedn(nnefdn_input))
    caffeop_crop.add_arg(dog.gen_arg_name(1), caffeop_reference.result_node)
    caffeop_crop.add_arg("axis", converter.default_to_none(axis, 2))
    caffeop_crop.add_arg("offset", offset)
    caffeop_crop.add_result(dog.gen_result_name(0), converter.make_caffedn(nnefop.result))
    converter.add_caffeop(caffeop_crop, nnefop)


DefaultConverters = {
    "external": convert_external,
    "variable": convert_ignore,
    "concat": convert_concat,
    "softmax": convert_softmax,
    "argmax_reduce": convert_argmax_reduce,
    "reshape": convert_reshape,
    "squeeze": generic_convert_squeeze_unsqueeze,
    "unsqueeze": generic_convert_squeeze_unsqueeze,
    "leaky_relu": convert_leaky_relu,
    "local_response_normalization": convert_local_response_normalization,
    "batch_normalization": convert_batch_normalization,
    "multilinear_upsample": convert_multilinear_upsample,
    "matmul": convert_matmul,
    "slice": convert_slice,
    "conv": partial(generic_convert_conv, target_name="Convolution"),
    "deconv": partial(generic_convert_conv, target_name="Deconvolution", is_upscale=True),
    "max_pool": partial(generic_convert_pool, pool=0),
    "avg_pool": partial(generic_convert_pool, pool=1),
    "max_reduce": partial(generic_convert_reduce, pool=0),
    "mean_reduce": partial(generic_convert_reduce, pool=1),
    "elu": partial(generic_convert_unary, target_name="ELU"),
    "relu": partial(generic_convert_unary, target_name="ReLU"),
    "prelu": convert_prelu,
    "sigmoid": partial(generic_convert_unary, target_name="Sigmoid"),
    "abs": partial(generic_convert_unary, target_name="AbsVal"),
    "tanh": partial(generic_convert_unary, target_name="TanH"),
    "softplus": partial(generic_convert_unary, target_name="BNLL"),
    "add": partial(generic_convert_binary,
                   is_commutative=True,
                   eltwise_operation=1,
                   eltwise_coeff=[1.0, 1.0],
                   learnable_name="Bias",
                   scalar_name="Power",
                   scalar_arg_name="shift"),
    "mul": partial(generic_convert_binary,
                   is_commutative=True,
                   eltwise_operation=0,
                   learnable_name="Scale",
                   scalar_name="Power",
                   scalar_arg_name="scale"),
    "max": partial(generic_convert_binary,
                   is_commutative=True,
                   eltwise_operation=2),
    "pow": partial(generic_convert_binary,
                   scalar_name="Power",
                   scalar_arg_name="power")
}  # type: Dict[str, Callable[(NnefOp, Converter), None]]
