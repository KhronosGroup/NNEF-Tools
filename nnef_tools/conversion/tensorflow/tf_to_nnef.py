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
import typing
from functools import partial

import numpy as np

from nnef_tools.conversion import converter
from nnef_tools.conversion import transforms
from nnef_tools.conversion.shape_utils import ShapeUtils
from nnef_tools.conversion.tensorflow import tf_to_nnef_passes
from nnef_tools.core import utils
from nnef_tools.io.nnef.nnef_graph import *
from nnef_tools.io.tensorflow.tf_graph import *

_nnef_dtype_by_tf_dtype = {
    "float16": 'scalar',
    "float32": 'scalar',
    "float64": 'scalar',
    "int8": 'integer',
    "uint8": 'integer',
    "int16": 'integer',
    "uint16": 'integer',
    "int32": 'integer',
    "uint32": 'integer',
    "int64": 'integer',
    "uint64": 'integer',
    "bool": 'logical'
}

_NNEFOpOrTupleOrList = typing.Union[NNEFOperation, typing.Tuple[NNEFOperation, ...], typing.List[NNEFOperation]]


class Converter(converter.Converter[TFTensor, TFOperation, TFGraph,
                                    NNEFTensor, NNEFOperation, NNEFGraph], ShapeUtils):

    def __init__(self,
                 enable_standard_converters=True,
                 enable_default_conversion=False,
                 enable_imprecise_image_resize=False,
                 custom_converter_by_op_name=None):
        converters = {}
        if enable_standard_converters:
            converters.update(_StandardConverters)
        if custom_converter_by_op_name is not None:
            converters.update(custom_converter_by_op_name)
        default_op_converter = convert_default if enable_default_conversion else None

        super(Converter, self).__init__(op_converter_by_name=converters,
                                        default_op_converter=default_op_converter)

        self.enable_imprecise_image_resize = enable_imprecise_image_resize
        self.is_quantized = False

    def create_graph(self, source_graph):
        # type: (TFGraph)->NNEFGraph
        return NNEFGraph(source_graph.name)

    def convert_tensor(self, source_tensor, target_graph):
        # type: (TFTensor, NNEFGraph)->NNEFTensor

        if source_tensor.quantization is None or source_tensor.quantization.all_zero():
            quantization = None
        else:
            if source_tensor.dtype not in ["uint8", "int32"]:
                print("Warning: non-int tensor had quantization: {}".format(source_tensor))
                quantization = None
            else:
                def none_to_float_zero(x):
                    return 0.0 if x is None else x

                def none_to_int_zero(x):
                    return 0 if x is None else x

                bits = 8 if source_tensor.dtype == "uint8" else 32
                quantization = NNEFQuantization(
                    name='tflite_quantize',
                    attribs=dict(min=none_to_float_zero(source_tensor.quantization.min),
                                 max=none_to_float_zero(source_tensor.quantization.max),
                                 scale=none_to_float_zero(source_tensor.quantization.scale),
                                 zero_point=none_to_int_zero(source_tensor.quantization.zero_point),
                                 bits=bits))

        return NNEFTensor(graph=target_graph,
                          name=None,
                          shape=list(source_tensor.shape),
                          dtype='scalar' if self.is_quantized else self.nnef_dtype(source_tensor.dtype),
                          data=(np.array(source_tensor.data, dtype=np.dtype(source_tensor.dtype)).flatten().tolist()
                                if source_tensor.is_constant else copy.copy(source_tensor.data)),
                          label=self.nnef_variable_label(source_tensor) if source_tensor.is_variable else None,
                          quantization=quantization)

    def convert_graph(self, source_graph):
        # type: (TFGraph)->NNEFGraph
        # TODO: we dont have to collapse conv/add or conv/pad if we use ext
        if any(t.quantization is not None for t in source_graph.tensors):
            self.is_quantized = True
        tf_to_nnef_passes.pre_conversion_pass(source_graph)
        # source_graph.dump()
        target_graph = super(Converter, self).convert_graph(source_graph)
        target_graph.generate_missing_names()
        return target_graph

    def can_include_in_conversion_info(self, source_tensor, target_tensor):
        return (super(Converter, self).can_include_in_conversion_info(source_tensor, target_tensor)
                and ":" in source_tensor.name)

    @staticmethod
    def is_nhwc(tf_op, default=True):
        # type: (TFOperation, bool)->bool
        if not tf_op.attribs.get("data_format"):
            return default
        return not tf_op.attribs["data_format"].upper().startswith("NC")

    @staticmethod
    def nnef_variable_label(tf_tensor):
        # type: (TFTensor)->None
        if tf_tensor.name.endswith("/read:0"):
            return tf_tensor.name[:-len("/read:0")]
        elif tf_tensor.name.endswith(":0"):
            return tf_tensor.name[:-2]
        else:
            return tf_tensor.name.replace(':', '_')

    @staticmethod
    def is_one_element_constant(tf_tensor):
        # type: (TFTensor)->bool
        return tf_tensor.is_constant and len(Converter.nnef_constant_value(tf_tensor)) == 1

    @staticmethod
    def nnef_constant_value(tf_tensor):
        # type: (TFTensor)->typing.List[typing.Union[int, float, bool]]
        assert isinstance(tf_tensor, TFTensor)
        assert tf_tensor.is_constant

        value = tf_tensor.data
        shape = tf_tensor.shape
        dtype = tf_tensor.dtype if tf_tensor.dtype else "float32"

        value = np.array(value, dtype=np.dtype(dtype))
        if np.all(value.flat == value.flat[0]):
            return [value.flat[0].item()]

        last_val = value.flat[-1]
        value2 = np.full(shape=shape, fill_value=last_val, dtype=np.dtype(dtype))
        value2.flat[:value.size] = value.flat

        return list(value2.flatten().tolist())

    @staticmethod
    def nnef_one_element_constant_value(tf_tensor):
        value = Converter.nnef_constant_value(tf_tensor)
        assert len(value) == 1
        return value[0]

    @staticmethod
    def spatial_size(shape, is_nhwc):
        return list(shape[1:-1] if is_nhwc else shape[2:])

    @staticmethod
    def full_size(shape, is_nhwc):
        return [1] + shape + [1] if is_nhwc else [1, 1] + shape

    @staticmethod
    def nnef_padding(padding, rank, is_nhwc, out_spatial):
        if out_spatial:
            assert is_nhwc is not None
        if isinstance(padding, (list, tuple)):
            assert len(padding) == rank
            full = [(a, b) for a, b in padding]
        else:
            full = [] if padding.upper() == 'SAME' else [(0, 0)] * rank
        return Converter.spatial_size(full, is_nhwc) if out_spatial else full

    @staticmethod
    def _nnef_stride_or_dilation(stride_or_dilation, rank, from_nhwc, out_spatial=False):
        assert (stride_or_dilation is None
                or len(stride_or_dilation) == 1
                or len(stride_or_dilation) == rank - 2
                or len(stride_or_dilation) == rank)

        if stride_or_dilation is None:
            spatial = [1] * (rank - 2)
        elif len(stride_or_dilation) == 1:
            spatial = stride_or_dilation * (rank - 2)
        elif len(stride_or_dilation) == rank - 2:
            spatial = stride_or_dilation
        elif len(stride_or_dilation) == rank:
            spatial = Converter.spatial_size(stride_or_dilation, from_nhwc)
        else:
            assert False

        return spatial if out_spatial else Converter.full_size(spatial, is_nhwc=False)

    @staticmethod
    def nnef_stride(stride, rank, from_nhwc, out_spatial=False):
        return Converter._nnef_stride_or_dilation(stride, rank, from_nhwc, out_spatial)

    @staticmethod
    def nnef_dilation(dilation, rank, from_nhwc, out_spatial=False):
        return Converter._nnef_stride_or_dilation(dilation, rank, from_nhwc, out_spatial)

    @staticmethod
    def nnef_border(border):
        border = border.lower()
        if border == 'symmetric':
            border = 'reflect-even'
        return border

    @staticmethod
    def nnef_axis(axis, rank):
        while axis < 0:
            axis += rank
        return axis

    @staticmethod
    def nnef_axes(axes, rank, none_means_all=False):
        if axes is None:
            if none_means_all:
                return list(range(rank))
            assert False, "Axes is None, use none_means_all if applicable"
        return [Converter.nnef_axis(a, rank) for a in axes]

    @staticmethod
    def nnef_dtype(tf_dtype):
        if tf_dtype is None:
            return None
        if tf_dtype.endswith('_ref'):
            tf_dtype = tf_dtype[:-len('_ref')]
        return _nnef_dtype_by_tf_dtype[tf_dtype]

    @staticmethod
    def reduced_shape(input_shape, axes):
        return [1 if i in axes else s for i, s in enumerate(input_shape)]

    @staticmethod
    def create_nchw_intermediate_for_nhwc_output(nnef_graph, nnef_tensor):
        return NNEFTensor(graph=nnef_graph,
                          shape=Converter.shape_nhwc_to_nchw(nnef_tensor.shape),
                          dtype=nnef_tensor.dtype,
                          data=nnef_tensor.data)

    @staticmethod
    def create_nchw_intermediate_for_hwcn_output(nnef_graph, nnef_tensor):
        return NNEFTensor(graph=nnef_graph,
                          shape=Converter.shape_hwcn_to_nchw(nnef_tensor.shape),
                          dtype=nnef_tensor.dtype,
                          data=nnef_tensor.data)

    @staticmethod
    def create_nchw_intermediate_for_hwcm_output(nnef_graph, nnef_tensor):
        return NNEFTensor(graph=nnef_graph,
                          shape=Converter.shape_hwcm_to_nchw(nnef_tensor.shape),
                          dtype=nnef_tensor.dtype,
                          data=nnef_tensor.data)

    @staticmethod
    def create_hwcm_intermediate_for_nchw_output(nnef_graph, nnef_tensor, input_channels):
        return NNEFTensor(graph=nnef_graph,
                          shape=Converter.shape_nchw_to_hwcm(nnef_tensor.shape, input_channels),
                          dtype=nnef_tensor.dtype,
                          data=nnef_tensor.data)

    @staticmethod
    def create_hwcn_intermediate_for_nchw_output(nnef_graph, nnef_tensor):
        return NNEFTensor(graph=nnef_graph,
                          shape=Converter.shape_nchw_to_hwcn(nnef_tensor.shape),
                          dtype=nnef_tensor.dtype,
                          data=nnef_tensor.data)

    @staticmethod
    def add_nhwc_to_nchw_transpose(nnef_graph, nnef_tensor, nchw_tensor=None):
        assert isinstance(nnef_graph, NNEFGraph)
        assert isinstance(nnef_tensor, NNEFTensor)

        if nchw_tensor is None:
            nchw_tensor = NNEFTensor(graph=nnef_graph,
                                     shape=Converter.shape_nhwc_to_nchw(nnef_tensor.shape),
                                     dtype=nnef_tensor.dtype)

        return NNEFOperation(graph=nnef_graph,
                             name="transpose",
                             inputs=nnef_tensor,
                             attribs=dict(axes=Converter.transpose_axes_nhwc_to_nchw(nnef_tensor.rank)),
                             outputs=nchw_tensor).output

    @staticmethod
    def add_nchw_to_nhwc_transpose(nnef_graph, nnef_tensor, nhwc_tensor=None):
        assert isinstance(nnef_graph, NNEFGraph)
        assert isinstance(nnef_tensor, NNEFTensor)
        assert nhwc_tensor is None or isinstance(nhwc_tensor, NNEFTensor)

        if nhwc_tensor is None:
            nhwc_tensor = NNEFTensor(graph=nnef_graph,
                                     shape=Converter.shape_nchw_to_nhwc(nnef_tensor.shape),
                                     dtype=nnef_tensor.dtype)
        NNEFOperation(graph=nnef_graph,
                      name="transpose",
                      inputs=nnef_tensor,
                      attribs=dict(axes=Converter.transpose_axes_nchw_to_nhwc(nnef_tensor.rank)),
                      outputs=nhwc_tensor)
        return nhwc_tensor

    @staticmethod
    def add_nchw_to_hwcn_transpose(nnef_graph, nnef_tensor, hwcn_tensor=None):
        assert isinstance(nnef_graph, NNEFGraph)
        assert isinstance(nnef_tensor, NNEFTensor)
        assert hwcn_tensor is None or isinstance(hwcn_tensor, NNEFTensor)

        if hwcn_tensor is None:
            hwcn_tensor = NNEFTensor(graph=nnef_graph,
                                     shape=Converter.shape_nchw_to_hwcn(nnef_tensor.shape),
                                     dtype=nnef_tensor.dtype)
        NNEFOperation(graph=nnef_graph,
                      name="transpose",
                      inputs=nnef_tensor,
                      attribs=dict(axes=Converter.transpose_axes_nchw_to_hwcn(nnef_tensor.rank)),
                      outputs=hwcn_tensor)
        return hwcn_tensor

    @staticmethod
    def add_hwcn_to_nchw_transpose(nnef_graph, nnef_tensor, nchw_tensor=None):
        assert isinstance(nnef_graph, NNEFGraph)
        assert isinstance(nnef_tensor, NNEFTensor)

        if nchw_tensor is None:
            nchw_tensor = NNEFTensor(graph=nnef_graph,
                                     shape=Converter.shape_hwcn_to_nchw(nnef_tensor.shape),
                                     dtype=nnef_tensor.dtype)

        return NNEFOperation(graph=nnef_graph,
                             name="transpose",
                             inputs=nnef_tensor,
                             attribs=dict(axes=Converter.transpose_axes_hwcn_to_nchw(nnef_tensor.rank)),
                             outputs=nchw_tensor).output

    @staticmethod
    def add_hwcm_to_hwcn_reshape(nnef_graph, nnef_tensor):
        # type: (NNEFGraph, NNEFTensor)->NNEFTensor
        assert isinstance(nnef_graph, NNEFGraph)
        assert isinstance(nnef_tensor, NNEFTensor)

        new_shape = nnef_tensor.shape[:-2] + [1, nnef_tensor.shape[-1] * nnef_tensor.shape[-2]]
        return NNEFOperation(graph=nnef_graph,
                             name="reshape",
                             inputs=nnef_tensor,
                             attribs=dict(shape=list(new_shape)),
                             outputs=NNEFTensor(graph=nnef_graph,
                                                shape=list(new_shape),
                                                dtype=nnef_tensor.dtype)).output

    @staticmethod
    def add_hwcn_to_hwcm_reshape(nnef_graph, nnef_tensor, hwcm_tensor=None, in_channels=None):
        assert isinstance(nnef_graph, NNEFGraph)
        assert isinstance(nnef_tensor, NNEFTensor)
        assert hwcm_tensor is None or isinstance(hwcm_tensor, NNEFTensor)
        assert hwcm_tensor is not None or in_channels is not None

        if hwcm_tensor is not None:
            new_shape = hwcm_tensor.shape
        else:
            in_channels = int(in_channels)
            assert in_channels % nnef_tensor.shape[-2] == 0
            groups = int(in_channels // nnef_tensor.shape[-2])
            assert nnef_tensor.shape[-1] % groups == 0
            new_shape = nnef_tensor.shape[:-2] + [in_channels, int(nnef_tensor.shape[-1] // groups)]
            hwcm_tensor = NNEFTensor(graph=nnef_graph,
                                     shape=list(new_shape),
                                     dtype=nnef_tensor.dtype)

        return NNEFOperation(graph=nnef_graph,
                             name="reshape",
                             inputs=nnef_tensor,
                             attribs=dict(shape=list(new_shape)),
                             outputs=hwcm_tensor).output

    @staticmethod
    def is_valid_tf_bias_shape(shape):
        # type: (typing.List[int])->bool
        return len(shape) == 1 or all(i == 1 or s == 1 for i, s in enumerate(shape))

    @staticmethod
    def add_unsqueeze(nnef_graph, nnef_tensor, axes):
        # type: (NNEFGraph, NNEFTensor, typing.List[int])->NNEFTensor
        return NNEFOperation(graph=nnef_graph,
                             name="unsqueeze",
                             inputs=nnef_tensor,
                             attribs=dict(axes=axes),
                             outputs=NNEFTensor(graph=nnef_graph,
                                                shape=transforms.unsqueezed_shape(nnef_tensor.shape, axes),
                                                dtype=nnef_tensor.dtype)).output

    @staticmethod
    def create_nhwc_intermediate_for_nchw_output(nnef_graph, nnef_tensor):
        return NNEFTensor(graph=nnef_graph,
                          shape=Converter.shape_nchw_to_nhwc(nnef_tensor.shape),
                          dtype=nnef_tensor.dtype,
                          data=nnef_tensor.data)

    @staticmethod
    def add_squeeze(nnef_graph, nnef_tensor, axes):
        # type: (NNEFGraph, NNEFTensor, typing.List[int])->NNEFTensor
        return NNEFOperation(graph=nnef_graph,
                             name="squeeze",
                             inputs=nnef_tensor,
                             attribs=dict(axes=axes),
                             outputs=NNEFTensor(graph=nnef_graph,
                                                shape=transforms.squeezed_shape(nnef_tensor.shape, axes),
                                                dtype=nnef_tensor.dtype)).output

    @staticmethod
    def nnef_zero_value(nnef_dtype):
        if nnef_dtype == "scalar":
            return 0.0
        elif nnef_dtype == "integer":
            return 0
        elif nnef_dtype == "logical":
            return False
        assert False, "Unsupported NNEF dtype: {}".format(nnef_dtype)

    @staticmethod
    def nnef_addition_op(nnef_dtype):
        if nnef_dtype == "scalar":
            return "add"
        elif nnef_dtype == "integer":
            return "add"
        elif nnef_dtype == "logical":
            return "or"
        assert False, "Unsupported NNEF dtype: {}".format(nnef_dtype)

    @staticmethod
    def has_gt_1(data):
        return utils.has_gt_1(data)

    @staticmethod
    def create_constant_nnef_tensor(nnef_graph, data, dtype="scalar"):
        if dtype == "scalar":
            data = float(data)
        return NNEFTensor(graph=nnef_graph, shape=[], dtype=dtype, data=[data])

    @staticmethod
    def can_broadcast_from_right(shape, other_shape):
        if len(other_shape) < len(shape):
            other_shape, shape = shape, other_shape

        for i in range(len(shape)):
            if shape[-1 - i] != 1 and other_shape[-1 - i] != 1 and shape[-1 - i] != other_shape[-1 - i]:
                return False
        return True


def convert_default(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    print("Warning: Converter of {} is not implemented, doing default conversion.".format(tf_op.name))

    NNEFOperation(graph=nnef_graph,
                  name=tf_op.name.replace('.', '_'),
                  inputs=converter.converted_tensors(tf_op.inputs),
                  attribs=utils.recursive_transform(tf_op.attribs, lambda x: x if x is not None else "None"),
                  outputs=converter.converted_tensors(tf_op.outputs))


def convert_assign(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None
    ref, value = tf_op.inputs
    NNEFOperation(graph=nnef_graph,
                  name="update",
                  inputs=converter.converted_tensors((ref, value)),
                  outputs=converter.converted_tensors(tf_op.outputs))


def generic_convert_conv(converter, tf_op, nnef_graph, is_deconv, is_planewise):
    # type: (Converter, TFOperation, NNEFGraph, bool, bool)->None
    is_nhwc = converter.is_nhwc(tf_op)

    input, filter = converter.converted_tensors(tf_op.inputs[0:2])
    bias = (converter.converted_tensor(tf_op.inputs[2]) if len(tf_op.inputs) >= 3
            else converter.create_constant_nnef_tensor(nnef_graph, 0.0))
    assert converter.is_valid_tf_bias_shape(bias.shape)

    output = converter.converted_tensor(tf_op.output)

    def reshape_transpose(filter_):
        return converter.add_hwcn_to_nchw_transpose(nnef_graph, converter.add_hwcm_to_hwcn_reshape(nnef_graph, filter_))

    nnef_op = NNEFOperation(
        graph=nnef_graph,
        name="deconv" if is_deconv else "conv",
        inputs=(converter.add_nhwc_to_nchw_transpose(nnef_graph, input) if is_nhwc else input,
                reshape_transpose(filter) if is_planewise else converter.add_hwcn_to_nchw_transpose(nnef_graph, filter),
                converter.add_unsqueeze(nnef_graph, bias, axes=[0]) if bias.rank == 1 else bias),
        attribs=dict(
            border=converter.nnef_border(tf_op.attribs.get('_border', 'constant')),
            padding=converter.nnef_padding(tf_op.attribs['padding'], input.rank, is_nhwc, out_spatial=True),
            stride=converter.nnef_stride(tf_op.attribs.get('stride'), input.rank, is_nhwc, out_spatial=True),
            dilation=converter.nnef_dilation(tf_op.attribs.get('dilation'), input.rank, is_nhwc, out_spatial=True),
            groups=0 if is_planewise else 1),
        outputs=converter.create_nchw_intermediate_for_nhwc_output(nnef_graph, output) if is_nhwc else output)

    if is_deconv:
        nnef_op.attribs["output_shape"] = list(nnef_op.output.shape)

    if is_nhwc:
        converter.add_nchw_to_nhwc_transpose(nnef_graph, nnef_op.output, nhwc_tensor=output)


def convert_separable_conv(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None
    is_nhwc = converter.is_nhwc(tf_op)

    input, plain_filter, point_filter = converter.converted_tensors(tf_op.inputs[0:3])
    bias = (converter.converted_tensor(tf_op.inputs[3]) if len(tf_op.inputs) >= 4
            else converter.create_constant_nnef_tensor(nnef_graph, 0.0))
    assert converter.is_valid_tf_bias_shape(bias.shape)

    output = converter.converted_tensor(tf_op.output)

    nnef_op = NNEFOperation(
        graph=nnef_graph,
        name="separable_conv",
        inputs=(converter.add_nhwc_to_nchw_transpose(nnef_graph, input) if is_nhwc else input,
                converter.add_hwcn_to_nchw_transpose(
                    nnef_graph, converter.add_hwcm_to_hwcn_reshape(nnef_graph, plain_filter)),
                converter.add_hwcn_to_nchw_transpose(nnef_graph, point_filter),
                converter.add_unsqueeze(nnef_graph, bias, axes=[0]) if bias.rank == 1 else bias),
        attribs=dict(
            border=converter.nnef_border(tf_op.attribs.get('_border', 'constant')),
            padding=converter.nnef_padding(tf_op.attribs['padding'], input.rank, is_nhwc, out_spatial=True),
            stride=converter.nnef_stride(tf_op.attribs.get('stride'), input.rank, is_nhwc, out_spatial=True),
            dilation=converter.nnef_dilation(tf_op.attribs.get('dilation'), input.rank, is_nhwc, out_spatial=True),
            groups=1),
        outputs=converter.create_nchw_intermediate_for_nhwc_output(nnef_graph, output) if is_nhwc else output)

    if is_nhwc:
        converter.add_nchw_to_nhwc_transpose(nnef_graph, nnef_op.output, nhwc_tensor=output)


def generic_convert_pool(converter, tf_op, nnef_graph, target_name):
    # type: (Converter, TFOperation, NNEFGraph, str)->None
    is_nhwc = converter.is_nhwc(tf_op)

    input = converter.converted_tensor(tf_op.input)
    outputs = converter.converted_tensors(tf_op.outputs)

    size = list(tf_op.attribs["size"])
    padding = converter.nnef_padding(tf_op.attribs["padding"], input.rank, is_nhwc, out_spatial=False)
    border = converter.nnef_border(tf_op.attribs.get('_border', 'ignore'))
    stride = converter.nnef_stride(tf_op.attribs.get('stride'), input.rank, is_nhwc, out_spatial=False)

    nnef_op = NNEFOperation(
        graph=nnef_graph,
        name=target_name,
        inputs=converter.add_nhwc_to_nchw_transpose(nnef_graph, input) if is_nhwc else input,
        attribs=dict(
            size=converter.shape_nhwc_to_nchw(size) if is_nhwc else list(size),
            padding=converter.shape_nhwc_to_nchw(padding) if is_nhwc else list(padding),
            border=border,
            stride=stride,
            dilation=[]),
        outputs=(tuple(converter.create_nchw_intermediate_for_nhwc_output(nnef_graph, output) for output in outputs)
                 if is_nhwc else outputs)
    )

    if is_nhwc:
        for intermediate, output in zip(nnef_op.outputs, outputs):
            converter.add_nchw_to_nhwc_transpose(nnef_graph, intermediate, nhwc_tensor=output)


def generic_convert_unary(converter, tf_op, nnef_graph, target_name):
    # type: (Converter, TFOperation, NNEFGraph, str)->None
    NNEFOperation(graph=nnef_graph,
                  name=target_name,
                  inputs=converter.converted_tensor(tf_op.input),
                  outputs=converter.converted_tensor(tf_op.output))


def generic_convert_binary(converter, tf_op, nnef_graph, target_name):
    # type: (Converter, TFOperation, NNEFGraph, str)->None

    x, y = converter.converted_tensors(tf_op.inputs)
    z = converter.converted_tensor(tf_op.output)
    assert converter.can_broadcast_from_right(x.shape, y.shape)

    NNEFOperation(
        graph=nnef_graph,
        name=target_name,
        inputs=(converter.add_unsqueeze(nnef_graph, x, list(range(y.rank - x.rank))) if 0 < x.rank < y.rank else x,
                converter.add_unsqueeze(nnef_graph, y, list(range(x.rank - y.rank))) if 0 < y.rank < x.rank else y),
        outputs=z)


def convert_leaky_relu(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    tf_x, tf_a = tf_op.inputs

    if converter.is_one_element_constant(tf_a):
        x = converter.converted_tensor(tf_x)
        a = converter.nnef_one_element_constant_value(tf_a)
        output = converter.converted_tensor(tf_op.output)

        NNEFOperation(
            graph=nnef_graph,
            name="leaky_relu",
            inputs=x,
            attribs=dict(alpha=a),
            outputs=output)
    else:
        x, a = converter.converted_tensors((tf_x, tf_a))
        output = converter.converted_tensor(tf_op.output)

        NNEFOperation(
            graph=nnef_graph,
            name="prelu",
            inputs=(x,
                    converter.add_unsqueeze(nnef_graph, a, list(range(x.rank - a.rank))) if 0 < a.rank < x.rank else a),
            outputs=output)


def convert_squared_difference(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    x, y = converter.converted_tensors(tf_op.inputs)
    z = converter.converted_tensor(tf_op.output)
    assert converter.can_broadcast_from_right(x.shape, y.shape)

    sub = NNEFOperation(
        graph=nnef_graph,
        name="sub",
        inputs=(converter.add_unsqueeze(nnef_graph, x, list(range(y.rank - x.rank))) if 0 < x.rank < y.rank else x,
                converter.add_unsqueeze(nnef_graph, y, list(range(x.rank - y.rank))) if 0 < y.rank < x.rank else y),
        outputs=NNEFTensor(graph=nnef_graph, shape=list(z.shape), dtype=z.dtype))

    NNEFOperation(graph=nnef_graph, name="sqr", inputs=sub.output, outputs=z)


def convert_where(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    assert len(tf_op.inputs) == 3, "Only 3-param tf.where is supported"
    cond, x, y = converter.converted_tensors(tf_op.inputs)
    z = converter.converted_tensor(tf_op.output)

    NNEFOperation(graph=nnef_graph, name="select", inputs=(cond, x, y), outputs=z)


def generic_convert_reduce(converter, tf_op, nnef_graph, target_name):
    # type: (Converter, TFOperation, NNEFGraph, str)->None

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))
    axes = utils.unique(sorted(converter.nnef_axes(axes=tf_op.attribs['axis'], rank=input.rank, none_means_all=True)))

    if tf_op.attribs["keepdims"]:
        NNEFOperation(graph=nnef_graph, name=target_name, inputs=input, attribs=dict(axes=axes), outputs=output)
    else:
        reduced_shape = converter.reduced_shape(input.shape, axes)
        reduce = NNEFOperation(graph=nnef_graph,
                               name=target_name,
                               inputs=input,
                               attribs=dict(axes=axes),
                               outputs=NNEFTensor(graph=nnef_graph, shape=reduced_shape, dtype=output.dtype))
        NNEFOperation(graph=nnef_graph,
                      name="squeeze",
                      inputs=reduce.output,
                      attribs=dict(axes=list(axes)),
                      outputs=output)


def generic_convert_arg_min_max(converter, tf_op, nnef_graph, target_name):
    # type: (Converter, TFOperation, NNEFGraph, str)->None

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))
    axes = sorted(converter.nnef_axes(axes=[tf_op.attribs['axis']], rank=input.rank, none_means_all=True))

    reduced_shape = converter.reduced_shape(input.shape, axes)
    reduce = NNEFOperation(graph=nnef_graph,
                           name=target_name,
                           inputs=input,
                           attribs=dict(axes=axes),
                           outputs=NNEFTensor(graph=nnef_graph,
                                              name=None,
                                              shape=reduced_shape,
                                              dtype=output.dtype))
    NNEFOperation(graph=nnef_graph, name="squeeze", inputs=reduce.output, attribs=dict(axes=axes), outputs=output)


def convert_lrn(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None
    is_nhwc = True

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))

    depth_radius = int(utils.first_set(tf_op.attribs.get('depth_radius'), 5))
    bias = float(utils.first_set(tf_op.attribs.get('bias'), 1.0))
    alpha = float(utils.first_set(tf_op.attribs.get('alpha'), 1.0))
    beta = float(utils.first_set(tf_op.attribs.get('beta'), 0.5))

    depth_size = 2 * depth_radius + 1

    nnef_op = NNEFOperation(
        graph=nnef_graph,
        name="local_response_normalization",
        inputs=converter.add_nhwc_to_nchw_transpose(nnef_graph, input) if is_nhwc else input,
        attribs=dict(size=[1, depth_size, 1, 1],
                     alpha=alpha * depth_size,
                     beta=beta,
                     bias=bias),
        outputs=converter.create_nchw_intermediate_for_nhwc_output(nnef_graph, output) if is_nhwc else output)

    if is_nhwc:
        converter.add_nchw_to_nhwc_transpose(nnef_graph, nnef_op.output, nhwc_tensor=output)


def convert_batch_normalization(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    input, mean, variance, offset, scale = converter.converted_tensors(tf_op.inputs)

    is_nhwc = (not (mean.rank >= 2 and mean.shape[1] > 1)
               and not tf_op.attribs.get("_data_format", "NHWC").upper().startswith("NC"))

    output = converter.converted_tensor(tf_op.output)

    def batch_norm_param(tensor):
        if tensor.rank == 1:
            return converter.add_unsqueeze(nnef_graph=nnef_graph, nnef_tensor=tensor, axes=[0])
        elif is_nhwc:
            return converter.add_nhwc_to_nchw_transpose(nnef_graph, tensor)
        else:
            return tensor

    nnef_op = NNEFOperation(
        graph=nnef_graph,
        name="batch_normalization",
        inputs=(converter.add_nhwc_to_nchw_transpose(nnef_graph, input) if is_nhwc else input,
                batch_norm_param(mean),
                batch_norm_param(variance),
                batch_norm_param(offset),
                batch_norm_param(scale)),
        attribs=dict(epsilon=float(tf_op.attribs['variance_epsilon'])),
        outputs=converter.create_nchw_intermediate_for_nhwc_output(nnef_graph, output) if is_nhwc else output)

    if is_nhwc:
        converter.add_nchw_to_nhwc_transpose(nnef_graph, nnef_op.output, nhwc_tensor=output)


def convert_l2_normalization(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))

    # TODO epsilon and bias are not really the same, is this a problem?
    NNEFOperation(graph=nnef_graph,
                  name="l2_normalization",
                  inputs=input,
                  attribs=dict(axes=sorted(converter.nnef_axes(axes=tf_op.attribs["axis"],
                                                               rank=input.rank,
                                                               none_means_all=True)),
                               bias=float(tf_op.attribs['epsilon'])),
                  outputs=output)


def convert_matmul(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    A, B = converter.converted_tensors(tf_op.inputs)
    output = converter.converted_tensor(tf_op.output)

    NNEFOperation(graph=nnef_graph,
                  name="matmul",
                  inputs=(A, B),
                  attribs=dict(transposeA=bool(tf_op.attribs["transpose_a"] or tf_op.attribs.get('adjoint_a')),
                               transposeB=bool(tf_op.attribs["transpose_b"] or tf_op.attribs.get('adjoint_b'))),
                  outputs=output)


def convert_add_n(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    NNEFOperation(graph=nnef_graph,
                  name="add_n",
                  inputs=converter.converted_tensors(tf_op.inputs),
                  outputs=converter.converted_tensors(tf_op.outputs))


def convert_bias_add(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    is_nhwc = converter.is_nhwc(tf_op)

    input, bias = converter.converted_tensors(tf_op.inputs)
    output = converter.converted_tensor(tf_op.output)

    nnef_op = NNEFOperation(
        graph=nnef_graph,
        name="add",
        inputs=(converter.add_nhwc_to_nchw_transpose(nnef_graph, input) if is_nhwc else input,
                converter.add_unsqueeze(nnef_graph, bias, axes=[0]) if bias.rank == 1 else bias),
        outputs=converter.create_nchw_intermediate_for_nhwc_output(nnef_graph, output) if is_nhwc else output)

    if is_nhwc:
        converter.add_nchw_to_nhwc_transpose(nnef_graph, nnef_op.output, nhwc_tensor=output)


def convert_concat(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None
    assert len(tf_op.inputs) > 0, "Concatenation of zero tensors is unsupported"

    inputs = list(converter.converted_tensors(tf_op.inputs))
    output = converter.converted_tensor(tf_op.output)

    NNEFOperation(graph=nnef_graph,
                  name="concat",
                  inputs=inputs,
                  attribs=dict(axis=converter.nnef_axis(tf_op.attribs["axis"], inputs[0].rank)),
                  outputs=output)


def convert_split(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None
    assert len(tf_op.outputs) > 0, "Split to zero tensors in unsupported"

    input = converter.converted_tensor(tf_op.input)
    outputs = list(converter.converted_tensors(tf_op.outputs))

    num_or_sizes = tf_op.attribs['num_or_size_splits']
    NNEFOperation(graph=nnef_graph,
                  name="split",
                  inputs=input,
                  attribs=dict(axis=converter.nnef_axis(tf_op.attribs["axis"], input.rank),
                               ratios=(list(num_or_sizes)
                                       if isinstance(num_or_sizes, (list, tuple))
                                       else [1] * num_or_sizes)),
                  outputs=outputs)


def convert_softmax(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))

    axis = tf_op.attribs["axis"] if tf_op.attribs.get("axis") is not None else -1

    NNEFOperation(graph=nnef_graph,
                  name='softmax',
                  inputs=input,
                  attribs=dict(axes=[converter.nnef_axis(axis, input.rank)]),
                  outputs=output)


def convert_moments(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    input = converter.converted_tensor(tf_op.input)
    mean, variance = converter.converted_tensors(tf_op.outputs)

    axes = converter.nnef_axes(tf_op.attribs["axes"], input.rank, none_means_all=True)

    if tf_op.attribs["keep_dims"]:
        NNEFOperation(graph=nnef_graph,
                      name="moments",
                      inputs=input,
                      attribs=dict(axes=axes),
                      outputs=(mean, variance))
    else:
        reduced_shape = converter.reduced_shape(input.shape, axes)
        moments = NNEFOperation(
            graph=nnef_graph,
            name="moments",
            inputs=input,
            attribs=dict(axes=axes),
            outputs=(NNEFTensor(graph=nnef_graph, shape=list(reduced_shape), dtype=input.dtype),
                     NNEFTensor(graph=nnef_graph, shape=list(reduced_shape), dtype=input.dtype)))
        NNEFOperation(graph=nnef_graph,
                      name="squeeze",
                      inputs=moments.outputs[0],
                      attribs=dict(axes=axes),
                      outputs=mean)
        NNEFOperation(graph=nnef_graph,
                      name="squeeze",
                      inputs=moments.outputs[1],
                      attribs=dict(axes=axes),
                      outputs=variance)


def convert_reshape(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))

    NNEFOperation(graph=nnef_graph,
                  name="reshape",
                  inputs=input,
                  attribs=dict(shape=list(tf_op.attribs["shape"]), axis_start=0, axis_count=-1),
                  outputs=output)


def convert_flatten(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))

    NNEFOperation(graph=nnef_graph,
                  name="reshape",
                  inputs=input,
                  attribs=dict(shape=[0, -1]),
                  outputs=output)


def convert_expand_dims(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))

    NNEFOperation(graph=nnef_graph,
                  name="unsqueeze",
                  inputs=input,
                  attribs=dict(axes=[converter.nnef_axis(tf_op.attribs["axis"], input.rank + 1)]),
                  outputs=output)


def convert_squeeze(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))

    if tf_op.attribs['axis'] is not None and tf_op.attribs['axis'] != []:
        axes = sorted(converter.nnef_axes(tf_op.attribs['axis'], input.rank))
    else:
        axes = [i for i in range(input.rank) if input.shape[i] == 1]

    NNEFOperation(graph=nnef_graph,
                  name="squeeze",
                  inputs=input,
                  attribs=dict(axes=axes),
                  outputs=output)


def convert_transpose(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))

    if tf_op.attribs['perm'] is not None:
        axes = converter.nnef_axes(tf_op.attribs['perm'], input.rank)
    else:
        axes = list(reversed(range(input.rank)))

    NNEFOperation(graph=nnef_graph,
                  name="transpose",
                  inputs=input,
                  attribs=dict(axes=axes),
                  outputs=output)


def convert_resize_bilinear(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None
    is_nhwc = True

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))

    in_size = converter.spatial_size(input.shape, is_nhwc)
    out_size = tf_op.attribs['size']

    if in_size == out_size:
        NNEFOperation(graph=nnef_graph, name="copy", inputs=input, outputs=output)
        return

    if converter.enable_imprecise_image_resize and out_size[0] < in_size[0] and out_size[1] < in_size[1]:
        print("Warning: Imprecise resize conversion: nearest instead of bilinear downsample")
        convert_resize_nearest_neighbor(converter, tf_op, nnef_graph)
        return

    assert out_size[0] > in_size[0] and out_size[1] > in_size[1], "Bilinear resize must be up-sampling or identity"
    assert out_size[0] % in_size[0] == 0 and out_size[1] % in_size[1] == 0, "Bilinear resize must have integer factor"

    nnef_op = NNEFOperation(
        graph=nnef_graph,
        name="multilinear_upsample",
        inputs=converter.add_nhwc_to_nchw_transpose(nnef_graph, input) if is_nhwc else input,
        attribs=dict(factor=[int(out_size[0] // in_size[0]), int(out_size[1] // in_size[1])],
                     method='aligned' if tf_op.attribs.get('align_corners', False) else 'asymmetric',
                     border='replicate'),
        outputs=converter.create_nchw_intermediate_for_nhwc_output(nnef_graph, output) if is_nhwc else output)

    if is_nhwc:
        converter.add_nchw_to_nhwc_transpose(nnef_graph, nnef_op.output, nhwc_tensor=output)


def convert_resize_bicubic(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    if converter.enable_imprecise_image_resize:
        print("Warning: Imprecise resize conversion: bilinear instead of bicubic")
        convert_resize_bilinear(converter, tf_op, nnef_graph)
        return

    assert False, "Bicubic resize is unsupported in NNEF."


def convert_resize_nearest_neighbor(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    is_nhwc = True

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))

    if tf_op.attribs.get("align_corners", False):
        if converter.enable_imprecise_image_resize:
            print("Warning: Imprecise resize_nearest_neighbor conversion: NNEF does not support align_corners "
                  "for nearest resize")
        else:
            assert False, "NNEF does not support align_corners for nearest resize"

    in_size = converter.spatial_size(input.shape, is_nhwc)
    out_size = tf_op.attribs['size']

    if in_size == out_size:
        NNEFOperation(graph=nnef_graph, name="copy", inputs=input, outputs=output)
        return

    assert ((out_size[0] > in_size[0] and out_size[1] > in_size[1])
            or (out_size[0] < in_size[0] and out_size[1] < in_size[1])), \
        "Nearest resize must not be mixed up/down sampling"

    if out_size[0] > in_size[0]:
        assert out_size[0] % in_size[0] == 0 and out_size[1] % in_size[1] == 0, \
            "Nearest resize must have integer factor"

        nnef_op = NNEFOperation(
            graph=nnef_graph,
            name="nearest_upsample",
            inputs=converter.add_nhwc_to_nchw_transpose(nnef_graph, input) if is_nhwc else input,
            attribs=dict(factor=[int(out_size[0] // in_size[0]), int(out_size[1] // in_size[1])]),
            outputs=converter.create_nchw_intermediate_for_nhwc_output(nnef_graph, output) if is_nhwc else output)

        if is_nhwc:
            converter.add_nchw_to_nhwc_transpose(nnef_graph, nnef_op.output, nhwc_tensor=output)

    else:
        assert in_size[0] % out_size[0] == 0 and in_size[1] % out_size[1] == 0, \
            "Nearest resize must have integer factor"

        nnef_op = NNEFOperation(
            graph=nnef_graph,
            name="nearest_downsample",
            inputs=converter.add_nhwc_to_nchw_transpose(nnef_graph, input) if is_nhwc else input,
            attribs=dict(factor=[int(in_size[0] // out_size[0]), int(in_size[1] // out_size[1])]),
            outputs=converter.create_nchw_intermediate_for_nhwc_output(nnef_graph, output) if is_nhwc else output)

        if is_nhwc:
            converter.add_nchw_to_nhwc_transpose(nnef_graph, nnef_op.output, nhwc_tensor=output)


def convert_resize_area(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    is_nhwc = True

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))

    if tf_op.attribs.get("align_corners", False):
        if converter.enable_imprecise_image_resize:
            print("Warning: Imprecise resize_area conversion: NNEF does not support align_corners")
        else:
            assert False

    in_size = converter.spatial_size(input.shape, is_nhwc)
    out_size = tf_op.attribs['size']

    if in_size == out_size:
        NNEFOperation(graph=nnef_graph, name="copy", inputs=input, outputs=output)
        return

    if converter.enable_imprecise_image_resize and out_size[0] > in_size[0] and out_size[1] > in_size[1]:
        print("Warning: Imprecise resize conversion: nearest instead of area upsample")
        convert_resize_nearest_neighbor(converter, tf_op, nnef_graph)
        return

    assert out_size[0] < in_size[0] and out_size[1] < in_size[1], "Area resize must be down-sampling or identity"

    assert in_size[0] % out_size[0] == 0 and in_size[1] % out_size[1] == 0, \
        "Area resize must have integer factor"

    nnef_op = NNEFOperation(
        graph=nnef_graph,
        name="area_downsample",
        inputs=converter.add_nhwc_to_nchw_transpose(nnef_graph, input) if is_nhwc else input,
        attribs=dict(factor=[int(in_size[0] // out_size[0]), int(in_size[1] // out_size[1])]),
        outputs=converter.create_nchw_intermediate_for_nhwc_output(nnef_graph, output) if is_nhwc else output)

    if is_nhwc:
        converter.add_nchw_to_nhwc_transpose(nnef_graph, nnef_op.output, nhwc_tensor=output)


def convert_resize_images(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    BILINEAR = 0
    NEAREST_NEIGHBOR = 1
    BICUBIC = 2
    AREA = 3

    method = tf_op.attribs["method"]

    if method == BILINEAR:
        convert_resize_bilinear(converter, tf_op, nnef_graph)
    elif method == NEAREST_NEIGHBOR:
        convert_resize_nearest_neighbor(converter, tf_op, nnef_graph)
    elif method == BICUBIC:
        convert_resize_bicubic(converter, tf_op, nnef_graph)
    elif method == AREA:
        convert_resize_area(converter, tf_op, nnef_graph)
    else:
        assert False, "Unsupported image resize method: {}".format(method)


def convert_clip_by_value(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    x, a, b = converter.converted_tensors(tf_op.inputs)
    output = converter.converted_tensor(tf_op.output)

    assert converter.can_broadcast_from_right(a.shape, x.shape)
    assert converter.can_broadcast_from_right(b.shape, x.shape)

    NNEFOperation(
        graph=nnef_graph,
        name="clamp",
        inputs=(x,
                converter.add_unsqueeze(nnef_graph, a, list(range(x.rank - a.rank))) if 0 < a.rank < x.rank else a,
                converter.add_unsqueeze(nnef_graph, b, list(range(x.rank - b.rank))) if 0 < b.rank < x.rank else b),
        outputs=output)


def convert_relu6(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))
    a = converter.create_constant_nnef_tensor(nnef_graph=nnef_graph, data=0.0, dtype=input.dtype)
    b = converter.create_constant_nnef_tensor(nnef_graph=nnef_graph, data=6.0, dtype=input.dtype)

    NNEFOperation(graph=nnef_graph,
                  name="clamp",
                  inputs=(input, a, b),
                  outputs=output)


def convert_slice(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))

    assert input.rank == len(tf_op.attribs["begin"]) == len(tf_op.attribs["size"])

    axes, begin, end = utils.zip_inverse(3, [
        (axis, begin, shape if size == -1 else begin + size)
        for axis, (begin, size, shape)
        in enumerate(zip(tf_op.attribs["begin"], tf_op.attribs["size"], input.shape))
        if not (begin == 0 and size == -1) and not (begin == 0 and size != -1 and begin + size == shape)
    ])

    NNEFOperation(graph=nnef_graph,
                  name="slice",
                  inputs=input,
                  attribs=dict(axes=axes, begin=begin, end=end),
                  outputs=output)


def convert_stack(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    inputs = list(converter.converted_tensors(tf_op.inputs))
    output = converter.converted_tensor(tf_op.output)

    assert len(tf_op.inputs) > 0, "Stacking zero tensors is not supported"

    NNEFOperation(graph=nnef_graph,
                  name="stack",
                  inputs=inputs,
                  attribs=dict(axis=converter.nnef_axis(tf_op.attribs['axis'], inputs[0].rank + 1)),
                  outputs=output)


def convert_unstack(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    input = converter.converted_tensor(tf_op.input)
    outputs = list(converter.converted_tensors(tf_op.outputs))

    num = tf_op.attribs.get('num')
    assert num is None or num == len(outputs), \
        "Num for unstack is {} but it has {} results".format(num, len(outputs))

    NNEFOperation(graph=nnef_graph,
                  name="unstack",
                  inputs=input,
                  attribs=dict(axis=converter.nnef_axis(tf_op.attribs['axis'], input.rank)),
                  outputs=outputs)


def convert_pad(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))

    assert tf_op.attribs["constant_values"] == 0, \
        "Only tf.pad constant_values=0 is supported, got: {}.".format(tf_op.attribs["constant_values"])

    NNEFOperation(graph=nnef_graph,
                  name="pad",
                  inputs=input,
                  attribs=dict(border=converter.nnef_border(tf_op.attribs["mode"]),
                               padding=converter.nnef_padding(padding=tf_op.attribs["paddings"],
                                                              rank=input.rank,
                                                              is_nhwc=None,
                                                              out_spatial=False)),
                  outputs=output)


def convert_tile(converter, tf_op, nnef_graph):
    # type: (Converter, TFOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((tf_op.input, tf_op.output))

    NNEFOperation(graph=nnef_graph,
                  name="tile",
                  inputs=input,
                  attribs=dict(repeats=list(tf_op.attribs["multiples"])),
                  outputs=output)


_StandardConverters = {
    "tf.assign": convert_assign,

    "_conv": partial(generic_convert_conv, is_planewise=False, is_deconv=False),
    "_planewise_conv": partial(generic_convert_conv, is_planewise=True, is_deconv=False),
    "_deconv": partial(generic_convert_conv, is_planewise=False, is_deconv=True),
    "_planewise_deconv": partial(generic_convert_conv, is_planewise=True, is_deconv=True),
    "_separable_conv": convert_separable_conv,

    "_max_pool": partial(generic_convert_pool, target_name="max_pool"),
    "_avg_pool": partial(generic_convert_pool, target_name="avg_pool"),
    "_max_pool_with_index": partial(generic_convert_pool, target_name="max_pool_with_index"),

    "tf.nn.elu": partial(generic_convert_unary, target_name="elu"),
    "tf.nn.relu": partial(generic_convert_unary, target_name="relu"),
    "tf.nn.softplus": partial(generic_convert_unary, target_name="softplus"),
    "tf.identity": partial(generic_convert_unary, target_name="copy"),
    "tf.reciprocal": partial(generic_convert_unary, target_name="rcp"),
    "tf.negative": partial(generic_convert_unary, target_name="neg"),
    "tf.logical_not": partial(generic_convert_unary, target_name="not"),
    "tf.abs": partial(generic_convert_unary, target_name="abs"),
    "tf.sign": partial(generic_convert_unary, target_name="sign"),
    "tf.exp": partial(generic_convert_unary, target_name="exp"),
    "tf.log": partial(generic_convert_unary, target_name="log"),
    "tf.sqrt": partial(generic_convert_unary, target_name="sqrt"),
    "tf.rsqrt": partial(generic_convert_unary, target_name="rsqrt"),
    "tf.square": partial(generic_convert_unary, target_name="sqr"),
    "tf.floor": partial(generic_convert_unary, target_name="floor"),
    "tf.ceil": partial(generic_convert_unary, target_name="ceil"),
    "tf.round": partial(generic_convert_unary, target_name="round"),
    "tf.nn.sigmoid": partial(generic_convert_unary, target_name="sigmoid"),
    "tf.nn.tanh": partial(generic_convert_unary, target_name="tanh"),

    "tf.add": partial(generic_convert_binary, target_name="add"),
    "tf.subtract": partial(generic_convert_binary, target_name="sub"),
    "tf.multiply": partial(generic_convert_binary, target_name="mul"),
    "tf.divide": partial(generic_convert_binary, target_name="div"),
    "tf.pow": partial(generic_convert_binary, target_name="pow"),
    "tf.logical_and": partial(generic_convert_binary, target_name="and"),
    "tf.logical_or": partial(generic_convert_binary, target_name="or"),
    "tf.greater": partial(generic_convert_binary, target_name="gt"),
    "tf.greater_equal": partial(generic_convert_binary, target_name="ge"),
    "tf.less": partial(generic_convert_binary, target_name="lt"),
    "tf.less_equal": partial(generic_convert_binary, target_name="le"),
    "tf.equal": partial(generic_convert_binary, target_name="eq"),
    "tf.not_equal": partial(generic_convert_binary, target_name="ne"),
    "tf.minimum": partial(generic_convert_binary, target_name="min"),
    "tf.maximum": partial(generic_convert_binary, target_name="max"),

    "tf.nn.leaky_relu": convert_leaky_relu,
    "tf.squared_difference": convert_squared_difference,
    "tf.where": convert_where,

    "tf.reduce_sum": partial(generic_convert_reduce, target_name="sum_reduce"),
    "tf.reduce_mean": partial(generic_convert_reduce, target_name="mean_reduce"),
    "tf.reduce_max": partial(generic_convert_reduce, target_name="max_reduce"),
    "tf.reduce_min": partial(generic_convert_reduce, target_name="min_reduce"),
    "tf.argmax": partial(generic_convert_arg_min_max, target_name="argmax_reduce"),
    "tf.argmin": partial(generic_convert_arg_min_max, target_name="argmin_reduce"),

    "tf.nn.lrn": convert_lrn,
    "tf.nn.batch_normalization": convert_batch_normalization,
    "tf.nn.l2_normalize": convert_l2_normalization,
    "tf.matmul": convert_matmul,

    "tf.add_n": convert_add_n,
    "tf.nn.bias_add": convert_bias_add,
    "tf.concat": convert_concat,
    "tf.split": convert_split,
    "tf.nn.softmax": convert_softmax,
    "tf.nn.moments": convert_moments,
    "tf.reshape": convert_reshape,
    "tf.layers.flatten": convert_flatten,
    "tf.expand_dims": convert_expand_dims,
    "tf.squeeze": convert_squeeze,
    "tf.transpose": convert_transpose,
    "tf.image.resize_bilinear": convert_resize_bilinear,
    "tf.image.resize_bicubic": convert_resize_bicubic,
    "tf.image.resize_nearest_neighbor": convert_resize_nearest_neighbor,
    "tf.image.resize_area": convert_resize_area,
    "tf.image.resize_images": convert_resize_images,
    "tf.clip_by_value": convert_clip_by_value,
    "tf.nn.relu6": convert_relu6,
    "tf.slice": convert_slice,
    "tf.stack": convert_stack,
    "tf.unstack": convert_unstack,
    "tf.pad": convert_pad,
    "tf.tile": convert_tile,

    "tf.reduce_any": partial(generic_convert_reduce, target_name="any_reduce"),
    "tf.reduce_all": partial(generic_convert_reduce, target_name="all_reduce"),
    "tf.sin": partial(generic_convert_unary, target_name="sin"),
    "tf.cos": partial(generic_convert_unary, target_name="cos"),
}
