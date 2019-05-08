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
from nnef_tools.conversion.tensorflow import nnef_to_tf_trafos
from nnef_tools.core import utils
from nnef_tools.io.nnef.nnef_graph import NNEFGraph, NNEFOperation, NNEFTensor
from nnef_tools.io.nnef.parser_config import NNEFParserConfig
from nnef_tools.io.tensorflow.tf_graph import TFGraph, TFTensor, TFOperation

# NNEF must be parsed with this before calling nnef_to_tf.Converter on it
ParserConfig = NNEFParserConfig(lowered=[
    "rms_pool",
    "softabs",
    "log2",
    "linear",
    "separable_conv",
    "separable_deconv",
    "l1_normalization",
])

_tf_dtype_by_nnef_dtype = {
    "scalar": "float32",
    "integer": 'int64',
    "logical": 'bool'
}


class Converter(converter.Converter[NNEFTensor, NNEFOperation, NNEFGraph,
                                    TFTensor, TFOperation, TFGraph], ShapeUtils):
    FORMAT_NHWC = "NHWC"
    FORMAT_NCHW = "NCHW"
    FORMAT_SPATIAL = "SPATIAL"

    def __init__(self,
                 prefer_nhwc=True,
                 enable_default_conversion=False,
                 enable_imprecise_image_resize=False,
                 enable_imprecise_padding_border=False,
                 custom_converter_by_op_name=None):
        default_op_converter = convert_default if enable_default_conversion else None
        converters = {}
        converters.update(_DefaultConverters)
        if custom_converter_by_op_name is not None:
            converters.update(custom_converter_by_op_name)
        super(Converter, self).__init__(op_converter_by_name=converters, default_op_converter=default_op_converter)

        self._preferred_format = self.FORMAT_NHWC if prefer_nhwc else self.FORMAT_NCHW
        self._enable_imprecise_image_resize = enable_imprecise_image_resize
        self._enable_imprecise_padding_border = enable_imprecise_padding_border

    @property
    def preferred_format(self):
        return self._preferred_format

    @property
    def prefer_nhwc(self):
        return self._preferred_format == self.FORMAT_NHWC

    def create_graph(self, source_graph):
        # type: (NNEFGraph)->TFGraph
        return TFGraph(source_graph.name)

    def convert_tensors(self, source_graph, target_graph):
        # type: (NNEFGraph, TFGraph)->None
        super(Converter, self).convert_tensors(source_graph, target_graph)

        if any(tensor.quantization is not None for tensor in target_graph.tensors):
            for tensor in target_graph.tensors:
                if tensor.dtype not in ["int32", "uint8"]:
                    tensor.dtype = "uint8"

    def convert_tensor(self, source_tensor, target_graph):
        # type: (NNEFTensor, TFGraph)->TFTensor

        quantization = None
        dtype = self.tf_dtype(source_tensor.dtype)
        if source_tensor.quantization:
            assert source_tensor.quantization.name == 'tflite_quantize'
            quantization = TFTensor.Quantization(min=source_tensor.quantization.attribs['min'],
                                                 max=source_tensor.quantization.attribs['max'],
                                                 scale=source_tensor.quantization.attribs['scale'],
                                                 zero_point=source_tensor.quantization.attribs['zero_point'])
            dtype = "uint8" if source_tensor.quantization.attribs['bits'] == 8 else "int32"
        return TFTensor(graph=target_graph,
                        label=source_tensor.label,  # can be None
                        shape=list(source_tensor.shape),
                        dtype=dtype,
                        data=copy.copy(source_tensor.data),
                        quantization=quantization)

    def convert_graph(self, source_graph):
        # type: (NNEFGraph)->TFGraph
        nnef_to_tf_trafos.pre_conversion_transform(source_graph)
        target_graph = super(Converter, self).convert_graph(source_graph)  # type: TFGraph
        target_graph.generate_missing_names()
        return target_graph

    @property
    def enable_imprecise_image_resize(self):
        return self._enable_imprecise_image_resize

    @property
    def enable_imprecise_padding_border(self):
        return self._enable_imprecise_padding_border

    @staticmethod
    def tf_dtype(nnef_dtype):
        return _tf_dtype_by_nnef_dtype[nnef_dtype]

    @staticmethod
    def create_constant_tf_tensor(tf_graph, data, dtype="float32"):
        if dtype.startswith("float"):
            data = float(data)
        return TFTensor(graph=tf_graph, shape=[], dtype=dtype, data=[data])

    @staticmethod
    def convert_format(shapelike, in_format, out_format, default_elem):
        if in_format == out_format:
            return list(shapelike)

        if in_format == Converter.FORMAT_SPATIAL:
            if out_format == Converter.FORMAT_NCHW:
                return [default_elem, default_elem] + shapelike
            elif out_format == Converter.FORMAT_NHWC:
                return [default_elem] + shapelike + [default_elem]
            else:
                assert False
        elif in_format == Converter.FORMAT_NCHW:
            if out_format == Converter.FORMAT_SPATIAL:
                return shapelike[2:]
            elif out_format == Converter.FORMAT_NHWC:
                return [shapelike[0]] + shapelike[2:] + [shapelike[1]]
            else:
                assert False
        elif in_format == Converter.FORMAT_NHWC:
            if out_format == Converter.FORMAT_SPATIAL:
                return shapelike[1:-1]
            elif out_format == Converter.FORMAT_NCHW:
                return [shapelike[0], shapelike[-1]] + shapelike[1:-1]
            else:
                assert False
        else:
            assert False

    @staticmethod
    def tf_data_format(data_format, rank):
        if rank == 3:
            return "NWC" if data_format == Converter.FORMAT_NHWC else "NCW"
        elif rank == 4:
            return "NHWC" if data_format == Converter.FORMAT_NHWC else "NCHW"
        elif rank == 5:
            return "NDHWC" if data_format == Converter.FORMAT_NHWC else "NCDHW"
        else:
            print("Warning: data format called for rank not in [3, 4, 5]")
            return "NHWC" if data_format == Converter.FORMAT_NHWC else "NCHW"

    @staticmethod
    def tf_border_mode(nnef_border_mode, enable_imprecise_padding_border=False):
        to_tf = {
            'ignore': None,
            'constant': 'CONSTANT',
            'replicate': None,
            'reflect': 'REFLECT',
            'reflect-even': 'SYMMETRIC'
        }

        if to_tf[nnef_border_mode] is None:
            if enable_imprecise_padding_border:
                print("Warning: Border mode {} is not supported in TensorFlow, "
                      "using CONSTANT because enable_imprecise_padding_border was True.".format(nnef_border_mode))
            else:
                assert False, \
                    "Error: Border mode {} is not supported in TensorFlow, " \
                    "use enable_imprecise_padding_border to suppress this error.".format(nnef_border_mode)
            return 'CONSTANT'
        else:
            return to_tf[nnef_border_mode]

    @staticmethod
    def nnef_reshaped_shape(input_shape, reshape_shape):
        for i in range(len(reshape_shape)):
            assert reshape_shape[i] != 0 or i <= len(input_shape), "Invalid input_shape and reshape_shape combination"
        reshape_shape = [input_shape[i] if reshape_shape[i] == 0 else reshape_shape[i] for i in
                         range(len(reshape_shape))]
        if -1 in reshape_shape:
            idx = reshape_shape.index(-1)
            reshape_shape2 = list(reshape_shape)
            reshape_shape2[idx] = 1
            rem = int(np.prod(input_shape)) % int(np.prod(reshape_shape2))
            assert rem == 0, "Invalid input_shape and reshape_shape combination"
            div = int(int(np.prod(input_shape)) / int(np.prod(reshape_shape2)))
            reshape_shape2[idx] = div
            return reshape_shape2
        return reshape_shape

    @staticmethod
    def nnef_flatten_shape(input_shape):
        return [input_shape[0], int(np.prod(input_shape[1:]))]

    @staticmethod
    def has_gt_1(data):
        return utils.has_gt_1(data)

    @staticmethod
    def add_nchw_to_nhwc_transpose(tf_graph, tf_tensor, nhwc_tensor=None):
        assert isinstance(tf_graph, TFGraph)
        assert isinstance(tf_tensor, TFTensor)
        assert nhwc_tensor is None or isinstance(nhwc_tensor, TFTensor)

        if nhwc_tensor is None:
            nhwc_tensor = TFTensor(graph=tf_graph,
                                   shape=Converter.shape_nchw_to_nhwc(tf_tensor.shape),
                                   dtype=tf_tensor.dtype)
        TFOperation(graph=tf_graph,
                    name="tf.transpose",
                    inputs=tf_tensor,
                    attribs=dict(perm=Converter.transpose_axes_nchw_to_nhwc(tf_tensor.rank)),
                    outputs=nhwc_tensor)
        return nhwc_tensor

    @staticmethod
    def add_squeeze(tf_graph, tf_tensor, axes):
        # type: (TFGraph, TFTensor, typing.List[int])->TFTensor
        assert isinstance(tf_graph, TFGraph)
        assert isinstance(tf_tensor, TFTensor)
        return TFOperation(graph=tf_graph,
                           name="tf.squeeze",
                           inputs=tf_tensor,
                           attribs=dict(axis=axes),
                           outputs=TFTensor(graph=tf_graph,
                                            shape=transforms.squeezed_shape(tf_tensor.shape, axes),
                                            dtype=tf_tensor.dtype)).output

    @staticmethod
    def add_nchw_to_hwcn_transpose(tf_graph, tf_tensor, hwcn_tensor=None):
        assert isinstance(tf_graph, TFGraph)
        assert isinstance(tf_tensor, TFTensor)
        assert hwcn_tensor is None or isinstance(hwcn_tensor, TFTensor)

        if hwcn_tensor is None:
            hwcn_tensor = TFTensor(graph=tf_graph,
                                   shape=Converter.shape_nchw_to_hwcn(tf_tensor.shape),
                                   dtype=tf_tensor.dtype)
        TFOperation(graph=tf_graph,
                    name="tf.transpose",
                    inputs=tf_tensor,
                    attribs=dict(perm=Converter.transpose_axes_nchw_to_hwcn(tf_tensor.rank)),
                    outputs=hwcn_tensor)
        return hwcn_tensor

    @staticmethod
    def add_hwcn_to_hwcm_reshape(tf_graph, tf_tensor, hwcm_tensor=None, in_channels=None):
        assert isinstance(tf_graph, TFGraph)
        assert isinstance(tf_tensor, TFTensor)
        assert hwcm_tensor is None or isinstance(hwcm_tensor, TFTensor)
        assert hwcm_tensor is not None or in_channels is not None

        if hwcm_tensor is not None:
            new_shape = hwcm_tensor.shape
        else:
            in_channels = int(in_channels)
            assert in_channels % tf_tensor.shape[-2] == 0
            groups = int(in_channels // tf_tensor.shape[-2])
            assert tf_tensor.shape[-1] % groups == 0
            new_shape = tf_tensor.shape[:-2] + [in_channels, int(tf_tensor.shape[-1] // groups)]
            hwcm_tensor = TFTensor(graph=tf_graph,
                                   shape=list(new_shape),
                                   dtype=tf_tensor.dtype)

        return TFOperation(graph=tf_graph,
                           name="tf.reshape",
                           inputs=tf_tensor,
                           attribs=dict(shape=list(new_shape)),
                           outputs=hwcm_tensor).output

    @staticmethod
    def create_nhwc_intermediate_for_nchw_output(tf_graph, tf_tensor):
        assert isinstance(tf_graph, TFGraph)
        assert isinstance(tf_tensor, TFTensor)

        return TFTensor(graph=tf_graph,
                        shape=Converter.shape_nchw_to_nhwc(tf_tensor.shape),
                        dtype=tf_tensor.dtype,
                        data=tf_tensor.data)

    @staticmethod
    def add_nhwc_to_nchw_transpose(tf_graph, tf_tensor, nchw_tensor=None):
        assert isinstance(tf_graph, TFGraph)
        assert isinstance(tf_tensor, TFTensor)
        assert nchw_tensor is None or isinstance(nchw_tensor, TFTensor)

        if nchw_tensor is None:
            nchw_tensor = TFTensor(graph=tf_graph,
                                   shape=Converter.shape_nhwc_to_nchw(tf_tensor.shape),
                                   dtype=tf_tensor.dtype)

        return TFOperation(graph=tf_graph,
                           name="tf.transpose",
                           inputs=tf_tensor,
                           attribs=dict(perm=Converter.transpose_axes_nhwc_to_nchw(tf_tensor.rank)),
                           outputs=nchw_tensor).output

    @staticmethod
    def create_hwcn_intermediate_for_nchw_output(tf_graph, tf_tensor):
        assert isinstance(tf_graph, TFGraph)
        assert isinstance(tf_tensor, TFTensor)

        return TFTensor(graph=tf_graph,
                        shape=Converter.shape_nchw_to_hwcn(tf_tensor.shape),
                        dtype=tf_tensor.dtype,
                        data=tf_tensor.data)

    @staticmethod
    def add_hwcn_to_nchw_transpose(tf_graph, tf_tensor, nchw_tensor=None):
        assert isinstance(tf_graph, TFGraph)
        assert isinstance(tf_tensor, TFTensor)
        assert nchw_tensor is None or isinstance(nchw_tensor, TFTensor)

        if nchw_tensor is None:
            nchw_tensor = TFTensor(graph=tf_graph,
                                   shape=Converter.shape_hwcn_to_nchw(tf_tensor.shape),
                                   dtype=tf_tensor.dtype)

        return TFOperation(graph=tf_graph,
                           name="tf.transpose",
                           inputs=tf_tensor,
                           attribs=dict(perm=Converter.transpose_axes_hwcn_to_nchw(tf_tensor.rank)),
                           outputs=nchw_tensor).output

    @staticmethod
    def create_hwcm_intermediate_for_nchw_output(tf_graph, tf_tensor, input_channels):
        assert isinstance(tf_graph, TFGraph)
        assert isinstance(tf_tensor, TFTensor)

        return TFTensor(graph=tf_graph,
                        shape=Converter.shape_nchw_to_hwcm(tf_tensor.shape, input_channels),
                        dtype=tf_tensor.dtype,
                        data=tf_tensor.data)

    @staticmethod
    def add_hwcm_to_hwcn_reshape(tf_graph, tf_tensor):
        assert isinstance(tf_graph, TFGraph)
        assert isinstance(tf_tensor, TFTensor)

        new_shape = tf_tensor.shape[:-2] + [1, tf_tensor.shape[-1] * tf_tensor.shape[-2]]
        return TFOperation(graph=tf_graph,
                           name="tf.reshape",
                           inputs=tf_tensor,
                           attribs=dict(shape=list(new_shape)),
                           outputs=TFTensor(graph=tf_graph,
                                            shape=list(new_shape),
                                            dtype=tf_tensor.dtype)).output

    @staticmethod
    def add_unsqueeze(tf_graph, tf_tensor, axes):
        assert isinstance(tf_graph, TFGraph)
        assert isinstance(tf_tensor, TFTensor)

        for axis in sorted(axes):
            tf_tensor = TFOperation(graph=tf_graph,
                                    name="tf.expand_dims",
                                    inputs=tf_tensor,
                                    attribs=dict(axis=axis),
                                    outputs=TFTensor(graph=tf_graph,
                                                     shape=transforms.unsqueezed_shape(tf_tensor.shape, [axis]),
                                                     dtype=tf_tensor.dtype)).output
        return tf_tensor

    @staticmethod
    def tf_zero_value(tf_dtype):
        if "float" in tf_dtype:
            return 0.0
        elif "int" in tf_dtype:
            return 0
        elif "bool" in tf_dtype:
            return False
        assert False, "Unsupported TF dtype: {}".format(tf_dtype)

    @staticmethod
    def tf_addition_op(tf_dtype):
        if "float" in tf_dtype:
            return "tf.add"
        elif "int" in tf_dtype:
            return "tf.add"
        elif "bool" in tf_dtype:
            return "tf.logical_or"
        assert False, "Unsupported TF dtype: {}".format(tf_dtype)


def convert_default(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    print("Warning: Converter of {} is not implemented, doing default conversion.".format(nnef_op.name))

    TFOperation(graph=tf_graph,
                name="nnef_" + nnef_op.name,
                inputs=converter.converted_tensors(nnef_op.inputs),
                attribs=utils.recursive_copy(nnef_op.attribs),
                outputs=converter.converted_tensors(nnef_op.outputs))


def convert_update(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    var, val = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    TFOperation(graph=tf_graph, name="tf.assign", inputs=(var, val), outputs=output)


def convert_reshape(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    new_shape = converter.nnef_reshaped_shape(nnef_op.input.shape, nnef_op.attribs["shape"])
    if nnef_op.input.rank >= 2 and new_shape == converter.nnef_flatten_shape(nnef_op.input.shape):
        TFOperation(graph=tf_graph, name="tf.layers.flatten", inputs=input, outputs=output)
    else:
        TFOperation(graph=tf_graph, name="tf.reshape", inputs=input, attribs=dict(shape=new_shape), outputs=output)


def convert_squeeze(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if nnef_op.attribs["axes"]:
        TFOperation(graph=tf_graph,
                    name="tf.squeeze",
                    inputs=input,
                    attribs=dict(axis=list(nnef_op.attribs["axes"])),
                    outputs=output)
    else:  # axes=[] is special in tf
        TFOperation(graph=tf_graph,
                    name="tf.identity",
                    inputs=input,
                    outputs=output)


def convert_unsqueeze(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    if len(nnef_op.attribs["axes"]) == 1:
        partial_convert_unsqueeze_to_expand_dims(converter, nnef_op, tf_graph)
    else:
        partial_convert_unsqueeze_to_reshape(converter, nnef_op, tf_graph)


def partial_convert_unsqueeze_to_reshape(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    TFOperation(graph=tf_graph,
                name="tf.reshape",
                inputs=input,
                attribs=dict(shape=transforms.unsqueezed_shape(input.shape, nnef_op.attribs["axes"])),
                outputs=output)


def partial_convert_unsqueeze_to_expand_dims(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    assert len(nnef_op.attribs["axes"]) == 1

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    TFOperation(graph=tf_graph,
                name="tf.expand_dims",
                inputs=input,
                attribs=dict(axis=nnef_op.attribs["axes"][0]),
                outputs=output)


def convert_transpose(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    axes = nnef_op.attribs["axes"]

    if len(axes) < nnef_op.input.rank:
        axes = axes + list(range(len(axes), nnef_op.input.rank))

    TFOperation(graph=tf_graph,
                name="tf.transpose",
                inputs=input,
                attribs=dict(perm=axes),
                outputs=output)


def convert_concat(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    inputs = list(converter.converted_tensors(nnef_op.inputs))
    output = converter.converted_tensor(nnef_op.output)

    TFOperation(graph=tf_graph,
                name="tf.concat",
                inputs=inputs,
                attribs=dict(axis=nnef_op.attribs["axis"]),
                outputs=output)


def convert_split(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input = converter.converted_tensor(nnef_op.input)
    outputs = list(converter.converted_tensors(nnef_op.outputs))

    nnef_axis = nnef_op.attribs["axis"]
    nnef_ratios = nnef_op.attribs["ratios"]

    if input.shape[nnef_axis] == 0:
        if len(np.unique(nnef_ratios)) == 1:
            tf_num_or_size_splits = len(nnef_ratios)
        else:
            assert False, "Split with different ratios on an axis ({}) with unspecified size.".format(nnef_axis)
    else:
        part_size = input.shape[nnef_axis] // int(np.sum(nnef_ratios))
        tf_num_or_size_splits = [ratio * part_size for ratio in nnef_ratios]

    TFOperation(graph=tf_graph,
                name="tf.split",
                inputs=input,
                attribs=dict(num_or_size_splits=tf_num_or_size_splits,
                             axis=nnef_axis),
                outputs=outputs)


def convert_bias_add(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    x, y = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    tf_op = TFOperation(graph=tf_graph,
                        name="tf.nn.bias_add",
                        inputs=(converter.add_nchw_to_nhwc_transpose(tf_graph, x) if converter.prefer_nhwc else x,
                                converter.add_squeeze(tf_graph, y, axes=[0])),
                        attribs=dict(data_format=converter.tf_data_format(converter.preferred_format, x.rank)),
                        outputs=(converter.create_nhwc_intermediate_for_nchw_output(tf_graph, output)
                                 if converter.prefer_nhwc else output))

    if converter.prefer_nhwc:
        converter.add_nhwc_to_nchw_transpose(tf_graph, tf_op.output, nchw_tensor=output)


def generic_convert_binary(converter, nnef_op, tf_graph, target_name):
    # type: (Converter, NNEFOperation, TFGraph, str)->None

    x, y = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    TFOperation(graph=tf_graph,
                name=target_name,
                inputs=((converter.add_unsqueeze(tf_graph, x, list(range(y.rank - x.rank, y.rank)))
                         if 0 < x.rank < y.rank else x),
                        (converter.add_unsqueeze(tf_graph, y, list(range(x.rank - y.rank, x.rank)))
                         if 0 < y.rank < x.rank else y)),
                outputs=output)


def generic_convert_unary(converter, nnef_op, tf_graph, target_name):
    # type: (Converter, NNEFOperation, TFGraph, str)->None

    TFOperation(graph=tf_graph,
                name=target_name,
                inputs=converter.converted_tensor(nnef_op.input),
                outputs=converter.converted_tensor(nnef_op.output))


def convert_rsqr(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))
    TFOperation(graph=tf_graph,
                name="tf.pow",
                inputs=(input, converter.create_constant_tf_tensor(tf_graph, -2.0)),
                outputs=output)


def convert_select(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    # TensorFlow does not support broadcast in tf.where

    condition, x, y = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    def fixed(tensor):
        if 0 < tensor.rank < output.rank:
            tensor = converter.add_unsqueeze(tf_graph, tensor, list(range(output.rank - tensor.rank, output.rank)))

        if tensor.shape != output.shape:
            tensor = TFOperation(graph=tf_graph,
                                 name=converter.tf_addition_op(tensor.dtype),
                                 inputs=(tensor, TFTensor(graph=tf_graph,
                                                          shape=list(output.shape),
                                                          dtype=tensor.dtype,
                                                          data=[converter.tf_zero_value(tensor.dtype)])),
                                 outputs=TFTensor(graph=tf_graph,
                                                  shape=list(output.shape),
                                                  dtype=tensor.dtype)).output
        return tensor

    TFOperation(graph=tf_graph,
                name="tf.where",
                inputs=(fixed(condition), fixed(x), fixed(y)),
                outputs=output)


def convert_clamp(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    x, a, b = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    TFOperation(graph=tf_graph,
                name="tf.clip_by_value",
                inputs=(x,
                        (converter.add_unsqueeze(tf_graph, a, list(range(x.rank - a.rank, x.rank)))
                         if 0 < a.rank < x.rank else a),
                        (converter.add_unsqueeze(tf_graph, b, list(range(x.rank - b.rank, x.rank)))
                         if 0 < b.rank < x.rank else b)),
                outputs=output)


def convert_matmul(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    a, b = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    TFOperation(graph=tf_graph,
                name="tf.matmul",
                inputs=(a, b),
                attribs=dict(transpose_a=nnef_op.attribs["transposeA"], transpose_b=nnef_op.attribs["transposeB"]),
                outputs=output)


def convert_conv(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    groups = nnef_op.attribs["groups"]
    input = nnef_op.inputs[0]
    d = input.rank - 2

    if groups == 1:
        if d in [2, 3]:
            partial_convert_conv_to_conv2d_or_conv3d(converter, nnef_op, tf_graph)
        else:
            partial_convert_conv_to_convolution(converter, nnef_op, tf_graph)
    else:
        if groups in [0, input.shape[1]] and d == 2:
            partial_convert_conv_to_deptwise_conv2d(converter, nnef_op, tf_graph)
        else:
            assert False, "Grouped convolutions are only supported if they can be converted to tf.nn.depthwise_conv2d."


def partial_convert_conv_to_conv2d_or_conv3d(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, filter = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    d = input.rank - 2
    assert d in [2, 3]

    tf_op = TFOperation(
        graph=tf_graph,
        name=("tf.nn.conv2d" if d == 2 else "tf.nn.conv3d"),
        inputs=(converter.add_nchw_to_nhwc_transpose(tf_graph, input) if converter.prefer_nhwc else input,
                converter.add_nchw_to_hwcn_transpose(tf_graph, filter)),
        attribs=dict(
            data_format=converter.tf_data_format(converter.preferred_format, input.rank),
            strides=converter.convert_format(
                nnef_op.attribs["stride"] if nnef_op.attribs["stride"] else [1] * d,
                in_format=converter.FORMAT_SPATIAL,
                out_format=converter.preferred_format,
                default_elem=1),
            dilations=converter.convert_format(
                nnef_op.attribs["dilation"] if nnef_op.attribs["dilation"] else [1] * d,
                in_format=converter.FORMAT_SPATIAL,
                out_format=converter.preferred_format,
                default_elem=1),
            padding=nnef_op.attribs["padding"]),
        outputs=(converter.create_nhwc_intermediate_for_nchw_output(tf_graph, output)
                 if converter.prefer_nhwc else output))

    if converter.prefer_nhwc:
        converter.add_nhwc_to_nchw_transpose(tf_graph, tf_op.output, nchw_tensor=output)


def partial_convert_conv_to_convolution(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, filter = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    assert not (converter.has_gt_1(nnef_op.attribs["stride"]) and converter.has_gt_1(nnef_op.attribs["dilation"])), \
        "Custom stride AND dilation is not supported by tf.nn.convolution."

    tf_op = TFOperation(
        graph=tf_graph,
        name="tf.nn.convolution",
        inputs=(converter.add_nchw_to_nhwc_transpose(tf_graph, input) if converter.prefer_nhwc else input,
                converter.add_nchw_to_hwcn_transpose(tf_graph, filter)),
        attribs=dict(data_format=converter.tf_data_format(converter.preferred_format, input.rank),
                     padding=nnef_op.attribs["padding"]),
        outputs=(converter.create_nhwc_intermediate_for_nchw_output(tf_graph, output)
                 if converter.prefer_nhwc else output))

    if converter.prefer_nhwc:
        converter.add_nhwc_to_nchw_transpose(tf_graph, tf_op.output, nchw_tensor=output)

    if utils.has_gt_1(nnef_op.attribs["stride"]):
        tf_op.attribs["strides"] = list(nnef_op.attribs["stride"])

    if utils.has_gt_1(nnef_op.attribs["dilation"]):
        tf_op.attribs["dilation_rate"] = list(nnef_op.attribs["dilation"])


def partial_convert_conv_to_deptwise_conv2d(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, filter = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)
    in_channels = input.shape[1]
    d = input.rank - 2
    assert d == 2

    assert not (utils.has_gt_1(nnef_op.attribs["stride"]) and utils.has_gt_1(nnef_op.attribs["dilation"])), \
        "Custom stride AND dilation is not supported by tf.nn.depthwise_conv2d."

    input = converter.add_nchw_to_nhwc_transpose(tf_graph, input) if converter.prefer_nhwc else input
    filter = converter.add_nchw_to_hwcn_transpose(tf_graph, filter)
    filter = converter.add_hwcn_to_hwcm_reshape(tf_graph, filter, in_channels=in_channels)

    tf_op = TFOperation(
        graph=tf_graph,
        name="tf.nn.depthwise_conv2d",
        inputs=(input, filter),
        attribs=dict(
            data_format=converter.tf_data_format(converter.preferred_format, input.rank),
            padding=nnef_op.attribs["padding"],
            strides=converter.convert_format(nnef_op.attribs["stride"] if nnef_op.attribs["stride"] else [1] * d,
                                             in_format=converter.FORMAT_SPATIAL,
                                             out_format=converter.preferred_format,
                                             default_elem=1)),
        outputs=(converter.create_nhwc_intermediate_for_nchw_output(tf_graph, output)
                 if converter.prefer_nhwc else output))

    if converter.prefer_nhwc:
        converter.add_nhwc_to_nchw_transpose(tf_graph, tf_op.output, nchw_tensor=output)

    if utils.has_gt_1(nnef_op.attribs["dilation"]):
        tf_op.attribs["rate"] = converter.convert_format(shapelike=nnef_op.attribs["dilation"],
                                                         in_format=converter.FORMAT_SPATIAL,
                                                         out_format=converter.preferred_format,
                                                         default_elem=1)


def convert_deconv(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    groups = nnef_op.attribs["groups"]
    input = nnef_op.inputs[0]
    d = input.rank - 2

    is_dilated = utils.has_gt_1(nnef_op.attribs["dilation"])

    if groups == 1:
        if not is_dilated and d in [2, 3]:
            partial_convert_deconv_to_conv2d_transpose_or_conv3d_transpose(converter, nnef_op, tf_graph)
        elif is_dilated and d == 2:
            partial_convert_deconv_to_atrous_conv2d_transpose(converter, nnef_op, tf_graph)
        else:
            assert False, \
                "{} dimensional{} deconv is not supported by TensorFlow.".format(d, " dilated" if is_dilated else "")
    else:
        if groups in [0, input.shape[1]] and d == 2:
            partial_convert_deconv_to_depthwise_conv2d_native_backprop_input(converter, nnef_op, tf_graph)
        else:
            assert False, \
                "Grouped deconvolutions are only supported if they can be " \
                "converted to tf.nn.depthwise_conv2d_native_backprop_input."


def partial_convert_deconv_to_conv2d_transpose_or_conv3d_transpose(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, filter = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)
    d = input.rank - 2
    assert d in [2, 3]

    target_format = converter.FORMAT_NHWC if d == 3 else converter.preferred_format

    assert not nnef_op.attribs["output_shape"] or nnef_op.attribs["output_shape"] == nnef_op.output.shape

    tf_op = TFOperation(
        graph=tf_graph,
        name=("tf.nn.conv2d_transpose" if d == 2 else "tf.nn.conv3d_transpose"),
        inputs=((converter.add_nchw_to_nhwc_transpose(tf_graph, input)
                 if target_format == converter.FORMAT_NHWC else input),
                converter.add_nchw_to_hwcn_transpose(tf_graph, filter)),
        attribs=dict(
            data_format=converter.tf_data_format(target_format, input.rank),
            strides=converter.convert_format(nnef_op.attribs["stride"] if nnef_op.attribs["stride"] else [1] * d,
                                             in_format=converter.FORMAT_SPATIAL,
                                             out_format=target_format,
                                             default_elem=1),
            padding=nnef_op.attribs["padding"],
            output_shape=(converter.shape_nchw_to_nhwc(output.shape)
                          if target_format == converter.FORMAT_NHWC else list(output.shape))),
        outputs=(converter.create_nhwc_intermediate_for_nchw_output(tf_graph, output)
                 if target_format == converter.FORMAT_NHWC else output))

    if target_format == converter.FORMAT_NHWC:
        converter.add_nhwc_to_nchw_transpose(tf_graph, tf_op.output, nchw_tensor=output)


def partial_convert_deconv_to_atrous_conv2d_transpose(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    # Only NHWC is supported

    input, filter = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    d = input.rank - 2
    assert d == 2

    assert len(utils.unique(nnef_op.attribs["dilation"])) <= 1, \
        "Cannot specify different x and y dilation in tf.nn.atrous_conv2d_transpose."
    assert not utils.has_gt_1(nnef_op.attribs["stride"]), \
        "Cannot use stride>1 in tf.nn.atrous_conv2d_transpose."
    assert not nnef_op.attribs["output_shape"] or nnef_op.attribs["output_shape"] == nnef_op.output.shape

    tf_op = TFOperation(
        graph=tf_graph,
        name="tf.nn.atrous_conv2d_transpose",
        inputs=(converter.add_nchw_to_nhwc_transpose(tf_graph, input),
                converter.add_nchw_to_hwcn_transpose(tf_graph, filter)),
        attribs=dict(rate=nnef_op.attribs["dilation"][0] if nnef_op.attribs["dilation"] else 1,
                     padding=nnef_op.attribs["padding"],
                     output_shape=converter.shape_nchw_to_nhwc(output.shape)),
        outputs=converter.create_nhwc_intermediate_for_nchw_output(tf_graph, output))

    converter.add_nhwc_to_nchw_transpose(tf_graph, tf_op.output, nchw_tensor=output)


def partial_convert_deconv_to_depthwise_conv2d_native_backprop_input(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, filter = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)
    in_channels = output.shape[1]

    d = input.rank - 2
    assert d == 2
    assert not nnef_op.attribs["output_shape"] or nnef_op.attribs["output_shape"] == nnef_op.output.shape
    assert not (utils.has_gt_1(nnef_op.attribs["stride"]) and utils.has_gt_1(nnef_op.attribs["dilation"])), \
        "Custom stride AND dilation is not supported by tf.nn.atrous_conv2d_transpose."

    input = converter.add_nchw_to_nhwc_transpose(tf_graph, input) if converter.prefer_nhwc else input
    filter = converter.add_nchw_to_hwcn_transpose(tf_graph, filter)
    filter = converter.add_hwcn_to_hwcm_reshape(tf_graph, filter, in_channels=in_channels)

    tf_op = TFOperation(
        graph=tf_graph,
        name="tf.nn.depthwise_conv2d_native_backprop_input",
        inputs=(filter, input),
        attribs=dict(
            data_format=converter.tf_data_format(converter.preferred_format, input.rank),
            strides=converter.convert_format(nnef_op.attribs["stride"] if nnef_op.attribs["stride"] else [1] * d,
                                             in_format=converter.FORMAT_SPATIAL,
                                             out_format=converter.preferred_format,
                                             default_elem=1),
            padding=nnef_op.attribs["padding"],
            input_sizes=converter.shape_nchw_to_nhwc(output.shape) if converter.prefer_nhwc else list(output.shape)),
        outputs=(converter.create_nhwc_intermediate_for_nchw_output(tf_graph, output)
                 if converter.prefer_nhwc else output))

    if converter.prefer_nhwc:
        converter.add_nhwc_to_nchw_transpose(tf_graph, tf_op.output, nchw_tensor=output)

    if utils.has_gt_1(nnef_op.attribs["dilation"]):
        tf_op.attribs["dilations"] = converter.convert_format(
            nnef_op.attribs["dilation"] if nnef_op.attribs["dilation"] else [1] * d,
            in_format=converter.FORMAT_SPATIAL,
            out_format=converter.preferred_format,
            default_elem=1)


def generic_convert_pooling(converter, nnef_op, tf_graph, target_name):
    # type: (Converter, NNEFOperation, TFGraph, str)->None

    assert not utils.has_gt_1(nnef_op.attribs["dilation"]), "Dilated pool is not supported in TensorFlow"

    input = converter.converted_tensor(nnef_op.input)
    outputs = converter.converted_tensors(nnef_op.outputs)

    size, stride, padding = utils.get_dict_items(nnef_op.attribs, "size", "stride", "padding")

    tf_op = TFOperation(
        graph=tf_graph,
        name=target_name,
        inputs=converter.add_nchw_to_nhwc_transpose(tf_graph, input) if converter.prefer_nhwc else input,
        attribs=dict(
            data_format=converter.tf_data_format(converter.preferred_format, input.rank),
            ksize=converter.shape_nchw_to_nhwc(size) if converter.prefer_nhwc else list(size),
            strides=converter.convert_format(stride if stride else [1] * input.rank,
                                             in_format=converter.FORMAT_NCHW,
                                             out_format=converter.preferred_format,
                                             default_elem=1),
            padding=padding),
        outputs=tuple(converter.create_nhwc_intermediate_for_nchw_output(tf_graph, output)
                      if converter.prefer_nhwc else output
                      for output in outputs))
    if converter.prefer_nhwc:
        for op_output, output in zip(tf_op.outputs, outputs):
            converter.add_nhwc_to_nchw_transpose(tf_graph, op_output, output)


def convert_max_pool_with_index(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    # always nhwc
    assert not utils.has_gt_1(nnef_op.attribs["dilation"]), "Dilated pool is not supported in TensorFlow"

    size, stride, padding = utils.get_dict_items(nnef_op.attribs, "size", "stride", "padding")

    input = converter.converted_tensor(nnef_op.input)
    outputs = converter.converted_tensors(nnef_op.outputs)

    tf_op = TFOperation(
        graph=tf_graph,
        name="tf.nn.max_pool_with_argmax",
        inputs=converter.add_nchw_to_nhwc_transpose(tf_graph, input),
        attribs=dict(
            ksize=converter.shape_nchw_to_nhwc(size),
            strides=converter.shape_nchw_to_nhwc(stride) if stride else [1] * input.rank,
            padding=padding),
        outputs=tuple(converter.create_nhwc_intermediate_for_nchw_output(tf_graph, output) for output in outputs))

    for op_output, output in zip(tf_op.outputs, outputs):
        converter.add_nhwc_to_nchw_transpose(tf_graph, op_output, output)


def convert_argmax_pool(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    # always nhwc
    assert not utils.has_gt_1(nnef_op.attribs["dilation"]), "Dilated pool is not supported in TensorFlow"

    size, stride, padding = utils.get_dict_items(nnef_op.attribs, "size", "stride", "padding")

    input = converter.converted_tensor(nnef_op.input)
    index = converter.converted_tensor(nnef_op.output)

    index_tmp = converter.create_nhwc_intermediate_for_nchw_output(tf_graph, index)
    output_tmp = TFTensor(graph=tf_graph, shape=index_tmp.shape, dtype="float32")

    tf_op = TFOperation(
        graph=tf_graph,
        name="tf.nn.max_pool_with_argmax",
        inputs=converter.add_nchw_to_nhwc_transpose(tf_graph, input),
        attribs=dict(
            ksize=converter.shape_nchw_to_nhwc(size),
            strides=converter.shape_nchw_to_nhwc(stride) if stride else [1] * input.rank,
            padding=padding),
        outputs=(output_tmp, index_tmp))

    converter.add_nhwc_to_nchw_transpose(tf_graph, index_tmp, index)


def generic_convert_upsample_downsample(converter, nnef_op, tf_graph, target_name, is_downsample):
    # type: (Converter, NNEFOperation, TFGraph, str, bool)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))
    factors = nnef_op.attribs["factor"]

    # always nhwc
    tf_op = TFOperation(
        graph=tf_graph,
        name=target_name,
        inputs=converter.add_nchw_to_nhwc_transpose(tf_graph, input),
        attribs=dict(size=[(int(i / f) if is_downsample else int(i * f))
                           for i, f in zip(input.shape[2:], factors)]),
        outputs=converter.create_nhwc_intermediate_for_nchw_output(tf_graph, output))

    converter.add_nhwc_to_nchw_transpose(tf_graph, tf_op.output, output)

    if nnef_op.name == "multilinear_upsample":
        if nnef_op.attribs["border"].lower() != "replicate":
            if converter.enable_imprecise_image_resize:
                print("Warning: border={} is unsupported in multilinear_upsample, "
                      "using replicate, because enable_imprecise_image_resize was True"
                      .format(nnef_op.attribs["border"]))
            else:
                assert False, "Error: border={} is unsupported in multilinear_upsample. " \
                              "Use enable_imprecise_image_resize=True, to suppress this error." \
                    .format(nnef_op.attribs["border"])
        if nnef_op.attribs["method"].lower() not in ["aligned", "asymmetric"]:
            if converter.enable_imprecise_image_resize:
                print("Warning: method={} is unsupported in multilinear_upsample, "
                      "using align_corners=False, because enable_imprecise_image_resize was True"
                      .format(nnef_op.attribs["method"]))
            else:
                assert False, "Error: method={} is unsupported in multilinear_upsample. " \
                              "Use enable_imprecise_image_resize=True, to suppress this error." \
                    .format(nnef_op.attribs["method"])

        tf_op.attribs["align_corners"] = (nnef_op.attribs["method"].lower() == 'aligned')


def generic_convert_reduce(converter, nnef_op, tf_graph, target_name, target_name_if_normalize=None):
    # type: (Converter, NNEFOperation, TFGraph, str, typing.Optional[str])->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if nnef_op.attribs.get("normalize"):
        assert target_name_if_normalize
        target_name = target_name_if_normalize

    TFOperation(graph=tf_graph,
                name=target_name,
                inputs=input,
                attribs=dict(axis=list(nnef_op.attribs["axes"]),
                             keepdims=True),
                outputs=output)


def generic_convert_argminmax_reduce(converter, nnef_op, tf_graph, target_name):
    # type: (Converter, NNEFOperation, TFGraph, str)->None

    assert len(nnef_op.attribs["axes"]) == 1, "{} is only supported for one axis".format(nnef_op.name)
    axis = nnef_op.attribs["axes"][0]

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    argminmax = TFOperation(graph=tf_graph,
                            name=target_name,
                            inputs=input,
                            attribs=dict(axis=axis),
                            outputs=TFTensor(graph=tf_graph,
                                             shape=transforms.squeezed_shape(input.shape, [axis],
                                                                             can_squeeze_not_one=True),
                                             dtype=converter.tf_dtype("integer")))

    TFOperation(graph=tf_graph,
                name="tf.expand_dims",
                inputs=argminmax.output,
                attribs=dict(axis=axis),
                outputs=output)


def convert_moments(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input = converter.converted_tensor(nnef_op.input)
    mean, variance = converter.converted_tensors(nnef_op.outputs)

    TFOperation(graph=tf_graph,
                name="tf.nn.moments",
                inputs=input,
                attribs=dict(axes=list(nnef_op.attribs["axes"]),
                             keep_dims=True),
                outputs=(mean, variance))


def convert_softmax(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    assert len(nnef_op.attribs["axes"]) == 1, "{} is only supported for one axis".format(nnef_op.name)
    axis = nnef_op.attribs["axes"][0]

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    TFOperation(graph=tf_graph,
                name="tf.nn.softmax",
                inputs=input,
                attribs=dict(axis=axis),
                outputs=output)


def convert_prelu(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, alpha = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    TFOperation(graph=tf_graph,
                name="tf.nn.leaky_relu",
                inputs=(input, alpha),
                outputs=output)


def convert_leaky_relu(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input = converter.converted_tensor(nnef_op.input)
    alpha = converter.create_constant_tf_tensor(tf_graph=tf_graph, data=nnef_op.attribs["alpha"])
    output = converter.converted_tensor(nnef_op.output)

    TFOperation(graph=tf_graph,
                name="tf.nn.leaky_relu",
                inputs=(input, alpha),
                outputs=output)


def convert_local_response_normalization(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    # Probably only NHWC is supported

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    nnefsize = converter.shape_nchw_to_nhwc(nnef_op.attribs["size"])

    assert len(nnefsize) >= 2, "Argument 'size' of local_response_normalization must have at least 2 elements"
    depth_size = nnefsize[-1]
    nnefsize[-1] = 1
    assert not utils.has_gt_1(nnefsize), \
        "local_response_normalization only supported when only the last element of 'size' is > 1"

    depth_radius = (depth_size - 1) / 2
    alpha = nnef_op.attribs["alpha"] / depth_size

    tf_op = TFOperation(graph=tf_graph,
                        name="tf.nn.lrn",
                        inputs=converter.add_nchw_to_nhwc_transpose(tf_graph, input),
                        attribs=dict(depth_radius=int(depth_radius),
                                     bias=nnef_op.attribs["bias"],
                                     alpha=alpha,
                                     beta=nnef_op.attribs["beta"]),
                        outputs=converter.create_nhwc_intermediate_for_nchw_output(tf_graph, output))

    converter.add_nhwc_to_nchw_transpose(tf_graph, tf_op.output, output)


def convert_l2_normalization(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    # TODO epsilon and bias are not really the same, is this a problem?
    TFOperation(graph=tf_graph,
                name="tf.nn.l2_normalize",
                inputs=input,
                attribs=dict(axis=nnef_op.attribs["axes"],
                             epsilon=nnef_op.attribs["bias"]),
                outputs=output)


def convert_batch_normalization(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, mean, variance, offset, scale = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    if converter.prefer_nhwc:
        def add_squeeze_or_transpose(tf_tensor):
            if tf_tensor.rank == 2 and tf_tensor.shape[0] == 1:
                return converter.add_squeeze(tf_graph, tf_tensor, axes=[0])
            else:
                return converter.add_nchw_to_nhwc_transpose(tf_graph, tf_tensor)

        tf_op = TFOperation(graph=tf_graph,
                            name="tf.nn.batch_normalization",
                            inputs=(converter.add_nchw_to_nhwc_transpose(tf_graph, input),
                                    add_squeeze_or_transpose(mean),
                                    add_squeeze_or_transpose(variance),
                                    add_squeeze_or_transpose(offset),
                                    add_squeeze_or_transpose(scale)),
                            attribs=dict(variance_epsilon=nnef_op.attribs["epsilon"]),
                            outputs=converter.create_nhwc_intermediate_for_nchw_output(tf_graph, output))

        converter.add_nhwc_to_nchw_transpose(tf_graph, tf_op.output, output)
    else:
        def add_unsqueeze(tf_tensor):
            return converter.add_unsqueeze(tf_graph, tf_tensor, list(range(input.rank - tf_tensor.rank, input.rank)))

        TFOperation(graph=tf_graph,
                    name="tf.nn.batch_normalization",
                    inputs=(input,
                            add_unsqueeze(mean) if 0 < mean.rank < input.rank else mean,
                            add_unsqueeze(variance) if 0 < variance.rank < input.rank else variance,
                            add_unsqueeze(offset) if 0 < offset.rank < input.rank else offset,
                            add_unsqueeze(scale) if 0 < scale.rank < input.rank else scale),
                    attribs=dict(variance_epsilon=nnef_op.attribs["epsilon"]),
                    outputs=output)


def convert_copy_n(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    assert len(nnef_op.outputs) == nnef_op.attribs["times"], "copy_n must have 'times' outputs"

    input = converter.converted_tensor(nnef_op.input)
    outputs = converter.converted_tensors(nnef_op.outputs)

    for output in outputs:
        TFOperation(graph=tf_graph,
                    name="tf.identity",
                    inputs=input,
                    outputs=output)


def convert_add_n(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    inputs = list(converter.converted_tensors(nnef_op.inputs))
    output = converter.converted_tensor(nnef_op.output)

    TFOperation(graph=tf_graph, name="tf.add_n", inputs=inputs, outputs=output)


def convert_slice(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    axes = nnef_op.attribs["axes"]
    begin = nnef_op.attribs["begin"]
    end = nnef_op.attribs["end"]

    size = [
        -1 if e == -1 else e - b
        for b, e in zip(begin, end)
    ]

    tfbegin, tfsize = utils.zip_inverse(2, [
        (begin[axes.index(i)], size[axes.index(i)]) if i in axes else (0, -1)
        for i in range(input.rank)
    ])

    TFOperation(graph=tf_graph,
                name="tf.slice",
                inputs=input,
                attribs=dict(begin=tfbegin,
                             size=tfsize),
                outputs=output)


def convert_stack(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    inputs = list(converter.converted_tensors(nnef_op.inputs))
    output = converter.converted_tensor(nnef_op.output)

    TFOperation(graph=tf_graph,
                name="tf.stack",
                inputs=inputs,
                attribs=dict(axis=nnef_op.attribs["axis"]),
                outputs=output)


def convert_unstack(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input = converter.converted_tensor(nnef_op.input)
    outputs = converter.converted_tensors(nnef_op.outputs)

    TFOperation(graph=tf_graph,
                name="tf.unstack",
                inputs=input,
                attribs=dict(num=len(outputs),
                             axis=nnef_op.attribs["axis"]),
                outputs=outputs)


def convert_box(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if (all(s == 1 for s in nnef_op.attribs["size"])
            and all(s == 1 for s in nnef_op.attribs["stride"])
            and all(d == 1 for d in nnef_op.attribs["dilation"])):
        tf_op = TFOperation(graph=tf_graph,
                            name="tf.pad",
                            inputs=input,
                            outputs=output)

        tf_border_mode = converter.tf_border_mode(nnef_op.attribs["border"],
                                                  converter.enable_imprecise_padding_border)
        if tf_border_mode != 'CONSTANT':
            tf_op.attribs["mode"] = tf_border_mode
    else:
        assert False, "Box is not yet fully supported, only for padding"


def convert_copy(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    TFOperation(graph=tf_graph,
                name="tf.identity",
                inputs=input,
                outputs=output)


def convert_box_to_pad(converter, nnef_op, tf_graph):
    # type: (Converter, NNEFOperation, TFGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    assert nnef_op.attribs["size"] == [1] * input.rank, "Only 1x1 box is supported (which is used to simulate padding)"

    tf_op = TFOperation(graph=tf_graph,
                        name="tf.pad",
                        inputs=input,
                        attribs=dict(paddings=list(nnef_op.attribs["padding"]), mode='CONSTANT', constant_values=0),
                        outputs=output)

    tf_pad_mode = converter.tf_border_mode(nnef_op.attribs["border"], converter.enable_imprecise_padding_border)

    if tf_pad_mode != "CONSTANT":
        tf_op.attribs["mode"] = tf_pad_mode


_DefaultConverters = {
    "update": convert_update,
    "reshape": convert_reshape,
    "squeeze": convert_squeeze,
    "unsqueeze": convert_unsqueeze,
    "transpose": convert_transpose,
    "concat": convert_concat,
    "split": convert_split,
    "add": partial(generic_convert_binary, target_name="tf.add"),
    "sub": partial(generic_convert_binary, target_name="tf.subtract"),
    "mul": partial(generic_convert_binary, target_name="tf.multiply"),
    "div": partial(generic_convert_binary, target_name="tf.divide"),
    "pow": partial(generic_convert_binary, target_name="tf.pow"),
    "lt": partial(generic_convert_binary, target_name="tf.less"),
    "gt": partial(generic_convert_binary, target_name="tf.greater"),
    "le": partial(generic_convert_binary, target_name="tf.less_equal"),
    "ge": partial(generic_convert_binary, target_name="tf.greater_equal"),
    "eq": partial(generic_convert_binary, target_name="tf.equal"),
    "ne": partial(generic_convert_binary, target_name="tf.not_equal"),
    "and": partial(generic_convert_binary, target_name="tf.logical_and"),
    "or": partial(generic_convert_binary, target_name="tf.logical_or"),
    "min": partial(generic_convert_binary, target_name="tf.minimum"),
    "max": partial(generic_convert_binary, target_name="tf.maximum"),
    "exp": partial(generic_convert_unary, target_name="tf.exp"),
    "log": partial(generic_convert_unary, target_name="tf.log"),
    "abs": partial(generic_convert_unary, target_name="tf.abs"),
    "sign": partial(generic_convert_unary, target_name="tf.sign"),
    "rcp": partial(generic_convert_unary, target_name="tf.reciprocal"),
    "neg": partial(generic_convert_unary, target_name="tf.negative"),
    "floor": partial(generic_convert_unary, target_name="tf.floor"),
    "ceil": partial(generic_convert_unary, target_name="tf.ceil"),
    "round": partial(generic_convert_unary, target_name="tf.round"),
    "sqr": partial(generic_convert_unary, target_name="tf.square"),
    "sqrt": partial(generic_convert_unary, target_name="tf.sqrt"),
    "rsqrt": partial(generic_convert_unary, target_name="tf.rsqrt"),
    "not": partial(generic_convert_unary, target_name="tf.logical_not"),
    "sigmoid": partial(generic_convert_unary, target_name="tf.nn.sigmoid"),
    "tanh": partial(generic_convert_unary, target_name="tf.nn.tanh"),
    "elu": partial(generic_convert_unary, target_name="tf.nn.elu"),
    "relu": partial(generic_convert_unary, target_name="tf.nn.relu"),
    "softplus": partial(generic_convert_unary, target_name="tf.nn.softplus"),
    "rsqr": convert_rsqr,
    "select": convert_select,
    "clamp": convert_clamp,
    "matmul": convert_matmul,
    "conv": convert_conv,
    "deconv": convert_deconv,
    "argmax_pool": convert_argmax_pool,
    "max_pool": partial(generic_convert_pooling, target_name="tf.nn.max_pool"),
    "avg_pool": partial(generic_convert_pooling, target_name="tf.nn.avg_pool"),
    "max_pool_with_index": convert_max_pool_with_index,
    "nearest_downsample": partial(generic_convert_upsample_downsample,
                                  target_name="tf.image.resize_nearest_neighbor",
                                  is_downsample=True),
    "area_downsample": partial(generic_convert_upsample_downsample,
                               target_name="tf.image.resize_area",
                               is_downsample=True),
    "nearest_upsample": partial(generic_convert_upsample_downsample,
                                target_name="tf.image.resize_nearest_neighbor",
                                is_downsample=False),
    "multilinear_upsample": partial(generic_convert_upsample_downsample,
                                    target_name="tf.image.resize_bilinear",
                                    is_downsample=False),
    "sum_reduce": partial(generic_convert_reduce, target_name="tf.reduce_sum", target_name_if_normalize="tf.reduce_mean"),
    "min_reduce": partial(generic_convert_reduce, target_name="tf.reduce_min"),
    "max_reduce": partial(generic_convert_reduce, target_name="tf.reduce_max"),
    "mean_reduce": partial(generic_convert_reduce, target_name="tf.reduce_mean"),
    "argmax_reduce": partial(generic_convert_argminmax_reduce, target_name="tf.argmax"),
    "argmin_reduce": partial(generic_convert_argminmax_reduce, target_name="tf.argmin"),
    "moments": convert_moments,
    "softmax": convert_softmax,
    "leaky_relu": convert_leaky_relu,
    "prelu": convert_prelu,
    "local_response_normalization": convert_local_response_normalization,
    "l2_normalization": convert_l2_normalization,
    "batch_normalization": convert_batch_normalization,
    "copy_n": convert_copy_n,
    "add_n": convert_add_n,
    "slice": convert_slice,
    "stack": convert_stack,
    "unstack": convert_unstack,
    "copy": convert_copy,
    "_bias_add": convert_bias_add,
    "box": convert_box_to_pad,
    # unsupported: debox, sample, desample, local_mean_normalization, local_variance_normalization,
    # local_contrast_normalization, linear_quantize, logarithmic_quantize, binary_quantize, ternary_quantize
}
