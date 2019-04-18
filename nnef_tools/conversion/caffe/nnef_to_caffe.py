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
import math
from functools import partial

import numpy as np
import six
import typing

from nnef_tools.conversion import converter as _converter
from nnef_tools.conversion.caffe import nnef_to_caffe_passes
from nnef_tools.core import utils
from nnef_tools.io.caffe.caffe_graph import *
from nnef_tools.io.nnef.nnef_graph import *
from nnef_tools.io.nnef.parser_config import NNEFParserConfig


class Converter(_converter.Converter[NNEFTensor, NNEFOperation, NNEFGraph,
                                     CaffeTensor, CaffeOperation, CaffeGraph]):

    def __init__(self,
                 enable_default_conversion=False,
                 custom_converter_by_op_name=None):
        converters = {}
        converters.update(_StandardConverters)
        if custom_converter_by_op_name is not None:
            converters.update(custom_converter_by_op_name)
        default_op_converter = convert_default if enable_default_conversion else None

        super(Converter, self).__init__(op_converter_by_name=converters,
                                        default_op_converter=default_op_converter)

        self.displayed_warnings = set()

    def create_graph(self, source_graph):
        # type:(NNEFGraph)->CaffeGraph
        return CaffeGraph(name=source_graph.name)

    def convert_tensor(self, source_tensor, target_graph):
        # type: (NNEFTensor, CaffeGraph)->CaffeTensor
        return CaffeTensor(graph=target_graph,
                           name=source_tensor.name,
                           shape=list(source_tensor.shape),
                           dtype=None,
                           data=copy.copy(source_tensor.get_numpy_array()))

    def convert_graph(self, source_graph):
        # type: (NNEFGraph)->CaffeGraph
        nnef_to_caffe_passes.pre_conversion_pass(source_graph)
        target_graph = super(Converter, self).convert_graph(source_graph)  # type: CaffeGraph
        nnef_to_caffe_passes.post_conversion_pass(target_graph)
        self.create_input_operation(target_graph)
        target_graph.generate_missing_names()
        return target_graph

    def create_input_operation(self, caffe_graph):
        # type: (CaffeGraph)->None
        CaffeOperation(graph=caffe_graph,
                       name='Input',
                       inputs=tuple(),
                       outputs=list(caffe_graph.inputs),
                       attribs=dict(shape=[list(t.shape) for t in caffe_graph.inputs]))

    @staticmethod
    def check_unique_variable_or_constant(nnef_tensor):
        # type: (NNEFTensor)->None
        assert isinstance(nnef_tensor, NNEFTensor)
        if len(nnef_tensor.consumers) > 1:
            raise utils.NNEFToolsException("Using a variable by multiple operations is unsupported in Caffe: {}"
                                           .format(nnef_tensor.name))
        if not (nnef_tensor.is_constant or nnef_tensor.is_variable):
            raise utils.NNEFToolsException("Using calculated weights is unsupported in Caffe: {}"
                                           .format(nnef_tensor.name))

    @staticmethod
    def is_unique_variable_or_constant(nnef_tensor):
        # type: (NNEFTensor)->bool
        assert isinstance(nnef_tensor, NNEFTensor)
        return len(nnef_tensor.consumers) == 1 and (nnef_tensor.is_constant or nnef_tensor.is_variable)

    @staticmethod
    def caffe_pad(nnef_padding):
        if not all(p == q for p, q in nnef_padding):
            raise utils.NNEFToolsException(
                "Asymmetric padding not supported, padding={}".format(nnef_padding))
        return [p for p, _q in nnef_padding]

    @staticmethod
    def caffe_set_channel_only_shape(bias, out_channels):
        assert isinstance(bias, CaffeTensor)
        assert all(dim == 1 or idx == 1 for idx, dim in enumerate(bias.shape))
        assert bias.rank < 2 or bias.shape[1] in [1, out_channels]
        bias.shape = [out_channels]
        bias.data = np.array(bias.data, dtype=np.float32).reshape((-1)) * np.ones(bias.shape, dtype=np.float32)

    @staticmethod
    def get_pool_output_size(h, k, p, q, s, ceil_mode=True):
        if ceil_mode:
            return int(math.ceil(float(h + p + q - k) / s)) + 1
        return int(math.floor(float(h + p + q - k) / s)) + 1

    @staticmethod
    def caffe_pool_pad(
            spatial_upscaled_size,
            spatial_filter_size,
            spatial_padding,
            spatial_stride
    ):
        nnef_output_shape = []
        caffe_output_shape = []

        for i, (h, k, (p, q), s) in enumerate(zip(spatial_upscaled_size,
                                                  spatial_filter_size,
                                                  spatial_padding,
                                                  spatial_stride)):
            nnef_output_size = Converter.get_pool_output_size(h, k, p, q, s, ceil_mode=False)
            caffe_output_size = Converter.get_pool_output_size(h, k, p, q, s, ceil_mode=True)
            nnef_output_shape.append(nnef_output_size)
            caffe_output_shape.append(caffe_output_size)

        ceil_mode = (nnef_output_shape == caffe_output_shape)

        spatial_padding_old = list(spatial_padding)
        for i, (h, k, (p, q), s) in enumerate(zip(spatial_upscaled_size,
                                                  spatial_filter_size,
                                                  spatial_padding_old,
                                                  spatial_stride)):
            asymmetric_size = Converter.get_pool_output_size(h, k, p, q, s, ceil_mode=ceil_mode)
            symmetric_size = Converter.get_pool_output_size(h, k, p, p, s, ceil_mode=ceil_mode)

            if asymmetric_size == symmetric_size:
                spatial_padding[i] = (p, p)

        if any(left != right for left, right in spatial_padding):
            raise utils.NNEFToolsException("Asymmetric pad is not supported, padding={}".format(spatial_padding))

        return [max(left, right) for left, right in spatial_padding], ceil_mode

    @staticmethod
    def fill_bilinear(caffe_tensor):
        # type: (CaffeTensor)->None
        # From: https://caffe.berkeleyvision.org/doxygen/filler_8hpp_source.html
        assert caffe_tensor.rank == 4
        N, C, H, W = caffe_tensor.shape
        assert H == W
        f = int(math.ceil(W / 2.0))
        c = float((2 * f - 1 - f % 2) / (2.0 * f))
        x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))
        data = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        caffe_tensor.data = np.tile(data.reshape((1, 1, H, W)), (N, C, 1, 1))


def convert_default(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    print("Warning: Converter of {} is not implemented, doing default conversion.".format(nnef_op.name))

    CaffeOperation(graph=caffe_graph,
                   name=nnef_op.name,
                   inputs=converter.converted_tensors(nnef_op.inputs),
                   attribs=utils.recursive_transform(nnef_op.attribs, lambda x: x if x is not None else "None"),
                   outputs=converter.converted_tensors(nnef_op.outputs))


def UNSUPPORTED(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    if converter.default_op_converter is not None:
        converter.default_op_converter(converter, nnef_op, caffe_graph)
    else:
        raise utils.NNEFToolsException('NNEF to Caffe: Unsupported op: {}'.format(nnef_op.name))


def UNNEEDED(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    assert False, "This should not be called!"


def NONATOMIC(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    assert False, "This should not be called!"


def convert_conv(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None
    nnef_input, nnef_filter, nnef_bias = nnef_op.inputs
    converter.check_unique_variable_or_constant(nnef_filter)
    converter.check_unique_variable_or_constant(nnef_bias)

    input, filter, bias = converter.converted_tensors((nnef_input, nnef_filter, nnef_bias))
    output = converter.converted_tensor(nnef_op.output)
    has_bias = np.any(nnef_bias.get_numpy_array() != 0.0)
    if has_bias:
        converter.caffe_set_channel_only_shape(bias, out_channels=output.shape[1])

    DEFAULT_ENGINE = 0
    CAFFE_ENGINE = 1
    _CUDNN_ENGINE = 2

    CaffeOperation(graph=caffe_graph,
                   name="Convolution",
                   inputs=(input, filter, bias) if has_bias else (input, filter),
                   outputs=output,
                   attribs=dict(
                       kernel_size=filter.shape[2:],
                       num_output=filter.shape[0],
                       bias_term=has_bias,
                       pad=converter.caffe_pad(nnef_op.attribs["padding"]),
                       stride=list(nnef_op.attribs['stride']),
                       dilation=list(nnef_op.attribs['dilation']),
                       group=nnef_op.attribs['groups'],
                       engine=DEFAULT_ENGINE if nnef_op.attribs["groups"] == 1 and input.rank == 4 else CAFFE_ENGINE,
                   ))


def convert_deconv(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None
    nnef_input, nnef_filter, nnef_bias = nnef_op.inputs
    converter.check_unique_variable_or_constant(nnef_filter)
    converter.check_unique_variable_or_constant(nnef_bias)

    input, filter, bias = converter.converted_tensors((nnef_input, nnef_filter, nnef_bias))
    output = converter.converted_tensor(nnef_op.output)
    has_bias = np.any(nnef_bias.get_numpy_array() != 0.0)
    if has_bias:
        converter.caffe_set_channel_only_shape(bias, out_channels=nnef_op.attribs['output_shape'][1])

    DEFAULT_ENGINE = 0
    CAFFE_ENGINE = 1
    _CUDNN_ENGINE = 2

    CaffeOperation(graph=caffe_graph,
                   name="Deconvolution",
                   inputs=(input, filter, bias) if has_bias else (input, filter),
                   outputs=output,
                   attribs=dict(
                       kernel_size=filter.shape[2:],
                       num_output=filter.shape[1] * nnef_op.attribs['groups'],
                       bias_term=has_bias,
                       pad=converter.caffe_pad(nnef_op.attribs["padding"]),
                       stride=list(nnef_op.attribs['stride']),
                       dilation=list(nnef_op.attribs['dilation']),
                       group=nnef_op.attribs['groups'],
                       engine=DEFAULT_ENGINE if nnef_op.attribs["groups"] == 1 and input.rank == 4 else CAFFE_ENGINE,
                   ))


def convert_multilinear_upsample(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    if (nnef_op.attribs['method'] != 'symmetric' or nnef_op.attribs['border'] != 'constant'
            or nnef_op.attribs['factor'][0] != nnef_op.attribs['factor'][1]):
        raise utils.NNEFToolsException(
            "Multilinear upsample is only supported if method, border == 'symmetric', 'constant', "
            "and factor[0] == factor[1]")

    input = converter.converted_tensor(nnef_op.input)
    output = converter.converted_tensor(nnef_op.output)

    factor = nnef_op.attribs["factor"][0]
    kernel_size = 2 * factor - factor % 2

    filter = CaffeTensor(graph=caffe_graph,
                         shape=[input.shape[1], 1, kernel_size, kernel_size])
    converter.fill_bilinear(filter)

    DEFAULT_ENGINE = 0
    CAFFE_ENGINE = 1
    _CUDNN_ENGINE = 2

    CaffeOperation(graph=caffe_graph,
                   name="Deconvolution",
                   inputs=(input, filter),
                   outputs=output,
                   attribs=dict(
                       weight_filler='bilinear',
                       kernel_size=(kernel_size, kernel_size),
                       num_output=output.shape[1],
                       bias_term=False,
                       pad=(int(math.ceil((factor - 1) / 2.0)),) * 2,
                       stride=(factor, factor),
                       dilation=(1, 1),
                       group=input.shape[1],
                       engine=DEFAULT_ENGINE if input.shape[1] == 1 else CAFFE_ENGINE,
                   ))


def generic_convert_unary(converter, nnef_op, caffe_graph, target_name):
    # type: (Converter, NNEFOperation, CaffeGraph, str)->None
    CaffeOperation(graph=caffe_graph,
                   name=target_name,
                   inputs=converter.converted_tensor(nnef_op.input),
                   outputs=converter.converted_tensor(nnef_op.output))


def convert_elu(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None
    CaffeOperation(graph=caffe_graph,
                   name='ELU',
                   inputs=converter.converted_tensor(nnef_op.input),
                   outputs=converter.converted_tensor(nnef_op.output),
                   attribs=dict(alpha=nnef_op.attribs.get('_alpha', 1.0)))


def convert_exp(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None
    CaffeOperation(graph=caffe_graph,
                   name='Exp',
                   inputs=converter.converted_tensor(nnef_op.input),
                   outputs=converter.converted_tensor(nnef_op.output),
                   attribs=dict(base=-1.0))  # -1 means e


def convert_log(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None
    CaffeOperation(graph=caffe_graph,
                   name='Log',
                   inputs=converter.converted_tensor(nnef_op.input),
                   outputs=converter.converted_tensor(nnef_op.output),
                   attribs=dict(base=-1.0))  # -1 means e


def convert_log2(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None
    CaffeOperation(graph=caffe_graph,
                   name='Log',
                   inputs=converter.converted_tensor(nnef_op.input),
                   outputs=converter.converted_tensor(nnef_op.output),
                   attribs=dict(base=2.0))


def convert_threshold(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None
    CaffeOperation(graph=caffe_graph,
                   name='Threshold',
                   inputs=converter.converted_tensor(nnef_op.input),
                   outputs=converter.converted_tensor(nnef_op.output),
                   attribs=dict(threshold=nnef_op.attribs['threshold']))


def convert_prelu(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    nnef_input, nnef_alpha = nnef_op.inputs
    converter.check_unique_variable_or_constant(nnef_alpha)
    input, alpha = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    if (alpha.is_variable or alpha.is_constant) and nnef_alpha.shape == []:
        CaffeOperation(graph=caffe_graph,
                       name='ReLU',
                       inputs=input,
                       outputs=output,
                       attribs=dict(negative_slope=float(alpha.data)))
    else:
        channel_shared = (alpha.count == 1 and input.shape[1] != 1)
        if channel_shared:
            alpha.shape = []
            alpha.data = alpha.data.reshape(alpha.shape)
        else:
            if not all(s == 1 or i == 1 for i, s in enumerate(alpha.shape)):
                raise utils.NNEFToolsException("prelu is only supported with channel-only or 0D alpha, got: {}"
                                               .format(alpha.shape))
            alpha.shape = [alpha.shape[1]]
            alpha.data = alpha.data.reshape(alpha.shape)

        CaffeOperation(graph=caffe_graph,
                       name='PReLU',
                       inputs=(input, alpha),
                       outputs=output,
                       attribs=dict(channel_shared=channel_shared))


def generic_convert_pool(converter, nnef_op, caffe_graph, pool):
    # type: (Converter, NNEFOperation, CaffeGraph, int)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if input.rank != 4:
        raise utils.NNEFToolsException("Pooling input '{}': rank != 4".format(input.name))

    if (any(s != 1 for s in nnef_op.attribs['size'][:2])
            or any(p != (0, 0) for p in nnef_op.attribs['padding'][:2])
            or any(s != 1 for s in nnef_op.attribs['stride'][:2])
            or any(d != 1 for d in nnef_op.attribs['dilation'])):
        raise utils.NNEFToolsException("Pooling is only supported in spatial dimensions and without dilation.")

    if nnef_op.name == "max_pool":
        if nnef_op.attribs["border"] != "ignore" and any(p != (0, 0) for p in nnef_op.attribs['padding']):
            raise utils.NNEFToolsException("max_pool is only supported with border: ignore or no padding")
    elif nnef_op.name == "avg_pool":
        if nnef_op.attribs["border"] != "constant" and any(p != (0, 0) for p in nnef_op.attribs['padding']):
            raise utils.NNEFToolsException("avg_pool is only supported with border: constant or no padding")

    pad, ceil_mode = converter.caffe_pool_pad(spatial_upscaled_size=input.shape[2:],
                                              spatial_filter_size=nnef_op.attribs['size'][2:],
                                              spatial_padding=nnef_op.attribs['padding'][2:],
                                              spatial_stride=nnef_op.attribs['stride'][2:])

    caffe_op = CaffeOperation(graph=caffe_graph,
                              name="Pooling",
                              inputs=input,
                              outputs=output,
                              attribs=dict(kernel_size=nnef_op.attribs['size'][2:],
                                           pool=pool,
                                           pad=pad,
                                           stride=nnef_op.attribs['stride'][2:]))

    if not ceil_mode:
        print("Warning: Pooling '{}' has to use round_mode: FLOOR".format(caffe_op.output.name))
        caffe_op.attribs['round_mode'] = 1  # FLOOR


def convert_max_reduce(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if input.rank != 4:
        raise utils.NNEFToolsException("Pooling input '{}': rank != 4".format(input.name))

    if sorted(nnef_op.attribs["axes"]) != [2, 3]:
        raise utils.NNEFToolsException("{}: not convertible to global pooling.".format(output.name))

    CaffeOperation(graph=caffe_graph,
                   name="Pooling",
                   inputs=input,
                   outputs=output,
                   attribs=dict(pool=0, global_pooling=True))


def convert_sum_or_mean_reduce(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    assert nnef_op.name in ["sum_reduce", "mean_reduce"]

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    axes = sorted(nnef_op.attribs['axes'])
    axis = min(axes)

    if axes != list(range(axis, input.rank)):
        raise utils.NNEFToolsException(
            "Reduce is only convertible if axes are in the form [rank-1, rank-2, ...]".format(input.name))

    if nnef_op.name == "mean_reduce" or nnef_op.attribs["normalize"]:
        operation = 4
    else:
        operation = 1

    reduction_op = CaffeOperation(graph=caffe_graph,
                                  name="Reduction",
                                  inputs=input,
                                  outputs=CaffeTensor(graph=caffe_graph,
                                                      shape=[1 if a in axes else d for a, d in enumerate(input.shape)]),
                                  attribs=dict(operation=operation, axis=axis, coeff=1.0))

    CaffeOperation(graph=caffe_graph,
                   name="Reshape",
                   inputs=reduction_op.output,
                   outputs=output,
                   attribs=dict(shape=list(output.shape)))


def convert_local_response_normalization(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None
    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    ACROSS_CHANNELS = 0
    WITHIN_CHANNEL = 1

    size = nnef_op.attribs['size']
    assert len(size) == input.rank

    if input.rank >= 2 and all(i == 1 or s == 1 for i, s in enumerate(size)) and size[1] % 2 == 1:
        CaffeOperation(graph=caffe_graph,
                       name="LRN",
                       inputs=input,
                       outputs=output,
                       attribs=dict(local_size=size[1],
                                    alpha=nnef_op.attribs["alpha"],
                                    beta=nnef_op.attribs["beta"],
                                    k=nnef_op.attribs['bias'],
                                    norm_region=ACROSS_CHANNELS))
    elif input.rank >= 3 and size[:2] == [1, 1] and size[2:] == [size[2]] * (input.rank - 2) and size[2] % 2 == 1:
        CaffeOperation(graph=caffe_graph,
                       name="LRN",
                       inputs=input,
                       outputs=output,
                       attribs=dict(local_size=size[2],
                                    alpha=nnef_op.attribs["alpha"],
                                    beta=nnef_op.attribs["beta"],
                                    k=nnef_op.attribs['bias'],
                                    norm_region=WITHIN_CHANNEL))
    else:
        raise utils.NNEFToolsException("This local_response_normalization can not be converted to Caffe.")


def convert_concat(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    inputs = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    CaffeOperation(graph=caffe_graph,
                   name="Concat",
                   inputs=inputs,
                   outputs=output,
                   attribs=dict(axis=nnef_op.attribs["axis"]))


def convert_matmul(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    nnef_A, nnef_B = nnef_op.inputs
    converter.check_unique_variable_or_constant(nnef_B)

    A, B = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    if nnef_op.attribs["transposeA"]:
        raise utils.NNEFToolsException("matmul: transposeA not yet supported")

    if not all(s == 1 for s in B.shape[:-2]):
        raise utils.NNEFToolsException("batch matmul is only supported if the leading dimensions of B are ones")

    B.shape = B.shape[-2:]
    B.data = B.data.reshape(B.shape)

    CaffeOperation(graph=caffe_graph,
                   name="InnerProduct",
                   inputs=(A, B),
                   outputs=output,
                   attribs=dict(num_output=output.shape[-1],
                                bias_term=False,
                                axis=-1,
                                transpose=not nnef_op.attribs['transposeB']
                                ))


def convert_linear(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    nnef_A, nnef_B, nnef_C = nnef_op.inputs
    converter.check_unique_variable_or_constant(nnef_B)
    converter.check_unique_variable_or_constant(nnef_C)

    A, B, C = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    if A.rank != 2 or B.rank != 2 or C.rank != 2:
        raise utils.NNEFToolsException("linear: only 2 dimensional is supported")

    converter.caffe_set_channel_only_shape(C, output.shape[1])

    CaffeOperation(graph=caffe_graph,
                   name="InnerProduct",
                   inputs=(A, B, C),
                   outputs=output,
                   attribs=dict(num_output=output.shape[-1],
                                bias_term=True,
                                axis=-1,
                                transpose=False))


def convert_reshape(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))
    shape = nnef_op.attribs["shape"]

    if sum(s == -1 for s in shape) == 1:
        axis = shape.index(-1)
        end_axis = axis + (input.rank - len(shape))
        if end_axis == input.rank - 1:
            end_axis = -1

        tmp_shape = list(shape)
        for _ in range(input.rank - len(shape)):
            tmp_shape.insert(axis, -1)

        if all(s == t or t in [0, -1] for s, t in zip(input.shape, tmp_shape)):
            CaffeOperation(graph=caffe_graph,
                           name="Flatten",
                           inputs=input,
                           outputs=output,
                           attribs=dict(axis=axis,
                                        end_axis=end_axis))
            return

    CaffeOperation(graph=caffe_graph,
                   name="Reshape",
                   inputs=input,
                   outputs=output,
                   attribs=dict(shape=shape))


def convert_softmax(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if len(nnef_op.attribs["axes"]) != 1:
        raise utils.NNEFToolsException("softmax: len(axes) != 1")

    CaffeOperation(graph=caffe_graph,
                   name="Softmax",
                   inputs=input,
                   outputs=output,
                   attribs=dict(axis=nnef_op.attribs['axes'][0]))


def generic_convert_binary(
        converter,  # type: Converter
        nnef_op,  # type: NNEFOperation
        caffe_graph,  # type: CaffeGraph
        is_commutative=False,  # type: bool
        eltwise_name=None,  # type: typing.Optional[str]
        eltwise_axis_needed=False,  # type: typing.Optional[bool]
        eltwise_operation=None,  # type: typing.Optional[int]
        eltwise_coeff=None,  # type: typing.Optional[typing.List[float]]
        learnable_name=None,  # type: typing.Optional[str]
        scalar_name=None,  # type: typing.Optional[str]
        scalar_arg_name=None,  # type: typing.Optional[str]
        scalar_transform=None,  # type: typing.Optional[typing.Callable[[float], float]]
        scalar_name_left=None,  # type: typing.Optional[str]
        scalar_arg_name_left=None,  # type: typing.Optional[str]
):
    # type: (...)->None

    if not scalar_transform:
        scalar_transform = lambda x: x

    def make_learnable(input, variable, output):
        # type: (CaffeTensor, CaffeTensor, CaffeTensor)->None

        axis = -1
        for j, (i, v) in enumerate(zip(input.shape, variable.shape)):
            if i == v and v != 1:
                axis = j
                break

        if axis == -1:
            if not utils.has_gt_1(variable.shape):
                if variable.rank >= 2 and input.rank >= 2:
                    axis = 1
                else:
                    axis = 0
            else:
                raise utils.NNEFToolsException("Cannot convert binary op: Variable and input shapes are incompatible")

        num_axes = variable.rank - axis

        variable.shape = variable.shape[axis:]
        variable.data = variable.data.reshape(variable.shape)

        CaffeOperation(graph=caffe_graph,
                       name=learnable_name,
                       inputs=(input, variable),
                       outputs=output,
                       attribs=dict(axis=axis,
                                    num_axes=num_axes))

    def make_eltwise(input1, input2, output):
        # type: (CaffeTensor, CaffeTensor, CaffeTensor)->None

        if eltwise_axis_needed:
            axis = -1
            for index, (i, j) in enumerate(zip(input1.shape, input2.shape)):
                if i == j and j != 1:
                    axis = index
                    break

            if axis == -1:
                if not utils.has_gt_1(input2.shape):
                    if input2.rank >= 2 and input1.rank >= 2:
                        axis = 1
                    else:
                        axis = 0
                else:
                    raise utils.NNEFToolsException(
                        "Cannot convert binary op: Input shapes are incompatible")
            if axis > 0:
                input2 = CaffeOperation(graph=caffe_graph,
                                        name="Reshape",
                                        inputs=input2,
                                        outputs=CaffeTensor(graph=caffe_graph,
                                                            shape=input2.shape[axis:]),
                                        attribs=dict(shape=input2.shape[axis:])).output
        else:
            axis = None

        eltwise = CaffeOperation(graph=caffe_graph,
                                 name=eltwise_name,
                                 inputs=(input1, input2),
                                 outputs=output)
        if eltwise_operation is not None:
            eltwise.attribs['operation'] = eltwise_operation
        if eltwise_coeff is not None:
            eltwise.attribs['coeff'] = eltwise_coeff
        if axis is not None:
            eltwise.attribs['axis'] = axis

    def make_scalar(input, output, scalar):
        # type: (CaffeTensor, CaffeTensor, typing.Union[bool, int, float])->None
        CaffeOperation(graph=caffe_graph,
                       name=scalar_name,
                       inputs=input,
                       outputs=output,
                       attribs={scalar_arg_name: scalar_transform(scalar)})

    def make_scalar_left(input, output, scalar):
        # type: (CaffeTensor, CaffeTensor, typing.Union[bool, int, float])->None
        CaffeOperation(graph=caffe_graph,
                       name=scalar_name_left,
                       inputs=input,
                       outputs=output,
                       attribs={scalar_arg_name_left: scalar})

    nnef_x, nnef_y = nnef_op.inputs
    x, y = converter.converted_tensors((nnef_x, nnef_y))
    z = converter.converted_tensor(nnef_op.output)

    if (learnable_name
            and nnef_x.data is None
            and converter.is_unique_variable_or_constant(nnef_y)
            and nnef_y.shape != []):
        make_learnable(input=x, variable=y, output=z)
    elif (learnable_name
          and is_commutative
          and nnef_y.data is None
          and converter.is_unique_variable_or_constant(nnef_x)
          and nnef_y.shape != []):
        make_learnable(input=y, variable=x, output=z)
    elif (scalar_name
          and nnef_x.data is None
          and nnef_y.data is not None and nnef_y.shape == []):
        make_scalar(input=x, output=z, scalar=float(nnef_y.get_numpy_array()))
    elif (scalar_name_left
          and nnef_y.data is None
          and nnef_x.data is not None and nnef_x.shape == []):
        make_scalar_left(input=y, output=z, scalar=float(nnef_x.get_numpy_array()))
    elif (scalar_name
          and is_commutative
          and nnef_y.data is None
          and nnef_x.data is not None and nnef_x.shape == []):
        make_scalar(input=y, output=z, scalar=float(nnef_x.get_numpy_array()))
    elif (eltwise_name is not None
          and nnef_x.data is None
          and nnef_y.data is None):
        make_eltwise(input1=x, input2=y, output=z)
    else:
        raise utils.NNEFToolsException("Unsupported binary operation: {}".format(nnef_op))


def convert_argmax_reduce(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if set(nnef_op.attribs["axes"]) == set(range(1, input.rank)):
        axis = None
    elif len(nnef_op.attribs["axes"]) == 1:
        axis = nnef_op.attribs["axes"][0]
    else:
        raise utils.NNEFToolsException("Unsupported axes arg: {}".format(nnef_op.attribs["axes"]))

    CaffeOperation(graph=caffe_graph,
                   name="ArgMax",
                   inputs=input,
                   outputs=output,
                   attribs=dict(axis=axis))


def convert_slice(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    axes = nnef_op.attribs["axes"]  # type: typing.List[int]
    begin = nnef_op.attribs["begin"]  # type: typing.List[int]
    end = nnef_op.attribs["end"]  # type: typing.List[int]
    axis = min(axes)
    offset = [begin[axes.index(a)] if a in axes else 0 for a in range(axis, input.rank)]
    new_shape = [end[axes.index(a)] - begin[axes.index(a)]
                 if a in axes
                 else (input.shape[a] if a >= axis else 1)
                 for a in range(input.rank)]

    reference = CaffeTensor(graph=caffe_graph, shape=new_shape)
    caffe_graph.inputs = list(caffe_graph.inputs) + [reference]
    CaffeOperation(graph=caffe_graph,
                   name='Crop',
                   inputs=(input, reference),
                   outputs=output,
                   attribs=dict(axis=axis, offset=offset))


def convert_split(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    input = converter.converted_tensor(nnef_op.input)
    outputs = converter.converted_tensors(nnef_op.outputs)
    axis = nnef_op.attribs['axis']
    sizes = [output.shape[axis] for output in outputs]
    slice_points = utils.prefix_sum(sizes)[:-1]
    CaffeOperation(graph=caffe_graph,
                   name='Slice',
                   inputs=(input,),
                   outputs=outputs,
                   attribs=dict(axis=axis, slice_point=slice_points))


def generic_convert_squeeze_unsqueeze(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    CaffeOperation(graph=caffe_graph,
                   name="Reshape",
                   inputs=input,
                   outputs=output,
                   attribs=dict(shape=output.shape))


def convert_batch_normalization(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    nnef_input, nnef_mean, nnef_variance, nnef_offset, nnef_scale = nnef_op.inputs

    converter.check_unique_variable_or_constant(nnef_mean)
    converter.check_unique_variable_or_constant(nnef_variance)
    converter.check_unique_variable_or_constant(nnef_offset)
    converter.check_unique_variable_or_constant(nnef_scale)

    if np.all(nnef_offset.get_numpy_array() == 0.0) and np.all(nnef_scale.get_numpy_array() == 1.0):
        input, mean, variance = converter.converted_tensors(nnef_op.inputs[:3])
        output = converter.converted_tensor(nnef_op.output)
        converter.caffe_set_channel_only_shape(mean, output.shape[1])
        converter.caffe_set_channel_only_shape(variance, output.shape[1])
        scale_factor = CaffeTensor(graph=caffe_graph, shape=[1], data=np.array([1.0], dtype=np.float32))
        CaffeOperation(graph=caffe_graph,
                       name="BatchNorm",
                       inputs=(input, mean, variance, scale_factor),
                       outputs=output,
                       attribs=dict(eps=nnef_op.attribs['epsilon']))

    else:
        input, mean, variance, offset, scale = converter.converted_tensors(nnef_op.inputs)
        output = converter.converted_tensor(nnef_op.output)
        converter.caffe_set_channel_only_shape(mean, output.shape[1])
        converter.caffe_set_channel_only_shape(variance, output.shape[1])
        converter.caffe_set_channel_only_shape(offset, output.shape[1])
        converter.caffe_set_channel_only_shape(scale, output.shape[1])
        scale_factor = CaffeTensor(graph=caffe_graph, shape=[1], data=np.array([1.0], dtype=np.float32))

        batch_norm = CaffeOperation(graph=caffe_graph,
                                    name="BatchNorm",
                                    inputs=(input, mean, variance, scale_factor),
                                    outputs=CaffeTensor(graph=caffe_graph, shape=list(input.shape)),
                                    attribs=dict(eps=nnef_op.attribs['epsilon']))

        CaffeOperation(graph=caffe_graph,
                       name="Scale",
                       inputs=(batch_norm.output, scale, offset),
                       outputs=output,
                       attribs=dict(bias_term=True))


def convert_copy_n(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    input = converter.converted_tensor(nnef_op.input)
    outputs = converter.converted_tensors(nnef_op.outputs)

    CaffeOperation(graph=caffe_graph,
                   name="Split",
                   inputs=input,
                   outputs=outputs)


def convert_local_contrast_normalization(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if input.rank <= 1:
        raise utils.NNEFToolsException("local_contrast_normalization: rank < 1 not supported")

    size_across_channels = list(input.shape)
    size_per_channel = list(input.shape)
    size_per_channel[1] = 1

    if nnef_op.attribs['size'] not in (size_across_channels, size_per_channel):
        raise utils.NNEFToolsException("local_contrast_normalization: size must be [N, C, H, W] or [N, 1, H, W]")

    CaffeOperation(graph=caffe_graph,
                   name="MVN",
                   inputs=input,
                   outputs=output,
                   attribs=dict(normalize_variance=True,
                                across_channels=nnef_op.attribs['size'] == size_across_channels,
                                eps=nnef_op.attribs['epsilon']))


def convert_local_mean_normalization(converter, nnef_op, caffe_graph):
    # type: (Converter, NNEFOperation, CaffeGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if input.rank <= 1:
        raise utils.NNEFToolsException("local_mean_normalization: rank < 1 not supported")

    size_across_channels = list(input.shape)
    size_per_channel = list(input.shape)
    size_per_channel[1] = 1

    if nnef_op.attribs['size'] not in (size_across_channels, size_per_channel):
        raise utils.NNEFToolsException("local_mean_normalization: size must be [N, C, H, W] or [N, 1, H, W]")

    CaffeOperation(graph=caffe_graph,
                   name="MVN",
                   inputs=input,
                   outputs=output,
                   attribs=dict(normalize_variance=False,
                                across_channels=nnef_op.attribs['size'] == size_across_channels))


_StandardConverters = {
    'separable_conv': NONATOMIC,
    'leaky_relu': NONATOMIC,
    'sqr': NONATOMIC,
    'conv': convert_conv,
    'deconv': convert_deconv,
    'prelu': convert_prelu,
    "max_pool": partial(generic_convert_pool, pool=0),
    "avg_pool": partial(generic_convert_pool, pool=1),
    "max_reduce": convert_max_reduce,
    "sum_reduce": convert_sum_or_mean_reduce,
    "mean_reduce": convert_sum_or_mean_reduce,
    "local_response_normalization": convert_local_response_normalization,
    "concat": convert_concat,
    "matmul": convert_matmul,
    "reshape": convert_reshape,
    "softmax": convert_softmax,
    "add": partial(generic_convert_binary,
                   is_commutative=True,
                   eltwise_name="Bias",
                   eltwise_axis_needed=True,
                   learnable_name="Bias",
                   scalar_name="Power",
                   scalar_arg_name="shift"),
    "sub": partial(generic_convert_binary,
                   scalar_name="Power",
                   scalar_arg_name="shift",
                   scalar_transform=lambda x: -x),
    "mul": partial(generic_convert_binary,
                   is_commutative=True,
                   eltwise_name="Scale",
                   eltwise_axis_needed=True,
                   learnable_name="Scale",
                   scalar_name="Power",
                   scalar_arg_name="scale"),
    "div": partial(generic_convert_binary,
                   scalar_name="Power",
                   scalar_arg_name="scale",
                   scalar_transform=lambda x: 1.0 / x),
    "max": partial(generic_convert_binary,
                   is_commutative=True,
                   eltwise_name="Eltwise",
                   eltwise_operation=2),
    "pow": partial(generic_convert_binary,
                   scalar_name="Power",
                   scalar_arg_name="power",
                   scalar_name_left="Exp",
                   scalar_arg_name_left="base"),
    "exp": convert_exp,
    "log": convert_log,
    "log2": convert_log2,
    "elu": convert_elu,
    "relu": partial(generic_convert_unary, target_name="ReLU"),
    "sigmoid": partial(generic_convert_unary, target_name="Sigmoid"),
    "abs": partial(generic_convert_unary, target_name="AbsVal"),
    "tanh": partial(generic_convert_unary, target_name="TanH"),
    "softplus": partial(generic_convert_unary, target_name="BNLL"),
    "multilinear_upsample": convert_multilinear_upsample,
    "argmax_reduce": convert_argmax_reduce,
    "squeeze": generic_convert_squeeze_unsqueeze,
    "unsqueeze": generic_convert_squeeze_unsqueeze,
    "batch_normalization": convert_batch_normalization,
    "slice": convert_slice,
    "split": convert_split,
    "copy_n": convert_copy_n,
    "local_contrast_normalization": convert_local_contrast_normalization,
    "local_mean_normalization": convert_local_mean_normalization,
    "linear": convert_linear,
    "_threshold": convert_threshold,
}

# NNEF must be parsed with this before calling nnef_to_caffe.Converter on it
ParserConfig = NNEFParserConfig(lowered=[k for k, v in six.iteritems(_StandardConverters) if v is NONATOMIC])
