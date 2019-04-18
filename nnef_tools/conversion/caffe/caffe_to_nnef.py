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
import re
from functools import partial

import typing

from nnef_tools.conversion import converter as _converter
from nnef_tools.conversion.caffe import caffe_to_nnef_passes
from nnef_tools.core import utils, graph_utils
from nnef_tools.io.caffe.caffe_graph import *
from nnef_tools.io.nnef.nnef_graph import *
from nnef_tools.shape_inference import shape_inference as infer


class Converter(_converter.Converter[CaffeTensor, CaffeOperation, CaffeGraph,
                                     NNEFTensor, NNEFOperation, NNEFGraph]):

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
        self.name_generator = utils.NameGenerator()

    def create_graph(self, source_graph):
        # type:(CaffeGraph)->NNEFGraph
        return NNEFGraph(name=self.identifier_compatible_name(source_graph.name))

    def convert_tensor(self, source_tensor, target_graph):
        # type: (CaffeTensor, NNEFGraph)->NNEFTensor
        name = self.name_generator.get_new_name(self.identifier_compatible_name(source_tensor.name))
        dtype = ('integer' if source_tensor.producer and source_tensor.producer.name == 'ArgMax' else 'scalar')
        return NNEFTensor(graph=target_graph,
                          name=name,
                          shape=list(source_tensor.shape),
                          dtype=dtype,
                          data=copy.copy(source_tensor.data),
                          label=name if source_tensor.is_variable else None)

    def convert_graph(self, source_graph):
        # type: (CaffeGraph)->NNEFGraph
        caffe_to_nnef_passes.pre_conversion_pass(source_graph)
        target_graph = super(Converter, self).convert_graph(source_graph)  # type: NNEFGraph
        graph_utils.remove_unreachable(target_graph)
        target_graph.generate_missing_names()
        return target_graph

    @staticmethod
    def identifier_compatible_name(name):
        return re.sub(r'\W', '_', name) if name is not None else None

    @staticmethod
    def nnef_padding(pad):
        # type: (typing.List[int])->typing.List[typing.Tuple[int, int]]
        return list(zip(pad, pad))

    @staticmethod
    def nnef_set_channel_only_shape(tensor):
        assert isinstance(tensor, NNEFTensor)
        assert tensor.rank == 0 or tensor.rank == 1 or (tensor.rank == 4 and tensor.shape[:3] == [1, 1, 1])
        if tensor.rank == 1:
            tensor.shape = [1, tensor.shape[0]]
            tensor.data = tensor.data.reshape(tensor.shape)
        elif tensor.rank == 4:
            tensor.shape = [1, tensor.shape[3]]
            tensor.data = tensor.data.reshape(tensor.shape)

    @staticmethod
    def nnef_add_dims_to_left(tensor, n):
        assert isinstance(tensor, NNEFTensor)
        tensor.shape = [1] * n + tensor.shape
        tensor.data = tensor.data.reshape(tensor.shape)

    @staticmethod
    def get_pooling_right_padding(h, k, p, q, s):
        # type: (int, int, int, int, int)->int
        a = int(math.ceil(float(h + p + q - k) / s))
        return s * a + k - h - p

    @staticmethod
    def nnef_axis(axis, rank):
        # type: (int, int)->int
        return axis if axis >= 0 else rank + axis


def convert_default(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    print("Warning: Converter of {} is not implemented, doing default conversion.".format(caffe_op.name))

    NNEFOperation(graph=nnef_graph,
                  name=caffe_op.name,
                  inputs=converter.converted_tensors(caffe_op.inputs),
                  attribs=utils.recursive_transform(caffe_op.attribs, lambda x: x if x is not None else "None"),
                  outputs=converter.converted_tensors(caffe_op.outputs))


def convert_input(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None
    # Not creating an operation for this
    pass


def convert_convolution(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None
    assert len(caffe_op.inputs) in [2, 3]
    assert bool(caffe_op.attribs['bias_term']) == (len(caffe_op.inputs) == 3)

    input, filter = converter.converted_tensors(caffe_op.inputs[:2])

    if caffe_op.attribs['bias_term']:
        bias = converter.converted_tensor(caffe_op.inputs[2])
        converter.nnef_set_channel_only_shape(bias)
    else:
        bias = NNEFTensor(graph=nnef_graph, shape=[], dtype='scalar', data=[0.0])

    output = converter.converted_tensor(caffe_op.output)

    NNEFOperation(graph=nnef_graph,
                  name='conv',
                  inputs=(input, filter, bias),
                  outputs=output,
                  attribs=dict(border='constant',
                               padding=converter.nnef_padding(caffe_op.attribs['pad']),
                               stride=list(caffe_op.attribs['stride']),
                               dilation=list(caffe_op.attribs['dilation']),
                               groups=caffe_op.attribs['group']))


def convert_deconvolution(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    assert len(caffe_op.inputs) in [2, 3]
    assert bool(caffe_op.attribs['bias_term']) == (len(caffe_op.inputs) == 3)

    input, filter = converter.converted_tensors(caffe_op.inputs[:2])
    output = converter.converted_tensor(caffe_op.output)

    factor = caffe_op.attribs["stride"][0]
    if (caffe_op.attribs["weight_filler"] == "bilinear"
            and caffe_op.attribs["num_output"] == input.shape[1]
            and not caffe_op.attribs["bias_term"]
            and caffe_op.attribs["kernel_size"] == (2 * factor - factor % 2,) * 2
            and caffe_op.attribs["stride"] == (factor,) * 2
            and caffe_op.attribs["pad"] == (factor // 2,) * 2
            and caffe_op.attribs["group"] == input.shape[1]):
        NNEFOperation(graph=nnef_graph,
                      name="multilinear_upsample",
                      inputs=input,
                      outputs=output,
                      attribs=dict(factor=2 * [factor],
                                   method='symmetric',
                                   border='constant'))
    else:
        input, filter = converter.converted_tensors(caffe_op.inputs[:2])

        if caffe_op.attribs['bias_term']:
            bias = converter.converted_tensor(caffe_op.inputs[2])
            converter.nnef_set_channel_only_shape(bias)
        else:
            bias = NNEFTensor(graph=nnef_graph, shape=[], dtype='scalar', data=[0.0])

        output = converter.converted_tensor(caffe_op.output)

        NNEFOperation(graph=nnef_graph,
                      name='deconv',
                      inputs=(input, filter, bias),
                      outputs=output,
                      attribs=dict(border='constant',
                                   padding=converter.nnef_padding(caffe_op.attribs['pad']),
                                   stride=list(caffe_op.attribs['stride']),
                                   dilation=list(caffe_op.attribs['dilation']),
                                   groups=caffe_op.attribs['group']))


def convert_relu(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None
    input, output = converter.converted_tensors((caffe_op.input, caffe_op.output))

    if caffe_op.attribs["negative_slope"] == 0:
        NNEFOperation(graph=nnef_graph, name="relu", inputs=input, outputs=output)
    else:
        NNEFOperation(graph=nnef_graph,
                      name="leaky_relu",
                      inputs=input,
                      outputs=output,
                      attribs=dict(alpha=caffe_op.attribs['negative_slope']))


def convert_prelu(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None
    input, alpha = converter.converted_tensors(caffe_op.inputs)
    output = converter.converted_tensor(caffe_op.output)

    if not caffe_op.attribs['channel_shared']:
        alpha.shape = [1] + alpha.shape
        alpha.data = alpha.data.reshape(alpha.shape)

    NNEFOperation(graph=nnef_graph, name="prelu", inputs=(input, alpha), outputs=output)


def convert_pooling(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe_op.input, caffe_op.output))

    POOL_MAX = 0
    POOL_AVE = 1

    border_by_pool = {
        POOL_MAX: 'ignore',
        POOL_AVE: 'constant',
    }

    pool_name_by_pool = {
        POOL_MAX: "max_pool",
        POOL_AVE: "avg_pool"
    }

    reduce_name_by_pool = {
        POOL_MAX: "max_reduce",
        POOL_AVE: "mean_reduce"
    }

    if caffe_op.attribs["pool"] not in reduce_name_by_pool:
        raise utils.NNEFToolsException("unsupported pool method {}".format(caffe_op.attribs["pool"]))

    if caffe_op.attribs["global_pooling"]:
        NNEFOperation(graph=nnef_graph,
                      name=reduce_name_by_pool[caffe_op.attribs["pool"]],
                      inputs=input,
                      outputs=output,
                      attribs=dict(axes=list(range(2, input.rank))))
    else:
        input_size = input.shape
        padding = converter.nnef_padding([0, 0] + list(caffe_op.attribs["pad"]))
        stride = [1, 1] + list(caffe_op.attribs["stride"])
        kernel_size = [1, 1] + list(caffe_op.attribs["kernel_size"])

        # compensate for caffe's pooling output size calculation
        # https://github.com/BVLC/caffe/issues/1318#issuecomment-59594323

        CEIL = 0
        if caffe_op.attribs.get('round_mode', CEIL) == CEIL:
            old_padding = padding
            padding = [(p, converter.get_pooling_right_padding(h, k, p, q, s))
                       for h, k, (p, q), s in zip(input_size, kernel_size, old_padding, stride)]

        NNEFOperation(graph=nnef_graph,
                      name=pool_name_by_pool[caffe_op.attribs["pool"]],
                      inputs=input,
                      outputs=output,
                      attribs=dict(size=kernel_size,
                                   border=border_by_pool[caffe_op.attribs["pool"]],
                                   padding=padding,
                                   stride=stride))


def convert_reduction(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    SUM = 1
    ASUM = 2
    SUMSQ = 3
    MEAN = 4

    reduction_ops = {SUM, ASUM, SUMSQ, MEAN}

    elementwise_by_op = {
        SUM: None,
        ASUM: "abs",
        SUMSQ: "sqr",
        MEAN: None
    }

    reduce_by_op = {
        SUM: "sum_reduce",
        ASUM: "sum_reduce",
        SUMSQ: "sum_reduce",
        MEAN: "mean_reduce"
    }

    input, output = converter.converted_tensors((caffe_op.input, caffe_op.output))

    axis = converter.nnef_axis(caffe_op.attribs['axis'], input.rank)
    axes = list(range(axis, input.rank))
    coeff = caffe_op.attribs['coeff']
    operation = caffe_op.attribs['operation']

    assert operation in reduction_ops

    elementwise = elementwise_by_op[operation]

    if elementwise:
        input = NNEFOperation(graph=nnef_graph,
                              name=elementwise,
                              inputs=input,
                              outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    reduce = reduce_by_op[operation]

    input = NNEFOperation(
        graph=nnef_graph,
        name=reduce,
        inputs=input,
        outputs=NNEFTensor(graph=nnef_graph,
                           shape=[1 if a in axes else d for a, d in enumerate(input.shape)],
                           dtype=output.dtype),
        attribs=dict(axes=list(axes))).output

    input = NNEFOperation(graph=nnef_graph,
                          name="squeeze",
                          inputs=input,
                          outputs=(output if coeff == 1 else
                                   NNEFTensor(graph=nnef_graph, shape=list(output.shape), dtype=output.dtype)),
                          attribs=dict(axes=list(axes))).output

    if coeff != 1:
        NNEFOperation(graph=nnef_graph,
                      name="mul",
                      inputs=(input, NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[coeff])),
                      outputs=output)


def convert_lrn(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe_op.input, caffe_op.output))

    ACROSS_CHANNELS = 0
    WITHIN_CHANNEL = 1

    if caffe_op.attribs["norm_region"] == ACROSS_CHANNELS:
        size = [caffe_op.attribs["local_size"] if i == 1 else 1 for i in range(input.rank)]
    elif caffe_op.attribs["norm_region"] == WITHIN_CHANNEL:
        size = [caffe_op.attribs["local_size"] if i >= 2 else 1 for i in range(input.rank)]
    else:
        assert False

    NNEFOperation(graph=nnef_graph,
                  name="local_response_normalization",
                  inputs=input,
                  outputs=output,
                  attribs=dict(size=size,
                               alpha=caffe_op.attribs['alpha'],
                               beta=caffe_op.attribs['beta'],
                               bias=caffe_op.attribs['k']))


def convert_mvn(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe_op.input, caffe_op.output))

    assert input.rank >= 1
    name = ("local_contrast_normalization"
            if caffe_op.attribs["normalize_variance"]
            else "local_mean_normalization")

    size = list(input.shape)
    if not caffe_op.attribs['across_channels']:
        size[1] = 1

    nnef_op = NNEFOperation(graph=nnef_graph,
                            name=name,
                            inputs=input,
                            outputs=output,
                            attribs=dict(size=size))

    if name == "local_contrast_normalization":
        nnef_op.attribs['epsilon'] = caffe_op.attribs["eps"]


def convert_concat(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    inputs = converter.converted_tensors(caffe_op.inputs)
    output = converter.converted_tensor(caffe_op.output)

    assert len(inputs) > 0

    NNEFOperation(graph=nnef_graph, name="concat",
                  inputs=inputs,
                  outputs=output,
                  attribs=dict(axis=converter.nnef_axis(caffe_op.attribs["axis"], inputs[0].rank)))


def convert_inner_product(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, weight = converter.converted_tensors(caffe_op.inputs[:2])
    output = converter.converted_tensor(caffe_op.output)

    transpose_a = False
    transpose_b = not caffe_op.attribs['transpose']

    axis = converter.nnef_axis(caffe_op.attribs["axis"], input.rank)

    if weight.rank == 4 and weight.shape[:2] == [1, 1]:
        weight.shape = weight.shape[2:]
        weight.data = weight.data.reshape(weight.shape)

    if axis > 1:
        weight.shape = [1] * (axis - 1) + weight.shape
        weight.data = weight.data.reshape(weight.shape)

    if axis != input.rank - 1:
        reshape_output = NNEFTensor(graph=nnef_graph,
                                    shape=[1] * int(axis == 0) + input.shape[:axis] + [
                                        utils.product(input.shape[axis:])],
                                    dtype=input.dtype)
        NNEFOperation(graph=nnef_graph,
                      name="reshape",
                      inputs=input,
                      outputs=reshape_output,
                      attribs=dict(shape=[1] * int(axis == 0) + input.shape[:axis] + [-1]))

        input = reshape_output

    if caffe_op.attribs["bias_term"] and transpose_b and axis == 1:
        bias = converter.converted_tensor(caffe_op.inputs[2])
        assert bias.rank == 1 or (bias.rank == 4 and bias.shape[:3] == [1, 1, 1])
        bias.shape = [1, bias.shape[-1]]
        bias.data = bias.data.reshape(bias.shape)
        if axis > 1:
            bias.shape = [1] * (axis - 1) + bias.shape
            bias.data = bias.data.reshape(bias.shape)

        NNEFOperation(graph=nnef_graph,
                      name="linear",
                      inputs=(input, weight, bias),
                      outputs=output)
    elif caffe_op.attribs["bias_term"]:
        matmul_output = NNEFTensor(graph=nnef_graph,
                                   shape=infer.matmul(a=input.shape,
                                                      b=weight.shape,
                                                      transpose_a=transpose_a,
                                                      transpose_b=transpose_b),
                                   dtype=input.dtype)

        add_output = NNEFTensor(graph=nnef_graph,
                                shape=list(matmul_output.shape),
                                dtype=input.dtype) if axis == 0 else output

        NNEFOperation(graph=nnef_graph,
                      name="matmul",
                      inputs=(input, weight),
                      outputs=matmul_output,
                      attribs=dict(transposeA=transpose_a, transposeB=transpose_b))

        bias = converter.converted_tensor(caffe_op.inputs[2])
        assert bias.rank == 1 or (bias.rank == 4 and bias.shape[:3] == [1, 1, 1])
        bias.shape = [1, bias.shape[-1]]
        bias.data = bias.data.reshape(bias.shape)
        if axis > 1:
            bias.shape = [1] * (axis - 1) + bias.shape
            bias.data = bias.data.reshape(bias.shape)

        NNEFOperation(graph=nnef_graph,
                      name="add",
                      inputs=(matmul_output, bias),
                      outputs=add_output)
        if axis == 0:
            NNEFOperation(graph=nnef_graph,
                          name="unsqueeze",
                          inputs=add_output,
                          outputs=output,
                          attribs=dict(axes=[0]))
    else:
        matmul_output = NNEFTensor(graph=nnef_graph,
                                   shape=infer.matmul(a=input.shape,
                                                      b=weight.shape,
                                                      transpose_a=transpose_a,
                                                      transpose_b=transpose_b),
                                   dtype=input.dtype) if axis == 0 else output

        NNEFOperation(graph=nnef_graph,
                      name="matmul",
                      inputs=(input, weight),
                      outputs=matmul_output,
                      attribs=dict(transposeA=transpose_a, transposeB=transpose_b))

        if axis == 0:
            NNEFOperation(graph=nnef_graph,
                          name="unsqueeze",
                          inputs=matmul_output,
                          outputs=output,
                          attribs=dict(axes=[0]))


def convert_reshape(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe_op.input, caffe_op.output))

    axis = converter.nnef_axis(caffe_op.attribs["axis"], input.rank)
    num_axes = input.rank if caffe_op.attribs["num_axes"] == -1 else caffe_op.attribs["num_axes"]
    new_shape = input.shape[:axis] + caffe_op.attribs["shape"] + input.shape[axis + num_axes:]

    NNEFOperation(graph=nnef_graph,
                  name="reshape",
                  inputs=input,
                  outputs=output,
                  attribs=dict(shape=new_shape))


def convert_split(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input = converter.converted_tensor(caffe_op.input)
    outputs = converter.converted_tensors(caffe_op.outputs)

    NNEFOperation(graph=nnef_graph,
                  name="copy_n",
                  inputs=input,
                  outputs=outputs,
                  attribs=dict(times=len(outputs)))


def convert_slice(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input = converter.converted_tensor(caffe_op.input)
    outputs = converter.converted_tensors(caffe_op.outputs)
    axis = converter.nnef_axis(caffe_op.attribs['axis'], rank=input.rank)

    NNEFOperation(graph=nnef_graph,
                  name="split",
                  inputs=input,
                  outputs=outputs,
                  attribs=dict(axis=axis, ratios=[output.shape[axis] for output in outputs]))


def convert_softmax(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe_op.input, caffe_op.output))

    NNEFOperation(graph=nnef_graph,
                  name="softmax",
                  inputs=input,
                  outputs=output,
                  attribs=dict(axes=[converter.nnef_axis(caffe_op.attribs["axis"], input.rank)]))


def generic_convert_unary(converter, Caffe_op, NNEF_graph, target_name):
    # type: (Converter, CaffeOperation, NNEFGraph, str)->None
    NNEFOperation(graph=NNEF_graph,
                  name=target_name,
                  inputs=converter.converted_tensor(Caffe_op.input),
                  outputs=converter.converted_tensor(Caffe_op.output))


def convert_bias(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    x, y = converter.converted_tensors(caffe_op.inputs)
    z = converter.converted_tensor(caffe_op.output)
    axis = caffe_op.attribs['axis']

    if y.is_variable:
        converter.nnef_add_dims_to_left(y, axis)
        NNEFOperation(graph=nnef_graph, name="add", inputs=(x, y), outputs=z)
    else:
        if axis > 0:
            y = NNEFOperation(graph=nnef_graph,
                              name="unsqueeze",
                              inputs=y,
                              outputs=NNEFTensor(graph=nnef_graph, shape=[1] * axis + y.shape, dtype=y.dtype),
                              attribs=dict(axes=list(range(axis)))).output
        NNEFOperation(graph=nnef_graph, name="add", inputs=(x, y), outputs=z)


def convert_scale(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    x, y = converter.converted_tensors(caffe_op.inputs[:2])
    z = converter.converted_tensor(caffe_op.output)
    axis = caffe_op.attribs['axis']

    if y.is_variable:
        converter.nnef_add_dims_to_left(y, axis)

        mul_output = (z
                      if not caffe_op.attribs['bias_term']
                      else NNEFTensor(graph=nnef_graph, shape=list(x.shape), dtype=x.dtype))

        NNEFOperation(graph=nnef_graph, name="mul", inputs=(x, y), outputs=mul_output)

        if caffe_op.attribs["bias_term"]:
            bias = converter.converted_tensor(caffe_op.inputs[2])
            converter.nnef_add_dims_to_left(bias, axis)
            NNEFOperation(graph=nnef_graph, name="add", inputs=(mul_output, bias), outputs=z)
    else:
        if axis > 0:
            y = NNEFOperation(graph=nnef_graph,
                              name="unsqueeze",
                              inputs=y,
                              outputs=NNEFTensor(graph=nnef_graph, shape=[1] * axis + y.shape, dtype=y.dtype),
                              attribs=dict(axes=list(range(axis)))).output

        mul_output = (z
                      if not caffe_op.attribs['bias_term']
                      else NNEFTensor(graph=nnef_graph, shape=list(x.shape), dtype=x.dtype))

        NNEFOperation(graph=nnef_graph, name="mul", inputs=(x, y), outputs=mul_output)

        if caffe_op.attribs["bias_term"]:
            bias = converter.converted_tensor(caffe_op.inputs[2])
            converter.nnef_add_dims_to_left(bias, axis)
            NNEFOperation(graph=nnef_graph, name="add", inputs=(mul_output, bias), outputs=z)


def convert_power(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe_op.input, caffe_op.output))

    if caffe_op.attribs["scale"] != 1:
        mul = NNEFOperation(graph=nnef_graph,
                            name="mul",
                            inputs=(input,
                                    NNEFTensor(graph=nnef_graph, shape=[], data=[float(caffe_op.attribs['scale'])])),
                            outputs=(output
                                     if caffe_op.attribs["shift"] == 0 and caffe_op.attribs["power"] == 1
                                     else NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)))
        input = mul.output
    if caffe_op.attribs["shift"] != 0:
        add = NNEFOperation(graph=nnef_graph,
                            name="add",
                            inputs=(input,
                                    NNEFTensor(graph=nnef_graph, shape=[], data=[float(caffe_op.attribs['shift'])])),
                            outputs=(output
                                     if caffe_op.attribs["power"] == 1
                                     else NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)))
        input = add.output
    if caffe_op.attribs["power"] != 1 or output.producer is None:
        NNEFOperation(graph=nnef_graph,
                      name="pow",
                      inputs=(input,
                              NNEFTensor(graph=nnef_graph, shape=[], data=[float(caffe_op.attribs['power'])])),
                      outputs=output)


def convert_exp(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe_op.input, caffe_op.output))

    assert caffe_op.attribs["base"] > 0 or caffe_op.attribs["base"] == -1

    if caffe_op.attribs["scale"] != 1:
        mul = NNEFOperation(graph=nnef_graph,
                            name="mul",
                            inputs=(input,
                                    NNEFTensor(graph=nnef_graph, shape=[],
                                               data=[float(caffe_op.attribs['scale'])])),
                            outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype))
        input = mul.output
    if caffe_op.attribs["shift"] != 0:
        add = NNEFOperation(graph=nnef_graph,
                            name="add",
                            inputs=(input,
                                    NNEFTensor(graph=nnef_graph, shape=[],
                                               data=[float(caffe_op.attribs['shift'])])),
                            outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype))
        input = add.output
    if caffe_op.attribs["base"] == -1:  # -1 means base is e
        NNEFOperation(graph=nnef_graph,
                      name="exp",
                      inputs=input,
                      outputs=output)
    else:
        NNEFOperation(graph=nnef_graph,
                      name="pow",
                      inputs=(NNEFTensor(graph=nnef_graph, shape=[], data=[float(caffe_op.attribs['base'])]), input),
                      outputs=output)


def convert_log(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe_op.input, caffe_op.output))

    assert caffe_op.attribs["base"] > 0 or caffe_op.attribs["base"] == -1

    if caffe_op.attribs["scale"] != 1:
        mul = NNEFOperation(graph=nnef_graph,
                            name="mul",
                            inputs=(input,
                                    NNEFTensor(graph=nnef_graph, shape=[],
                                               data=[float(caffe_op.attribs['scale'])])),
                            outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype))
        input = mul.output
    if caffe_op.attribs["shift"] != 0:
        add = NNEFOperation(graph=nnef_graph,
                            name="add",
                            inputs=(input,
                                    NNEFTensor(graph=nnef_graph, shape=[],
                                               data=[float(caffe_op.attribs['shift'])])),
                            outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype))
        input = add.output
    if caffe_op.attribs["base"] == -1:  # -1 means base is e
        NNEFOperation(graph=nnef_graph,
                      name="log",
                      inputs=input,
                      outputs=output)
    elif caffe_op.attribs["base"] == 2:
        NNEFOperation(graph=nnef_graph,
                      name="log2",
                      inputs=input,
                      outputs=output)
    else:
        log = NNEFOperation(graph=nnef_graph,
                            name="log",
                            inputs=input,
                            outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype))
        NNEFOperation(graph=nnef_graph,
                      name="div",
                      inputs=(log.output,
                              NNEFTensor(graph=nnef_graph, shape=[], data=[math.log(float(caffe_op.attribs['base']))])),
                      outputs=output)


def convert_flatten(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe_op.input, caffe_op.output))

    axis = converter.nnef_axis(caffe_op.attribs["axis"], input.rank)
    end_axis = converter.nnef_axis(caffe_op.attribs["end_axis"], input.rank)
    new_shape = input.shape[:axis] + [-1] + input.shape[end_axis + 1:]

    NNEFOperation(graph=nnef_graph,
                  name="reshape",
                  inputs=input,
                  outputs=output,
                  attribs=dict(shape=new_shape))


def convert_argmax(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe_op.input, caffe_op.output))

    if caffe_op.attribs["top_k"] != 1 or caffe_op.attribs["out_max_val"]:
        raise utils.NNEFToolsException("Argmax params not supported yet: {}".format(caffe_op))

    if caffe_op.attribs["axis"] is None:
        axes = list(range(1, input.rank))
    else:
        axes = [converter.nnef_axis(caffe_op.attribs["axis"], input.rank)]

    NNEFOperation(graph=nnef_graph,
                  name="argmax_reduce",
                  inputs=input,
                  outputs=output,
                  attribs=dict(axes=axes))


def convert_elu(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe_op.input, caffe_op.output))

    if caffe_op.attribs["alpha"] == 1:
        NNEFOperation(graph=nnef_graph, name="elu", inputs=input, outputs=output)
    else:
        op_gt = NNEFOperation(graph=nnef_graph,
                              name="gt",
                              inputs=(input, NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[0.0])),
                              outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype='logical'))

        op_exp = NNEFOperation(graph=nnef_graph,
                               name="exp",
                               inputs=input,
                               outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype))

        op_sub = NNEFOperation(graph=nnef_graph,
                               name="sub",
                               inputs=(op_exp.output,
                                       NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[1.0])),
                               outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype))

        op_mul = NNEFOperation(graph=nnef_graph,
                               name="mul",
                               inputs=(op_sub.output, NNEFTensor(graph=nnef_graph,
                                                                 shape=[],
                                                                 dtype=input.dtype,
                                                                 data=[float(caffe_op.attribs['alpha'])])),
                               outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype))

        NNEFOperation(graph=nnef_graph, name="select", inputs=(op_gt.output, input, op_mul.output), outputs=output)


def convert_eltwise(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None
    inputs = converter.converted_tensors(caffe_op.inputs)
    output = converter.converted_tensor(caffe_op.output)
    assert len(inputs) >= 2

    name = ("mul", "add", "max")[caffe_op.attribs["operation"]]
    coeffs = caffe_op.attribs['coeff']

    if coeffs:
        assert name == 'add'

        inputs = [NNEFOperation(graph=nnef_graph,
                                name="mul",
                                inputs=(input,
                                        NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[coeff])),
                                outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output
                  for input, coeff in zip(inputs, coeffs)]

    left = inputs[0]
    for i, right in enumerate(inputs[1:]):
        if 1 + i == len(inputs) - 1:
            new_left = output
        else:
            new_left = NNEFTensor(graph=nnef_graph, name=None, shape=list(left.shape), dtype=left.dtype)
        NNEFOperation(graph=nnef_graph, name=name, inputs=(left, right), outputs=new_left)
        left = new_left


def convert_crop(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    caffe_input, caffe_reference = caffe_op.inputs

    input, reference = converter.converted_tensors(caffe_op.inputs)
    output = converter.converted_tensor(caffe_op.output)

    if len(caffe_reference.consumers) == 1 and reference in nnef_graph.inputs and reference not in nnef_graph.outputs:
        nnef_graph.inputs = [i for i in nnef_graph.inputs if i is not reference]

    axis = caffe_op.attribs["axis"]
    offset = caffe_op.attribs["offset"]
    if len(offset) == 1:
        offset = (input.rank - axis) * offset

    axes = list(range(axis, input.rank))
    begin = list(offset)
    end = [o + s for o, s in zip(offset, reference.shape[axis:])]

    NNEFOperation(graph=nnef_graph,
                  name="slice",
                  inputs=input,
                  outputs=output,
                  attribs=dict(axes=axes,
                               begin=begin,
                               end=end))


def convert_batch_norm(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, mean, variance, scale_factor = converter.converted_tensors(caffe_op.inputs)
    output = converter.converted_tensor(caffe_op.output)

    assert scale_factor.shape == [1]

    scale_factor = float(scale_factor.get_numpy_array()[0])
    norm = 0.0 if scale_factor == 0.0 else 1.0 / scale_factor

    converter.nnef_set_channel_only_shape(mean)
    mean.data *= norm
    converter.nnef_set_channel_only_shape(variance)
    variance.data *= norm

    offset = NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[0.0])
    scale = NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[1.0])

    NNEFOperation(graph=nnef_graph,
                  name="batch_normalization",
                  inputs=(input,
                          mean,
                          variance,
                          offset,
                          scale),
                  outputs=output,
                  attribs=dict(epsilon=caffe_op.attribs["eps"]))


def convert_scaled_batch_norm(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, mean, variance, scale_factor, offset, scale = converter.converted_tensors(caffe_op.inputs)
    output = converter.converted_tensor(caffe_op.output)

    assert scale_factor.shape == [1]

    scale_factor = float(scale_factor.get_numpy_array()[0])
    norm = 0.0 if scale_factor == 0.0 else 1.0 / scale_factor

    converter.nnef_set_channel_only_shape(mean)
    mean.data *= norm
    converter.nnef_set_channel_only_shape(variance)
    variance.data *= norm
    converter.nnef_set_channel_only_shape(offset)
    converter.nnef_set_channel_only_shape(scale)

    NNEFOperation(graph=nnef_graph,
                  name="batch_normalization",
                  inputs=(input, mean, variance, offset, scale),
                  outputs=output,
                  attribs=dict(epsilon=caffe_op.attribs["eps"]))


def convert_threshold(converter, caffe_op, nnef_graph):
    # type: (Converter, CaffeOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe_op.input, caffe_op.output))

    threshold = NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[caffe_op.attribs['threshold']])
    one = NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[1.0])
    zero = NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[0.0])
    gt_output = NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype='logical')

    NNEFOperation(graph=nnef_graph,
                  name="gt",
                  inputs=(input, threshold),
                  outputs=gt_output)
    NNEFOperation(graph=nnef_graph,
                  name="select",
                  inputs=(gt_output, one, zero),
                  outputs=output)


_StandardConverters = {
    # Data layers
    # These don't seem to be important in inference mode:
    # ImageData, Data, HDF5Data, HDF5Output, WindowData, MemoryData, DummyData
    'Input': convert_input,

    # Vision layers
    'Convolution': convert_convolution,
    'Pooling': convert_pooling,
    'Crop': convert_crop,
    'Deconvolution': convert_deconvolution,
    # SPP
    # Im2Col (legacy)

    # Recurrent layers
    # Recurrent, RNN, LSTM

    # Common layers
    'InnerProduct': convert_inner_product,
    # Dropout (we remove it from the graph)
    # Embed

    # Normalization layers
    'LRN': convert_lrn,
    'MVN': convert_mvn,
    'BatchNorm': convert_batch_norm,
    'BatchNorm+Scale': convert_scaled_batch_norm,  # created in caffe_to_nnef_passes.py

    # Activation / neuron layers
    'ReLU': convert_relu,
    'PReLU': convert_prelu,
    'ELU': convert_elu,
    'Sigmoid': partial(generic_convert_unary, target_name='sigmoid'),
    'TanH': partial(generic_convert_unary, target_name='tanh'),
    'AbsVal': partial(generic_convert_unary, target_name='abs'),
    'Power': convert_power,
    'Exp': convert_exp,
    'Log': convert_log,
    'BNLL': partial(generic_convert_unary, target_name='softplus'),
    'Threshold': convert_threshold,
    'Bias': convert_bias,
    'Scale': convert_scale,

    # Utility layers
    'Flatten': convert_flatten,
    'Reshape': convert_reshape,
    # BatchReindex
    'Split': convert_split,  # (to copy_n)
    'Concat': convert_concat,
    'Slice': convert_slice,
    'Eltwise': convert_eltwise,
    # Filter
    # Parameter
    'Reduction': convert_reduction,
    # Silence (we remove it from the graph)
    'ArgMax': convert_argmax,
    'Softmax': convert_softmax,
    # Python

    # Loss layers
    # These are not important in inference mode.
}
