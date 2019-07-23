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
import six

from nnef_tools.conversion.caffe2.nnef_to_caffe2_passes import pre_conversion_pass
from nnef_tools.io.caffe2.caffe2_graph import *
from nnef_tools.conversion import converter as _converter
from nnef_tools.conversion import transforms
from nnef_tools.core import utils
from nnef_tools.io.nnef.nnef_graph import *
from nnef_tools.io.nnef.parser_config import NNEFParserConfig
from nnef_tools.shape_inference import shape_inference as infer


class Converter(_converter.Converter[NNEFTensor, NNEFOperation, NNEFGraph,
                                     Caffe2Tensor, Caffe2Operation, Caffe2Graph]):

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
        # type:(NNEFGraph)->Caffe2Graph
        return Caffe2Graph(name=source_graph.name)

    def convert_tensor(self, source_tensor, target_graph):
        # type: (NNEFTensor, Caffe2Graph)->Caffe2Tensor
        return Caffe2Tensor(graph=target_graph,
                            name=source_tensor.name,
                            shape=list(source_tensor.shape),
                            dtype={'scalar': 'FLOAT',
                                   'integer': 'INT32',
                                   'logical': 'BOOL'}[source_tensor.dtype],
                            data=copy.copy(source_tensor.get_numpy_array()))

    def convert_graph(self, source_graph):
        # type: (NNEFGraph)->Caffe2Graph
        pre_conversion_pass(source_graph)
        target_graph = super(Converter, self).convert_graph(source_graph)  # type: Caffe2Graph
        target_graph.generate_missing_names()
        return target_graph

    @staticmethod
    def nnef_padding_to_caffe2_pads(padding):
        ps, qs = utils.zip_inverse(2, padding)
        return ps + qs

    @staticmethod
    def nnef_border_to_caffe2_mode(mode):
        return {'constant': 'constant',
                'reflect': 'reflect',
                'replicate': 'edge'}[mode]

    @staticmethod
    def add_squeeze(caffe2_graph, caffe2_tensor, dims):
        # type: (Caffe2Graph, Caffe2Tensor, typing.List[int])->Caffe2Tensor
        return Caffe2Operation(graph=caffe2_graph,
                               name="Squeeze",
                               inputs=caffe2_tensor,
                               attribs=dict(dims=dims),
                               outputs=Caffe2Tensor(graph=caffe2_graph,
                                                    shape=transforms.squeezed_shape(caffe2_tensor.shape, dims),
                                                    dtype=caffe2_tensor.dtype)).output


def convert_default(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    print("Warning: Converter of {} is not implemented, doing default conversion.".format(nnef_op.name))

    Caffe2Operation(graph=caffe2_graph,
                    name=nnef_op.name,
                    inputs=converter.converted_tensors(nnef_op.inputs),
                    attribs=utils.recursive_transform(nnef_op.attribs, lambda x: x if x is not None else "None"),
                    outputs=converter.converted_tensors(nnef_op.outputs))


def generic_convert_unary(converter, nnef_op, caffe2_graph, target_name):
    # type: (Converter, NNEFOperation, Caffe2Graph, str)->None

    Caffe2Operation(graph=caffe2_graph,
                    name=target_name,
                    inputs=converter.converted_tensor(nnef_op.input),
                    outputs=converter.converted_tensor(nnef_op.output))


def generic_convert_to_binary(converter, nnef_op, caffe2_graph, target_name, commutative=False):
    # type: (Converter, NNEFOperation, Caffe2Graph, str, bool)->None

    a, b = converter.converted_tensors(nnef_op.inputs)
    c = converter.converted_tensor(nnef_op.output)

    def can_caffe2_broadcast_to(a_shape, b_shape):
        # This function assumes that the broadcast axis is 0
        if utils.starts_with(a_shape, b_shape):
            return True
        # If there are dimensions that differ and none of them is one, we obviously can't broadcast
        if any(a_dim != b_dim and 1 not in [a_dim, b_dim] for a_dim, b_dim in zip(a_shape, b_shape)):
            return False
        # Unidirectional broadcast, so len(a_shape) >= len(b_shape) must be true
        if len(a_shape) < len(b_shape):
            return False
        # Unidirectional broadcast, so a_dim >= b_dim must be true
        if any(a_dim < b_dim for a_dim, b_dim in zip(a_shape, b_shape)):
            return False

        # When both dims are 1, it can be regarded as broadcast or no-broadcast too. Let's remove those dims.
        a_shape, b_shape = utils.filter_multi(lambda a, b: not (a == b == 1), a_shape, b_shape)
        # Create a string: 'b' for broadcast, 'n' for no broadcast
        broadcast_str = ''.join('b' if a_dim != b_dim else 'n' for a_dim, b_dim in zip(a_shape, b_shape))
        # Remove consecutive repetitions, e.g. bbbnnbb -> bnb
        broadcast_str = utils.without_consecutive_repetitions(broadcast_str)
        # It is not supported iff we don't broadcast, then broadcast, then don't broadcast again
        return 'nbn' not in broadcast_str

    swapped = False
    if not can_caffe2_broadcast_to(a.shape, b.shape):
        swap_needed = can_caffe2_broadcast_to(b.shape, a.shape)
        if swap_needed and (commutative or nnef_op.name in ('sub',)):
            a, b = b, a
            swapped = True
        else:
            raise utils.NNEFToolsException(
                'Binary ops are only supported if the second input can be broadcasted to the first.'
                'The broadcast is unidirectional and it works only in the leading and trailing dimensions.'
                'Parameters in add, sub, mul, and, or, eq and ne will be swapped if needed.')

    if swapped and nnef_op.name == 'sub':
        binary_op = Caffe2Operation(graph=caffe2_graph,
                                    name=target_name,
                                    inputs=(a, b),
                                    outputs=Caffe2Tensor(graph=caffe2_graph, shape=list(c.shape), dtype=c.dtype))
        Caffe2Operation(graph=caffe2_graph,
                        name='Negative',
                        inputs=binary_op.output,
                        outputs=c)
    else:
        binary_op = Caffe2Operation(graph=caffe2_graph,
                                    name=target_name,
                                    inputs=(a, b),
                                    outputs=c)
    if a.shape == b.shape:
        binary_op.attribs['broadcast'] = 0
    else:
        binary_op.attribs['broadcast'] = 1
        binary_op.attribs['axis'] = 0


def generic_convert_pool(converter, nnef_op, caffe2_graph, target_name, allowed_borders):
    # type: (Converter, NNEFOperation, Caffe2Graph, str, typing.Tuple[str])->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if not nnef_op.attribs['border'] in allowed_borders:
        raise utils.NNEFToolsException(
            '{} is not supported with border={}'.format(nnef_op.name, nnef_op.attribs['border']))
    if any(d != 1 for d in nnef_op.attribs['dilation']):
        raise utils.NNEFToolsException('Dilation is not supported for pooling')
    if (nnef_op.attribs['size'][:2] != [1, 1]
            or nnef_op.attribs['stride'][:2] != [1, 1]
            or nnef_op.attribs['padding'][:2] != [(0, 0), (0, 0)]):
        raise utils.NNEFToolsException('Pooling is only supported in spatial dimensions (NCHW)')

    pool_op = Caffe2Operation(graph=caffe2_graph,
                              name=target_name,
                              inputs=input,
                              outputs=output,
                              attribs=dict(kernels=list(nnef_op.attribs['size'][2:]),
                                           strides=list(nnef_op.attribs['stride'][2:]),
                                           pads=converter.nnef_padding_to_caffe2_pads(nnef_op.attribs['padding'][2:])))
    if nnef_op.name == '_lp_pool':
        pool_op.attribs['p'] = nnef_op.attribs['p']


def generic_convert_lp_normalization(converter, nnef_op, caffe2_graph, target_name):
    # type: (Converter, NNEFOperation, Caffe2Graph, str)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if len(nnef_op.attribs['axes']) != 1:
        raise utils.NNEFToolsException('l1 or l2 normalization is only supported for a single axis')
    if nnef_op.attribs['bias'] != 0:
        raise utils.NNEFToolsException('l1 or l2 normalization is only supported for bias=0')

    if abs(nnef_op.attribs['epsilon'] - 1e-12) > 1e-13:
        print('Info: epsilon is ignored for lp_normalization, always using 1e-12')

    Caffe2Operation(graph=caffe2_graph,
                    name=target_name,
                    inputs=input,
                    outputs=output,
                    attribs=dict(axis=nnef_op.attribs['axes'][0]))


def generic_convert_reduce(converter, nnef_op, caffe2_graph, multi_axis, target_name, target_name_if_normalized=""):
    # type: (Converter, NNEFOperation, Caffe2Graph, bool, str, str)->None
    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if not nnef_op.attribs['axes']:
        Caffe2Operation(graph=caffe2_graph, name='Copy', inputs=input, outputs=output)
        return

    if not multi_axis and len(nnef_op.attribs['axes']) != 1:
        raise utils.NNEFToolsException("{} supports only one axis in Caffe2".format(target_name))

    caffe2_op = Caffe2Operation(graph=caffe2_graph,
                                name=(target_name_if_normalized
                                      if target_name_if_normalized and nnef_op.attribs.get('normalize', False)
                                      else target_name),
                                inputs=input,
                                attribs=dict(keepdims=1),
                                outputs=output)

    if multi_axis:
        caffe2_op.attribs['axes'] = list(nnef_op.attribs['axes'])
    else:
        caffe2_op.attribs['axis'] = nnef_op.attribs['axes'][0]


def convert_min_max(converter, nnef_op, caffe2_graph, target_name):
    # type: (Converter, NNEFOperation, Caffe2Graph, str)->None

    assert nnef_op.name in ('min', 'max')

    a, b = converter.converted_tensors(nnef_op.inputs)
    c = converter.converted_tensor(nnef_op.output)

    if a.shape == b.shape:
        Caffe2Operation(graph=caffe2_graph,
                        name='Min' if nnef_op.name == 'min' else 'Max',
                        inputs=(a, b),
                        outputs=c)
    elif (not a.shape and a.data is not None) or (not b.shape and b.data is not None):
        if b.shape or b.data is None:
            a, b = b, a
        clip = Caffe2Operation(graph=caffe2_graph,
                               name='Clip',
                               inputs=a,
                               outputs=c)
        if nnef_op.name == 'min':
            clip.attribs['max'] = np.array(b.data).item()
        else:
            clip.attribs['min'] = np.array(b.data).item()
    else:
        raise utils.NNEFToolsException(
            '{} is only supported for equal input shapes, or one tensor and one scalar constant'.format(nnef_op.name))


def convert_squeeze(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if nnef_op.attribs['axes']:
        Caffe2Operation(graph=caffe2_graph,
                        name='Squeeze',
                        inputs=input,
                        outputs=output,
                        attribs=dict(dims=nnef_op.attribs['axes']))
    else:
        Caffe2Operation(graph=caffe2_graph,
                        name='Copy',
                        inputs=input,
                        outputs=output)


def convert_unsqueeze(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if nnef_op.attribs['axes']:
        Caffe2Operation(graph=caffe2_graph,
                        name='ExpandDims',
                        inputs=input,
                        outputs=output,
                        attribs=dict(dims=nnef_op.attribs['axes']))
    else:
        Caffe2Operation(graph=caffe2_graph,
                        name='Copy',
                        inputs=input,
                        outputs=output)


def convert_concat(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    inputs = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)
    Caffe2Operation(graph=caffe2_graph,
                    name='Concat',
                    inputs=inputs,
                    outputs=(output, Caffe2Tensor(graph=caffe2_graph, shape=[len(nnef_op.inputs)], dtype='INT32')),
                    attribs=dict(axis=nnef_op.attribs['axis']))


def convert_select(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    cond, true_value, false_value = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    if any(x.rank < 1 for x in (cond, true_value, false_value)):
        raise utils.NNEFToolsException('Select with rank < 1 is not supported')
    if cond.shape == [true_value.shape[0]] and true_value.shape == false_value.shape and true_value.rank > 1:
        Caffe2Operation(graph=caffe2_graph,
                        name='Conditional',
                        inputs=(cond, true_value, false_value),
                        outputs=output)
    elif cond.shape == true_value.shape == false_value.shape:
        Caffe2Operation(graph=caffe2_graph,
                        name='Where',
                        inputs=(cond, true_value, false_value),
                        outputs=output)
    else:
        raise utils.NNEFToolsException('Broadcasting is not supported for select in general.')


def convert_conv(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input, filter, bias = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    if not nnef_op.attribs['border'] in ('constant',):
        raise utils.NNEFToolsException(
            '{} is not supported with border={}'.format(nnef_op.name, nnef_op.attribs['border']))

    if nnef_op.inputs[2].data is not None and np.all(nnef_op.inputs[2].get_numpy_array() == 0.0):
        bias = None
    else:
        if bias.rank != 2 and bias.shape[0] != 1:
            raise utils.NNEFToolsException('conv bias is only supported if its shape is [1, C] or it is 0')
        bias = converter.add_squeeze(caffe2_graph, bias, [0])

    Caffe2Operation(graph=caffe2_graph,
                    name='Conv',
                    inputs=(input, filter, bias) if bias else (input, filter),
                    outputs=output,
                    attribs=dict(kernels=filter.shape[2:],
                                 strides=list(nnef_op.attribs['stride']),
                                 dilations=list(nnef_op.attribs['dilation']),
                                 pads=converter.nnef_padding_to_caffe2_pads(nnef_op.attribs['padding']),
                                 group=nnef_op.attribs['groups']))


def convert_deconv(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input, filter, bias = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    if not nnef_op.attribs['border'] in ('constant',):
        raise utils.NNEFToolsException(
            '{} is not supported with border={}'.format(nnef_op.name, nnef_op.attribs['border']))

    if any(d != 1 for d in nnef_op.attribs['dilation']):
        raise utils.NNEFToolsException('Dilation is not supported for deconv')

    if nnef_op.inputs[2].data is not None and np.all(nnef_op.inputs[2].get_numpy_array() == 0.0):
        bias = None
    else:
        if bias.rank != 2 and bias.shape[0] != 1:
            raise utils.NNEFToolsException('deconv bias is only supported if its shape is [1, C] or it is 0')
        bias = converter.add_squeeze(caffe2_graph, bias, [0])

    output_padding = infer.get_deconv_output_padding(output=output.shape,
                                                     input=input.shape,
                                                     filter=filter.shape[2:],
                                                     padding=nnef_op.attribs['padding'],
                                                     stride=nnef_op.attribs['stride'],
                                                     dilation=nnef_op.attribs['dilation'],
                                                     groups=nnef_op.attribs['groups'],
                                                     format=infer.Format.NCHW)

    Caffe2Operation(graph=caffe2_graph,
                    name='ConvTranspose',
                    inputs=(input, filter, bias) if bias else (input, filter),
                    outputs=output,
                    attribs=dict(kernels=filter.shape[2:],
                                 strides=list(nnef_op.attribs['stride']),
                                 pads=converter.nnef_padding_to_caffe2_pads(nnef_op.attribs['padding']),
                                 group=nnef_op.attribs['groups'],
                                 adjs=[q for p, q in output_padding]))


def convert_split(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input = converter.converted_tensor(nnef_op.input)
    outputs = converter.converted_tensors(nnef_op.outputs)
    axis = nnef_op.attribs['axis']
    split = [output.shape[axis] for output in outputs]
    Caffe2Operation(graph=caffe2_graph,
                    name='Split',
                    inputs=input,
                    outputs=outputs,
                    attribs=dict(axis=axis, split=split))


def convert_linear(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    x, w, b = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    assert x.rank == 2 and w.rank == 2

    if nnef_op.inputs[2].data is not None and np.all(nnef_op.inputs[2].get_numpy_array() == 0.0):
        b = Caffe2Tensor(graph=caffe2_graph,
                         shape=[nnef_op.output.shape[1]],
                         dtype=nnef_op.output.dtype,
                         data=np.zeros([nnef_op.output.shape[1]]))
    else:
        if b.rank != 2 and b.shape[0] != 1:
            raise utils.NNEFToolsException('linear bias is only supported if its shape is [1, C] or it is 0')
        b = converter.add_squeeze(caffe2_graph, b, [0])

    Caffe2Operation(graph=caffe2_graph,
                    name='FC',
                    inputs=(x, w, b),
                    outputs=output)


def convert_matmul(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    a, b = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    if not (a.rank == b.rank >= 2):
        raise utils.NNEFToolsException('matmul is only supported with equal input ranks in at least 2D')

    trans_a = int(nnef_op.attribs['transposeA'])
    trans_b = int(nnef_op.attribs['transposeB'])

    if a.rank == 2:
        Caffe2Operation(graph=caffe2_graph,
                        name='MatMul',
                        inputs=(a, b),
                        outputs=output,
                        attribs=dict(axis_a=1,
                                     axis_b=1,
                                     trans_a=trans_a,
                                     trans_b=trans_b))
    else:
        broadcast = int(not (a.shape[:-2] == b.shape[:-2]
                             and a.shape[-2 if trans_a else -1] == b.shape[-1 if trans_b else -2]))
        Caffe2Operation(graph=caffe2_graph,
                        name='BatchMatMul',
                        inputs=(a, b),
                        outputs=output,
                        attribs=dict(broadcast=broadcast,
                                     trans_a=int(nnef_op.attribs['transposeA']),
                                     trans_b=int(nnef_op.attribs['transposeB'])))


def convert_reshape(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input = converter.converted_tensor(nnef_op.input)
    output = converter.converted_tensor(nnef_op.output)

    shape = nnef_op.attribs['shape']
    axis_start = nnef_op.attribs['axis_start']
    axis_count = nnef_op.attribs['axis_count']
    if axis_count == -1:
        axis_count = input.rank - axis_start

    caffe2_shape = input.shape[:axis_start] + shape + input.shape[axis_start + axis_count:]

    Caffe2Operation(graph=caffe2_graph,
                    name='Reshape',
                    inputs=input,
                    outputs=(output, Caffe2Tensor(graph=caffe2_graph, shape=[nnef_op.input.rank], dtype='INT64')),
                    attribs=dict(shape=caffe2_shape))


def convert_leaky_relu(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input = converter.converted_tensor(nnef_op.input)
    output = converter.converted_tensor(nnef_op.output)
    Caffe2Operation(graph=caffe2_graph,
                    name='LeakyRelu',
                    inputs=input,
                    outputs=output,
                    attribs=dict(alpha=nnef_op.attribs['alpha']))


def convert_prelu(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input, slope = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    if slope.rank not in [2, input.rank] or slope.shape[:-1] != [1] * (slope.rank - 1):
        raise utils.NNEFToolsException('prelu is only supported when all axes except the channel are one')
    is_nhwc = (slope.rank == input.rank)
    slope = converter.add_squeeze(caffe2_graph, slope, dims=list(range(slope.rank - 1)))

    Caffe2Operation(graph=caffe2_graph,
                    name='PRelu',
                    inputs=(input, slope),
                    outputs=output,
                    attribs=dict(order='NHWC' if is_nhwc else 'NCHW'))


def convert_local_response_normalization(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None
    input = converter.converted_tensor(nnef_op.input)
    output = converter.converted_tensor(nnef_op.output)
    if input.rank != 4:
        raise utils.NNEFToolsException('local_response_normalization is only supported for 4D input')
    size = nnef_op.attribs['size']
    if sum(s != 1 for s in size) > 1:
        raise utils.NNEFToolsException(
            'local_response_normalization is only supported in the second or the last dimension')
    if all(s == 1 or i == 1 for i, s in enumerate(size)):
        is_nhwc = False
        size = size[1]
    elif all(s == 1 or i == input.rank - 1 for i, s in enumerate(size)):
        is_nhwc = True
        size = size[-1]
    else:
        raise utils.NNEFToolsException(
            'local_response_normalization is only supported in the second or the last dimension')
    if size % 2 != 1:
        raise utils.NNEFToolsException(
            'local_response_normalization is only supported for odd size')
    caffe2_op = Caffe2Operation(graph=caffe2_graph,
                                name='LRN',
                                inputs=input,
                                outputs=output,
                                attribs=dict(size=size,
                                             alpha=nnef_op.attribs['alpha'],
                                             beta=nnef_op.attribs['beta'],
                                             bias=nnef_op.attribs['bias']))
    if is_nhwc:
        caffe2_op.attribs['order'] = 'NHWC'


def convert_transpose(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input = converter.converted_tensor(nnef_op.input)
    output = converter.converted_tensor(nnef_op.output)
    axes = list(nnef_op.attribs['axes'])
    axes += list(range(len(axes), input.rank))
    Caffe2Operation(graph=caffe2_graph,
                    name='Transpose',
                    inputs=input,
                    outputs=output,
                    attribs=dict(axes=axes))


def convert_add_n(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    inputs = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    if len(inputs) == 0 or any(input.shape != inputs[0].shape for input in inputs):
        raise utils.NNEFToolsException('add_n is only supported if there is at least one input, '
                                       'and all inputs have the same shape')
    Caffe2Operation(graph=caffe2_graph,
                    name='Sum',
                    inputs=inputs,
                    outputs=output)


def convert_softmax(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input = converter.converted_tensor(nnef_op.input)
    output = converter.converted_tensor(nnef_op.output)
    axes = sorted(nnef_op.attribs['axes'])
    if not axes or axes != list(range(axes[0], input.rank)):
        raise utils.NNEFToolsException(
            "Softmax is only supported if the axes are a contiguous range ending with the last axis.")
    Caffe2Operation(graph=caffe2_graph,
                    name='Softmax',
                    inputs=input,
                    outputs=output,
                    attribs=dict(axis=axes[0]))


def convert_batch_normalization(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    x, mean, var, bias, scale = converter.converted_tensors(nnef_op.inputs)
    y = converter.converted_tensor(nnef_op.output)

    if not (mean.rank == var.rank == bias.rank == scale.rank) or not all(s == 1
                                                                         for t in (mean, var, bias, scale)
                                                                         for s in t.shape[:-1]):
        raise utils.NNEFToolsException('batch_normalization is only supported in axis 1 or -1')

    is_nhwc = (x.rank != 2 and mean.rank == x.rank)
    squeeze_axes = list(range(x.rank - 1)) if is_nhwc else [0]
    scale, bias, mean, var = [converter.add_squeeze(caffe2_graph, t, squeeze_axes) for t in (scale, bias, mean, var)]

    caffe2_op = Caffe2Operation(graph=caffe2_graph,
                                name='SpatialBN',
                                inputs=(x, scale, bias, mean, var),
                                outputs=y,
                                attribs=dict(is_test=1, epsilon=nnef_op.attribs['epsilon']))
    if is_nhwc:
        caffe2_op.attribs['order'] = 'NHWC'


def convert_max_pool_with_index(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input = converter.converted_tensor(nnef_op.input)
    output, index = converter.converted_tensors(nnef_op.outputs)

    print("Warning: Caffe2 does not have CPU implementation for MaxPoolWithIndex")

    if nnef_op.attribs['border'] != 'ignore':
        raise utils.NNEFToolsException(
            '{} is not supported with border={}'.format(nnef_op.name, nnef_op.attribs['border']))
    if any(d != 1 for d in nnef_op.attribs['dilation']):
        raise utils.NNEFToolsException('Dilation is not supported for pooling')
    if (nnef_op.attribs['size'][:2] != [1, 1]
            or nnef_op.attribs['stride'][:2] != [1, 1]
            or nnef_op.attribs['padding'][:2] != [(0, 0), (0, 0)]):
        raise utils.NNEFToolsException('Pooling is only supported in spatial dimensions (NCHW)')

    Caffe2Operation(graph=caffe2_graph,
                    name='MaxPoolWithIndex',
                    inputs=input,
                    outputs=(output, index),
                    attribs=dict(kernels=list(nnef_op.attribs['size'][2:]),
                                 strides=list(nnef_op.attribs['stride'][2:]),
                                 pads=converter.nnef_padding_to_caffe2_pads(nnef_op.attribs['padding'][2:])))


def convert_clamp(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input, min, max = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    if min.shape or min.data is None or max.shape or max.data is None:
        raise utils.NNEFToolsException('clamp is only supported with constant min/max')

    Caffe2Operation(graph=caffe2_graph,
                    name='Clip',
                    inputs=input,
                    outputs=output,
                    attribs=dict(min=np.array(min.data).item(), max=np.array(max.data).item()))


def convert_slice(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input = converter.converted_tensor(nnef_op.input)
    output = converter.converted_tensor(nnef_op.output)

    axes = []
    starts = []
    ends = []
    for axis, begin, end in sorted(zip(nnef_op.attribs['axes'], nnef_op.attribs['begin'], nnef_op.attribs['end'])):
        assert axis >= 0
        if begin < 0:
            begin += input.shape[axis]
        if end <= 0:
            end += input.shape[axis]
        if end == input.shape[axis]:
            end = -1
        if (begin, end) != (0, -1):
            axes.append(axis)
            starts.append(begin)
            ends.append(end)

    if not axes:
        Caffe2Operation(graph=caffe2_graph,
                        name='Copy',
                        inputs=input,
                        outputs=output)
    for i, (axis, start, end) in enumerate(zip(axes, starts, ends)):
        starts_attr = [0] * input.rank
        starts_attr[axis] = start
        ends_attr = [-1] * input.rank
        ends_attr[axis] = end

        tmp = output if i == len(axes) - 1 else Caffe2Tensor(graph=caffe2_graph,
                                                             shape=infer.slice(input=input.shape,
                                                                               begin=[start],
                                                                               end=[0 if end == -1 else end],
                                                                               axes=[axis],
                                                                               zero_means_all=True),
                                                             dtype=input.dtype)

        Caffe2Operation(graph=caffe2_graph,
                        name='Slice',
                        inputs=input,
                        outputs=tmp,
                        attribs=dict(starts=starts_attr, ends=ends_attr))
        input = tmp


def convert_nearest_upsample(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))
    if input.rank != 4:
        raise utils.NNEFToolsException('nearest_upsample is only supported for 4D tensor')

    Caffe2Operation(graph=caffe2_graph,
                    name='ResizeNearest',
                    inputs=input,
                    outputs=output,
                    attribs=dict(width_scale=float(nnef_op.attribs['factor'][1]),
                                 height_scale=float(nnef_op.attribs['factor'][0])))


def convert_nearest_downsample(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))
    if input.rank != 4:
        raise utils.NNEFToolsException('nearest_downsample is only supported for 4D tensor')

    Caffe2Operation(graph=caffe2_graph,
                    name='ResizeNearest',
                    inputs=input,
                    outputs=output,
                    attribs=dict(width_scale=float(1.0 / nnef_op.attribs['factor'][1]),
                                 height_scale=float(1.0 / nnef_op.attribs['factor'][0])))


def convert_tile(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    repeats = nnef_op.attribs["repeats"]

    if all(r == 1 for r in repeats):
        Caffe2Operation(graph=caffe2_graph, name='Copy', inputs=input, outputs=output)
        return

    for axis, repeat in enumerate(repeats):
        if repeat != 1:
            input = Caffe2Operation(graph=caffe2_graph,
                                    name='Tile',
                                    attribs=dict(axis=axis, tiles=repeat),
                                    inputs=input,
                                    outputs=(output
                                             if all(r == 1 for r in repeats[axis + 1:])
                                             else Caffe2Tensor(graph=caffe2_graph,
                                                               shape=[s * repeat if a == axis else s for a, s in
                                                                      enumerate(input.shape)],
                                                               dtype=input.dtype))).output


def convert_pad(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    padding = nnef_op.attribs['padding']

    if (len(padding) >= 1 and padding[0] != (0, 0)) or (len(padding) >= 2 and padding[1] != (0, 0)):
        raise utils.NNEFToolsException("Padding is not supported in N,C dimensions")

    onnx_op = Caffe2Operation(graph=caffe2_graph,
                              name='PadImage',
                              inputs=input,
                              attribs=dict(mode=converter.nnef_border_to_caffe2_mode(nnef_op.attribs['border']),
                                           pads=converter.nnef_padding_to_caffe2_pads(padding[2:])),
                              outputs=output)

    if onnx_op.attribs['mode'] == 'constant':
        onnx_op.attribs['value'] = 0.0


def NONATOMIC(converter, nnef_op, caffe2_graph):
    # type: (Converter, NNEFOperation, Caffe2Graph)->None

    assert False, "This should not be called!"


_StandardConverters = {
    'moments': NONATOMIC,

    'abs': partial(generic_convert_unary, target_name='Abs'),
    'ceil': partial(generic_convert_unary, target_name='Ceil'),
    'copy': partial(generic_convert_unary, target_name='Copy'),
    'elu': partial(generic_convert_unary, target_name='Elu'),
    'exp': partial(generic_convert_unary, target_name='Exp'),
    'floor': partial(generic_convert_unary, target_name='Floor'),
    'log': partial(generic_convert_unary, target_name='Log'),
    'neg': partial(generic_convert_unary, target_name='Negative'),
    'not': partial(generic_convert_unary, target_name='Not'),
    'relu': partial(generic_convert_unary, target_name='Relu'),
    'sigmoid': partial(generic_convert_unary, target_name='Sigmoid'),
    'sign': partial(generic_convert_unary, target_name='Sign'),
    'softplus': partial(generic_convert_unary, target_name='Softplus'),
    'sqr': partial(generic_convert_unary, target_name='Sqr'),
    'sqrt': partial(generic_convert_unary, target_name='Sqrt'),
    'tanh': partial(generic_convert_unary, target_name='Tanh'),

    'add': partial(generic_convert_to_binary, target_name='Add', commutative=True),
    'and': partial(generic_convert_to_binary, target_name='And', commutative=True),
    'div': partial(generic_convert_to_binary, target_name='Div'),
    'eq': partial(generic_convert_to_binary, target_name='EQ', commutative=True),
    'ne': partial(generic_convert_to_binary, target_name='NE', commutative=True),
    'ge': partial(generic_convert_to_binary, target_name='GE'),
    'gt': partial(generic_convert_to_binary, target_name='GT'),
    'le': partial(generic_convert_to_binary, target_name='LE'),
    'lt': partial(generic_convert_to_binary, target_name='LT'),
    'mul': partial(generic_convert_to_binary, target_name='Mul', commutative=True),
    'or': partial(generic_convert_to_binary, target_name='Or', commutative=True),
    'sub': partial(generic_convert_to_binary, target_name='Sub'),
    'pow': partial(generic_convert_to_binary, target_name='Pow'),

    'avg_pool': partial(generic_convert_pool, target_name='AveragePool', allowed_borders=('ignore',)),
    'max_pool': partial(generic_convert_pool, target_name='MaxPool', allowed_borders=('ignore',)),
    '_lp_pool': partial(generic_convert_pool, target_name='LpPool', allowed_borders=('ignore', 'constant')),

    'l1_normalization': partial(generic_convert_lp_normalization, target_name='NormalizeL1'),
    'l2_normalization': partial(generic_convert_lp_normalization, target_name='Normalize'),

    'argmax_reduce': partial(generic_convert_reduce, multi_axis=False, target_name='ArgMax'),
    'argmin_reduce': partial(generic_convert_reduce, multi_axis=False, target_name='ArgMin'),
    'min_reduce': partial(generic_convert_reduce, multi_axis=True, target_name='ReduceMin'),
    'max_reduce': partial(generic_convert_reduce, multi_axis=True, target_name='ReduceMax'),
    'sum_reduce': partial(generic_convert_reduce,
                          multi_axis=True,
                          target_name='ReduceSum',
                          target_name_if_normalized='ReduceMean'),
    'mean_reduce': partial(generic_convert_reduce, multi_axis=True, target_name='ReduceMean'),

    'min': partial(convert_min_max, target_name='Min'),
    'max': partial(convert_min_max, target_name='Max'),

    'concat': convert_concat,
    'select': convert_select,
    'conv': convert_conv,
    'deconv': convert_deconv,
    'split': convert_split,
    'linear': convert_linear,
    'reshape': convert_reshape,
    'leaky_relu': convert_leaky_relu,
    'prelu': convert_prelu,
    'local_response_normalization': convert_local_response_normalization,
    'matmul': convert_matmul,
    'transpose': convert_transpose,
    'add_n': convert_add_n,
    'softmax': convert_softmax,
    'batch_normalization': convert_batch_normalization,
    'squeeze': convert_squeeze,
    'unsqueeze': convert_unsqueeze,
    'max_pool_with_index': convert_max_pool_with_index,
    'clamp': convert_clamp,
    'slice': convert_slice,
    'nearest_upsample': convert_nearest_upsample,
    'nearest_downsample': convert_nearest_downsample,

    "sin": partial(generic_convert_unary, target_name='Sin'),
    "cos": partial(generic_convert_unary, target_name='Cos'),
    "tile": convert_tile,
    "pad": convert_pad,
}

# NNEF must be parsed with this before calling nnef_to_caffe.Converter on it
ParserConfig = NNEFParserConfig(lowered=[k for k, v in six.iteritems(_StandardConverters) if v is NONATOMIC])
