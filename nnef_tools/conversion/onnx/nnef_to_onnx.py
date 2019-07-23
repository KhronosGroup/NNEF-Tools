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

from nnef_tools.conversion import converter
from nnef_tools.conversion import transforms
from nnef_tools.core import graph_utils
from nnef_tools.core import utils
from nnef_tools.io.nnef.nnef_graph import *
from nnef_tools.io.nnef.parser_config import NNEFParserConfig
from nnef_tools.io.onnx.onnx_graph import *
from nnef_tools.io.onnx.onnx_io import NumpyDTypeByONNXDType
from nnef_tools.shape_inference import shape_inference as infer

_onnx_dtype_by_numpy_dtype = {str(np.dtype(v)): k
                              for k, v in six.iteritems(NumpyDTypeByONNXDType)
                              if v is not None}

_onnx_dtype_by_nnef_dtype = {
    'scalar': 'FLOAT',
    'integer': 'INT64',
    'logical': 'BOOL',
}

_numpy_dtype_by_nnef_dtype = {
    'scalar': 'float32',
    'integer': 'int64',
    'logical': 'bool',
}

_onnx_pad_mode_by_nnef_border = {'constant': 'constant',
                                 'reflect': 'reflect',
                                 'replicate': 'edge'}


class Converter(converter.Converter[NNEFTensor, NNEFOperation, NNEFGraph,
                                    ONNXTensor, ONNXOperation, ONNXGraph]):

    def __init__(self,
                 custom_converter_by_op_name=None,
                 enable_default_conversion=False,
                 enable_imprecise_image_resize=False):
        converters = {}
        converters.update(_DefaultConverters)
        if custom_converter_by_op_name is not None:
            converters.update(custom_converter_by_op_name)
        default_op_converter = convert_default if enable_default_conversion else None

        super(Converter, self).__init__(op_converter_by_name=converters,
                                        default_op_converter=default_op_converter)

        self._enable_imprecise_image_resize = enable_imprecise_image_resize

    def create_graph(self, source_graph):
        # type: (NNEFGraph)->ONNXGraph

        return ONNXGraph(name=source_graph.name)

    def convert_tensor(self, source_tensor, target_graph):
        # type: (NNEFTensor, ONNXGraph)->ONNXTensor
        # TODO handle constants here
        return ONNXTensor(graph=target_graph,
                          name=source_tensor.name,
                          shape=list(source_tensor.shape),
                          dtype=(self.numpy_dtype_to_onnx_dtype(source_tensor.data.dtype)
                                 if isinstance(source_tensor.data, np.ndarray)
                                 else self.nnef_dtype_to_onnx_dtype(source_tensor.dtype)),
                          data=copy.copy(source_tensor.data))

    def convert_graph(self, source_graph):
        graph_utils.remove_unreachable(source_graph)
        target_graph = super(Converter, self).convert_graph(source_graph)  # type: NNEFGraph
        graph_utils.remove_unreachable(target_graph)
        target_graph.generate_missing_names()
        return target_graph

    @property
    def enable_imprecise_image_resize(self):
        return self._enable_imprecise_image_resize

    @staticmethod
    def numpy_dtype_to_onnx_dtype(numpy_dtype):
        return _onnx_dtype_by_numpy_dtype[str(np.dtype(numpy_dtype))]

    @staticmethod
    def nnef_dtype_to_onnx_dtype(nnef_dtype):
        return _onnx_dtype_by_nnef_dtype[nnef_dtype]

    @staticmethod
    def nnef_dtype_to_numpy_dtype(nnef_dtype):
        return np.dtype(_numpy_dtype_by_nnef_dtype[nnef_dtype])

    @staticmethod
    def add_squeeze(onnx_graph, onnx_tensor, axes, squeezed_tensor=None):
        # type: (ONNXGraph, ONNXTensor, typing.List[int], typing.Optional[ONNXTensor])->ONNXTensor
        if squeezed_tensor is None:
            squeezed_tensor = ONNXTensor(graph=onnx_graph,
                                         shape=transforms.squeezed_shape(onnx_tensor.shape, axes),
                                         dtype=onnx_tensor.dtype)
        return ONNXOperation(graph=onnx_graph,
                             name="Squeeze",
                             inputs=onnx_tensor,
                             attribs=dict(axes=axes),
                             outputs=squeezed_tensor).output

    @staticmethod
    def add_unsqueeze(onnx_graph, onnx_tensor, axes, unsqueezed_tensor=None):
        # type: (ONNXGraph, ONNXTensor, typing.List[int], typing.Optional[ONNXTensor])->ONNXTensor
        if unsqueezed_tensor is None:
            unsqueezed_tensor = ONNXTensor(graph=onnx_graph,
                                           shape=transforms.unsqueezed_shape(onnx_tensor.shape, axes),
                                           dtype=onnx_tensor.dtype)
        return ONNXOperation(graph=onnx_graph,
                             name="Unsqueeze",
                             inputs=onnx_tensor,
                             attribs=dict(axes=axes),
                             outputs=unsqueezed_tensor).output

    @staticmethod
    def add_expand(onnx_graph, onnx_tensor, shape):
        # type: (ONNXGraph, ONNXTensor, typing.List[int])->ONNXTensor
        return ONNXOperation(graph=onnx_graph,
                             name="Expand",
                             inputs=(onnx_tensor,
                                     ONNXTensor(graph=onnx_graph,
                                                shape=[4],
                                                dtype='INT64',
                                                data=list(shape))),
                             outputs=ONNXTensor(graph=onnx_graph,
                                                shape=list(shape),
                                                dtype=onnx_tensor.dtype)).output

    @staticmethod
    def onnx_pads(nnef_padding):
        # type: (typing.List[typing.Tuple[int, int]])->typing.List[int]
        return utils.concat_lists(utils.zip_inverse(2, nnef_padding))

    @staticmethod
    def onnx_pad_mode(nnef_border):
        assert nnef_border in _onnx_pad_mode_by_nnef_border
        return _onnx_pad_mode_by_nnef_border[nnef_border]

    @staticmethod
    def constant_1d_tensor(graph, list_, dtype):
        # type:(ONNXGraph,  typing.List[typing.Any], str)->ONNXTensor
        return ONNXTensor(graph=graph, shape=[len(list_)], dtype=dtype, data=list(list_))

    @staticmethod
    def constant_0d_tensor(graph, value, dtype):
        # type:(ONNXGraph,  typing.Any, str)->ONNXTensor
        return ONNXTensor(graph=graph, shape=[], dtype=dtype, data=[value])


def convert_default(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    print("Warning: Converter of {} is not implemented, doing default conversion.".format(nnef_op.name))

    def flatten(x):
        return utils.flatten(x) if isinstance(x, (list, tuple)) else x

    ONNXOperation(graph=onnx_graph,
                  name=nnef_op.name,
                  inputs=converter.converted_tensors(nnef_op.inputs),
                  attribs={k: flatten(v) for k, v in six.iteritems(nnef_op.attribs)},
                  outputs=converter.converted_tensors(nnef_op.outputs))


def UNSUPPORTED(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    if converter.default_op_converter is not None:
        converter.default_op_converter(converter, nnef_op, onnx_graph)
    else:
        raise utils.NNEFToolsException('NNEF to ONNX: Unsupported op: {}'.format(nnef_op.name))


def UNNEEDED(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    assert False, "This should not be called!"


def NONATOMIC(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    assert False, "This should not be called!"


def generic_convert_binary(converter, nnef_op, onnx_graph, target_name, broadcast_workaround=False):
    # type: (Converter, NNEFOperation, ONNXGraph, str, bool)->None

    if broadcast_workaround:
        x, y = converter.converted_tensors(nnef_op.inputs)
        z = converter.converted_tensor(nnef_op.output)

        # Caffe2 onnx backend cannot broadcast in Pow
        if x.shape != z.shape:
            x = converter.add_expand(onnx_graph, x, z.shape)
        if y.shape != z.shape:
            y = converter.add_expand(onnx_graph, y, z.shape)

        ONNXOperation(
            graph=onnx_graph,
            name=target_name,
            inputs=(x, y),
            outputs=z)
    else:
        x, y = converter.converted_tensors(nnef_op.inputs)
        z = converter.converted_tensor(nnef_op.output)

        ONNXOperation(
            graph=onnx_graph,
            name=target_name,
            inputs=(converter.add_unsqueeze(onnx_graph, x, list(range(x.rank, y.rank))) if 0 < x.rank < y.rank else x,
                    converter.add_unsqueeze(onnx_graph, y, list(range(y.rank, x.rank))) if 0 < y.rank < x.rank else y),
            outputs=z)


def generic_convert_unary(converter, nnef_op, onnx_graph, target_name, copy_attribs=None):
    # type: (Converter, NNEFOperation, ONNXGraph, str, typing.List[str])->None
    onnx_op = ONNXOperation(graph=onnx_graph,
                            name=target_name,
                            inputs=converter.converted_tensor(nnef_op.input),
                            outputs=converter.converted_tensor(nnef_op.output))
    if copy_attribs:
        for attr in copy_attribs:
            onnx_op.attribs[attr] = copy.deepcopy(nnef_op.attribs[attr])


def convert_lrn(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))
    nnef_size = nnef_op.attribs['size']

    assert len(nnef_size) >= 2 and all(dim == 1 or i == 1 for i, dim in enumerate(nnef_size)), \
        'Only channel LRN is supported'

    ONNXOperation(graph=onnx_graph,
                  name='LRN',
                  inputs=input,
                  outputs=output,
                  attribs=dict(size=nnef_size[1],
                               alpha=nnef_op.attribs['alpha'],
                               beta=nnef_op.attribs['beta'],
                               bias=nnef_op.attribs['bias']))


def convert_matmul(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None
    A, B = converter.converted_tensors(nnef_op.inputs)
    C = converter.converted_tensor(nnef_op.output)

    assert A.rank <= 2 and B.rank <= 2, "Batch matmul is unsupported in ONNX"

    ONNXOperation(graph=onnx_graph,
                  name='Gemm',
                  inputs=(A, B, converter.constant_0d_tensor(graph=onnx_graph, value=0.0, dtype=C.dtype)),
                  outputs=C,
                  attribs=dict(transA=1 if nnef_op.attribs['transposeA'] else 0,
                               transB=1 if nnef_op.attribs['transposeB'] else 0))


def convert_linear(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None
    A, B, C = converter.converted_tensors(nnef_op.inputs)
    D = converter.converted_tensor(nnef_op.output)

    assert A.rank <= 2 and B.rank <= 2 and C.rank <= 2, "Batch matmul is unsupported in ONNX"

    ONNXOperation(graph=onnx_graph,
                  name='Gemm',
                  inputs=(A, B, C),
                  outputs=D,
                  attribs=dict(transA=0, transB=1))


def partial_convert_conv_deconv(converter, nnef_op, onnx_graph, is_deconv, input, filter, bias, output):
    # type: (Converter, NNEFOperation, ONNXGraph, bool, ONNXTensor, ONNXTensor, ONNXTensor, ONNXTensor)->None

    strides = list(nnef_op.attribs['stride'])
    if not strides:
        strides = [1] * (input.rank - 2)

    dilations = nnef_op.attribs['dilation']
    if not dilations:
        dilations = [1] * (input.rank - 2)

    groups = nnef_op.attribs.get('groups', 1)  # default needed because box does not have groups
    if groups == 0:
        groups = input.shape[1]

    assert nnef_op.attribs['border'] == 'constant'

    if is_deconv:
        pads = nnef_op.attribs['padding']
        if not pads:  # auto pad
            calc_output_size = [i * s for i, s in (input.shape[2:], strides)]
            pads = infer.same_padding(upscaled_input=calc_output_size,
                                      filter=filter.shape[2:],
                                      stride=strides,
                                      dilation=dilations)
        else:
            calc_output_size = infer.conv(input=input.shape,
                                          filter=filter.shape[2:],
                                          padding=pads,
                                          stride=strides,
                                          dilation=dilations,
                                          groups=groups,
                                          output_channels=output.shape[1],
                                          format=infer.Format.NCHW,
                                          deconv=True)[2:]
        output_size = output.shape[2:]
        output_padding = [o - c for o, c in zip(output_size, calc_output_size)]
    else:
        pads = nnef_op.attribs['padding']
        if not pads:
            pads = infer.same_padding(upscaled_input=input.shape[2:],
                                      filter=filter.shape[2:],
                                      stride=strides,
                                      dilation=dilations)
        output_padding = [0] * len(pads)

    pads = converter.onnx_pads(pads)

    if bias.is_constant and bias.data == [0.0]:
        inputs = (input, filter)
    else:
        if bias.rank == 2:
            assert bias.shape[0] == 1
            bias = converter.add_squeeze(onnx_graph=onnx_graph, onnx_tensor=bias, axes=[0])
        inputs = (input, filter, bias)

    op = ONNXOperation(graph=onnx_graph,
                       name='ConvTranspose' if is_deconv else 'Conv',
                       inputs=inputs,
                       attribs=dict(
                           kernel_shape=filter.shape[2:],  # Not mandatory, but Caffe2 fails without this
                           strides=strides,
                           dilations=dilations,
                           pads=pads,
                           group=groups),
                       outputs=output)

    if is_deconv:
        op.attribs['output_padding'] = output_padding


def generic_convert_conv_deconv(converter, nnef_op, onnx_graph, is_deconv):
    # type: (Converter, NNEFOperation, ONNXGraph, bool)->None

    input, filter, bias = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)
    partial_convert_conv_deconv(converter, nnef_op, onnx_graph, is_deconv, input, filter, bias, output)


def convert_box(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if nnef_op.attribs['size'] == [1] * input.rank:
        onnx_op = ONNXOperation(graph=onnx_graph,
                                name='Pad',
                                inputs=input,
                                attribs=dict(mode=converter.onnx_pad_mode(nnef_op.attribs['border']),
                                             pads=converter.onnx_pads(nnef_op.attribs['padding'])),
                                outputs=output)

        if onnx_op.attribs['mode'] == 'constant':
            onnx_op.attribs['value'] = 0.0
        return

    if nnef_op.attribs['normalize']:
        partial_convert_pool(converter, nnef_op, onnx_graph,
                             target_name='AveragePool', input=input, outputs=(output,))
    else:
        temporary = ONNXTensor(graph=onnx_graph, shape=list(output.shape), dtype=output.dtype)
        partial_convert_pool(converter, nnef_op, onnx_graph,
                             target_name='AveragePool', input=input, outputs=(temporary,), force_constant=True)
        ONNXOperation(
            graph=onnx_graph,
            name='Mul',
            inputs=(temporary,
                    converter.constant_0d_tensor(onnx_graph, float(np.product(nnef_op.attribs['size'])), 'FLOAT')),
            outputs=output)


def generic_convert_pool(converter, nnef_op, onnx_graph, target_name):
    # type: (Converter, NNEFOperation, ONNXGraph, str)->None

    input = converter.converted_tensor(nnef_op.input)
    outputs = converter.converted_tensors(nnef_op.outputs)
    partial_convert_pool(converter, nnef_op, onnx_graph, target_name, input, outputs)


def partial_convert_pool(converter, nnef_op, onnx_graph, target_name, input, outputs, force_constant=False):
    # type: (Converter, NNEFOperation, ONNXGraph, str, ONNXTensor, typing.Tuple[ONNXTensor, ...], bool)->None
    if nnef_op.name == 'argmax_pool':
        outputs = (ONNXTensor(graph=onnx_graph, shape=list(outputs[0].shape), dtype=input.dtype),) + tuple(outputs)

    strides = list(nnef_op.attribs['stride'])
    if not strides:
        strides = [1] * input.rank
    dilations = nnef_op.attribs['dilation']
    if not dilations:
        dilations = [1] * input.rank

    assert nnef_op.attribs['border'] in ['constant', 'ignore']

    pads = nnef_op.attribs['padding']
    if not pads:
        pads = infer.same_padding(upscaled_input=input.shape[2:],
                                  filter=nnef_op.attribs['size'][2:],
                                  stride=strides[2:],
                                  dilation=dilations[2:])

    assert pads[:2] == [(0, 0), (0, 0)], "Padding in batch and channel dimensions is not supported in ONNX"
    pads = pads[2:]
    pads = converter.onnx_pads(pads)

    assert nnef_op.attribs['size'][:2] == strides[:2] == dilations[:2] == [1, 1], \
        'Pooling in batch and channel dimensions is not supported in ONNX'
    strides = strides[2:]
    dilations = dilations[2:]

    assert all(d == 1 for d in dilations), 'Dilation is not supported for pooling in ONNX'

    onnx_op = ONNXOperation(graph=onnx_graph,
                            name=target_name,
                            inputs=input,
                            attribs=dict(kernel_shape=nnef_op.attribs['size'][2:],
                                         pads=pads,
                                         strides=strides),
                            outputs=outputs)

    if target_name == 'AveragePool':
        onnx_op.attribs['count_include_pad'] = 1 if (nnef_op.attribs['border'] == 'constant' or force_constant) else 0


def convert_desample(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    input, indices = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    strides = list(nnef_op.attribs['stride'])
    if not strides:
        strides = [1] * input.rank
    dilations = nnef_op.attribs['dilation']
    if not dilations:
        dilations = [1] * input.rank

    assert nnef_op.attribs['border'] in ['constant', 'ignore']

    pads = nnef_op.attribs['padding']
    if not pads:  # auto pad
        calc_output_size = [i * s for i, s in (input.shape[2:], strides)]
        pads = infer.same_padding(upscaled_input=calc_output_size,
                                  filter=nnef_op.attribs['size'],
                                  stride=strides,
                                  dilation=dilations)

    assert pads[:2] == [(0, 0), (0, 0)], "Padding in batch and channel dimensions is not supported in ONNX"
    pads = pads[2:]
    pads = converter.onnx_pads(pads)

    assert nnef_op.attribs['size'][:2] == strides[:2] == dilations[:2] == [1, 1], \
        'Pooling in batch and channel dimensions is not supported in ONNX'
    strides = strides[2:]
    dilations = dilations[2:]

    assert all(d == 1 for d in dilations), 'Dilation is not supported for pooling in ONNX'

    ONNXOperation(graph=onnx_graph,
                  name='MaxUnpool',
                  inputs=(input, indices),
                  attribs=dict(kernel_shape=nnef_op.attribs['size'][2:],
                               pads=pads,
                               strides=strides),
                  outputs=output)


def convert_reshape(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    shape = nnef_op.attribs['shape']
    axis_start = nnef_op.attribs['axis_start']
    axis_count = nnef_op.attribs['axis_count']
    if axis_count == -1:
        axis_count = input.rank - axis_start

    onnx_shape = input.shape[:axis_start] + shape + input.shape[axis_start + axis_count:]

    ONNXOperation(graph=onnx_graph,
                  name='Reshape',
                  inputs=(input, converter.constant_1d_tensor(graph=onnx_graph,
                                                              list_=onnx_shape,
                                                              dtype='INT64')),
                  outputs=output)


def convert_transpose(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if not nnef_op.attribs['axes']:
        ONNXOperation(graph=onnx_graph, name='Identity', inputs=input, outputs=output)
        return

    perm = list(nnef_op.attribs['axes'])
    if len(perm) < input.rank:
        perm += list(range(len(perm), input.rank))

    ONNXOperation(graph=onnx_graph,
                  name='Transpose',
                  inputs=input,
                  attribs=dict(perm=perm),
                  outputs=output)


def convert_concat(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    inputs = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    ONNXOperation(graph=onnx_graph,
                  name='Concat',
                  inputs=inputs,
                  attribs=dict(axis=nnef_op.attribs['axis']),
                  outputs=output)


def convert_split(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    input = converter.converted_tensor(nnef_op.input)
    outputs = converter.converted_tensors(nnef_op.outputs)

    axis = nnef_op.attribs['axis']
    ratios = nnef_op.attribs['ratios']
    assert input.shape[axis] % sum(ratios) == 0
    unit = input.shape[axis] // sum(ratios)

    ONNXOperation(graph=onnx_graph,
                  name='Split',
                  inputs=input,
                  attribs=dict(axis=axis,
                               split=[ratio * unit for ratio in ratios]),
                  outputs=outputs)


def convert_stack(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    inputs = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    inputs = [converter.add_unsqueeze(onnx_graph, input, [nnef_op.attribs['axis']]) for input in inputs]

    ONNXOperation(graph=onnx_graph,
                  name='Concat',
                  inputs=inputs,
                  attribs=dict(axis=nnef_op.attribs['axis']),
                  outputs=output)


def convert_unstack(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    input = converter.converted_tensor(nnef_op.input)
    outputs = converter.converted_tensors(nnef_op.outputs)

    axis = nnef_op.attribs['axis']
    num_parts = input.shape[axis]

    def unsqueeze(shape, axis):
        shape = list(shape)
        shape.insert(axis, 1)
        return shape

    intermediates = [ONNXTensor(graph=onnx_graph, shape=unsqueeze(output.shape, axis), dtype=output.dtype)
                     for output in outputs]

    ONNXOperation(graph=onnx_graph,
                  name='Split',
                  inputs=input,
                  attribs=dict(axis=axis, split=[1] * num_parts),
                  outputs=intermediates)

    for intermediate, output in zip(intermediates, outputs):
        converter.add_unsqueeze(onnx_graph, intermediate, axis, unsqueezed_tensor=output)


def convert_slice(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if not nnef_op.attribs['axes']:
        ONNXOperation(graph=onnx_graph, name='Identity', inputs=input, outputs=output)
        return

    axes = list(nnef_op.attribs['axes'])
    starts = list(nnef_op.attribs['begin'])

    ends = list(utils.INT32_MAX if e == 0 else e for e in nnef_op.attribs['end'])

    ONNXOperation(graph=onnx_graph,
                  name='Slice',
                  inputs=input,
                  attribs=dict(axes=axes, starts=starts, ends=ends),
                  outputs=output)


def generic_convert_binary1_or_binary2(converter, nnef_op, onnx_graph, target_name1, target_name2):
    # type: (Converter, NNEFOperation, ONNXGraph, str, str)->None

    x, y = converter.converted_tensors(nnef_op.inputs)
    z = converter.converted_tensor(nnef_op.output)

    binary1 = ONNXOperation(
        graph=onnx_graph,
        name=target_name1,
        inputs=(converter.add_unsqueeze(onnx_graph, x, list(range(x.rank, y.rank))) if 0 < x.rank < y.rank else x,
                converter.add_unsqueeze(onnx_graph, y, list(range(y.rank, x.rank))) if 0 < y.rank < x.rank else y),
        outputs=ONNXTensor(graph=onnx_graph, shape=list(z.shape), dtype=z.dtype))

    binary2 = ONNXOperation(
        graph=onnx_graph,
        name=target_name2,
        inputs=(converter.add_unsqueeze(onnx_graph, x, list(range(x.rank, y.rank))) if 0 < x.rank < y.rank else x,
                converter.add_unsqueeze(onnx_graph, y, list(range(y.rank, x.rank))) if 0 < y.rank < x.rank else y),
        outputs=ONNXTensor(graph=onnx_graph, shape=list(z.shape), dtype=z.dtype))

    ONNXOperation(
        graph=onnx_graph,
        name="Or",
        inputs=(binary1.output, binary2.output),
        outputs=z)


def convert_ne(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    x, y = converter.converted_tensors(nnef_op.inputs)
    z = converter.converted_tensor(nnef_op.output)

    equal = ONNXOperation(
        graph=onnx_graph,
        name='Equal',
        inputs=(converter.add_unsqueeze(onnx_graph, x, list(range(x.rank, y.rank))) if 0 < x.rank < y.rank else x,
                converter.add_unsqueeze(onnx_graph, y, list(range(y.rank, x.rank))) if 0 < y.rank < x.rank else y),
        outputs=ONNXTensor(graph=onnx_graph, shape=list(z.shape), dtype=z.dtype))

    ONNXOperation(
        graph=onnx_graph,
        name="Not",
        inputs=equal.output,
        outputs=z)


def convert_round(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    add = ONNXOperation(graph=onnx_graph,
                        name='Add',
                        inputs=(input, converter.constant_0d_tensor(graph=onnx_graph, value=0.5, dtype=input.dtype)),
                        outputs=ONNXTensor(graph=onnx_graph, shape=list(output.shape), dtype=output.dtype))

    ONNXOperation(graph=onnx_graph,
                  name='Floor',
                  inputs=add.output,
                  outputs=output)


def convert_select(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    cond, true_value, false_value = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    max_rank = max(t.rank for t in [cond, true_value, false_value])

    def broadcast(t):
        return converter.add_unsqueeze(onnx_graph, t, list(range(t.rank, max_rank))) if 0 < t.rank < max_rank else t

    ONNXOperation(
        graph=onnx_graph,
        name='Where',
        inputs=(broadcast(cond), broadcast(true_value), broadcast(false_value)),
        outputs=output)


def convert_clamp(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    x, a, b = converter.converted_tensors(nnef_op.inputs)
    y = converter.converted_tensor(nnef_op.output)

    if a.is_constant and a.shape == [] and b.is_constant and b.shape == []:
        ONNXOperation(
            graph=onnx_graph,
            name='Clip',
            inputs=x,
            attribs=dict(min=a.data[0], max=b.data[0]),
            outputs=y)
    else:
        max_rank = max(t.rank for t in [x, a, b])

        def broadcast(t):
            return converter.add_unsqueeze(onnx_graph, t, list(range(t.rank, max_rank))) if 0 < t.rank < max_rank else t

        min = ONNXOperation(
            graph=onnx_graph,
            name='Min',
            inputs=(broadcast(x), broadcast(b)),
            outputs=ONNXTensor(graph=onnx_graph, shape=list(y.shape), dtype=y.dtype),
        )

        ONNXOperation(
            graph=onnx_graph,
            name='Max',
            inputs=(min.output, broadcast(a)),
            outputs=y,
        )


def convert_multilinear_upsample(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None
    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if nnef_op.attribs['method'] != 'symmetric':
        if converter.enable_imprecise_image_resize:
            print("Warning: method={} is unsupported in multilinear_upsample, "
                  "using symmetric, because enable_imprecise_image_resize was True"
                  .format(nnef_op.attribs["method"]))
        else:
            assert False, "Error: method={} is unsupported in multilinear_upsample. " \
                          "Use enable_imprecise_image_resize=True, to suppress this error." \
                .format(nnef_op.attribs["method"])

    if nnef_op.attribs['border'] != 'replicate':
        if converter.enable_imprecise_image_resize:
            print("Warning: border={} is unsupported in multilinear_upsample, "
                  "using replicate, because enable_imprecise_image_resize was True"
                  .format(nnef_op.attribs["border"]))
        else:
            assert False, "Error: border={} is unsupported in multilinear_upsample. " \
                          "Use enable_imprecise_image_resize=True, to suppress this error." \
                .format(nnef_op.attribs["border"])

    scales = [float(f) for f in [1, 1] + nnef_op.attribs['factor']]

    ONNXOperation(graph=onnx_graph,
                  name='Upsample',
                  inputs=(input, converter.constant_1d_tensor(graph=onnx_graph, list_=scales, dtype='FLOAT')),
                  attribs=dict(mode='linear'),
                  outputs=output)


def generic_convert_reduce(converter, nnef_op, onnx_graph, target_name, target_name_if_normalized="", one_axis=False):
    # type: (Converter, NNEFOperation, ONNXGraph, str, str, bool)->None
    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    if not nnef_op.attribs['axes']:
        ONNXOperation(graph=onnx_graph, name='Identity', inputs=input, outputs=output)
        return

    if one_axis:
        assert len(nnef_op.attribs['axes']) == 1, "{} supports only one axis in ONNX".format(target_name)

    onnx_op = ONNXOperation(graph=onnx_graph,
                            name=(target_name_if_normalized
                                  if target_name_if_normalized and nnef_op.attribs['normalize']
                                  else target_name),
                            inputs=input,
                            attribs=dict(keepdims=1),
                            outputs=output)

    if one_axis:
        onnx_op.attribs['axis'] = nnef_op.attribs['axes'][0]
    else:
        onnx_op.attribs['axes'] = list(nnef_op.attribs['axes'])


def convert_prelu(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None
    input, slope = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    ONNXOperation(graph=onnx_graph,
                  name='PRelu',
                  inputs=(input, converter.add_squeeze(onnx_graph, slope, [0])),
                  outputs=output)


def convert_batch_normalization(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    input, mean, variance, offset, scale = converter.converted_tensors(nnef_op.inputs)
    output = converter.converted_tensor(nnef_op.output)

    if input.rank < 3:  # Caffe2 BatchNorm works only for rank >= 3
        axes = list(range(input.rank, 3))
        input = converter.add_unsqueeze(onnx_graph=onnx_graph, onnx_tensor=input, axes=axes)
        final_output = output
        output = ONNXTensor(graph=onnx_graph, shape=list(input.shape), dtype=input.dtype)
        converter.add_squeeze(onnx_graph=onnx_graph, onnx_tensor=output, axes=axes, squeezed_tensor=final_output)

    ONNXOperation(graph=onnx_graph,
                  name='BatchNormalization',
                  inputs=(input,
                          converter.add_squeeze(onnx_graph, scale, [0]),
                          converter.add_squeeze(onnx_graph, offset, [0]),
                          converter.add_squeeze(onnx_graph, mean, [0]),
                          converter.add_squeeze(onnx_graph, variance, [0])),
                  attribs=dict(epsilon=nnef_op.attribs['epsilon']),
                  outputs=output)


def convert_copy_n(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    input = converter.converted_tensor(nnef_op.input)
    outputs = converter.converted_tensors(nnef_op.outputs)

    for output in outputs:
        ONNXOperation(graph=onnx_graph,
                      name='Identity',
                      inputs=input,
                      outputs=output)


def convert_nearest_upsample(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    scales = [float(f) for f in [1, 1] + nnef_op.attribs['factor']]

    ONNXOperation(graph=onnx_graph,
                  name='Upsample',
                  inputs=(input, converter.constant_1d_tensor(onnx_graph, scales, 'FLOAT')),
                  attribs=dict(mode='nearest'),
                  outputs=output)


def convert_tile(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    ONNXOperation(graph=onnx_graph,
                  name='Tile',
                  inputs=(input,
                          converter.constant_1d_tensor(graph=onnx_graph,
                                                       list_=nnef_op.attribs['repeats'],
                                                       dtype='INT64')),
                  outputs=output)


def convert_pad(converter, nnef_op, onnx_graph):
    # type: (Converter, NNEFOperation, ONNXGraph)->None

    input, output = converter.converted_tensors((nnef_op.input, nnef_op.output))

    onnx_op = ONNXOperation(graph=onnx_graph,
                            name='Pad',
                            inputs=input,
                            attribs=dict(mode=converter.onnx_pad_mode(nnef_op.attribs['border']),
                                         pads=converter.onnx_pads(nnef_op.attribs['padding'])),
                            outputs=output)

    if onnx_op.attribs['mode'] == 'constant':
        onnx_op.attribs['value'] = 0.0


_DefaultConverters = {
    'external': UNNEEDED,
    'variable': UNNEEDED,
    'constant': UNNEEDED,
    'update': UNSUPPORTED,
    'reshape': convert_reshape,
    'transpose': convert_transpose,
    'concat': convert_concat,
    'split': convert_split,
    'slice': convert_slice,
    'squeeze': partial(generic_convert_unary, target_name='Squeeze', copy_attribs=['axes']),
    'unsqueeze': partial(generic_convert_unary, target_name='Unsqueeze', copy_attribs=['axes']),
    'stack': convert_stack,
    'unstack': convert_unstack,
    'add': partial(generic_convert_binary, target_name='Add'),
    'sub': partial(generic_convert_binary, target_name='Sub'),
    'mul': partial(generic_convert_binary, target_name='Mul'),
    'div': partial(generic_convert_binary, target_name='Div'),
    'pow': partial(generic_convert_binary, target_name='Pow', broadcast_workaround=True),
    'exp': partial(generic_convert_unary, target_name='Exp'),
    'log': partial(generic_convert_unary, target_name='Log'),
    'abs': partial(generic_convert_unary, target_name='Abs'),
    'sign': partial(generic_convert_unary, target_name='Sign'),
    'rcp': partial(generic_convert_unary, target_name='Reciprocal'),
    'neg': partial(generic_convert_unary, target_name='Neg'),
    'copy': partial(generic_convert_unary, target_name='Identity'),
    'lt': partial(generic_convert_binary, target_name='Less'),
    'gt': partial(generic_convert_binary, target_name='Greater'),
    'le': partial(generic_convert_binary1_or_binary2, target_name1='Less', target_name2='Equal'),
    'ge': partial(generic_convert_binary1_or_binary2, target_name1='Greater', target_name2='Equal'),
    'eq': partial(generic_convert_binary, target_name='Equal'),
    'ne': convert_ne,
    'and': partial(generic_convert_binary, target_name='And'),
    'or': partial(generic_convert_binary, target_name='Or'),
    'not': partial(generic_convert_unary, target_name='Not'),
    'floor': partial(generic_convert_unary, target_name='Floor'),
    'ceil': partial(generic_convert_unary, target_name='Ceil'),
    'round': convert_round,
    'select': convert_select,
    'sqr': NONATOMIC,
    'sqrt': partial(generic_convert_unary, target_name='Sqrt'),
    'rsqr': NONATOMIC,
    'rsqrt': NONATOMIC,
    'log2': NONATOMIC,
    'min': partial(generic_convert_binary, target_name='Min', broadcast_workaround=True),
    'max': partial(generic_convert_binary, target_name='Max', broadcast_workaround=True),
    'clamp': convert_clamp,
    'matmul': convert_matmul,
    'conv': partial(generic_convert_conv_deconv, is_deconv=False),
    'deconv': partial(generic_convert_conv_deconv, is_deconv=True),
    'box': convert_box,
    'debox': UNSUPPORTED,
    'argmax_pool': partial(generic_convert_pool, target_name='MaxPool'),  # not typo
    'sample': UNSUPPORTED,
    'desample': convert_desample,
    'nearest_downsample': NONATOMIC,
    'area_downsample': NONATOMIC,
    'nearest_upsample': convert_nearest_upsample,
    'multilinear_upsample': convert_multilinear_upsample,
    'sum_reduce': partial(generic_convert_reduce, target_name='ReduceSum', target_name_if_normalized='ReduceMean'),
    'max_reduce': partial(generic_convert_reduce, target_name='ReduceMax'),
    'min_reduce': partial(generic_convert_reduce, target_name='ReduceMin'),
    'argmax_reduce': partial(generic_convert_reduce, target_name='ArgMax', one_axis=True),
    'argmin_reduce': partial(generic_convert_reduce, target_name='ArgMin', one_axis=True),
    'mean_reduce': partial(generic_convert_reduce, target_name='ReduceMean'),
    'moments': NONATOMIC,
    'relu': partial(generic_convert_unary, target_name='Relu'),
    'sigmoid': partial(generic_convert_unary, target_name='Sigmoid'),
    'tanh': partial(generic_convert_unary, target_name='Tanh'),
    'softabs': NONATOMIC,
    'softmax': partial(generic_convert_unary, target_name='Softmax'),
    'softplus': partial(generic_convert_unary, target_name='Softplus'),
    'elu': partial(generic_convert_unary, target_name='Elu'),
    'prelu': convert_prelu,
    'leaky_relu': partial(generic_convert_unary, target_name='LeakyRelu', copy_attribs=['alpha']),
    'max_pool_with_index': partial(generic_convert_pool, target_name='MaxPool'),  # not typo
    'max_pool': partial(generic_convert_pool, target_name='MaxPool'),
    'avg_pool': partial(generic_convert_pool, target_name='AveragePool'),
    'rms_pool': NONATOMIC,
    'linear': convert_linear,
    'separable_conv': NONATOMIC,
    'separable_deconv': NONATOMIC,
    'local_response_normalization': convert_lrn,
    'local_mean_normalization': NONATOMIC,
    'local_variance_normalization': NONATOMIC,
    'local_contrast_normalization': NONATOMIC,
    'l1_normalization': NONATOMIC,
    'l2_normalization': NONATOMIC,
    'batch_normalization': convert_batch_normalization,
    'avg_roi_pool': UNSUPPORTED,
    'max_roi_pool': UNSUPPORTED,  # maybe supported
    'roi_resample': UNSUPPORTED,
    'avg_roi_align': UNSUPPORTED,
    'max_roi_align': UNSUPPORTED,
    'linear_quantize': NONATOMIC,
    'logarithmic_quantize': NONATOMIC,
    'copy_n': convert_copy_n,
    'tile': convert_tile,
    'pad': convert_pad,
    'sin': partial(generic_convert_unary, target_name="Sin"),
    'cos': partial(generic_convert_unary, target_name="Cos"),
}

# TODO add to class as static(?) method
# NNEF must be parsed with this before calling nnef_to_tf.Converter on it
ParserConfig = NNEFParserConfig(lowered=[k for k, v in six.iteritems(_DefaultConverters) if v is NONATOMIC])
