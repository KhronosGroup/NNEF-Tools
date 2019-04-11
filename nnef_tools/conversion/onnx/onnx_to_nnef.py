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

import six

from nnef_tools.conversion import converter as _converter
from nnef_tools.conversion import transforms
from nnef_tools.conversion.onnx import onnx_to_nnef_trafos
from nnef_tools.core import graph_utils
from nnef_tools.core import utils
from nnef_tools.io.nnef.nnef_graph import *
from nnef_tools.io.nnef.nnef_io import NNEFDTypeByNumpyDType
from nnef_tools.io.onnx import onnx_shape_inference
from nnef_tools.io.onnx.onnx_graph import *
from nnef_tools.io.onnx.onnx_io import NumpyDTypeByONNXDType
from nnef_tools.shape_inference import shape_inference as infer

_nnef_dtype_by_onnx_dtype = {k: NNEFDTypeByNumpyDType[v.__name__]
                             for k, v in six.iteritems(NumpyDTypeByONNXDType)
                             if v is not None and v.__name__ in NNEFDTypeByNumpyDType}


class Converter(_converter.Converter[ONNXTensor, ONNXOperation, ONNXGraph,
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

    def create_graph(self, source_graph):
        # type: (ONNXGraph)->NNEFGraph

        return NNEFGraph(name=self._to_identifier_or_none(source_graph.name))

    def convert_tensor(self, source_tensor, target_graph):
        # type: (ONNXTensor, NNEFGraph)->NNEFTensor
        return NNEFTensor(graph=target_graph,
                          name=self._to_identifier_or_none(source_tensor.name),
                          shape=list(source_tensor.shape),
                          dtype=self.nnef_dtype(source_tensor.dtype),
                          data=copy.copy(source_tensor.data),
                          label=source_tensor.name if source_tensor.is_variable else None)

    def convert_graph(self, source_graph):
        graph_utils.remove_unreachable(source_graph)
        target_graph = super(Converter, self).convert_graph(source_graph)  # type: NNEFGraph
        # onnx_to_nnef_trafos.small_variables_to_consts(target_graph)
        onnx_to_nnef_trafos.merge_pads(target_graph)
        graph_utils.remove_unreachable(target_graph)
        target_graph.generate_missing_names()
        return target_graph

    @staticmethod
    def _to_identifier_or_none(s):
        if not any(c.isalpha() for c in s):
            return None
        cc = []
        for c in s:
            if not c.isalnum() and c != "_":
                c = "_"
            cc.append(c)
        s = ''.join(cc)
        if s[0] != '_' and not s[0].isalpha():
            s2 = "_" + s
            s = s2
        return s.lower()

    @staticmethod
    def nnef_dtype(onnx_dtype):
        assert onnx_dtype in _nnef_dtype_by_onnx_dtype
        return _nnef_dtype_by_onnx_dtype[onnx_dtype]

    @staticmethod
    def nnef_padding(custom_padding):
        return onnx_shape_inference.to_nnef_padding(custom_padding)

    @staticmethod
    def nnef_padding_ex(auto_padding, custom_padding, upscaled_shape, filter_shape, stride, dilation):
        return onnx_shape_inference.get_concrete_padding(
            auto_padding, custom_padding, upscaled_shape, filter_shape, stride, dilation)

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
    def broadcasted_from_right(input_shapes):
        return infer.elementwise(inputs=input_shapes, broadcast=infer.Broadcast.FROM_RIGHT)

    @staticmethod
    def nnef_border(onnx_border):
        return {'constant': 'constant',
                'reflect': 'reflect',
                'edge': 'replicate'}[onnx_border]

    @staticmethod
    def reduced_shape(input_shape, axes):
        return infer.reduce(input_shape, axes)

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


def convert_default(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    print("Warning: Converter of {} is not implemented, doing default conversion.".format(onnx_op.name))

    NNEFOperation(graph=nnef_graph,
                  name=onnx_op.name.replace('.', '_'),
                  inputs=converter.converted_tensors(onnx_op.inputs),
                  attribs=utils.recursive_transform(onnx_op.attribs, lambda x: x if x is not None else "None"),
                  outputs=converter.converted_tensors(onnx_op.outputs))


def generic_convert_unary(converter, onnx_op, nnef_graph, target_name, copy_attribs=None):
    # type: (Converter, ONNXOperation, NNEFGraph, str, typing.List[str])->None
    nnef_op = NNEFOperation(graph=nnef_graph,
                            name=target_name,
                            inputs=converter.converted_tensor(onnx_op.input),
                            outputs=converter.converted_tensor(onnx_op.output))
    if copy_attribs:
        for attr in copy_attribs:
            nnef_op.attribs[attr] = copy.deepcopy(onnx_op.attribs[attr])


def convert_lrn(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None
    # TODO make sure that size=4 is like o x o o in NNEF
    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))
    NNEFOperation(graph=nnef_graph,
                  name='local_response_normalization',
                  inputs=input,
                  outputs=output,
                  attribs=dict(size=[onnx_op.attribs['size'] if i == 1 else 1 for i in range(input.rank)],
                               alpha=onnx_op.attribs.get('alpha', 0.0001),
                               beta=onnx_op.attribs.get('beta', 0.75),
                               bias=onnx_op.attribs.get('bias', 1.0)))


def convert_softmax(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None
    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))
    NNEFOperation(graph=nnef_graph,
                  name='softmax',
                  inputs=input,
                  outputs=output,
                  attribs=dict(axes=list(range(onnx_op.attribs.get('axis', 1), input.rank))))


def convert_conv(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input, filter = converter.converted_tensors(onnx_op.inputs[:2])
    if len(onnx_op.inputs) < 3 or onnx_op.inputs[2].is_null:
        inputs = (input, filter)
    else:
        bias = converter.converted_tensor(onnx_op.inputs[2])
        bias = converter.add_unsqueeze(nnef_graph, bias, [0])
        inputs = (input, filter, bias)

    output = converter.converted_tensor(onnx_op.output)

    filter_size = filter.shape[2:]
    stride = onnx_op.attribs.get('strides', [1] * len(filter_size))
    dilation = onnx_op.attribs.get('dilations', [1] * len(filter_size))
    padding = converter.nnef_padding_ex(auto_padding=onnx_op.attribs.get('auto_pad'),
                                        custom_padding=onnx_op.attribs.get('pads'),
                                        upscaled_shape=input.shape[2:],
                                        filter_shape=filter_size,
                                        stride=stride,
                                        dilation=dilation)
    groups = onnx_op.attribs.get('group', 1)
    if groups == input.shape[1]:
        groups = 0

    NNEFOperation(graph=nnef_graph,
                  name='conv',
                  inputs=inputs,
                  attribs=dict(border='constant',
                               padding=padding,
                               stride=stride,
                               dilation=dilation,
                               groups=groups),
                  outputs=output)


def _get_reverse_sliding_window_padding(input_size, filter_size, stride, output_padding, output_shape,
                                        left_bigger=False):
    pads = [(0, 0)] * len(input_size)
    total_padding = [0] * len(input_size)
    for i in range(len(input_size)):
        total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + filter_size[i] - output_shape[i]
        if not left_bigger:
            pads[i] = (total_padding[i] // 2, total_padding[i] - (total_padding[i] // 2))
        else:
            pads[i] = (total_padding[i] // 2, total_padding[i] - (total_padding[i] // 2))
    return pads


def convert_conv_transpose(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input, filter = converter.converted_tensors(onnx_op.inputs[:2])
    if len(onnx_op.inputs) < 3 or onnx_op.inputs[2].is_null:
        inputs = (input, filter)
    else:
        bias = converter.converted_tensor(onnx_op.inputs[2])
        bias = converter.add_unsqueeze(nnef_graph, bias, [0])
        inputs = (input, filter, bias)

    output = converter.converted_tensor(onnx_op.output)

    groups = onnx_op.attribs.get('group', 1)
    if groups == input.shape[1]:
        groups = 0

    filter_size = filter.shape[2:]
    stride = onnx_op.attribs.get('strides', [1] * len(filter_size))
    dilation = onnx_op.attribs.get('dilations', [1] * len(filter_size))
    output_padding = onnx_op.attribs.get('output_padding', [0] * len(filter_size))

    if 'output_shape' in onnx_op.attribs:
        padding = _get_reverse_sliding_window_padding(input_size=input.shape[2:],
                                                      filter_size=filter_size,
                                                      stride=stride,
                                                      output_padding=output_padding,
                                                      output_shape=onnx_op.attribs['output_shape'],
                                                      left_bigger=onnx_op.attribs.get('auto_pad') == 'SAME_LOWER')
    else:
        padding = converter.nnef_padding_ex(auto_padding=onnx_op.attribs.get('auto_pad'),
                                            custom_padding=onnx_op.attribs.get('pads'),
                                            upscaled_shape=input.shape[2:],
                                            filter_shape=filter_size,
                                            stride=stride,
                                            dilation=dilation)

    NNEFOperation(graph=nnef_graph,
                  name='deconv',
                  inputs=inputs,
                  attribs=dict(border='constant',
                               padding=padding,
                               stride=stride,
                               dilation=dilation,
                               groups=groups,
                               output_shape=list(output.shape)),
                  outputs=output)


def convert_batch_normalization(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    assert len(onnx_op.outputs) == 1 or all(output.is_null for output in onnx_op.outputs), \
        "Batch normalization is only supported in 'test' mode"

    input, scale, bias, mean, variance = converter.converted_tensors(onnx_op.inputs)
    output = converter.converted_tensor(onnx_op.outputs[0])

    NNEFOperation(graph=nnef_graph,
                  name='batch_normalization',
                  inputs=(input,
                          converter.add_unsqueeze(nnef_graph, mean, [0]),
                          converter.add_unsqueeze(nnef_graph, variance, [0]),
                          converter.add_unsqueeze(nnef_graph, bias, [0]),
                          converter.add_unsqueeze(nnef_graph, scale, [0])),
                  attribs=dict(epsilon=onnx_op.attribs.get('epsilon', 1e-05)),
                  outputs=output)


def generic_convert_variadic(converter, onnx_op, nnef_graph, target_name, normalize):
    # type: (Converter, ONNXOperation, NNEFGraph, str, bool)->None

    inputs = converter.converted_tensors(onnx_op.inputs)
    output = converter.converted_tensor(onnx_op.output)

    assert len(inputs) >= 1
    if len(inputs) == 1:
        NNEFOperation(graph=nnef_graph, name='copy', inputs=inputs, outputs=output)
    else:
        left = inputs[0]

        for i, right in enumerate(inputs[1:]):
            if 1 + i == len(inputs) - 1 and not normalize:
                new_left = output
            else:
                new_left = NNEFTensor(graph=nnef_graph,
                                      name=None,
                                      shape=converter.broadcasted_from_right([left.shape, right.shape]),
                                      dtype=left.dtype)
            NNEFOperation(graph=nnef_graph,
                          name=target_name,
                          inputs=(left, right),
                          outputs=new_left)
            left = new_left

        if normalize:
            n_tensor = converter.create_constant_nnef_tensor(nnef_graph=nnef_graph, data=float(len(inputs)),
                                                             dtype=output.dtype)
            NNEFOperation(graph=nnef_graph,
                          name='div',
                          inputs=(left, n_tensor),
                          outputs=output)


def generic_convert_binary_with_axis(converter, onnx_op, nnef_graph, target_name):
    # type: (Converter, ONNXOperation, NNEFGraph, str)->None

    x, y = converter.converted_tensors(onnx_op.inputs)
    z = converter.converted_tensor(onnx_op.output)

    if 'axis' in onnx_op.attribs:
        if onnx_op.attribs['axis'] == 0:
            NNEFOperation(
                graph=nnef_graph,
                name=target_name,
                inputs=(x, y),
                outputs=z)
        else:
            NNEFOperation(
                graph=nnef_graph,
                name=target_name,
                inputs=(x, converter.add_unsqueeze(nnef_graph, y, list(range(onnx_op.attribs['axis'])))),
                outputs=z)

    else:
        assert converter.can_broadcast_from_right(x.shape, y.shape)
        NNEFOperation(
            graph=nnef_graph,
            name=target_name,
            inputs=(converter.add_unsqueeze(nnef_graph, x, list(range(y.rank - x.rank))) if 0 < x.rank < y.rank else x,
                    converter.add_unsqueeze(nnef_graph, y, list(range(x.rank - y.rank))) if 0 < y.rank < x.rank else y),
            outputs=z)


def generic_convert_global_pooling(converter, onnx_op, nnef_graph, target_name, before='', after=''):
    # type: (Converter, ONNXOperation, NNEFGraph, str, str, str)->None

    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))

    assert input.rank >= 3

    if before:
        input = NNEFOperation(graph=nnef_graph,
                              name=before,
                              inputs=input,
                              outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    if input.rank == 4:
        if after:
            final_output = output
            output = NNEFTensor(graph=nnef_graph, shape=list(final_output.shape), dtype=final_output.dtype)
            NNEFOperation(graph=nnef_graph,
                          name=after,
                          inputs=output,
                          outputs=final_output)

        NNEFOperation(graph=nnef_graph,
                      name=target_name,
                      inputs=input,
                      attribs=dict(axes=[2, 3]),
                      outputs=output)
    else:
        reduced = NNEFOperation(graph=nnef_graph,
                                name=target_name,
                                inputs=input,
                                attribs=dict(axes=list(range(2, input.rank))),
                                outputs=NNEFTensor(graph=nnef_graph,
                                                   name=None,
                                                   shape=input.shape[:2] + [1] * (input.rank - 2),
                                                   dtype=output.dtype)).output

        if after:
            reduced = NNEFOperation(graph=nnef_graph,
                                    name=after,
                                    inputs=reduced,
                                    outputs=NNEFTensor(graph=nnef_graph,
                                                       shape=list(reduced.shape),
                                                       dtype=reduced.dtype)).output

        if input.rank == 3:
            NNEFOperation(graph=nnef_graph,
                          name='unsqueeze',
                          inputs=reduced,
                          attribs=dict(axes=[3]),
                          outputs=output)
        else:
            NNEFOperation(graph=nnef_graph,
                          name='squeeze',
                          inputs=reduced,
                          attribs=dict(axes=list(range(4, input.rank))),
                          outputs=output)


def convert_global_lp_pooling(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None
    p = onnx_op.attribs.get('p', 2.0)
    assert p in [1.0, 2.0], 'Lp pooling is only supported for L1 and L2.'
    if p == 1.0:
        generic_convert_global_pooling(converter, onnx_op, nnef_graph, target_name='sum_reduce',
                                       before='abs')
    elif p == 2.0:
        generic_convert_global_pooling(converter, onnx_op, nnef_graph, target_name='sum_reduce',
                                       before='sqr', after='sqrt')
    else:
        assert False


def convert_reshape(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    if 'shape' in onnx_op.attribs:
        input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))
        NNEFOperation(graph=nnef_graph,
                      name='reshape',
                      inputs=input,
                      attribs=dict(shape=onnx_op.attribs['shape']),
                      outputs=output)
    else:
        onnx_input, onnx_shape = onnx_op.inputs
        input = converter.converted_tensor(onnx_input)
        output = converter.converted_tensor(onnx_op.output)

        NNEFOperation(graph=nnef_graph,
                      name='reshape',
                      inputs=input,
                      attribs=dict(shape=onnx_shape_inference.evaluate_shape_tensor_simple(onnx_shape)),
                      outputs=output)


def generic_convert_pool(converter, onnx_op, nnef_graph, target_name):
    # type: (Converter, ONNXOperation, NNEFGraph, str)->None

    input = converter.converted_tensor(onnx_op.input)
    outputs = converter.converted_tensors(onnx_op.outputs)

    filter_size = onnx_op.attribs['kernel_shape']
    stride = onnx_op.attribs.get('strides', [1] * len(filter_size))
    dilation = [1] * len(filter_size)
    padding = converter.nnef_padding_ex(auto_padding=onnx_op.attribs.get('auto_pad'),
                                        custom_padding=onnx_op.attribs.get('pads'),
                                        upscaled_shape=onnx_op.input.shape[2:],
                                        filter_shape=filter_size,
                                        stride=stride,
                                        dilation=dilation)

    assert len(outputs) in [1, 2]
    assert onnx_op.attribs.get('storage_order', 0) == 0, 'Only row major storage order is supported'

    NNEFOperation(graph=nnef_graph,
                  name=target_name + ('_with_index' if len(outputs) == 2 else ''),
                  inputs=input,
                  attribs=dict(size=[1, 1] + filter_size,
                               border='ignore' if onnx_op.attribs.get('count_include_pad', 0) == 0 else 'constant',
                               padding=[(0, 0), (0, 0)] + padding,
                               stride=[1, 1] + stride,
                               dilation=[1, 1] + dilation),
                  outputs=outputs)


def convert_matmul(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    A, B = converter.converted_tensors(onnx_op.inputs)
    if A.rank > 2:
        A = converter.add_squeeze(nnef_graph, A, list(range(2, A.rank)))
    if B.rank > 2:
        B = converter.add_squeeze(nnef_graph, B, list(range(2, B.rank)))
    Y = converter.converted_tensor(onnx_op.output)
    NNEFOperation(graph=nnef_graph,
                  name='matmul',
                  inputs=(A, B),
                  outputs=Y)


def convert_gemm(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    A, B, C = converter.converted_tensors(onnx_op.inputs)
    if A.rank > 2:
        A = converter.add_squeeze(nnef_graph, A, list(range(2, A.rank)))
    if B.rank > 2:
        B = converter.add_squeeze(nnef_graph, B, list(range(2, B.rank)))
    if C.rank > 2:
        C = converter.add_squeeze(nnef_graph, C, list(range(2, C.rank)))

    Y = converter.converted_tensor(onnx_op.output)
    alpha_tensor = converter.create_constant_nnef_tensor(nnef_graph=nnef_graph,
                                                         data=onnx_op.attribs.get('alpha', 1.0),
                                                         dtype=Y.dtype)
    beta_tensor = converter.create_constant_nnef_tensor(nnef_graph=nnef_graph,
                                                        data=onnx_op.attribs.get('beta', 1.0),
                                                        dtype=Y.dtype)

    nnef_op_matmul = NNEFOperation(
        graph=nnef_graph,
        name='matmul',
        inputs=(A, B),
        attribs=dict(transposeA=bool(onnx_op.attribs.get('transA', False)),
                     transposeB=bool(onnx_op.attribs.get('transB', False))),
        outputs=NNEFTensor(graph=nnef_graph, name=None, shape=list(Y.shape), dtype=Y.dtype))

    nnef_op_mul = NNEFOperation(
        graph=nnef_graph,
        name='mul',
        inputs=(alpha_tensor, nnef_op_matmul.output),
        outputs=NNEFTensor(graph=nnef_graph, name=None, shape=list(Y.shape), dtype=Y.dtype))

    mul2 = NNEFOperation(
        graph=nnef_graph,
        name='mul',
        inputs=(beta_tensor, C),
        outputs=NNEFTensor(graph=nnef_graph, name=None, shape=list(C.shape), dtype=C.dtype)).output

    if Y.rank - C.rank > 0:
        mul2 = converter.add_unsqueeze(nnef_graph, mul2, list(range(Y.rank - C.rank)))

    NNEFOperation(
        graph=nnef_graph,
        name='add',
        inputs=(nnef_op_mul.output, mul2),
        outputs=Y)


def convert_concat(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None
    inputs = list(converter.converted_tensors(onnx_op.inputs))
    output = converter.converted_tensor(onnx_op.output)

    NNEFOperation(graph=nnef_graph,
                  name='concat',
                  inputs=inputs,
                  attribs=dict(axis=onnx_op.attribs.get('axis', 1)),
                  outputs=output)


def convert_dropout(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    assert len(onnx_op.outputs) == 1 or len(onnx_op.outputs[1].consumers) == 0, 'Using dropout mask is not supported'

    if 'dropout' not in converter.displayed_warnings:
        print('Warning: One or more Dropouts have been eliminated from the graph.')
        converter.displayed_warnings.add('dropout')

    input, output = converter.converted_tensors((onnx_op.input, onnx_op.outputs[0]))

    NNEFOperation(graph=nnef_graph, name='copy', inputs=input, outputs=output)


def convert_image_scaler(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))

    bias = onnx_op.attribs.get('bias', [0.0] * input.shape[1])
    scale = onnx_op.attribs.get('scale', 0.0)

    scale_tensor = NNEFTensor(graph=nnef_graph, name=None, shape=[1, input.shape[1]], dtype=input.dtype, data=bias)

    mul = NNEFOperation(graph=nnef_graph,
                        name='mul',
                        inputs=(converter.create_constant_nnef_tensor(nnef_graph, scale, input.dtype),
                                input),
                        outputs=NNEFTensor(graph=nnef_graph,
                                           name=None,
                                           shape=list(input.shape),
                                           dtype=input.dtype))

    NNEFOperation(graph=nnef_graph,
                  name='add',
                  inputs=(mul.output, scale_tensor),
                  outputs=output)


def convert_flatten(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None
    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))
    axis = onnx_op.attribs.get('axis', 1)
    if axis < 0:
        axis += onnx_op.input.rank
    if axis == 1:
        NNEFOperation(graph=nnef_graph,
                      name='reshape',
                      inputs=input,
                      attribs=dict(shape=[0, -1]),
                      outputs=output)
    else:
        NNEFOperation(graph=nnef_graph,
                      name='reshape',
                      inputs=input,
                      attribs=dict(shape=[infer.volume(input.shape[:axis]), -1]),
                      outputs=output)


def convert_transpose(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None
    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))
    perm = onnx_op.attribs['perm'] if 'perm' in onnx_op.attribs else list(reversed(range(input.rank)))
    NNEFOperation(graph=nnef_graph, name='transpose', inputs=input, attribs=dict(axes=perm), outputs=output)


def convert_identity(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None
    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))
    NNEFOperation(graph=nnef_graph, name='copy', inputs=input, outputs=output)


def convert_pad(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None
    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))

    value = onnx_op.attribs.get('value', 0.0)

    if 'paddings' in onnx_op.attribs:
        paddings = onnx_op.attribs['paddings']
    elif 'pads' in onnx_op.attribs:
        paddings = onnx_op.attribs['pads']
    else:
        assert False

    paddings = converter.nnef_padding(paddings)

    border = converter.nnef_border(onnx_op.attribs.get("mode", 'constant'))

    if all(p == 0 and q == 0 for p, q in paddings):
        NNEFOperation(graph=nnef_graph, name='copy', inputs=input, outputs=output)
        return

    box_op = NNEFOperation(
        graph=nnef_graph,
        name="box",
        inputs=input,
        attribs=dict(size=[1] * input.rank,
                     border=border,
                     padding=paddings),
        outputs=output)

    if value != 0:
        box_op.attribs['_value'] = value


def convert_prelu(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None
    input, slope = converter.converted_tensors(onnx_op.inputs)
    output = converter.converted_tensor(onnx_op.output)

    if slope.rank == 1:
        slope = converter.add_unsqueeze(nnef_graph, slope, [0])

    NNEFOperation(graph=nnef_graph,
                  name='prelu',
                  inputs=(input, slope),
                  outputs=output)


def generic_convert_reduce(converter, onnx_op, nnef_graph, target_name, multi_axis, before='', after=''):
    # type: (Converter, ONNXOperation, NNEFGraph, str, bool, str, str)->None

    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))

    if before:
        input = NNEFOperation(graph=nnef_graph,
                              name=before,
                              inputs=input,
                              outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    if not multi_axis:
        axis = onnx_op.attribs.get('axis', 0)
        if axis < 0:
            axis += input.rank
        axes = [axis]
    else:
        axes = onnx_op.attribs.get('axes', list(range(input.rank)))

    keepdims = onnx_op.attribs.get('keepdims', 1)

    if keepdims:
        if after:
            final_output = output
            output = NNEFTensor(graph=nnef_graph, shape=list(final_output.shape), dtype=final_output.dtype)
            NNEFOperation(graph=nnef_graph,
                          name=after,
                          inputs=output,
                          outputs=final_output)

        NNEFOperation(graph=nnef_graph,
                      name=target_name,
                      inputs=input,
                      attribs=dict(axes=axes),
                      outputs=output)
    else:

        reduce = NNEFOperation(graph=nnef_graph,
                               name=target_name,
                               inputs=input,
                               attribs=dict(axes=axes),
                               outputs=NNEFTensor(graph=nnef_graph,
                                                  name=None,
                                                  shape=converter.reduced_shape(input.shape, axes),
                                                  dtype=output.dtype))
        if after:
            reduce = NNEFOperation(graph=nnef_graph,
                                   name=after,
                                   inputs=output,
                                   outputs=NNEFTensor(graph=nnef_graph,
                                                      shape=list(reduce.output.shape),
                                                      dtype=reduce.output.dtype))

        NNEFOperation(graph=nnef_graph, name="squeeze", inputs=reduce.output, attribs=dict(axes=axes), outputs=output)


def convert_lp_normalization(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))

    p = onnx_op.attribs.get('p', 2.0)
    axis = onnx_op.attribs.get('axis', -1)
    if axis < 0:
        axis += input.rank

    assert p in [1.0, 2.0], 'Lp normalization is only supported for L1 and L2.'

    NNEFOperation(graph=nnef_graph,
                  name='l{}_normalization'.format(int(p)),
                  inputs=input,
                  attribs=dict(axes=[axis]),
                  outputs=output)


def convert_cast(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))

    if input.dtype == output.dtype:
        NNEFOperation(graph=nnef_graph, name="copy", inputs=input, outputs=output)
    elif input.dtype == "logical" and output.dtype == 'scalar':
        zeros = NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype='scalar', data=[0.0])
        ones = NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype='scalar', data=[1.0])
        NNEFOperation(graph=nnef_graph, name="select", inputs=(input, ones, zeros), outputs=output)
    elif input.dtype == "logical" and output.dtype == 'integer':
        zeros = NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype='integer', data=[0])
        ones = NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype='integer', data=[1])
        NNEFOperation(graph=nnef_graph, name="select", inputs=(input, ones, zeros), outputs=output)
    elif input.dtype == "scalar" and output.dtype == "logical":
        zeros = NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype='scalar', data=[0.0])
        NNEFOperation(graph=nnef_graph, name="ne", inputs=(input, zeros), outputs=output)
    else:
        assert False, ("Unsupported cast: {} -> {}."
                       "Supported casts: cast to same type, logical->scalar, logical->integer, scalar->logical."
                       .format(input.dtype, output.dtype))


def convert_clip(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))

    min_tensor = converter.create_constant_nnef_tensor(nnef_graph,
                                                       data=onnx_op.attribs.get('min', -3.4028234663852886e+38),
                                                       dtype='scalar')
    max_tensor = converter.create_constant_nnef_tensor(nnef_graph,
                                                       data=onnx_op.attribs.get('max', 3.4028234663852886e+38),
                                                       dtype='scalar')

    NNEFOperation(graph=nnef_graph,
                  name='clamp',
                  inputs=(input, min_tensor, max_tensor),
                  outputs=output)


def convert_constant_of_shape(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None
    # Not creating an operation for this
    converter.converted_tensor(onnx_op.output).data = [onnx_op.attribs['value']]


def convert_shape(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None
    # Not creating an operation for this
    converter.converted_tensor(onnx_op.output).data = list(onnx_op.input.shape)


def convert_size(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None
    # Not creating an operation for this
    converter.converted_tensor(onnx_op.output).data = [onnx_op.input.count]


def convert_elu(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))

    alpha = onnx_op.attribs.get('alpha', 1.0)

    if alpha == 1.0:
        NNEFOperation(graph=nnef_graph, name='elu', inputs=input, outputs=output)
    else:
        def create_tensor():
            return NNEFTensor(graph=nnef_graph, name=None, shape=list(input.shape), dtype=input.dtype)

        zero_tensor = converter.create_constant_nnef_tensor(nnef_graph=nnef_graph, data=0.0, dtype=input.dtype)
        one_tensor = converter.create_constant_nnef_tensor(nnef_graph=nnef_graph, data=1.0, dtype=input.dtype)
        alpha_tensor = converter.create_constant_nnef_tensor(nnef_graph=nnef_graph, data=alpha, dtype=input.dtype)

        exp = NNEFOperation(graph=nnef_graph, name='exp', inputs=input, outputs=create_tensor())
        sub = NNEFOperation(graph=nnef_graph, name='sub', inputs=(exp.output, one_tensor), outputs=create_tensor())
        mul = NNEFOperation(graph=nnef_graph, name='mul', inputs=(alpha_tensor, sub.output), outputs=create_tensor())
        less = NNEFOperation(graph=nnef_graph, name='less', inputs=(input, zero_tensor), outputs=create_tensor())
        NNEFOperation(graph=nnef_graph, name='select', inputs=(less.output, mul.output, input), outputs=output)


def convert_expand(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None
    onnx_input, onnx_shape = onnx_op.inputs
    input = converter.converted_tensor(onnx_input)
    shape = onnx_shape_inference.evaluate_shape_tensor_simple(onnx_shape)
    output = converter.converted_tensor(onnx_op.output)

    zero_tensor = NNEFTensor(graph=nnef_graph,
                             name=None,
                             shape=list(shape),
                             dtype=input.dtype,
                             data=[converter.nnef_zero_value(input.dtype)])
    NNEFOperation(
        graph=nnef_graph,
        name=converter.nnef_addition_op(input.dtype),
        inputs=((converter.add_unsqueeze(nnef_graph, input, list(range(zero_tensor.rank - input.rank)))
                 if 0 < input.rank < zero_tensor.rank else input),
                (converter.add_unsqueeze(nnef_graph, zero_tensor, list(range(input.rank - zero_tensor.rank)))
                 if 0 < zero_tensor.rank < input.rank else zero_tensor)),
        outputs=output)


def convert_max_unpool(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input, index = converter.converted_tensors(onnx_op.inputs[:2])
    output_shape = (onnx_shape_inference.evaluate_shape_tensor_simple(onnx_op.inputs[2])
                    if len(onnx_op.inputs) >= 3 and not onnx_op.inputs[2].is_null else None)
    output = converter.converted_tensor(onnx_op.output)

    filter_size = onnx_op.attribs['kernel_shape']
    stride = onnx_op.attribs.get('strides', [1] * len(filter_size))
    dilation = [1] * len(filter_size)

    if output_shape is not None:
        padding = _get_reverse_sliding_window_padding(input_size=input.shape[2:],
                                                      filter_size=filter_size,
                                                      stride=stride,
                                                      output_padding=[0] * len(filter_size),
                                                      output_shape=output_shape,
                                                      left_bigger=False)
    else:
        padding = converter.nnef_padding(onnx_op.attribs.get('pads', [0] * 2 * len(filter_size)))

    NNEFOperation(graph=nnef_graph,
                  name='desample',
                  inputs=(input, index),
                  attribs=dict(size=[1, 1] + filter_size,
                               border='constant',
                               padding=[(0, 0), (0, 0)] + padding,
                               stride=[1, 1] + stride,
                               dilation=[1, 1] + dilation,
                               output_shape=output.shape),
                  outputs=output)


def convert_slice(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))

    starts = onnx_op.attribs['starts']
    ends = onnx_op.attribs['ends']
    axes = onnx_op.attribs.get('axes', list(range(len(starts))))

    ends = [0 if end >= input.shape[axis] else end for end, axis in zip(ends, axes)]

    NNEFOperation(graph=nnef_graph,
                  name='slice',
                  inputs=input,
                  attribs=dict(axes=axes, begin=starts, end=ends),
                  outputs=output)


def convert_split(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input = converter.converted_tensor(onnx_op.input)
    outputs = converter.converted_tensors(onnx_op.outputs)
    axis = onnx_op.attribs.get('axis', 0)

    if 'split' in onnx_op.attribs:
        ratios = onnx_op.attribs['split']
    else:
        assert input.shape[axis] % len(onnx_op.outputs) == 0
        ratios = [1] * len(onnx_op.outputs)

    NNEFOperation(graph=nnef_graph,
                  name='split',
                  inputs=input,
                  attribs=dict(axis=axis, ratios=ratios),
                  outputs=list(outputs))


def convert_squeeze(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))

    axes = onnx_op.attribs.get('axes', None)
    if axes is None:
        axes = [i for i, dim in enumerate(input.shape) if dim == 1]

    NNEFOperation(graph=nnef_graph,
                  name='squeeze',
                  inputs=input,
                  attribs=dict(axes=axes),
                  outputs=output)


def convert_unsqueeze(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))

    axes = onnx_op.attribs['axes']

    NNEFOperation(graph=nnef_graph,
                  name='unsqueeze',
                  inputs=input,
                  attribs=dict(axes=axes),
                  outputs=output)


def convert_tile(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    if len(onnx_op.inputs) == 3:
        onnx_input, onnx_tiles, onnx_axis = onnx_op.inputs

        input = converter.converted_tensor(onnx_input)
        output = converter.converted_tensor(onnx_op.output)
        tiles = onnx_shape_inference.evaluate_scalar_int_tensor_simple(onnx_tiles)
        axis = onnx_shape_inference.evaluate_scalar_int_tensor_simple(onnx_axis)

        repeats = [1] * input.rank
        repeats[axis] = tiles

    elif len(onnx_op.inputs) == 2:
        onnx_input, onnx_repeats = onnx_op.inputs

        input = converter.converted_tensor(onnx_input)
        output = converter.converted_tensor(onnx_op.output)
        repeats = onnx_shape_inference.evaluate_shape_tensor_simple(onnx_repeats)
    else:
        assert False, 'Tile is only supported with 2 or 3 inputs'

    input_shape = input.shape
    input_dtype = input.dtype

    assert input.rank == len(repeats)

    if input_dtype == 'scalar':
        broadcasts = [m if s == 1 and m != 1 else 1 for m, s in zip(repeats, input_shape)]
        concats = [m if s != 1 and m != 1 else 1 for m, s in zip(repeats, input_shape)]
    else:
        broadcasts = [1] * len(repeats)
        concats = list(repeats)

    needs_broadcast = any(b != 1 for b in broadcasts)
    needed_concats = sum(c != 1 for c in concats)

    if not needs_broadcast and not needed_concats:
        NNEFOperation(graph=nnef_graph, name="copy", inputs=input, outputs=output)
        return

    if needs_broadcast:
        zeros = NNEFOperation(graph=nnef_graph,
                              name="constant",
                              inputs=input,
                              attribs=dict(shape=list(broadcasts), value=[0.0]),
                              outputs=NNEFTensor(graph=nnef_graph,
                                                 shape=list(broadcasts),
                                                 dtype=input_dtype))

        if needed_concats:
            add_output = NNEFTensor(graph=nnef_graph,
                                    shape=[s * b for s, b in zip(input_shape, broadcasts)],
                                    dtype=input_dtype)
        else:
            add_output = output

        NNEFOperation(graph=nnef_graph,
                      name="add",
                      inputs=(input, zeros.output),
                      outputs=add_output)
        input = add_output

    if needed_concats:
        concats_created = 0
        for i, c in enumerate(concats):
            if c != 1:
                if concats_created < needed_concats - 1:
                    concat_output = NNEFTensor(graph=nnef_graph,
                                               name=None,
                                               shape=[s_ * b_ * c_ if j_ <= i else s_ * b_
                                                      for j_, (s_, b_, c_)
                                                      in enumerate(zip(input_shape, broadcasts, concats))],
                                               dtype=input_dtype)
                else:
                    concat_output = output

                NNEFOperation(graph=nnef_graph,
                              name="concat",
                              inputs=[input] * c,
                              attribs=dict(axis=i),
                              outputs=concat_output)

                input = concat_output
                concats_created += 1

        if 'tile' not in converter.displayed_warnings:
            print("Warning: Simulating tile with {} concats.".format(concats_created))
            converter.displayed_warnings.add('tile')


def convert_upsample(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input = converter.converted_tensor(onnx_op.inputs[0])
    output = converter.converted_tensor(onnx_op.output)

    if 'scales' in onnx_op.attribs:
        scales = onnx_op.attribs['scales']
    else:
        scales = onnx_shape_inference.evaluate_float_list_tensor_simple(onnx_op.inputs[1])

    assert all(s == int(s) for s in scales), 'Only integer scales are supported'
    assert scales[:2] == [1, 1], 'Scale is only supported in spatial dimensions'

    mode = onnx_op.attribs.get('mode', 'nearest').lower()
    assert mode in ['nearest', 'linear']

    if mode == 'nearest':
        NNEFOperation(graph=nnef_graph,
                      name='nearest_upsample',
                      inputs=input,
                      attribs=dict(factor=[utils.anyint_to_int(s) for s in scales[2:]]),
                      outputs=output)
    elif mode == 'linear':
        NNEFOperation(graph=nnef_graph,
                      name='multilinear_upsample',
                      inputs=input,
                      attribs=dict(factor=[utils.anyint_to_int(s) for s in scales[2:]],
                                   method='symmetric',
                                   border='replicate'),
                      outputs=output)


def convert_where(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    cond, true_val, false_val = converter.converted_tensors(onnx_op.inputs)
    output = converter.converted_tensor(onnx_op.output)

    max_rank = max(cond.rank, true_val.rank, false_val.rank)

    def broadcast(tensor):
        return (converter.add_unsqueeze(nnef_graph, tensor, list(range(max_rank - tensor.rank)))
                if 0 < tensor.rank < max_rank else tensor)

    NNEFOperation(graph=nnef_graph,
                  name='select',
                  inputs=(broadcast(cond), broadcast(true_val), broadcast(false_val)),
                  outputs=output)


def convert_xor(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    x, y = converter.converted_tensors(onnx_op.inputs)
    z = converter.converted_tensor(onnx_op.output)

    x = converter.add_unsqueeze(nnef_graph, x, list(range(y.rank - x.rank))) if 0 < x.rank < y.rank else x
    y = converter.add_unsqueeze(nnef_graph, y, list(range(x.rank - y.rank))) if 0 < y.rank < x.rank else y

    not_x = NNEFOperation(graph=nnef_graph,
                          name='not',
                          inputs=x,
                          outputs=NNEFTensor(graph=nnef_graph, shape=list(x.shape), dtype=x.dtype)).output

    not_y = NNEFOperation(graph=nnef_graph,
                          name='not',
                          inputs=y,
                          outputs=NNEFTensor(graph=nnef_graph, shape=list(y.shape), dtype=y.dtype)).output

    and1 = NNEFOperation(graph=nnef_graph,
                         name='and',
                         inputs=(x, not_y),
                         outputs=NNEFTensor(graph=nnef_graph, shape=list(z.shape), dtype=z.dtype)).output

    and2 = NNEFOperation(graph=nnef_graph,
                         name='and',
                         inputs=(not_x, y),
                         outputs=NNEFTensor(graph=nnef_graph, shape=list(z.shape), dtype=z.dtype)).output

    NNEFOperation(graph=nnef_graph,
                  name='or',
                  inputs=(and1, and2),
                  outputs=z)


def convert_depth_to_space(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    x, y = converter.converted_tensors((onnx_op.input, onnx_op.output))

    n, c, h, w = x.shape

    block_size = onnx_op.attribs['blocksize']
    shape1 = [n, block_size, block_size, c // (block_size ** 2), h, w]
    reshape1 = NNEFOperation(
        graph=nnef_graph,
        name='reshape',
        inputs=x,
        attribs=dict(shape=shape1),
        outputs=NNEFTensor(graph=nnef_graph, shape=list(shape1), dtype=x.dtype))

    axes = [0, 3, 4, 1, 5, 2]
    transpose = NNEFOperation(
        graph=nnef_graph,
        name='transpose',
        attribs=dict(axes=axes),
        inputs=reshape1.output,
        outputs=NNEFTensor(graph=nnef_graph, shape=infer.transpose(input=shape1, axes=axes), dtype=x.dtype))

    shape2 = [n, c // (block_size ** 2), h * block_size, w * block_size]

    NNEFOperation(
        graph=nnef_graph,
        name='reshape',
        inputs=transpose.output,
        attribs=dict(shape=shape2),
        outputs=y)


def convert_hard_sigmoid(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    x, y = converter.converted_tensors((onnx_op.input, onnx_op.output))
    alpha_tensor = converter.create_constant_nnef_tensor(nnef_graph, onnx_op.attribs.get('alpha', 0.2), dtype=x.dtype)
    beta_tensor = converter.create_constant_nnef_tensor(nnef_graph, onnx_op.attribs.get('beta', 0.5), dtype=x.dtype)
    zero_tensor = converter.create_constant_nnef_tensor(nnef_graph, 0.0, dtype=x.dtype)
    one_tensor = converter.create_constant_nnef_tensor(nnef_graph, 1.0, dtype=x.dtype)

    mul = NNEFOperation(
        graph=nnef_graph,
        name='mul',
        inputs=(alpha_tensor, x),
        outputs=NNEFTensor(graph=nnef_graph, shape=list(x.shape), dtype=x.dtype))

    add = NNEFOperation(graph=nnef_graph,
                        name='add',
                        inputs=(mul.output, beta_tensor),
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(x.shape), dtype=x.dtype))

    NNEFOperation(graph=nnef_graph,
                  name='clamp',
                  inputs=(add.output, zero_tensor, one_tensor),
                  outputs=y)


def convert_instance_normalization(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input, scale, bias = converter.converted_tensors(onnx_op.inputs)
    output = converter.converted_tensor(onnx_op.output)

    epsilon_tensor = converter.create_constant_nnef_tensor(nnef_graph,
                                                           onnx_op.attribs.get('epsilon', 1e-5),
                                                           dtype=input.dtype)
    scale = converter.add_unsqueeze(nnef_graph=nnef_graph, nnef_tensor=scale, axes=[0])
    bias = converter.add_unsqueeze(nnef_graph=nnef_graph, nnef_tensor=bias, axes=[0])

    axes = list(range(2, input.rank))
    mean, variance = NNEFOperation(
        graph=nnef_graph,
        name='moments',
        inputs=input,
        attribs=dict(axes=axes),
        outputs=(NNEFTensor(graph=nnef_graph, shape=infer.reduce(input=input.shape, axes=axes), dtype=input.dtype),
                 NNEFTensor(graph=nnef_graph,
                            shape=infer.reduce(input=input.shape, axes=axes),
                            dtype=input.dtype))).outputs

    sub = NNEFOperation(graph=nnef_graph,
                        name='sub',
                        inputs=(input, mean),
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    mul = NNEFOperation(graph=nnef_graph,
                        name='mul',
                        inputs=(scale, sub),
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    add = NNEFOperation(graph=nnef_graph,
                        name='add',
                        inputs=(variance, epsilon_tensor),
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(variance.shape), dtype=variance.dtype)).output

    sqrt = NNEFOperation(graph=nnef_graph,
                         name='sqrt',
                         inputs=add,
                         outputs=NNEFTensor(graph=nnef_graph, shape=list(add.shape), dtype=add.dtype)).output

    div = NNEFOperation(graph=nnef_graph,
                        name='div',
                        inputs=(mul, sqrt),
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(mul.shape), dtype=mul.dtype)).output

    NNEFOperation(graph=nnef_graph,
                  name='add',
                  inputs=(div, bias),
                  outputs=output)


def convert_log_softmax(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    input, output = converter.converted_tensors((onnx_op.input, onnx_op.output))
    softmax = NNEFOperation(graph=nnef_graph,
                            name='softmax',
                            inputs=input,
                            outputs=NNEFTensor(graph=nnef_graph, shape=list(output.shape), dtype=output.dtype),
                            attribs=dict(axes=list(range(onnx_op.attribs.get('axis', 1), input.rank)))).output
    NNEFOperation(graph=nnef_graph,
                  name='log',
                  inputs=softmax,
                  outputs=output)


def UNSUPPORTED(converter, onnx_op, nnef_graph):
    # type: (Converter, ONNXOperation, NNEFGraph)->None

    if converter.default_op_converter is not None:
        converter.default_op_converter(converter, onnx_op, nnef_graph)
    else:
        raise utils.NNEFToolsException('ONNX to NNEF: Unsupported op: {}'.format(onnx_op.name))


_StandardConverters = {
    'Abs': partial(generic_convert_unary, target_name='abs'),
    'Acos': UNSUPPORTED,
    'Acosh': UNSUPPORTED,
    'Add': partial(generic_convert_binary_with_axis, target_name='add'),
    'And': partial(generic_convert_binary_with_axis, target_name='and'),
    'ArgMax': partial(generic_convert_reduce, multi_axis=False, target_name='argmax_reduce'),
    'ArgMin': partial(generic_convert_reduce, multi_axis=False, target_name='argmin_reduce'),
    'Asin': UNSUPPORTED,
    'Asinh': UNSUPPORTED,
    'Atan': UNSUPPORTED,
    'Atanh': UNSUPPORTED,
    'AveragePool': partial(generic_convert_pool, target_name='avg_pool'),
    'BatchNormalization': convert_batch_normalization,
    'Cast': convert_cast,  # workaround
    'Ceil': partial(generic_convert_unary, target_name='ceil'),
    'Clip': convert_clip,
    'Compress': UNSUPPORTED,  # It will probably be converted to "where" if that is standardized
    'Concat': convert_concat,
    # Constant: no operation was made for this, so no need to convert
    'ConstantOfShape': convert_constant_of_shape,
    'Conv': convert_conv,
    'ConvTranspose': convert_conv_transpose,
    'Cos': UNSUPPORTED,
    'Cosh': UNSUPPORTED,
    'DepthToSpace': convert_depth_to_space,
    'Div': partial(generic_convert_binary_with_axis, target_name='div'),
    'Dropout': convert_dropout,
    'Elu': convert_elu,  # workaround when alpha != 1.0
    'Equal': partial(generic_convert_binary_with_axis, target_name='eq'),
    'Erf': UNSUPPORTED,
    'Exp': partial(generic_convert_unary, target_name='exp'),
    'Expand': partial(convert_expand),
    'EyeLike': UNSUPPORTED,
    'Flatten': convert_flatten,
    'Floor': partial(generic_convert_unary, target_name='floor'),
    'GRU': UNSUPPORTED,
    'Gather': UNSUPPORTED,
    'Gemm': convert_gemm,
    'GlobalAveragePool': partial(generic_convert_global_pooling, target_name='mean_reduce'),
    'GlobalLpPool': convert_global_lp_pooling,
    'GlobalMaxPool': partial(generic_convert_global_pooling, target_name='max_reduce'),
    'Greater': partial(generic_convert_binary_with_axis, target_name='gt'),
    'HardSigmoid': convert_hard_sigmoid,
    'HardMax': UNSUPPORTED,
    'Identity': convert_identity,
    'If': UNSUPPORTED,
    'InstanceNormalization': convert_instance_normalization,
    'IsNan': UNSUPPORTED,
    'LRN': convert_lrn,
    'LSTM': UNSUPPORTED,
    'LeakyRelu': partial(generic_convert_unary, target_name='leaky_relu', copy_attribs=['alpha']),
    'Less': partial(generic_convert_binary_with_axis, target_name='lt'),
    'Log': partial(generic_convert_unary, target_name='log'),
    'LogSoftmax': convert_log_softmax,
    'Loop': UNSUPPORTED,
    'LpNormalization': convert_lp_normalization,
    'LpPool': UNSUPPORTED,
    'MatMul': convert_matmul,
    'Max': partial(generic_convert_variadic, target_name='max', normalize=False),
    'MaxPool': partial(generic_convert_pool, target_name='max_pool'),
    'MaxRoiPool': UNSUPPORTED,  # maybe support?
    'MaxUnpool': convert_max_unpool,
    'Mean': partial(generic_convert_variadic, target_name='add', normalize=True),
    'Min': partial(generic_convert_variadic, target_name='min', normalize=False),
    'Mul': partial(generic_convert_binary_with_axis, target_name='mul'),
    'Multinomial': UNSUPPORTED,
    'Neg': partial(generic_convert_unary, target_name='neg'),
    'Not': partial(generic_convert_unary, target_name='not'),
    'OneHot': UNSUPPORTED,
    'Or': partial(generic_convert_binary_with_axis, target_name='or'),
    'PRelu': convert_prelu,
    'Pad': partial(convert_pad),  # workaround
    'Pow': partial(generic_convert_binary_with_axis, target_name='pow'),
    'RNN': UNSUPPORTED,
    'RandomNormal': UNSUPPORTED,
    'RandomNormalLike': UNSUPPORTED,
    'RandomUniform': UNSUPPORTED,
    'RandomUniformLike': UNSUPPORTED,
    'Reciprocal': partial(generic_convert_unary, target_name='rcp'),
    'ReduceL1': partial(generic_convert_reduce, multi_axis=True, target_name='sum_reduce', before='abs'),
    'ReduceL2': partial(generic_convert_reduce, multi_axis=True, target_name='sum_reduce', before='sqr', after='sqrt'),
    'ReduceLogSum': partial(generic_convert_reduce, multi_axis=True, target_name='sum_reduce', after='log'),
    'ReduceLogSumExp': partial(generic_convert_reduce, multi_axis=True, target_name='sum_reduce', before='exp',
                               after='log'),
    'ReduceMax': partial(generic_convert_reduce, multi_axis=True, target_name='max_reduce'),
    'ReduceMean': partial(generic_convert_reduce, multi_axis=True, target_name='mean_reduce'),
    'ReduceMin': partial(generic_convert_reduce, multi_axis=True, target_name='min_reduce'),
    'ReduceProd': UNSUPPORTED,
    'ReduceSum': partial(generic_convert_reduce, multi_axis=True, target_name='sum_reduce'),
    'ReduceSumSquare': partial(generic_convert_reduce, multi_axis=True, target_name='sum_reduce', before='sqr'),
    "Relu": partial(generic_convert_unary, target_name='relu'),
    'Reshape': convert_reshape,
    'Scan': UNSUPPORTED,
    'Scatter': UNSUPPORTED,  # consider support
    'Selu': UNSUPPORTED,
    'Shape': convert_shape,
    'Shrink': UNSUPPORTED,
    'Sigmoid': partial(generic_convert_unary, target_name='sigmoid'),
    'Sign': partial(generic_convert_unary, target_name='sign'),
    'Sin': UNSUPPORTED,
    'Sinh': UNSUPPORTED,
    'Size': convert_size,
    'Slice': convert_slice,
    'Softmax': convert_softmax,
    'Softplus': partial(generic_convert_unary, target_name='softplus'),
    'Softsign': UNSUPPORTED,
    'SpaceToDepth': UNSUPPORTED,
    'Split': convert_split,
    'Sqrt': partial(generic_convert_unary, target_name='sqrt'),
    'Squeeze': convert_squeeze,
    'Sub': partial(generic_convert_binary_with_axis, target_name='sub'),
    'Sum': partial(generic_convert_variadic, target_name='add', normalize=False),
    'Tan': UNSUPPORTED,
    'Tanh': partial(generic_convert_unary, target_name="tanh"),
    'Tile': partial(convert_tile),
    'TopK': UNSUPPORTED,
    'Transpose': convert_transpose,
    'Unsqueeze': convert_unsqueeze,
    'Upsample': convert_upsample,
    'Where': convert_where,
    'Xor': convert_xor,

    # Experimental ONNX ops (We will not support these in general!)
    # It appears in model zoo, so we support it. But the order of add and mul is not sure.
    'ImageScaler': convert_image_scaler,
}
