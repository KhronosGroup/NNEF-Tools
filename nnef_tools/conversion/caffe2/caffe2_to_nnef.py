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
import re
import typing
from functools import partial

import numpy as np

from nnef_tools.io.caffe2.caffe2_graph import *
from nnef_tools.conversion import converter as _converter
from nnef_tools.conversion import transforms
from nnef_tools.core import utils, graph_utils
from nnef_tools.io.nnef.nnef_graph import *
from nnef_tools.shape_inference import shape_inference as infer


class Converter(_converter.Converter[Caffe2Tensor, Caffe2Operation, Caffe2Graph,
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

        self.name_generator = utils.NameGenerator()

    def create_graph(self, source_graph):
        # type:(Caffe2Graph)->NNEFGraph
        return NNEFGraph(name=self.identifier_compatible_name(source_graph.name))

    def convert_tensor(self, source_tensor, target_graph):
        # type: (Caffe2Tensor, NNEFGraph)->NNEFTensor
        name = self.name_generator.get_new_name(self.identifier_compatible_name(source_tensor.name))
        return NNEFTensor(graph=target_graph,
                          name=name,
                          shape=list(source_tensor.shape),
                          dtype=self.caffe2_dtype_to_nnef(source_tensor.dtype),
                          data=copy.copy(source_tensor.data),
                          label=name if source_tensor.is_variable else None)

    def convert_graph(self, source_graph):
        # type: (Caffe2Graph)->NNEFGraph
        target_graph = super(Converter, self).convert_graph(source_graph)  # type: NNEFGraph
        graph_utils.remove_unreachable(target_graph)
        target_graph.generate_missing_names()
        return target_graph

    @staticmethod
    def identifier_compatible_name(name):
        return re.sub(r'\W', '_', name) if name is not None else None

    @staticmethod
    def caffe2_dtype_to_nnef(dtype):
        assert isinstance(dtype, str)
        if 'FLOAT' in dtype or 'DOUBLE' == dtype:
            return 'scalar'
        elif 'INT' in dtype or 'BYTE' == dtype:
            return 'integer'
        elif 'BOOL' == dtype:
            return 'logical'
        else:
            raise utils.NNEFToolsException('Unsupported Caffe2 dtype: {}'.format(dtype))

    @staticmethod
    def add_unsqueeze(nnef_graph, nnef_tensor, axes):
        # type: (NNEFGraph, NNEFTensor, typing.List[int])->NNEFTensor
        return NNEFOperation(graph=nnef_graph,
                             name="unsqueeze",
                             inputs=nnef_tensor,
                             attribs=dict(axes=list(axes)),
                             outputs=NNEFTensor(graph=nnef_graph,
                                                shape=transforms.unsqueezed_shape(nnef_tensor.shape, axes),
                                                dtype=nnef_tensor.dtype)).output

    @staticmethod
    def add_squeeze(nnef_graph, nnef_tensor, axes, squeezed_tensor=None):
        # type: (NNEFGraph, NNEFTensor, typing.List[int], typing.Optional[NNEFTensor])->NNEFTensor
        if squeezed_tensor is None:
            squeezed_tensor = NNEFTensor(graph=nnef_graph,
                                         shape=transforms.squeezed_shape(nnef_tensor.shape, axes),
                                         dtype=nnef_tensor.dtype)
        return NNEFOperation(graph=nnef_graph,
                             name="squeeze",
                             inputs=nnef_tensor,
                             attribs=dict(axes=list(axes)),
                             outputs=squeezed_tensor).output

    @staticmethod
    def add_reshape(nnef_graph, nnef_tensor, shape, reshaped_tensor=None):
        # type: (NNEFGraph, NNEFTensor, typing.List[int], NNEFTensor)->NNEFTensor

        inferred_shape = infer.reshape(nnef_tensor.shape, shape)
        reshaped_tensor = reshaped_tensor if reshaped_tensor else NNEFTensor(graph=nnef_graph,
                                                                             shape=list(inferred_shape),
                                                                             dtype=nnef_tensor.dtype)
        assert reshaped_tensor.shape == inferred_shape
        return NNEFOperation(graph=nnef_graph,
                             name="reshape",
                             inputs=nnef_tensor,
                             attribs=dict(shape=list(shape)),
                             outputs=reshaped_tensor).output

    @staticmethod
    def add_reshape_to_2d(nnef_graph, nnef_tensor, axis, use_neg_one=False, reshaped_tensor=None):
        # type: (NNEFGraph, NNEFTensor, int, bool, typing.Optional[NNEFTensor])->NNEFTensor

        shape = [utils.product(nnef_tensor.shape[:axis]), utils.product(nnef_tensor.shape[axis:])]
        shape_with_neg_one = [-1, utils.product(nnef_tensor.shape[axis:])]
        reshaped_tensor = reshaped_tensor if reshaped_tensor else NNEFTensor(graph=nnef_graph,
                                                                             shape=shape,
                                                                             dtype=nnef_tensor.dtype)
        assert reshaped_tensor.shape == shape
        return NNEFOperation(graph=nnef_graph,
                             name="reshape",
                             inputs=nnef_tensor,
                             attribs=dict(shape=shape_with_neg_one if use_neg_one else shape),
                             outputs=reshaped_tensor).output

    @staticmethod
    def caffe2_pads_to_nnef_padding(pads):
        assert len(pads) % 2 == 0
        return list(zip(pads[:len(pads) // 2], pads[len(pads) // 2:]))

    @staticmethod
    def is_tensor_used_or_io(caffe2_tensor):
        # type: (Caffe2Tensor)->bool
        return (bool(caffe2_tensor.consumers)
                or caffe2_tensor in caffe2_tensor.graph.inputs
                or caffe2_tensor in caffe2_tensor.graph.outputs)


def convert_default(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    print("Warning: Converter of {} is not implemented, doing default conversion.".format(caffe2_op.name))

    NNEFOperation(graph=nnef_graph,
                  name=caffe2_op.name,
                  inputs=converter.converted_tensors(caffe2_op.inputs),
                  attribs=utils.recursive_transform(caffe2_op.attribs, lambda x: x if x is not None else "None"),
                  outputs=converter.converted_tensors(caffe2_op.outputs))


def generic_convert_unary(converter, caffe2_op, nnef_graph, target_name):
    # type: (Converter, Caffe2Operation, NNEFGraph, str)->None

    NNEFOperation(graph=nnef_graph,
                  name=target_name,
                  inputs=converter.converted_tensor(caffe2_op.input),
                  outputs=converter.converted_tensor(caffe2_op.output))


def generic_convert_binary(converter, caffe2_op, nnef_graph, target_name):
    # type: (Converter, Caffe2Operation, NNEFGraph, str)->None

    a, b = converter.converted_tensors(caffe2_op.inputs)
    c = converter.converted_tensor(caffe2_op.output)

    axis = caffe2_op.attribs.get('axis', 0)
    assert axis >= 0, 'Binary op axis should be non-negative after unification'

    if axis != 0 and b.rank != 0:
        b = converter.add_unsqueeze(nnef_graph, b, list(range(axis)))

    NNEFOperation(graph=nnef_graph,
                  name=target_name,
                  inputs=(a, b),
                  outputs=c)


def generic_convert_pool(converter, caffe2_op, nnef_graph, target_name, global_name, border):
    # type: (Converter, Caffe2Operation, NNEFGraph, str, str, str)->None

    is_nhwc = caffe2_op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    if is_nhwc:
        raise utils.NNEFToolsException('NHWC order is not supported')

    input, output = converter.converted_tensors((caffe2_op.input, caffe2_op.output))

    if caffe2_op.attribs.get('global_pooling'):
        assert all(s == 1 for s in caffe2_op.attribs['strides'])
        assert all(p == 0 for p in caffe2_op.attribs['pads'])
        NNEFOperation(graph=nnef_graph,
                      name=global_name,
                      inputs=input,
                      outputs=output,
                      attribs=dict(axes=list(range(2, input.rank))))
    else:
        NNEFOperation(graph=nnef_graph,
                      name=target_name,
                      inputs=input,
                      outputs=output,
                      attribs=dict(size=[1, 1] + list(caffe2_op.attribs['kernels']),
                                   stride=[1, 1] + list(caffe2_op.attribs['strides']),
                                   padding=[(0, 0), (0, 0)] + converter.caffe2_pads_to_nnef_padding(
                                       caffe2_op.attribs['pads']),
                                   dilation=[1] * input.rank,
                                   border=border))


def generic_convert_lp_normalization(converter, caffe2_op, nnef_graph, target_name):
    # type: (Converter, Caffe2Operation, NNEFGraph, str)->None
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    axis = caffe2_op.attribs.get('axis', -1)
    if axis < 0:
        axis += input.rank

    NNEFOperation(graph=nnef_graph,
                  name=target_name,
                  inputs=input,
                  outputs=output,
                  attribs=dict(axes=[axis],
                               epsilon=1e-12,
                               bias=0.0))


def generic_convert_variadic(converter, caffe2_op, nnef_graph, target_name, normalize=False, variadic_target_name=None):
    # type: (Converter, Caffe2Operation, NNEFGraph, str, bool, typing.Optional[str])->None

    inputs = converter.converted_tensors(caffe2_op.inputs)
    output = converter.converted_tensor(caffe2_op.output)

    assert all(input.shape == output.shape and input.dtype == output.dtype for input in inputs)

    assert len(inputs) >= 1
    if len(inputs) == 1:
        NNEFOperation(graph=nnef_graph, name='copy', inputs=inputs[0], outputs=output)
    else:
        if len(inputs) > 2 and variadic_target_name:
            if not normalize:
                tmp = output
            else:
                tmp = NNEFTensor(graph=nnef_graph,
                                 name=None,
                                 shape=output.shape,
                                 dtype=output.dtype)
            NNEFOperation(graph=nnef_graph,
                          name=variadic_target_name,
                          inputs=inputs,
                          outputs=tmp)
        else:
            left = inputs[0]
            for i, right in enumerate(inputs[1:]):
                if 1 + i == len(inputs) - 1 and not normalize:
                    new_left = output
                else:
                    new_left = NNEFTensor(graph=nnef_graph,
                                          name=None,
                                          shape=left.shape,
                                          dtype=left.dtype)
                NNEFOperation(graph=nnef_graph,
                              name=target_name,
                              inputs=(left, right),
                              outputs=new_left)
                left = new_left
            tmp = left

        if normalize:
            n_tensor = NNEFTensor(graph=nnef_graph, shape=[], data=[float(len(inputs))], dtype=output.dtype)
            NNEFOperation(graph=nnef_graph,
                          name='div',
                          inputs=(tmp, n_tensor),
                          outputs=output)


def generic_convert_reduce(converter, caffe2_op, nnef_graph, target_name, multi_axis, before='', after=''):
    # type: (Converter, Caffe2Operation, NNEFGraph, str, bool, str, str)->None

    input, output = converter.converted_tensors((caffe2_op.input, caffe2_op.output))

    if before:
        input = NNEFOperation(graph=nnef_graph,
                              name=before,
                              inputs=input,
                              outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    if not multi_axis:
        axis = caffe2_op.attribs.get('axis', -1)
        axes = [axis + input.rank if axis < 0 else axis]
    else:
        axes = [axis + input.rank if axis < 0 else axis for axis in caffe2_op.attribs['axes']]

    keepdims = caffe2_op.attribs.get('keepdims', True)

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
                                                  shape=infer.reduce(input.shape, axes),
                                                  dtype=output.dtype))
        if after:
            reduce = NNEFOperation(graph=nnef_graph,
                                   name=after,
                                   inputs=reduce.output,
                                   outputs=NNEFTensor(graph=nnef_graph,
                                                      shape=list(reduce.output.shape),
                                                      dtype=reduce.output.dtype))

        NNEFOperation(graph=nnef_graph, name="squeeze", inputs=reduce.output, attribs=dict(axes=axes), outputs=output)


def convert_concat(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    assert len(caffe2_op.outputs) == 1 or not converter.is_tensor_used_or_io(caffe2_op.outputs[1])

    inputs = converter.converted_tensors(caffe2_op.inputs)
    output = converter.converted_tensor(caffe2_op.outputs[0])
    NNEFOperation(graph=nnef_graph,
                  name='concat',
                  inputs=inputs,
                  outputs=output,
                  attribs=dict(axis=caffe2_op.attribs['axis']))


def convert_conditional_or_where(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    input, true_value, false_value = converter.converted_tensors(caffe2_op.inputs)
    output = converter.converted_tensor(caffe2_op.output)
    NNEFOperation(graph=nnef_graph,
                  name='select',
                  inputs=(input, true_value, false_value),
                  outputs=output)


def convert_conv(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    is_nhwc = caffe2_op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    if is_nhwc:
        raise utils.NNEFToolsException('NHWC order is not supported')

    input, filter = converter.converted_tensors((caffe2_op.inputs[0], caffe2_op.inputs[1]))
    if len(caffe2_op.inputs) == 3:
        bias = converter.converted_tensor(caffe2_op.inputs[2])
        bias = converter.add_unsqueeze(nnef_graph, bias, [0])
    else:
        bias = NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[0.0])
    output = converter.converted_tensor(caffe2_op.output)

    NNEFOperation(graph=nnef_graph,
                  name='conv',
                  inputs=(input, filter, bias),
                  outputs=output,
                  attribs=dict(stride=list(caffe2_op.attribs['strides']),
                               padding=converter.caffe2_pads_to_nnef_padding(caffe2_op.attribs['pads']),
                               dilation=list(caffe2_op.attribs['dilations']),
                               border='constant',
                               groups=caffe2_op.attribs['group']))


def convert_conv_transpose(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    is_nhwc = caffe2_op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    if is_nhwc:
        raise utils.NNEFToolsException('NHWC order is not supported')

    input, filter = converter.converted_tensors((caffe2_op.inputs[0], caffe2_op.inputs[1]))
    if len(caffe2_op.inputs) == 3:
        bias = converter.converted_tensor(caffe2_op.inputs[2])
        bias = converter.add_unsqueeze(nnef_graph, bias, [0])
    else:
        bias = NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[0.0])
    output = converter.converted_tensor(caffe2_op.output)

    NNEFOperation(graph=nnef_graph,
                  name='deconv',
                  inputs=(input, filter, bias),
                  outputs=output,
                  attribs=dict(stride=list(caffe2_op.attribs['strides']),
                               padding=converter.caffe2_pads_to_nnef_padding(caffe2_op.attribs['pads']),
                               dilation=[1] * (input.rank - 2),
                               border='constant',
                               groups=caffe2_op.attribs['group'],
                               output_shape=output.shape))


def convert_split(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    input = converter.converted_tensor(caffe2_op.input)
    outputs = converter.converted_tensors(caffe2_op.outputs)
    NNEFOperation(graph=nnef_graph,
                  name='split',
                  inputs=input,
                  outputs=outputs,
                  attribs=dict(axis=caffe2_op.attribs['axis'], ratios=caffe2_op.attribs['split']))


def convert_dot_product(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    a, b = converter.converted_tensors(caffe2_op.inputs)
    c = converter.converted_tensor(caffe2_op.output)

    assert a.rank == b.rank
    if a.rank == 1:
        NNEFOperation(graph=nnef_graph,
                      name='mul',
                      inputs=(a, b),
                      outputs=c)
    else:
        multiplied = NNEFOperation(graph=nnef_graph,
                                   name='mul',
                                   inputs=(a, b),
                                   outputs=NNEFTensor(graph=nnef_graph,
                                                      shape=[max(a_dim, b_dim) for a_dim, b_dim in
                                                             zip(a.shape, b.shape)],
                                                      dtype=c.dtype)).output
        reduced = NNEFOperation(graph=nnef_graph,
                                name='sum_reduce',
                                inputs=multiplied,
                                outputs=NNEFTensor(graph=nnef_graph,
                                                   shape=[max(a.shape[0], b.shape[0])] + [1] * (a.rank - 1),
                                                   dtype=c.dtype),
                                attribs=dict(axes=list(range(1, a.rank)))).output
        NNEFOperation(graph=nnef_graph,
                      name='squeeze',
                      inputs=reduced,
                      outputs=c,
                      attribs=dict(axes=list(range(1, a.rank))))


def convert_fc(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    x, w, b = converter.converted_tensors(caffe2_op.inputs)
    b = converter.add_unsqueeze(nnef_graph, b, [0])
    y = converter.converted_tensor(caffe2_op.output)

    axis = caffe2_op.attribs.get('axis', 1)
    axis_w = caffe2_op.attribs.get('axis_w', 1)
    if x.rank == w.rank == 2 and axis == 1 and axis_w == 1:
        NNEFOperation(graph=nnef_graph,
                      name='linear',
                      inputs=(x, w, b),
                      outputs=y)
    else:
        x = converter.add_reshape_to_2d(nnef_graph, x, axis)
        w = converter.add_reshape_to_2d(nnef_graph, w, axis_w)

        tmp = NNEFOperation(graph=nnef_graph,
                            name='linear',
                            inputs=(x, w, b),
                            outputs=NNEFTensor(graph=nnef_graph, shape=[x.shape[0], w.shape[0]], dtype=y.dtype)).output

        NNEFOperation(graph=nnef_graph,
                      name='reshape',
                      inputs=tmp,
                      outputs=y,
                      attribs=dict(shape=y.shape))


def convert_fc_transposed(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    x, w, b = converter.converted_tensors(caffe2_op.inputs)
    b = converter.add_unsqueeze(nnef_graph, b, [0])
    y = converter.converted_tensor(caffe2_op.output)

    axis = caffe2_op.attribs.get('axis', 1)
    axis_w = caffe2_op.attribs.get('axis_w', 1)
    if x.rank == w.rank == 2 and axis == 1 and axis_w == 1:
        tmp = NNEFTensor(graph=nnef_graph, shape=list(y.shape), dtype=y.dtype)
        NNEFOperation(graph=nnef_graph,
                      name='matmul',
                      inputs=(x, w),
                      outputs=tmp)
        NNEFOperation(graph=nnef_graph,
                      name='add',
                      inputs=(tmp, b),
                      outputs=y)
    else:
        x = converter.add_reshape_to_2d(nnef_graph, x, axis)
        w = converter.add_reshape_to_2d(nnef_graph, w, axis_w)

        matmul = NNEFOperation(graph=nnef_graph,
                               name='matmul',
                               inputs=(x, w),
                               outputs=NNEFTensor(graph=nnef_graph,
                                                  shape=[x.shape[0], w.shape[1]],
                                                  dtype=y.dtype)).output
        add = NNEFOperation(graph=nnef_graph,
                            name='add',
                            inputs=(matmul, b),
                            outputs=NNEFTensor(graph=nnef_graph, shape=[x.shape[0], w.shape[1]], dtype=y.dtype)).output
        NNEFOperation(graph=nnef_graph,
                      name='reshape',
                      inputs=add,
                      outputs=y,
                      attribs=dict(shape=y.shape))


def convert_matmul(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    a, b = converter.converted_tensors(caffe2_op.inputs)
    c = converter.converted_tensor(caffe2_op.output)

    axis_a = caffe2_op.attribs.get('axis_a', 1)
    axis_b = caffe2_op.attribs.get('axis_b', 1)
    trans_a = bool(caffe2_op.attribs.get('trans_a', 0))
    trans_b = bool(caffe2_op.attribs.get('trans_b', 0))

    if a.rank == b.rank == 2 and axis_a == axis_b == 1:
        NNEFOperation(graph=nnef_graph,
                      name='matmul',
                      inputs=(a, b),
                      outputs=c,
                      attribs=dict(transposeA=trans_a,
                                   transposeB=trans_b))
    else:
        a = converter.add_reshape_to_2d(nnef_graph, a, axis_a)
        b = converter.add_reshape_to_2d(nnef_graph, b, axis_b)

        tmp = NNEFOperation(graph=nnef_graph,
                            name='matmul',
                            inputs=(a, b),
                            outputs=NNEFTensor(graph=nnef_graph,
                                               shape=[a.shape[1 if trans_a else 0], b.shape[0 if trans_b else 1]],
                                               dtype=b.dtype),
                            attribs=dict(transposeA=trans_a,
                                         transposeB=trans_b)).output

        NNEFOperation(graph=nnef_graph,
                      name='reshape',
                      inputs=tmp,
                      outputs=c,
                      attribs=dict(shape=c.shape))


def convert_batch_matmul(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    a, b = converter.converted_tensors(caffe2_op.inputs)
    c = converter.converted_tensor(caffe2_op.output)

    trans_a = bool(caffe2_op.attribs.get('trans_a', 0)) and a.rank > 1
    trans_b = bool(caffe2_op.attribs.get('trans_b', 0)) and b.rank > 1

    shape_a = list(a.shape)
    shape_b = list(b.shape)
    if len(shape_a) == 1:
        shape_a = [1, shape_a[0]]
    if len(shape_b) == 1:
        shape_b = [shape_b[0], 1]

    rank = max(len(shape_a), len(shape_b))
    shape_a = [1] * (rank - len(shape_a)) + shape_a
    shape_b = [1] * (rank - len(shape_b)) + shape_b

    if a.shape != shape_a:
        a = converter.add_reshape(nnef_graph, a, shape_a)
    if b.shape != shape_b:
        b = converter.add_reshape(nnef_graph, b, shape_b)

    transposed_shape_a = shape_a[:-2] + list(reversed(shape_a[-2:])) if trans_a else shape_a
    transposed_shape_b = shape_b[:-2] + list(reversed(shape_b[-2:])) if trans_b else shape_b
    calculated_output_shape = ([max(a, b) for a, b in zip(transposed_shape_a[:-2], transposed_shape_b[:-2])]
                               + [transposed_shape_a[-2], transposed_shape_b[-1]])

    tmp = c if c.shape == calculated_output_shape else NNEFTensor(graph=nnef_graph,
                                                                  shape=calculated_output_shape,
                                                                  dtype=b.dtype)

    NNEFOperation(graph=nnef_graph,
                  name='matmul',
                  inputs=(a, b),
                  outputs=tmp,
                  attribs=dict(transposeA=trans_a,
                               transposeB=trans_b))

    if c.shape != calculated_output_shape:
        converter.add_reshape(nnef_graph, tmp, c.shape, reshaped_tensor=c)


def convert_softmax(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    axis = caffe2_op.attribs.get('axis', 1)
    NNEFOperation(graph=nnef_graph,
                  name='softmax',
                  inputs=input,
                  outputs=output,
                  attribs=dict(axes=list(range(axis, input.rank))))


def convert_reshape(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    assert len(caffe2_op.outputs) == 1 or not converter.is_tensor_used_or_io(caffe2_op.outputs[1])
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.outputs[0])

    NNEFOperation(graph=nnef_graph,
                  name='reshape',
                  inputs=input,
                  outputs=output,
                  attribs=dict(shape=caffe2_op.attribs['shape']))


def convert_merge_dim(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    shape = list(output.shape)
    shape[0] = -1
    NNEFOperation(graph=nnef_graph,
                  name='reshape',
                  inputs=input,
                  outputs=output,
                  attribs=dict(shape=shape))


def convert_prepend_dim(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    shape = list(output.shape)
    shape[1] = -1
    NNEFOperation(graph=nnef_graph,
                  name='reshape',
                  inputs=input,
                  outputs=output,
                  attribs=dict(shape=shape))


def convert_resize_like(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    input, ref = converter.converted_tensors(caffe2_op.inputs)
    output = converter.converted_tensor(caffe2_op.output)

    NNEFOperation(graph=nnef_graph,
                  name='reshape',
                  inputs=input,
                  outputs=output,
                  attribs=dict(shape=list(ref.shape)))


def convert_flatten(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)
    converter.add_reshape_to_2d(nnef_graph, input,
                                axis=caffe2_op.attribs.get('axis', 1),
                                use_neg_one=True,
                                reshaped_tensor=output)


def convert_flatten_to_vec(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)
    converter.add_reshape(nnef_graph, input, shape=[-1], reshaped_tensor=output)


def convert_leaky_relu(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)
    NNEFOperation(graph=nnef_graph,
                  name='leaky_relu',
                  inputs=input,
                  outputs=output,
                  attribs=dict(alpha=caffe2_op.attribs.get('alpha', 0.01)))


def convert_lrn(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    if len(caffe2_op.outputs) == 2 and converter.is_tensor_used_or_io(caffe2_op.outputs[1]):
        raise utils.NNEFToolsException("Using LRN scale is unsupported.")

    is_nhwc = caffe2_op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.outputs[0])
    assert input.rank == 4
    size = caffe2_op.attribs['size']
    NNEFOperation(graph=nnef_graph,
                  name='local_response_normalization',
                  inputs=input,
                  outputs=output,
                  attribs=dict(size=[1, 1, 1, size] if is_nhwc else [1, size, 1, 1],
                               alpha=caffe2_op.attribs['alpha'],
                               beta=caffe2_op.attribs['beta'],
                               bias=caffe2_op.attribs.get('bias', 1.0)))


def convert_transpose(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    if 'axes' in caffe2_op.attribs:
        axes = list(caffe2_op.attribs['axes'])
    else:
        axes = list(reversed(range(input.rank)))

    NNEFOperation(graph=nnef_graph,
                  name='transpose',
                  inputs=input,
                  outputs=output,
                  attribs=dict(axes=axes))


def convert_spatial_bn(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    assert (len(caffe2_op.outputs) == 1
            or all(not converter.is_tensor_used_or_io(output) for output in caffe2_op.outputs[1:]))
    x, scale, bias, mean, var = converter.converted_tensors(caffe2_op.inputs[:5])
    y = converter.converted_tensor(caffe2_op.outputs[0])

    is_nhwc = caffe2_op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    unsqueeze_axes = list(range(x.rank - 1)) if is_nhwc else [0]
    scale, bias, mean, var = [converter.add_unsqueeze(nnef_graph, t, unsqueeze_axes) for t in (scale, bias, mean, var)]

    NNEFOperation(graph=nnef_graph,
                  name='batch_normalization',
                  inputs=(x, mean, var, bias, scale),
                  outputs=y,
                  attribs=dict(epsilon=caffe2_op.attribs.get('epsilon', 1e-5)))


def convert_max_pool_with_index(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    is_nhwc = caffe2_op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    if is_nhwc:
        raise utils.NNEFToolsException('NHWC order is not supported')

    input = converter.converted_tensor(caffe2_op.input)
    output, index = converter.converted_tensors(caffe2_op.outputs)

    kernels = (caffe2_op.input.shape[2:] if caffe2_op.attribs.get('global_pooling')
               else list(caffe2_op.attribs['kernels']))

    NNEFOperation(graph=nnef_graph,
                  name='max_pool_with_index',
                  inputs=input,
                  outputs=(output, index),
                  attribs=dict(size=[1, 1] + kernels,
                               stride=[1, 1] + list(caffe2_op.attribs['strides']),
                               padding=[(0, 0), (0, 0)] + converter.caffe2_pads_to_nnef_padding(
                                   caffe2_op.attribs['pads']),
                               dilation=[1] * input.rank,
                               border='ignore'))


def convert_dropout(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    assert len(caffe2_op.outputs) == 1 or not converter.is_tensor_used_or_io(caffe2_op.outputs[1])
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.outputs[0])

    NNEFOperation(graph=nnef_graph,
                  name='copy',
                  inputs=input,
                  outputs=output)


def convert_expand_dims(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    NNEFOperation(graph=nnef_graph,
                  name='unsqueeze',
                  inputs=input,
                  outputs=output,
                  attribs=dict(axes=caffe2_op.attribs.get('dims')))


def convert_squeeze(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    NNEFOperation(graph=nnef_graph,
                  name='squeeze',
                  inputs=input,
                  outputs=output,
                  attribs=dict(axes=caffe2_op.attribs.get('dims')))


def convert_cast(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

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
        raise utils.NNEFToolsException(
            "Unsupported cast: {} -> {}."
            "Supported casts: cast to same type, logical->scalar, logical->integer, scalar->logical."
                .format(input.dtype, output.dtype))


def convert_clip(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    has_min = 'min' in caffe2_op.attribs and caffe2_op.attribs['min'] != float('-inf')
    has_max = 'max' in caffe2_op.attribs and caffe2_op.attribs['max'] != float('inf')

    if has_min and has_max:
        min = NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[caffe2_op.attribs.get('min')])
        max = NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[caffe2_op.attribs.get('max')])
        NNEFOperation(graph=nnef_graph,
                      name='clamp',
                      inputs=(input, min, max),
                      outputs=output)
    elif has_min:
        min = NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[caffe2_op.attribs.get('min')])
        NNEFOperation(graph=nnef_graph,
                      name='max',
                      inputs=(input, min),
                      outputs=output)
    elif has_max:
        max = NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[caffe2_op.attribs.get('max')])
        NNEFOperation(graph=nnef_graph,
                      name='min',
                      inputs=(input, max),
                      outputs=output)
    else:
        NNEFOperation(graph=nnef_graph,
                      name='copy',
                      inputs=input,
                      outputs=output)


def convert_scale(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    NNEFOperation(graph=nnef_graph,
                  name='mul',
                  inputs=(input, NNEFTensor(graph=nnef_graph,
                                            shape=[],
                                            dtype=input.dtype,
                                            data=[float(caffe2_op.attribs.get('scale', 1.0))])),
                  outputs=output)


def convert_slice(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    NNEFOperation(graph=nnef_graph,
                  name='slice',
                  inputs=input,
                  outputs=output,
                  attribs=dict(axes=list(range(input.rank)),
                               begin=caffe2_op.attribs['starts'],
                               end=[e + 1 if e < 0 else e for e in caffe2_op.attribs['ends']]))


def convert_channel_shuffle(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    channel_axis = input.rank - 1 if caffe2_op.attribs.get('order', 'NCHW').upper() == 'NHWC' else 1
    channels = input.shape[channel_axis]
    groups = int(caffe2_op.attribs.get('group', 1))
    assert channels % groups == 0

    shape = list(input.shape)
    shape.insert(channel_axis, groups)
    shape[channel_axis + 1] = int(channels / groups)
    reshape = NNEFOperation(graph=nnef_graph,
                            name='reshape',
                            inputs=input,
                            outputs=NNEFTensor(graph=nnef_graph, shape=list(shape), dtype=input.dtype),
                            attribs=dict(shape=list(shape))).output
    axes = list(range(reshape.rank))
    axes[channel_axis], axes[channel_axis + 1] = axes[channel_axis + 1], axes[channel_axis]
    shape = infer.transpose(shape, axes)
    transpose = NNEFOperation(graph=nnef_graph,
                              name='transpose',
                              inputs=reshape,
                              outputs=NNEFTensor(graph=nnef_graph, shape=list(shape), dtype=input.dtype),
                              attribs=dict(axes=axes)).output
    NNEFOperation(graph=nnef_graph,
                  name='reshape',
                  inputs=transpose,
                  outputs=output,
                  attribs=dict(shape=list(input.shape)))


def convert_selu(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    alpha = caffe2_op.attribs.get('alpha', 1.6732632423543772848170429916717)
    scale = caffe2_op.attribs.get('scale', 1.0507009873554804934193349852946)

    exp = NNEFOperation(graph=nnef_graph,
                        name='exp',
                        inputs=input,
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    mul = NNEFOperation(graph=nnef_graph,
                        name='mul',
                        inputs=(exp, NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[float(alpha)])),
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    sub = NNEFOperation(graph=nnef_graph,
                        name='sub',
                        inputs=(mul, NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[float(alpha)])),
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    gt = NNEFOperation(graph=nnef_graph,
                       name='gt',
                       inputs=(input, NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[0.0])),
                       outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype='logical')).output

    select = NNEFOperation(graph=nnef_graph,
                           name='select',
                           inputs=(gt, input, sub),
                           outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    NNEFOperation(graph=nnef_graph,
                  name='mul',
                  inputs=(select, NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[float(scale)])),
                  outputs=output)


def convert_softsign(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    abs = NNEFOperation(graph=nnef_graph,
                        name='abs',
                        inputs=input,
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    add = NNEFOperation(graph=nnef_graph,
                        name='add',
                        inputs=(abs, NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[1.0])),
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    NNEFOperation(graph=nnef_graph,
                  name='div',
                  inputs=(input, add),
                  outputs=output)


def convert_stump_func(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    threshold = caffe2_op.attribs.get('threshold', 0.0)
    low_value = caffe2_op.attribs.get('low_value', 0.0)
    high_value = caffe2_op.attribs.get('high_value', 0.0)

    gt = NNEFOperation(graph=nnef_graph,
                       name='gt',
                       inputs=(input, NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[threshold])),
                       outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype='logical')).output

    NNEFOperation(graph=nnef_graph,
                  name='select',
                  inputs=(gt,
                          NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype,
                                     data=[high_value]),
                          NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype,
                                     data=[low_value])),
                  outputs=output)


def convert_elementwise_linear(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    x, w, b = converter.converted_tensors(caffe2_op.inputs)
    w = converter.add_unsqueeze(nnef_graph, w, [0])
    b = converter.add_unsqueeze(nnef_graph, b, [0])
    c = converter.converted_tensor(caffe2_op.output)

    axis = caffe2_op.attribs.get('axis', 1)

    if axis == 1 and x.rank == 2:
        mul = NNEFOperation(graph=nnef_graph,
                            name='mul',
                            inputs=(x, w),
                            outputs=NNEFTensor(graph=nnef_graph, shape=list(x.shape), dtype=x.dtype)).output
        NNEFOperation(graph=nnef_graph,
                      name='add',
                      inputs=(mul, b),
                      outputs=c)
    else:
        x = converter.add_reshape_to_2d(nnef_graph, x, axis)

        mul = NNEFOperation(graph=nnef_graph,
                            name='mul',
                            inputs=(x, w),
                            outputs=NNEFTensor(graph=nnef_graph, shape=list(x.shape), dtype=x.dtype)).output
        add = NNEFOperation(graph=nnef_graph,
                            name='add',
                            inputs=(mul, b),
                            outputs=NNEFTensor(graph=nnef_graph, shape=list(x.shape), dtype=x.dtype)).output

        NNEFOperation(graph=nnef_graph,
                      name='reshape',
                      inputs=add,
                      outputs=c,
                      attribs=dict(shape=c.shape))


def convert_instance_norm(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    is_nhwc = caffe2_op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    if is_nhwc:
        raise utils.NNEFToolsException('NHWC order is not supported')

    assert (len(caffe2_op.outputs) == 1
            or all(not converter.is_tensor_used_or_io(output) for output in caffe2_op.outputs[1:]))

    input, scale, bias = converter.converted_tensors(caffe2_op.inputs)
    output = converter.converted_tensor(caffe2_op.outputs[0])

    epsilon_tensor = NNEFTensor(graph=nnef_graph,
                                shape=[],
                                dtype=input.dtype,
                                data=[caffe2_op.attribs.get('epsilon', 1e-5)])

    channel_axis = 1

    scale = converter.add_unsqueeze(nnef_graph=nnef_graph, nnef_tensor=scale, axes=list(range(channel_axis)))
    bias = converter.add_unsqueeze(nnef_graph=nnef_graph, nnef_tensor=bias, axes=list(range(channel_axis)))

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
                        inputs=(sub, scale),
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


def convert_l1_distance(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    a, b = converter.converted_tensors(caffe2_op.inputs)
    c = converter.converted_tensor(caffe2_op.output)

    assert a.rank == b.rank

    sub_shape = [max(a_dim, b_dim) for a_dim, b_dim in zip(a.shape, b.shape)]

    if a.rank == 1:
        sub = NNEFOperation(graph=nnef_graph,
                            name='sub',
                            inputs=(a, b),
                            outputs=NNEFTensor(graph=nnef_graph,
                                               shape=list(sub_shape),
                                               dtype=c.dtype)).output
        NNEFOperation(graph=nnef_graph,
                      name='abs',
                      inputs=sub,
                      outputs=c)
    else:
        sub = NNEFOperation(graph=nnef_graph,
                            name='sub',
                            inputs=(a, b),
                            outputs=NNEFTensor(graph=nnef_graph,
                                               shape=list(sub_shape),
                                               dtype=c.dtype)).output
        abs = NNEFOperation(graph=nnef_graph,
                            name='abs',
                            inputs=sub,
                            outputs=NNEFTensor(graph=nnef_graph,
                                               shape=list(sub_shape),
                                               dtype=c.dtype)).output

        reduced = NNEFOperation(graph=nnef_graph,
                                name='sum_reduce',
                                inputs=abs,
                                outputs=NNEFTensor(graph=nnef_graph,
                                                   shape=[sub_shape[0]] + [1] * (a.rank - 1),
                                                   dtype=c.dtype),
                                attribs=dict(axes=list(range(1, a.rank)))).output
        NNEFOperation(graph=nnef_graph,
                      name='squeeze',
                      inputs=reduced,
                      outputs=c,
                      attribs=dict(axes=list(range(1, a.rank))))


def convert_layer_norm(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    if len(caffe2_op.inputs) == 3:
        raise utils.NNEFToolsException('Layer norm with 3 inputs in unsupported.')

    input = converter.converted_tensor(caffe2_op.inputs[0])
    output, mean, std = converter.converted_tensors(caffe2_op.outputs)

    epsilon_tensor = NNEFTensor(graph=nnef_graph,
                                shape=[],
                                dtype=input.dtype,
                                data=[caffe2_op.attribs.get('epsilon', 1e-5)])
    axis = caffe2_op.attribs.get('axis', 1)
    if axis < 0:
        axis += input.rank

    axes = list(range(axis, input.rank))
    mean_unsqueezed, std_unsqueezed = NNEFOperation(
        graph=nnef_graph,
        name='moments',
        inputs=input,
        attribs=dict(axes=axes),
        outputs=(NNEFTensor(graph=nnef_graph,
                            shape=infer.reduce(input=input.shape, axes=axes),
                            dtype=input.dtype),
                 NNEFTensor(graph=nnef_graph,
                            shape=infer.reduce(input=input.shape, axes=axes),
                            dtype=input.dtype))).outputs

    sub = NNEFOperation(graph=nnef_graph,
                        name='sub',
                        inputs=(input, mean_unsqueezed),
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    add = NNEFOperation(graph=nnef_graph,
                        name='add',
                        inputs=(std_unsqueezed, epsilon_tensor),
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(std_unsqueezed.shape),
                                           dtype=std_unsqueezed.dtype)).output

    sqrt = NNEFOperation(graph=nnef_graph,
                         name='sqrt',
                         inputs=add,
                         outputs=NNEFTensor(graph=nnef_graph, shape=list(add.shape), dtype=add.dtype)).output

    NNEFOperation(graph=nnef_graph,
                  name='div',
                  inputs=(sub, sqrt),
                  outputs=output)

    squeeze_axes = axes[1:]
    converter.add_squeeze(nnef_graph, mean_unsqueezed, squeeze_axes, mean)
    converter.add_squeeze(nnef_graph, sqrt, squeeze_axes, std)


def convert_logit(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe2_op.input, caffe2_op.output))

    eps = caffe2_op.attribs.get('eps', 1e-6)
    min = NNEFTensor(graph=nnef_graph, shape=[], dtype=output.dtype, data=[eps])
    max = NNEFTensor(graph=nnef_graph, shape=[], dtype=output.dtype, data=[1.0 - eps])
    one = NNEFTensor(graph=nnef_graph, shape=[], dtype=output.dtype, data=[1.0])

    clamp = NNEFOperation(graph=nnef_graph,
                          name='clamp',
                          inputs=(input, min, max),
                          outputs=NNEFTensor(graph=nnef_graph, shape=list(output.shape), dtype=output.dtype)).output
    sub = NNEFOperation(graph=nnef_graph,
                        name='sub',
                        inputs=(one, clamp),
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(output.shape), dtype=output.dtype)).output
    div = NNEFOperation(graph=nnef_graph,
                        name='div',
                        inputs=(clamp, sub),
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(output.shape), dtype=output.dtype)).output
    NNEFOperation(graph=nnef_graph,
                  name='log',
                  inputs=div,
                  outputs=output)


def convert_lp_norm(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe2_op.input, caffe2_op.output))
    p = caffe2_op.attribs.get('p', 2)
    average = caffe2_op.attribs.get('average', False)
    assert p in [1, 2]

    if input.rank != 1:
        input = NNEFOperation(graph=nnef_graph,
                              name='reshape',
                              inputs=input,
                              outputs=NNEFTensor(graph=nnef_graph, shape=[utils.product(input.shape)],
                                                 dtype=input.dtype),
                              attribs=dict(shape=[-1])).output

    abs = NNEFOperation(graph=nnef_graph,
                        name='abs' if p == 1 else 'sqr',
                        inputs=input,
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output
    NNEFOperation(graph=nnef_graph,
                  name='mean_reduce' if average else 'sum_reduce',
                  inputs=abs,
                  outputs=output,
                  attribs=dict(axes=[0]))


def convert_lp_pool(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    is_nhwc = caffe2_op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    if is_nhwc:
        raise utils.NNEFToolsException('NHWC order is not supported')

    input, output = converter.converted_tensors((caffe2_op.input, caffe2_op.output))

    p = caffe2_op.attribs.get('p', 2.0)
    assert p != 0

    input = NNEFOperation(graph=nnef_graph,
                          name='abs',
                          inputs=input,
                          outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    input = NNEFOperation(graph=nnef_graph,
                          name='pow',
                          inputs=(input, NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[p])),
                          outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output

    if caffe2_op.attribs.get('global_pooling'):
        assert all(s == 1 for s in caffe2_op.attribs['strides'])
        assert all(p == 0 for p in caffe2_op.attribs['pads'])
        input = NNEFOperation(graph=nnef_graph,
                              name='sum_reduce',
                              inputs=input,
                              outputs=NNEFTensor(graph=nnef_graph, shape=list(output.shape), dtype=output.dtype),
                              attribs=dict(axes=list(range(2, input.rank)))).output
    else:
        input = NNEFOperation(graph=nnef_graph,
                              name='box',
                              inputs=input,
                              outputs=NNEFTensor(graph=nnef_graph, shape=list(output.shape), dtype=output.dtype),
                              attribs=dict(size=[1, 1] + list(caffe2_op.attribs['kernels']),
                                           stride=[1, 1] + list(caffe2_op.attribs['strides']),
                                           padding=[(0, 0), (0, 0)] + converter.caffe2_pads_to_nnef_padding(
                                               caffe2_op.attribs['pads']),
                                           dilation=[1] * input.rank,
                                           border='ignore')).output

    NNEFOperation(graph=nnef_graph,
                  name='pow',
                  inputs=(input, NNEFTensor(graph=nnef_graph, shape=[], dtype=input.dtype, data=[1.0 / p])),
                  outputs=output)


def convert_pow(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    if len(caffe2_op.inputs) == 2:
        generic_convert_binary(converter, caffe2_op, nnef_graph, target_name='pow')
    elif len(caffe2_op.inputs) == 1:
        input, output = converter.converted_tensors((caffe2_op.input, caffe2_op.output))
        NNEFOperation(graph=nnef_graph,
                      name='pow',
                      inputs=(input, NNEFTensor(graph=nnef_graph,
                                                shape=[],
                                                dtype=input.dtype,
                                                data=[caffe2_op.attribs.get('exponent', 0.0)])),
                      outputs=output)
    else:
        assert False


def convert_prelu(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    x, slope = converter.converted_tensors(caffe2_op.inputs)
    output = converter.converted_tensor(caffe2_op.output)
    is_nhwc = caffe2_op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    channel_axis = x.rank - 1 if is_nhwc else 1
    slope = converter.add_unsqueeze(nnef_graph, slope, list(range(channel_axis)))
    NNEFOperation(graph=nnef_graph,
                  name='prelu',
                  inputs=(x, slope),
                  outputs=output)


def convert_range(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    output = converter.converted_tensor(caffe2_op.output)

    output.data = np.arange(caffe2_op.inputs[0].data.item(),
                            caffe2_op.inputs[1].data.item() if len(caffe2_op.inputs) >= 2 else None,
                            caffe2_op.inputs[2].data.item() if len(caffe2_op.inputs) >= 3 else None,
                            dtype=output.get_numpy_dtype()).tolist()


def convert_resize_nearest(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    input = converter.converted_tensor(caffe2_op.input)
    output = converter.converted_tensor(caffe2_op.output)

    is_nhwc = caffe2_op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    if is_nhwc:
        raise utils.NNEFToolsException('NHWC order is not supported')

    if output.shape[2] == input.shape[2] and output.shape[3] == input.shape[3]:
        NNEFOperation(graph=nnef_graph,
                      name='copy',
                      inputs=input,
                      outputs=output)
    elif output.shape[2] <= input.shape[2] and output.shape[3] <= input.shape[3]:
        if input.shape[2] % output.shape[2] != 0 or input.shape[3] % output.shape[3] != 0:
            raise utils.NNEFToolsException('resize_nearest is only supported for integer resize factor')
        NNEFOperation(graph=nnef_graph,
                      name='nearest_downsample',
                      inputs=input,
                      outputs=output,
                      attribs=dict(factor=[input.shape[2] // output.shape[2], input.shape[3] // output.shape[3]]))
    elif output.shape[2] >= input.shape[2] and output.shape[3] >= input.shape[3]:
        if output.shape[2] % input.shape[2] != 0 or output.shape[3] % input.shape[3] != 0:
            raise utils.NNEFToolsException('resize_nearest is only supported for integer resize factor')
        NNEFOperation(graph=nnef_graph,
                      name='nearest_upsample',
                      inputs=input,
                      outputs=output,
                      attribs=dict(factor=[output.shape[2] // input.shape[2], output.shape[3] // input.shape[3]]))
    elif output.shape[2] > input.shape[2] and output.shape[3] < input.shape[3]:
        if output.shape[2] % input.shape[2] != 0 or input.shape[3] % output.shape[3] != 0:
            raise utils.NNEFToolsException('resize_nearest is only supported for integer resize factor')
        tmp = NNEFOperation(graph=nnef_graph,
                            name='nearest_downsample',
                            inputs=input,
                            outputs=NNEFTensor(graph=nnef_graph,
                                               shape=output.shape[:2] + [input.shape[2], output.shape[3]],
                                               dtype=output.dtype),
                            attribs=dict(factor=[1, input.shape[3] // output.shape[3]])).output
        NNEFOperation(graph=nnef_graph,
                      name='nearest_upsample',
                      inputs=tmp,
                      outputs=output,
                      attribs=dict(factor=[output.shape[2] // input.shape[2], 1]))
    elif output.shape[2] < input.shape[2] and output.shape[3] > input.shape[3]:
        if input.shape[2] % output.shape[2] != 0 or output.shape[3] % input.shape[3] != 0:
            raise utils.NNEFToolsException('resize_nearest is only supported for integer resize factor')
        tmp = NNEFOperation(graph=nnef_graph,
                            name='nearest_downsample',
                            inputs=input,
                            outputs=NNEFTensor(graph=nnef_graph,
                                               shape=output.shape[:3] + [input.shape[3]],
                                               dtype=output.dtype),
                            attribs=dict(factor=[input.shape[2] // output.shape[2], 1])).output
        NNEFOperation(graph=nnef_graph,
                      name='nearest_upsample',
                      inputs=tmp,
                      outputs=output,
                      attribs=dict(factor=[1, output.shape[3] // input.shape[3]]))
    else:
        assert False


def convert_row_mul(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    mat, w = converter.converted_tensors(caffe2_op.inputs)
    z = converter.converted_tensor(caffe2_op.output)

    if w.rank != 1:
        w = NNEFOperation(graph=nnef_graph,
                          name='reshape',
                          inputs=w,
                          outputs=NNEFTensor(graph=nnef_graph, shape=[utils.product(mat.shape)], dtype=mat.dtype),
                          attribs=dict(shape=[-1])).output

    NNEFOperation(graph=nnef_graph,
                  name='mul',
                  inputs=(mat, w),
                  outputs=z)


def convert_squared_l2_distance(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None
    a, b = converter.converted_tensors(caffe2_op.inputs)
    c = converter.converted_tensor(caffe2_op.output)

    assert a.rank == b.rank

    sub_shape = [max(a_dim, b_dim) for a_dim, b_dim in zip(a.shape, b.shape)]

    sub = NNEFOperation(graph=nnef_graph,
                        name='sub',
                        inputs=(a, b),
                        outputs=NNEFTensor(graph=nnef_graph,
                                           shape=list(sub_shape),
                                           dtype=c.dtype)).output
    sqr = NNEFOperation(graph=nnef_graph,
                        name='sqr',
                        inputs=sub,
                        outputs=NNEFTensor(graph=nnef_graph,
                                           shape=list(sub_shape),
                                           dtype=c.dtype)).output

    div = c if a.rank == 1 else NNEFTensor(graph=nnef_graph,
                                           shape=list(sub_shape),
                                           dtype=c.dtype)
    NNEFOperation(graph=nnef_graph,
                  name='div',
                  inputs=(sqr, NNEFTensor(graph=nnef_graph, shape=[], dtype=c.dtype, data=[2.0])),
                  outputs=div)

    if a.rank != 1:
        reduced = NNEFOperation(graph=nnef_graph,
                                name='sum_reduce',
                                inputs=div,
                                outputs=NNEFTensor(graph=nnef_graph,
                                                   shape=[sub_shape[0]] + [1] * (a.rank - 1),
                                                   dtype=c.dtype),
                                attribs=dict(axes=list(range(1, a.rank)))).output
        NNEFOperation(graph=nnef_graph,
                      name='squeeze',
                      inputs=reduced,
                      outputs=c,
                      attribs=dict(axes=list(range(1, a.rank))))


def convert_sum_sqr_elements(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe2_op.input, caffe2_op.output))

    axes = list(range(input.rank))
    sqr = NNEFOperation(graph=nnef_graph,
                        name='sqr',
                        inputs=input,
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(input.shape), dtype=input.dtype)).output
    sum = NNEFOperation(graph=nnef_graph,
                        name='mean_reduce' if caffe2_op.attribs.get('average', False) else 'sum_reduce',
                        inputs=sqr,
                        outputs=NNEFTensor(graph=nnef_graph, shape=[1] * input.rank, dtype=input.dtype),
                        attribs=dict(axes=list(axes))).output
    NNEFOperation(graph=nnef_graph,
                  name='squeeze',
                  inputs=sum,
                  outputs=output,
                  attribs=dict(axes=list(axes)))


def convert_sum_reduce_like(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    input, reference = converter.converted_tensors(caffe2_op.inputs)
    output = converter.converted_tensor(caffe2_op.output)

    axis = caffe2_op.attribs['axis']
    target = [None if i < axis or i >= axis + reference.rank else min(input.shape[i], reference.shape[i - axis])
              for i in range(input.rank)]
    reduced_shape = [1 if dim is None else dim for dim in target]
    reduce_axes = [i for i in range(input.rank) if (target[i] in [None, 1]) and input.shape[i] != 1]
    squeeze_axes = [i for i in range(input.rank) if target[i] is None]

    if not reduce_axes and not squeeze_axes:
        NNEFOperation(graph=nnef_graph,
                      name='copy',
                      inputs=input,
                      outputs=output)
    else:
        if reduce_axes:
            input = NNEFOperation(graph=nnef_graph,
                                  name="sum_reduce",
                                  inputs=input,
                                  attribs=dict(axes=reduce_axes),
                                  outputs=NNEFTensor(graph=nnef_graph,
                                                     name=None,
                                                     shape=reduced_shape,
                                                     dtype=output.dtype) if squeeze_axes else output).output
        if squeeze_axes:
            NNEFOperation(graph=nnef_graph,
                          name="squeeze",
                          inputs=input,
                          attribs=dict(axes=squeeze_axes),
                          outputs=output)


def convert_summarize(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    input, output = converter.converted_tensors((caffe2_op.input, caffe2_op.output))

    axes = list(range(input.rank))

    min = NNEFOperation(graph=nnef_graph,
                        name='min_reduce',
                        inputs=input,
                        outputs=NNEFTensor(graph=nnef_graph, shape=[1] * input.rank, dtype=input.dtype),
                        attribs=dict(axes=list(axes))).output

    max = NNEFOperation(graph=nnef_graph,
                        name='max_reduce',
                        inputs=input,
                        outputs=NNEFTensor(graph=nnef_graph, shape=[1] * input.rank, dtype=input.dtype),
                        attribs=dict(axes=list(axes))).output

    mean, std = NNEFOperation(graph=nnef_graph,
                              name='moments',
                              inputs=input,
                              outputs=(NNEFTensor(graph=nnef_graph, shape=[1] * input.rank, dtype=input.dtype),
                                       NNEFTensor(graph=nnef_graph, shape=[1] * input.rank, dtype=input.dtype)),
                              attribs=dict(axes=list(axes))).outputs

    min = converter.add_reshape(nnef_graph, min, shape=[1])
    max = converter.add_reshape(nnef_graph, max, shape=[1])
    mean = converter.add_reshape(nnef_graph, mean, shape=[1])
    std = converter.add_reshape(nnef_graph, std, shape=[1])

    std = NNEFOperation(graph=nnef_graph,
                        name='mul',
                        inputs=(std, NNEFTensor(graph=nnef_graph,
                                                shape=[],
                                                dtype=input.dtype,
                                                data=[input.count / float(input.count - 1)])),
                        outputs=NNEFTensor(graph=nnef_graph, shape=[1] * input.rank, dtype=input.dtype)).output

    std = NNEFOperation(graph=nnef_graph,
                        name='sqrt',
                        inputs=std,
                        outputs=NNEFTensor(graph=nnef_graph, shape=[1] * input.rank, dtype=input.dtype)).output

    NNEFOperation(graph=nnef_graph,
                  name='concat',
                  inputs=[min, max, mean, std],
                  outputs=output,
                  attribs=dict(axis=0))


def convert_swish(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    x, y = converter.converted_tensors((caffe2_op.input, caffe2_op.output))

    neg = NNEFOperation(graph=nnef_graph,
                        name='neg',
                        inputs=x,
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(x.shape), dtype=x.dtype)).output

    exp = NNEFOperation(graph=nnef_graph,
                        name='exp',
                        inputs=neg,
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(neg.shape), dtype=neg.dtype)).output

    add = NNEFOperation(graph=nnef_graph,
                        name='add',
                        inputs=(exp, NNEFTensor(graph=nnef_graph, shape=[], dtype=neg.dtype, data=[1.0])),
                        outputs=NNEFTensor(graph=nnef_graph, shape=list(exp.shape), dtype=exp.dtype)).output

    NNEFOperation(graph=nnef_graph,
                  name='div',
                  inputs=(x, add),
                  outputs=y)


def convert_thresholded_relu(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    x, y = converter.converted_tensors((caffe2_op.input, caffe2_op.output))

    alpha = caffe2_op.attribs.get('alpha', 1.0)

    gt = NNEFOperation(graph=nnef_graph,
                       name='gt',
                       inputs=(x, NNEFTensor(graph=nnef_graph, shape=[], dtype=x.dtype, data=[alpha])),
                       outputs=NNEFTensor(graph=nnef_graph, shape=list(x.shape), dtype='logical')).output

    NNEFOperation(graph=nnef_graph,
                  name='select',
                  inputs=(gt,
                          x,
                          NNEFTensor(graph=nnef_graph, shape=list(x.shape), dtype=x.dtype,
                                     data=[0.0])),
                  outputs=y)


def convert_xor(converter, caffe2_op, nnef_graph):
    # type: (Converter, Caffe2Operation, NNEFGraph)->None

    a, b = converter.converted_tensors(caffe2_op.inputs)
    c = converter.converted_tensor(caffe2_op.output)

    axis = caffe2_op.attribs.get('axis', 0)
    assert axis >= 0, 'Binary op axis should be non-negative after unification'

    if axis != 0 and b.rank != 0:
        b = converter.add_unsqueeze(nnef_graph, b, list(range(axis)))

    def tensor_like(t):
        assert isinstance(t, NNEFTensor)
        return NNEFTensor(graph=nnef_graph, shape=list(t.shape), dtype=t.dtype)

    not_a = NNEFOperation(graph=nnef_graph,
                          name='not',
                          inputs=a,
                          outputs=tensor_like(a)).output
    not_b = NNEFOperation(graph=nnef_graph,
                          name='not',
                          inputs=b,
                          outputs=tensor_like(b)).output
    a_and_not_b = NNEFOperation(graph=nnef_graph,
                                name='and',
                                inputs=(a, not_b),
                                outputs=tensor_like(a)).output
    b_and_not_a = NNEFOperation(graph=nnef_graph,
                                name='and',
                                inputs=(b, not_a),
                                outputs=tensor_like(a)).output
    NNEFOperation(graph=nnef_graph,
                  name='or',
                  inputs=(a_and_not_b, b_and_not_a),
                  outputs=c)


_StandardConverters = {
    'Abs': partial(generic_convert_unary, target_name='abs'),
    'Add': partial(generic_convert_binary, target_name='add'),
    'Alias': None,
    'And': partial(generic_convert_binary, target_name='and'),
    'ArgMax': partial(generic_convert_reduce, multi_axis=False, target_name='argmax_reduce'),
    'ArgMin': partial(generic_convert_reduce, multi_axis=False, target_name='argmin_reduce'),
    'AveragePool': partial(generic_convert_pool, target_name='avg_pool', global_name='mean_reduce', border='ignore'),
    'BBoxTransform': None,
    'BRGNCHWCToPackedInt8BGRAStylizerDeprocess': None,
    'BatchGather': None,
    'BatchMatMul': convert_batch_matmul,
    'BatchToSpace': None,
    'BooleanMask': None,
    'BooleanUnmask': None,
    'BoxWithNMSLimit': None,
    'Cast': convert_cast,
    'Ceil': partial(generic_convert_unary, target_name='ceil'),
    'ChannelShuffle': convert_channel_shuffle,
    'ChannelStats': None,
    'Clip': convert_clip,
    'ClipTensorByScaling': None,
    'Col2Im': None,
    'Concat': convert_concat,
    'Conditional': convert_conditional_or_where,
    'Conv': convert_conv,
    'ConvTranspose': convert_conv_transpose,
    'Copy': partial(generic_convert_unary, target_name='copy'),
    'Cos': None,
    'Div': partial(generic_convert_binary, target_name='div'),
    'DotProduct': convert_dot_product,
    'DotProductWithPadding': None,
    'Dropout': convert_dropout,
    'EQ': partial(generic_convert_binary, target_name='eq'),
    'ElementwiseLinear': convert_elementwise_linear,
    'Elu': partial(generic_convert_unary, target_name='elu'),
    'Exp': partial(generic_convert_unary, target_name='exp'),
    'ExpandDims': convert_expand_dims,
    'FC': convert_fc,
    'FCTransposed': convert_fc_transposed,
    'Flatten': convert_flatten,
    'FlattenToVec': convert_flatten_to_vec,
    'FlexibleTopK': None,
    'Floor': partial(generic_convert_unary, target_name='floor'),
    'GE': partial(generic_convert_binary, target_name='ge'),
    'GRUUnit': None,
    'GT': partial(generic_convert_binary, target_name='gt'),
    'Gather': None,
    'GenerateProposals': None,
    'Glu': None,
    'Im2Col': None,
    'InstanceNorm': convert_instance_norm,
    'L1Distance': convert_l1_distance,
    'LE': partial(generic_convert_binary, target_name='le'),
    'LRN': convert_lrn,
    'LT': partial(generic_convert_binary, target_name='lt'),
    'LayerNorm': convert_layer_norm,
    'LeakyRelu': convert_leaky_relu,
    'Log': partial(generic_convert_unary, target_name='log'),
    'Logit': convert_logit,
    'LpNorm': convert_lp_norm,
    'LpPool': convert_lp_pool,
    'MatMul': convert_matmul,
    'Max': partial(generic_convert_variadic, target_name='max', variadic_target_name=None, normalize=False),
    'MaxPool': partial(generic_convert_pool, target_name='max_pool', global_name='max_reduce', border='ignore'),
    'MaxPoolWithIndex': convert_max_pool_with_index,
    'Mean': partial(generic_convert_variadic, target_name='add', variadic_target_name='add_n', normalize=True),
    'MergeDim': convert_merge_dim,
    'Min': partial(generic_convert_variadic, target_name='min', variadic_target_name=None, normalize=False),
    'Mod': None,
    'Mul': partial(generic_convert_binary, target_name='mul'),
    'NanCheck': None,
    'NE': partial(generic_convert_binary, target_name='ne'),
    'Negative': partial(generic_convert_unary, target_name='neg'),
    'Normalize': partial(generic_convert_lp_normalization, target_name='l2_normalization'),
    'NormalizeL1': partial(generic_convert_lp_normalization, target_name='l1_normalization'),
    'NormalizePlanarYUV': None,
    'Not': partial(generic_convert_unary, target_name='not'),
    'Or': partial(generic_convert_binary, target_name='or'),
    'PRelu': convert_prelu,
    'PackedInt8BGRANHWCToNCHWCStylizerPreprocess': None,
    'PadImage': None,
    'Perplexity': None,
    'PiecewiseLinearTransform': None,
    'Pow': convert_pow,
    'PrependDim': convert_prepend_dim,
    'QuantDecode': None,
    'Range': convert_range,
    'ReduceMin': partial(generic_convert_reduce, multi_axis=True, target_name='min_reduce'),
    'ReduceMax': partial(generic_convert_reduce, multi_axis=True, target_name='max_reduce'),
    'ReduceSum': partial(generic_convert_reduce, multi_axis=True, target_name='sum_reduce'),
    'ReduceMean': partial(generic_convert_reduce, multi_axis=True, target_name='mean_reduce'),
    'ReduceL1': partial(generic_convert_reduce, multi_axis=True, target_name='sum_reduce', before='abs'),
    'ReduceL2': partial(generic_convert_reduce, multi_axis=True, target_name='sum_reduce', before='sqr', after='sqrt'),
    'Relu': partial(generic_convert_unary, target_name='relu'),
    'ReplaceNaN': None,
    'Reshape': convert_reshape,
    'ResizeLike': convert_resize_like,
    'ResizeNearest': convert_resize_nearest,
    'RoIAlign': None,
    'RoIPool': None,
    'RowMul': convert_row_mul,
    'Scale': convert_scale,
    'Selu': convert_selu,
    'Sigmoid': partial(generic_convert_unary, target_name='sigmoid'),
    'Sign': partial(generic_convert_unary, target_name='sign'),
    'Sin': None,
    'Slice': convert_slice,
    'Softmax': convert_softmax,
    'Softplus': partial(generic_convert_unary, target_name='softplus'),
    'Softsign': convert_softsign,
    'SpaceToBatch': None,
    'SpatialBN': convert_spatial_bn,
    'Split': convert_split,
    'Sqr': partial(generic_convert_unary, target_name='sqr'),
    'Sqrt': partial(generic_convert_unary, target_name='sqrt'),
    'SquaredL2Distance': convert_squared_l2_distance,
    'Squeeze': convert_squeeze,
    'StopGradient': None,
    'StumpFunc': convert_stump_func,
    'Sub': partial(generic_convert_binary, target_name='sub'),
    'Sum': partial(generic_convert_variadic, target_name='add', variadic_target_name='add_n', normalize=False),
    'SumInt': None,
    'SumSqrElements': convert_sum_sqr_elements,
    'SumReduceLike': convert_sum_reduce_like,
    'Summarize': convert_summarize,
    'Swish': convert_swish,
    'TT': None,
    'Tanh': partial(generic_convert_unary, target_name='tanh'),
    'ThresholdedRelu': convert_thresholded_relu,
    'Tile': None,
    'TopK': None,
    'Transpose': convert_transpose,
    'Unique': None,
    'WallClockTime': None,
    'WeightedSum': None,
    'Where': convert_conditional_or_where,
    'Xor': convert_xor,
}
