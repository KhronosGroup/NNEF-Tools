# Copyright (c) 2020 The Khronos Group Inc.
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

from ..model.utils import replace_chain
from ..model.graph import *
from ..utils.types import from_numpy


class Optimizer:

    def __init__(self, keep_io_names=False):
        self._keep_io_names = keep_io_names

    def __call__(self, graph, only_required=False):
        self._fix_inputs_without_producer(graph)
        replace_chain(graph, ['SpaceToBatchND', 'Conv2D', 'BatchToSpaceND'], self._replace_dilated_conv)
        if not only_required:
            self._remove_unused_constants(graph)
        return graph

    @staticmethod
    def _fix_inputs_without_producer(graph):
        idx = 0
        for tensor in graph.inputs:
            if tensor.producer is None:
                cnt = len(graph.operations)
                Operation(tensor.graph, type='Placeholder', name=Optimizer._op_name(tensor.name), outputs=tensor,
                          attribs={'shape': tensor.shape, 'dtype': tensor.dtype})
                graph.move_operation(cnt, idx)
                idx += 1
        return idx > 0

    @staticmethod
    def _op_name(tensor_name):
        idx = tensor_name.find(':')
        return tensor_name[:idx] if idx != -1 else tensor_name

    @staticmethod
    def _remove_unused_constants(graph):
        ops = [op for op in graph.operations if op.type == 'Const' and len(op.output.consumers) == 0]
        tensors = [op.output for op in ops]
        graph.remove_operations(ops, unlink=True)
        graph.remove_tensors(tensors)

    @staticmethod
    def _replace_dilated_conv(space_to_batch, conv, batch_to_space):
        if not Optimizer._is_constant(space_to_batch.inputs[1]) or not Optimizer._is_constant(batch_to_space.inputs[1]):
            return False

        block_shape1 = Optimizer._read_constant(space_to_batch.inputs[1])
        block_shape2 = Optimizer._read_constant(batch_to_space.inputs[1])

        if not np.all(block_shape1 == block_shape2):
            return False

        if conv.attribs['padding'] != 'VALID':
            return False

        dilations = from_numpy(block_shape1)

        input = space_to_batch.inputs[0]
        filter = conv.inputs[1]
        output = batch_to_space.outputs[0]

        is_nxc = Optimizer._is_nxc(conv.attribs['data_format'])
        same_padding = Optimizer._is_same_padded(input.shape, output.shape, conv.attribs['strides'], is_nxc)

        if not same_padding:
            return False

        op = conv.copy_with(inputs=(input, filter), outputs=output, attribs=dict(conv.attribs))

        op.attribs['dilations'] = [1] + dilations + [1] if is_nxc else [1, 1] + dilations
        op.attribs['padding'] = 'SAME'

    @staticmethod
    def _is_constant(tensor):
        return tensor.producer.type == 'Const' if tensor.producer else tensor.data is not None

    @staticmethod
    def _read_constant(tensor):
        return tensor.producer.attribs['value'] if tensor.producer else tensor.data

    @staticmethod
    def _is_nxc(format):
        return format[0] == 'N' and format[-1] == 'C' and len(format) > 2

    @staticmethod
    def _is_same_padded(input, output, stride, is_nxc):
        rank = len(input)
        return all(output[i] == (input[i] + stride[i] - 1) // stride[i]
                   for i in (range(1, rank - 1) if is_nxc else range(2, rank)))
