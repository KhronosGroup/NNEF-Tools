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

import typing
from collections import deque, OrderedDict

from nnef_tools.core.graph import *

_OpOrOps = typing.Union[Operation, typing.Tuple[Operation, ...], typing.List[Operation]]
_TensorOrTensors = typing.Union[Tensor, typing.Tuple[Tensor, ...], typing.List[Tensor]]


def remove_subgraph(graph, operations):
    # type: (Graph, typing.List[Operation])->None
    """
    Deletes the given ops and the tensors that are enclosed in their sub-graph
    """

    tensors = set()
    for operation in operations:
        for tensor in operation.inputs:
            if (tensor not in tensors
                    and tensor.producers
                    and all(producer in operations for producer in tensor.producers)):
                tensors.add(tensor)

    for tensor in tensors:
        for operation in tensor.consumers:
            assert operation in operations

    graph.remove_operations(operations, unlink=True)
    graph.remove_tensors(tensors)


def remove_unreachable(graph):
    # type: (Graph)->None

    visited_ops = set()  # type: typing.Set[Operation]

    q = deque()  # type: typing.Deque[Operation]

    output_ops = set([producer
                      for t in graph.outputs
                      for producer in t.producers])

    for op in output_ops:
        visited_ops.add(op)
        q.append(op)

    while q:
        op = q.popleft()

        for tensor in op.inputs:
            for producer in tensor.producers:
                if producer not in visited_ops:
                    visited_ops.add(producer)
                    q.append(producer)

    graph.remove_operations([op for op in graph.operations if op not in visited_ops], unlink=True)

    def can_remove(g, t):
        return len(t.producers) == 0 and len(t.consumers) == 0 and t not in g.outputs and t not in g.inputs

    graph.remove_tensors([t for t in graph.tensors if can_remove(graph, t)])


def replace_tensor_in_inputs(graph, old_tensor, new_tensor, remove=False):
    if graph.input_ids is not None:
        graph.inputs = OrderedDict((name, new_tensor if t is old_tensor else t)
                                   for name, t in zip(graph.input_ids, graph.inputs))
    else:
        graph.inputs = [new_tensor if t is old_tensor else t for t in graph.inputs]

    if remove:
        graph.remove_tensor(old_tensor)


def replace_tensor_in_outputs(graph, old_tensor, new_tensor, remove=False):
    if graph.output_ids is not None:
        graph.outputs = OrderedDict((name, new_tensor if t is old_tensor else t)
                                    for name, t in zip(graph.output_ids, graph.outputs))
    else:
        graph.outputs = [new_tensor if t is old_tensor else t for t in graph.outputs]

    if remove:
        graph.remove_tensor(old_tensor)


def replace_tensor_in_consumers(graph, old_tensor, new_tensor, remove=False):
    # type: (Graph, Tensor, Tensor, bool)->None

    for consumer in list(old_tensor.consumers):
        if isinstance(consumer.inputs, tuple):
            consumer.inputs = tuple(new_tensor if t is old_tensor else t for t in consumer.inputs)
        else:
            consumer.inputs = [new_tensor if t is old_tensor else t for t in consumer.inputs]

    replace_tensor_in_outputs(graph, old_tensor, new_tensor)

    if remove:
        graph.remove_tensor(old_tensor)


def remove_passthrough(g, op):
    # type: (Graph, Operation)->None
    assert len(op.outputs) == 1 and len(op.inputs) == 1
    op_input = op.input
    op_output = op.output

    g.remove_operation(op, unlink=True)
    replace_tensor_in_consumers(g, op_output, op_input, remove=True)


def remove_passthroughs(g, is_passthrough):
    # type: (Graph, typing.Callable[[Operation], bool])->None
    for op in list(g.operations):
        if is_passthrough(op):
            remove_passthrough(g, op)


def _with_tensor_replaced(tuple_or_listlike, old, new):
    is_tuple = isinstance(tuple_or_listlike, tuple)
    list_ = [new if t is old else t for t in tuple_or_listlike]
    return tuple(list_) if is_tuple else list_


def resolve_tensor_overwrite(g, duplicate):
    # type: (Graph, typing.Callable[[Tensor], Tensor])->None
    for op in g.operations:
        assert len(op.outputs) == len(set(id(output) for output in op.outputs))
    outputs_of_earlier_ops = set()
    for i, op in enumerate(g.operations):
        for tensor in list(op.outputs):
            if tensor in op.inputs or tensor in outputs_of_earlier_ops:
                new_tensor = duplicate(tensor)
                op.outputs = _with_tensor_replaced(op.outputs, tensor, new_tensor)
                for op2 in g.operations[i + 1:]:
                    if tensor in op2.inputs:
                        op2.inputs = _with_tensor_replaced(op2.inputs, tensor, new_tensor)
                    if tensor in op2.outputs:
                        new_tensor = duplicate(tensor)
                        op2.outputs = _with_tensor_replaced(op2.outputs, tensor, new_tensor)
                replace_tensor_in_outputs(g, tensor, new_tensor)
        outputs_of_earlier_ops.update(list(op.outputs))
