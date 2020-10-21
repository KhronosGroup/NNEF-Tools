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


def generate_tensor_names_from_op_type(graph, keep_io_names=False):
    op_counts = {}
    for tensor in graph.tensors:
        if keep_io_names and tensor.name is not None and (tensor in graph.inputs or tensor in graph.outputs):
            continue

        if tensor.producer is not None:
            op_type = tensor.producer.type
            idx = op_counts.get(op_type, 0) + 1
            op_counts[op_type] = idx
            tensor.name = op_type + str(idx)
        else:
            tensor.name = None


def generate_op_names_from_op_type(graph):
    op_counts = {}
    for op in graph.operations:
        idx = op_counts.get(op.type, 0) + 1
        op_counts[op.type] = idx
        op.name = op.type + str(idx)


def replace_tensor_in_graph_inputs(graph, old_tensor, new_tensor):
    graph.inputs = [new_tensor if t is old_tensor else t for t in graph.inputs]


def replace_tensor_in_graph_outputs(graph, old_tensor, new_tensor):
    graph.outputs = [new_tensor if t is old_tensor else t for t in graph.outputs]


def replace_tensor_in_consumers(graph, old_tensor, new_tensor):
    for consumer in list(old_tensor.consumers):     # copy list to avoid changes during iteration
        sequence = tuple if isinstance(consumer.inputs, tuple) else list
        consumer.inputs = sequence(new_tensor if t is old_tensor else t for t in consumer.inputs)

    replace_tensor_in_graph_outputs(graph, old_tensor, new_tensor)


def replace_tensor_in_producers(graph, old_tensor, new_tensor):
    for producer in list(old_tensor.producers):     # copy list to avoid changes during iteration
        sequence = tuple if isinstance(producer.outputs, tuple) else list
        producer.outputs = sequence(new_tensor if t is old_tensor else t for t in producer.outputs)

    replace_tensor_in_graph_inputs(graph, old_tensor, new_tensor)


def bypass_and_remove(graph, op, remove_input_not_output=False):
    assert len(op.outputs) == 1 and len(op.inputs) == 1

    op_input = op.input
    op_output = op.output

    graph.remove_operation(op, unlink=True)

    if remove_input_not_output:
        replace_tensor_in_consumers(graph, op_input, op_output)
        replace_tensor_in_producers(graph, op_input, op_output)
        graph.remove_tensor(op_input)
    else:
        replace_tensor_in_consumers(graph, op_output, op_input)
        replace_tensor_in_producers(graph, op_output, op_input)
        graph.remove_tensor(op_output)


def replace_chain(graph, types, func, allow_forks=False):
    def _match_type(type, template):
        return type in template if isinstance(template, set) else type == template

    def _match_link(op, template, is_last):
        return _match_type(op.type, template) and (len(op.outputs) == 1 or is_last)

    def _match_chain(op, types, allow_forks):
        if not _match_link(op, types[0], is_last=len(types) == 1):
            return None

        chain = [op]
        tensor = op.output
        for idx, type in enumerate(types[1:]):
            is_last = idx + 1 == len(types) - 1

            if not allow_forks and len(tensor.consumers) > 1:
                return None

            op = next((consumer for consumer in tensor.consumers if _match_link(consumer, type, is_last)), None)
            if op is None:
                return None

            chain.append(op)
            if not is_last:
                tensor = op.output

        return chain

    changed = False
    i = 0
    while i < len(graph.operations):
        count = len(graph.operations)
        chain = _match_chain(graph.operations[i], types, allow_forks)
        if chain is not None and func(*chain) is not False:
            k = i
            while graph.operations[k] is not chain[-1]:
                k += 1

            for j in range(count, len(graph.operations)):
                graph.move_operation(j, k)
                k += 1

            offs = len(chain) - 1
            while offs > 0 and len(chain[offs - 1].output.consumers) == 1:
                offs -= 1

            interns = [op.output for op in chain[offs:-1]]
            graph.remove_operations(chain[offs:], unlink=True)
            graph.remove_tensors(interns)
            changed = True
        else:
            i += 1
    return changed


def remove_unreachable(graph):
    visited = {tensor.producer for tensor in graph.outputs}
    queue = list(visited)

    k = 0
    while k < len(queue):
        op = queue[k]
        k += 1

        for tensor in op.inputs:
            if tensor.producer is not None and tensor.producer not in visited and \
                    (tensor not in graph.inputs or len(tensor.producer.inputs) == 0):
                visited.add(tensor.producer)
                queue.append(tensor.producer)

    graph.remove_operations({op for op in graph.operations if op not in visited}, unlink=True)
    graph.remove_tensors({tensor for tensor in graph.tensors
                          if len(tensor.producers) == 0 and len(tensor.consumers) == 0
                          and tensor not in graph.inputs and tensor not in graph.outputs})
