from . import *
import re


def _split_counter_from_name(str):
    if len(str) == 0 or not str[-1].isdigit():
        return str, None

    i = len(str)
    while i > 0:
        if not str[i-1].isdigit():
            return str[:i], int(str[i:])
        i -= 1
    return None, int(str)


def generate_tensor_names_from_op_type(model, keep_io_names=False):
    used_names = set()
    if keep_io_names:
        for graph in model.graphs:
            used_names.update(tensor.name for tensor in graph.inputs if tensor.name is not None)
            used_names.update(tensor.name for tensor in graph.outputs if tensor.name is not None)

    op_counts = {}

    for graph in model.graphs:
        for op in graph.operations:
            for tensor in op.outputs:
                if keep_io_names and tensor.name is not None and (tensor in graph.inputs or tensor in graph.outputs):
                    continue

                idx = op_counts.get(op.type, 0) + 1
                while op.type + str(idx) in used_names:
                    idx += 1

                op_counts[op.type] = idx
                tensor.name = op.type + str(idx)

        for tensor in graph.tensors:
            if tensor.data is not None:
                tensor.name = None


def generate_missing_tensor_names_from_op_type(model):
    counters = {}
    for graph in model.graphs:
        for tensor in graph.tensors:
            if tensor.name is not None:
                name, count = _split_counter_from_name(tensor.name)
                if name is not None and count is not None:
                    counters[name] = max(counters.get(name, 0), count)

        for tensor in graph.tensors:
            if tensor.name is None and tensor.producer is not None:
                op = tensor.producer
                idx = counters.get(op.type, 0) + 1
                counters[op.type] = idx
                tensor.name = op.type + str(idx)


def generate_op_names_from_op_type(model):
    op_counts = {}
    for graph in model.graphs:
        for op in graph.operations:
            idx = op_counts.get(op.type, 0) + 1
            op_counts[op.type] = idx
            op.name = op.type + str(idx)


def valid_id(name):
    id = re.sub('[^~_0-9a-zA-Z]+', '_', name)
    if len(id) > 0 and id[0].isdigit():
        id = "__" + id
    return id


def ensure_valid_ids(model):
    if model.name is not None:
        model.name = valid_id(model.name)

    graph_names = {}
    for graph in model.graphs:
        if graph.name is not None:
            graph.name = valid_id(graph.name)
            count = graph_names.get(graph.name, 0)
            graph_names[graph.name] = count + 1
            if count:
                graph.name += f'_{count + 1}'

    tensor_names = {}
    for graph in model.graphs:
        for tensor in graph.tensors:
            if tensor.name is not None:
                tensor.name = valid_id(tensor.name)
                count = tensor_names.get(tensor.name, 0)
                tensor_names[tensor.name] = count + 1
                if count:
                    tensor.name += f'_{count + 1}'


def replace_tensor_in_graph_inputs(graph, old_tensor, new_tensor):
    graph.inputs = tuple(new_tensor if t is old_tensor else t for t in graph.inputs)


def replace_tensor_in_graph_outputs(graph, old_tensor, new_tensor):
    graph.outputs = tuple(new_tensor if t is old_tensor else t for t in graph.outputs)


def replace_tensor_in_consumers(old_tensor, new_tensor):
    for consumer in list(old_tensor.consumers):     # copy list to avoid changes during iteration
        assert isinstance(consumer.inputs, tuple)
        consumer.inputs = replace_tensor_in_sequence_nested(consumer.inputs, old_tensor, new_tensor)


def replace_tensor_in_producers(old_tensor, new_tensor):
    producer = old_tensor.producer
    if producer is not None:
        assert isinstance(producer.outputs, tuple)
        producer.outputs = replace_tensor_in_sequence_nested(producer.outputs, old_tensor, new_tensor)


def replace_tensor_in_sequence_nested(sequence, old_tensor, new_tensor):
    type = tuple if isinstance(sequence, tuple) else list
    return type((new_tensor if item is old_tensor else item) if isinstance(item, Tensor) or item is None
                else replace_tensor_in_sequence_nested(item, old_tensor, new_tensor)
                for item in sequence)


def bypass_and_remove(graph, op, remove_input_not_output=False):
    assert len(op.outputs) == 1 and len(op.inputs) == 1

    op_input = op.input
    op_output = op.output

    graph.remove_operation(op, unlink=True)

    if remove_input_not_output:
        replace_tensor_in_consumers(op_input, op_output)
        replace_tensor_in_producers(op_input, op_output)
        replace_tensor_in_graph_inputs(graph, op_input, op_output)
        replace_tensor_in_graph_outputs(graph, op_input, op_output)
        graph.remove_tensor(op_input)
    else:
        replace_tensor_in_consumers(op_output, op_input)
        replace_tensor_in_producers(op_output, op_input)
        replace_tensor_in_graph_inputs(graph, op_output, op_input)
        replace_tensor_in_graph_outputs(graph, op_output, op_input)
        graph.remove_tensor(op_output)


def _replace_chain(graph, types, func, allow_forks=False):
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


def replace_chain(arg, types, func, allow_forks=False):
    if isinstance(arg, Graph):
        return _replace_chain(arg, types, func, allow_forks)
    else:
        return any(_replace_chain(graph, types, func, allow_forks) for graph in arg.graphs)


def _remove_unreachables(graph):
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
                          if not tensor.has_producer and not tensor.has_consumer
                          and tensor not in graph.inputs and tensor not in graph.outputs})


def remove_unreachables(arg):
    if isinstance(arg, Graph):
        _remove_unreachables(arg)
    else:
        for graph in arg.graphs:
            _remove_unreachables(graph)


def _remove_unused_tensors(graph):
    graph.remove_tensors([tensor for tensor in graph.tensors
                          if not tensor.has_producer and tensor not in graph.inputs and
                          not tensor.has_consumer and not (tensor in graph.outputs and tensor.data is not None)])


def remove_unused_tensors(arg):
    if isinstance(arg, Graph):
        _remove_unused_tensors(arg)
    else:
        for graph in arg.graphs:
            _remove_unused_tensors(graph)


def recursive_itemize(arg):
    if type(arg) is list or type(arg) is tuple:
        for item in arg:
            yield from recursive_itemize(item)
    else:
        yield arg
