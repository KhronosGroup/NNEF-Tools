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

from .protobuf import GraphDef, NodeDef
from .writer import _build_attribute
from .utils import import_graph_def
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
from collections.abc import Sequence
import inspect


class _Composite:

    instances = []

    def __init__(self, id, func, attribs, inputs, outputs):
        self.id = id
        self.func = func
        self.attribs = attribs
        self.inputs = inputs
        self.outputs = outputs

    @staticmethod
    def function(func):
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)

            name = kwargs.get('name')
            if name is not None:
                del kwargs['name']
            id = name or len(_Composite.instances)

            signature = inspect.signature(func)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            attribs = {name: value for name, value in bound.arguments.items()
                       if not isinstance(value, tf.Tensor) and value is not None}
            inputs = [value for value in bound.arguments.values()
                      if isinstance(value, tf.Tensor)]
            outputs = (results,) if not isinstance(results, (list, tuple)) else results

            assert all(isinstance(value, tf.Tensor) for value in outputs), \
                "Results of composite function must be tensors"
            assert not any(tensor in inputs for tensor in outputs), \
                "Results of composite function cannot be input arguments at the same time"

            _Composite.instances.append(_Composite(id, func, attribs, inputs, outputs))

            return results
        return wrapper

    @property
    def name(self):
        return self.id if isinstance(self.id, str) else 'Composite' + str(self.id)


function = _Composite.function


def _is_tensor(value):
    return isinstance(value, tf.Tensor) or (isinstance(value, Sequence) and all(_is_tensor(item) for item in value))


def _node_name_from_tensor(name):
    if name[0] == '^':
        name = name[1:]
    pos = name.find(':')
    if pos != -1 and name[pos+1:].isdigit():
        name = name[:pos]
    return name


def _input_name_from_tensor(name):
    return name[:-2] if name.endswith(':0') else name


def _build_node_def(composite):
    node_def = NodeDef()
    node_def.op = 'PyFunc'
    node_def.name = composite.name
    node_def.input.extend([_input_name_from_tensor(arg.name) for arg in composite.inputs])

    input_dtypes = [tensor.dtype.as_numpy_dtype for tensor in composite.inputs]
    output_dtypes = [tensor.dtype.as_numpy_dtype for tensor in composite.outputs]
    output_shapes = [tuple(tensor.shape.as_list()) for tensor in composite.outputs]

    _build_attribute(node_def.attr['Tin'], input_dtypes)
    _build_attribute(node_def.attr['Tout'], output_dtypes)
    _build_attribute(node_def.attr['token'], composite.func.__name__)
    _build_attribute(node_def.attr['_output_shapes'], output_shapes)
    for name, value in composite.attribs.items():
        _build_attribute(node_def.attr['_$' + name + '$_'], value)
    return node_def


def _remap_tensors(tensors, graph):
    return type(tensors)(graph.get_tensor_by_name(tensor.name) for tensor in tensors)


def _tensor_producers_and_consumers(graph):
    producers_and_consumers = {tensor: [tensor.op] for op in graph.get_operations() for tensor in op.outputs}
    for op in graph.get_operations():
        for tensor in op.inputs:
            ops = producers_and_consumers[tensor]
            if op not in ops:
                ops.append(op)

    return producers_and_consumers


def _find_subgraph(composite, producers_and_consumers):
    queue = [tensor.op for tensor in composite.outputs]
    subgraph = {item.name for item in queue}

    idx = 0
    while idx < len(queue):
        op = queue[idx]
        idx += 1
        tensors = [tensor for tensor in op.inputs if tensor not in composite.inputs] + \
                  [tensor for tensor in op.outputs if tensor not in composite.outputs]
        for tensor in tensors:
            for op in producers_and_consumers[tensor]:
                if op.name not in subgraph:
                    subgraph.add(op.name)
                    queue.append(op)
    return subgraph


def replace_composites_with_py_functions(graph_def):
    graph = import_graph_def(graph_def)
    for composite in _Composite.instances:
        composite.inputs = _remap_tensors(composite.inputs, graph)
        composite.outputs = _remap_tensors(composite.outputs, graph)

    producers_and_consumers = _tensor_producers_and_consumers(graph)

    tensor_remap = {}
    subgraph_ops = set()
    for composite in _Composite.instances:
        subgraph_ops.update(_find_subgraph(composite, producers_and_consumers))

        for idx, tensor in enumerate(composite.outputs):
            tensor_remap[_input_name_from_tensor(tensor.name)] = \
                composite.name + ':' + str(idx) if idx > 0 else composite.name

    new_graph_def = GraphDef()
    for node in graph_def.node:
        if node.name not in subgraph_ops:
            new_graph_def.node.append(node)

    for composite in _Composite.instances:
        new_graph_def.node.append(_build_node_def(composite))

    for node in new_graph_def.node:
        for i in range(len(node.input)):
            remapped = tensor_remap.get(node.input[i])
            if remapped is not None:
                node.input[i] = remapped

    return new_graph_def


def reset_composites():
    _Composite.instances = []
