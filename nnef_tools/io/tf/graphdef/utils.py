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

from .protobuf import *
from .writer import _build_attribute
from .reader import _get_attributes
import numpy as np
import six
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf


def import_graph_def(graph_def):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return graph


def export_graph_def(graph):
    return graph.as_graph_def(add_shapes=True)


def reinfer_shapes(graph_def):
    return export_graph_def(import_graph_def(graph_def))


def _try_eval(tensor, session):
    try:
        if tensor.dtype == tf.resource or tensor.dtype == tf.string:
            return None
        value = tensor.eval(session=session)
        print("Evaluated constant tensor '{}'".format(tensor.name))
        return value
    except:
        return None


def _build_node(type, name, attribs, inputs):
    node_def = NodeDef()
    node_def.op = type
    node_def.name = name
    if len(inputs):
        node_def.input.extend(inputs)
    for name, value in attribs.items():
        _build_attribute(node_def.attr[name], value)
    return node_def


def _make_const_node(value, name):
    return _build_node('Const', name, {'dtype': value.dtype.type, 'value': value, '_output_shapes': [value.shape]}, [])


def _make_identity_node(input, name, dtype, shape):
    return _build_node('Identity', name, {'T': dtype, '_output_shapes': [shape]}, [input])


def _freeze_shape_tensors(graph_def):
    graph = import_graph_def(graph_def)

    evaluated = {}
    for op in graph.get_operations():
        if op.type == 'Shape':
            shape = op.inputs[0].shape
            if shape.dims is not None and all(item is not None for item in shape.as_list()):
                evaluated[op.name] = np.array(shape, dtype=np.int32)
                print("Evaluated Shape op '{}' to {}".format(op.name, str(shape)))

    changed = False
    new_graph_def = GraphDef()
    for node in graph_def.node:
        value = evaluated.get(node.name)
        if value is not None:
            new_graph_def.node.append(_make_const_node(value, node.name))
            changed = True
        else:
            new_graph_def.node.append(node)

    return new_graph_def, changed


def _remove_const_control_dependencies(graph_def):
    for node in graph_def.node:
        if node.op == 'Const':
            for idx in reversed(range(len(node.input))):
                name = node.input[idx]
                if name[0] == '^':
                    del node.input[idx]

    return graph_def


def _remove_zero_index(name):
    return name[:-2] if name.endswith(':0') else name


def _remove_const_identities(graph_def):
    graph = import_graph_def(graph_def)

    removables = {op.name: _remove_zero_index(op.inputs[0].name)
                  for op in graph.get_operations()
                  if op.type == 'Identity' and op.inputs[0].op.type == 'Const'}

    for node in graph_def.node:
        for i in range(len(node.input)):
            replacement = removables.get(_op_name_from_tensor(node.input[i]))
            if replacement:
                node.input[i] = replacement

    new_graph_def = GraphDef()
    for node in graph_def.node:
        if node.name not in removables:
            new_graph_def.node.append(node)

    return new_graph_def


def _eval_candidates(graph):
    evaluables = set()
    changed = True
    while changed:
        changed = False
        for op in graph.get_operations():
            if op not in evaluables and all(tensor.op in evaluables for tensor in op.inputs) and op.type != 'Placeholder':
                evaluables.add(op)
                changed = True

    candidates = set()
    for op in graph.get_operations():
        if op not in evaluables:
            for tensor in op.inputs:
                if tensor.op in evaluables and tensor.op.type != 'Const':
                    candidates.add(tensor)

    return candidates


def _fold_constant_tensors(graph_def):
    graph = import_graph_def(graph_def)

    evaluated = {}
    with tf.Session(graph=graph) as session:
        for tensor in _eval_candidates(graph):
            evaluated[tensor.name] = _try_eval(tensor, session)

    results = {}
    for op in graph.get_operations():
        results[op.name] = [evaluated.get(tensor.name) for tensor in op.outputs]

    remap = {}
    changed = False
    new_graph_def = GraphDef()
    for node in graph_def.node:
        values = results[node.name]
        all_evaluated = all(value is not None for value in values) and len(values) > 0

        for idx, value in enumerate(values):
            if value is not None:
                arg_name = node.name if idx == 0 else node.name + ':{}'.format(idx)
                const_name = node.name if idx == 0 and all_evaluated else node.name + '//{}'.format(idx)
                remap[arg_name] = const_name
                new_graph_def.node.append(_make_const_node(value, const_name))
                changed = True
        if not all_evaluated:
            new_graph_def.node.append(node)

    for node in new_graph_def.node:
        for i in range(len(node.input)):
            remapped = remap.get(node.input[i])
            if remapped is not None:
                node.input[i] = remapped

    return new_graph_def, changed


def _find_reachables_forward(graph, reachables):
    changed = True
    while changed:
        changed = False
        for op in graph.get_operations():
            if op.name not in reachables and any(tensor.op.name in reachables for tensor in op.inputs):
                reachables.add(op.name)
                changed = True
    return reachables


def _find_reachables_backward(graph, reachables):
    changed = True
    while changed:
        changed = False
        for op in reversed(graph.get_operations()):
            if op.name in reachables:
                for tensor in op.inputs:
                    if tensor.op.name not in reachables:
                        reachables.add(tensor.op.name)
                        changed = True
    return reachables


def _retain_nodes(graph_def, node_names):
    new_graph_def = GraphDef()
    for node in graph_def.node:
        if node.name in node_names:
            new_graph_def.node.append(node)

    for node in new_graph_def.node:
        for idx in reversed(range(len(node.input))):
            name = node.input[idx]
            if name[0] == '^' and name[1:] not in node_names:
                del node.input[idx]

    return new_graph_def


def _retain_reachables_from_placeholders(graph_def):
    graph = import_graph_def(graph_def)

    reachables = {op.name for op in graph.get_operations() if op.type == 'Placeholder'}
    if len(reachables) == 0:
        return graph_def

    reachables = _find_reachables_forward(graph, reachables)
    reachables = _find_reachables_backward(graph, reachables)

    return _retain_nodes(graph_def, reachables)


def _op_name_from_tensor(name):
    if name[0] == '^':
        name = name[1:]
    pos = name.find(':')
    if pos != -1 and name[pos+1:].isdigit():
        name = name[:pos]
    return name


def fold_constant_tensors(graph_def):
    graph_def = _remove_const_control_dependencies(graph_def)
    graph_def = _remove_const_identities(graph_def)

    graph_def, changed = _freeze_shape_tensors(graph_def)
    graph_def, changed = _fold_constant_tensors(graph_def)
    while changed:
        graph_def, changed = _freeze_shape_tensors(graph_def)
        if changed:
            graph_def, changed = _fold_constant_tensors(graph_def)

    graph_def = _retain_reachables_from_placeholders(graph_def)

    return reinfer_shapes(graph_def)


def set_input_shapes(graph_def, input_shapes):
    graph = import_graph_def(graph_def)
    placeholders = {op.name: (op.outputs[0].shape, op.outputs[0].dtype)
                    for op in graph.get_operations() if op.type == 'Placeholder'}

    graph = tf.Graph()
    with graph.as_default():
        input_map = {}
        for name, shape in six.iteritems(input_shapes):
            if name not in placeholders:
                raise IOError("Model has no input named '{}'".format(name))

            orig_shape, dtype = placeholders[name]
            if orig_shape.rank is not None and len(shape) != orig_shape.rank:
                raise IOError("Shape rank for input '{}' does not match that of the model ({} vs {})"
                              .format(name, len(shape), orig_shape.rank))

            input_map[name] = tf.placeholder(shape=shape, dtype=dtype, name=name)

        for name, (shape, dtype) in placeholders.items():
            if name not in input_map:
                input_map[name] = tf.placeholder(shape=shape, dtype=dtype, name=name)

        tf.import_graph_def(graph_def, name='', input_map=input_map)

    used = {tensor.op.name for op in graph.get_operations() for tensor in op.inputs}

    graph_def = GraphDef()
    for op in graph.get_operations():
        if op.type != 'Placeholder' or op.name in used:
            graph_def.node.append(op.node_def)

    return reinfer_shapes(graph_def)


def retain_reachables_from_outputs(graph_def, output_names):
    graph = import_graph_def(graph_def)

    reachables = _find_reachables_backward(graph, set(output_names))

    return _retain_nodes(graph_def, reachables)


def insert_rename_identities(graph_def, tensor_rename):
    for tensor, name in six.iteritems(tensor_rename):
        tensor_name = _remove_zero_index(tensor.name)
        if name != tensor_name:
            graph_def.node.append(_make_identity_node(tensor_name, name,
                                                      tensor.dtype.as_numpy_dtype,
                                                      tuple(tensor.shape.as_list())))
    return graph_def


def check_finite(graph_def):
    for node in graph_def.node:
        attribs = _get_attributes(node.attr)
        for key, value in six.iteritems(attribs):
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
                raise ValueError("Attribute '{}' of op '{}' named '{}' contains nan or inf".
                                 format(key, node.op, node.name))


def check_variables(session):
    variables = tf.global_variables()
    for variable in variables:
        value = session.run(variable)
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError("Variable '{}' contains nan or inf".format(variable.name))
