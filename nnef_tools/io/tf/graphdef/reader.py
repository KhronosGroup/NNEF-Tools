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

from __future__ import division, print_function, absolute_import

from collections import namedtuple
from ....model import *
from ....utils.types import as_str
from .protobuf import *
import numpy as np
import six


Function = namedtuple('Function', ['name', 'attrs'])


_DtypeToNumpy = {
    'DT_INVALID': None,
    'DT_RESOURCE': np.dtype([('resource', np.int32)]),
    'DT_HALF': np.float16,
    'DT_FLOAT': np.float32,
    'DT_DOUBLE': np.float64,
    'DT_INT8': np.int8,
    'DT_INT16': np.int16,
    'DT_INT32': np.int32,
    'DT_INT64': np.int64,
    'DT_UINT8': np.uint8,
    'DT_UINT16': np.uint16,
    'DT_UINT32': np.uint32,
    'DT_UINT64': np.uint64,
    'DT_BOOL': np.bool_,
    'DT_STRING': np.str_,
    'DT_COMPLEX64': np.complex64,
    'DT_COMPLEX128': np.complex128,
}


def _get_shape(shape_proto):
    return tuple(int(dim.size) if dim.size >= 0 else None for dim in shape_proto.dim) \
        if not shape_proto.unknown_rank else None


def _get_dtype(dtype_enum):
    dtype = _DtypeToNumpy[DataType.Name(dtype_enum)]
    assert dtype is not None, "non-numeric dtype '{}' in attribute".format(DataType.Name(dtype_enum))
    return dtype


def _get_nonempty_items(message, fields):
    for field in fields:
        items = getattr(message, field)
        if len(items):
            return field, items

    return None, None


def _get_tensor(tensor_proto):
    shape = _get_shape(tensor_proto.tensor_shape)
    dtype = _get_dtype(tensor_proto.dtype)

    if len(tensor_proto.tensor_content):
        data = np.frombuffer(tensor_proto.tensor_content, dtype=dtype).reshape(shape)
    else:
        field, items = _get_nonempty_items(tensor_proto,
                                           fields=['half_val', 'float_val', 'double_val', 'int_val', 'int64_val',
                                                   'bool_val', 'string_val', 'uint32_val', 'uint64_val',
                                                   'resource_handle_val', 'scomplex_val', 'dcomplex_val'])

        if items is None and any(s == 0 for s in shape):
            items = []

        assert items is not None, "tensor items are empty, dtype = {}, shape = {}".format(dtype, shape)

        items = [item for item in items]
        if len(items) == int(np.prod(shape)):
            data = np.array(items, dtype=dtype).reshape(shape)
        else:
            assert len(items) == 1
            data = np.full(shape=shape, dtype=dtype, fill_value=items[0])

    return data


def _get_func(name_attrlist_proto):
    return Function(name_attrlist_proto.name, _get_attributes(name_attrlist_proto.attr))


def _get_attribute(field, value):
    if field == 'i' or field == 'f' or field == 'b' or field == 'placeholder':
        return value
    elif field == 's':
        return as_str(value.decode())
    elif field == 'shape':
        return _get_shape(value)
    elif field == 'type':
        return _get_dtype(value)
    elif field == 'tensor':
        return _get_tensor(value)
    elif field == 'func':
        return _get_func(value)
    elif field == 'list':
        field, items = _get_nonempty_items(value, fields=['i', 'f', 'b', 's', 'shape', 'type', 'tensor', 'func'])
        return [_get_attribute(field, item) for item in items] if items is not None else []

    assert False


def _get_attributes(attr_map_proto):
    attributes = {}
    for name, value in attr_map_proto.items():
        field = value.WhichOneof('value')
        if field is not None:
            value = getattr(value, field)
            attributes[as_str(name)] = _get_attribute(field, value)
        else:
            attributes[as_str(name)] = None

    return attributes


def _get_output_name(node_name, idx):
    return node_name + ':' + str(idx) if idx > 0 else node_name


def _has_output_shapes(graph_def):
    return all('_output_shapes' in node.attr and node.attr['_output_shapes'].WhichOneof('value') is not None
               for node in graph_def.node)


def _add_output_shapes(graph_def):
    try:
        import tensorflow.compat.v1 as tf
    except ImportError:
        import tensorflow as tf

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
        return graph.as_graph_def(add_shapes=True)


def _get_dtypes(graph_def):
    try:
        import tensorflow.compat.v1 as tf
    except ImportError:
        import tensorflow as tf

    dtypes = {}

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
        for op in graph.get_operations():
            for tensor in op.outputs:
                name = tensor.name[:-2] if tensor.name.endswith(':0') else tensor.name
                dtypes[name] = tensor.dtype.as_numpy_dtype if tensor.dtype != tf.resource else _DtypeToNumpy['DT_RESOURCE'].type

    return dtypes


def _get_output_shapes(attr_map_proto):
    value = attr_map_proto['_output_shapes']
    field = value.WhichOneof('value')
    if field is None:
        return None

    value = getattr(value, field)
    return _get_attribute(field, value)


def build_graph(graph_def):
    graph = Graph()

    dtypes = _get_dtypes(graph_def)

    # create tensors
    node_outputs = {}
    for node in graph_def.node:
        output_shapes = _get_output_shapes(node.attr)
        if output_shapes is not None:
            name = as_str(node.name)
            node_outputs[name] = [Tensor(graph, _get_output_name(name, idx), shape=shape, dtype=dtypes.get(name))
                                  for idx, shape in enumerate(output_shapes)]

    tensors = {tensor.name: tensor for outputs in six.itervalues(node_outputs) for tensor in outputs}

    # create ops
    for node in graph_def.node:
        attributes = _get_attributes(node.attr)
        inputs = [tensors[name] for name in node.input if not name.startswith('^')]
        outputs = node_outputs[node.name] if node.name in node_outputs else []

        Operation(graph,
                  type=as_str(node.op),
                  name=as_str(node.name),
                  inputs=inputs,
                  outputs=outputs,
                  attribs=attributes)

    graph.inputs = [node_outputs[node.name][0] for node in graph_def.node if node.op == 'Placeholder']
    graph.outputs = [output for op in graph.operations if all(len(output.consumers) == 0 for output in op.outputs)
                     for output in op.outputs]
    return graph


def _unpack_custom_ops(graph):
    for op in graph.operations:
        if op.type == 'PyFunc':
            op.custom = True
            op.type = op.attribs['token']
            op.attribs = {key[2:-2]: value for key, value in six.iteritems(op.attribs)
                          if key.startswith('_$') and key.endswith('$_')}


def read_graphdef(filename, input_shapes, fold_constants):
    graph_def = GraphDef()
    with open(filename, 'rb') as file:
        graph_def.ParseFromString(file.read())

    if not _has_output_shapes(graph_def):
        graph_def = _add_output_shapes(graph_def)

    if input_shapes is not None:
        from .utils import set_input_shapes
        graph_def = set_input_shapes(graph_def, input_shapes)

    if fold_constants:
        from .utils import fold_constant_tensors
        graph_def = fold_constant_tensors(graph_def)

    graph = build_graph(graph_def)
    _unpack_custom_ops(graph)

    return graph


class Reader(object):

    def __init__(self, fold_constants=False):
        self._fold_constants = fold_constants

    def __call__(self, filename, input_shapes=None):
        return read_graphdef(filename, input_shapes, self._fold_constants)
