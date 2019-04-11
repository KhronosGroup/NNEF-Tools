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

from collections import namedtuple, OrderedDict

import numpy as np
import six

import nnef_tools.io.tensorflow.tf_pb as tf_pb
from nnef_tools.conversion import conversion_info
from nnef_tools.conversion.tensorflow import tf_pb_to_tf_py, tf_py_to_tf_pb
from nnef_tools.core import utils
from nnef_tools.io.tensorflow.tf_graph import *

_NumpyDtype = {
    'DT_INVALID': None,
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
    'DT_BOOL': np.bool,
    'DT_STRING': np.str,
    'DT_COMPLEX64': np.complex64,
    'DT_COMPLEX128': np.complex128,
}

# Ops with one output are not listed
_OutputCount = {
    'FusedBatchNorm': 5,
    'MaxPoolWithArgmax': 2,
    'Split': 'num_split',
    'SplitV': 'num_split',
}

Function = namedtuple('Function', ['name', 'attrs'])


class _DType(object):
    def __init__(self, value):
        self.value = value


def _get_shape(shape_proto):
    return [utils.anyint_to_int(dim.size) if utils.is_anyint(dim.size) else dim.size for dim in shape_proto.dim]


def _get_dtype(dtype_enum):
    return tf_pb.DataType.Name(dtype_enum)


def _get_nonempty_items(message, fields):
    for field in fields:
        items = getattr(message, field)
        if len(items):
            return field, items

    return None, None


def _get_tensor(tensor_proto, graph):
    shape = _get_shape(tensor_proto.tensor_shape)
    dtype = _get_dtype(tensor_proto.dtype)
    if len(tensor_proto.tensor_content):
        arr = np.frombuffer(tensor_proto.tensor_content, dtype=_NumpyDtype[dtype])
        data = arr.reshape(shape)
    else:
        field, items = _get_nonempty_items(tensor_proto, fields=['float_val',
                                                                 'double_val',
                                                                 'int_val',
                                                                 'int64_val',
                                                                 'bool_val'])
        if items is not None:
            items = [item for item in items]
            if len(items) == int(np.prod(shape)):
                data = np.array(items).reshape(shape)
            else:
                assert len(items) == 1
                data = np.full(shape=shape, fill_value=items[0])
        else:
            data = None

    return TFTensor(graph, name=None, shape=shape, dtype=dtype, data=data)


def _get_func(name_attrlist_proto):
    return Function(name_attrlist_proto.name, _get_attributes(name_attrlist_proto.attr))


def _get_attribute(field, value, graph):
    if field == 'i' or field == 'f' or field == 'b' or field == 'placeholder':
        if utils.is_anyint(value):
            return utils.anyint_to_int(value)
        return value
    elif field == 's':
        return utils.anystr_to_str(value.decode())
    elif field == 'shape':
        return _get_shape(value)
    elif field == 'type':
        return _get_dtype(value)
    elif field == 'tensor':
        return _get_tensor(value, graph)
    elif field == 'func':
        return _get_func(value)
    elif field == 'list':
        field, items = _get_nonempty_items(value, fields=['i', 'f', 'b', 's', 'shape', 'type', 'tensor', 'func'])
        if items is None:
            return []
        return [_get_attribute(field, item, graph) for item in items]

    assert False


def _get_attributes(attr_map_proto, graph):
    attributes = {}
    for name, value in attr_map_proto.items():
        if not name.startswith('_'):
            field = value.WhichOneof('value')
            value = getattr(value, field)
            attributes[utils.anystr_to_str(name)] = _get_attribute(field, value, graph)

    return attributes


def _build_shape(shape_proto, shape):
    for item in shape:
        dim = shape_proto.dim.add()
        dim.size = item


def _build_dtype(name):
    return tf_pb.DataType.Value(name)


def _build_tensor(tensor_proto, tensor):
    if tensor.dtype is not None:
        tensor_proto.dtype = _build_dtype(tensor.dtype)

    if tensor.shape is not None:
        _build_shape(tensor_proto.tensor_shape, tensor.shape)

    if tensor.data is not None:
        if isinstance(tensor.data, np.ndarray):
            tensor_proto.tensor_content = \
                tensor.data.astype(_NumpyDtype[tensor.dtype]).reshape([-1]).view(np.uint8).tobytes()
        elif isinstance(tensor.data, list):
            if tensor.dtype == 'DT_BOOL':
                tensor_proto.bool_val.extend(tensor.data)
            elif tensor.dtype == 'DT_INT32':
                tensor_proto.int_val.extend(tensor.data)
            elif tensor.dtype == 'DT_INT64':
                tensor_proto.int64_val.extend(tensor.data)
            elif tensor.dtype == 'DT_FLOAT':
                tensor_proto.float_val.extend(tensor.data)
            elif tensor.dtype == 'DT_DOUBLE':
                tensor_proto.double_val.extend(tensor.data)
            else:
                raise TypeError('unable to build tensor proto message from data type: ' + tensor.dtype)
        elif tensor.dtype == 'DT_BOOL':
            tensor_proto.bool_val.append(tensor.data)
        elif tensor.dtype == 'DT_INT32':
            tensor_proto.int_val.append(tensor.data)
        elif tensor.dtype == 'DT_INT64':
            tensor_proto.int64_val.append(tensor.data)
        elif tensor.dtype == 'DT_FLOAT':
            tensor_proto.float_val.append(tensor.data)
        elif tensor.dtype == 'DT_DOUBLE':
            tensor_proto.double_val.append(tensor.data)
        else:
            raise TypeError('unable to build tensor proto message from data type: ' + tensor.dtype)

    return tensor_proto


def _build_attribute(attr_proto, value):
    if isinstance(value, bool):  # must be before int
        attr_proto.b = value
    elif isinstance(value, int):
        attr_proto.i = value
    elif isinstance(value, float):
        attr_proto.f = value
    elif isinstance(value, str):
        attr_proto.s = value.encode()
    elif isinstance(value, _DType):
        attr_proto.type = value.value
    elif isinstance(value, TFTensor):
        _build_tensor(attr_proto.tensor, value)
    elif isinstance(value, list):
        if isinstance(value[0], int):
            attr_proto.list.i.extend(value)
        elif isinstance(value[0], float):
            attr_proto.list.f.extend(value)
        elif isinstance(value[0], bool):
            attr_proto.list.b.extend(value)
        elif isinstance(value[0], str):
            attr_proto.list.s.extend([item.encode() for item in value])
        elif isinstance(value[0], TFTensor):
            for item in value:
                _build_tensor(attr_proto.list.add(), item)
        else:
            raise TypeError('unable to build attribute proto message from type: ' + type(value[0]))
    else:
        raise TypeError('unable to build attribute proto message from type: ' + type(value))

    return attr_proto


def _build_node(node_def, operation, names):
    node_def.name = names[operation]
    node_def.op = operation.name
    node_def.input.extend([names[tensor] for tensor in operation.inputs])
    for name, value in operation.attribs.items():
        _build_attribute(node_def.attr[name],
                         _DType(_build_dtype(value)) if name in ['T', 'Targmax', 'Index'] else value)
    return node_def


def _generate_names(graph):
    ng = utils.NameGenerator()
    names = {}
    for op in graph.operations:
        new_name = ng.get_new_name(op.name)
        names[op] = new_name
        for i, tensor in enumerate(op.outputs):
            if i == 0:
                names[tensor] = new_name
            else:
                names[tensor] = new_name + ':' + str(i)
    for tensor in graph.tensors:
        if tensor.producer is None:
            if tensor.data is None:
                if tensor in graph.inputs and graph.input_ids:
                    names[tensor] = ng.get_new_name(graph.input_ids[graph.inputs.index(tensor)])
                else:
                    names[tensor] = ng.get_new_name('Placeholder')
            elif isinstance(tensor.data, np.ndarray):
                if tensor.label:
                    names[tensor] = ng.get_new_name(tensor.label)
                else:
                    names[tensor] = ng.get_new_name('Variable')
            else:
                names[tensor] = ng.get_new_name('Const')
    return names


def _build_graph(graph, names):
    graph_def = tf_pb.GraphDef()

    for tensor in graph.tensors:
        if tensor.producer is None:
            node_def = graph_def.node.add()
            node_def.name = names[tensor]
            if tensor.data is None:
                node_def.op = 'Placeholder'
                _build_shape(node_def.attr['shape'].shape, tensor.shape)
                node_def.attr['dtype'].type = _build_dtype(tensor.dtype)
            else:
                node_def.op = 'Const'
                _build_attribute(node_def.attr['value'], tensor)
                node_def.attr['dtype'].type = _build_dtype(tensor.dtype)

    for operation in graph.operations:
        node_def = graph_def.node.add()
        _build_node(node_def, operation, names)

    return graph_def


def read_tf_graph_from_protobuf(filename):
    graph_def = tf_pb.GraphDef()
    with open(filename, 'rb') as file:
        graph_def.ParseFromString(file.read())

    graph = TFGraph()
    # just a graph to contain the tensors that are in attributes
    # no need to return this
    attrib_graph = TFGraph()

    attributes_by_node_id = {}
    outputs_by_node_id = {}

    for node in graph_def.node:
        outputs = []
        attributes = _get_attributes(node.attr, attrib_graph)
        output_count = _OutputCount.get(node.op, 1)
        if isinstance(output_count, str):
            output_count = attributes[output_count]
        assert isinstance(output_count, int)
        if output_count >= 1:
            output = TFTensor(graph, utils.anystr_to_str(node.name))
            outputs.append(output)
            for i in range(1, output_count):
                tensor_name = utils.anystr_to_str(node.name) + ':' + str(i)
                output = TFTensor(graph, tensor_name)
                outputs.append(output)
        outputs_by_node_id[id(node)] = outputs
        attributes_by_node_id[id(node)] = attributes

    tensor_by_name = {tensor.name: tensor
                      for outputs in six.itervalues(outputs_by_node_id)
                      for tensor in outputs}
    placeholders = []
    for node in graph_def.node:
        attributes = attributes_by_node_id[id(node)]
        outputs = outputs_by_node_id[id(node)]

        if node.op == 'Placeholder':
            assert len(outputs) == 1
            tensor = outputs[0]
            tensor.shape = attributes['shape'] if 'shape' in attributes else None
            tensor.dtype = attributes['dtype'] if 'dtype' in attributes else None
            placeholders.append(tensor)
        elif node.op == 'Const':
            assert len(outputs) == 1
            tensor = outputs[0]
            value = attributes['value']
            if isinstance(value, TFTensor):
                tensor.shape = value.shape
                tensor.dtype = value.dtype
                tensor.data = value.data
            else:
                tensor.data = value
        else:
            inputs = tuple([tensor_by_name[name] for name in node.input])
            TFOperation(graph,
                        name=utils.anystr_to_str(node.op),
                        inputs=inputs,
                        outputs=outputs,
                        attribs=attributes)

    for tensor in graph.tensors:
        if tensor.name is not None and ':' not in tensor.name:
            tensor.name += ':0'

    graph.inputs = OrderedDict([(tensor.name.split(':')[0], tensor) for tensor in placeholders])
    graph_outputs = []
    for op in graph.operations:
        if all(len(output.consumers) == 0 for output in op.outputs):
            for output in op.outputs:
                graph_outputs.append(output)

    graph.outputs = OrderedDict([('output' + str(i) if len(graph_outputs) > 1 else 'output', tensor)
                                 for i, tensor in enumerate(graph_outputs)])

    return graph


def write_tf_graph_to_protobuf(graph, filename, write_pbtxt=False, convert_from_tf_py=False):
    if convert_from_tf_py:
        tf_py_to_tf_pb.convert(graph)

    names = _generate_names(graph)
    graph_def = _build_graph(graph, names)

    with open(filename, 'wb') as file:
        file.write(graph_def.SerializeToString())

    if write_pbtxt:
        import tensorflow as tf
        if filename.endswith('.pb'):
            filename = filename[:-3]
        tf.train.write_graph(graph_def, "", filename + ".pbtxt")

    return _get_rename_info(graph, names)


def _get_rename_info(graph, names):
    tensor_infos = []
    for tensor in graph.tensors:
        if tensor.name and names[tensor]:
            name = names[tensor]
            name = name if ':' in name else name + ':0'
            tensor_infos.append(conversion_info.TensorInfo(source_name=tensor.name,
                                                           target_name=name,
                                                           target_shape=list(tensor.shape),
                                                           target_dtype=tensor.dtype,
                                                           is_input=tensor in graph.inputs,
                                                           is_output=tensor in graph.outputs,
                                                           is_variable=tensor.is_variable))
    return conversion_info.ConversionInfo(tensor_infos)


class Reader(object):

    # input shape: (dtype, shape) tuple or name->(dtype, shape) dict
    def __init__(self, convert_to_tf_py=False, input_shape=None):
        self._convert_to_tf_py = convert_to_tf_py
        self._input_shape = input_shape

    def __call__(self, filename):
        g = read_tf_graph_from_protobuf(filename)
        if self._convert_to_tf_py:

            if self._input_shape is None:
                source_dtypes = None
                source_shapes = None
            elif isinstance(self._input_shape, dict):
                source_dtypes = {k: v[0] for k, v in six.iteritems(self._input_shape)}
                source_shapes = {k: v[1] for k, v in six.iteritems(self._input_shape)}
            else:
                source_dtypes = {i.name: self._input_shape[0] for i in g.inputs}
                source_shapes = {i.name: self._input_shape[1] for i in g.inputs}

            tf_pb_to_tf_py.evaluate_and_convert(g, source_dtypes=source_dtypes, source_shapes=source_shapes)
        return g


class Writer(object):

    def __init__(self, convert_from_tf_py=False):
        self._convert_from_tf_py = convert_from_tf_py

    def __call__(self, graph, filename):
        return write_tf_graph_to_protobuf(graph, filename, convert_from_tf_py=self._convert_from_tf_py)
