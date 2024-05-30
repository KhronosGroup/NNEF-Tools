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

from .protobuf import *
import numpy as np
import six


_DtypeFromNumpy = {
    None: 'DT_INVALID',
    np.float16: 'DT_HALF',
    np.float32: 'DT_FLOAT',
    np.float64: 'DT_DOUBLE',
    np.int8: 'DT_INT8',
    np.int16: 'DT_INT16',
    np.int32: 'DT_INT32',
    np.int64: 'DT_INT64',
    np.uint8: 'DT_UINT8',
    np.uint16: 'DT_UINT16',
    np.uint32: 'DT_UINT32',
    np.uint64: 'DT_UINT64',
    np.bool_: 'DT_BOOL',
    np.str_: 'DT_STRING',
    np.complex64: 'DT_COMPLEX64',
    np.complex128: 'DT_COMPLEX128',
    np.dtype([('resource', np.int32)]): 'DT_RESOURCE',
}

_NumpyDtypes = {
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float32, np.float64,
    np.complex64, np.complex128,
    np.bool_, np.str_,
    np.dtype([('resource', np.int32)]),
}


def _build_shape(shape_proto, shape):
    shape_proto.unknown_rank = (shape is None)
    if shape is not None:
        for item in shape:
            dim = shape_proto.dim.add()
            dim.size = item if item is not None else -1


def _build_dtype(dtype):
    return DataType.Value(_DtypeFromNumpy[dtype])


def _build_tensor(tensor_proto, data):
    if data.dtype is not None:
        tensor_proto.dtype = _build_dtype(data.dtype.type)

    if data.shape is not None:
        _build_shape(tensor_proto.tensor_shape, data.shape)

    tensor_proto.tensor_content = data.reshape([-1]).view(np.uint8).tobytes()
    return tensor_proto


def _build_attribute(attr_proto, value):
    if value is None:
        return attr_proto

    if type(value) in _NumpyDtypes:
        value = np.array(value)

    if isinstance(value, bool):  # must be before int
        attr_proto.b = value
    elif isinstance(value, int):
        attr_proto.i = value
    elif isinstance(value, float):
        attr_proto.f = value
    elif isinstance(value, str):
        attr_proto.s = value.encode()
    elif isinstance(value, (type, np.dtype)):
        attr_proto.type = _build_dtype(value)
    elif isinstance(value, tuple):
        _build_shape(attr_proto.shape, value)
    elif isinstance(value, np.ndarray):
        _build_tensor(attr_proto.tensor, value)
    elif isinstance(value, list):
        if len(value) == 0:
            attr_proto.list.i.extend([])     # to signal that the 'list' is the active in the oneof field
        else:
            first = value[0]
            if isinstance(first, int):
                attr_proto.list.i.extend(value)
            elif isinstance(first, float):
                attr_proto.list.f.extend(value)
            elif isinstance(first, bool):
                attr_proto.list.b.extend(value)
            elif isinstance(first, str):
                attr_proto.list.s.extend([item.encode() for item in value])
            elif isinstance(first, (type, np.dtype)):
                attr_proto.list.type.extend([_build_dtype(item) for item in value])
            elif isinstance(first, tuple):
                for item in value:
                    _build_shape(attr_proto.list.shape.add(), item)
            elif isinstance(first, np.ndarray):
                for item in value:
                    _build_tensor(attr_proto.list.tensor.add(), item)
            else:
                raise TypeError('unable to build attribute proto message from type: ' + str(type(first)))
    else:
        raise TypeError('unable to build attribute proto message from type: ' + str(type(value)))

    return attr_proto


def _build_output_shapes(attr_proto, output_shapes):
    for item in output_shapes:
        _build_shape(attr_proto.list.shape.add(), item)


def _tensor_name(tensor):
    name = tensor.producer.name
    idx = tensor.producer.outputs.index(tensor)
    return name + ':' + str(idx) if idx > 0 else name


def _custom_attribs(operation):
    attribs = {'_$' + key + '$_': value for key, value in six.iteritems(operation.attribs)}
    attribs['token'] = operation.type
    attribs['Tin'] = [tensor.dtype for tensor in operation.inputs]
    attribs['Tout'] = [tensor.dtype for tensor in operation.outputs]


def _build_node(node_def, operation):
    node_def.op = operation.type if not operation.custom else 'PyFunc'
    node_def.name = operation.name
    node_def.input.extend([_tensor_name(tensor) for tensor in operation.inputs])

    attribs = operation.attribs if not operation.custom else _custom_attribs(operation)

    output_shapes = attribs.get('_output_shapes')
    if output_shapes is not None:
        _build_output_shapes(node_def.attr['_output_shapes'], output_shapes)
        del attribs['_output_shapes']
    else:
        _build_output_shapes(node_def.attr['_output_shapes'],
                             [tensor.shape for tensor in operation.outputs])

    for name, value in attribs.items():
        _build_attribute(node_def.attr[name], value)

    return node_def


def build_graphdef(graph):
    graph_def = GraphDef()
    for operation in graph.operations:
        node_def = graph_def.node.add()
        _build_node(node_def, operation)
    return graph_def


def write_graphdef(graph, filename):
    graph_def = build_graphdef(graph)

    with open(filename, 'wb') as file:
        file.write(graph_def.SerializeToString())


class Writer(object):

    def __call__(self, graph, filename):
        return write_graphdef(graph, filename)
