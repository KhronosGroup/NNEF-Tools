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

import sys

import numpy as np
import six
import typing

from nnef_tools.core import utils
from nnef_tools.io.onnx.onnx_graph import *
from nnef_tools.io.onnx.onnx_pb import onnx_pb2
from nnef_tools.io.onnx import onnx_shape_inference

OUTPUT_IR_VERSION = 3
OUTPUT_OPSET_VERSION = 9
LAST_SUPPORTED_IR_VERSION = 4
LAST_SUPPORTED_OPSET_VERSION = 9
PRODUCER_NAME = 'NNEF Tools'
PRODUCER_VERSION = 'v0.0.1'


class ParseException(Exception):
    pass


_is_little_endian_system = (sys.byteorder == 'little')

NumpyDTypeByONNXDType = {
    'UNDEFINED': None,
    'FLOAT': np.float32,
    'UINT8': np.uint8,
    'INT8': np.int8,
    'UINT16': np.uint16,
    'INT16': np.int16,
    'INT32': np.int32,
    'INT64': np.int64,
    'STRING': np.str,
    'BOOL': np.bool,
    'FLOAT16': np.float16,
    'DOUBLE': np.float64,
    'UINT32': np.uint32,
    'UINT64': np.uint64,
    'COMPLEX64': np.complex64,
    'COMPLEX128': np.complex128,
    'BFLOAT16': None,

}


def _fixint(i):
    return utils.anyint_to_int(i) if i is not None else None


def _fixstr(s):
    return utils.anystr_to_str(s) if s is not None else None


def _get_shape(tensor_shape_proto):
    # TODO dim_param
    return ([_fixint(dim.dim_value) if dim.HasField('dim_value') else -1 for dim in tensor_shape_proto.dim]
            if tensor_shape_proto is not None else None)


def _get_dtype(dtype_int):
    return _fixstr(onnx_pb2.TensorProto.DataType.Name(dtype_int))


def _get_field(proto, name, default=None):
    return getattr(proto, name) if proto.HasField(name) else default


def _get_value_info(value_info_proto):
    name = _fixstr(value_info_proto.name)
    shape = _get_shape(_get_field(value_info_proto.type.tensor_type, 'shape'))
    dtype = _get_dtype(value_info_proto.type.tensor_type.elem_type)
    doc_string = _fixstr(_get_field(value_info_proto, 'doc_string'))
    return name, shape, dtype, doc_string


def _get_tensor(tensor_proto):
    if tensor_proto.HasField('segment'):
        raise ParseException('TensorProto.segment is not yet supported.',
                             (_fixint(tensor_proto.segment.begin), _fixint(tensor_proto.segment.end)))
    name = _fixstr(tensor_proto.name)
    shape = [utils.anyint_to_int(dim) for dim in tensor_proto.dims]
    dtype = _get_dtype(tensor_proto.data_type)

    if not NumpyDTypeByONNXDType.get(dtype):
        raise ParseException("Unsupported '{}' dtype for '{}'".format(dtype, name))

    if tensor_proto.HasField('raw_data'):
        if dtype == 'STRING':
            raise ParseException('Unexpected raw_data when dtype is STRING')

        data = np.frombuffer(tensor_proto.raw_data, NumpyDTypeByONNXDType[dtype])
        if not _is_little_endian_system:
            data = data.byteswap()
    else:
        if dtype == 'FLOAT':
            data = np.array(tensor_proto.float_data, NumpyDTypeByONNXDType[dtype])
        elif dtype == 'DOUBLE':
            data = np.array(tensor_proto.double_data, NumpyDTypeByONNXDType[dtype])
        elif dtype == 'INT64':
            data = np.array(tensor_proto.int64_data, NumpyDTypeByONNXDType[dtype])
        elif dtype == 'STRING':
            data = np.array(_fixstr(tensor_proto.string_data))
        elif dtype == 'FLOAT16':
            data = np.array(tensor_proto.int32_data, np.uint16).view(np.float16)
        elif dtype == 'COMPLEX64':
            data = np.array(tensor_proto.float_data, np.float32)
            data = data[0::2] + data[1::2] * 1j
        elif dtype == 'COMPLEX128':
            data = np.array(tensor_proto.double_data, np.float64)
            data = data[0::2] + data[1::2] * 1j
        elif dtype in ['INT8', 'UINT8', 'INT16', 'UINT16', 'INT32', 'BOOL']:
            data = np.array(tensor_proto.int32_data, NumpyDTypeByONNXDType[dtype])
        elif dtype in ['UINT32', 'UINT64']:
            data = np.array(tensor_proto.uint64_data, NumpyDTypeByONNXDType[dtype])
        else:
            raise ParseException('Unsupported dtype: {}'.format(dtype))
    data = data.reshape(shape)
    doc_string = _fixstr(_get_field(tensor_proto, 'doc_string'))
    return name, shape, dtype, data, doc_string


def _get_node(node_proto):
    inputs = [_fixstr(input) for input in node_proto.input]
    outputs = [_fixstr(output) for output in node_proto.output]
    name = _fixstr(_get_field(node_proto, 'name'))
    domain = _fixstr(_get_field(node_proto, 'domain'))
    op_type = _fixstr(node_proto.op_type)
    attributes = {}
    for attribute in node_proto.attribute:
        name, value = _get_attribute(attribute)
        attributes[name] = value
    doc_string = _fixstr(_get_field(node_proto, 'doc_string'))
    return inputs, outputs, name, domain, op_type, attributes, doc_string


def _get_attribute(attribute_proto):
    if attribute_proto.HasField('ref_attr_name'):
        raise ParseException('Unexpected ref_attr_name in main graph')

    name = _fixstr(attribute_proto.name)

    if attribute_proto.HasField('f'):
        value = float(attribute_proto.f)
    elif attribute_proto.HasField('i'):
        value = utils.anyint_to_int(attribute_proto.i)
    elif attribute_proto.HasField('s'):
        value = utils.anystr_to_str(attribute_proto.s)
    elif attribute_proto.HasField('t'):
        value = _get_tensor(attribute_proto.t)
        # raise ParseException("Attribute '{}' with type TENSOR in unsupported".format(name))
    elif attribute_proto.HasField('g'):
        value = _get_graph(attribute_proto.g)
        # raise ParseException("Attribute '{}' with type GRAPH in unsupported".format(name))
    elif attribute_proto.floats:
        value = [float(f) for f in attribute_proto.floats]
    elif attribute_proto.ints:
        value = [utils.anyint_to_int(i) for i in attribute_proto.ints]
    elif attribute_proto.strings:
        value = [utils.anystr_to_str(s) for s in attribute_proto.strings]
    elif attribute_proto.tensors:
        # raise ParseException("Attribute '{}' with type TENSOR LIST in unsupported".format(name))
        value = [_get_tensor(t) for t in attribute_proto.tensors]
    elif attribute_proto.graphs:
        # raise ParseException("Attribute '{}' with type GRAPH LIST in unsupported".format(name))
        value = [_get_graph(g) for g in attribute_proto.graphs]
    else:
        value = []

    return name, value


def _get_graph(graph_proto):
    graph = ONNXGraph(name=_fixstr(_get_field(graph_proto, 'name')))

    if graph.name and not (graph.name[0].isalpha() or graph.name[0] == '_'):
        graph.name = 'graph_' + graph.name

    tensors_by_name = {}
    for node in graph_proto.node:
        for tensor_name in node.output:
            tensor_name = _fixstr(tensor_name)
            if tensor_name not in tensors_by_name:
                tensors_by_name[tensor_name] = ONNXTensor(graph=graph, name=tensor_name)
    for value_info in graph_proto.input:
        tensor_name = _fixstr(value_info.name)
        if tensor_name not in tensors_by_name:
            tensors_by_name[tensor_name] = ONNXTensor(graph=graph, name=tensor_name)
    for value_info in graph_proto.output:
        tensor_name = _fixstr(value_info.name)
        if tensor_name not in tensors_by_name:
            tensors_by_name[tensor_name] = ONNXTensor(graph=graph, name=tensor_name)
    for value_info in graph_proto.value_info:
        tensor_name = _fixstr(value_info.name)
        if tensor_name not in tensors_by_name:
            tensors_by_name[tensor_name] = ONNXTensor(graph=graph, name=tensor_name)
    for tensor_proto in graph_proto.initializer:
        tensor_name = _fixstr(tensor_proto.name)
        if tensor_name not in tensors_by_name:
            tensors_by_name[tensor_name] = ONNXTensor(graph=graph, name=tensor_name)

    const_or_var_names = {_fixstr(model_proto.name) for model_proto in graph_proto.initializer}
    input_names = [_fixstr(value_info.name)
                   for value_info in graph_proto.input
                   if _fixstr(value_info.name) not in const_or_var_names]
    output_names = [_fixstr(value_info.name) for value_info in graph_proto.output]
    graph.inputs = [tensors_by_name[name] for name in input_names]
    graph.outputs = [tensors_by_name[name] for name in output_names]

    for value_info in graph_proto.input:
        name, shape, dtype, doc_string = _get_value_info(value_info)
        tensor = tensors_by_name[name]
        tensor.shape, tensor.dtype = shape, dtype
    for value_info in graph_proto.output:
        name, shape, dtype, doc_string = _get_value_info(value_info)
        tensor = tensors_by_name[name]
        tensor.shape, tensor.dtype = shape, dtype
    for value_info in graph_proto.value_info:
        name, shape, dtype, doc_string = _get_value_info(value_info)
        tensor = tensors_by_name[name]
        tensor.shape, tensor.dtype = shape, dtype
    for tensor_proto in graph_proto.initializer:
        name, shape, dtype, data, doc_string = _get_tensor(tensor_proto)
        tensor = tensors_by_name[name]
        tensor.shape, tensor.dtype, tensor.data = shape, dtype, data

    for node in graph_proto.node:
        if _fixstr(node.op_type) == 'Constant':
            inputs, outputs, name, domain, op_type, attributes, doc_string = _get_node(node)
            if len(outputs) != 1:
                raise ParseException('Constant must have one output, we have: {}'.format(len(outputs)))
            tensor = tensors_by_name[outputs[0]]
            _name, tensor.shape, tensor.dtype, tensor.data, _doc_string = attributes['value']
            if not tensor.shape:
                tensor.data = tensor.data.flatten().tolist()
                if utils.is_anyint(tensor.data[0]):
                    tensor.data[0] = utils.anyint_to_int(tensor.data[0])
        else:
            inputs, outputs, name, domain, op_type, attributes, doc_string = _get_node(node)
            if op_type == 'ConstantOfShape':
                if 'value' in attributes:
                    _tensor_name, _tensor_shape, tensor_dtype, tensor_data, _tensor_doc_string = attributes['value']
                    attributes['dtype'] = tensor_dtype
                    attributes['value'] = (utils.anyint_to_int(tensor_data.item())
                                           if 'INT' in tensor_dtype
                                           else tensor_data.item())
                else:
                    attributes['dtype'] = 'FLOAT'
                    attributes['value'] = 0.0
            ONNXOperation(
                graph=graph,
                name=op_type,
                inputs=tuple([tensors_by_name[name] if name else ONNXTensor.create_null(graph) for name in inputs]),
                outputs=tuple([tensors_by_name[name] if name else ONNXTensor.create_null(graph) for name in outputs]),
                attribs=attributes)
    return graph


def read_onnx_from_protobuf(filename):
    # type: (str)->ONNXGraph
    model_proto = onnx_pb2.ModelProto()

    with open(filename, 'rb') as f:
        model_proto.ParseFromString(f.read())

    if not model_proto.HasField('ir_version'):
        print('Warning: ModelProto has no ir_version!')
    elif model_proto.ir_version > LAST_SUPPORTED_IR_VERSION:
        print('Warning: ModelProto has newer ir_version than what we support. ({} > {})'.format(
            model_proto.ir_version, LAST_SUPPORTED_IR_VERSION))

    if len(model_proto.opset_import) == 0:
        print('Warning: ModelProto has no opset import!')
    else:
        for opset in model_proto.opset_import:
            if opset.domain != '':
                print('Warning: ModelProto has an unsupported opset domain: {}'.format(opset.domain))
            elif opset.version > LAST_SUPPORTED_OPSET_VERSION:
                print('Warning: ModelProto has newer opset than what we support. ({} > {})'.format(
                    opset.version, LAST_SUPPORTED_OPSET_VERSION))

    return _get_graph(model_proto.graph)


def write_onnx_to_protobuf(graph, filename):
    # type: (ONNXGraph, str)->None

    graph.sort()

    model_proto = build_model(graph)
    with open(filename, 'wb') as file:
        file.write(model_proto.SerializeToString())


def build_model(graph):
    # type: (ONNXGraph)->onnx_pb2.ModelProto

    graph.generate_missing_names()

    model_proto = onnx_pb2.ModelProto()
    model_proto.ir_version = OUTPUT_IR_VERSION

    opset = model_proto.opset_import.add()
    opset.domain = ''
    opset.version = OUTPUT_OPSET_VERSION

    model_proto.producer_name = PRODUCER_NAME
    model_proto.producer_version = PRODUCER_VERSION

    if graph.domain is not None:
        model_proto.domain = graph.domain
    if graph.version is not None:
        model_proto.model_version = graph.version

    build_graph(graph, model_proto.graph)

    return model_proto


def build_graph(graph, graph_proto):
    # type: (ONNXGraph, onnx_pb2.GraphProto)->None

    for tensor in graph.tensors:
        if tensor.is_constant and not (len(tensor.consumers) == 1
                                       and tensor.consumers[0].name == 'ConstantOfShape'
                                       and tensor.consumers[0].inputs[1] is tensor):
            node_proto = graph_proto.node.add()
            build_constant_node(tensor, node_proto)

    for op in graph.operations:
        node_proto = graph_proto.node.add()
        build_node(op, node_proto)

    if graph.name is not None:
        graph_proto.name = graph.name

    for input in list(graph.inputs) + list(t for t in graph.tensors if t.is_variable and not t.is_null):
        value_info_proto = graph_proto.input.add()
        build_value_info(input, value_info_proto)

    for output in graph.outputs:
        value_info_proto = graph_proto.output.add()
        build_value_info(output, value_info_proto)

    for tensor in graph.tensors:
        if tensor.is_variable and not tensor.is_null:
            tensor_proto = graph_proto.initializer.add()
            build_tensor_proto(tensor, tensor_proto)


def build_value_info(tensor, value_info_proto):
    # type: (ONNXTensor, onnx_pb2.ValueInfoProto)->None

    value_info_proto.name = tensor.name
    value_info_proto.type.tensor_type.elem_type = build_dtype(tensor.dtype)

    if tensor.shape:
        for s in tensor.shape:
            dim = value_info_proto.type.tensor_type.shape.dim.add()
            dim.dim_value = s
    else:
        value_info_proto.type.tensor_type.shape.SetInParent()


def build_dtype(name):
    return onnx_pb2.TensorProto.DataType.Value(name)


def build_attribute_type(name):
    return onnx_pb2.AttributeProto.AttributeType.Value(name)


def build_tensor_proto(tensor, tensor_proto):
    # type: (ONNXTensor, onnx_pb2.TensorProto)->None
    if isinstance(tensor.data, np.ndarray):
        data = tensor.data
    else:
        if len(tensor.data) == 1:
            data = np.full(shape=tensor.shape, fill_value=tensor.data[0], dtype=NumpyDTypeByONNXDType[tensor.dtype])
        else:
            data = np.array(tensor.data, dtype=NumpyDTypeByONNXDType[tensor.dtype]).reshape(tensor.shape)

    # data = np.array([0], dtype=NumpyDTypeByONNXDType[tensor.dtype]) # to write without much data (for debug)

    for s in tensor.shape:
        tensor_proto.dims.append(s)

    tensor_proto.data_type = build_dtype(tensor.dtype)

    if tensor.dtype == 'STRING':
        tensor_proto.string_data = str(data)
    else:
        data = data.flatten().astype(NumpyDTypeByONNXDType[tensor.dtype])

        if tensor.dtype in ['COMPLEX64', 'COMPLEX128']:
            data = np.column_stack((np.real(data), np.imag(data))).flatten()

        if data.dtype.str[0] != "<":
            data = data.byteswap()
        tensor_proto.raw_data = data.tobytes()

    tensor_proto.name = tensor.name


def build_node(op, node_proto):
    # type: (ONNXOperation, onnx_pb2.NodeProto)->None

    if op.name == 'ConstantOfShape':
        inputs = op.inputs[:1]
        attribs = dict(op.attribs)
        attribs['value'] = op.inputs[1]
    else:
        inputs = op.inputs
        attribs = op.attribs

    for input in inputs:
        node_proto.input.append(input.name)
    for output in op.outputs:
        node_proto.output.append(output.name)

    node_proto.op_type = op.name

    for k, v in six.iteritems(attribs):
        attribute_proto = node_proto.attribute.add()
        build_attribute(k, v, attribute_proto)


def build_constant_node(tensor, node_proto):
    # type: (ONNXTensor, onnx_pb2.NodeProto)->None

    node_proto.output.append(tensor.name)

    node_proto.op_type = 'Constant'

    attribute_proto = node_proto.attribute.add()
    build_attribute('value', tensor, attribute_proto)


def build_attribute(key, value, attribute_proto):
    # type: (str, typing.Any, onnx_pb2.AttributeProto)->None

    attribute_proto.name = key

    if isinstance(value, ONNXTensor):
        attribute_proto.type = build_attribute_type('TENSOR')
        old_name = value.name
        value.name = "__attribute__" + old_name
        build_tensor_proto(value, attribute_proto.t)
        value.name = old_name
    elif isinstance(value, int):
        attribute_proto.type = build_attribute_type('INT')
        attribute_proto.i = value
    elif isinstance(value, float):
        attribute_proto.type = build_attribute_type('FLOAT')
        attribute_proto.f = value
    elif isinstance(value, str):
        attribute_proto.type = build_attribute_type('STRING')
        attribute_proto.s = value.encode('utf-8')
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            attribute_proto.type = build_attribute_type('INTS')  # TODO better
        else:
            if isinstance(value[0], int):
                attribute_proto.type = build_attribute_type('INTS')
                for v in value:
                    attribute_proto.ints.append(v)
            elif isinstance(value[0], float):
                attribute_proto.type = build_attribute_type('FLOATS')
                for v in value:
                    attribute_proto.floats.append(v)
            elif isinstance(value[0], str):
                attribute_proto.type = build_attribute_type("STRINGS")
                for v in value:
                    attribute_proto.strings.append(v.encode('utf-8'))
            else:
                assert False, \
                    "Unsupported attribute: {}: {} of type: List[{}]".format(key, value, type(value[0]).__name__)

    else:
        assert False, "Unsupported attribute: {}: {} of type: {}".format(key, value, type(value).__name__)


_DTypeShapeTuple = typing.Tuple[str, typing.List[int]]


class Reader(object):

    def __init__(self, propagate_shapes=False, input_shape=None):
        # type: (bool, typing.Union[typing.Dict[str, _DTypeShapeTuple], _DTypeShapeTuple, None])->None
        self._propagate_shapes = propagate_shapes
        self._input_shape = input_shape

    def __call__(self, filename):
        # type: (str)->ONNXGraph

        g = read_onnx_from_protobuf(filename)

        if self._propagate_shapes:
            if self._input_shape is None:
                source_dtypes = None
                source_shapes = None
            elif isinstance(self._input_shape, dict):
                source_dtypes = {k: v[0] for k, v in six.iteritems(self._input_shape)}
                source_shapes = {k: v[1] for k, v in six.iteritems(self._input_shape)}
            else:
                source_dtypes = {i.name: self._input_shape[0] for i in g.inputs}
                source_shapes = {i.name: self._input_shape[1] for i in g.inputs}

            onnx_shape_inference.propagate(g, source_shapes=source_shapes, source_dtypes=source_dtypes)

        return g


class Writer(object):

    def __call__(self, graph, filename):
        write_onnx_to_protobuf(graph, filename)
        return None
