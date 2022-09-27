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

from ...model import *
import numpy as np
import six
import onnx


_DtypeFromNumpy = {
    None: 'UNDEFINED',
    np.float32: 'FLOAT',
    np.uint8: 'UINT8',
    np.int8: 'INT8',
    np.uint16: 'UINT16',
    np.int16: 'INT16',
    np.int32: 'INT32',
    np.int64: 'INT64',
    np.str_: 'STRING',
    np.bool_: 'BOOL',
    np.float16: 'FLOAT16',
    np.float64: 'DOUBLE',
    np.uint32: 'UINT32',
    np.uint64: 'UINT64',
    np.complex64: 'COMPLEX64',
    np.complex128: 'COMPLEX128',
}


def build_model(graph, ir_version, opset_version):
    # type: (Graph)->onnx.ModelProto

    model_proto = onnx.ModelProto()
    build_graph(graph, model_proto.graph)

    model_proto.ir_version = ir_version
    model_proto.opset_import.add()
    model_proto.opset_import[0].version = opset_version

    return model_proto


def build_graph(graph, graph_proto):
    # type: (Graph, onnx.GraphProto)->None

    for idx, op in enumerate(graph.operations):
        node_proto = graph_proto.node.add()
        build_node(op, node_proto, idx)

    if graph.name is not None:
        graph_proto.name = graph.name

    for input in list(graph.inputs) + list(t for t in graph.tensors if t.is_constant and t.name != ''):
        value_info_proto = graph_proto.input.add()
        build_value_info(input, value_info_proto)

    for output in graph.outputs:
        value_info_proto = graph_proto.output.add()
        build_value_info(output, value_info_proto)

    for tensor in graph.tensors:
        if tensor.is_constant and tensor.name != '':
            tensor_proto = graph_proto.initializer.add()
            build_tensor_proto(tensor, tensor_proto)

        if tensor.quant:
            build_quantization(tensor, graph_proto)


def build_value_info(tensor, value_info_proto):
    # type: (Tensor, onnx.ValueInfoProto)->None

    value_info_proto.name = tensor.name
    value_info_proto.type.tensor_type.elem_type = build_dtype(tensor.dtype)

    if tensor.shape:
        for s in tensor.shape:
            dim = value_info_proto.type.tensor_type.shape.dim.add()
            if s is not None:
                dim.dim_value = s
    else:
        value_info_proto.type.tensor_type.shape.SetInParent()


def build_dtype(dtype):
    dtype = dtype.type if isinstance(dtype, np.dtype) else dtype
    return onnx.TensorProto.DataType.Value(_DtypeFromNumpy[dtype])


def build_attribute_type(name):
    return onnx.AttributeProto.AttributeType.Value(name)


def build_tensor_data(data, tensor_proto):
    # type: (np.ndarray, onnx.TensorProto)->None
    for s in data.shape:
        tensor_proto.dims.append(s)

    tensor_proto.data_type = build_dtype(data.dtype)

    if data.dtype == np.str_:
        tensor_proto.string_data = str(data)
    else:
        data = data.flatten().astype(data.dtype)

        if data.dtype in [np.complex64, np.complex128]:
            data = np.column_stack((np.real(data), np.imag(data))).flatten()

        if data.dtype.str[0] != "<":
            data = data.byteswap()
        tensor_proto.raw_data = data.tobytes()


def build_tensor_proto(tensor, tensor_proto):
    # type: (Tensor, onnx.TensorProto)->None
    if isinstance(tensor.data, np.ndarray):
        data = tensor.data
    elif isinstance(tensor.data, (list, tuple)):
        data = np.array(tensor.data, dtype=tensor.dtype).reshape(tensor.shape)
    else:
        data = np.full(shape=tensor.shape, fill_value=tensor.data, dtype=tensor.dtype)

    build_tensor_data(data, tensor_proto)
    tensor_proto.name = tensor.name


def build_quantization(tensor, graph_proto):
    tensor_annotation = graph_proto.quantization_annotation.add()
    tensor_annotation.tensor_name = tensor.name

    for key, value in tensor.quant:
        value_tensor_name = tensor.name + '/' + key
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        tensor_proto = graph_proto.initializer.add()
        build_tensor_data(value, tensor_proto)
        tensor_proto.name = value_tensor_name

        item = tensor_annotation.quant_parameter_tensor_names.add()
        item.key = key
        item.value = value_tensor_name


def build_node(op, node_proto, idx):
    # type: (Operation, onnx.NodeProto)->None

    inputs = op.inputs
    attribs = op.attribs

    for input in inputs:
        node_proto.input.append(input.name)
    for output in op.outputs:
        node_proto.output.append(output.name)

    node_proto.op_type = op.type
    node_proto.name = op.name or (op.type + str(idx))

    for k, v in six.iteritems(attribs):
        attribute_proto = node_proto.attribute.add()
        build_attribute(k, v, attribute_proto)


def build_attribute(key, value, attribute_proto):
    # type: (str, typing.Any, onnx.AttributeProto)->None

    attribute_proto.name = key

    if isinstance(value, np.ndarray):
        attribute_proto.type = build_attribute_type('TENSOR')
        build_tensor_data(value, attribute_proto.t)
    elif isinstance(value, int):
        attribute_proto.type = build_attribute_type('INT')
        attribute_proto.i = value
    elif isinstance(value, float):
        attribute_proto.type = build_attribute_type('FLOAT')
        attribute_proto.f = value
    elif isinstance(value, str):
        attribute_proto.type = build_attribute_type('STRING')
        attribute_proto.s = value.encode('utf-8')
    elif isinstance(value, (type, np.dtype)):
        attribute_proto.type = build_attribute_type('INT')
        attribute_proto.i = build_dtype(value)
    elif isinstance(value, Graph):
        attribute_proto.type = build_attribute_type('GRAPH')
        build_graph(value, attribute_proto.g)
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
                attribute_proto.type = build_attribute_type('STRINGS')
                for v in value:
                    attribute_proto.strings.append(v.encode('utf-8'))
            elif isinstance(value[0], Graph):
                attribute_proto.type = build_attribute_type('GRAPHS')
                for v in value:
                    g = attribute_proto.graphs.add()
                    build_graph(v, g)
            else:
                assert False, \
                    "Unsupported attribute: {}: {} of type: List[{}]".format(key, value, type(value[0]).__name__)
    else:
        assert False, "Unsupported attribute: {}: {} of type: {}".format(key, value, type(value).__name__)


class Writer(object):

    def __init__(self, ir_version=6, opset_version=11):
        self._ir_version = ir_version
        self._opset_version = opset_version

    def __call__(self, graph, filename):
        model_proto = build_model(graph, self._ir_version, self._opset_version)
        onnx.checker.check_model(model_proto)
        with open(filename, 'wb') as file:
            file.write(model_proto.SerializeToString())
