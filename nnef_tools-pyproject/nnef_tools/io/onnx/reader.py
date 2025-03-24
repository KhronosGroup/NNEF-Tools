from __future__ import division, print_function, absolute_import

from ...model import *
from ...model.utils import recursive_itemize
from onnx.shape_inference import infer_shapes
import numpy as np
import onnx
import sys


_is_little_endian_system = (sys.byteorder == 'little')


_DtypeToNumpy = {
    'UNDEFINED': None,
    'FLOAT': np.float32,
    'UINT8': np.uint8,
    'INT8': np.int8,
    'UINT16': np.uint16,
    'INT16': np.int16,
    'INT32': np.int32,
    'INT64': np.int64,
    'STRING': np.str_,
    'BOOL': np.bool_,
    'FLOAT16': np.float16,
    'DOUBLE': np.float64,
    'UINT32': np.uint32,
    'UINT64': np.uint64,
    'COMPLEX64': np.complex64,
    'COMPLEX128': np.complex128,
}


# noinspection PyUnresolvedReferences
def as_str(s):
    if sys.version_info[0] >= 3:
        return s.decode('utf-8') if isinstance(s, bytes) else s
    else:
        return s.encode('utf-8') if isinstance(s, unicode) else s


def _get_shape(tensor_shape_proto):
    # TODO dim_param
    return ([int(dim.dim_value) if dim.HasField('dim_value') else None for dim in tensor_shape_proto.dim]
            if tensor_shape_proto is not None else None)


def _get_dtype(dtype_int):
    return _DtypeToNumpy[onnx.TensorProto.DataType.Name(dtype_int)]


def _get_field(proto, name, default=None):
    return getattr(proto, name) if proto.HasField(name) else default


def _get_value_info(value_info_proto):
    name = as_str(value_info_proto.name)
    shape = _get_shape(_get_field(value_info_proto.type.tensor_type, 'shape'))
    dtype = _get_dtype(value_info_proto.type.tensor_type.elem_type)
    return name, shape, dtype


def _get_tensor(tensor_proto):
    assert not tensor_proto.HasField('segment'), 'TensorProto.segment is not supported'

    name = as_str(tensor_proto.name)
    shape = tuple(int(dim) for dim in tensor_proto.dims)
    dtype = _get_dtype(tensor_proto.data_type)
    assert dtype is not None

    if tensor_proto.HasField('raw_data'):
        assert dtype != np.str_

        data = np.frombuffer(tensor_proto.raw_data, dtype)
        if not _is_little_endian_system:
            data = data.byteswap()
    else:
        if dtype == np.float32:
            data = np.array(tensor_proto.float_data, dtype)
        elif dtype == np.float64:
            data = np.array(tensor_proto.double_data, dtype)
        elif dtype == np.int64:
            data = np.array(tensor_proto.int64_data, dtype)
        elif dtype == np.str_:
            data = np.array(as_str(tensor_proto.string_data))
        elif dtype == np.float16:
            data = np.array(tensor_proto.int32_data, np.uint16).view(np.float16)
        elif dtype == np.complex64:
            data = np.array(tensor_proto.float_data, np.float32)
            data = data[0::2] + data[1::2] * 1j
        elif dtype == np.complex128:
            data = np.array(tensor_proto.double_data, np.float64)
            data = data[0::2] + data[1::2] * 1j
        elif dtype in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.bool_]:
            data = np.array(tensor_proto.int32_data, dtype)
        elif dtype in [np.uint32, np.uint64]:
            data = np.array(tensor_proto.uint64_data, dtype)
        else:
            assert False

    data = data.reshape(shape)
    return name, shape, dtype, data


def _get_tensor_data(tensor_proto):
    name, shape, dtype, data = _get_tensor(tensor_proto)
    return data


def _get_node(node_proto, model, graphs_by_name, tensors_by_name):
    inputs = [as_str(input) for input in node_proto.input]
    outputs = [as_str(output) for output in node_proto.output]
    name = as_str(_get_field(node_proto, 'name'))
    domain = as_str(_get_field(node_proto, 'domain'))
    op_type = as_str(node_proto.op_type)
    attributes = {}
    for attribute in node_proto.attribute:
        key, value = _get_attribute(attribute, model, graphs_by_name, tensors_by_name)
        attributes[key] = value
    return inputs, outputs, name, domain, op_type, attributes


def _get_attribute(attribute_proto, model, graphs_by_name, tensors_by_name):
    assert not attribute_proto.HasField('ref_attr_name')

    name = as_str(attribute_proto.name)

    if attribute_proto.HasField('f'):
        value = float(attribute_proto.f)
    elif attribute_proto.HasField('i'):
        value = int(attribute_proto.i)
    elif attribute_proto.HasField('s'):
        value = as_str(attribute_proto.s)
    elif attribute_proto.HasField('t'):
        value = _get_tensor_data(attribute_proto.t)
    elif attribute_proto.HasField('g'):
        value = _get_graph(attribute_proto.g, model, graphs_by_name, tensors_by_name)
    elif attribute_proto.floats:
        value = [float(f) for f in attribute_proto.floats]
    elif attribute_proto.ints:
        value = [int(i) for i in attribute_proto.ints]
    elif attribute_proto.strings:
        value = [as_str(s) for s in attribute_proto.strings]
    elif attribute_proto.tensors:
        value = [_get_tensor_data(t) for t in attribute_proto.tensors]
    elif attribute_proto.graphs:
        value = [_get_graph(g, model, graphs_by_name, tensors_by_name) for g in attribute_proto.graphs]
    else:
        value = []

    return name, value


def _get_tensors(graph_proto, graph, tensors_by_name):
    for tensor_proto in graph_proto.initializer:
        name, shape, dtype, data = _get_tensor(tensor_proto)
        if name not in tensors_by_name:
            tensors_by_name[name] = Tensor(graph=graph, name=name, shape=shape, dtype=dtype, data=data)
    for value_info in graph_proto.input:
        name, shape, dtype = _get_value_info(value_info)
        if name not in tensors_by_name:
            tensors_by_name[name] = Tensor(graph=graph, name=name, shape=shape, dtype=dtype)
    for value_info in graph_proto.output:
        name, shape, dtype = _get_value_info(value_info)
        if name not in tensors_by_name:
            tensors_by_name[name] = Tensor(graph=graph, name=name, shape=shape, dtype=dtype)
    for value_info in graph_proto.value_info:
        name, shape, dtype = _get_value_info(value_info)
        if name not in tensors_by_name:
            tensors_by_name[name] = Tensor(graph=graph, name=name, shape=shape, dtype=dtype)

    for node in graph_proto.node:
        for name in node.output:
            name = as_str(name)
            if name not in tensors_by_name:
                tensors_by_name[name] = Tensor(graph=graph, name=name)

    return tensors_by_name


def _get_graph(graph_proto, model, graphs_by_name, tensors_by_name):
    name = as_str(_get_field(graph_proto, 'name'))
    graph = graphs_by_name.get(name)
    if graph is not None:
        return graph

    graph = Graph(model, name=name)

    if len(tensors_by_name) == 0:
        tensors_by_name[''] = Tensor(graph, name='', shape=(), dtype=np.void, data=np.zeros(shape=(), dtype=np.float32))

    _get_tensors(graph_proto, graph, tensors_by_name)

    initializer_names = {as_str(value_info.name) for value_info in graph_proto.initializer}
    input_names = [as_str(value_info.name) for value_info in graph_proto.input
                   if as_str(value_info.name) not in initializer_names]
    output_names = [as_str(value_info.name) for value_info in graph_proto.output]
    graph.inputs = tuple(tensors_by_name[name] for name in input_names)
    graph.outputs = tuple(tensors_by_name[name] for name in output_names)

    for annotation in graph_proto.quantization_annotation:
        tensor = tensors_by_name[annotation.tensor_name]
        tensor.quant = {item.key: tensors_by_name[item.value].data for item in annotation.quant_parameter_tensor_names}

    for node in graph_proto.node:
        inputs, outputs, name, domain, op_type, attributes = _get_node(node, model, graphs_by_name, tensors_by_name)

        Operation(
            graph=graph,
            type=op_type,
            name=name,
            inputs=tuple(tensors_by_name[input] for input in inputs),
            outputs=tuple(tensors_by_name[output] for output in outputs),
            attribs=attributes)

    graphs_by_name[graph.name] = graph
    return graph


def _get_tensor_shapes_as_dict(value_infos):
    shapes = dict()
    for value_info in value_infos:
        name, shape, dtype = _get_value_info(value_info)
        shapes[name] = shape
    return shapes


def _set_tensor_shapes_from_dict(value_infos, shapes):
    for value_info in value_infos:
        name, shape, dtype = _get_value_info(value_info)
        input_shape = shapes.get(name)
        if input_shape is not None:
            assert len(input_shape) == len(shape)
            assert all(s is None or z is None or z == s for s, z in zip(shape, input_shape))

            del value_info.type.tensor_type.shape.dim[:]
            for s in input_shape:
                dim = value_info.type.tensor_type.shape.dim.add()
                if s is not None:
                    dim.dim_value = s


# This is for working around a bug in ONNX IR, see https://github.com/onnx/onnx/issues/2903
def _add_value_info_for_constants(model: onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph: onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)

    return add_const_value_infos_to_graph(model.graph)


def _add_graph_inputs(model):
    for graph in model.graphs:
        locals = {tensor for op in graph.operations for tensor in recursive_itemize(op.outputs)}
        locals.update(tensor for tensor in graph.tensors if tensor.data is not None)
        locals.update(graph.inputs)

        inputs = {tensor: None for op in graph.operations for tensor in recursive_itemize(op.inputs)
                  if tensor not in locals}
        graph.inputs = graph.inputs + tuple(inputs.keys())


def _add_implicit_loop_and_branch_inputs(model):
    for graph in model.graphs:
        for op in graph.operations:
            if op.type == 'Loop':
                body = op.attribs['body']
                if len(body.inputs) > len(op.inputs):
                    op.attribs['_implicit_input_count_'] = len(body.inputs) - len(op.inputs)
                    op.inputs = op.inputs + body.inputs[len(op.inputs):]
                else:
                    op.attribs['_implicit_input_count_'] = 0
            if op.type == 'If':
                then_branch = op.attribs['then_branch']
                else_branch = op.attribs['else_branch']
                op.attribs['_implicit_input_count_'] = [len(then_branch.inputs), len(else_branch.inputs)]
                op.inputs = op.inputs + then_branch.inputs + else_branch.inputs


def from_onnx_model(onnx_model):
    model = Model(name=as_str(_get_field(onnx_model.graph, 'name')), version=onnx_model.opset_import[0].version)
    _get_graph(onnx_model.graph, model, {}, {})
    return model


def read_tensor(filename):
    with open(filename, 'rb') as file:
        return _get_tensor_data(onnx.load_tensor(file))


class Reader(object):

    def __init__(self, simplify=False, add_implicit_inputs=True, enforce_output_shapes=False):
        self._simplify = simplify
        self._add_implicit_inputs = add_implicit_inputs
        self._enforce_output_shapes = enforce_output_shapes

    def __call__(self, filename, input_shapes=None):
        model_proto = onnx.load_model(filename)
        _add_value_info_for_constants(model_proto)

        if self._simplify:
            from onnxsim import simplify
            model_proto, _ = simplify(model_proto, input_shapes=input_shapes, perform_optimization=False)
        if input_shapes:
            _set_tensor_shapes_from_dict(model_proto.graph.input, input_shapes)

        output_shapes = _get_tensor_shapes_as_dict(model_proto.graph.output) if self._enforce_output_shapes else None
        model_proto = infer_shapes(model_proto)
        if self._enforce_output_shapes:
            _set_tensor_shapes_from_dict(model_proto.graph.output, output_shapes)

        model = from_onnx_model(model_proto)

        if self._add_implicit_inputs:
            _add_graph_inputs(model)
            _add_implicit_loop_and_branch_inputs(model)

        return model
