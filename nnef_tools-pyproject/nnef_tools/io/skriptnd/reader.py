import skriptnd as nd
from ...model import *


def _remap(tensor, tensor_map):
    def fetch(tensor):
        return tensor_map[tensor.name] if tensor is not None else None

    return [fetch(t) for t in tensor] if isinstance(tensor, (list, nd.TensorPack)) else fetch(tensor)


def _build_tensor(graph, ts_tensor):
    return Tensor(graph,
                  name=ts_tensor.name,
                  shape=ts_tensor.shape,
                  dtype=nd.DtypeToNumpy[ts_tensor.dtype],
                  data=ts_tensor.value,
                  quant=ts_tensor.quant)


def _build_operation(graph, ts_operation, tensor_map):
    attribs = dict(ts_operation.attribs)
    if ts_operation.dtypes:
        attribs['dtypes'] = {k: nd.DtypeToNumpy[t] for k, t in ts_operation.dtypes.items()}

    return Operation(graph,
                     type=ts_operation.name,
                     attribs=attribs,
                     inputs=tuple(_remap(input, tensor_map) for input in ts_operation.inputs),
                     outputs=tuple(_remap(output, tensor_map) for output in ts_operation.outputs))


def _build_graph(model, ts_graph):
    graph = Graph(model, name=ts_graph.name)

    tensor_map = {}
    for tensor in ts_graph.tensors:
        tensor_map[tensor.name] = _build_tensor(graph, tensor)

    graph.inputs = tuple(_remap(input, tensor_map) for input in ts_graph.inputs)
    graph.outputs = tuple(_remap(output, tensor_map) for output in ts_graph.outputs)

    for operation in ts_graph.operations:
        _build_operation(graph, operation, tensor_map)

    return graph


def _build_model(ts_model):
    model = Model(name=ts_model.name)
    for graph in ts_model.graphs:
        _build_graph(model, graph)
    return model


class Reader(object):

    def __init__(self, atomics=None):
        self._atomics = atomics

    def __call__(self, filename, attribs=None):
        ts_model = nd.read_model(filename, atomic=self._atomics, attribs=attribs)
        if ts_model is None:
            raise IOError('could not read model')
        return _build_model(ts_model)
