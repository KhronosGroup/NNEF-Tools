import skriptnd as nd
from ...model import *


def _remap(item, tensor_map):
    def fetch(item):
        return tensor_map[item.name] if item is not None else None

    return [fetch(x) for x in item] if isinstance(item, list) and not isinstance(item, nd.TensorPack) else fetch(item)


def _build_tensor(graph, ts_tensor):
    return Tensor(graph,
                  name=ts_tensor.name,
                  shape=ts_tensor.shape,
                  dtype=nd.DtypeToNumpy[ts_tensor.dtype],
                  data=ts_tensor.value,
                  quant=ts_tensor.quant,
                  variable=ts_tensor.variable)


def _build_tensor_pack(graph, ts_pack, tensor_map):
    return TensorPack(graph,
                      name=ts_pack.name,
                      shape=ts_pack.shape,
                      dtype=nd.DtypeToNumpy[ts_pack.dtype],
                      size=ts_pack.size,
                      items=[_remap(item, tensor_map) for item in ts_pack])


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

    for pack in ts_graph.packs:
        tensor_map[pack.name] = _build_tensor_pack(graph, pack, tensor_map)

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
