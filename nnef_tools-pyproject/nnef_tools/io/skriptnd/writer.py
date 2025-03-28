import skriptnd as nd
from ...model import *
from ...utils.tgz import compress
import tempfile
import shutil
import six
import os


def _remap(tensor, tensor_map):
    def fetch(tensor):
        return tensor_map[tensor.name] if tensor is not None else None

    return [fetch(t) for t in tensor] if isinstance(tensor, (list, nd.TensorPack)) else fetch(tensor)


def _build_model(model):
    tensor_map = {tensor.name: _build_tensor(tensor) for graph in model.graphs for tensor in graph.tensors}
    graph_map = {graph.name: _build_graph(graph, tensor_map) for graph in model.graphs}

    ts_model = nd.Model(name=model.name, graphs=[graph_map[graph.name] for graph in model.graphs])

    for graph in ts_model.graphs:
        for op in graph.operations:
            for key, value in op.attribs.items():
                if isinstance(value, Graph):
                    op.attribs[key] = graph_map[value.name]
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, Graph):
                            value[i] = graph_map[item.name]

    return ts_model


def _build_tensor(tensor):
    return nd.Tensor(name=tensor.name or "~",
                     shape=tensor.shape,
                     max_shape=tensor.shape,
                     dtype=nd.DtypeFromNumpy[tensor.dtype] if tensor.dtype else None,
                     quant=tensor.quant,
                     value=tensor.data,
                     variable=tensor.is_variable)


def _build_graph(graph, tensor_map):
    nd_graph = nd.Graph(name=graph.name,
                        inputs=tuple(_remap(input, tensor_map) for input in graph.inputs),
                        outputs=tuple(_remap(output, tensor_map) for output in graph.outputs),
                        operations=[_build_operation(op, tensor_map) for op in graph.operations],
                        tensors=[tensor_map[tensor.name] for tensor in graph.tensors],
                        packs=[])

    for op in nd_graph.operations:
        for key, value in op.attribs.items():
            if isinstance(value, Tensor):
                op.attribs[key] = tensor_map[value.name]
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Tensor):
                        value[i] = tensor_map[item.name]

    return nd_graph


def _build_operation(operation, tensor_map):
    attribs = {k: list(v) if isinstance(v, list) else v for k, v in operation.attribs.items() if v is not None}
    dtypes = attribs.get('dtypes')
    if dtypes is not None:
        dtypes = {k: nd.DtypeFromNumpy[t] for k, t in dtypes.items()}
        del attribs['dtypes']
    else:
        dtypes = {}

    return nd.Operation(name=operation.type, dtypes=dtypes, attribs=attribs,
                        inputs=tuple(_remap(input, tensor_map) for input in operation.inputs),
                        outputs=tuple(_remap(output, tensor_map) for output in operation.outputs),
                        contractions=[],
                        asserts=[])


class Writer(object):

    def __init__(self, operators=None, imports=None, compression=None, inline_subgraphs=False):
        self._operators = operators
        self._imports = imports
        self._compression = compression
        self._inline_subgraphs = inline_subgraphs

    def __call__(self, model, path):
        folder = None
        try:
            if self._compression is not None:
                folder = tempfile.mkdtemp(prefix="tensorscript_")
            else:
                folder = path
                if not os.path.exists(folder):
                    os.makedirs(folder)

            operators = None
            if self._operators is not None:
                used_operators = {op.type for graph in model.graphs for op in graph.operations}
                operators = [text for name, text in six.iteritems(self._operators) if name in used_operators]

            nd_model = _build_model(model)
            nd.write_model(nd_model, folder, operators=operators, imports=self._imports,
                           inline_subgraphs=self._inline_subgraphs)
        finally:
            if self._compression is not None and folder:
                compress(folder, path + '.tgz', compression_level=self._compression)
                shutil.rmtree(folder)
