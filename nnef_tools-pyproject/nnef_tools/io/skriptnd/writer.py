# Copyright (c) 2017-2025 The Khronos Group Inc.
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

import skriptnd as sknd
from ...model import *
from .utils import *
from ...utils.tgz import compress
import tempfile
import shutil
import six
import os


def _build_model(model):
    tensor_map = {}
    for graph in model.graphs:
        for tensor in graph.tensors:
            tensor_map[tensor.name] = _build_tensor(tensor)
        for pack in graph.packs:
            tensor_map[pack.name] = _build_tensor_pack(pack, tensor_map)

    for name, tensor in tensor_map.items():
        remap_tensors_in_expr(tensor.shape, tensor_map)
        if isinstance(tensor, TensorPack):
            remap_tensors_in_expr(tensor.size, tensor_map)

    graph_map = {graph.name: _build_graph(graph, tensor_map) for graph in model.graphs}

    sknd_model = sknd.Model(name=model.name, graphs=[graph_map[graph.name] for graph in model.graphs])

    for graph in sknd_model.graphs:
        for op in graph.operations:
            for key, value in op.attribs.items():
                if isinstance(value, Graph):
                    op.attribs[key] = graph_map[value.name]
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, Graph):
                            value[i] = graph_map[item.name]

    return sknd_model


def _build_tensor(tensor):
    return sknd.Tensor(name=tensor.name or "~",
                       shape=tensor.shape,
                       max_shape=tensor.shape,
                       dtype=sknd.DtypeFromNumpy[tensor.dtype] if tensor.dtype else None,
                       quant=tensor.quant,
                       value=tensor.data,
                       variable=tensor.is_variable)


def _build_tensor_pack(pack, tensor_map):
    return sknd.TensorPack(name=pack.name,
                           shape=pack.shape,
                           max_shape=pack.shape,
                           dtype=sknd.DtypeFromNumpy[pack.dtype] if pack.dtype else None,
                           size=pack.size,
                           items=[remap_tensor(item, tensor_map) for item in pack])


def _build_graph(graph, tensor_map):
    sknd_graph = sknd.Graph(name=graph.name,
                            inputs=tuple(remap_tensor(input, tensor_map) for input in graph.inputs),
                            outputs=tuple(remap_tensor(output, tensor_map) for output in graph.outputs),
                            operations=[_build_operation(op, tensor_map) for op in graph.operations],
                            tensors=[tensor_map[tensor.name] for tensor in graph.tensors],
                            packs=[])

    for op in sknd_graph.operations:
        for key, value in op.attribs.items():
            if isinstance(value, Tensor):
                op.attribs[key] = tensor_map[value.name]
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Tensor):
                        value[i] = tensor_map[item.name]

    return sknd_graph


def _build_operation(operation, tensor_map):
    attribs = {k: list(v) if isinstance(v, list) else v for k, v in operation.attribs.items() if v is not None}
    dtypes = {k: sknd.DtypeFromNumpy[t] for k, t in operation.dtypes.items()}

    for key, value in attribs.items():
        remap_tensors_in_expr(value, tensor_map)

    return sknd.Operation(name=operation.type, dtypes=dtypes, attribs=attribs,
                          inputs=tuple(remap_tensor(input, tensor_map) for input in operation.inputs),
                          outputs=tuple(remap_tensor(output, tensor_map) for output in operation.outputs),
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

            sknd_model = _build_model(model)
            sknd.write_model(sknd_model, folder, operators=operators, imports=self._imports,
                             inline_subgraphs=self._inline_subgraphs)
        finally:
            if self._compression is not None and folder:
                compress(folder, path + '.tgz', compression_level=self._compression)
                shutil.rmtree(folder)
