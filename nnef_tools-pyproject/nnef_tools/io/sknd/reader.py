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


def _build_tensor(graph, sknd_tensor):
    return Tensor(graph,
                  name=sknd_tensor.name,
                  shape=sknd_tensor.shape,
                  dtype=sknd.DtypeToNumpy[sknd_tensor.dtype],
                  data=sknd_tensor.value,
                  quant=sknd_tensor.quant,
                  variable=sknd_tensor.variable)


def _build_tensor_pack(graph, sknd_pack, tensor_map):
    return TensorPack(graph,
                      name=sknd_pack.name,
                      shape=sknd_pack.shape,
                      dtype=sknd.DtypeToNumpy[sknd_pack.dtype],
                      size=sknd_pack.size,
                      items=[remap_tensor(item, tensor_map) for item in sknd_pack])


def _build_operation(graph, sknd_operation, tensor_map, graph_map):
    attribs = dict(sknd_operation.attribs)
    dtypes = {k: sknd.DtypeToNumpy[t] for k, t in sknd_operation.dtypes.items()}

    for key, value in attribs.items():
        remap_tensors_in_expr(value, tensor_map)

    inputs = tuple(remap_tensor(tensor, tensor_map) for tensor in sknd_operation.inputs)
    outputs = tuple(remap_tensor(tensor, tensor_map) for tensor in sknd_operation.outputs)
    internals = list(remap_tensor(tensor, tensor_map) for tensor in sknd_operation.internals)
    subgraphs = list(graph_map[graph.name] for graph in sknd_operation.subgraphs)

    return Operation(graph,
                     type=sknd_operation.name,
                     dtypes=dtypes,
                     attribs=attribs,
                     inputs=inputs,
                     outputs=outputs,
                     internals=internals,
                     subgraphs=subgraphs)


def _build_graph(sknd_graph, graph_map, tensor_map):
    graph = graph_map[sknd_graph.name]

    for tensor in sknd_graph.tensors:
        tensor_map[tensor.name] = _build_tensor(graph, tensor)

    for pack in sknd_graph.packs:
        tensor_map[pack.name] = _build_tensor_pack(graph, pack, tensor_map)

    for tensor in graph.tensors:
        remap_tensors_in_expr(tensor.shape, tensor_map)

    for pack in graph.packs:
        remap_tensors_in_expr(pack.shape, tensor_map)
        remap_tensors_in_expr(pack.size, tensor_map)

    graph.inputs = tuple(remap_tensor(input, tensor_map) for input in sknd_graph.inputs)
    graph.outputs = tuple(remap_tensor(output, tensor_map) for output in sknd_graph.outputs)

    for operation in sknd_graph.operations:
        _build_operation(graph, operation, tensor_map, graph_map)

    return graph


def _build_model(sknd_model):
    model = Model(name=sknd_model.name)
    tensor_map = {}
    graph_map = {}
    for graph in sknd_model.graphs:
        parent = graph_map[graph.parent.name] if graph.parent else None
        graph_map[graph.name] = Graph(model, parent=parent, name=graph.name)
    for graph in sknd_model.graphs:
        _build_graph(graph, graph_map, tensor_map)
    return model


class Reader(object):

    def __init__(self, inline=None, atomic=None):
        self._inline = inline
        self._atomic = atomic

    def __call__(self, filename, attribs=None, init_data=True):
        sknd_model = sknd.read_model(filename, attribs=attribs, init_data=init_data)
        if sknd_model is None:
            raise IOError('could not read model')
        if self._inline:
            sknd.inline_compounds(sknd_model, filter=self._inline)
        if self._atomic:
            sknd.atomize_compounds(sknd_model, filter=self._atomic)
        return _build_model(sknd_model)
