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


def _build_operation(graph, sknd_operation, tensor_map):
    attribs = dict(sknd_operation.attribs)
    dtypes = {k: sknd.DtypeToNumpy[t] for k, t in sknd_operation.dtypes.items()}

    for key, value in attribs.items():
        remap_tensors_in_expr(value, tensor_map)

    return Operation(graph,
                     type=sknd_operation.name,
                     dtypes=dtypes,
                     attribs=attribs,
                     inputs=tuple(remap_tensor(input, tensor_map) for input in sknd_operation.inputs),
                     outputs=tuple(remap_tensor(output, tensor_map) for output in sknd_operation.outputs))


def _build_graph(model, sknd_graph):
    graph = Graph(model, name=sknd_graph.name)

    tensor_map = {}
    for tensor in sknd_graph.tensors:
        tensor_map[tensor.name] = _build_tensor(graph, tensor)

    for pack in sknd_graph.packs:
        tensor_map[pack.name] = _build_tensor_pack(graph, pack, tensor_map)

    for name, tensor in tensor_map.items():
        remap_tensors_in_expr(tensor.shape, tensor_map)
        if isinstance(tensor, TensorPack):
            remap_tensors_in_expr(tensor.size, tensor_map)

    graph.inputs = tuple(remap_tensor(input, tensor_map) for input in sknd_graph.inputs)
    graph.outputs = tuple(remap_tensor(output, tensor_map) for output in sknd_graph.outputs)

    for operation in sknd_graph.operations:
        _build_operation(graph, operation, tensor_map)

    return graph


def _build_model(sknd_model):
    model = Model(name=sknd_model.name)
    for graph in sknd_model.graphs:
        _build_graph(model, graph)
    return model


class Reader(object):

    def __init__(self, atomic=None):
        self._atomic = atomic

    def __call__(self, filename, attribs=None, init_data=True):
        sknd_model = sknd.read_model(filename, atomic=self._atomic, attribs=attribs, init_data=init_data)
        if sknd_model is None:
            raise IOError('could not read model')
        return _build_model(sknd_model)
