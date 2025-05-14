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

from __future__ import division, print_function, absolute_import

import os
import nnef
import shutil
import tempfile
from ...model import *
from ...utils import types, tgz


_DtypeToNumpy = {
    'scalar': np.float32,
    'integer': np.int64,
    'logical': np.bool_,
}


def _make_constant_tensor(graph, value):
    value = types.to_numpy(value)
    return Tensor(graph=graph, shape=(), dtype=value.dtype.type, data=value)


def _make_tensor(graph, nnef_tensor):
    dtype = nnef_tensor.data.dtype.type if isinstance(nnef_tensor.data, np.ndarray) else _DtypeToNumpy[nnef_tensor.dtype]
    return Tensor(graph=graph, name=nnef_tensor.name, shape=tuple(nnef_tensor.shape) if nnef_tensor.shape is not None else None,
                  dtype=dtype, data=nnef_tensor.data, quant=nnef_tensor.quantization)


def _remap_item(arg, graph, tensor_by_name, only_tensors):
    return tensor_by_name[str(arg)] if isinstance(arg, nnef.Identifier) or only_tensors else _make_constant_tensor(graph, arg)


def _remap(arg, graph, tensor_by_name, only_tensors=False):
    return [_remap_item(item, graph, tensor_by_name, only_tensors) for item in arg] \
        if isinstance(arg, list) else _remap_item(arg, graph, tensor_by_name, only_tensors)


def _build_model(nnef_graph):
    model = Model(name=nnef_graph.name)
    graph = Graph(model=model, name=nnef_graph.name)

    tensor_by_name = {str(item): _make_tensor(graph, nnef_graph.tensors[str(item)]) for item in nnef_graph.tensors}

    for nnef_op in nnef_graph.operations:
        inputs = tuple(_remap(item, graph, tensor_by_name, only_tensors=False) for item in six.itervalues(nnef_op.inputs))
        outputs = tuple(_remap(item, graph, tensor_by_name, only_tensors=True) for item in six.itervalues(nnef_op.outputs))

        attribs = dict(nnef_op.attribs)
        if nnef_op.dtype is not None:
            attribs['dtype'] = outputs[0].dtype if nnef_op.name == 'constant' or nnef_op.name == 'variable' else \
                _DtypeToNumpy[nnef_op.dtype]

        _substitute_empty_array(nnef_op.name, 'stride', attribs, inputs)
        _substitute_empty_array(nnef_op.name, 'dilation', attribs, inputs)

        custom = nnef_op.name not in nnef.StandardOperations

        Operation(graph=graph, type=nnef_op.name, attribs=attribs, inputs=inputs, outputs=outputs, custom=custom)

    graph.inputs = tuple(_remap(item, graph, tensor_by_name, only_tensors=True)for item in nnef_graph.inputs)
    graph.outputs = tuple(_remap(item, graph, tensor_by_name, only_tensors=True) for item in nnef_graph.outputs)

    return model


def _substitute_empty_array(op, key, attribs, inputs):
    value = attribs.get(key)
    if value is not None and len(value) == 0:
        rank = None
        if op == 'slice':
            rank = len(attribs['axes'])
        elif len(inputs) > 0 and inputs[0].rank is not None:
            rank = inputs[0].rank - 2 if op.endswith('conv') else inputs[0].rank

        if rank is not None:
            attribs[key] = [1] * rank


class Reader(object):

    def __init__(self, stdlib=None, decomposed=None, custom_shapes=None, infer_shapes=True, load_variables=True):
        self._stdlib = stdlib
        self._decomposed = decomposed
        self._custom_shapes = custom_shapes
        self._infer_shapes = infer_shapes
        self._load_variables = load_variables

    def __call__(self, path, input_shapes=None):
        filename, extension = os.path.splitext(path)
        compressed = extension in ['.tgz', '.gz'] and not os.path.isdir(path)

        folder = None
        try:
            if compressed:
                folder = tempfile.mkdtemp(prefix="nnef_")
                tgz.extract(path, folder)
                path = folder

            if not os.path.isdir(path):
                raise IOError("NNEF model must be a (compressed) folder, but an uncompressed file was provided")

            nnef_graph = nnef.load_graph(path, stdlib=self._stdlib, lowered=self._decomposed, load_variables=self._load_variables)
            if self._infer_shapes:
                nnef.infer_shapes(nnef_graph, external_shapes=input_shapes or {}, custom_shapes=self._custom_shapes or {})

            return _build_model(nnef_graph)
        finally:
            if folder is not None:
                shutil.rmtree(folder)
