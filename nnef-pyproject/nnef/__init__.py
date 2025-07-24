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

import _nnef
from .parser import *
from .printer import *
from .binary import read_tensor, write_tensor
from .shapes import infer_shapes, _StandardShapeFuncs
import os


Identifier = _nnef.Identifier   # subclass of str
Error = _nnef.Error             # subclass of exception

Graph = _nnef.Graph             # namedtuple('Graph', ['name': str, 'tensors': typing.Dict[str, Tensor], 'operations': typing.List[Operation],
                                #                       'inputs': typing.List[str], 'outputs': typing.List['str']])
Tensor = _nnef.Tensor           # namedtuple('Tensor', ['name': str, 'dtype': str, 'shape': typing.List[int], 'data': numpy.ndarray,
                                #                       'quantization': Dict[str, object]])
Operation = _nnef.Operation     # namedtuple('Operation', ['name': str, 'attribs': OrderedDict[str, object], 'inputs': OrderedDict[str, object],
                                #                           'outputs': OrderedDict[str, object], 'dtype': str])


Tensor.__new__.__defaults__ = (None, None, None)
Operation.__new__.__defaults__ = (None,)


StandardOperations = set(_StandardShapeFuncs.keys())


def load_graph(path, stdlib=None, lowered=None, load_variables=True):
    if os.path.isfile(path):
        return parse_file(path, stdlib=stdlib, lowered=lowered)

    graph_fn = os.path.join(path, 'graph.nnef')
    quant_fn = os.path.join(path, 'graph.quant')

    graph = parse_file(graph_fn, quant_fn if os.path.isfile(quant_fn) else None, stdlib=stdlib, lowered=lowered)

    if load_variables:
        for operation in graph.operations:
            if operation.name == 'variable':
                variable_filename = operation.attribs['label'] + '.dat'
                if variable_filename.startswith('/'):
                    variable_filename = variable_filename[1:]
                variable_filename = os.path.join(path, variable_filename)
                tensor_name = operation.outputs['output']
                with open(variable_filename) as variable_file:
                    data = read_tensor(variable_file)

                data_shape = list(data.shape)
                shape = operation.attribs['shape']
                if data_shape != shape:
                    raise _nnef.Error('shape {} in variable file does not match shape {} defined in network structure'
                                      .format(data_shape, shape))

                tensor = graph.tensors[tensor_name]
                graph.tensors[tensor_name] = _nnef.Tensor(tensor.name, tensor.dtype, data_shape, data, tensor.quantization)

    return graph


def save_graph(graph, path, annotate_shapes=False):
    if os.path.exists(path):
        raise RuntimeError("folder already exists: '{}'".format(path))

    os.makedirs(path)

    text = format_graph(graph.name, graph.inputs, graph.outputs, graph.operations, graph.tensors, annotate_shapes=annotate_shapes)

    with open(os.path.join(path, 'graph.nnef'), mode='w') as file:
        file.write('version 1.0;\n\n')
        file.write(text)

    for operation in graph.operations:
        if operation.name == 'variable':
            variable_filename = operation.attribs['label'] + '.dat'
            if variable_filename.startswith('/'):
                variable_filename = variable_filename[1:]
            variable_filename = os.path.join(path, variable_filename)
            os.makedirs(os.path.split(variable_filename)[0], exist_ok=True)

            tensor_name = operation.outputs['output']
            tensor = graph.tensors[tensor_name]
            if tensor.data is not None:
                with open(variable_filename, 'wb') as variable_file:
                    write_tensor(variable_file, tensor.data, quantized=bool(tensor.quantization))


class Session:

    def __init__(self, path, stdlib=None, lowered=None):
        self._handle = _nnef.create_session(path, stdlib=stdlib, lowered=lowered)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _nnef.cleanup_session(self._handle)

    def __call__(self, *inputs):
        return _nnef.execute_session(self._handle, tuple(inputs))
