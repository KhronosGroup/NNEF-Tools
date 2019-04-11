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
import os
from nnef.binary import read_tensor


def parse_file(graph_fn, quant_fn=None, stdlib=None, lowered=[]):
    return _nnef.parse_file(graph_fn, quantization=quant_fn, stdlib=stdlib, lowered=lowered)


def parse_string(graph_str, quant_str=None, stdlib=None, lowered=[]):
    return _nnef.parse_string(graph_str, quantization=quant_str, stdlib=stdlib, lowered=lowered)


def load_graph(path, stdlib=None, lowered=[]):
    if os.path.isfile(path):
        return parse_file(path, stdlib=stdlib, lowered=lowered)

    graph_fn = os.path.join(path, 'graph.nnef')
    quant_fn = os.path.join(path, 'graph.quant')

    graph = parse_file(graph_fn, quant_fn if os.path.isfile(quant_fn) else None, stdlib=stdlib, lowered=lowered)

    for operation in graph.operations:
        if operation.name == 'variable':
            variable_filename = os.path.join(path, operation.attribs['label'] + '.dat')
            tensor_name = operation.outputs['output']
            with open(variable_filename) as variable_file:
                data, compression = read_tensor(variable_file)

            data_shape = list(data.shape);
            shape = operation.attribs['shape']
            if data_shape != shape:
                raise _nnef.Error('shape {} in variable file does not match shape {} defined in network structure'
                                  .format(data_shape, shape))

            tensor = graph.tensors[tensor_name]
            graph.tensors[tensor_name] = _nnef.Tensor(tensor.name, tensor.dtype, data_shape, data, compression, tensor.quantization)

    return graph
