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

import torch
import nnef
import skriptnd as sknd
import os

from .nnef_module import NNEFModule
from .sknd_module import SKNDModule
from .. import compute_statistics


class NNEFInterpreter:

    def __init__(self, model, device=None, decomposed=None, custom_operators=None):
        if isinstance(model, nnef.Graph):
            self._nnef_graph = model
        else:
            self._nnef_graph = nnef.parse_file(os.path.join(model, 'graph.nnef'), lowered=decomposed)
        self._init_input_shapes(self._nnef_graph)

        self._nnef_module = NNEFModule(model=model, custom_operators=custom_operators, decomposed=decomposed)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._nnef_module.to(device)
        self._nnef_module.eval()
        self._device = device

    def __call__(self, inputs, output_names=None, statistics=None):
        outputs = {}

        def callback(name, tensor):
            if output_names is not None and name in output_names:
                outputs[name] = tensor.detach().cpu().numpy()
            if statistics is not None:
                statistics[name] = compute_statistics(tensor)

        if output_names is not None:
            assert all(name in self._nnef_graph.tensors for name in output_names), \
                "could not find tensor(s) named {}".format({name for name in output_names
                                                            if name not in self._nnef_graph.tensors})

        if output_names is not None or statistics is not None:
            self._nnef_module.activation_callback = callback

        torch_inputs = [torch.tensor(input).to(self._device) for input in inputs]
        with torch.no_grad():  # Without this, gradients are calculated even in eval mode
            torch_outputs = self._nnef_module.forward(*torch_inputs)

        self._nnef_module.activation_callback = None

        if output_names is None:
            outputs = {name: torch_tensor.detach().cpu().numpy()
                       for name, torch_tensor in zip(self._nnef_graph.outputs, torch_outputs)}

        return outputs

    def input_details(self):
        return [self._nnef_graph.tensors[name] for name in self._nnef_graph.inputs]

    def output_details(self):
        return [self._nnef_graph.tensors[name] for name in self._nnef_graph.outputs]

    def tensor_details(self):
        return self._nnef_graph.tensors.values()

    @staticmethod
    def _init_input_shapes(graph):
        from nnef.shapes import _set_shape
        for op in graph.operations:
            if op.name == 'external':
                _set_shape(graph, op.outputs['output'], op.attribs['shape'])


class SKNDInterpreter:

    def __init__(self, model, device=None, decomposed=None):
        if isinstance(model, sknd.Model):
            self._sknd_model = model
        else:
            self._sknd_model = sknd.parse_file(os.path.join(model, 'main.sknd'))

        self._sknd_module = SKNDModule(model=model, decomposed=decomposed)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._sknd_module.to(device)
        self._sknd_module.eval()
        self._device = device

    def __call__(self, inputs, output_names=None, statistics=None):
        outputs = {}

        def callback(name, tensor):
            if output_names is not None and name in output_names:
                outputs[name] = tensor.detach().cpu().numpy()
            if statistics is not None:
                statistics[name] = compute_statistics(tensor)

        sknd_graph = self._sknd_model.graphs[0]
        if output_names is not None:
            assert all(name in sknd_graph.tensors for name in output_names), \
                "could not find tensor(s) named {}".format({name for name in output_names
                                                            if name not in sknd_graph.tensors})

        if output_names is not None or statistics is not None:
            self._sknd_module.activation_callback = callback

        torch_inputs = [torch.tensor(input).to(self._device) for input in inputs]
        with torch.no_grad():  # Without this, gradients are calculated even in eval mode
            torch_outputs = self._sknd_module.forward(*torch_inputs)

        self._sknd_module.activation_callback = None

        if output_names is None:
            outputs = {name: torch_tensor.detach().cpu().numpy()
                       for name, torch_tensor in zip(sknd_graph.outputs, torch_outputs)}

        return outputs

    def input_details(self):
        return [tensor for tensor in self._sknd_model.graphs[0].inputs]

    def output_details(self):
        return [tensor for tensor in self._sknd_model.graphs[0].outputs]

    def tensor_details(self):
        return [tensor for graph in self._sknd_model.graphs for tensor in graph.tensors]
