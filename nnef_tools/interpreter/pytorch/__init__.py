# Copyright (c) 2020 The Khronos Group Inc.
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
import os

from .nnef_module import NNEFModule
from .. import Statistics


class Interpreter:

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

    def __call__(self, inputs, output_names=None, collect_statistics=False):
        outputs = {}
        statistics = {} if collect_statistics else None

        def callback(name, tensor):
            if output_names is not None and name in output_names:
                outputs[name] = tensor.detach().cpu().numpy()
            if collect_statistics:
                statistics[name] = self._compute_statistics(tensor)

        if output_names is not None:
            assert all(name in self._nnef_graph.tensors for name in output_names), \
                "could not find tensor(s) named {}".format({name for name in output_names
                                                            if name not in self._nnef_graph.tensors})

        if output_names is not None or collect_statistics:
            self._nnef_module.activation_callback = callback

        torch_inputs = [torch.tensor(input).to(self._device) for input in inputs]
        with torch.no_grad():  # Without this, gradients are calculated even in eval mode
            torch_outputs = self._nnef_module.forward(*torch_inputs)

        self._nnef_module.activation_callback = None

        if output_names is None:
            outputs = {name: torch_tensor.detach().cpu().numpy()
                       for name, torch_tensor in zip(self._nnef_graph.outputs, torch_outputs)}

        return (outputs, statistics) if collect_statistics else outputs

    def input_details(self):
        return [self._nnef_graph.tensors[name] for name in self._nnef_graph.inputs]

    def output_details(self):
        return [self._nnef_graph.tensors[name] for name in self._nnef_graph.outputs]

    def tensor_details(self):
        return self._nnef_graph.tensors.values()

    @staticmethod
    def _compute_statistics(torch_tensor):
        num = torch_tensor.numel()
        if num == 0:
            return Statistics(num=0, min=0.0, max=0.0, sum=0.0, ssum=0.0)
        else:
            return Statistics(
                num=num,
                min=float(torch.min(torch_tensor)),
                max=float(torch.max(torch_tensor)),
                sum=float(torch.sum(torch_tensor)),
                ssum=float(torch.sum(torch_tensor * torch_tensor)),
            )

    @staticmethod
    def _init_input_shapes(graph):
        from nnef.shapes import _set_shape
        for op in graph.operations:
            if op.name == 'external':
                _set_shape(graph, op.outputs['output'], op.attribs['shape'])
