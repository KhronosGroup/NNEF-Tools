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

from __future__ import division, print_function, absolute_import

import numpy as np
import torch
import typing

from nnef_tools.backend.pytorch import operations
from nnef_tools.core import utils
from nnef_tools.io.nnef import nnef_io
from nnef_tools.io.nnef.nnef_graph import *

_TensorHookType = typing.Callable[[NNEFTensor, torch.Tensor], None]


class NNEFModule(torch.nn.Module):
    """
    This a PyTorch NN Module which runs the given NNEF graph
    """

    def __init__(self,
                 nnef_graph,  # type: NNEFGraph
                 custom_operations=None,  # type: typing.Optional[typing.Dict[str, typing.Callable]]
                 batch_normalization_momentum=0.1,  # type: float
                 fix_batch_size=False,  # type: bool
                 permissive=False,  # type: bool
                 save_memory=False,  # type: bool
                 tensor_hooks=None,  # type: typing.Optional[typing.List[_TensorHookType]]
                 ):
        # type: (...)->None
        """
            nnef_graph might be modified by this class if save_memory is True or training and write_nnef is used
        """
        super(NNEFModule, self).__init__()
        self._nnef_graph = nnef_graph
        self._save_memory = save_memory

        for nnef_tensor in self._nnef_graph.tensors:
            if nnef_tensor.is_constant:
                np_array = nnef_tensor.get_numpy_array()
                self.register_buffer(self._safe_name(nnef_tensor.name),
                                     self.to_torch_tensor(np_array, nnef_dtype=nnef_tensor.dtype))
            elif nnef_tensor.is_variable:
                np_array = nnef_tensor.get_numpy_array()
                self.register_parameter(
                    self._safe_name(nnef_tensor.name),
                    torch.nn.Parameter(self.to_torch_tensor(np_array, nnef_dtype=nnef_tensor.dtype)))
                if self._save_memory:
                    nnef_tensor.data = np.array([])

        self._ref_count = {}  # tensor name -> int
        self._torch_tensors = {}  # tensor name -> tensor

        self._operations = {}
        self._operations.update(operations.operations)
        if custom_operations:
            self._operations.update(custom_operations)
        self._batch_normalization_momentum = batch_normalization_momentum
        self._fix_batch_size = fix_batch_size
        self._permissive = permissive
        self._tensor_hooks = tensor_hooks if tensor_hooks else []

    def forward(self, *inputs):
        operations.context.reset(is_training=self.training,
                                 batch_normalization_momentum=self._batch_normalization_momentum,
                                 fix_batch_size=self._fix_batch_size,
                                 permissive=self._permissive)
        try:
            self._ref_count = {t.name: (sum(input is t
                                            for consumer in t.consumers
                                            for input in consumer.inputs)
                                        + (1 if t in self._nnef_graph.outputs else 0))  # don't remove outputs
                               for t in self._nnef_graph.tensors}

            assert len(inputs) == len(self._nnef_graph.inputs)
            for torch_tensor, nnef_tensor in zip(inputs, self._nnef_graph.inputs):
                self._ready(nnef_tensor, torch_tensor)
                for hook in self._tensor_hooks:
                    hook(nnef_tensor, torch_tensor)

            if self._tensor_hooks:
                for tensor in self._nnef_graph.tensors:
                    if tensor.is_constant or tensor.is_variable:
                        for hook in self._tensor_hooks:
                            hook(tensor, self._get_torch_tensor(tensor))

            for op in self._nnef_graph.operations:
                if op.name not in self._operations:
                    raise utils.NNEFToolsException("Unsupported operation: {}".format(op.name))
                fun = self._operations[op.name]
                assert all(self._has_torch_tensor(t) for t in op.inputs)
                if isinstance(op.inputs, tuple):
                    inputs = tuple(self._get_torch_tensor(t) for t in op.inputs)
                else:
                    inputs = ([self._get_torch_tensor(t) for t in op.inputs],)
                outputs = fun(*inputs, **op.attribs)
                if not isinstance(outputs, (list, tuple)):
                    outputs = (outputs,)
                for t, output in zip(op.outputs, outputs):
                    self._ready(t, output)
                    for hook in self._tensor_hooks:
                        hook(t, output)
                for t in op.inputs:
                    if not t.is_constant and not t.is_variable:
                        self._unref(t)

            outputs = [self._get_torch_tensor(t) for t in self._nnef_graph.outputs]
            for t in self._nnef_graph.outputs:
                self._unref(t)

            assert not self._torch_tensors, "Memory leak in PyTorch NNEF Backend"
            return tuple(outputs)
        finally:
            operations.context.reset()

    def reset_parameters(self):
        biases = set()
        for op in self._nnef_graph.operations:
            if op.name in ('conv', 'separable_conv', 'deconv', 'separable_deconv'):
                biases.add(op.inputs[2].name)
            elif op.name in ('batch_normalization',):
                biases.add(op.inputs[3].name)

        for name, param in self.named_parameters():
            if self._unsafe_name(name) in biases:
                param.data.fill_(0.0)
            elif len(param.shape) <= 2:
                param.data.fill_(1.0)
            else:
                torch.nn.init.xavier_uniform_(param)

    def write_nnef(self, nnef_path):
        for nnef_tensor in self._nnef_graph.tensors:
            if nnef_tensor.is_variable:
                nnef_tensor.data = self.to_numpy_array(self._get_torch_tensor(nnef_tensor), nnef_tensor.dtype)
        writer = nnef_io.Writer()
        writer(self._nnef_graph, nnef_path)

        if self._save_memory:
            for nnef_tensor in self._nnef_graph.tensors:
                if nnef_tensor.is_variable:
                    nnef_tensor.data = np.array([])

    @staticmethod
    def to_torch_tensor(np_array, nnef_dtype):
        return torch.tensor(np_array, dtype={'logical': torch.uint8,
                                             'scalar': torch.float32,
                                             'integer': torch.int64}[nnef_dtype])

    @staticmethod
    def to_numpy_array(torch_tensor, nnef_dtype):
        return torch_tensor.detach().cpu().numpy().astype({'logical': np.bool,
                                                           'scalar': np.float32,
                                                           'integer': np.int32}[nnef_dtype])

    def _get_torch_tensor(self, nnef_tensor):
        if nnef_tensor.is_constant or nnef_tensor.is_variable:
            return getattr(self, self._safe_name(nnef_tensor.name))
        else:
            return self._torch_tensors[nnef_tensor.name]

    def _has_torch_tensor(self, nnef_tensor):
        if nnef_tensor.is_constant or nnef_tensor.is_variable:
            return hasattr(self, self._safe_name(nnef_tensor.name))
        else:
            return nnef_tensor.name in self._torch_tensors

    def _unref(self, nnef_tensor):
        self._ref_count[nnef_tensor.name] -= 1
        if self._ref_count[nnef_tensor.name] <= 0:
            del self._torch_tensors[nnef_tensor.name]

    def _ready(self, nnef_tensor, torch_tensor):
        if self._ref_count[nnef_tensor.name] > 0:
            self._torch_tensors[nnef_tensor.name] = torch_tensor

    @staticmethod
    def _safe_name(name):
        return '_nnef_tensor_' + name

    @staticmethod
    def _unsafe_name(name):
        assert name.startswith('_nnef_tensor_')
        return name[len('_nnef_tensor_'):]
