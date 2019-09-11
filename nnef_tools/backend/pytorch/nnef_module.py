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
                 tensor_hooks=None,  # type: typing.Optional[typing.List[_TensorHookType]]
                 deallocate_nnef_tensors=False,  # type: bool
                 ):
        # type: (...)->None
        """
            nnef_graph might be modified by this class if deallocate_nnef_tensors is True or training and write_nnef is used
        """
        super(NNEFModule, self).__init__()
        self._nnef_graph = nnef_graph
        self._deallocate_nnef_tensors = deallocate_nnef_tensors

        for nnef_tensor in self._nnef_graph.tensors:
            if nnef_tensor.is_constant:
                np_array = nnef_tensor.get_numpy_array()
                self.register_buffer(self._safe_name(nnef_tensor.name),
                                     to_torch_tensor(np_array, nnef_dtype=nnef_tensor.dtype))
            elif nnef_tensor.is_variable:
                np_array = nnef_tensor.get_numpy_array()
                self.register_parameter(
                    self._safe_name(nnef_tensor.name),
                    torch.nn.Parameter(to_torch_tensor(np_array, nnef_dtype=nnef_tensor.dtype)))
                if self._deallocate_nnef_tensors:
                    nnef_tensor.data = np.array([])

        self._operations = {}
        self._operations.update(operations.operations)
        if custom_operations:
            self._operations.update(custom_operations)
        self._batch_normalization_momentum = batch_normalization_momentum
        self._tensor_hooks = tensor_hooks if tensor_hooks else []

    def forward(self, *inputs):
        activation_tensors = _RefCountedDict({
            t.name: (sum(input is t
                         for consumer in t.consumers
                         for input in consumer.inputs)
                     + (1 if t in self._nnef_graph.outputs else 0))
            for t in self._nnef_graph.tensors})

        def get_tensor(name):
            if hasattr(self, self._safe_name(name)):
                return getattr(self, self._safe_name(name))
            else:
                return activation_tensors[name]

        def has_tensor(name):
            return hasattr(self, self._safe_name(name)) or name in activation_tensors

        assert len(inputs) == len(self._nnef_graph.inputs)
        for torch_tensor, nnef_tensor in zip(inputs, self._nnef_graph.inputs):
            activation_tensors.ready(nnef_tensor.name, torch_tensor)
            utils.call_each(self._tensor_hooks, nnef_tensor, torch_tensor)

        if self._tensor_hooks:
            for nnef_tensor in self._nnef_graph.tensors:
                if nnef_tensor.is_constant or nnef_tensor.is_variable:
                    utils.call_each(self._tensor_hooks, nnef_tensor, get_tensor(nnef_tensor.name))

        for op in self._nnef_graph.operations:
            if op.name not in self._operations:
                raise utils.NNEFToolsException("Unsupported operation: {}".format(op.name))
            fun = self._operations[op.name]
            assert all(has_tensor(t.name) for t in op.inputs)
            if isinstance(op.inputs, tuple):
                inputs = tuple(get_tensor(t.name) for t in op.inputs)
            else:
                inputs = ([get_tensor(t.name) for t in op.inputs],)
            outputs = fun(*inputs, **utils.dict_union(op.attribs, self._get_extra_attributes(op.name)))
            if not isinstance(outputs, (list, tuple)):
                outputs = (outputs,)
            for t, output in zip(op.outputs, outputs):
                activation_tensors.ready(t.name, output)
                utils.call_each(self._tensor_hooks, t, output)

            for t in op.inputs:
                if not t.is_constant and not t.is_variable:
                    activation_tensors.release(t.name)

        outputs = [get_tensor(t.name) for t in self._nnef_graph.outputs]
        for t in self._nnef_graph.outputs:
            activation_tensors.release(t.name)

        assert not activation_tensors, "Memory leak in PyTorch NNEF Backend"
        return tuple(outputs)

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
                nnef_tensor.data = to_numpy_array(getattr(self, self._safe_name(nnef_tensor.name)),
                                                  nnef_tensor.dtype)
        writer = nnef_io.Writer()
        writer(self._nnef_graph, nnef_path)

        if self._deallocate_nnef_tensors:
            for nnef_tensor in self._nnef_graph.tensors:
                if nnef_tensor.is_variable:
                    nnef_tensor.data = np.array([])

    def _get_extra_attributes(self, op_name):
        if op_name == "batch_normalization":
            return {'is_training': self.training,
                    'momentum': self._batch_normalization_momentum}
        return {}

    @staticmethod
    def _safe_name(name):
        return '_nnef_tensor_' + name

    @staticmethod
    def _unsafe_name(name):
        assert name.startswith('_nnef_tensor_')
        return name[len('_nnef_tensor_'):]


class _RefCountedDict(object):
    """
        This is a special container to retain objects that are needed by a predefined number of consumers
        - The number of consumers must be given in the constructor
        - The ready method must be used to store the objects when they become ready
        - The release method must be used to signal that a consumer of the object has already run
    """

    def __init__(self, ref_count_by_name):
        self._ref_count = ref_count_by_name
        self._value = {}

    def ready(self, name, value):
        assert name in self._ref_count
        if self._ref_count[name] > 0:
            self._value[name] = value

    def release(self, name):
        assert name in self._ref_count and self._ref_count[name] > 0
        self._ref_count[name] -= 1
        if self._ref_count[name] == 0:
            del self._value[name]

    def __getitem__(self, name):
        assert name in self._value
        return self._value[name]

    def __contains__(self, name):
        return name in self._value

    def __bool__(self):
        return bool(self._value)

    def __nonzero__(self):  # python2 compatibility
        return self.__bool__()


def to_torch_tensor(np_array, nnef_dtype):
    return torch.tensor(np_array, dtype={'logical': torch.uint8,
                                         'scalar': torch.float32,
                                         'integer': torch.int64}[nnef_dtype])


def to_numpy_array(torch_tensor, nnef_dtype):
    return torch_tensor.detach().cpu().numpy().astype({'logical': np.bool,
                                                       'scalar': np.float32,
                                                       'integer': np.int32}[nnef_dtype])
