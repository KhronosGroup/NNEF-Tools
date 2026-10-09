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

import numpy as np

import skriptnd as sknd
import torch
import keyword
import inspect

from . import sknd_operators
from ...io import sknd as sknd_io
from ...io.sknd.reader import _build_model
from ...model import *
from ...model.utils import recursive_itemize
from collections.abc import Iterable


class SKNDModule(torch.nn.Module):

    """
    A torch.nn.Module that interprets the given NNEF model
    """

    _Atomics = {
        'layout.reshape': lambda op: True,
        'nn.softmax': lambda op: True,
        'nn.avg_pool': lambda op: True,
        'nn.local_response_norm': lambda op: len(op.attribs['axes']) == 1,
        'nn.batch_norm': lambda op: True,
        'nn.lstm': lambda op: True,
        'math.cumsum': lambda op: not op.attribs['exclusive'] and not op.attribs['reverse'],
    }

    def __init__(self,
                 model,  # type: str
                 decomposed=None,           # type: typing.Optional[typing.List[str]]
                 activation_callback=None,  # type: typing.Optional[typing.Callable[[str, torch.Tensor], None]]
                 training_attributes=None,  # type: typing.Optional[typing.Dict[str, typing.Dict[str, typing.Any]]]
                 ):
        # type: (...)->None
        """
            nnef_graph might be modified by this class if training and write_nnef is used
        """
        super(SKNDModule, self).__init__()

        def inline_filter(op):
            if decomposed and op.name in decomposed:
                return True
            func = self._Atomics.get(op.name)
            if not func:
                return True
            return not func(op)

        if isinstance(model, sknd.Model):
            sknd.inline_compounds(model, filter=inline_filter)
            self._sknd_model = _build_model(model)
        else:
            reader = sknd_io.Reader(inline=inline_filter)
            self._sknd_model = reader(model)

        for graph in self._sknd_model.graphs:
            for tensor in graph.tensors:
                if tensor.is_variable:
                    name = self._registered_name(tensor.name)
                    data = self._dequantize(tensor.data, tensor.quant) \
                        if tensor.quant else tensor.data
                    data = self.normalize_dtype(data)
                    self.register_parameter(name, torch.nn.Parameter(torch.tensor(data), requires_grad=data.dtype == np.float32))
                elif tensor.is_constant:
                    name = self._registered_name(tensor.name)
                    data = tensor.data if isinstance(tensor.data, np.ndarray) \
                        else np.array(tensor.data, dtype=tensor.dtype).reshape(tensor.shape) if isinstance(tensor.data, Iterable) \
                        else np.full(tensor.shape, tensor.data, dtype=tensor.dtype)
                    data = self.normalize_dtype(data)
                    self.register_buffer(name, torch.tensor(data))

        self._operators = sknd_operators.Operators
        self._activation_callback = activation_callback
        self._training_attributes = training_attributes or {}

    def forward(self, *inputs):
        graph = self._sknd_model.main
        assert len(inputs) == len(graph.inputs)
        activations = {nnef_tensor.name: torch_tensor for torch_tensor, nnef_tensor
                       in zip(inputs, graph.inputs)}

        def get_tensor(name):
            if hasattr(self, self._registered_name(name)):
                return getattr(self, self._registered_name(name))
            else:
                return activations[name]

        def get_tensors(query):
            if query is None:
                return None
            return [get_tensor(item.name) for item in query] if isinstance(query, list) else get_tensor(query.name)

        def has_tensor(name):
            return hasattr(self, self._registered_name(name)) or name in activations

        def has_tensors(query):
            if query is None:
                return True
            return all(has_tensor(item.name) for item in query) if isinstance(query, list) else has_tensor(query.name)

        for op in graph.operations:
            assert op.type in self._operators, "Unsupported operation: {}".format(op.type)
            func = self._operators[op.type]
            params = inspect.signature(func).parameters if inspect.isfunction(func) else {}

            assert all(has_tensors(input) for input in op.inputs),\
                "could not fetch input tensor(s) {} for operation {}"\
                    .format({[item.name for item in input] if isinstance(input, list) else input.name
                             for input in op.inputs}, op.type)

            dtype_attribs = {name: type for name, type in op.dtypes.items() if name in params}
            training_attribs = self._training_attributes.get(op.type, {})
            attribs = {**op.attribs, **dtype_attribs, **training_attribs}
            attribs = {self._escape_keyword(name): value for name, value in six.iteritems(attribs)}

            inputs = [get_tensors(input) for input in op.inputs]
            outputs = func(*inputs, **attribs)

            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            def itemize(query):
                if isinstance(query, (list, tuple)):
                    for item in query:
                        yield from itemize(item)
                else:
                    yield query

            for sknd_tensor, output in zip(itemize(op.outputs), itemize(outputs)):
                if sknd_tensor.quant and not sknd_tensor._is_variable:
                    output = self._fake_quantize(output, sknd_tensor.quant)

                activations[sknd_tensor.name] = output
                if self._activation_callback:
                    self._activation_callback(sknd_tensor.name, output)

            # optimization: remove activations that are not needed any more
            for sknd_tensor in recursive_itemize(op.inputs):
                if sknd_tensor is not None and sknd_tensor.name in activations and op is sknd_tensor.consumers[-1] and sknd_tensor not in graph.outputs:
                    del activations[sknd_tensor.name]

        return tuple(get_tensors(output) for output in graph.outputs)

    def save_sknd(self, path):
        for graph in self._sknd_model.graphs:
            for sknd_tensor in graph.tensors:
                if sknd_tensor.is_variable:
                    torch_tensor = getattr(self, self._registered_name(sknd_tensor.name))
                    sknd_tensor.data = torch_tensor.detach().cpu().numpy().astype(sknd_tensor.dtype)

        writer = sknd_io.Writer()
        writer(self._sknd_model, path)

    @property
    def activation_callback(self):
        return self._activation_callback

    @activation_callback.setter
    def activation_callback(self, callback):
        self._activation_callback = callback

    @staticmethod
    def _registered_name(name):
        return name.replace('.', '_')

    @staticmethod
    def _escape_keyword(name):
        return name if not keyword.iskeyword(name) else '_' + name + '_'

    @staticmethod
    def _dequantize(data, quant):
        op_name = quant['op-name']
        rank = len(data.shape)
        if op_name == 'quant.zero_point_linear_quantize':
            channel_axis = quant['channel_axis']
            zero_point = SKNDModule._as_array(quant['zero_point'], rank, channel_axis)
            scale = SKNDModule._as_array(quant['scale'], rank, channel_axis)
            return SKNDModule._dequantize_zero_point(data, zero_point, scale)
        elif op_name == 'quant.min_max_linear_quantize' or op_name == 'linear_quantize':
            channel_axis = quant['channel_axis']
            min = SKNDModule._as_array(quant['min'], rank, channel_axis)
            max = SKNDModule._as_array(quant['max'], rank, channel_axis)
            bits = quant['bits']
            signed = quant['signed']
            symmetric = quant['symmetric']
            return SKNDModule._dequantize_min_max(data, min, max, bits, signed, symmetric)
        else:
            raise ValueError("Quantization operation '{}' not implemented".format(op_name))

    @staticmethod
    def _dequantize_zero_point(data, zero_point, scale):
        return (data - zero_point) * scale

    @staticmethod
    def _dequantize_min_max(data, min, max, bits, signed, symmetric):
        if signed:
            data += 2 ** (bits - 1) - int(symmetric)
        r = 2 ** bits - 1 - int(signed and symmetric)
        return data * ((max - min) / r) + min

    def _fake_quantize(self, tensor, quant):
        op_type = quant['op-name']
        attribs = {key: value for key, value in six.iteritems(quant) if key != 'op-name'}

        assert op_type in self._operators, "Unsupported quantization operation: {}".format(op_type)
        func = self._operators[op_type]
        return func(tensor, **attribs)

    @staticmethod
    def _as_array(value, rank, axis):
        array = np.array(value)
        return np.reshape(array, newshape=(1,) * axis + array.shape + (1,) * (rank - axis - len(array.shape)))

    @staticmethod
    def normalize_dtype(data):
        dtype = SKNDModule._dtypeRemap.get(data.dtype.type)
        return data.astype(dtype) if dtype is not None else data

    _dtypeRemap = {
        np.float16: np.float32,
        np.float64: np.float32,
        np.int8: np.int64,
        np.uint8: np.int64,
        np.int16: np.int64,
        np.uint16: np.int64,
        np.int32: np.int64,
        np.uint32: np.int64,
        np.uint64: np.int64,
    }
