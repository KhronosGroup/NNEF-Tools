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

from collections import OrderedDict

import six

from nnef_tools.conversion.conversion_info import *

_SourceTensorT = typing.TypeVar('_SourceTensorT', bound='Tensor')
_SourceOperationT = typing.TypeVar('_SourceOperationT', bound='Operation')
_SourceGraphT = typing.TypeVar('_SourceGraphT', bound='Graph')
_TargetTensorT = typing.TypeVar('_TargetTensorT', bound='Tensor')
_TargetOperationT = typing.TypeVar('_TargetOperationT', bound='Operation')
_TargetGraphT = typing.TypeVar('_TargetGraphT', bound='Graph')

_OpConverter = typing.Callable[["Converter", _SourceOperationT, _TargetGraphT], None]
_TargetTensorListOrTuple = typing.Union[typing.List[_TargetTensorT], typing.Tuple[_TargetTensorT]]


class Converter(typing.Generic[_SourceTensorT, _SourceOperationT, _SourceGraphT,
                               _TargetTensorT, _TargetOperationT, _TargetGraphT]):

    def __init__(self, op_converter_by_name, default_op_converter=None):
        # type: (typing.Dict[str, _OpConverter], typing.Optional[_OpConverter])->None
        self._op_converter_by_name = op_converter_by_name
        self._default_op_converter = default_op_converter
        self._target_tensor_by_source = {}  # type: typing.Dict[_SourceTensorT, _TargetTensorT]

    @property
    def default_op_converter(self):
        # type: ()->_OpConverter
        return self._default_op_converter

    def converted_tensor(self, source_tensor):
        # type: (_SourceTensorT) -> _TargetTensorT
        return self._target_tensor_by_source[source_tensor]

    def converted_tensors(self, source_tensors):
        # type: (typing.Iterable[_SourceTensorT]) -> _TargetTensorListOrTuple
        if isinstance(source_tensors, tuple):
            return tuple(self._target_tensor_by_source[t] for t in source_tensors)
        return list(self._target_tensor_by_source[t] for t in source_tensors)

    def create_graph(self, source_graph):
        # type:(_SourceGraphT)->_TargetGraphT
        raise NotImplementedError()

    def convert_tensors(self, source_graph, target_graph):
        # type: (_SourceGraphT, _TargetGraphT)->None
        self._target_tensor_by_source = {}
        for source_tensor in source_graph.tensors:
            self._target_tensor_by_source[source_tensor] = self.convert_tensor(source_tensor, target_graph)

    def set_inputs_and_outputs(self, source_graph, target_graph):
        # type: (_SourceGraphT, _TargetGraphT)->None

        if source_graph.input_ids is not None:
            target_graph.inputs = OrderedDict((name, self._target_tensor_by_source[tensor])
                                              for name, tensor in zip(source_graph.input_ids, source_graph.inputs))
        else:
            target_graph.inputs = [self._target_tensor_by_source[tensor] for tensor in source_graph.inputs]

        if source_graph.output_ids is not None:
            target_graph.outputs = OrderedDict((name, self._target_tensor_by_source[tensor])
                                               for name, tensor in zip(source_graph.output_ids, source_graph.outputs))
        else:
            target_graph.outputs = [self._target_tensor_by_source[tensor] for tensor in source_graph.outputs]

    def convert_tensor(self, source_tensor, target_graph):
        # type: (_SourceTensorT, _TargetGraphT)->_TargetTensorT
        raise NotImplementedError()

    def convert_operations(self, source_graph, target_graph):
        # type: (_SourceGraphT, _TargetGraphT)->None
        for source_op in source_graph.operations:
            self.convert_operation(source_op, target_graph)

    def convert_operation(self, source_op, target_graph):
        # type: (_SourceOperationT, _TargetGraphT)->None

        assert hasattr(source_op, "name"), \
            "If the source operations do not have names, you have to override this method"

        op_converter = self._op_converter_by_name.get(source_op.name)
        if op_converter is not None:
            op_converter(self, source_op, target_graph)
        elif self._default_op_converter is not None:
            self._default_op_converter(self, source_op, target_graph)
        else:
            assert False, "No converter for operation '{}'".format(source_op.name)

    # noinspection PyMethodMayBeStatic
    def can_include_in_conversion_info(self, source_tensor, target_tensor):
        # type: (_SourceTensorT, _TargetTensorT)->bool

        return source_tensor.name and target_tensor.name

    def create_conversion_info(self, source_graph, target_graph):
        # type: (_SourceGraphT, _TargetGraphT)->ConversionInfo

        source_tensor_by_target = {target: source for source, target in six.iteritems(self._target_tensor_by_source)}

        return ConversionInfo([
            TensorInfo(source_name=source_tensor_by_target[t].name,
                       target_name=t.name,
                       target_shape=t.shape,
                       target_dtype=t.dtype,
                       is_input=t in target_graph.inputs,
                       is_output=t in target_graph.outputs,
                       is_variable=t.is_variable)
            for t in target_graph.tensors
            if t in source_tensor_by_target and self.can_include_in_conversion_info(source_tensor_by_target[t], t)
        ])

    def convert_graph(self, source_graph):
        # type: (_SourceGraphT)->_TargetGraphT
        target_graph = self.create_graph(source_graph)
        self.convert_tensors(source_graph, target_graph)
        self.set_inputs_and_outputs(source_graph, target_graph)
        self.convert_operations(source_graph, target_graph)
        return target_graph

    def __call__(self, source_graph):
        # type: (_SourceGraphT)->typing.Tuple[_TargetGraphT, ConversionInfo]
        target_graph = self.convert_graph(source_graph)
        conversion_info = self.create_conversion_info(source_graph, target_graph)
        return target_graph, conversion_info
