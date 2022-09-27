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

from collections.abc import Sequence
from functools import reduce

import six
import typing
import numpy as np


# noinspection PyProtectedMember
class Tensor:

    def __init__(self,
                 graph,             # type: Graph,
                 name=None,         # type: typing.Optional[str]
                 shape=None,        # type: typing.Optional[typing.Tuple[int, ...]]
                 dtype=None,        # type: typing.Optional[type]
                 data=None,         # type: typing.Union[None, np.ndarray, typing.Any]
                 quant=None         # type: typing.Optional[typing.Dict[str, typing.Any]]
                 ):
        # type: (...)->None
        self._graph = graph
        self._producers = []
        self._consumers = []

        self.name = name            # type: typing.Optional[str]
        self.shape = shape          # type: typing.Optional[typing.Tuple[int, ...]]
        self.dtype = dtype          # type: typing.Optional[type]
        self.data = data            # type: typing.Union[None, np.ndarray, typing.Any]
        self.quant = quant or {}    # type: typing.Optional[typing.Dict[str, typing.Any]]

        assert isinstance(graph, Graph)
        graph._tensors.append(self)

    def copy_with(self, graph=None, name=None, dtype=None, shape=None, data=None, quant=None):
        return Tensor(graph=graph if graph is not None else self.graph,
                      name=name if name is not None else self.name,
                      dtype=dtype if dtype is not None else self.dtype,
                      shape=shape if shape is not None else self.shape,
                      data=data if data is not None else self.data,
                      quant=quant if quant is not None else self.quant)

    @property
    def graph(self):
        # type: ()->typing.Optional[Graph]
        return self._graph

    @property
    def has_producer(self):
        return len(self._producers) != 0

    @property
    def producers(self):
        # type: ()->typing.List[Operation]
        return self._producers

    @property
    def producer(self):
        # type: ()->typing.Optional[Operation]
        assert len(self._producers) <= 1
        return self._producers[0] if len(self._producers) == 1 else None

    @property
    def has_consumer(self):
        return len(self._consumers) != 0

    @property
    def consumers(self):
        # type: ()->typing.List[Operation]
        return self._consumers

    @property
    def consumer(self):
        # type: ()->typing.Optional[Operation]
        return self._consumers[0] if len(self._consumers) == 1 else None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert name is None or isinstance(name, str)
        self._name = name

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        assert shape is None or isinstance(shape, (list, tuple))
        assert shape is None or all(s is None or isinstance(s, int) for s in shape)
        self._shape = tuple(shape) if shape is not None else None

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        assert dtype is None or isinstance(dtype, type)
        self._dtype = dtype

    @property
    def rank(self):
        # type: ()->typing.Optional[int]
        return len(self.shape) if self.shape is not None else None

    @property
    def volume(self):
        # type: ()->typing.Optional[int]
        return reduce((lambda x, y: x * y), self.shape) if self.shape is not None and \
                                                           all(s is not None for s in self.shape) else None

    @property
    def is_constant(self):
        # type: ()->bool
        return self.data is not None

    def __repr__(self):
        return self.name if self.name is not None else _hex_id(self)

    def __str__(self):
        return '{name}: {dtype}[{shape}]'.format(
            name=self.name if self.name is not None else _hex_id(self),
            dtype=self.dtype.__name__,
            shape=', '.join(str(s) for s in self.shape) if self.shape else '...')


_TensorListOrTupleT = typing.Union[typing.List[Tensor], typing.Tuple[Tensor, ...]]


# noinspection PyProtectedMember
class Operation:

    def __init__(self,
                 graph,             # type: Graph
                 type=None,         # type: typing.Optional[str]
                 name=None,         # type: typing.Optional[str]
                 attribs=None,      # type: typing.Dict[str, typing.Any]
                 inputs=None,       # type: typing.Union[None, Tensor, _TensorListOrTuple]
                 outputs=None,      # type: typing.Union[None, Tensor, _TensorListOrTuple]
                 custom=False,      # type: bool
                 ):
        # type:(...)->None
        self._graph = graph
        self._inputs = tuple()
        self._outputs = tuple()

        assert name is None or isinstance(name, str)
        if attribs is not None:
            assert isinstance(attribs, dict)
            assert all(isinstance(key, str) for key in six.iterkeys(attribs))
            assert all(not isinstance(value, Tensor) for value in six.itervalues(attribs))

        self.type = type                # type: typing.Optional[str]
        self.name = name                # type: typing.Optional[str]
        self.attribs = attribs or {}    # type: typing.Dict[str, typing.Any]
        self.custom = custom            # type: bool

        assert isinstance(graph, Graph)
        graph._operations.append(self)

        if inputs is not None:
            self.inputs = inputs
        if outputs is not None:
            self.outputs = outputs

    def copy_with(self, graph=None, type=None, name=None, attribs=None, inputs=None, outputs=None, custom=None):
        return Operation(graph=graph if graph is not None else self.graph,
                         type=type if type is not None else self.type,
                         name=name if name is not None else self.name,
                         attribs=attribs if attribs is not None else self.attribs,
                         inputs=inputs if inputs is not None else self.inputs,
                         outputs=outputs if outputs is not None else self.outputs,
                         custom=custom if custom is not None else self.custom)

    @property
    def graph(self):
        # type: ()->typing.Optional[Graph]
        return self._graph

    @property
    def inputs(self):
        # type: ()->_TensorListOrTupleT
        return self._inputs

    @property
    def input(self):
        # type: ()->Tensor
        assert len(self._inputs) == 1
        return self._inputs[0]

    @inputs.setter
    def inputs(self, tensors):
        # type: (typing.Union[Tensor, _TensorListOrTupleT])->None
        if isinstance(tensors, Tensor):
            tensors = (tensors,)

        for tensor in self._inputs:
            assert self in tensor._consumers
        for tensor in self._inputs:
            if self in tensor._consumers:
                tensor._consumers.remove(self)

        self._inputs = _ListView(tensors) if isinstance(tensors, list) else tensors
        for tensor in tensors:
            assert isinstance(tensor, Tensor), "got {}".format(type(tensor))
            if self not in tensor._consumers:
                tensor._consumers.append(self)

    @property
    def outputs(self):
        # type: ()->_TensorListOrTupleT
        return self._outputs

    @property
    def output(self):
        # type: ()->Tensor
        assert len(self._outputs) == 1
        return self._outputs[0]

    @outputs.setter
    def outputs(self, tensors):
        # type: (typing.Union[Tensor, _TensorListOrTupleT])->None

        if isinstance(tensors, Tensor):
            tensors = (tensors,)

        for tensor in self._outputs:
            assert self in tensor._producers
            tensor._producers.remove(self)

        self._outputs = _ListView(tensors) if isinstance(tensors, list) else tensors
        for tensor in tensors:
            assert isinstance(tensor, Tensor), "got {}".format(type(tensor))
            assert self not in tensor._producers
            tensor._producers.append(self)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        assert type is None or isinstance(type, str), "got '{}'".format(type)
        self._type = type

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert name is None or isinstance(name, str), "got '{}'".format(name)
        self._name = name

    def __repr__(self):
        return self.type if self.type is not None else _hex_id(self)

    def __str__(self):
        return '{outputs} = {op}{{{attribs}}}({inputs})'.format(
            op=self.type if self.type is not None else _hex_id(self),
            inputs=', '.join(repr(tensor) for tensor in self._inputs),
            outputs=', '.join(str(tensor) for tensor in self._outputs),
            attribs=', '.join('{}={}'.format(key, value) for key, value in self.attribs.items()))


# noinspection PyProtectedMember
class Graph:

    def __init__(self, name=None):
        # type:(typing.Optional[str])->None
        self._operations = []
        self._tensors = []
        self._inputs = []
        self._outputs = []
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert name is None or isinstance(name, str)
        self._name = name

    @property
    def operations(self):
        # type: ()->typing.Sequence[Operation]
        return _ListView(self._operations)

    @property
    def tensors(self):
        # type: ()->typing.Sequence[Tensor]
        return _ListView(self._tensors)

    @property
    def inputs(self):
        # type: ()->typing.Sequence[Tensor]
        return _ListView(self._inputs)

    @inputs.setter
    def inputs(self, tensors):
        # type: (_TensorListOrTupleT)->None
        assert isinstance(tensors, (list, tuple))

        self._inputs = tensors

        for tensor in self._inputs:
            assert isinstance(tensor, Tensor)

    @property
    def outputs(self):
        # type: ()->typing.Sequence[Tensor]
        return _ListView(self._outputs)

    @outputs.setter
    def outputs(self, tensors):
        # type: (_TensorListOrTupleT)->None
        assert isinstance(tensors, (list, tuple))

        self._outputs = tensors

        for tensor in self._outputs:
            assert isinstance(tensor, Tensor)

    def remove_tensor(self, tensor):
        # type: (Tensor)->None
        assert len(tensor.producers) == 0
        assert len(tensor.consumers) == 0
        assert tensor not in self._inputs
        assert tensor not in self._outputs
        self._tensors.remove(tensor)
        tensor._graph = None

    def remove_tensors(self, tensors):
        # type: (typing.Iterable[Tensor])->None
        for tensor in tensors:
            assert len(tensor.producers) == 0
            assert len(tensor.consumers) == 0
            assert tensor not in self._inputs
            assert tensor not in self._outputs
        self._tensors = [tensor for tensor in self._tensors if tensor not in tensors]
        for tensor in tensors:
            tensor._graph = None

    def remove_operation(self, operation, unlink=False):
        # type: (Operation, bool)->None
        if unlink:
            operation.inputs = []
            operation.outputs = []
        else:
            assert len(operation.inputs) == 0
            assert len(operation.outputs) == 0
        self._operations.remove(operation)
        operation._graph = None

    def remove_operations(self, operations, unlink=False):
        # type: (typing.Iterable[Operation], bool)->None
        operations = operations if isinstance(operations, set) else set(operations)
        for operation in operations:
            if unlink:
                operation.inputs = []
                operation.outputs = []
            else:
                assert len(operation.inputs) == 0
                assert len(operation.outputs) == 0
        self._operations = [op for op in self._operations if op not in operations]
        for operation in operations:
            operation._graph = None

    def is_unique(self):
        return all(len(t.producers) <= 1 for t in self.tensors)

    def is_sorted(self):
        seen = set()
        for op in self.operations:
            for tensor in op.inputs:
                for producer in tensor.producers:
                    if producer not in seen:
                        return False
            seen.add(op)
        return True

    def sort(self, offset=0):
        count = len(self._operations)
        sorted = {op: False for op in self._operations[offset:]}
        for idx in range(offset, count):
            i = idx
            while i < count and not all(sorted.get(tensor.producer, True) for tensor in self._operations[i].inputs):
                i += 1
            if i == count:  # the graph contains a loop
                return False
            while i > idx:
                self._operations[i-1], self._operations[i] = self._operations[i], self._operations[i-1]
                i -= 1
            sorted[self._operations[i]] = True
        return True

    def move_operation(self, at_idx, to_idx):
        self._operations.insert(to_idx, self._operations.pop(at_idx))

    def reverse(self, offset=0):
        self._operations[offset:] = reversed(self._operations[offset:])

    def __repr__(self):
        return self.name if self.name is not None else _hex_id(self)

    def __str__(self):
        return "graph {name}({inputs}) -> ({outputs})".format(
            name=repr(self),
            inputs=', '.join(repr(input) for input in self.inputs),
            outputs=', '.join(repr(input) for input in self.outputs),
        )

    def print(self, file=None):
        print(f'graph {repr(self)} {{', file=file)

        print(f'\tinputs {{', file=file)
        for tensor in self.inputs:
            print('\t\t' + str(tensor) + ',', file=file)
        print(f'\t}}', file=file)

        print(f'\toutputs {{', file=file)
        for tensor in self.outputs:
            print('\t\t' + str(tensor) + ',', file=file)
        print(f'\t}}', file=file)

        print(f'\tparams {{', file=file)
        for tensor in self.tensors:
            if tensor.producer is None and tensor.data is not None:
                print('\t\t' + str(tensor) + ',', file=file)
        print(f'\t}}', file=file)

        print(f'\toperators {{', file=file)
        for operation in self._operations:
            print('\t\t' + str(operation) + ',', file=file)
        print(f'\t}}', file=file)

        print(f'}}')

    def assert_consistent(self):
        assert len(self.tensors) == len(set(self.tensors))
        assert len(self.operations) == len(set(self.operations))
        for t in self.tensors:
            assert t._graph == self
            assert all(t in consumer.inputs for consumer in t.consumers)
            assert all(t in producer.outputs for producer in t.producers)
            assert all(consumer in self.operations for consumer in t.consumers)
            assert all(producer in self.operations for producer in t.producers)
        for op in self.operations:
            assert op._graph == self
            assert all(op in t.consumers for t in op.inputs)
            assert all(op in t.producers for t in op.outputs)
        for t in self.inputs:
            assert t in self.tensors
        for t in self.outputs:
            assert t in self.tensors


class _ListView(Sequence):

    def __init__(self, lst):
        self._list = lst

    def __len__(self):
        return self._list.__len__()

    def __getitem__(self, item):
        return self._list.__getitem__(item)

    def __iter__(self):
        return self._list.__iter__()

    def __repr__(self):
        return self._list.__repr__()

    def __str__(self):
        return self._list.__str__()

    def __contains__(self, item):
        return self._list.__contains__(item)

    def __reversed__(self):
        return reversed(self._list)


def _hex_id(obj):
    return '@' + hex(id(obj))[2:]
