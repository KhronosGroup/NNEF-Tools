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

from collections.abc import Sequence
from functools import reduce

import six
import typing
import numpy as np


# noinspection PyProtectedMember
class Tensor:

    def __init__(self,
                 graph,             # type: Model,
                 name=None,         # type: typing.Optional[str]
                 shape=None,        # type: typing.Optional[typing.Tuple[int, ...]]
                 dtype=None,        # type: typing.Optional[type]
                 data=None,         # type: typing.Union[None, np.ndarray, typing.Any]
                 quant=None,        # type: typing.Optional[typing.Dict[str, typing.Any]]
                 variable=None,     # type: typing.Optional[bool]
                 ):
        # type: (...)->None
        self._graph = graph
        self._producer = None
        self._consumers = []
        self._variable = variable if variable is not None else isinstance(data, np.ndarray)

        self.name = name            # type: typing.Optional[str]
        self.shape = shape          # type: typing.Optional[typing.Tuple[int, ...]]
        self.dtype = dtype          # type: typing.Optional[type]
        self.data = data            # type: typing.Union[None, np.ndarray, typing.Any]
        self.quant = quant or {}    # type: typing.Optional[typing.Dict[str, typing.Any]]

        assert isinstance(graph, Graph)
        graph._tensors.append(self)

    def copy_with(self, graph=None, name=None, dtype=None, shape=None, data=None, quant=None, variable=None):
        return Tensor(graph=graph if graph is not None else self.graph,
                      name=name if name is not None else self.name,
                      dtype=dtype if dtype is not None else self.dtype,
                      shape=shape if shape is not None else self.shape,
                      data=data if data is not None else self.data,
                      quant=quant if quant is not None else self.quant,
                      variable=variable if variable is not None else self.is_variable)

    @property
    def graph(self):
        # type: ()->typing.Optional[Graph]
        return self._graph

    @property
    def has_producer(self):
        return self._producer is not None

    @property
    def producer(self):
        # type: ()->typing.Optional[Operation]
        return self._producer

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
        return self.data is not None and not self._variable

    @property
    def is_variable(self):
        # type: ()->bool
        return self._variable

    @property
    def is_activation(self):
        # type: ()->bool
        return self.data is None

    @property
    def is_null(self):
        return self.dtype == np.void

    def set_data(self, data, variable=None):
        self.data = data
        self._variable = variable if variable is not None else isinstance(data, np.ndarray)

    def __repr__(self):
        return self.name if self.name is not None else _hex_id(self)

    def __str__(self):
        return '{name}: {dtype}[{shape}]'.format(
            name=self.name if self.name is not None else _hex_id(self),
            dtype=self.dtype.__name__ if self.dtype is not None else 'void',
            shape=', '.join(str(s) for s in self.shape) if self.shape is not None else '...')


class TensorPack(list):

    def __init__(self, graph, name, shape, dtype, size, items):
        super().__init__(items)
        self._graph = graph
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.size = size

        assert isinstance(graph, Graph)
        graph._packs.append(self)

    def copy_with(self, graph=None, name=None, dtype=None, shape=None, size=None, items=None):
        return TensorPack(graph=graph if graph is not None else self.graph,
                          name=name if name is not None else self.name,
                          dtype=dtype if dtype is not None else self.dtype,
                          shape=shape if shape is not None else self.shape,
                          size=size if size is not None else self.size,
                          items=items if items is not None else self)

    @property
    def graph(self):
        # type: ()->typing.Optional[Graph]
        return self._graph

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
        self._shape = tuple(shape) if shape is not None else None

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        assert dtype is None or isinstance(dtype, type)
        self._dtype = dtype

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = size

    @property
    def rank(self):
        # type: ()->typing.Optional[int]
        return len(self.shape) if self.shape is not None else None

    @property
    def volume(self):
        # type: ()->typing.Optional[int]
        return reduce((lambda x, y: x * y), self.shape) if self.shape is not None and \
                                                           all(s is not None for s in self.shape) else None

    def __repr__(self):
        return self.name if self.name is not None else _hex_id(self)

    def __str__(self):
        return '{name}: {dtype}[{shape}]'.format(
            name=self.name if self.name is not None else _hex_id(self),
            dtype=self.dtype.__name__ if self.dtype is not None else 'void',
            shape=', '.join(str(s) for s in self.shape) if self.shape is not None else '...')


# noinspection PyProtectedMember
class Operation:

    def __init__(self,
                 graph,         # type: Graph
                 type=None,     # type: typing.Optional[str]
                 name=None,     # type: typing.Optional[str]
                 dtypes=None,   # type: typind.Dict[str, np.dtype]
                 attribs=None,  # type: typing.Dict[str, typing.Any]
                 inputs=None,   # type: typing.Union[None, Tensor, _TensorListOrTuple]
                 outputs=None,  # type: typing.Union[None, Tensor, _TensorListOrTuple]
                 custom=False,  # type: bool
                 ):
        # type:(...)->None
        self._graph = graph
        self._inputs = tuple()
        self._outputs = tuple()

        assert name is None or isinstance(name, str)
        if attribs is not None:
            assert isinstance(attribs, dict)
            assert all(isinstance(key, str) for key in six.iterkeys(attribs))

        self.type = type                # type: typing.Optional[str]
        self.name = name                # type: typing.Optional[str]
        self.dtypes = dtypes or {}      # type: typing.Dict[str, np.dtype]
        self.attribs = attribs or {}    # type: typing.Dict[str, typing.Any]
        self.custom = custom            # type: bool

        assert isinstance(graph, Graph)
        graph._operations.append(self)

        if inputs is not None:
            self.inputs = inputs
        if outputs is not None:
            self.outputs = outputs

    def copy_with(self, graph=None, type=None, name=None, dtypes=None, attribs=None, inputs=None, outputs=None, custom=None):
        return Operation(graph=graph if graph is not None else self.graph,
                         type=type if type is not None else self.type,
                         name=name if name is not None else self.name,
                         dtypes=dtypes if dtypes is not None else self.dtypes,
                         attribs=attribs if attribs is not None else self.attribs,
                         inputs=inputs if inputs is not None else self.inputs,
                         outputs=outputs if outputs is not None else self.outputs,
                         custom=custom if custom is not None else self.custom)

    @property
    def graph(self):
        # type: ()->Graph
        return self._graph

    @property
    def inputs(self):
        # type: ()->typing.Tuple[Tensor]
        return self._inputs

    @property
    def input(self):
        # type: ()->Tensor
        assert len(self._inputs) == 1
        return self._inputs[0]

    @inputs.setter
    def inputs(self, tensors):
        # type: (typing.Union[Tensor, typing.List[Tensor], typing.Tuple[Tensor]])->None
        if isinstance(tensors, (Tensor, list)):
            tensors = (tensors,)

        assert isinstance(tensors, tuple)
        for tensor in tensors:
            if isinstance(tensor, list):
                assert all(isinstance(t, Tensor) or t is None for t in tensor)
            else:
                assert isinstance(tensor, Tensor) or tensor is None

        for tensor in self._inputs:
            if isinstance(tensor, list):
                assert all(t is None or self in t._consumers for t in tensor)
            else:
                assert tensor is None or self in tensor._consumers

        for tensor in self._inputs:
            if isinstance(tensor, list):
                for t in tensor:
                    if t is not None and self in t._consumers:
                        t._consumers.remove(self)
            else:
                if tensor is not None and self in tensor._consumers:
                    tensor._consumers.remove(self)

        self._inputs = tensors
        for tensor in tensors:
            if isinstance(tensor, list):
                for t in tensor:
                    if t is not None and self not in t._consumers:
                        t._consumers.append(self)
            else:
                if tensor is not None and self not in tensor._consumers:
                    tensor._consumers.append(self)

    @property
    def outputs(self):
        # type: ()->typing.Tuple[Tensor]
        return self._outputs

    @property
    def output(self):
        # type: ()->Tensor
        assert len(self._outputs) == 1
        return self._outputs[0]

    @outputs.setter
    def outputs(self, tensors):
        # type: (typing.Union[Tensor, typing.List[Tensor], typing.Tuple[Tensor]])->None

        if isinstance(tensors, (Tensor, list)):
            tensors = (tensors,)

        assert isinstance(tensors, tuple)
        for tensor in tensors:
            if isinstance(tensor, list):
                assert all(isinstance(t, Tensor) for t in tensor)
                assert all(t.graph is self.graph for t in tensor)
            else:
                assert isinstance(tensor, Tensor)
                assert tensor.graph is self.graph

        for tensor in self._outputs:
            if isinstance(tensor, list):
                assert all(self is t._producer for t in tensor)
            else:
                assert self is tensor._producer

        for tensor in self._outputs:
            if isinstance(tensor, list):
                for t in tensor:
                    t._producer = None
            else:
                tensor._producer = None

        self._outputs = tensors
        for tensor in tensors:
            if isinstance(tensor, list):
                for t in tensor:
                    if not t.is_null:
                        assert self is not t._producer
                        t._producer = self
            else:
                if not tensor.is_null:
                    assert self is not tensor._producer
                    tensor._producer = self

    def detach_output(self):
        output = self.output
        self._outputs = ()
        return output

    def detach_outputs(self):
        outputs = self.outputs
        self._outputs = ()
        return outputs

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

    def __init__(self, model, name=None, inputs=None, outputs=None):
        self._model = model
        self._inputs = tuple()
        self._outputs = tuple()
        self._tensors = []
        self._packs = []
        self._operations = []

        self.name = name

        if inputs is not None:
            self.inputs = inputs

        if outputs is not None:
            self.outputs = outputs

        assert isinstance(model, Model)
        model._graphs.append(self)

    @property
    def model(self):
        # type: ()->Model
        return self._model

    @property
    def inputs(self):
        # type: ()->typing.Tuple[Tensor]
        return self._inputs

    @inputs.setter
    def inputs(self, tensors):
        # type: (typing.Tuple[Tensor])->None
        assert isinstance(tensors, tuple)

        for tensor in self._inputs:
            if isinstance(tensor, list):
                assert all(isinstance(t, Tensor) for t in tensor)
            else:
                assert isinstance(tensor, Tensor)

        self._inputs = tensors

    @property
    def outputs(self):
        # type: ()->typing.Tuple[Tensor]
        return self._outputs

    @outputs.setter
    def outputs(self, tensors):
        # type: (typing.Tuple[Tensor])->None
        assert isinstance(tensors, tuple)
        for tensor in self._outputs:
            if isinstance(tensor, list):
                assert all(isinstance(t, Tensor) for t in tensor)
                assert all(t.graph is self for t in tensor)
            else:
                assert isinstance(tensor, Tensor)
                assert tensor.graph is self

        self._outputs = tensors

    @property
    def tensors(self):
        # type: ()->typing.Sequence[Tensor]
        return _ListView(self._tensors)

    @property
    def packs(self):
        # type: ()->typing.Sequence[TensorPack]
        return _ListView(self._packs)

    @property
    def operations(self):
        # type: ()->typing.Sequence[Operation]
        return _ListView(self._operations)

    def remove_tensor(self, tensor):
        # type: (Tensor)->None
        assert not tensor.has_producer
        assert not tensor.has_consumer
        assert tensor not in _recursive_itemize(self.inputs)
        assert tensor not in _recursive_itemize(self.outputs)
        self._tensors.remove(tensor)
        tensor._graph = None

    def remove_tensors(self, tensors):
        # type: (typing.Iterable[Tensor])->None
        for tensor in tensors:
            assert not tensor.has_producer
            assert not tensor.has_consumer
            assert tensor not in _recursive_itemize(self.inputs)
            assert tensor not in _recursive_itemize(self.outputs) or not tensor.is_constant
        self._tensors = [tensor for tensor in self._tensors if tensor not in tensors]
        self._outputs = tuple(tensor for tensor in self._outputs if tensor not in tensors)
        for tensor in tensors:
            tensor._graph = None

    def remove_pack(self, pack):
        assert pack not in self.inputs
        assert pack not in self.outputs
        self._packs.remove(pack)
        pack._graph = None

    def remove_packs(self, packs):
        for pack in packs:
            assert pack not in self.inputs
            assert pack not in self.outputs
        self._packs = [pack for pack in self._packs if pack not in packs]
        self._outputs = tuple(item for item in self._outputs if item not in packs)
        for pack in packs:
            pack._graph = None

    def remove_operation(self, operation, unlink=False):
        # type: (Operation, bool)->None
        if unlink:
            operation.inputs = ()
            operation.outputs = ()
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
                operation.inputs = ()
                operation.outputs = ()
            else:
                assert len(operation.inputs) == 0
                assert len(operation.outputs) == 0
        self._operations = [op for op in self._operations if op not in operations]
        for operation in operations:
            operation._graph = None

    def move_operation(self, at_idx, to_idx):
        self._operations.insert(to_idx, self._operations.pop(at_idx))

    def reverse(self, offset=0):
        self._operations[offset:] = reversed(self._operations[offset:])

    def is_sorted(self):
        seen = set()
        for op in self._operations:
            for tensor in _recursive_itemize(op.inputs):
                if tensor.has_producer and tensor.producer not in seen:
                    return False
            seen.add(op)
        return True

    def sort(self, offset=0):
        count = len(self._operations)
        sorted = {op: False for op in self._operations[offset:]}
        for idx in range(offset, count):
            i = idx
            while i < count and not all(sorted.get(tensor.producer, True)
                                        for tensor in _recursive_itemize(self._operations[i].inputs)
                                        if tensor is not None):
                i += 1
            if i == count:  # the graph contains a loop
                return False
            while i > idx:
                self._operations[i-1], self._operations[i] = self._operations[i], self._operations[i-1]
                i -= 1
            sorted[self._operations[i]] = True
        return True


# noinspection PyProtectedMember
class Model:

    def __init__(self, name=None, version=None):
        # type:(typing.Optional[str], typing.Optional[typing.Any])->None
        self._graphs = []
        self._name = name
        self._version = version

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert name is None or isinstance(name, str)
        self._name = name

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, version):
        self._version = version

    @property
    def graphs(self):
        # type: ()->typing.Sequence[Graph]
        return _ListView(self._graphs)

    @property
    def main(self):
        # type: ()->Graph
        return self._graphs[0] if len(self._graphs) else None

    def remove_graph(self, graph):
        for g in self.graphs:
            for op in g.operations:
                for key, value in op.attribs.items():
                    if isinstance(value, Graph):
                        assert value is not graph
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, Graph):
                                assert item is not graph
        self._graphs.remove(graph)
        graph._model = None

    def remove_graphs(self, graphs):
        for g in self.graphs:
            for op in g.operations:
                for key, value in op.attribs.items():
                    if isinstance(value, Graph):
                        assert value not in graphs
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, Graph):
                                assert item not in graphs
        self._graphs = [graph for graph in self.graphs if graph not in graphs]
        for graph in graphs:
            graph._model = None

    def __repr__(self):
        return self.name if self.name is not None else _hex_id(self)

    def __str__(self):
        str = "model {name}".format(name=repr(self))
        main = self.main
        if main:
            str += "({inputs}) -> ({outputs})".format(
                name=repr(self),
                inputs=', '.join(repr(input) for input in main.inputs),
                outputs=', '.join(repr(input) for input in main.outputs),
            )
        return str

    def print(self, file=None):
        print(f'model {repr(self)} {{', file=file)

        for i, graph in enumerate(self.graphs):
            print(f'\tgraph{i} {{', file=file)

            print(f'\t\tinputs {{', file=file)
            for tensor in graph._inputs:
                print('\t\t\t' + str(tensor) + ',', file=file)
            print(f'\t\t}}', file=file)

            print(f'\t\toutputs {{', file=file)
            for tensor in graph.outputs:
                print('\t\t\t' + str(tensor) + ',', file=file)
            print(f'\t\t}}', file=file)

            if sum(1 if not tensor.has_producer and tensor.data is not None else 0 for tensor in graph.tensors) > 0:
                print(f'\tparams {{', file=file)
                for tensor in graph.tensors:
                    if not tensor.has_producer and tensor.data is not None:
                        print('\t\t' + str(tensor) + ',', file=file)
                print(f'\t}}', file=file)

            print(f'\t\toperators {{', file=file)
            for operation in graph._operations:
                print('\t\t\t' + str(operation) + ',', file=file)
            print(f'\t\t}}', file=file)

            print(f'\t}}', file=file)

        print(f'}}', file=file)

    def assert_consistent(self):
        for graph in self.graphs:
            assert len(graph.tensors) == len(set(graph.tensors))
            assert len(graph.operations) == len(set(graph.operations))
            for op in graph.operations:
                assert op.graph == graph
                assert all(op in t.consumers for t in op.inputs)
                assert all(op is t.producer for t in op.outputs)
            for t in graph.inputs:
                assert t in self.tensors
            for t in graph.outputs:
                assert t in self.tensors
            for t in graph.tensors:
                assert t._graph == graph
                assert all(t in consumer.inputs for consumer in t.consumers)
                assert all(consumer in graph.operations for consumer in t.consumers)
                assert t in t.producer.outputs
                assert t.producer in graph.operations

    def is_sorted(self):
        return all(graph.is_sorted() for graph in self.graphs)

    def sort(self):
        for graph in self.graphs:
            graph.sort()


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


def _recursive_itemize(arg):
    if type(arg) is list or type(arg) is tuple:
        for item in arg:
            yield from _recursive_itemize(item)
    else:
        yield arg
