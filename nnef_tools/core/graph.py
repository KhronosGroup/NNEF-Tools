from __future__ import division, print_function, absolute_import

import typing
from collections import OrderedDict, Sequence

import six

_TensorT = typing.TypeVar('_TensorT', bound='Tensor')
_OperationT = typing.TypeVar('_OperationT', bound='Operation')
_GraphT = typing.TypeVar('_GraphT', bound='Graph')

_TensorListOrTuple = typing.Union[typing.List[_TensorT], typing.Tuple[_TensorT, ...]]


# noinspection PyProtectedMember
class Tensor(typing.Generic[_GraphT, _OperationT]):

    def __init__(self, graph):
        # type: (_GraphT)->None
        self._graph = graph
        self._producers = []
        self._consumers = []

        assert isinstance(graph, Graph)
        graph._tensors.append(self)

    @property
    def graph(self):
        # type: ()->typing.Optional[_GraphT]
        return self._graph

    @property
    def producers(self):
        # type: ()->typing.List[_OperationT]
        return self._producers

    @property
    def producer(self):
        # type: ()->typing.Optional[_OperationT]
        assert len(self._producers) <= 1
        return self._producers[0] if len(self._producers) == 1 else None

    @property
    def consumers(self):
        # type: ()->typing.List[_OperationT]
        return self._consumers

    def __repr__(self):
        return 'T({})'.format(_hex_id(self))

    def __str__(self):
        return 'Tensor({}, producers={}, consumers={}'.format(_hex_id(self), self._producers, self._consumers)


# noinspection PyProtectedMember
class Operation(typing.Generic[_GraphT, _TensorT]):

    def __init__(self,
                 graph,  # type: _GraphT
                 inputs=None,  # type: typing.Union[None, Tensor, _TensorListOrTuple]
                 outputs=None  # type: typing.Union[None, Tensor, _TensorListOrTuple]
                 ):
        # type:(...)->None
        self._graph = graph
        self._inputs = tuple()
        self._outputs = tuple()

        assert isinstance(graph, Graph)
        graph._operations.append(self)

        if inputs is not None:
            self.inputs = inputs
        if outputs is not None:
            self.outputs = outputs

    @property
    def graph(self):
        # type: ()->typing.Optional[_GraphT]
        return self._graph

    @property
    def inputs(self):
        # type: ()->_TensorListOrTuple
        return self._inputs

    @property
    def input(self):
        # type: ()->_TensorT
        assert len(self._inputs) == 1
        return self._inputs[0]

    @inputs.setter
    def inputs(self, tensors):
        # type: (typing.Union[_TensorT, _TensorListOrTuple])->None
        if isinstance(tensors, Tensor):
            tensors = (tensors,)

        for tensor in self._inputs:
            assert self in tensor._consumers
        for tensor in self._inputs:
            if self in tensor._consumers:
                tensor._consumers.remove(self)

        self._inputs = _ListView(tensors) if isinstance(tensors, list) else tensors
        for tensor in tensors:
            assert isinstance(tensor, Tensor)
            assert tensor.graph is self.graph
            if self not in tensor._consumers:
                tensor._consumers.append(self)

    @property
    def outputs(self):
        # type: ()->_TensorListOrTuple
        return self._outputs

    @property
    def output(self):
        # type: ()->_TensorT
        assert len(self._outputs) == 1
        return self._outputs[0]

    @outputs.setter
    def outputs(self, tensors):
        # type: (typing.Union[_TensorT, _TensorListOrTuple])->None

        if isinstance(tensors, Tensor):
            tensors = (tensors,)

        for tensor in self._outputs:
            assert self in tensor._producers
            tensor._producers.remove(self)

        self._outputs = _ListView(tensors) if isinstance(tensors, list) else tensors
        for tensor in tensors:
            assert isinstance(tensor, Tensor)
            assert tensor.graph is self.graph
            assert self not in tensor._producers
            tensor._producers.append(self)

    def __repr__(self):
        return 'O({})'.format(_hex_id(self))

    def __str__(self):
        return 'Operation({}, inputs={}, outputs={})'.format(_hex_id(self), self._inputs, self._outputs)


# noinspection PyProtectedMember
class Graph(typing.Generic[_TensorT, _OperationT]):

    def __init__(self):
        self._operations = []
        self._tensors = []
        self._inputs = []
        self._outputs = []
        self._input_ids = None
        self._output_ids = None

    @property
    def operations(self):
        # type: ()->typing.Sequence[_OperationT]
        return _ListView(self._operations)

    @property
    def tensors(self):
        # type: ()->typing.Sequence[_TensorT]
        return _ListView(self._tensors)

    @property
    def inputs(self):
        # type: ()->typing.Sequence[_TensorT]
        return _ListView(self._inputs)

    @property
    def input_ids(self):
        # type: ()->typing.Optional[typing.Sequence[str]]
        return _ListView(self._input_ids) if self._input_ids is not None else None

    @inputs.setter
    def inputs(self, tensors):
        # type: (typing.Union[_TensorListOrTuple, OrderedDict[str, typing.Any]])->None
        assert isinstance(tensors, (list, tuple, OrderedDict))

        if isinstance(tensors, OrderedDict):
            self._inputs = list(six.itervalues(tensors))
            self._input_ids = list(six.iterkeys(tensors))
        else:
            self._inputs = tensors
            self._input_ids = None

        for tensor in self._inputs:
            assert isinstance(tensor, Tensor)
            assert tensor.graph is self

    @property
    def outputs(self):
        # type: ()->typing.Sequence[_TensorT]
        return _ListView(self._outputs)

    @property
    def output_ids(self):
        # type: ()->typing.Optional[typing.Sequence[str]]
        return _ListView(self._output_ids) if self._output_ids is not None else None

    @outputs.setter
    def outputs(self, tensors):
        # type: (typing.Union[_TensorListOrTuple, typing.Dict[str, typing.Any]])->None
        assert isinstance(tensors, (list, tuple, OrderedDict))

        if isinstance(tensors, OrderedDict):
            self._outputs = list(six.itervalues(tensors))
            self._output_ids = list(six.iterkeys(tensors))
        else:
            self._outputs = tensors
            self._output_ids = None

        for tensor in self._outputs:
            assert isinstance(tensor, Tensor)
            assert tensor.graph is self

    @property
    def is_unique(self):
        return all(len(t.producers) <= 1 for t in self.tensors)

    def remove_tensor(self, tensor):
        # type: (_TensorT)->None
        assert len(tensor.producers) == 0
        assert len(tensor.consumers) == 0
        assert tensor not in self._inputs
        assert tensor not in self._outputs
        self._tensors.remove(tensor)
        tensor._graph = None

    def remove_tensors(self, tensors):
        # type: (typing.Iterable[_TensorT])->None
        for tensor in tensors:
            assert len(tensor.producers) == 0
            assert len(tensor.consumers) == 0
            assert tensor not in self._inputs
            assert tensor not in self._outputs
        self._tensors = [tensor for tensor in self._tensors if tensor not in tensors]
        for tensor in tensors:
            tensor._graph = None

    def remove_operation(self, operation, unlink=False):
        # type: (_OperationT, bool)->None
        if unlink:
            operation.inputs = []
            operation.outputs = []
        else:
            assert len(operation.inputs) == 0
            assert len(operation.outputs) == 0
        self._operations.remove(operation)
        operation._graph = None

    def remove_operations(self, operations, unlink=False):
        # type: (typing.Iterable[_OperationT], bool)->None
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

    def sort(self):
        # TODO change to iterative

        # Ensure that we get the same order for the same graph
        # including when the ops are regenerated for some source tensors
        self._operations.sort(key=lambda op_: id(op_.outputs[0]) if len(op_.outputs) > 0 else id(op_))

        visited = set()
        operations = []

        def dfs(op_):
            if op_ in visited:
                return
            visited.add(op_)
            for t in op_.outputs:
                for consumer in t.consumers:
                    dfs(consumer)
            operations.append(op_)

        for op in self.operations:
            dfs(op)

        self._operations = operations[::-1]

    def __repr__(self):
        return 'G({})'.format(_hex_id(self))

    def __str__(self):
        return 'Graph({}, inputs={}, outputs={})'.format(_hex_id(self), self._inputs, self._outputs)

    def dump(self, file=None):
        # type: (typing.TextIO)->None
        print(str(self), file=file)
        print('--Tensors--')
        for tensor in self._tensors:
            print(str(tensor), file=file)
        print('--Operations--')
        for operation in self._operations:
            print(str(operation), file=file)

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
        assert self.input_ids is None or len(self.input_ids) == len(self.inputs)
        assert self.output_ids is None or len(self.output_ids) == len(self.outputs)


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
        return self._list.__reversed__()


def _hex_id(obj):
    return '@' + hex(id(obj))[2:]


__all__ = [
    'Tensor',
    'Operation',
    'Graph'
]
