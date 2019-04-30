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
import numpy as np
import six
import typing

from nnef_tools.core import utils
from nnef_tools.core.graph import *
# noinspection PyProtectedMember
from nnef_tools.core.graph import _hex_id, _TensorT, _GraphT, _OperationT, _TensorListOrTuple


class BaseTensor(Tensor[_GraphT, _OperationT]):

    def __init__(self,
                 graph,  # type: _GraphT
                 name=None,  # type: typing.Optional[str]
                 shape=None,  # type: typing.Optional[typing.List[int]]
                 dtype=None,  # type: typing.Optional[str]
                 data=None,  # type: typing.Union[None, np.ndarray, typing.Any]
                 ):
        # type: (...)->None
        super(BaseTensor, self).__init__(graph)

        assert isinstance(graph, Graph)

        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.data = data

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
        assert shape is None or isinstance(shape, list)
        assert shape is None or all(isinstance(dim, int) for dim in shape)
        self._shape = shape

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        assert dtype is None or isinstance(dtype, str)
        self._dtype = dtype

    @property
    def rank(self):
        # type: ()->typing.Optional[int]
        return len(self.shape) if self.shape is not None else None

    @property
    def count(self):
        # type: ()->typing.Optional[int]
        return int(np.prod(self.shape)) if self.shape is not None else None

    @property
    def is_variable(self):
        # type: ()->bool
        return self.data is not None and isinstance(self.data, np.ndarray)

    @property
    def is_constant(self):
        # type: ()->bool
        return self.data is not None and not isinstance(self.data, np.ndarray)

    def __repr__(self):
        return 'T({})'.format(_name_or_id(self))

    def __str__(self):
        return _str_format(self, 'Tensor')

    def _str_dict(self):
        max_data_length_to_print = 8

        if self.data is None:
            data_str = "None"
        elif isinstance(self.data, list):
            data_str = "list("
            if len(self.data) <= max_data_length_to_print:
                data_str += str(self.data)
            else:
                data_str += "..."
            data_str += ")"
        else:
            assert isinstance(self.data, np.ndarray)
            data_str = "ndarray("
            if self.data.size <= max_data_length_to_print:
                data_str += str(self.data.tolist())
            else:
                data_str += "..."
            data_str += ")"

        return OrderedDict([('shape', self.shape), ('dtype', self.dtype),
                            ('producers', self._producers), ('consumers', self._consumers),
                            ('data', data_str)])


class BaseOperation(Operation[_GraphT, _TensorT]):

    def __init__(self,
                 graph,  # type: _GraphT
                 name=None,  # type: typing.Optional[str]
                 inputs=None,  # type: typing.Union[None, Tensor, _TensorListOrTuple]
                 outputs=None,  # type: typing.Union[None, Tensor, _TensorListOrTuple]
                 attribs=None  # type: typing.Dict[str, typing.Any]
                 ):
        # type: (...)->None
        super(BaseOperation, self).__init__(graph, inputs, outputs)

        assert name is None or isinstance(name, str)
        assert attribs is None or isinstance(attribs, dict)
        assert attribs is None or all(isinstance(key, str) for key in six.iterkeys(attribs))
        assert attribs is None or not utils.recursive_any(attribs, lambda x: isinstance(x, Tensor))

        self.name = name  # type: typing.Optional[str]
        self.attribs = attribs if attribs is not None else {}  # type: typing.Dict[str, typing.Any]

    def __repr__(self):
        return 'O({})'.format(_name_or_id(self))

    def __str__(self):
        return _str_format(self, 'Operation')

    def _str_dict(self):
        return OrderedDict([('inputs', self._inputs), ('outputs', self._outputs), ('attribs', self.attribs)])


class BaseGraph(Graph[_TensorT, _OperationT]):

    def __init__(self, name=None):
        # type: (typing.Optional[str])->None
        super(BaseGraph, self).__init__()
        self.name = name  # type: typing.Optional[str]

    def list_variables(self):
        # type: ()->typing.List[_TensorT]
        return [t for t in self.tensors if t.is_variable]

    def list_constants(self):
        # type: ()->typing.List[_TensorT]
        return [t for t in self.tensors if t.is_constant]

    def __repr__(self):
        return 'G({})'.format(_name_or_id(self))

    def __str__(self):
        return _str_format(self, 'Graph')

    def _str_dict(self):
        return OrderedDict([('inputs', self._inputs), ('outputs', self._outputs)])


def _name_or_id(obj):
    return obj.name if obj.name is not None else _hex_id(obj)


def _str_format(obj, kind):
    attribs = ', '.join('{}={}'.format(key, value) for key, value in six.iteritems(obj._str_dict()))
    return '{}({}, {})'.format(kind, _name_or_id(obj), attribs)


__all__ = [
    'BaseTensor',
    'BaseOperation',
    'BaseGraph'
]
