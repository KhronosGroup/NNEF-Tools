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

import typing

import numpy as np
import six

from nnef_tools.core import utils
from nnef_tools.core.base_graph import *


class NNEFQuantization(object):
    def __init__(self, name, attribs=None):
        if attribs is None:
            attribs = {}
        self.name = name
        self.attribs = attribs

    def __repr__(self):
        return "Q({}, {})".format(self.name, self.attribs)

    def is_close_to(self, other):
        if not isinstance(other, NNEFQuantization):
            return False
        if self.name != other.name:
            return False
        if set(six.iterkeys(self.attribs)) != set(six.iterkeys(other.attribs)):
            return False
        for key in six.iterkeys(self.attribs):
            value1 = self.attribs[key]
            value2 = other.attribs[key]
            if isinstance(value1, float):
                if not np.allclose(np.array(value1), np.array(value2)):
                    return False
            else:
                if value1 != value2:
                    return False
        return True


class NNEFTensor(BaseTensor["NNEFGraph", "NNEFOperation"]):
    def __init__(self,
                 graph,  # type: NNEFGraph
                 name=None,  # type: typing.Optional[str]
                 shape=None,  # type: typing.Optional[typing.List[int]]
                 dtype=None,  # type: typing.Optional[str]
                 data=None,  # type: typing.Union[None, np.ndarray, typing.Any]
                 label=None,  # type: typing.Optional[str],
                 quantization=None,  # type: typing.Optional[NNEFQuantization]
                 ):
        # type: (...)->None
        super(NNEFTensor, self).__init__(graph=graph, name=name, shape=shape, dtype=dtype, data=data)
        assert bool(self.is_variable) == bool(label is not None)
        self.label = label
        self.quantization = quantization

    def _str_dict(self):
        _dict = super(NNEFTensor, self)._str_dict()
        _dict.update([('quant', self.quantization)])
        return _dict

    # experimental
    def get_numpy_array(self):
        # type: ()->typing.Optional[np.ndarray]
        if self.data is None:
            return None

        if isinstance(self.data, np.ndarray):
            assert list(self.shape) == list(self.data.shape)
            return self.data
        else:
            assert self.shape is not None
            assert self.dtype is not None

            if len(self.data) == 1:
                return np.full(shape=self.shape, fill_value=self.data[0]).astype(self.get_numpy_dtype())
            else:
                return np.array(self.data).reshape(self.shape).astype(self.get_numpy_dtype())

    # experimental
    def get_numpy_dtype(self):
        # type: ()->typing.Optional[typing.Any]
        if isinstance(self.data, np.ndarray):
            return self.data.dtype
        if self.dtype is not None:
            return {'scalar': np.float32, 'integer': np.int32, 'logical': np.bool}[self.dtype]
        return None


_OptTensorOrListOrTuple = typing.Union[None, NNEFTensor, typing.List[NNEFTensor], typing.Tuple[NNEFTensor, ...]]


class NNEFOperation(BaseOperation["NNEFGraph", "NNEFTensor"]):
    def __init__(self,
                 graph,  # type: NNEFGraph
                 name=None,  # type: typing.Optional[str]
                 inputs=None,  # type: _OptTensorOrListOrTuple
                 outputs=None,  # type: _OptTensorOrListOrTuple
                 attribs=None,  # type: typing.Optional[typing.Dict[str, typing.Any]]
                 comment=None,  # type: typing.Optional[str]
                 ):
        super(NNEFOperation, self).__init__(graph=graph, name=name, inputs=inputs, outputs=outputs, attribs=attribs)
        self.comment = comment


class NNEFGraph(BaseGraph["NNEFTensor", "NNEFOperation"]):

    def generate_missing_names(self):
        name_generator = utils.NameGenerator(used_names=set(t.name for t in self.tensors if t.name))

        for t in self.tensors:
            if not t.name:
                if self.input_ids is not None and t in self.inputs:
                    t.name = name_generator.get_new_name(self.input_ids[list(self.inputs).index(t)])
                elif self.output_ids is not None and t in self.outputs:
                    t.name = name_generator.get_new_name(self.output_ids[list(self.outputs).index(t)])
                elif t.is_variable:
                    t.name = name_generator.get_new_name('variable')
                elif t.is_constant:
                    t.name = name_generator.get_new_name('constant')
                elif t.producer is None:
                    t.name = name_generator.get_new_name('external')
                else:
                    t.name = name_generator.get_new_name(t.producer.name)

        label_generator = utils.NameGenerator(used_names=set(t.label for t in self.tensors if t.label))
        for t in self.tensors:
            if t.is_variable and not t.label:
                t.label = label_generator.get_new_name('variable')

        if not self.name:
            self.name = "network"


__all__ = [
    'NNEFTensor',
    'NNEFOperation',
    'NNEFGraph',
    'NNEFQuantization'
]
