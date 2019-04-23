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
        # type: ()->None
        # assert self.is_unique

        ng = utils.NameGenerator(used_names=set(t.name for t in self.tensors if t.name))

        for t in self.tensors:
            if t.name:
                pass
            elif self.input_ids is not None and t in self.inputs:
                t.name = ng.get_new_name(self.input_ids[list(self.inputs).index(t)])
            elif self.output_ids is not None and t in self.outputs:
                t.name = ng.get_new_name(self.output_ids[list(self.outputs).index(t)])
            elif t.is_variable:
                t.name = ng.get_new_name('variable')
            elif t.is_constant:
                t.name = ng.get_new_name('constant')
            elif t.producer is None:
                t.name = ng.get_new_name('external')
            else:
                t.name = ng.get_new_name(t.producer.name.split('.')[-1])


__all__ = [
    'NNEFTensor',
    'NNEFOperation',
    'NNEFGraph',
    'NNEFQuantization'
]
