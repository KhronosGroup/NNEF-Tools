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

from nnef_tools.core import utils
from nnef_tools.core.base_graph import *


class CaffeTensor(BaseTensor['CaffeGraph', 'CaffeOperation']):

    def __init__(self,
                 graph,  # type: CaffeGraph
                 name=None,  # type: typing.Optional[str]
                 shape=None,  # type: typing.Optional[typing.List[int]]
                 dtype=None,  # type: typing.Optional[str]
                 data=None,  # type: typing.Union[None, np.ndarray, typing.Any]
                 ):
        # type: (...)->None
        super(CaffeTensor, self).__init__(graph=graph, name=name, shape=shape, dtype=dtype, data=data)


class CaffeOperation(BaseOperation['CaffeGraph', 'CaffeTensor']):

    def __init__(self, graph, name=None, inputs=None, outputs=None, attribs=None, label=None):
        super(CaffeOperation, self).__init__(graph, name=name, inputs=inputs, outputs=outputs, attribs=attribs)
        self.label = label

    def _str_dict(self):
        _dict = super(CaffeOperation, self)._str_dict()
        _dict.update([('label', self.label)])
        return _dict


class CaffeGraph(BaseGraph['CaffeTensor', 'CaffeOperation']):

    def __init__(self, name=None):
        # type: (typing.Optional[str])->None
        super(CaffeGraph, self).__init__(name=name)

    def generate_missing_names(self):
        name_generator = utils.NameGenerator(used_names=set(t.name for t in self.tensors if t.name))

        for t in self.tensors:
            if not t.name:
                if t.is_variable:
                    t.name = name_generator.get_new_name('variable')
                elif t.is_constant:
                    t.name = name_generator.get_new_name('constant')
                else:
                    assert t.producer
                    t.name = name_generator.get_new_name(t.producer.name)

        label_generator = utils.NameGenerator(used_names=set(op.label for op in self.operations if op.label))
        for op in self.operations:
            if not op.label:
                op.label = label_generator.get_new_name(op.outputs[0].name)


__all__ = [
    'CaffeTensor',
    'CaffeOperation',
    'CaffeGraph'
]
