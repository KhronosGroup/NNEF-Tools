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

from nnef_tools.core.base_graph import *
from nnef_tools.core import utils


class ONNXTensor(BaseTensor['ONNXGraph', 'ONNXOperation']):
    def __init__(self,
                 graph,  # type: ONNXGraph
                 name=None,  # type: typing.Optional[str]
                 shape=None,  # type: typing.Optional[typing.List[int]]
                 dtype=None,  # type: typing.Optional[str]
                 data=None,  # type: typing.Union[None, np.ndarray, typing.Any]
                 ):
        # type: (...)->None
        super(ONNXTensor, self).__init__(graph=graph, name=name, shape=shape, dtype=dtype, data=data)

    @property
    def is_null(self):
        return self.shape == [0]

    @staticmethod
    def create_null(graph):
        return ONNXTensor(graph=graph, name=None, shape=[0], dtype=None, data=None)


class ONNXOperation(BaseOperation['ONNXGraph', 'ONNXTensor']):
    def __init__(self, graph, name=None, inputs=None, outputs=None, attribs=None):
        super(ONNXOperation, self).__init__(graph, name=name, inputs=inputs, outputs=outputs, attribs=attribs)


class ONNXGraph(BaseGraph['ONNXTensor', 'ONNXOperation']):
    def __init__(self, name=None, domain=None, version=None):
        # type: (typing.Optional[str], typing.Optional[str], typing.Optional[int])->None
        super(ONNXGraph, self).__init__(name=name)
        self.domain = domain
        self.version = version

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
    'ONNXTensor',
    'ONNXOperation',
    'ONNXGraph'
]
