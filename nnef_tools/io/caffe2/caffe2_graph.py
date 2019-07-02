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


class Caffe2Quantization(object):
    def __init__(self, scale, zero_point):
        self.scale = scale
        self.zero_point = zero_point

    def __str__(self):
        return "Q({}, {})".format(self.scale, self.zero_point)


class Caffe2Tensor(BaseTensor['Caffe2Graph', 'Caffe2Operation']):

    def __init__(self,
                 graph,  # type: Caffe2Graph
                 name=None,  # type: typing.Optional[str]
                 shape=None,  # type: typing.Optional[typing.List[int]]
                 dtype=None,  # type: typing.Optional[str]
                 data=None,  # type: typing.Union[None, np.ndarray, typing.Any]
                 quantization=None,  # type: typing.Optional[Caffe2Quantization]
                 ):
        # type: (...)->None
        super(Caffe2Tensor, self).__init__(graph=graph, name=name, shape=shape, dtype=dtype, data=data)
        self.quantization = quantization

    def _str_dict(self):
        _dict = super(Caffe2Tensor, self)._str_dict()
        _dict.update([('quantization', self.quantization)])
        return _dict


class Caffe2Operation(BaseOperation['Caffe2Graph', 'Caffe2Tensor']):

    def __init__(self, graph, name=None, inputs=None, outputs=None, attribs=None, label=None):
        super(Caffe2Operation, self).__init__(graph, name=name, inputs=inputs, outputs=outputs, attribs=attribs)
        self.label = label

    def _str_dict(self):
        _dict = super(Caffe2Operation, self)._str_dict()
        _dict.update([('label', self.label)])
        return _dict


class Caffe2Graph(BaseGraph['Caffe2Tensor', 'Caffe2Operation']):

    def __init__(self, name=None):
        # type: (typing.Optional[str])->None
        super(Caffe2Graph, self).__init__(name=name)

    def generate_missing_names(self, labels_too=True):
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

        if labels_too:
            label_generator = utils.NameGenerator(used_names=set(op.label for op in self.operations if op.label))
            for op in self.operations:
                if not op.label:
                    op.label = label_generator.get_new_name(op.outputs[0].name)


__all__ = [
    'Caffe2Quantization',
    'Caffe2Tensor',
    'Caffe2Operation',
    'Caffe2Graph',
]
