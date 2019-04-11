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

import numpy as np
import typing

import nnef_tools.core.utils as utils
from nnef_tools.core.base_graph import *


class TFTensor(BaseTensor['TFGraph', 'TFOperation']):
    class Quantization(object):
        def __init__(self, min, max, scale, zero_point):
            self.min = min
            self.max = max
            self.scale = scale
            self.zero_point = zero_point

        def all_zero(self):
            return self.min == self.max == self.scale == self.zero_point == 0

        def is_close_to(self, other):
            if not isinstance(other, TFTensor.Quantization):
                return False
            return np.allclose(np.array([self.min, self.max, self.scale, self.zero_point], dtype=np.float32),
                               np.array([other.min, other.max, other.scale, other.zero_point], dtype=np.float32))

        def __repr__(self):
            return repr((self.min, self.max, self.scale, self.zero_point))

    def __init__(self,
                 graph,  # type: TFGraph
                 name=None,  # type: typing.Optional[str]
                 shape=None,  # type: typing.Optional[typing.List[int]]
                 dtype=None,  # type: typing.Optional[str]
                 data=None,  # type: typing.Union[None, np.ndarray, typing.Any]
                 label=None,  # type: typing.Optional[str],
                 quantization=None,  # type: typing.Optional[TFTensor.Quantization]
                 ):
        # type: (...)->None
        super(TFTensor, self).__init__(graph=graph, name=name, shape=shape, dtype=dtype, data=data)
        self.label = label
        self.quantization = quantization

    def _str_dict(self):
        _dict = super(TFTensor, self)._str_dict()
        _dict.update([('quant', self.quantization)])
        return _dict


class TFOperation(BaseOperation['TFGraph', 'TFTensor']):
    def __init__(self, graph, name=None, inputs=None, outputs=None, attribs=None, comment=None, location=None):
        super(TFOperation, self).__init__(graph, name=name, inputs=inputs, outputs=outputs, attribs=attribs)
        self.comment = comment
        self.location = location


class TFGraph(BaseGraph['TFTensor', 'TFOperation']):

    def generate_missing_names(self):
        # type: (TFGraph)->None
        # assert self.is_unique

        ng = utils.NameGenerator()

        for t in self.tensors:
            if t.name and ng.is_available(t.name):
                t.name = ng.get_new_name(t.name)
            elif t.producer:
                t.name = ng.get_new_name(t.producer.name.split('.')[-1])
            elif t.is_variable:
                t.name = ng.get_new_name("variable")
            elif t.is_constant:
                t.name = ng.get_new_name("constant")
            else:
                t.name = ng.get_new_name("placeholder")


__all__ = [
    'TFTensor',
    'TFOperation',
    'TFGraph'
]
