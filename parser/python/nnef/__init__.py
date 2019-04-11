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

import _nnef
from .parser import *
from .printer import *
from .binary import read_tensor, write_tensor
from .shapes import infer_shapes


Identifier = _nnef.Identifier   # subclass of str
Error = _nnef.Error             # subclass of exception

Graph = _nnef.Graph             # namedtuple('Graph', ['name': str, 'tensors': typing.Dict[str, Tensor], 'operations': typing.List[Operation],
                                #                       'inputs': typing.List[str], 'outputs': typing.List['str']])
Tensor = _nnef.Tensor           # namedtuple('Tensor', ['name': str, 'dtype': str, 'shape': typing.List[int], 'data': numpy.ndarray,
                                #                       'compression': typing.Dict[str, object], 'quantization': Dict[str, object]])
Operation = _nnef.Operation     # namedtuple('Operation', ['name': str, 'attribs': OrderedDict[str, object], 'inputs': OrderedDict[str, object],
                                #                           'outputs': OrderedDict[str, object], 'dtype': str])
