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

import nnef
import numpy as np
from collections import OrderedDict


input = nnef.Tensor('input', dtype='scalar')
filter = nnef.Tensor('filter', dtype='scalar', data=np.random.randn(32,3,5,5))
output = nnef.Tensor('output', dtype='scalar')

external = nnef.Operation('external', attribs={'shape': [1,3,224,224]},
                          inputs=OrderedDict(),
                          outputs=OrderedDict([('output', nnef.Identifier('input'))]))
variable = nnef.Operation('variable', attribs={'shape': [32,3,5,5], 'label': 'conv/filter'},
                          inputs=OrderedDict(),
                          outputs=OrderedDict([('output', nnef.Identifier('filter'))]))
conv = nnef.Operation('conv', attribs={},
                      inputs=OrderedDict([('input', nnef.Identifier('input')), ('filter', nnef.Identifier('filter'))]),
                      outputs=OrderedDict([('output', nnef.Identifier('output'))]))

graph = nnef.Graph('G', inputs=['input'], outputs=['output'], operations=[external, variable, conv],
                   tensors={'input': input, 'filter': filter, 'output': output})

nnef.save_graph(graph, 'G', annotate_shapes=True)
