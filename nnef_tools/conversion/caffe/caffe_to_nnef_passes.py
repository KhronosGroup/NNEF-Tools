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

from nnef_tools.core import graph_utils
from nnef_tools.core import matcher
from nnef_tools.io.caffe.caffe_graph import *


def pre_conversion_pass(g):
    # type: (CaffeGraph)->None
    graph_utils.remove_unreachable(g)
    graph_utils.remove_passthroughs(g, is_passthrough=lambda op: op.name in ('Dropout', 'Silence'))
    _merge_batch_norm_and_scale(g)
    graph_utils.remove_unreachable(g)


def _merge_batch_norm_and_scale(g):
    # type: (CaffeGraph)->None

    input, mean, variance, scale_factor, offset, scale, normed, output = matcher.tensors(8)

    batch_norm_op = matcher.Operation(name='BatchNorm',
                                      inputs=(input, mean, variance, scale_factor),
                                      outputs=normed)
    scale_op = matcher.Operation(name='Scale',
                                 inputs=(normed, scale, offset),
                                 outputs=output,
                                 attribs=dict(axis=1, num_axes=1))

    matcher.replace(g, scale_op,
                    lambda m: CaffeOperation(
                        graph=g,
                        name='BatchNorm+Scale',
                        inputs=(m[input], m[mean], m[variance], m[scale_factor], m[offset], m[scale]),
                        outputs=m[output],
                        attribs=dict(eps=m[batch_norm_op].attribs['eps'])))
