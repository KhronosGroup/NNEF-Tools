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
from nnef_tools.core import utils
from nnef_tools.io.nnef.nnef_graph import *


def post_conversion_pass(g):
    # type: (NNEFGraph)->None
    graph_utils.remove_unreachable(g)
    # _small_variables_to_consts(g)
    _merge_pads(g)
    graph_utils.remove_unreachable(g)


def _merge_pads(g):
    # type: (NNEFGraph)->None

    t = matcher.Tensor()
    pad = matcher.Operation(name=['box', 'pad'], outputs=t)
    sliding = matcher.Operation(name=['argmax_pool', 'max_pool', 'max_pool_with_index', 'avg_pool', 'conv'],
                                inputs={0: t})

    def condition(m):
        # type: (matcher.Match)->bool
        if not (m[pad].name == 'pad' or
                (m[pad].name == 'box'
                 and all(s == 1 for s in m[pad].attribs.get('size', []))
                 and all(s == 1 for s in m[pad].attribs.get('stride', []))
                 and all(s == 1 for s in m[pad].attribs.get('dilation', []))
                 and not m[pad].attribs.get('normalize', False))):
            return False

        value = m[pad].attribs.get('_value', 0.0)

        if value not in [0.0, float('-inf')]:
            return False

        if value == float('-inf'):
            if not m[sliding].name in ['argmax_pool', 'max_pool', 'max_pool_with_index']:
                return False

        if m[pad].attribs.get('border', 'constant') != 'constant':
            return False

        if (m[sliding].attribs.get('border', 'constant') != 'constant'
                and any(p != 0 or q != 0 for p, q in m[sliding].attribs.get('padding', []))):
            return False

        if m[sliding].name in ['conv'] and any(p != 0 or q != 0 for p, q in m[pad].attribs.get('padding', [])[:2]):
            return False

        return True

    def action(m):
        # type: (matcher.Match)->None
        value = m[pad].attribs.get('_value', 0.0)
        pad_padding = m[pad].attribs.get('padding', [(0, 0) * m[t].rank])
        sliding_padding = m[sliding].attribs.get('padding', [(0, 0) * m[t].rank])

        if m[sliding].name in ['conv']:
            pad_padding = pad_padding[2:]

        assert len(pad_padding) == len(sliding_padding)

        m[sliding].attribs['padding'] = [(p + pp, q + qq) for (p, q), (pp, qq) in zip(pad_padding, sliding_padding)]
        m[sliding].attribs['border'] = 'ignore' if value == float('-inf') else 'constant'

        graph_utils.remove_passthrough(g, m[pad])

    matcher.for_each(graph=g, pattern=sliding, action=action, condition=condition)

    for op in g.operations:
        if op.name in ['box', 'pad'] and '_value' in op.attribs:
            raise utils.NNEFToolsException('Could not export {} with value={}'.format(op.name, op.attribs['_value']))


def _small_variables_to_consts(g):
    # type: (NNEFGraph)->None

    MaxSize = 4
    MaxNonOneDims = 1

    for tensor in g.tensors:
        if tensor.is_variable and tensor.count <= MaxSize and sum(dim > 1 for dim in tensor.shape) <= MaxNonOneDims:
            tensor.data = tensor.data.flatten().tolist()
            if tensor.dtype == 'integer':
                tensor.data = [utils.anyint_to_int(i) for i in tensor.data]
