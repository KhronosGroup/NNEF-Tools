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

from nnef_tools.core import graph_utils, utils
from nnef_tools.core import matcher
from nnef_tools.io.caffe.caffe_graph import *
from nnef_tools.io.nnef.nnef_graph import *


def pre_conversion_pass(g):
    # type: (NNEFGraph)->None
    graph_utils.remove_unreachable(g)
    _create_thresholds(g)
    _create_elus(g)
    graph_utils.remove_unreachable(g)


def post_conversion_pass(g):
    # type: (CaffeGraph)->None
    graph_utils.remove_unreachable(g)
    _unite_powers(g)
    _merge_up_bias(g)
    graph_utils.remove_unreachable(g)


def _create_thresholds(g):
    # type: (NNEFGraph)->None

    input, threshold, gt_output, one, zero, output = matcher.tensors(6)
    _gt_op = matcher.Operation(name="gt",
                               inputs=(input, threshold),
                               outputs=gt_output)
    select_op = matcher.Operation(
        name="select",
        inputs=(gt_output, one, zero),
        outputs=output)

    def condition(m):
        return (m[output].shape == m[input].shape
                and m[threshold].data is not None
                and m[one].data is not None
                and m[zero].data is not None
                and m[threshold].rank == 0
                and np.all(m[one].get_numpy_array() == 1.0)
                and np.all(m[zero].get_numpy_array() == 0.0))

    def replacement(m):
        NNEFOperation(graph=g,
                      name="_threshold",
                      inputs=m[input],
                      outputs=m[output],
                      attribs=dict(threshold=float(m[threshold].get_numpy_array())))

    matcher.replace(g, select_op, replacement, condition)


def _create_elus(g):
    # type: (NNEFGraph)->None

    input, gt_out, exp_out, sub_out, mul_out, output, zero, one, alpha = matcher.tensors(9)

    _gt = matcher.Operation(name='gt', inputs=(input, zero), outputs=gt_out)
    _exp = matcher.Operation(name='exp', inputs=input, outputs=exp_out)
    _sub = matcher.Operation(name='sub', inputs=(exp_out, one), outputs=sub_out)
    _mul = matcher.Operation(name='mul', inputs=(sub_out, alpha), outputs=mul_out)
    select = matcher.Operation(name='select', inputs=(gt_out, input, mul_out), outputs=output)

    matcher.replace(
        g, select,
        lambda m: NNEFOperation(graph=g,
                                name="elu",
                                inputs=m[input],
                                outputs=m[output],
                                attribs=dict(_alpha=float(m[alpha].data[0]))),
        lambda m: m[zero].data == [0.0] and m[one].data == [1.0] and m[alpha].rank == 0 and m[alpha].data is not None)


def _unite_powers(g):
    # type: (CaffeGraph)->None

    input, scaled, shifted, powered = matcher.tensors(4)

    scale_op = matcher.Operation(name=['Power'], inputs=input, outputs=scaled)
    shift_op = matcher.Operation(name=['Power'], inputs=scaled, outputs=shifted)
    power_op = matcher.Operation(name=['Power', 'Exp'], inputs=shifted, outputs=powered)

    def condition(m):
        return (list(m[scale_op].attribs.keys()) == ['scale']
                and list(m[shift_op].attribs.keys()) == ['shift']
                and list(m[power_op].attribs.keys()) in (['power'], ['base']))

    def replacement(m):
        CaffeOperation(graph=g,
                       name=m[power_op].name,
                       inputs=m[input],
                       outputs=m[powered],
                       attribs=utils.dict_union(m[scale_op].attribs, m[shift_op].attribs, m[power_op].attribs))

    matcher.replace(g, power_op, replacement, condition)

    input, output1, output2 = matcher.tensors(3)

    op1 = matcher.Operation(name='Power', inputs=input, outputs=output1)
    op2 = matcher.Operation(name=['Power', 'Exp'], inputs=output1, outputs=output2)

    def condition(m):
        return ((list(m[op2].attribs.keys()) in (['power'], ['base'])
                 and list(m[op1].attribs.keys()) == ['shift'])
                or (list(m[op2].attribs.keys()) in (['power'], ['base'])
                    and list(m[op1].attribs.keys()) == ['scale'])
                or (list(m[op2].attribs.keys()) == ['shift']
                    and list(m[op1].attribs.keys()) == ['scale']))

    def replacement(m):
        CaffeOperation(graph=g,
                       name=m[op2].name,
                       inputs=m[input],
                       outputs=m[output2],
                       attribs=utils.dict_union(m[op2].attribs, m[op1].attribs))

    matcher.replace(g, op2, replacement, condition)


def _merge_up_bias(g):
    # type: (CaffeGraph)->None

    input, scale, scaled, bias, output = matcher.tensors(5)

    scale_op = matcher.Operation(name=['Scale', 'InnerProduct'], inputs=(input, scale), outputs=scaled)
    bias_op = matcher.Operation(name='Bias', inputs=(scaled, bias), outputs=output)

    matcher.replace(g, bias_op, lambda m: CaffeOperation(graph=g,
                                                         name=m[scale_op].name,
                                                         inputs=(m[input], m[scale], m[bias]),
                                                         outputs=m[output],
                                                         attribs=utils.updated_dict(m[scale_op].attribs,
                                                                                    bias_term=True)))
