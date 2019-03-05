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

from __future__ import division, print_function

from collections import OrderedDict

import nnef

from . import dog
from . import utils
from .nnef_dog_types import NnefDN, NnefOp, NnefGraph


def nnefgraph_to_nnefdog(nnefgraph, with_weights=True):
    ops = []
    dn_by_name = {}

    def transform_arg(arg, op):
        if isinstance(arg, nnef.Identifier):
            dn = dn_by_name.get(str(arg))
            if dn is None:
                utils.print_error("DataNode {} not defined before use".format(str(arg)))
                return utils.REMOVE
            if op not in dn.consumers:  # can be multiple times, eg: matmul(a, a)
                dn.consumers.append(op)
            return dn
        else:
            return arg

    def transform_result(result, op):

        if isinstance(result, nnef.Identifier):
            dn = NnefDN(str(result))
            dn.shape = list(nnefgraph.tensors[str(result)].shape)
            dn.dtype = nnefgraph.tensors[str(result)].dtype
            dn.producer = op
            if dn.name in dn_by_name:
                utils.print_error("DataNode {} defined multiple times".format(dn.name))
                return utils.REMOVE
            dn_by_name[dn.name] = dn
            return dn
        else:
            return result

    def transform_tensor_to_dn(tensor):
        dn = dn_by_name.get(str(tensor))
        if dn is None:
            utils.print_error("DataNode {} not defined before use".format(str(tensor)))
            return utils.REMOVE
        return dn

    for nnefop in nnefgraph.operations:
        op = NnefOp(nnefop.name)

        args = OrderedDict()
        args.update(nnefop.inputs)
        args.update(nnefop.attribs)

        op.args = utils.recursive_transform(args, lambda arg: transform_arg(arg, op))
        op.results = utils.recursive_transform(nnefop.outputs, lambda result: transform_result(result, op))
        ops.append(op)

        if with_weights and op.name == "variable":
            op.result_node.extra[dog.EXTRA_WEIGHTS] = nnefgraph.tensors[nnefop.outputs["output"]].data
            assert op.result_node.extra[dog.EXTRA_WEIGHTS] is not None

    input_dn_names = [dn.name for dn in utils.recursive_transform(nnefgraph.inputs, transform_tensor_to_dn)]
    output_dn_names = [dn.name for dn in utils.recursive_transform(nnefgraph.outputs, transform_tensor_to_dn)]

    return NnefGraph(nnefgraph.name, ops, dn_by_name, input_dn_names, output_dn_names)
