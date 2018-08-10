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

import re

import numpy as np

from ..common import CaffeDN, CaffeGraph
from ..common import EXTRA_WEIGHTS, EXTRA_ACTIVATIONS
from ...common import dog
from ...common import utils
from ...common.matchers import match_arg_chain, Op, OpAndArg
from ...common.name_generator import NameGenerator
from ...common.nnef_dog_types import NnefGraph
from ...common.types import *


def _fix_name(name, name_gen=None):
    # type: (AnyStr, Optional[NameGenerator])->str
    fixed = utils.ensure_not_unicode_in_python2(name)
    fixed = re.sub(r'\W', '_', fixed)
    if name_gen:
        return name_gen.get_new_name(fixed)
    return fixed


def fix_names(caffedog):
    # type: (CaffeGraph)->None
    caffedog.name = _fix_name(caffedog.name)

    name_gen = NameGenerator()
    fixed_dnname_by_dnname = {}  # type: Dict[str, str]
    for dn in caffedog.dn_by_name.values():
        fixed = _fix_name(dn.name, name_gen)
        fixed_dnname_by_dnname[dn.name] = fixed
        dn.name = fixed
    caffedog.dn_by_name = {dn.name: dn for dn in caffedog.dn_by_name.values()}
    caffedog.input_dn_names = [fixed_dnname_by_dnname[dnname] for dnname in caffedog.input_dn_names]
    caffedog.output_dn_names = [fixed_dnname_by_dnname[dnname] for dnname in caffedog.output_dn_names]

    name_gen = NameGenerator()
    for op in caffedog.ops:
        op.set_arg("name", _fix_name(op.args["name"], name_gen))
        op.name = _fix_name(op.name)


def resolve_inplace(caffedog):
    # type: (CaffeGraph)->None
    for i, op in enumerate(caffedog.ops):
        for result_name, result in list(op.results.items()):
            if not isinstance(result, CaffeDN):
                continue
            if result in op.get_arg_nodes():
                new_result = CaffeDN(caffedog.name_generator.get_new_name(result.name))
                new_result.shape = result.shape
                if EXTRA_ACTIVATIONS in result.extra:
                    new_result.extra[EXTRA_ACTIVATIONS] = result.extra[EXTRA_ACTIVATIONS]
                    result.extra[EXTRA_ACTIVATIONS] = None
                op.set_result(result_name, new_result)
                caffedog.dn_by_name[new_result.name] = new_result
                for op2 in caffedog.ops[i + 1:]:
                    for arg2_name, arg2 in op2.args.items():
                        if arg2 is result:
                            op2.set_arg(arg2_name, new_result)
                    for result2_name, result2 in op2.results.items():
                        if result2 is result:
                            op2.set_result(result2_name, new_result, overwrite_producer=False)
    caffedog.output_dn_names = [dn.name for dn in caffedog.dn_by_name.values() if not dn.consumers]


def merge_batch_norm_and_scale(caffedog):
    # type: (CaffeGraph)->None
    matches = []
    for op in caffedog.ops:
        match = match_arg_chain(op, [
            OpAndArg("Scale", dog.gen_arg_name(0)),
            Op("BatchNorm")
        ], output_dn_names=caffedog.output_dn_names)
        if match:
            scale, _, batch_norm = match
            if scale.args["bias_term"] and scale.args["axis"] == 1 and scale.args["num_axes"] == 1:
                matches.append((scale, batch_norm))
    to_remove = set()
    for scale, batch_norm in matches:
        scale_weights = scale.extra[EXTRA_WEIGHTS]
        scale_result = scale.result_node
        scale.unlink()
        to_remove.add(scale)

        batch_norm.name = "_ScaledBatchNorm"
        batch_norm.set_result(dog.gen_result_name(0), scale_result)
        batch_norm.extra[EXTRA_WEIGHTS].update(scale_weights)
    caffedog.remove_ops(to_remove)


def batch_norm_to_scale(caffedog):
    # type: (CaffeGraph)->None
    for op in caffedog.ops:
        if op.name == "BatchNorm":
            weights = op.extra[EXTRA_WEIGHTS]
            eps = op.args["eps"]

            if weights["scale_factor"].shape != (1,):
                utils.print_error("{}: scale_factor.shape must be [1]".format(op.args["name"]))

            scale_factor = weights["scale_factor"][0]
            norm = 0.0 if scale_factor == 0.0 else (1.0 / scale_factor)

            var_normed = weights["variance"] * norm + eps
            mean_normed = weights["mean"] * norm

            weight = 1.0 / np.sqrt(var_normed)
            bias = - mean_normed / np.sqrt(var_normed)

            op.name = "Scale"
            op.remove_arg("eps")
            op.add_arg("bias_term", True)
            op.extra[EXTRA_WEIGHTS] = {
                "weight": weight,
                "bias": bias
            }


def remove_passthroughs(caffedog):
    # type: (CaffeGraph)->None
    passthroughs = {"Dropout"}
    to_remove = set()
    for i, op in enumerate(caffedog.ops):
        if op.name in passthroughs:
            caffedog.unlink_passthrough(op, dn_arg=op.args[dog.gen_arg_name(0)], dn_result=op.result_node)
            to_remove.add(op)
    caffedog.remove_ops(to_remove)


def reorder(nnefdog):
    # type: (NnefGraph)->None
    externals = []
    variables = []
    others = []
    for op in nnefdog.ops:
        if op.name == "external":
            externals.append(op)
        elif op.name == "variable":
            variables.append(op)
        else:
            others.append(op)
    nnefdog.ops = externals + variables + others
