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

from .dog_to_caffe import EXTRA_CAFFE_PARAM_NAME
from ..common import EXTRA_VARIABLE_LABELS
from ...common import dog
from ...common.matchers import Op, OpAndArg, OneOf, match_arg_chain
from ...common.nnef_dog_types import NnefGraph

EXTRA_MERGED_POWER = "merged_power"


def constant_to_variable(nnefdog):
    # type: (NnefGraph)->None
    for op in nnefdog.ops:
        if op.name == "constant":
            op.name = "variable"
            op.add_arg("label", op.args["value"])
            op.remove_arg("value")


def unite_powers(caffedog):
    # type: (CaffeGraph)->None

    while True:
        matches = []

        for op in caffedog.ops:
            match = match_arg_chain(op, [
                OpAndArg("Power", dog.gen_arg_name(0)),
                OpAndArg("Power", dog.gen_arg_name(0)),
                Op("Power")
            ], output_dn_names=caffedog.output_dn_names)
            if match:
                power1, _, power2, _, power3 = match
                if not any(c.name == "Power" and not c.extra.get(EXTRA_MERGED_POWER)
                           for c in power1.result_node.consumers):
                    if power1.args.keys() == [dog.gen_arg_name(0), "power"]:
                        if power2.args.keys() == [dog.gen_arg_name(0), "shift"]:
                            if power3.args.keys() == [dog.gen_arg_name(0), "scale"]:
                                matches.append((power1, power2, power3))
                                continue

            match = match_arg_chain(op, [
                OpAndArg("Power", dog.gen_arg_name(0)),
                Op("Power")
            ], output_dn_names=caffedog.output_dn_names)

            if match:
                power1, _, power2 = match
                if not any(c.name == "Power" and not c.extra.get(EXTRA_MERGED_POWER)
                           for c in power1.result_node.consumers):
                    if power1.args.keys() == [dog.gen_arg_name(0), "power"]:
                        if power2.args.keys() == [dog.gen_arg_name(0), "shift"]:
                            matches.append((power1, power2))
                            continue
                    if power1.args.keys() == [dog.gen_arg_name(0), "power"]:
                        if power2.args.keys() == [dog.gen_arg_name(0), "scale"]:
                            matches.append((power1, power2))
                            continue
                    if power1.args.keys() == [dog.gen_arg_name(0), "shift"]:
                        if power2.args.keys() == [dog.gen_arg_name(0), "scale"]:
                            matches.append((power1, power2))
                            continue

        to_remove = set()
        for powers in matches:
            args = OrderedDict([(dog.gen_arg_name(0), powers[-1].args[dog.gen_arg_name(0)])])
            for power in powers:
                power.remove_arg(dog.gen_arg_name(0))
                args.update(power.args)
                power.remove_args()
            powers[0].add_args(args)
            powers[0].extra[EXTRA_MERGED_POWER] = True
            to_remove.update(list(powers[1:]))

        if not to_remove:
            break

        caffedog.remove_ops(to_remove)


def merge_up_bias(caffedog):
    # type: (CaffeGraph)->None

    matches = []

    for op in caffedog.ops:
        match = match_arg_chain(op, [
            OpAndArg("Bias", dog.gen_arg_name(0)),
            Op(OneOf("Scale", "InnerProduct"))
        ], output_dn_names=caffedog.output_dn_names)

        if match:
            bias, _, other = match
            if not other.args.get("bias_term", False):
                matches.append((bias, other))

    to_remove = set()
    for bias, other in matches:
        other_name = other.name
        other_args = OrderedDict(other.args)
        other_args["bias_term"] = True
        other_var_labels = other.extra[EXTRA_VARIABLE_LABELS]
        other_param_name = other.extra.get(EXTRA_CAFFE_PARAM_NAME, None)
        other.unlink()
        to_remove.add(other)

        bias.name = other_name
        bias.remove_args()
        bias.add_args(other_args)
        bias.extra[EXTRA_VARIABLE_LABELS] = other_var_labels + bias.extra[EXTRA_VARIABLE_LABELS]
        if other_param_name:
            bias.extra[EXTRA_CAFFE_PARAM_NAME] = other_param_name

    caffedog.remove_ops(to_remove)
