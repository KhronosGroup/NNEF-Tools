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

from . import dog
from . import utils
from .types import *


class OneOf(object):
    def __init__(self, *args):
        # type: (Union[Op, OpAndArg, str, Callable], ...)->None

        self.values = args


class OpAndArg(object):
    def __init__(self, op, arg):
        # type: (Union[OneOf, str, Callable], Union[OneOf, str])->None

        self.op = op
        self.arg = arg


class Op(object):
    def __init__(self, op):
        # type: (Union[OneOf, str, Callable])->None

        self.op = op


def match_arg_chain(
        op,  # type: dog.OperationNode
        matchers,  # type: List[Union[OneOf, Op, OpAndArg]]
        only_single_consumer=True,  # type: bool
        output_dn_names=None,  # type: Optional[List[str]]
        tf_hacks=False,  # type: bool
):
    # type: (...)->Optional[Tuple[Union[dog.OperationNode, str], ...]]
    # returns None or ([op: dog.OperationNode, arg_name: str]*, [op: dog.OperationNode])
    # if output_dn_names is set and any results of the returned ops (except the first op), are outputs, we return None

    result = _match_arg_chain_impl(op, _matchers_to_names(matchers, tf_hacks=tf_hacks),
                                   only_single_consumer=only_single_consumer, tf_hacks=tf_hacks)
    if not result or not output_dn_names:
        return result

    for r in result[1:]:
        if isinstance(r, dog.OperationNode):
            for dn in r.get_result_nodes():
                if dn.name in output_dn_names:
                    return None

    return result


def _matchers_to_names(matchers, tf_hacks=False):
    names = []
    for matcher in matchers:
        matchers2 = list(matcher.values) if isinstance(matcher, OneOf) else [matcher]
        op_names = []
        arg_name_lists = []
        for matcher2 in matchers2:
            if isinstance(matcher2, (Op, OpAndArg)):
                op_matchers = list(matcher2.op.values) if isinstance(matcher2.op, OneOf) else [matcher2.op]
                for op_matcher in op_matchers:
                    if tf_hacks and not isinstance(op_matcher, str):
                        op_name = utils.get_qualified_name(op_matcher)
                    else:
                        op_name = op_matcher
                    op_names.append(op_name)
                if isinstance(matcher2, OpAndArg):
                    arg_names = list(matcher2.arg.values) if isinstance(matcher2.arg, OneOf) else [matcher2.arg]
                    arg_name_lists = arg_name_lists + len(op_matchers) * [arg_names]
            else:
                assert False, "Expected Op or OpAndArg"
        if len(op_names) > 0:
            names.append(op_names)
        if len(arg_name_lists) > 0:
            names.append(arg_name_lists)
    return names


def _match_arg_chain_impl(op, names, only_single_consumer=True, tf_hacks=False):
    if len(names) == 0:
        return ()
    name_list = names[0] if isinstance(names[0], list) else [names[0]]
    if op is not None and op.name in name_list:
        name_idx = name_list.index(op.name)
    else:
        return None
    if len(names) >= 2:
        name_list = names[1] if isinstance(names[1], list) else [names[1]]
        name_list = name_list[name_idx] if len(name_list) != 1 else name_list[0]
        name_list = name_list if isinstance(name_list, list) else [name_list]
        for arg_name in name_list:
            if arg_name in op.args:
                dn_arg = op.args[arg_name]
                if isinstance(dn_arg, dog.DataNode):
                    is_single_consumer = True
                    for consumer_op in dn_arg.consumers:
                        if consumer_op != op:
                            if tf_hacks:
                                from .. tf_converters.tf_to_nnef import tf_to_dog
                                if consumer_op.extra.get(tf_to_dog.EXTRA_NESTING_LEVEL, -1) == 0:
                                    is_single_consumer = False
                            else:
                                is_single_consumer = False
                    if is_single_consumer or not only_single_consumer:
                        res = _match_arg_chain_impl(dn_arg.producer, names[2:], only_single_consumer)
                        if res is not None:
                            return (op, arg_name) + res
        return None
    else:
        return op,
