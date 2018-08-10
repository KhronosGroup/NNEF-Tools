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

from collections import OrderedDict, deque

import numpy as np

from . import utils
from .name_generator import NameGenerator
from .types import *

EXTRA_COMMENT = "comment"
EXTRA_WEIGHTS = "weights"
EXTRA_HEATMAP = "heatmap"


class DataNode(object):
    def __init__(self, name=None):
        self.name = name  # type: str
        # name in the source framework
        self.source_name = None  # type: str
        self.shape = None  # type: List[int]
        self.dtype = None  # type: str
        self.producer = None  # type: OperationNode
        self.consumers = []  # type: List[OperationNode]
        self.extra = {}  # type: Dict[str, Any]

    def __repr__(self):
        return "DN#{}: {}".format(id(self), self.name)


class OperationNode(object):
    def __init__(self, name=None):
        self.name = name  # type: str
        self.args = OrderedDict()  # type: OrderedDict[str, Any]
        self.results = OrderedDict()  # type: OrderedDict[str, Any]
        self.extra = {}  # type: Dict[str, Any]

    def add_arg(self, arg_name, arg):

        def transform_new(a):
            if isinstance(a, DataNode) and self not in a.consumers:
                a.consumers.append(self)
            return a

        def transform_old(a):
            if (isinstance(a, DataNode)
                    and self in a.consumers
                    and not utils.recursive_contains_by_pointer(self.args, a)):
                a.consumers.remove(self)
            return a

        old_value = self.args.get(arg_name)

        self.args[arg_name] = utils.recursive_transform(arg, transform_new)

        utils.recursive_transform(old_value, transform_old)

    def set_arg(self, arg_name, arg):
        self.add_arg(arg_name, arg)

    def remove_arg(self, arg_name):
        self.add_arg(arg_name, None)
        del self.args[arg_name]

    def add_args(self, args):
        # type: (OrderedDict)->None
        for k, v in args.items():
            self.add_arg(k, v)

    def remove_args(self):
        # type: ()->None
        for arg_name in list(self.args.keys()):
            self.remove_arg(arg_name)

    def remove_results(self):
        # type: ()->None
        for result_name in list(self.results.keys()):
            self.remove_result(result_name)

    def unlink(self):
        # type: ()->None
        self.remove_args()
        self.remove_results()

    def add_result(self, result_name, result, overwrite_producer=True):

        def transform_new(r):
            if isinstance(r, DataNode) and (r.producer is None or overwrite_producer):
                r.producer = self
            return r

        def transform_old(r):
            if (isinstance(r, DataNode)
                    and r.producer is self
                    and not utils.recursive_contains_by_pointer(self.results, r)):
                r.producer = None
            return r

        old_value = self.results.get(result_name)

        self.results[result_name] = utils.recursive_transform(result, transform_new)

        utils.recursive_transform(old_value, transform_old)

    def set_result(self, result_name, result, overwrite_producer=True):
        self.add_result(result_name, result, overwrite_producer=overwrite_producer)

    def remove_result(self, result_name):
        self.add_result(result_name, None)
        del self.results[result_name]

    @property
    def result(self):
        assert len(self.results) == 1
        return list(self.results.values())[0]

    @property
    def result_node(self):
        # type: ()->DataNode
        assert len(self.results) == 1
        result = list(self.results.values())[0]
        assert isinstance(result, DataNode)
        return result

    def get_arg_nodes(self, except_args=None):
        if except_args is None:
            except_args = []
        return self._get_data_nodes(self.args, except_args)

    def get_result_nodes(self, except_args=None):
        if except_args is None:
            except_args = []
        return self._get_data_nodes(self.results, except_args)

    @staticmethod
    def _get_data_nodes(dict_, except_args):
        data_nodes = []

        for arg_name, arg in dict_.items():
            if arg_name in except_args:
                continue
            arg_elems = arg if isinstance(arg, (list, tuple)) else (arg,)
            for arg_elem in arg_elems:
                if isinstance(arg_elem, DataNode):
                    data_nodes.append(arg_elem)

        return data_nodes

    def get_nth_arg(self, n):
        return list(self.args.values())[n]

    def get_nth_result(self, n):
        return list(self.results.values())[n]

    def get_result_name(self, data_node):
        for k, v in self.results.items():
            if v is data_node:
                return k
        return None

    def num_gen_args(self):
        return len([key for key in self.args.keys() if key.startswith("arg")])

    def num_gen_results(self):
        return len([key for key in self.results.keys() if key.startswith("result")])

    def __repr__(self):
        return "OP#{}: {}".format(id(self), self.name)


class Graph(object):
    def __init__(self, graph_name, ops, dn_by_name, input_dn_names, output_dn_names):
        # type: (str, List[OperationNode], Dict[str, DataNode], List[str], List[str])->None

        self.name = graph_name  # type: str
        self.ops = ops  # type: List[OperationNode]
        self.dn_by_name = dn_by_name  # type: Dict[str, DataNode]
        self.input_dn_names = input_dn_names  # type: List[str]
        self.output_dn_names = output_dn_names  # type: List[str]
        self.extra = {}  # type: Dict[str, Any]
        self.name_generator = NameGenerator(used_names=list(dn_by_name.keys()))

    def debug_print(self):
        def transform(data):
            if isinstance(data, DataNode):
                return data.name
            elif isinstance(data, np.ndarray):
                return data.shape
            else:
                return data

        print("dog", self.name,
              "inputs", self.input_dn_names,
              "outputs", self.output_dn_names)

        for op in self.ops:
            print(utils.recursive_transform(op.results, transform),
                  "=", op.name, utils.recursive_transform(op.args, transform),
                  "extra", utils.recursive_transform(op.extra, transform))

        for dn in self.dn_by_name.values():
            print(dn.name, "extra", utils.recursive_transform(dn.extra, transform))

        print()

    def remove_ops(self, ops_to_remove):
        # type: (Set[OperationNode])->None

        dn_names_to_remove = set(dn.name for dn in self.dn_by_name.values() if dn.producer in ops_to_remove)

        self.ops = [op for op in self.ops if op not in ops_to_remove]
        self.dn_by_name = {name: dn for name, dn in self.dn_by_name.items() if dn.producer not in ops_to_remove}
        self.input_dn_names = [name for name in self.input_dn_names if name not in dn_names_to_remove]
        self.output_dn_names = [name for name in self.output_dn_names if name not in dn_names_to_remove]

    # You have to also remove it later
    def unlink_passthrough(self, op, dn_arg, dn_result):
        # type: (OperationNode, DataNode, DataNode)->None

        op.unlink()

        if dn_result.name in self.output_dn_names:
            if dn_arg.name in self.output_dn_names:
                self.output_dn_names.remove(op.result_node.name)
            else:
                def replace(name):
                    if name == dn_result.name:
                        return dn_arg.name
                    return name

                self.output_dn_names = utils.recursive_transform(self.output_dn_names, replace)

        for consumer in dn_result.consumers:
            def replace(x):
                if x is dn_result:
                    return dn_arg
                return x

            consumer.args = utils.recursive_transform(consumer.args, replace)
            consumer.results = utils.recursive_transform(consumer.results, replace)  # normally it is not needed
            dn_arg.consumers.append(consumer)

    def remove_unreachables(self):
        # type: ()->None
        visited_ops = set()

        q = deque()

        output_ops = []
        for op in self.ops:
            if any([result.name in self.output_dn_names for result in op.get_result_nodes()]):
                output_ops.append(op)

        for op in utils.unique(output_ops):
            q.append(op)
            visited_ops.add(op)

        while q:
            op = q.popleft()

            for dn in op.get_arg_nodes():
                if dn.producer and dn.producer not in visited_ops:
                    visited_ops.add(dn.producer)
                    q.append(dn.producer)

        self.remove_ops(set([op for op in self.ops if op not in visited_ops]))


def get_shape_safe(dn_or_other):
    if isinstance(dn_or_other, DataNode):
        return dn_or_other.shape
    elif isinstance(dn_or_other, np.ndarray):
        return list(dn_or_other.shape)
    elif isinstance(dn_or_other, (list, tuple)):
        return list(np.array(dn_or_other).shape)
    else:
        return []


def get_rank_safe(dn_or_other):
    return len(get_shape_safe(dn_or_other))


def get_extra_safe(dn_or_other):
    if isinstance(dn_or_other, DataNode):
        return dn_or_other.extra
    else:
        return {}


def get_name_safe(dn_or_other):
    if isinstance(dn_or_other, DataNode):
        return dn_or_other.extra
    else:
        return "literal({})".format(dn_or_other)


def get_dummy_dn():
    dummy_op = OperationNode("<<<ERROR>>>")
    dummy_dn = DataNode("<<<ERROR>>>")
    dummy_dn.shape = []
    dummy_op.add_result("result", dummy_dn)
    return dummy_dn


def gen_arg_name(i=0):
    return "arg{}".format(i)


def gen_result_name(i=0):
    return "result{}".format(i)


class Factory(object):
    def __init__(self, graph_class, dn_class, op_class):
        # type: (type, type, type)->None
        self.graph_class = graph_class
        self.dn_class = dn_class
        self.op_class = op_class

    def make_graph(self, graph_name, ops, dn_by_name, input_dn_names, output_dn_names):
        # type: (str, List[OperationNode], Dict[str, DataNode], List[str], List[str])->Graph
        return self.graph_class(graph_name, ops, dn_by_name, input_dn_names, output_dn_names)

    def make_dn(self, name=None):
        # type: (Optional[str])->DataNode
        return self.dn_class(name)

    def make_op(self, name=None):
        # type: (Optional[str])->OperationNode
        return self.op_class(name)


if has_typing:
    DataNodeLike = Union[DataNode, bool, int, float]
