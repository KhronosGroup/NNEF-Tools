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

from . import dog, utils
from .name_generator import NameGenerator
from .types import *

if has_typing:
    DataNodeLike = Union[dog.DataNode, bool, int, float]
    LogAnchor = Union[dog.OperationNode, dog.DataNode, str]


class ConverterBase(object):
    def __init__(self,
                 sourcedog,  # type: dog.Graph
                 source_factory,  # type: dog.Factory
                 target_factory,  # type: dog.Factory
                 converters  # type:  Dict[str, Callable[(dog.OperationNode, ConverterBase), None]]
                 ):
        # type: (...)->None

        self.sourcedog = sourcedog  # type: dog.Graph
        self.name_generator = NameGenerator()  # type: NameGenerator

        self.targetdn_by_source_name = OrderedDict()  # type: OrderedDict[str, dog.DataNode]
        self.sourcedn_by_targetdn = OrderedDict()  # type: OrderedDict[dog.DataNode, Optional[dog.DataNode]]
        self.sourceop_by_targetop = OrderedDict()  # type: OrderedDict[dog.OperationNode, Optional[dog.OperationNode]]

        self.source_factory = source_factory  # type: dog.Factory
        self.target_factory = target_factory  # type: dog.Factory
        self.converters = converters

        self.used = False  # type: bool

    def make_targetdn(self, sourcedn=None, name=None, discriminator=None):
        # type: (Optional[dog.DataNode], Optional[str], Optional[str]) -> dog.DataNode

        assert (sourcedn is None) != (name is None)

        if name is None:
            name = sourcedn.name
        if discriminator:
            name = name + "_" + discriminator
        name = self.name_generator.get_new_name(name)

        targetdn = self.target_factory.make_dn(name)
        if sourcedn:
            targetdn.source_name = sourcedn.name
            self.targetdn_by_source_name[sourcedn.name] = targetdn

        self.sourcedn_by_targetdn[targetdn] = sourcedn

        return targetdn

    def get_targetdn(self, sourcednlike):
        # type: (DataNodeLike)->DataNodeLike

        if not isinstance(sourcednlike, dog.DataNode):
            return sourcednlike

        if sourcednlike.name not in self.targetdn_by_source_name:
            self.print_error(sourcednlike, "Corresponding targetdn not defined")
            return self.make_targetdn(sourcedn=sourcednlike)

        return self.targetdn_by_source_name[sourcednlike.name]

    def add_targetop(self, targetop, sourceop=None):
        # type: (dog.OperationNode, Optional[dog.OperationNode])->None

        assert targetop not in self.sourceop_by_targetop
        self.sourceop_by_targetop[targetop] = sourceop

    def add_targetop_ex(
            self,
            sourceop,  # type: dog.OperationNode
            op_name_target,  # type: str
            args=None,  # type: Optional[OrderedDict[str, Any]]
            results=None,  # type: Optional[OrderedDict[str, Any]]
            extra=None  # type: Optional[Dict[str, Any]]
    ):
        # type: (...)->dog.OperationNode

        if args is None:
            args = OrderedDict()
        if results is None:
            results = OrderedDict([(dog.gen_result_name(i), r) for i, r in enumerate(sourceop.results.values())])

        assert isinstance(sourceop, self.source_factory.op_class)
        assert all(isinstance(r, (self.source_factory.dn_class, self.target_factory.dn_class))
                   for r in results.values())

        targetop = self.target_factory.make_op(op_name_target)

        def input_value(val):
            # type: (Any)->Any
            if isinstance(val, self.source_factory.dn_class):
                val = self.get_targetdn(val)
            return val

        def output_value(val):
            # type: (dog.DataNode)->dog.DataNode
            if isinstance(val, self.source_factory.dn_class):
                return self.make_targetdn(sourcedn=val)
            return val

        for k, v in args.items():
            targetop.add_arg(k, utils.recursive_transform(v, input_value))

        for k, v in results.items():
            targetop.add_result(k, utils.recursive_transform(v, output_value))

        if extra:
            targetop.extra.update(extra)

        self.add_targetop(targetop, sourceop=sourceop)
        return targetop

    def convert(self):
        # type: ()->dog.Graph

        assert not self.used
        self.used = True

        for sourceop in self.sourcedog.ops:
            if sourceop.name in self.converters:
                self.converters[sourceop.name](sourceop, self)
            else:
                utils.print_error("No converter for {}".format(sourceop.name))

        def get_name_by_source_name(source_name):
            if source_name in self.targetdn_by_source_name:
                return self.targetdn_by_source_name[source_name].name
            return "<<<ERROR>>>"

        return self.target_factory.make_graph(
            graph_name=self.sourcedog.name,
            ops=list(self.sourceop_by_targetop.keys()),
            dn_by_name={targetdn.name: targetdn for targetdn in self.sourcedn_by_targetdn.keys()},
            input_dn_names=[get_name_by_source_name(name) for name in self.sourcedog.input_dn_names],
            output_dn_names=[get_name_by_source_name(name) for name in self.sourcedog.output_dn_names])

    @staticmethod
    def _print_message(anchor, message, type):
        # type: (LogAnchor, str, str)->None

        print_funs = {"info": utils.print_info,
                      "warning": utils.print_warning,
                      "error": utils.print_error}
        print_fun = print_funs[type]

        if isinstance(anchor, dog.OperationNode):
            print_fun("{}={}(): {}".format(
                ", ".join([dn.name for dn in anchor.get_result_nodes()]),
                anchor.name,
                message
            ))
        elif isinstance(anchor, dog.DataNode):
            print_fun("{}: {}".format(anchor.name, message))
        else:
            print_fun("{}: {}".format(anchor, message))

    @staticmethod
    def print_error(anchor, message):
        # type: (LogAnchor, str)->None
        ConverterBase._print_message(anchor, message, "error")

    @staticmethod
    def print_warning(anchor, message):
        # type: (LogAnchor, str)->None
        ConverterBase._print_message(anchor, message, "warning")

    @staticmethod
    def print_info(anchor, message):
        # type: (LogAnchor, str)->None
        ConverterBase._print_message(anchor, message, "info")
