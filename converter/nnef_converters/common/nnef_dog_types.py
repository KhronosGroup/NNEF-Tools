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
from .name_generator import NameGenerator
from .types import *


# Remark: The typed graphs might be removed in the future,
# and just dog.Graph/dog.OperationNode/dog.DataNode will remain

class NnefOp(dog.OperationNode):
    pass


class NnefDN(dog.DataNode):
    pass


class NnefGraph(dog.Graph):
    def __init__(self, graph_name, ops, dn_by_name, input_dn_names, output_dn_names):
        # type: (str, List[dog.OperationNode], Dict[str, dog.DataNode], List[str], List[str])->None
        super(NnefGraph, self).__init__(graph_name, ops, dn_by_name, input_dn_names, output_dn_names)
        self.label_name_generator = NameGenerator(used_names=[op.args["label"]
                                                              for op in self.ops
                                                              if op.name == "variable"])


nnef_factory = dog.Factory(NnefGraph, NnefDN, NnefOp)

if has_typing:
    NnefDNLike = Union[NnefDN, bool, int, float]
else:
    NnefDNLike = object
