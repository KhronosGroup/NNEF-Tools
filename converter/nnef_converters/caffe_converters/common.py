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

from ..common import dog
from ..common.types import *

# caffe to nnef
EXTRA_WEIGHTS = "weights"  # value in caffedog: Dict[str, np.ndarray], value in nnefdog: np.ndarray
EXTRA_ACTIVATIONS = "activations"  # value: np.ndarray

# nnef to caffe

# order: [weight/scale/mean/alpha, bias/variance, other]
# value can be str, List[float] or VARIABLE_LABEL_SKIP
EXTRA_VARIABLE_LABELS = "variable_labels"
VARIABLE_LABEL_SKIP = ("skip",)


def get_layer_name(caffeop):
    # type: (dog.OperationNode)->str
    return caffeop.get_result_nodes()[0].name


class CaffeOp(dog.OperationNode):
    pass


class CaffeDN(dog.DataNode):
    pass


class CaffeGraph(dog.Graph):
    pass


caffe_factory = dog.Factory(CaffeGraph, CaffeDN, CaffeOp)

if has_typing:
    CaffeDNLike = Union[CaffeDN, bool, int, float]
