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

from nnef_tools.core import matcher
from nnef_tools.core import utils
from nnef_tools.io.nnef.nnef_graph import *


def pre_conversion_pass(g):
    # type: (NNEFGraph)->None
    _create_lp_pools(g)


def _create_lp_pools(g):
    # type: (NNEFGraph)->None

    input, abs_out, pow_out, box_out, output, p, q = matcher.tensors(7)

    _abs_op = matcher.Operation(name='abs', inputs=input, outputs=abs_out)
    _pow_op = matcher.Operation(name='pow', inputs=(abs_out, p), outputs=pow_out)
    box_op = matcher.Operation(name='box', inputs=pow_out, outputs=box_out)
    pow2_op = matcher.Operation(name='pow', inputs=(box_out, q), outputs=output)

    matcher.replace(
        g, pow2_op,
        lambda m: NNEFOperation(graph=g,
                                name="_lp_pool",
                                inputs=m[input],
                                outputs=m[output],
                                attribs=utils.dict_union(m[box_op].attribs,
                                                         dict(p=float(m[p].get_numpy_array().item())))),
        lambda m: (m[p].rank == 0 and m[p].data is not None and m[p].get_numpy_array().item() != 0
                   and m[q].rank == 0 and m[q].data is not None and m[p].get_numpy_array().item() != 0
                   and np.allclose(1.0 / m[p].get_numpy_array().item(), m[q].get_numpy_array().item())))
