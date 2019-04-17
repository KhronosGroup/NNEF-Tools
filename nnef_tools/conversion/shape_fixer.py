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

import typing

from nnef_tools.core import utils
from nnef_tools.core.base_graph import *


def fix_input_shapes(graph, source_shapes):
    # type: (BaseGraph, typing.Union[typing.Dict[str, typing.List[int]], typing.List[int], int, None])->None

    def get_shape_for(name):
        if isinstance(source_shapes, dict) and name in source_shapes:
            return source_shapes[name]
        elif isinstance(source_shapes, list):
            return list(source_shapes)
        elif utils.is_anyint(source_shapes):
            return utils.anyint_to_int(source_shapes)
        return None

    placeholders = [tensor for tensor in graph.tensors
                    if len(tensor.producers) == 0 and not tensor.is_constant and not tensor.is_variable]

    if source_shapes is None:
        if any(tensor.shape is None or -1 in tensor.shape for tensor in placeholders):
            for tensor in placeholders:
                print("Info: Input shape: {}: {}".format(tensor.name, tensor.shape))

    for tensor in placeholders:  # type: BaseTensor
        shape_for_this = get_shape_for(tensor.name) if tensor.name else None
        if isinstance(shape_for_this, list):
            if not utils.compatible_shapes(tensor.shape, shape_for_this):
                raise utils.NNEFToolsException(
                    "The specified shape is incompatible with the original shape for {}. {} vs. {}".format(
                        tensor.name, shape_for_this, tensor.shape))
            tensor.shape = shape_for_this
        elif shape_for_this is None or isinstance(shape_for_this, int):
            if tensor.shape is None:
                raise utils.NNEFToolsException(
                    "The full shape must be specified for {}, because it is unknown.".format(tensor.name))
            elif -1 in tensor.shape:
                if shape_for_this is None:
                    shape_for_this = 1
                    print("Warning: Incomplete input shape is auto-fixed: {}. {} -> {}. "
                          "Use --input-shape if other shape is desired.".format(
                        tensor.name, tensor.shape, [shape_for_this if dim == -1 else dim for dim in tensor.shape]))
                tensor.shape = [shape_for_this if dim == -1 else dim for dim in tensor.shape]
        else:
            assert False

        if tensor.dtype is None:
            raise utils.NNEFToolsException("An input tensor has incomplete dtype, "
                                           "we have thought that this is impossible, "
                                           "please file a bug report to NNEF Tools.")
