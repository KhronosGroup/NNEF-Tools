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

from nnef_tools.io.nnef.nnef_graph import NNEFGraph, NNEFOperation, NNEFTensor


def pre_conversion_pass(g):
    # type: (NNEFGraph)->None
    _transform_extract_bias_add(g)
    g.generate_missing_names()
    g.assert_consistent()


def _transform_extract_bias_add(g):
    # type:(NNEFGraph)->None

    supported_ops = {"conv", "deconv"}

    for nnefop in list(g.operations):
        if nnefop.name in supported_ops and len(nnefop.inputs) >= 3:
            bias = nnefop.inputs[2]
            nnefop.inputs = tuple(nnefop.inputs)[:2]

            if not (bias.is_constant and bias.data == [0]):
                output_with_bias = nnefop.output
                output_without_bias = NNEFTensor(graph=g,
                                                 name=None,
                                                 dtype=output_with_bias.dtype,
                                                 shape=output_with_bias.shape)
                nnefop.outputs = output_without_bias

                NNEFOperation(graph=g,
                              name="_bias_add",
                              inputs=(output_without_bias, bias),
                              outputs=output_with_bias)
