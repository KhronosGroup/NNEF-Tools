# Copyright (c) 2017-2025 The Khronos Group Inc.
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

from ..model.utils import replace_chain
from ..model import Operation
import numpy as np
import six


class Optimizer:

    _QuantDtype = {np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32}

    def __init__(self, custom_optimizers=None):
        self._custom_optimizers = custom_optimizers or {}

    def __call__(self, model, only_required=False):
        for graph in model.graphs:
            self._fix_batchnorm_spatial(graph)
            replace_chain(graph, ['QuantizeLinear', 'DequantizeLinear'], self._fuse_quantize_linear)
            for chain, replacer in six.iteritems(self._custom_optimizers):
                replace_chain(graph, chain, replacer)

    @staticmethod
    def _fuse_quantize_linear(quant, dequant):
        if not Optimizer._same_constant(quant.inputs[1], dequant.inputs[1]):
            return False
        if not Optimizer._same_zero_point(quant, dequant):
            return False
        if quant.attribs.get('axis', 1) != dequant.attribs.get('axis', 1):
            return False
        zero_point = Optimizer._zero_point(quant)
        dtype = zero_point.dtype if zero_point is not None else np.uint8
        if dtype not in Optimizer._QuantDtype:
            return False

        output = dequant.detach_output()
        Operation(quant.graph, type='FusedQuantizeLinear', name=quant.name or dequant.name,
                  inputs=quant.inputs, outputs=output,
                  attribs={'axis': quant.attribs.get('axis', 1)})

    @staticmethod
    def _zero_point(op):
        if len(op.inputs) > 2 and op.inputs[2].name != '':
            return op.inputs[2]
        return None

    @staticmethod
    def _same_zero_point(quant, dequant):
        left = Optimizer._zero_point(quant)
        right = Optimizer._zero_point(dequant)
        if left is None and right is None:
            return True
        if left is None or right is None:
            present = left if left is not None else right
            return present.data is not None and np.all(np.asarray(present.data) == 0)
        return Optimizer._same_constant(left, right)

    @staticmethod
    def _same_constant(left, right):
        if left is right:
            return True
        if left.data is None or right.data is None:
            return False
        return np.array_equal(np.asarray(left.data), np.asarray(right.data))

    @staticmethod
    def _fix_batchnorm_spatial(graph):
        for op in graph.operations:
            if op.type == 'BatchNormalization':
                spatial = op.attribs.get('spatial')
                if spatial == 0 and op.inputs[1].rank == 1:
                    del op.attribs['spatial']
