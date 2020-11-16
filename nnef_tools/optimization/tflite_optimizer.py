# Copyright (c) 2020 The Khronos Group Inc.
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
from ..model.graph import *


class Optimizer:

    def __init__(self, keep_io_names=False, custom_optimizers=None):
        self._keep_io_names = keep_io_names
        self._custom_optimizers = custom_optimizers or {}

    def __call__(self, graph, only_required=False):
        replace_chain(graph, ['RESHAPE', 'RESHAPE', 'PACK', 'PACK', 'RESHAPE'], self._replace_resize_nearest)
        for chain, replacer in six.iteritems(self._custom_optimizers):
            replace_chain(graph, chain, replacer)
        return graph

    @staticmethod
    def _replace_resize_nearest(reshape1, reshape2, pack1, pack2, reshape3):
        def _all_inputs_same(op):
            return len(op.inputs) > 0 and all(tensor is op.inputs[0] for tensor in op.inputs)

        if not (reshape2.output.shape == reshape1.inputs[0].shape and
                _all_inputs_same(pack1) and _all_inputs_same(pack2) and
                len(pack1.inputs) == len(pack2.inputs)):
            return False

        input = reshape1.inputs[0]
        output = reshape3.output
        size = Tensor(input.graph, shape=(len(output.shape) - 2,), dtype=np.int32, data=np.array(output.shape[1:-1]),
                      name=output.name + '/size')
        Operation(input.graph,
                  type='RESIZE_NEAREST_NEIGHBOR',
                  inputs=(input, size),
                  outputs=output,
                  attribs={
                      'align_corners': False,
                      'half_pixel_centers': False,
                  })
