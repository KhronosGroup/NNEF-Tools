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

from ..model.utils import replace_chain, bypass_and_remove
from ..model.graph import *
from ..utils.types import from_numpy


class Optimizer:

    def __init__(self, custom_optimizers=None):
        self._custom_optimizers = custom_optimizers or {}

    def __call__(self, graph, only_required=False):
        Optimizer._eliminate_variable_dequantize_ops(graph)
        replace_chain(graph, ['SPACE_TO_BATCH_ND', {'CONV_2D', 'DEPTHWISE_CONV_2D'}, 'BATCH_TO_SPACE_ND'], self._replace_dilated_conv)
        replace_chain(graph, ['RESHAPE', 'RESHAPE', 'PACK', 'PACK', 'RESHAPE'], self._replace_resize_nearest)
        replace_chain(graph, ['SHAPE'], self._replace_const_shape)
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

    @staticmethod
    def _replace_dilated_conv(space_to_batch, conv, batch_to_space):
        if not Optimizer._is_constant(space_to_batch.inputs[1]) or not Optimizer._is_constant(batch_to_space.inputs[1]):
            return False

        block_shape1 = Optimizer._read_constant(space_to_batch.inputs[1])
        block_shape2 = Optimizer._read_constant(batch_to_space.inputs[1])

        if not np.all(block_shape1 == block_shape2):
            return False

        if conv.attribs['padding'] != 'VALID':
            return False

        strides = [1, conv.attribs['stride_h'], conv.attribs['stride_w'], 1]
        dilations = from_numpy(block_shape1)

        input = space_to_batch.inputs[0]
        filter = conv.inputs[1]
        output = batch_to_space.outputs[0]

        same_padding = Optimizer._is_same_padded(input.shape, output.shape, strides)

        if not same_padding:
            return False

        op = conv.copy_with(inputs=(input, filter, *conv.inputs[2:]), outputs=output, attribs=dict(conv.attribs))

        op.attribs['dilation_h_factor'] = dilations[0]
        op.attribs['dilation_w_factor'] = dilations[1]
        op.attribs['padding'] = 'SAME'

    @staticmethod
    def _is_constant(tensor):
        return tensor.producer is None and tensor.data is not None

    @staticmethod
    def _read_constant(tensor):
        return tensor.data

    @staticmethod
    def _is_same_padded(input, output, stride, is_nxc=True):
        rank = len(input)
        return all(output[i] == (input[i] + stride[i] - 1) // stride[i]
                   for i in (range(1, rank - 1) if is_nxc else range(2, rank)))

    @staticmethod
    def _eliminate_variable_dequantize_ops(graph):
        for op in list(graph.operations):
            if op.type == 'DEQUANTIZE' and Optimizer._is_constant(op.input):
                variable = op.input

                if 'zero_point' in variable.quant and 'scale' in variable.quant:
                    zero_point = variable.quant['zero_point']
                    scale = variable.quant['scale']
                    variable.data = (variable.data - zero_point) * scale

                variable.data = variable.data.astype(np.float32)
                variable.dtype = np.float32
                variable.quant = None

                bypass_and_remove(graph, op)

    @staticmethod
    def _replace_const_shape(shape):
        if shape.input.shape is not None:
            shape.output.data = np.array(shape.input.shape)
