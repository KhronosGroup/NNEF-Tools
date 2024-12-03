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

from ..model.utils import bypass_and_remove, replace_chain
from ..model.utils import generate_tensor_names_from_op_type, generate_missing_tensor_names_from_op_type
from ..model.graph import *


class Optimizer:

    def __init__(self, keep_tensor_names=True, custom_optimizers=None, dequantize=False):
        self._keep_tensor_names = keep_tensor_names
        self._custom_optimizers = custom_optimizers or {}
        self._dequantize = dequantize

    def __call__(self, graph, only_required=False):
        self._fix_inputs_as_output(graph)
        self._fix_inputs_without_producer(graph)

        if not only_required:
            changed = True
            while changed:
                changed = False

                changed |= self._remove_identity_ops(graph, 'copy', lambda op: True)
                changed |= self._remove_identity_ops(graph, 'transpose', lambda op:
                    self._is_sorted(op.attribs['axes']))
                changed |= self._remove_identity_ops(graph, 'reshape', lambda op:
                    op.output.shape == op.input.shape)
                changed |= self._remove_identity_ops(graph, 'squeeze', lambda op:
                    op.attribs['axes'] == [])
                changed |= self._remove_identity_ops(graph, 'unsqueeze', lambda op:
                    op.attribs['axes'] == [])
                changed |= self._remove_identity_ops(graph, 'mul', lambda op:
                    self._is_constant(op.inputs[0], 1.0) or self._is_constant(op.inputs[1], 1.0))
                changed |= self._remove_identity_ops(graph, 'add', lambda op:
                    self._is_constant(op.inputs[0], 0.0) or self._is_constant(op.inputs[1], 0.0))
                changed |= self._remove_identity_ops(graph, ('box', 'debox', 'avg_pool', 'max_pool'), lambda op:
                    self._is_uniform(op.attribs['size'], 1) and
                    self._is_uniform(op.attribs['stride'], 1) and
                    self._is_uniform(op.attribs['dilation'], 1) and
                    self._is_uniform(op.attribs['padding'], 0))
                changed |= self._remove_identity_ops(graph,
                    ('nearest_downsample', 'area_downsample', 'nearest_upsample', 'multilinear_upsample'), lambda op:
                    self._is_uniform(op.attribs['factor'], 1))

                changed |= self._remove_inverse_ops(graph, 'squeeze', 'unsqueeze', lambda op1, op2:
                    op1.attribs['axes'] == op2.attribs['axes'])
                changed |= self._remove_inverse_ops(graph, 'unsqueeze', 'squeeze', lambda op1, op2:
                    op1.attribs['axes'] == op2.attribs['axes'])
                changed |= self._remove_inverse_ops(graph, 'transpose', 'transpose', lambda op1, op2:
                    self._is_sorted(Optimizer._permute(op1.attribs['axes'], op2.attribs['axes'])))

                changed |= self._merge_op_into_variables_and_constants(graph, 'transpose', lambda data, attribs:
                    data.transpose(attribs['axes']))
                changed |= self._merge_op_into_variables_and_constants(graph, 'reshape', lambda data, attribs:
                    data.reshape(self._get_reshape_shape(data, attribs)))
                changed |= self._merge_op_into_variables_and_constants(graph, 'squeeze', lambda data, attribs:
                    data.squeeze(tuple(attribs['axes'])))
                changed |= self._merge_op_into_variables_and_constants(graph, 'unsqueeze', lambda data, attribs:
                    data.reshape(self._unsqueeze_shape(data.shape, attribs['axes'])))

                changed |= self._merge_reshape_sequence(graph)

                changed |= replace_chain(graph, ['pad', {'conv', 'deconv', 'max_pool', 'avg_pool'}],
                                         self._merge_pad_with_sliding)
                changed |= replace_chain(graph, [{'mul', 'div'}, {'conv', 'deconv', 'linear'}],
                                         self._merge_mul_linear, allow_forks=True)
                changed |= replace_chain(graph, [{'conv', 'deconv', 'linear'}, {'add', 'sub'}],
                                         self._merge_linear_add)
                changed |= replace_chain(graph, [{'conv', 'deconv', 'linear'}, {'mul', 'div'}],
                                         self._merge_linear_mul)
                changed |= replace_chain(graph, ['matmul', {'add', 'sub'}],
                                         self._merge_matmul_bias)
                changed |= replace_chain(graph, [{'conv', 'deconv'}, 'batch_normalization'],
                                         self._merge_conv_batch_norm)
                changed |= replace_chain(graph, ['batch_normalization'], self._merge_batch_norm)
                changed |= replace_chain(graph, ['transpose', 'squeeze'], self._merge_transpose_squeeze)
                changed |= replace_chain(graph, ['reshape'], self._substitute_squeeze)

                for chain, replacer in six.iteritems(self._custom_optimizers):
                    changed |= replace_chain(graph, chain, replacer)

                changed |= self._remove_unused_variables_and_constants(graph)

        if self._keep_tensor_names:
            generate_missing_tensor_names_from_op_type(graph)
        else:
            generate_tensor_names_from_op_type(graph)

        if self._dequantize:
            Optimizer._dequantize_variables(graph)
            Optimizer._remove_quantization_attribs(graph)

        return graph

    @staticmethod
    def _fix_inputs_without_producer(graph):
        idx = 0
        for tensor in graph.inputs:
            if tensor.producer is None:
                cnt = len(graph.operations)
                Operation(tensor.graph, type='external', outputs=tensor,
                          attribs={'shape': list(tensor.shape), 'dtype': tensor.dtype})
                graph.move_operation(cnt, idx)
                idx += 1
        return idx > 0

    @staticmethod
    def _fix_inputs_as_output(graph):
        graph.outputs = [Optimizer._insert_copy(tensor) if tensor in graph.inputs else tensor
                         for tensor in graph.outputs]

    @staticmethod
    def _insert_copy(tensor, copy=None):
        if copy is None:
            copy = Tensor(tensor.graph, name=tensor.name + '_copy', dtype=tensor.dtype, shape=tensor.shape,
                          data=tensor.data, quant=tensor.quant)
        Operation(tensor.graph, type='copy', inputs=tensor, outputs=copy)
        return copy

    @staticmethod
    def _match_op_type(type, types):
        return type in types if isinstance(types, tuple) else type == types

    def _remove_identity_ops(self, graph, type, cond):
        changed = False
        for op in graph.operations:
            if self._match_op_type(op.type, type) and cond(op) and op.input.quant == op.output.quant:
                changed |= self._bypass_and_remove(graph, op)

        return changed

    def _merge_op_into_variables_and_constants(self, graph, type, func):
        changed = False
        for op in graph.operations:
            if (op.type == 'variable' or op.type == 'constant') and len(op.output.consumers) > 0:
                if self._all_consumers_same(op.output, type):
                    data = op.output.data if op.output.data is not None else np.zeros(op.output.shape)
                    attribs = op.output.consumers[0].attribs
                    data = func(data, attribs)
                    if op.output.data is not None:
                        op.output.data = data
                    op.output.shape = data.shape
                    op.attribs['shape'] = list(data.shape)
                    for consumer in list(op.output.consumers):  # copy the list before removals!
                        changed |= self._bypass_and_remove(graph, consumer)

        return changed

    def _remove_inverse_ops(self, graph, type1, type2, cond):
        changed = False
        for op in graph.operations:
            if op.type == type1 and len(op.output.consumers) == 1:
                consumer = op.output.consumer
                if consumer.type == type2 and cond(op, consumer):
                    changed |= self._bypass_and_remove(graph, op)
                    changed |= self._bypass_and_remove(graph, consumer)

        return changed

    def _merge_reshape_sequence(self, graph):
        changed = False
        for op in graph.operations:
            if op.type == 'reshape' and len(op.output.consumers) == 1:
                consumer = op.output.consumer
                if consumer.type == 'reshape':
                    new_shape = self._get_reshape_shape(consumer.input, consumer.attribs)
                    if any(s == 0 for s in new_shape):
                        old_shape = self._get_reshape_shape(op.input, op.attribs)
                        new_shape = [old_shape[i] if s == 0 else s for i, s in enumerate(new_shape)]

                    consumer.attribs['shape'] = new_shape
                    del consumer.attribs['axis_start']
                    del consumer.attribs['axis_count']

                    changed |= self._bypass_and_remove(graph, op)

        return changed

    def _get_reshape_shape(self, input, attribs):
        start = attribs.get('axis_start', 0)
        count = attribs.get('axis_count', len(input.shape) - start)
        shape = attribs['shape']
        return input.shape[:start] + tuple(shape) + input.shape[start + count:]

    def _bypass_and_remove(self, graph, op):
        if op.output in graph.outputs and (op.input in graph.inputs or op.input in graph.outputs):
            self._insert_copy(op.input, op.output)
            graph.remove_operation(op, unlink=True)
            return False
        else:
            bypass_and_remove(graph, op, remove_input_not_output=op.output in graph.outputs)
            return True

    @staticmethod
    def _is_channelwise_shape(shape):
        return len(shape) <= 1 or all(s == 1 or i == 1 for i, s in enumerate(shape))

    @staticmethod
    def _merge_linear_add(linear, add, type=None):
        bias = add.inputs[1] if add.inputs[0] == linear.output else add.inputs[0]
        if bias.data is None or not Optimizer._is_channelwise_shape(bias.shape):
            return False

        if len(linear.inputs) > 2 and linear.inputs[2].data is None:
            return None

        if len(bias.shape) == 0:
            bias.data = np.expand_dims(bias.data, axis=0)
        elif len(bias.shape) >= 2:
            bias.data = Optimizer._squeeze_batch_and_spatial_dims(bias.data)

        bias.shape = bias.data.shape

        if add.type == 'sub':
            bias.data = -bias.data

        if len(linear.inputs) == 2:
            new_shape = (1, 1) if len(bias.shape) == 0 else (1, *bias.shape) if len(bias.shape) == 1 else None
            if new_shape is not None:
                bias.data = np.reshape(bias.data, newshape=new_shape)
                bias.shape = new_shape
        else:
            bias.data = linear.inputs[2].data + bias.data
            bias.shape = bias.data.shape

        Optimizer._ensure_variable_producer(bias, label=linear.output.name + '_bias')

        linear.copy_with(type=type or linear.type,
                         attribs=linear.attribs if type != 'linear' else {},
                         inputs=(linear.inputs[0], linear.inputs[1], bias),
                         outputs=add.output)

    @staticmethod
    def _merge_matmul_bias(matmul, add):
        bias = add.inputs[1] if add.inputs[0] == matmul.output else add.inputs[0]
        if not Optimizer._is_channelwise_shape(bias.shape):
            return False

        transposeA = matmul.attribs.get('transposeA') or False
        transposeB = matmul.attribs.get('transposeB') or False

        if transposeA:
            return False

        if not transposeB:
            producer = matmul.inputs[1].producer
            data = matmul.inputs[1].data
            if data is None or producer.type != 'variable':
                return False

            rank = len(data.shape)
            data = np.transpose(data, axes=list(range(rank - 2)) + [rank - 1, rank - 2])
            matmul.inputs[1].data = data
            producer.attribs['shape'] = list(data.shape)
            matmul.attribs['transposeB'] = True

        return Optimizer._merge_linear_add(matmul, add, type='linear')

    @staticmethod
    def _is_sorted(array):
        return all(array[i] <= array[i + 1] for i in range(len(array) - 1))

    @staticmethod
    def _all_consumers_same(tensor, type):
        attribs = tensor.consumers[0].attribs
        return all(consumer.type == type and consumer.attribs == attribs for consumer in tensor.consumers)

    @staticmethod
    def _unsqueeze_shape(shape, axes):
        for axis in axes:
            shape = shape[:axis] + (1,) + shape[axis:]
        return shape

    @staticmethod
    def _permute(items, perm):
        permuted = list(items)
        for i in range(len(perm)):
            permuted[i] = items[perm[i]]
        return type(items)(permuted)

    @staticmethod
    def _add_variable(graph, data, name, label=None):
        output = Tensor(graph, name=name, shape=data.shape, dtype=data.dtype.type, data=data)
        Operation(graph, type='variable', outputs=output, attribs={'shape': list(data.shape), 'label': label or name})
        return output

    @staticmethod
    def _ensure_variable_producer(tensor, label):
        if tensor.producer is None and len(tensor.shape) != 0:
            Operation(tensor.graph, type='variable', outputs=tensor,
                      attribs={'shape': list(tensor.shape), 'label': label})
        elif tensor.producer is not None:
            tensor.producer.attribs['shape'] = list(tensor.shape)

    @staticmethod
    def _merged_conv_batch_norm_params(weights, bias, mean, variance, offset, scale, epsilon, axis):
        std = np.sqrt(variance + epsilon)
        factor = scale / std
        new_weights = weights * np.reshape(factor, newshape=(1,) * axis + factor.shape + (1,) * (len(weights.shape) - axis - 1))
        new_bias = (bias - mean) * factor + offset
        return new_weights, new_bias

    @staticmethod
    def _merge_conv_batch_norm(conv, bn):
        if any(tensor.quant for tensor in conv.inputs) or any(tensor.quant for tensor in bn.inputs):
            return False

        if conv.inputs[1].data is None:
            return False

        weights, bias = Optimizer._merged_conv_batch_norm_params(conv.inputs[1].data,
                                    np.squeeze(conv.inputs[2].data if len(conv.inputs) > 2 else 0, axis=0),
                                    np.squeeze(bn.inputs[1].data, axis=0),
                                    np.squeeze(bn.inputs[2].data, axis=0),
                                    np.squeeze(bn.inputs[3].data if len(bn.inputs) > 3 else 0, axis=0),
                                    np.squeeze(bn.inputs[4].data if len(bn.inputs) > 4 else 1, axis=0),
                                    bn.attribs['epsilon'],
                                    axis=1 if conv.type == 'deconv' else 0)

        bias = np.expand_dims(bias, axis=0)

        conv.inputs[1].data = weights

        if len(conv.inputs) > 2:
            conv.inputs[2].data = bias
            conv.inputs[2].shape = bias.shape
            Optimizer._ensure_variable_producer(conv.inputs[2], label=conv.output.name + '_bias')
            conv.copy_with(outputs=bn.output)
        else:
            bias = Optimizer._add_variable(conv.graph, data=bias, name=conv.output.name + '_bias')
            conv.copy_with(inputs=(*conv.inputs[:2], bias), outputs=bn.output)

    @staticmethod
    def _merged_batch_norm_params(mean, variance, offset, scale, epsilon):
        std = np.sqrt(variance + epsilon)
        factor = scale / std
        return factor, offset - factor * mean

    @staticmethod
    def _merge_batch_norm(bn):
        if any(tensor.quant for tensor in bn.inputs):
            return False

        scale, offset = Optimizer._merged_batch_norm_params(
                                    bn.inputs[1].data,
                                    bn.inputs[2].data,
                                    bn.inputs[3].data if len(bn.inputs) > 3 else 0,
                                    bn.inputs[4].data if len(bn.inputs) > 4 else 1,
                                    bn.attribs['epsilon'])

        scale = Optimizer._add_variable(bn.graph, data=scale, name=bn.output.name + '_scale')
        offset = Optimizer._add_variable(bn.graph, data=offset, name=bn.output.name + '_offset')

        scaled = Tensor(graph=bn.graph, name=bn.output.name + '_scaled', shape=bn.output.shape, dtype=bn.output.dtype)

        Operation(graph=bn.graph, type='mul', inputs=(bn.inputs[0], scale), outputs=scaled)
        Operation(graph=bn.graph, type='add', inputs=(scaled, offset), outputs=bn.output)

    @staticmethod
    def _merge_mul_linear(mul, linear):
        which = 0 if mul.inputs[0].data is not None else 1
        other = 1 - which

        variable = mul.inputs[which]
        if variable.data is None or not Optimizer._is_channelwise_shape(variable.shape):
            return False

        if len(variable.shape) == 0:
            scale = np.expand_dims(variable.data, axis=0)
        elif len(variable.shape) >= 2:
            scale = Optimizer._squeeze_batch_and_spatial_dims(variable.data)

        weights = linear.inputs[1]
        if weights.data is None:
            return False

        rank = len(weights.shape)
        shape = scale.shape + (1,) * (rank - 1) if linear.type == 'deconv' else (1,) + scale.shape + (1,) * (rank - 2)
        scale = np.reshape(scale, newshape=shape)

        weights.data = weights.data * scale if mul.type != 'div' else weights.data / scale

        linear.copy_with(inputs=(mul.inputs[other], weights, *linear.inputs[2:]), outputs=linear.output)

    @staticmethod
    def _merge_linear_mul(linear, mul):
        variable = mul.inputs[1] if mul.inputs[0] == linear.output else mul.inputs[0]
        if variable.data is None or not Optimizer._is_channelwise_shape(variable.shape):
            return False

        if len(variable.shape) == 0:
            scale = np.expand_dims(variable.data, axis=0)
        elif len(variable.shape) >= 2:
            scale = Optimizer._squeeze_batch_and_spatial_dims(variable.data)

        negate = mul.type == 'div'

        weights = linear.inputs[1]
        if weights.data is None:
            return False

        if len(linear.inputs) > 2:
            bias = linear.inputs[2]
            if bias.data is None:
                return False

            bias.data = bias.data * scale if not negate else bias.data / scale
            bias.shape = bias.data.shape

            Optimizer._ensure_variable_producer(bias, label=linear.output.name + '_bias')

        rank = len(weights.shape)
        shape = (1,) + scale.shape + (1,) * (rank - 2) if linear.type == 'deconv' else scale.shape + (1,) * (rank - 1)
        scale = np.reshape(scale, newshape=shape)

        weights.data = weights.data * scale if not negate else weights.data / scale

        linear.copy_with(inputs=(linear.inputs[0], weights, *linear.inputs[2:]), outputs=mul.output)

    @staticmethod
    def _remove_unused_variables_and_constants(graph):
        ops = {op for op in graph.operations
               if (op.type == 'variable' or op.type == 'constant') and len(op.output.consumers) == 0}
        tensors = {op.output for op in ops}
        graph.remove_operations(ops, unlink=True)
        graph.remove_tensors(tensors)
        return len(ops) != 0

    @staticmethod
    def _merge_pad_with_sliding(pad, sliding):
        offset = 2 if sliding.type == 'conv' or sliding.type == 'deconv' else 0
        padding = pad.attribs['padding']

        if not all(p == 0 and q == 0 for p, q in sliding.attribs['padding']) or \
                len(padding) < offset or not all(p == 0 and q == 0 for p, q in padding[:offset]):
            return False

        attribs = dict(sliding.attribs)
        attribs['padding'] = pad.attribs['padding'][offset:]
        attribs['border'] = pad.attribs['border']

        sliding.copy_with(inputs=(pad.input, *sliding.inputs[1:]), attribs=attribs)

    @staticmethod
    def _squeeze_batch_and_spatial_dims(data):
        return np.squeeze(data, axis=(0,) + tuple(i for i in range(2, len(data.shape))))

    @staticmethod
    def _is_constant(tensor, value):
        if tensor.producer is not None and tensor.producer.name == 'constant':
            data = tensor.attribs['value']
        else:
            data = tensor.data

        return (not isinstance(tensor.data, np.ndarray) or data.shape == ()) and data == value

    @staticmethod
    def _is_uniform(array, value):
        return all(item == value for item in array)

    @staticmethod
    def _merge_transpose_squeeze(transpose, squeeze):
        transpose_axes = transpose.attribs['axes']
        squeeze_axes = squeeze.attribs['axes']

        squeezed = [x for i, x in enumerate(transpose_axes) if i not in squeeze_axes]
        is_identity = squeezed == list(range(len(squeezed)))

        if not is_identity:
            return False

        attribs = dict(squeeze.attribs)
        attribs['axes'] = [transpose_axes[x] for x in squeeze_axes]

        squeeze.copy_with(inputs=transpose.input, attribs=attribs)

    @staticmethod
    def _substitute_squeeze(reshape):
        input_shape = reshape.input.shape
        output_shape = reshape.output.shape

        if not len(output_shape) < len(input_shape):
            return False

        k = 0
        axes = []
        for i in range(len(input_shape)):
            if k < len(output_shape) and input_shape[i] == output_shape[k]:
                k += 1
            elif input_shape[i] == 1:
                axes.append(i)
            else:
                return False

        attribs = {'axes': axes}
        dtype = reshape.attribs.get('dtype')
        if dtype is not None:
            attribs['dtype'] = dtype

        Operation(reshape.graph, type='squeeze', name=reshape.name, inputs=reshape.input, outputs=reshape.output,
                  attribs=attribs)

    @staticmethod
    def _dequantize_variables(graph):
        for tensor in graph.tensors:
            if tensor.quant and tensor.data is not None:
                rank = len(tensor.data.shape)
                scale = Optimizer._ensure_quant_param_rank(tensor.quant.get('scale'), rank)
                zero_point = Optimizer._ensure_quant_param_rank(tensor.quant.get('zero_point'), rank)
                if isinstance(zero_point, np.ndarray):
                    assert Optimizer._broadcastable(zero_point.shape, tensor.shape), \
                        f"zero-point shape {zero_point.shape} cannot be broadcast to tensor shape {tensor.shape} " \
                        f"for tensor '{tensor.name}'"
                if isinstance(scale, np.ndarray):
                    assert Optimizer._broadcastable(scale.shape, tensor.shape), \
                        f"scale shape {scale.shape} cannot be broadcast to tensor shape {tensor.shape} " \
                        f"for tensor '{tensor.name}'"
                if scale is not None and not Optimizer._is_zero(scale):
                    dequantized = (tensor.data - zero_point) * scale
                    tensor.data = dequantized.astype(np.float32)
                    tensor.quant = None

    @staticmethod
    def _remove_quantization_attribs(graph):
        for tensor in graph.tensors:
            tensor.quant = None

    @staticmethod
    def _ensure_quant_param_rank(param, rank, offset=0):
        return np.reshape(param, newshape=(1,) * offset + param.shape + (1,) * (rank - 1 - offset)) \
            if isinstance(param, np.ndarray) and len(param.shape) == 1 else param

    @staticmethod
    def _broadcastable(x, y):
        return all(xi == yi or xi == 1 for xi, yi in zip(x, y))

    @staticmethod
    def _is_zero(value):
        return np.all(value == 0) if isinstance(value, np.ndarray) else value == 0
