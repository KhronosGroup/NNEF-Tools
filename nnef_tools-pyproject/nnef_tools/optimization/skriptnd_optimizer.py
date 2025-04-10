from ..model.utils import bypass_and_remove, replace_chain
from ..model.utils import generate_tensor_names_from_op_type, generate_missing_tensor_names_from_op_type
from ..model import *
import numpy as np
import skriptnd as nd


class Optimizer:

    def __init__(self, keep_tensor_names=True, custom_optimizers=None, dequantize=False):
        self._keep_tensor_names = keep_tensor_names
        self._custom_optimizers = custom_optimizers or {}
        self._dequantize = dequantize

    def __call__(self, model, only_required=False):
        self._collect_shape_referenced_tensors(model)

        for graph in model.graphs:
            changed = True
            while changed:
                changed = False

                changed |= self._remove_identity_ops(graph, 'layout.reshape',
                                                     lambda op: self._resolve_shape_references(op.output.shape, op.input) == op.input.shape)
                changed |= self._remove_identity_ops(graph, 'layout.flatten',
                                                     lambda op: op.attribs['rank'] <= 1)
                changed |= self._remove_identity_ops(graph, 'layout.unflatten',
                                                     lambda op: len(op.attribs['shape']) == 1)
                changed |= self._remove_identity_ops(graph, 'layout.transpose',
                                                     lambda op: self._is_range(op.attribs['perm'], op.attribs['axis']))
                changed |= self._remove_identity_ops(graph, ('layout.squeeze', 'layout.unsqueeze'),
                                                     lambda op: op.attribs['axes'] == [])
                changed |= self._remove_identity_ops(graph, 'math.mul',
                                                     lambda op: self._is_constant(op.inputs[0], 1.0), input_index=1)
                changed |= self._remove_identity_ops(graph, 'math.mul',
                                                     lambda op: self._is_constant(op.inputs[1], 1.0), input_index=0)
                changed |= self._remove_identity_ops(graph, 'math.add',
                                                     lambda op: self._is_constant(op.inputs[0], 0.0), input_index=1)
                changed |= self._remove_identity_ops(graph, 'math.add',
                                                     lambda op: self._is_constant(op.inputs[1], 0.0), input_index=0)
                changed |= self._remove_identity_ops(graph, ('nn.avg_pool', 'nn.max_pool'),
                                                     lambda op: self._is_uniform(op.attribs['size'], 1) and
                                                                self._is_uniform(op.attribs['stride'], 1) and
                                                                self._is_uniform(op.attribs['dilation'], 1) and
                                                                ('padding' not in op.attribs or
                                                                 self._is_uniform(op.attribs['padding'], 0)))
                changed |= self._remove_identity_ops(graph,('image.nearest_downsample', 'image.area_downsample',
                                                            'image.nearest_upsample', 'image.multilinear_upsample'),
                                                     lambda op: self._is_uniform(op.attribs['factor'], 1))

                changed |= self._remove_inverse_ops(graph, 'layout.squeeze', 'layout.unsqueeze',
                                                    lambda op1, op2: op1.attribs['axes'] == op2.attribs['axes'])
                changed |= self._remove_inverse_ops(graph, 'layout.unsqueeze', 'layout.squeeze',
                                                    lambda op1, op2: op1.attribs['axes'] == op2.attribs['axes'])
                changed |= self._remove_inverse_ops(graph, 'layout.transpose', 'layout.transpose',
                                                    lambda op1, op2: op2.output.shape == op1.input.shape)

                changed |= self._merge_op_into_variables_and_constants(graph, 'layout.transpose',
                                   lambda data, attribs: data.transpose(self._transpose_axes(data.shape, attribs)))
                changed |= self._merge_op_into_variables_and_constants(graph, 'layout.reshape',
                                   lambda data, attribs: data.reshape(self._reshape_shape(data.shape, attribs)))
                changed |= self._merge_op_into_variables_and_constants(graph, 'layout.squeeze',
                                   lambda data, attribs: data.squeeze(tuple(attribs['axes'])))
                changed |= self._merge_op_into_variables_and_constants(graph, 'layout.unsqueeze',
                                   lambda data, attribs: data.reshape(self._unsqueeze_shape(data.shape, attribs)))

                changed |= self._merge_reshape_sequence(graph)

                changed |= replace_chain(graph, ['layout.pad', {'nn.conv', 'nn.deconv', 'nn.max_pool', 'nn.avg_pool'}],
                                         self._merge_pad_with_sliding)
                changed |= replace_chain(graph, [{'math.mul', 'math.div'}, {'nn.conv', 'nn.deconv', 'nn.linear'}],
                                         self._merge_mul_linear, allow_forks=True)
                changed |= replace_chain(graph, [{'nn.conv', 'nn.deconv', 'nn.linear'}, {'math.add', 'math.sub'}],
                                         self._merge_linear_add)
                changed |= replace_chain(graph, [{'nn.conv', 'nn.deconv', 'nn.linear'}, {'math.mul', 'math.div'}],
                                         self._merge_linear_mul)
                changed |= replace_chain(graph, ['linalg.matmul', {'math.add', 'math.sub'}],
                                         self._merge_matmul_bias)
                changed |= replace_chain(graph, [{'nn.conv', 'nn.deconv'}, 'nn.batch_norm'],
                                         self._merge_conv_batch_norm)
                changed |= replace_chain(graph, ['nn.batch_norm'], self._split_batch_norm)

    def _collect_shape_referenced_tensors(self, model):
        self._tensor_references = {}
        for graph in model.graphs:
            for op in graph.operations:
                for key, value in op.attribs.items():
                    self._collect_shape_referenced_tensors_from_expr(value, op)
            for tensor in graph.tensors:
                for item in tensor.shape:
                    self._collect_shape_referenced_tensors_from_expr(item, tensor)
            for pack in graph.packs:
                for item in pack.shape:
                    self._collect_shape_referenced_tensors_from_expr(item, pack)
                self._collect_shape_referenced_tensors_from_expr(pack.size, pack)

    def _collect_shape_referenced_tensors_from_expr(self, value, referrer):
        if isinstance(value, nd.Expr):
            for expr in nd.recursive_enumerate_expr(value):
                if isinstance(expr, nd.ShapeAccess):
                    self._add_referenced_tensor(expr.tensor.name, referrer)
                if isinstance(expr, nd.SizeAccess):
                    self._add_referenced_tensor(expr.pack.name, referrer)

    def _add_referenced_tensor(self, name, referrer):
        referrers = self._tensor_references.get(name)
        if referrers is None:
            self._tensor_references[name] = [referrer]
        elif referrer not in referrers:
            referrers.append(referrer)

    def _resolve_shape_references(self, shape, target, referrer=None):
        def func(x):
            return x.tensor.shape[x.dim] if isinstance(x, nd.ShapeAccess) and x.tensor is target else \
                x.size if isinstance(x, nd.SizeAccess) and x.pack is target else None

        if isinstance(shape, tuple):
            resolved = tuple(nd.transform_expr(expr, func) for expr in shape)
            if referrer:
                for item in resolved:
                    self._collect_shape_referenced_tensors_from_expr(item, referrer)
        else:
            resolved = nd.transform_expr(shape, func)
            if referrer:
                self._collect_shape_referenced_tensors_from_expr(shape, referrer)
        return resolved

    def _redirect_shape_references(self, target):
        references = self._tensor_references.get(target.name)
        if references is not None:
            for reference in references:
                if isinstance(reference, Tensor):
                    reference.shape = self._resolve_shape_references(reference.shape, target, reference)
                elif isinstance(reference, TensorPack):
                    reference.shape = self._resolve_shape_references(reference.shape, target, reference)
                    reference.size = self._resolve_shape_references(reference.size, target, reference)
                elif isinstance(reference, Operation):
                    for key, value in reference.attribs.items():
                        reference.attribs[key] = self._resolve_shape_references(value, target, reference)
            del self._tensor_references[target.name]

    @staticmethod
    def _match_op_type(type, types):
        return type in types if isinstance(types, tuple) else type == types

    @staticmethod
    def _is_range(items, first):
        return all(items[i] == first + i for i in range(len(items)))

    @staticmethod
    def _is_constant(tensor, value):
        return tensor.is_constant and tensor.data == value

    @staticmethod
    def _is_uniform(array, value):
        return all(item == value for item in array)

    def _remove_identity_ops(self, graph, type, cond, input_index=0):
        changed = False
        for op in graph.operations:
            if self._match_op_type(op.type, type) and cond(op):
                if op.inputs[input_index].quant == op.output.quant:
                    changed |= self._bypass_and_remove(graph, op, input_index=input_index)

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

    @staticmethod
    def _permute(items, perm):
        permuted = list(items)
        for i in range(len(perm)):
            permuted[i] = items[perm[i]]
        return type(items)(permuted)

    def _bypass_and_remove(self, graph, op, input_index=0):
        input = op.inputs[input_index]
        if op.output in graph.outputs and (input in graph.inputs or input in graph.outputs):
            self._insert_copy(input, op.detach_output())
            graph.remove_operation(op, unlink=True)
            return False
        else:
            remove_input_not_output = op.output in graph.outputs
            self._redirect_shape_references(input if remove_input_not_output else op.output)
            bypass_and_remove(graph, op, remove_input_not_output=remove_input_not_output, input_index=input_index)
            return True

    @staticmethod
    def _insert_copy(tensor, copy=None):
        if copy is None:
            copy = Tensor(tensor.graph, name=tensor.name + '_copy', dtype=tensor.dtype, shape=tensor.shape,
                          data=tensor.data, quant=tensor.quant)
        Operation(tensor.graph, type='', inputs=tensor, outputs=copy)
        return copy

    def _merge_op_into_variables_and_constants(self, graph, type, func):
        changed = False
        for tensor in graph.tensors:
            if tensor.data is not None:
                if len(tensor.consumers) > 0 and self._all_consumers_same(tensor, type):
                    tensor.data = func(tensor.data, tensor.consumers[0].attribs)
                    tensor.shape = tensor.data.shape
                    for consumer in list(tensor.consumers):  # copy the list before removals!
                        changed |= self._bypass_and_remove(graph, consumer)
        return changed

    @staticmethod
    def _all_consumers_same(tensor, type):
        attribs = tensor.consumers[0].attribs
        return all(consumer.type == type and consumer.attribs == attribs for consumer in tensor.consumers)

    @staticmethod
    def _reshape_shape(input_shape, attribs):
        axis = attribs.get('axis', 0)
        rank = attribs.get('rank', len(input_shape) - axis)
        shape = attribs['shape']
        return input_shape[:axis] + tuple(shape) + input_shape[axis + rank:]

    @staticmethod
    def _transpose_axes(input_shape, attribs):
        axis = attribs.get('axis', 0)
        perm = attribs['perm']
        axes = list(range(len(input_shape)))
        axes[axis:axis + len(perm)] = perm
        return axes

    @staticmethod
    def _unsqueeze_shape(input_shape, attribs):
        axes = attribs['axes']
        shape = input_shape
        for axis in axes:
            shape = shape[:axis] + (1,) + shape[axis:]
        return shape

    def _merge_reshape_sequence(self, graph):
        changed = False
        for op in graph.operations:
            if op.type == 'layout.reshape' and len(op.output.consumers) == 1:
                consumer = op.output.consumer
                if consumer.type == 'layout.reshape':
                    new_shape = self._reshape_shape(consumer.input.shape, consumer.attribs)
                    if any(s == 0 for s in new_shape):
                        old_shape = self._reshape_shape(op.input.shape, op.attribs)
                        new_shape = tuple(old_shape[i] if s == 0 else s for i, s in enumerate(new_shape))

                    consumer.attribs['shape'] = list(new_shape)
                    del consumer.attribs['axis']
                    del consumer.attribs['rank']

                    changed |= self._bypass_and_remove(graph, op)

        return changed

    @staticmethod
    def _interleave(a):
        n = len(a)
        return list(zip(a[:n], a[n:]))

    @staticmethod
    def _uninterleave(a):
        return [x for x, y in a] + [y for x, y in a]

    def _merge_pad_with_sliding(self, pad, sliding):
        offset = 2 if sliding.type == 'nn.conv' or sliding.type == 'nn.deconv' else 0
        padding = self._interleave(pad.attribs['padding'])

        if not all(p == 0 and q == 0 for p, q in Optimizer._interleave(sliding.attribs['padding'])) or \
                len(padding) < offset or not all(p == 0 and q == 0 for p, q in padding[:offset]):
            return False

        attribs = dict(sliding.attribs)
        attribs['padding'] = self._uninterleave(padding[offset:])

        self._redirect_shape_references(pad.output)

        sliding.copy_with(inputs=(pad.input, *sliding.inputs[1:]), outputs=sliding.detach_outputs(), attribs=attribs)

    @staticmethod
    def _is_channelwise_shape(shape):
        return len(shape) <= 1 or all(s == 1 or i == 1 for i, s in enumerate(shape))

    @staticmethod
    def _squeeze_batch_and_spatial_dims(data):
        return np.squeeze(data, axis=(0,) + tuple(i for i in range(2, len(data.shape))))

    def _merge_mul_linear(self, mul, linear):
        which = 0 if mul.inputs[0].data is not None else 1
        other = 1 - which

        variable = mul.inputs[which]
        if variable.data is None or not self._is_channelwise_shape(variable.shape):
            return False

        if len(variable.shape) == 0:
            scale = np.expand_dims(variable.data, axis=0)
        elif len(variable.shape) >= 2:
            scale = self._squeeze_batch_and_spatial_dims(variable.data)

        weights = linear.inputs[1]
        if weights.data is None:
            return False

        rank = len(weights.shape)
        shape = scale.shape + (1,) * (rank - 1) if linear.type == 'nn.deconv' else (1,) + scale.shape + (1,) * (rank - 2)
        scale = np.reshape(scale, newshape=shape)

        weights.data = weights.data * scale if mul.type != 'math.div' else weights.data / scale

        self._redirect_shape_references(mul.output)

        linear.copy_with(inputs=(mul.inputs[other], weights, *linear.inputs[2:]), outputs=linear.detach_output())

    def _merge_linear_add(self, linear, add, type=None):
        bias = add.inputs[1] if add.inputs[0] == linear.output else add.inputs[0]
        if bias.data is None or not self._is_channelwise_shape(bias.shape):
            return False

        if len(linear.inputs) > 2 and linear.inputs[2] is not None and linear.inputs[2].data is None:
            return None

        if len(bias.shape) == 0:
            bias.data = np.expand_dims(bias.data, axis=0)
        elif len(bias.shape) >= 2:
            bias.data = self._squeeze_batch_and_spatial_dims(bias.data)

        bias.shape = bias.data.shape

        if bias.shape[0] == 1 and linear.inputs[1].shape[0] != 1:
            bias.data = np.tile(bias.data, linear.inputs[1].shape[0])
            bias.shape = bias.data.shape

        if len(linear.inputs) > 2 and linear.inputs[2] is not None:
            bias.data = linear.inputs[2].data - bias.data if add.type == 'math.sub' else linear.inputs[2].data + bias.data

        self._redirect_shape_references(linear.output)

        linear.copy_with(type=type or linear.type,
                         attribs=linear.attribs if type != 'nn.linear' else {},
                         inputs=(linear.inputs[0], linear.inputs[1], bias),
                         outputs=add.detach_output())

    def _merge_linear_mul(self, linear, mul):
        variable = mul.inputs[1] if mul.inputs[0] == linear.output else mul.inputs[0]
        if variable.data is None or not self._is_channelwise_shape(variable.shape):
            return False

        if len(variable.shape) == 0:
            scale = np.expand_dims(variable.data, axis=0)
        elif len(variable.shape) >= 2:
            scale = self._squeeze_batch_and_spatial_dims(variable.data)

        negate = mul.type == 'math.div'

        weights = linear.inputs[1]
        if weights.data is None:
            return False

        if len(linear.inputs) > 2 and linear.inputs[2] is not None:
            bias = linear.inputs[2]
            if bias.data is None:
                return False

            bias.data = bias.data * scale if not negate else bias.data / scale
            bias.shape = bias.data.shape

        rank = len(weights.shape)
        shape = (1,) + scale.shape + (1,) * (rank - 2) if linear.type == 'nn.deconv' else scale.shape + (1,) * (rank - 1)
        scale = np.reshape(scale, newshape=shape)

        weights.data = weights.data * scale if not negate else weights.data / scale

        self._redirect_shape_references(linear.output)

        linear.copy_with(inputs=(linear.inputs[0], weights, *linear.inputs[2:]), outputs=mul.detach_output())

    def _merge_matmul_bias(self, matmul, add):
        if matmul.inputs[0].rank != 2:
            return False

        bias = add.inputs[1] if add.inputs[0] == matmul.output else add.inputs[0]
        if not self._is_channelwise_shape(bias.shape):
            return False

        transposeA = matmul.attribs.get('transA') or False
        transposeB = matmul.attribs.get('transB') or False

        if transposeA:
            return False

        if not transposeB:
            B = matmul.inputs[1]
            if not B.is_variable:
                return False

            rank = len(B.data.shape)
            B.data = np.transpose(B.data, axes=list(range(rank - 2)) + [rank - 1, rank - 2])
            B.shape = B.data.shape
            matmul.attribs['transB'] = True

        return self._merge_linear_add(matmul, add, type='nn.linear')

    @staticmethod
    def _merged_conv_batch_norm_params(weights, bias, mean, variance, offset, scale, epsilon, axis):
        std = np.sqrt(variance + epsilon)
        factor = scale / std
        new_weights = weights * np.reshape(factor, shape=(1,) * axis + factor.shape + (1,) * (len(weights.shape) - axis - 1))
        new_bias = (bias - mean) * factor + offset
        return new_weights, new_bias

    def _merge_conv_batch_norm(self, conv, bn):
        if any(tensor.quant for tensor in conv.inputs if tensor is not None) or \
                any(tensor.quant for tensor in bn.inputs if tensor is not None):
            return False

        if conv.inputs[1].data is None:
            return False

        weights, bias = self._merged_conv_batch_norm_params(conv.inputs[1].data,
                                                            conv.inputs[2].data if conv.inputs[2] else 0,
                                                            bn.inputs[1].data,
                                                            bn.inputs[2].data,
                                                            bn.inputs[3].data if bn.inputs[3] else 0,
                                                            bn.inputs[4].data if bn.inputs[4] else 1,
                                                            bn.attribs['epsilon'],
                                                            axis=1 if conv.type == 'nn.deconv' else 0)
        conv.inputs[1].data = weights

        if conv.inputs[2]:
            conv.inputs[2].data = bias
            conv.copy_with(outputs=bn.detach_output())
        else:
            bias = Tensor(conv.graph, name=conv.output.name + '_bias', shape=bias.shape, dtype=bias.dtype.type, data=bias)
            conv.copy_with(inputs=(*conv.inputs[:2], bias), outputs=bn.detach_output())

        self._redirect_shape_references(conv.output)

    @staticmethod
    def _merged_batch_norm_params(mean, variance, offset, scale, epsilon):
        std = np.sqrt(variance + epsilon)
        factor = scale / std
        return factor, offset - factor * mean

    def _split_batch_norm(self, bn):
        if any(tensor.quant for tensor in bn.inputs if tensor is not None):
            return False

        scale, offset = self._merged_batch_norm_params(bn.inputs[1].data,
                                                       bn.inputs[2].data,
                                                       bn.inputs[3].data if bn.inputs[3] else 0,
                                                       bn.inputs[4].data if bn.inputs[4] else 1,
                                                       bn.attribs['epsilon'])

        scale = Tensor(bn.graph, data=scale, name=bn.output.name + '_scale', shape=scale.shape, dtype=scale.dtype.type)
        offset = Tensor(bn.graph, data=offset, name=bn.output.name + '_offset', shape=offset.shape, dtype=offset.dtype.type)
        scaled = Tensor(graph=bn.graph, name=bn.output.name + '_scaled', shape=bn.output.shape, dtype=bn.output.dtype)

        Operation(graph=bn.graph, type='math.mul', inputs=(bn.inputs[0], scale), outputs=scaled, attribs={'rhs_align': 1})
        Operation(graph=bn.graph, type='math.add', inputs=(scaled, offset), outputs=bn.detach_output(), attribs={'rhs_align': 1})
