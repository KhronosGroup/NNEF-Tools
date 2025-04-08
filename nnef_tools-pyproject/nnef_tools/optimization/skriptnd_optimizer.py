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
        self._tensor_references = self._collect_shape_referenced_tensors(model)

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

    @staticmethod
    def _collect_shape_referenced_tensors(model):
        references = {}
        for graph in model.graphs:
            for op in graph.operations:
                for key, value in op.attribs.items():
                    Optimizer._collect_shape_referenced_tensors_from_expr(value, references, op)
            for tensor in graph.tensors:
                for item in tensor.shape:
                    Optimizer._collect_shape_referenced_tensors_from_expr(item, references, tensor)
            for pack in graph.packs:
                for item in pack.shape:
                    Optimizer._collect_shape_referenced_tensors_from_expr(item, references, pack)
                Optimizer._collect_shape_referenced_tensors_from_expr(pack.size, references, pack)
        return references

    @staticmethod
    def _collect_shape_referenced_tensors_from_expr(value, references, referrer):
        if isinstance(value, nd.Expr):
            for expr in nd.recursive_enumerate_expr(value):
                if isinstance(expr, nd.ShapeAccess):
                    Optimizer._add_referenced_tensor(references, expr.tensor.name, referrer)
                if isinstance(expr, nd.SizeAccess):
                    Optimizer._add_referenced_tensor(references, expr.pack.name, referrer)

    @staticmethod
    def _add_referenced_tensor(references, name, referrer):
        referrers = references.get(name)
        if referrers is None:
            references[name] = [referrer]
        elif referrers[-1] is not referrer:
            referrers.append(referrer)

    def _is_referenced(self, tensor):
        return tensor.name in self._tensor_references

    def _is_referenced_except(self, tensor, referrer):
        referrers = self._tensor_references.get(tensor.name)
        if not referrers:
            return False
        return any(item is not referrer for item in referrers)

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

    def _remove_identity_ops(self, graph, type, cond, input_index=None):
        changed = False
        for op in graph.operations:
            if self._match_op_type(op.type, type) and cond(op):
                input = op.input if input_index is None else op.inputs[input_index]
                if not self._is_referenced(op.output) and input.quant == op.output.quant:
                    changed |= self._bypass_and_remove(graph, op, input_index=input_index)

        return changed

    def _remove_inverse_ops(self, graph, type1, type2, cond):
        changed = False
        for op in graph.operations:
            if op.type == type1 and len(op.output.consumers) == 1:
                consumer = op.output.consumer
                if consumer.type == type2 and cond(op, consumer) and \
                        not self._is_referenced(op.output) and not self._is_referenced(consumer.output):
                    changed |= self._bypass_and_remove(graph, op)
                    changed |= self._bypass_and_remove(graph, consumer)

        return changed

    @staticmethod
    def _permute(items, perm):
        permuted = list(items)
        for i in range(len(perm)):
            permuted[i] = items[perm[i]]
        return type(items)(permuted)

    def _bypass_and_remove(self, graph, op, input_index=None):
        input = op.input if input_index is None else op.inputs[input_index]
        if op.output in graph.outputs and (input in graph.inputs or input in graph.outputs):
            self._insert_copy(input, op.detach_output())
            graph.remove_operation(op, unlink=True)
            return False
        else:
            bypass_and_remove(graph, op, remove_input_not_output=op.output in graph.outputs, input_index=input_index)
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

    @staticmethod
    def _resolve_shape_references(shape, target):
        return tuple(nd.transform_expr(expr, lambda x: x.tensor.shape[x.dim] if isinstance(x, nd.ShapeAccess) and x.tensor is target else None)
                     for expr in shape)

    def _merge_reshape_sequence(self, graph):
        changed = False
        for op in graph.operations:
            if op.type == 'layout.reshape' and len(op.output.consumers) == 1:
                consumer = op.output.consumer
                if consumer.type == 'layout.reshape' and not self._is_referenced(op.output):
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
        padding = Optimizer._interleave(pad.attribs['padding'])

        if self._is_referenced_except(pad.output, sliding.output):
            return False

        if not all(p == 0 and q == 0 for p, q in Optimizer._interleave(sliding.attribs['padding'])) or \
                len(padding) < offset or not all(p == 0 and q == 0 for p, q in padding[:offset]):
            return False

        attribs = dict(sliding.attribs)
        attribs['padding'] = Optimizer._uninterleave(padding[offset:])

        sliding.output.shape = self._resolve_shape_references(sliding.output.shape, pad.output)

        sliding.copy_with(inputs=(pad.input, *sliding.inputs[1:]), outputs=sliding.detach_outputs(), attribs=attribs)

    @staticmethod
    def _is_channelwise_shape(shape):
        return len(shape) <= 1 or all(s == 1 or i == 1 for i, s in enumerate(shape))

    @staticmethod
    def _squeeze_batch_and_spatial_dims(data):
        return np.squeeze(data, axis=(0,) + tuple(i for i in range(2, len(data.shape))))

    def _merge_mul_linear(self, mul, linear):
        if self._is_referenced_except(mul.output, linear.output):
            return False

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
        shape = scale.shape + (1,) * (rank - 1) if linear.type == 'nn.deconv' else (1,) + scale.shape + (1,) * (rank - 2)
        scale = np.reshape(scale, newshape=shape)

        weights.data = weights.data * scale if mul.type != 'math.div' else weights.data / scale

        linear.output.shape = self._resolve_shape_references(linear.output.shape, mul.output)

        linear.copy_with(inputs=(mul.inputs[other], weights, *linear.inputs[2:]), outputs=linear.detach_output())

    def _merge_linear_add(self, linear, add, type=None):
        if self._is_referenced_except(linear.output, add.output):
            return False

        bias = add.inputs[1] if add.inputs[0] == linear.output else add.inputs[0]
        if bias.data is None or not Optimizer._is_channelwise_shape(bias.shape):
            return False

        if len(linear.inputs) > 2 and linear.inputs[2] is not None and linear.inputs[2].data is None:
            return None

        if len(bias.shape) == 0:
            bias.data = np.expand_dims(bias.data, axis=0)
        elif len(bias.shape) >= 2:
            bias.data = Optimizer._squeeze_batch_and_spatial_dims(bias.data)

        bias.shape = bias.data.shape

        if bias.shape[0] == 1 and linear.inputs[1].shape[0] != 1:
            bias.data = np.tile(bias.data, linear.inputs[1].shape[0])
            bias.shape = bias.data.shape

        if len(linear.inputs) > 2 and linear.inputs[2] is not None:
            bias.data = linear.inputs[2].data - bias.data if add.type == 'math.sub' else linear.inputs[2].data + bias.data

        add.output.shape = self._resolve_shape_references(add.output.shape, linear.output)

        linear.copy_with(type=type or linear.type,
                         attribs=linear.attribs if type != 'nn.linear' else {},
                         inputs=(linear.inputs[0], linear.inputs[1], bias),
                         outputs=add.detach_output())

    def _merge_linear_mul(self, linear, mul):
        if self._is_referenced_except(linear.output, mul.output):
            return False

        variable = mul.inputs[1] if mul.inputs[0] == linear.output else mul.inputs[0]
        if variable.data is None or not Optimizer._is_channelwise_shape(variable.shape):
            return False

        if len(variable.shape) == 0:
            scale = np.expand_dims(variable.data, axis=0)
        elif len(variable.shape) >= 2:
            scale = Optimizer._squeeze_batch_and_spatial_dims(variable.data)

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

        mul.output.shape = self._resolve_shape_references(mul.output.shape, linear.output)

        linear.copy_with(inputs=(linear.inputs[0], weights, *linear.inputs[2:]), outputs=mul.detach_output())
