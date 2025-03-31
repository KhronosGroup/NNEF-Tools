from ..model.utils import bypass_and_remove, replace_chain
from ..model.utils import generate_tensor_names_from_op_type, generate_missing_tensor_names_from_op_type
from ..model import *


class Optimizer:

    def __init__(self, keep_tensor_names=True, custom_optimizers=None, dequantize=False):
        self._keep_tensor_names = keep_tensor_names
        self._custom_optimizers = custom_optimizers or {}
        self._dequantize = dequantize

    def __call__(self, model, only_required=False):
        for graph in model.graphs:
            changed = True
            while changed:
                changed = False

                changed |= self._remove_identity_ops(graph, ('layout.reshape', 'layout.flatten', 'layout.unflatten'),
                                                     lambda op: op.output.shape == op.input.shape)
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
                if input.quant == op.output.quant:
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
