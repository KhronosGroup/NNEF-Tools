from __future__ import division, print_function, absolute_import
from .converter import ConverterToTS as _Converter, Transform, ConversionError
from ..model.utils import *
from ..model import *
from ..utils import types
from collections import OrderedDict
from .converter import ShapeExpr, optimize_shape_expr, check_shape_expr
import numpy as np
import copy


_INT_MAX = 2 ** 31 - 1
_FLT_POS_INF = ShapeExpr(ShapeExpr.Op.Const, args=[float('inf')])
_FLT_NEG_INF = ShapeExpr(ShapeExpr.Op.Const, args=[float('-inf')])
_INT_POS_INF = ShapeExpr(ShapeExpr.Op.Cast, args=[_FLT_POS_INF, 'int'])
_INT_NEG_INF = ShapeExpr(ShapeExpr.Op.Cast, args=[_FLT_NEG_INF, 'int'])


ONNX_SOFTMAX = """
operator onnx_softmax {
    @attrib {
        axis: int;
    }
    @input {
        input: real[s..(d)];
    }
    @output {
        output: real[s..];
    }
    @assert {
        axis >= -d && axis < d;
    }
    @compose {
        output = nn.softmax{axes=[axis:d]}(input);
    }
}
"""

ONNX_GLOBAL_AVG_POOL = """
operator onnx_global_avg_pool {
    @input {
        input: real[b,c,s..(d)];
    }
    @output {
        output: real[b,c,1 ..(d)];
    }
    @compose {
        output = math.mean_reduce{axes=[2:d+2]}(input);
    }
}
"""

ONNX_GLOBAL_MAX_POOL = """
operator onnx_global_max_pool {
    @input {
        input: real[b,c,s..(d)];
    }
    @output {
        output: real[b,c,1 ..(d)];
    }
    @compose {
        output = math.max_reduce{axes=[2:d+2]}(input);
    }
}
"""

ONNX_GLOBAL_LP_POOL = """
operator onnx_global_lp_pool {
    @attrib {
        p: real;
    }
    @input {
        input: real[b,c,s..(d)];
    }
    @output {
        output: real[b,c,1 ..(d)];
    }
    @compose {
        output = math.lp_reduce{axes=[2:d+2], p=p}(input);
    }
}
"""


class Converter(_Converter):

    @staticmethod
    def defined_operations():
        return {
            'onnx_softmax': ONNX_SOFTMAX,
            'onnx_global_avg_pool': ONNX_GLOBAL_AVG_POOL,
            'onnx_global_max_pool': ONNX_GLOBAL_MAX_POOL,
            'onnx_global_lp_pool': ONNX_GLOBAL_LP_POOL,
        }

    @staticmethod
    def defined_imports():
        return {'nn', 'math'}

    ShapeExprArgs = {
        'ReduceMin': [1],
        'ReduceMax': [1],
        'ReduceMean': [1],
        'ReduceSum': [1],
        'ReduceProd': [1],
        'ReduceL1': [1],
        'ReduceL2': [1],
        'Reshape': [1],
        'Squeeze': [1],
        'Unsqueeze': [1],
        'Split': [1],
        'Pad': [1, 3],
        'Tile': [1],
        'Expand': [1],
        'Slice': [1, 2, 3, 4],
        'Upsample': [1],
        'Resize': [1, 2, 3],
        'TopK': [1],
        'NonMaxSuppression': [2, 3, 4],
        'ConstantOfShape': [0],
        'Range': [0, 1, 2],
    }

    OpsWithDistinguishedInput = [
        'Slice',
        'Gather',
        'Squeeze',
        'Unqueeze',
        'Expand',
        'Transpose',
        'Reshape',
    ]

    def __init__(self, custom_transforms=None, custom_functions=None, mirror_unsupported=False):
        _Converter.__init__(self, transforms=self.merge_transforms(_Transforms, custom_transforms),
                            functions=custom_functions, mirror_unsupported=mirror_unsupported)

    def __call__(self, model):
        self._eliminate_constant_ops(model)
        self._collect_shape_ops(model)
        model = _Converter.__call__(self, model)
        self._add_zero_copy_for_constant_outputs(model)
        self._eliminate_empty_subgraphs(model)
        self.remove_unused_constants(model)
        self._fix_constant_names(model)
        ensure_valid_ids(model)
        self._fix_loops(model)
        self._fix_shape_expr_args(model)
        generate_missing_tensor_names_from_op_type(model)
        return model

    def should_skip_conversion(self, op):
        return op in self._shape_ops

    def _eliminate_constant_ops(self, model):
        for graph in model.graphs:
            removed = []
            for op in graph.operations:
                if op.type == 'Constant':
                    value = op.attribs['value']
                    op.output.set_data(value)
                    removed.append(op)
                elif op.type == 'ConstantOfShape' and op.input.data is not None:
                    shape = op.input.data
                    value = op.attribs['value']
                    op.output.set_data(value.item() if value.size == 1 else np.reshape(value, shape))
                    removed.append(op)

            graph.remove_operations(removed, unlink=True)

    def _add_zero_copy_for_constant_outputs(self, model):
        for graph in model.graphs:
            graph.outputs = tuple(self.zero_copy(tensor) if tensor.is_constant or tensor.is_variable else tensor
                                  for tensor in graph.outputs)

    def _eliminate_empty_subgraphs(self, model):
        for graph in model.graphs:
            for op in graph.operations:
                for key, value in op.attribs.items():
                    if isinstance(value, Graph) and len(value.operations) == 0 and len(value.outputs) == 1:
                        op.attribs[key] = value = value.outputs[0]
                        op.inputs = op.inputs + (value,)
                    elif isinstance(value, list):
                        for idx, item in enumerate(value):
                            if isinstance(item, Graph) and len(item.operations) == 0 and len(item.outputs) == 1:
                                op.attribs[key][idx] = item = item.outputs[0]
                                op.inputs = op.inputs + (item,)

        removed = [graph for graph in model.graphs[1:] if len(graph.operations) == 0 and len(graph.outputs) == 1]
        model.remove_graphs(removed)

    def _collect_shape_ops(self, model):
        self._shape_ops = set()
        for graph in model.graphs:
            for op in reversed(graph.operations):
                if op in self._shape_ops:
                    if op.type != 'Shape':
                        self._shape_ops.update({input.producer for input in op.inputs if input.producer is not None})
                else:
                    shape_args = self.ShapeExprArgs.get(op.type, [])
                    for idx, input in enumerate(op.inputs):
                        if input.producer is not None and idx in shape_args:
                            self._shape_ops.add(input.producer)

    def _is_constant(self, tensor):
        return tensor.data is not None

    def _read_constant(self, tensor, type=None):
        if tensor.data is None:
            raise ConversionError(f"trying to evaluate non-constant tensor '{tensor.name}'")

        value = tensor.data
        return types.from_numpy(value, type=type) if isinstance(value, np.ndarray) else \
                types.cast(value, type=type) if type else value

    def _fix_constant_names(self, model):
        used = {tensor.name for graph in model.graphs for tensor in graph.tensors}

        counter = 0
        for graph in model.graphs:
            for tensor in graph.tensors:
                if tensor.name is None:
                    counter += 1
                    name = '/' + str(counter)
                    while name in used:
                        counter += 1
                        name = '/' + str(counter)

                    tensor.name = name

    def _fix_loops(self, model):
        body_graphs = {op.attribs['body_graph']: op.attribs.get('cond_graph') is not None
                       for graph in model.graphs for op in graph.operations if op.type == 'do'}

        for body, has_cond in body_graphs.items():
            body.inputs = self.tupled(body.inputs[1], has_cond) + body.inputs[2:] + (body.inputs[0],)

            if not has_cond:
                body.remove_operation(body.outputs[0].producer, unlink=True)

            body.outputs = self.tupled(body.outputs[0], has_cond) + body.outputs[1:]

        for graph in model.graphs:
            for op in graph.operations:
                if op.type == 'do':
                    cond = op.attribs.get('cond_graph')
                    if cond is not None:
                        op.outputs = (Tensor(graph, name='', dtype=np.void, shape=()),) + op.outputs

    def _is_shape_expr(self, tensor):
        original = self._tensor_map.get(tensor)
        return original.producer in self._shape_ops if original else False

    def _fix_shape_expr_args(self, model):
        for graph in model.graphs:
            count = len(graph.operations)
            for i in range(count):
                op = graph.operations[i]
                for input in op.inputs:
                    if isinstance(input, list):
                        for item in input:
                            if item is not None and not item.has_producer and self._is_shape_expr(item):
                                self._fix_shape_expr_arg(item, graph)
                    else:
                        if input is not None and not input.has_producer and self._is_shape_expr(input):
                            self._fix_shape_expr_arg(input, graph)
            graph.sort()

    def _fix_shape_expr_arg(self, arg, graph):
        expr = self.arg_as_attrib(arg)
        if expr.op == ShapeExpr.Op.Const:
            arg.set_data(expr.args[0], variable=False)
        else:
            Operation(graph, type="layout.constant", attribs={"shape": list(arg.shape), "value": expr}, inputs=(), outputs=(arg,))

    @staticmethod
    def _interleave(items):
        return [item[0] for item in items] + [item[1] for item in items]

    @staticmethod
    def _uninterleave(items):
        count = len(items) // 2
        return list(zip(items[:count], items[count:]))

    def convert_padding(self, pads, auto_pad, output_padding=None):
        if auto_pad == "NOTSET":
            if output_padding is not None:
                padding = list(pads) if pads is not None else [0] * (2 * len(output_padding))
                offs = len(output_padding)
                for i in range(len(output_padding)):
                    padding[i + offs] -= output_padding[i]
                return padding
            else:
                return pads or 0
        elif auto_pad == "VALID":
            if output_padding is not None:
                return [0] * len(output_padding) + [-p for p in output_padding]
            else:
                return 0
        elif auto_pad == "SAME_LOWER" or auto_pad == "SAME_UPPER":
            return None
        else:
            assert False

    def squeeze_input(self, tensor, axes, keep_dims=False):
        return self._pre_squeeze(tensor, axes=axes) if not keep_dims and len(axes) else tensor

    def squeeze_output(self, tensor, axes, keep_dims=False):
        return self._post_squeeze(tensor, axes=axes) if not keep_dims and len(axes) else tensor

    def squeeze_param(self, tensor, axes):
        tensor.data = tensor.data.squeeze(tuple(axes))
        tensor.shape = tensor.data.shape
        return tensor

    def unsqueeze_input(self, tensor, axes, keep_dims=False):
        return self._pre_unsqueeze(tensor, axes=axes) if not keep_dims and len(axes) else tensor

    def unsqueeze_output(self, tensor, axes, keep_dims=False):
        return self._post_unsqueeze(tensor, axes=axes) if not keep_dims and len(axes) else tensor

    def unsqueeze_vector(self, tensor):
        original = self._tensor_map[tensor]
        if self._is_constant(original) and len(original.consumers) == 1:
            self._transform_constant(tensor, lambda data: np.expand_dims(data, 0))
            return tensor
        else:
            return self.unsqueeze_input(tensor, axes=[0])

    def transpose_matrix(self, tensor):
        original = self._tensor_map[tensor]
        if self._is_constant(original) and len(original.consumers) == 1:
            self._transform_constant(tensor, lambda data: np.transpose(data))
            return tensor
        else:
            return self._pre_transpose(tensor, perm=[1, 0])

    def stack_output(self, output, axis=0):
        input = Tensor(output.graph, dtype=output.dtype, shape=output.shape, quant=copy.deepcopy(output.quant))
        self._stack_operation(input, output, axis)
        return input

    def zero_copy(self, tensor):
        result = Tensor(tensor.graph, dtype=tensor.dtype, shape=tensor.shape, quant=copy.deepcopy(tensor.quant))
        self._zero_copy_operation(tensor, result)
        return result

    def ensure_list(self, arg):
        return [arg] if not isinstance(arg, list) else arg

    def _convert_int_inf(self, x):
        return _INT_POS_INF if x > _INT_MAX else _INT_NEG_INF if x < -_INT_MAX else x

    def is_unused(self, tensor):
        if len(tensor.name) == 0:
            return True
        original = self._tensor_map[tensor]
        return len(original.consumers) == 0

    def leading_zeros(self, items):
        for i, item in enumerate(items):
            if item != 0:
                return i
        return len(items)

    def is_continuous_range(self, items):
        return items == list(range(items[0], items[0] + len(items)))

    def is_const_true_cond(self, cond_in, cond_out, cond_init):
        cond_in = self._tensor_map[cond_in]
        cond_out = self._tensor_map[cond_out]
        cond_init = self._tensor_map[cond_init]
        return (cond_init.name == '' and cond_out.data is not None and self.from_numpy(cond_out.data) == True) or \
               (cond_out.producer.type == 'Identity' and cond_out.producer.input is cond_in and self.from_numpy(cond_init.data) == True)

    def is_const_int_max(self, arg):
        if not self.is_const(arg):
            return False
        value = self.as_const(arg)
        if isinstance(value, list):
            value = value[0]
        return value >= _INT_MAX

    def tupled(self, x, c):
        return (x,) if c else ()

    def conflate_lstm_bias(self, bias):
        if isinstance(bias.data, np.ndarray):
            b1, b2 = np.split(bias.data, indices_or_sections=2, axis=1)
            bias.data = b1 + b2
            bias.shape = bias.data.shape
            return bias
        else:
            b1, b2 = self._split(bias, axis=1, count=2)
            return self._add(b1, b2)

    def coordinate_transform(self, coordinate_transformation_mode):
        return 'SYMMETRIC' if coordinate_transformation_mode == 'half_pixel' or \
                            coordinate_transformation_mode == 'pytorch_half_pixel' else \
               'ASYMMETRIC' if coordinate_transformation_mode == 'asymmetric' else \
               'ALIGNED' if coordinate_transformation_mode == 'align_corners' else None

    @staticmethod
    def _non_singleton_rank(shape):
        return sum(s != 1 for s in shape)

    @staticmethod
    def _ensure_zero_rank(expr):
        if expr.rank == 0:
            return expr
        elif expr.op == ShapeExpr.Op.UpRank and expr.arg.rank == 0:
            return expr.arg
        else:
            return ShapeExpr(ShapeExpr.Op.DownRank, expr.rank)

    @staticmethod
    def as_list(obj):
        return obj.tolist() if isinstance(obj, np.ndarray) else list(obj)

    def _eval_symbolic_shape(self, tensor):
        op = tensor.producer
        if op is None and tensor.data is None:
            raise ConversionError(f"Cannot convert symbolic shape expression that contains (sub)graph input '{tensor.name}'")
        elif tensor.data is not None:
            value = self._read_constant(tensor)
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            effective_rank = self._non_singleton_rank(value.shape)
            if effective_rank > 1:
                raise ConversionError(f"Symbolic shape must not contain constants with effective rank > 1; "
                                      f"found tensor '{tensor.name}' of rank {effective_rank}")
            rank = len(value.shape)
            if rank > effective_rank:
                expr = ShapeExpr(ShapeExpr.Op.Const, args=[np.squeeze(value)])
                return ShapeExpr(ShapeExpr.Op.UpRank, args=[expr, rank - effective_rank])
            else:
                return ShapeExpr(ShapeExpr.Op.Const, args=[value])
        elif op.type == 'Shape':
            return ShapeExpr(ShapeExpr.Op.Shape, args=[self._map_tensor(op.input)])
        elif op.type == 'If':
            condition = self._eval_symbolic_shape(op.inputs[0])
            then_branch = self._eval_symbolic_shape(op.attribs['then_branch'].outputs[0])
            else_branch = self._eval_symbolic_shape(op.attribs['else_branch'].outputs[0])
            return ShapeExpr(ShapeExpr.Op.Select, args=[condition, then_branch, else_branch])
        else:
            inputs = [self._eval_symbolic_shape(input) for input in op.inputs]
            if op.type == 'Add':
                return ShapeExpr(ShapeExpr.Op.Add, args=inputs)
            elif op.type == 'Sub':
                return ShapeExpr(ShapeExpr.Op.Sub, args=inputs)
            elif op.type == 'Mul':
                return ShapeExpr(ShapeExpr.Op.Mul, args=inputs)
            elif op.type == 'Div':
                return ShapeExpr(ShapeExpr.Op.Div, args=inputs)
            elif op.type == 'Min':
                return ShapeExpr(ShapeExpr.Op.Min, args=inputs)
            elif op.type == 'Max':
                return ShapeExpr(ShapeExpr.Op.Max, args=inputs)
            elif op.type == 'Less':
                return ShapeExpr(ShapeExpr.Op.Less, args=inputs)
            elif op.type == 'Greater':
                return ShapeExpr(ShapeExpr.Op.Greater, args=inputs)
            elif op.type == 'LessOrEqual':
                return ShapeExpr(ShapeExpr.Op.LessEqual, args=inputs)
            elif op.type == 'GreaterOrEqual':
                return ShapeExpr(ShapeExpr.Op.GreaterEqual, args=inputs)
            elif op.type == 'Equal':
                return ShapeExpr(ShapeExpr.Op.Equal, args=inputs)
            elif op.type == 'And':
                return ShapeExpr(ShapeExpr.Op.And, args=inputs)
            elif op.type == 'Or':
                return ShapeExpr(ShapeExpr.Op.Or, args=inputs)
            elif op.type == 'Xor':
                return ShapeExpr(ShapeExpr.Op.Xor, args=inputs)
            elif op.type == 'Ceil':
                assert inputs[0].op == ShapeExpr.Op.Div, (f"'Ceil' op can only be converted when preceded by 'Div'"
                                                          f" op in shape expression; found operation '{op.name}' preceded"
                                                          f" by operation '{inputs[0].op.name}' of type '{inputs[0].op.type}'")
                inputs[0].op = ShapeExpr.Op.CeilDiv
                return inputs[0]
            elif op.type == 'Gather':
                assert inputs[1].rank <= 1, f"Operation '{op.name}' of type 'Gather' must have index of rank <= 1 in shape expression"
                return ShapeExpr(ShapeExpr.Op.Subscript, args=[inputs[0], inputs[1]])
            elif op.type == 'Concat':
                return ShapeExpr(ShapeExpr.Op.Concat, args=inputs)
            elif op.type == 'Slice':
                assert inputs[1].effective_rank == 0, f"Operation '{op.name}' of type 'Slice' must have 'starts' of length 1"
                assert inputs[2].effective_rank == 0, f"Operation '{op.name}' of type 'Slice' must have 'ends' of length 1"
                beg = self._ensure_zero_rank(inputs[1])
                end = self._ensure_zero_rank(inputs[2])
                return ShapeExpr(ShapeExpr.Op.Slice, args=[inputs[0], beg, end])
            elif op.type == 'Squeeze':
                axes = op.attribs.get('axes') or self.as_list(self.as_const(op.inputs[1]))
                return ShapeExpr(ShapeExpr.Op.DownRank, args=[inputs[0], len(axes)])
            elif op.type == 'Unsqueeze':
                axes = op.attribs.get('axes') or self.as_list(self.as_const(op.inputs[1]))
                return ShapeExpr(ShapeExpr.Op.UpRank, args=[inputs[0], len(axes)])
            elif op.type == 'Cast':
                output_type = self.ts_dtype(op.output.dtype)
                return ShapeExpr(ShapeExpr.Op.Cast, args=[inputs[0], output_type])
            elif op.type == 'Expand':
                assert inputs[0].effective_rank == 0, f"Operation '{op.name}' of type 'Expand' must have input of singular dimensions"
                repeats = ShapeExpr(ShapeExpr.Op.DownRank, args=[inputs[1], 1])
                return ShapeExpr(ShapeExpr.Op.Expand, args=[inputs[0], repeats])
            elif op.type == 'Transpose':
                return inputs[0]
            elif op.type == 'Reshape':
                shape = op.attribs.get('shape')
                if shape is None and self._is_constant(op.inputs[1]):
                    shape = self.as_list(self.as_const(op.inputs[1]))
                if shape is None:
                    raise ConversionError(f"Shape argument of operation '{op.name}' of type 'Reshape' must be constant in shape expression")

                if len(shape) > inputs[0].rank:
                    return ShapeExpr(ShapeExpr.Op.UpRank, args=[inputs[0], len(shape) - inputs[0].rank])
                elif len(shape) < inputs[0].rank:
                    return ShapeExpr(ShapeExpr.Op.DownRank, args=[inputs[0], inputs[0].rank - len(shape)])
                else:
                    return inputs[0]
            elif op.type == 'Range':
                return ShapeExpr(ShapeExpr.Op.Range, args=[inputs[0], inputs[1], inputs[2]])
            elif op.type == 'ReduceSum':
                expr = ShapeExpr(ShapeExpr.Op.Sum, args=[inputs[0]])
                return expr if op.attribs.get('keepdims') == 0 else \
                    ShapeExpr(ShapeExpr.Op.UpRank, args=[expr, 1])
            elif op.type == 'ReduceProd':
                expr = ShapeExpr(ShapeExpr.Op.Prod, args=[inputs[0]])
                return expr if op.attribs.get('keepdims') == 0 else \
                    ShapeExpr(ShapeExpr.Op.UpRank, args=[expr, 1])
            elif op.type == 'ReduceMin':
                expr = ShapeExpr(ShapeExpr.Op.Minimize, args=[inputs[0]])
                return expr if op.attribs.get('keepdims') == 0 else \
                    ShapeExpr(ShapeExpr.Op.UpRank, args=[expr, 1])
            elif op.type == 'ReduceMax':
                expr = ShapeExpr(ShapeExpr.Op.Maximize, args=[inputs[0]])
                return expr if op.attribs.get('keepdims') == 0 else \
                    ShapeExpr(ShapeExpr.Op.UpRank, args=[expr, 1])
            else:
                raise ConversionError(f"Conversion of operator '{op.name}' of type '{op.type}' "
                                      f"as a symbolic shape expression is not implemented")

    def eval_symbolic_shape(self, tensor, as_scalar=False):
        symbolic = self._eval_symbolic_shape(self._tensor_map[tensor])
        if as_scalar:
            subscript = ShapeExpr(ShapeExpr.Op.Const, args=[np.array(0)])
            symbolic = ShapeExpr(ShapeExpr.Op.Subscript, args=[symbolic, subscript])
        symbolic = optimize_shape_expr(symbolic)
        check_shape_expr(symbolic)
        return symbolic

    def arg_as_attrib(self, arg, as_scalar=False, convert_int_inf=False, none_on_failure=False):
        if isinstance(arg, list):
            return [self.arg_as_attrib(item, as_scalar=as_scalar, convert_int_inf=convert_int_inf) for item in arg]

        if self.is_const(arg):
            value = self.as_const(arg)
            if convert_int_inf:
                value = [self._convert_int_inf(x) for x in value] if isinstance(value, list) else self._convert_int_inf(value)
            return value[0] if as_scalar and isinstance(value, list) and len(value) == 1 else value
        else:
            try:
                return self.eval_symbolic_shape(arg, as_scalar=as_scalar)
            except AssertionError as e:
                if none_on_failure:
                    return None
                raise ConversionError(f"Conversion of shape expression is not possible: " + str(e))


_Transforms = Converter.unpack_transforms({
    ('Conv', 'ConvTranspose'):
        Transform(
            type=('nn.conv', 'nn.deconv'),
            defaults={
                'strides': None,
                'dilations': None,
                'pads': None,
                'auto_pad': "NOTSET",
                'group': 1,
                'output_shape': None,
                'output_padding': None,
            },
            inputs=(
                '!I[0]',
                '!I[1]',
                '!I[2] if len(I) > 2 else None',
            ),
            outputs='!O[0]',
            attribs={
                'stride': '!strides',
                'dilation': '!dilations',
                'padding': '!convert_padding(pads, auto_pad, output_padding)',
                'padding_align': '!"UPPER" if auto_pad == "SAME_UPPER" else None',
                'groups': '!group',
                'output_size': '!output_shape[2:] if output_shape is not None else None',
            }
        ),
    ('MaxPool', 'AveragePool', 'LpPool'):
        Transform(
            type=('nn.max_pool', 'nn.avg_pool', 'nn.lp_pool'),
            defaults={
                'strides': None,
                'dilations': None,
                'pads': None,
                'auto_pad': "NOTSET",
                'ceil_mode': 0,
                'storage_order': 0,
                'count_include_pad': 0,
                'p': 2,
            },
            cond={
                '!storage_order == 0': 'storage_order must be 0',
                '!auto_pad != "SAME_LOWER"': 'auto_pad must not be SAME_LOWER',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'size': '!kernel_shape',
                'stride': '!strides',
                'dilation': '!dilations',
                'padding': '!convert_padding(pads, auto_pad)',
                'padding_align': '!"UPPER" if auto_pad == "SAME_UPPER" else None',
                'ignore_border': '!count_include_pad == 0 if _type_ != "LpPool" else None',
                'ceil_mode': '!ceil_mode != 0',
                'p': '!float(p) if _type_ == "LpPool" else None',
            }
        ),
    ('GlobalMaxPool', 'GlobalAveragePool', 'GlobalLpPool'):
        Transform(
            type=('!"math.max_reduce" if known_rank else "onnx_global_max_pool"',
                  '!"math.mean_reduce" if known_rank else "onnx_global_avg_pool"',
                  '!"math.lp_reduce" if known_rank else "onnx_global_lp_pool"'),
            using={
                'known_rank': '!I[0].rank != None',
            },
            defaults={
                'p': 2,
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!list(range(2, I[0].rank)) if known_rank else None',
                'p': '!float(p) if _type_ == "GlobalLpPool" else None',
            }
        ),
    ('ReduceMin', 'ReduceMax', 'ReduceMean', 'ReduceSum', 'ReduceProd', 'ReduceL1', 'ReduceL2'):
        Transform(
            type=('math.min_reduce', 'math.max_reduce', 'math.mean_reduce', 'math.sum_reduce',
                  'math.prod_reduce', 'math.lp_reduce', 'math.lp_reduce'),
            defaults={
                'keepdims': 1,
                'axes': '!arg_as_attrib(I[1]) if len(I) > 1 else None',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
                'squeeze': '!keepdims == 0',
                'p': '!1.0 if _type_ == "ReduceL1" else 2.0 if _type_ == "ReduceL2" else None',
            }
        ),
    ('ArgMin', 'ArgMax'):
        Transform(
            type=('math.argmin', 'math.argmax'),
            defaults={
                'axis': 0,
                'keepdims': 1,
                'select_last_index': 0,
            },
            cond={
                '!select_last_index == 0': 'select_last_index must be 0',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axis': '!axis',
                'squeeze': '!keepdims == 0',
            }
        ),
    'BatchNormalization':
        Transform(
            type='nn.batch_norm',
            defaults={
                'epsilon': 1e-5,
                'spatial': 1,
            },
            inputs=(
                '!I[0]',
                '!I[3]',
                '!I[4]',
                '!I[2]',
                '!I[1]',
            ),
            outputs='!O[0]',
            attribs={
                'epsilon': '!epsilon',
            }
        ),
    ('Relu', 'Sigmoid', 'Softplus', 'Selu', 'Not', 'Identity', 'Elu', 'Erf', 'Abs', 'Sign',
     'Sin', 'Cos', 'Tan', 'Asin', 'Acos', 'Atan', 'Sinh', 'Cosh', 'Tanh', 'Asinh', 'Acosh', 'Atanh',
     'Exp', 'Log', 'Neg', 'Sqrt', 'Ceil', 'Floor', 'Round'):
        Transform(
            type=('nn.relu', 'nn.sigmoid', 'nn.softplus', 'nn.selu', 'math.not', 'math.iden', 'nn.elu', 'nn.erf',
                  'math.abs', 'math.sign', 'math.sin', 'math.cos', 'math.tan', 'math.asin', 'math.acos', 'math.atan',
                  'math.sinh', 'math.cosh', 'math.tanh', 'math.asinh', 'math.acosh', 'math.atanh',
                  'math.exp', 'math.log', 'math.neg', 'math.sqrt', 'math.ceil', 'math.floor', 'math.round'),
            inputs='!I[0]',
            outputs='!O[0]',
        ),
    ('Add', 'Sub', 'Mul', 'Div', 'Mod', 'Pow', 'Min', 'Max',
     'And', 'Or', 'Xor', 'Equal', 'Less', 'Greater', 'LessOrEqual', 'GreaterOrEqual'):
        Transform(
            type=('math.add', 'math.sub', 'math.mul', 'math.div', 'math.mod', 'math.pow', 'math.min', 'math.max',
                  'math.and', 'math.or', 'math.xor', 'math.eq', 'math.lt', 'math.gt', 'math.le', 'math.ge'),
            inputs=(
                '!I[0]',
                '!I[1]',
            ),
            outputs='!O[0]',
        ),
    'Gelu':
        Transform(
            type='nn.gelu',
            defaults={
                'approximate': 'none',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'approximate': '!approximate.upper() if approximate != "none" else None',
            }
        ),
    'LeakyRelu':
        Transform(
            type='nn.relu',
            defaults={
                'alpha': 0.01,
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'alpha': '!alpha',
            }
        ),
    'PRelu':
        Transform(
            type='nn.prelu',
            inputs=(
                '!I[0]',
                '!squeeze_param(I[1], axes=list(range(1,len(I[1].data.shape)))) if I[1].data is not None else I[1]',
            ),
            outputs='!O[0]',
        ),
    'ThresholdedRelu':
        Transform(
            type='nn.thresholded_relu',
            defaults={
                'alpha': 1.0,
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'theta': '!alpha',
            },
        ),
    'Transpose':
        Transform(
            type='layout.transpose',
            defaults={
                'perm': None,
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'perm': '!perm',
                'axis': 0,
            }
        ),
    'Reshape':
        Transform(
            type='layout.reshape',
            defaults={
                'shape': '!arg_as_attrib(I[1])',
            },
            using={
                'symbolic': '!not is_const(I[1])',
                'axis': '!leading_zeros(shape) if not symbolic else None',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axis': '!axis',
                'shape': '!shape[axis:] if not symbolic else shape',
            }
        ),
    'Flatten':
        Transform(
            type='layout.flatten',
            defaults={
                'axis': 1,
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axis': '!axis'
            }
        ),
    'Squeeze':
        Transform(
            type='layout.squeeze',
            defaults={
                'axes': '!arg_as_attrib(I[1])',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
            }
        ),
    'Unsqueeze':
        Transform(
            type='layout.unsqueeze',
            defaults={
                'axes': '!arg_as_attrib(I[1])',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
            }
        ),
    'MatMul':
        Transform(
            type='linalg.matmul',
            inputs=(
                '!I[0]',
                '!I[1]',
            ),
            outputs=(
                '!O[0]',
            ),
        ),
    'Gemm':
        Transform(
            type='!"nn.linear" if is_linear else "linalg.matmul"',
            defaults={
                'alpha': 1.0,
                'beta': 1.0,
                'transA': 0,
                'transB': 0,
            },
            cond={
                '!alpha == 1.0': 'alpha must be 1',
                '!beta == 1.0 or len(I) == 2': 'beta must be 1',
            },
            using={
                'is_linear': '!len(I) > 2 and I[2].rank == 1 and not transA',
            },
            inputs=(
                '!I[0]',
                '!transpose_matrix(I[1]) if is_linear and not transB else I[1]',
                '!I[2] if len(I) > 2 else None',
            ),
            outputs='!O[0]',
            attribs={
                'transA': '!bool(transA) if not is_linear else None',
                'transB': '!bool(transB) if not is_linear else None',
            }
        ),
    'LRN':
        Transform(
            type='nn.local_response_norm',
            defaults={
                'alpha': 0.0001,
                'beta': 0.75,
                'bias': 1.0,
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'alpha': '!alpha',
                'beta': '!beta',
                'bias': '!bias',
                'axes': [1],
                'size': '!size',
            }
        ),
    'Concat':
        Transform(
            type='layout.concat',
            defaults={
                'axis': 1,
            },
            inputs='!list(I)',
            outputs='!O[0]',
            attribs={
                'axis': '!axis',
            }
        ),
    'Split':
        Transform(
            type='layout.split',
            defaults={
                'axis': 0,
                'num_outputs': '!len(O)',
                'split': '!arg_as_attrib(I[1]) if len(I) > 1 else None',
            },
            inputs='!I[0]',
            outputs='!list(O)',
            attribs={
                'axis': '!axis',
                'count': '!num_outputs',
                'sizes': '!split',
            }
        ),
    'Dropout':
        Transform(
            type='math.iden',
            inputs='!I[0]',
            outputs='!O[0]',
        ),
    'Softmax':
        Transform(
            type='!"nn.softmax" if _version_ >= 13 or known_rank else "onnx_softmax"',
            using={
                'known_rank': '!I[0].rank != None',
            },
            defaults={
                'axis': '!1 if _version_ < 13 else -1',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '![axis] if _version_ >= 13 else list(range(axis, I[0].rank)) if known_rank else None',
                'axis': '!axis if _version_ < 13 and not known_rank else None',
            }
        ),
    'Sum':
        Transform(
            type='math.sum_n',
            inputs='!list(I)',
            outputs='!O[0]',
        ),
    'Where':
        Transform(
            type='math.select',
            inputs=(
                '!I[0]',
                '!I[1]',
                '!I[2]',
            ),
            outputs='!O[0]',
        ),
    'Clip':
        Transform(
            type='!"math.max" if I[2].name == "" else "math.min" if I[1].name == "" else "math.clamp"',
            inputs=(
                '!I[0]',
                '!I[1] if I[1].name != "" else None',
                '!I[2] if I[2].name != "" else None',
            ),
            outputs='!O[0]',
        ),
    'Pad':
        Transform(
            type='layout.pad',
            defaults={
                'mode': "constant",
                'value': 0.0,
            },
            inputs='!(I[0], I[2]) if len(I) > 2 else I[0]',
            outputs='!O[0]',
            attribs={
                'padding': '!arg_as_attrib(I[1]) if len(I) > 1 else pads',
                'method': '!"REPLICATE" if mode == "edge" else mode.upper()',
                'axes': '!arg_as_attrib(I[3]) if len(I) > 3 else None',
            }
        ),
    'Tile':
        Transform(
            type='layout.tile',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'repeats': '!arg_as_attrib(I[1])',
            }
        ),
    'Expand':
        Transform(
            type='!"layout.uniform" if I[0].rank == 0 else "layout.broadcast"',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'shape': '!arg_as_attrib(I[1])',
            }
        ),
    'Slice':
        Transform(
            type='layout.slice',
            defaults={
                'starts': '!arg_as_attrib(I[1], convert_int_inf=True) if len(I) > 1 else None',
                'ends': '!arg_as_attrib(I[2], convert_int_inf=True) if len(I) > 2 else None',
                'axes': '!arg_as_attrib(I[3]) if len(I) > 3 else None',
                'steps': '!arg_as_attrib(I[4]) if len(I) > 4 else None',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
                'begin': '!starts',
                'end': '!ends',
                'stride': '!steps',
            }
        ),
    'LpNormalization':
        Transform(
            type='!"nn.l1_norm" if p == 1 else "nn.l2_norm"',
            defaults={
                'axis': -1,
                'p': 2,
            },
            cond={
                '!p == 1 or p == 2': 'p must be 1 or 2',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '![axis]',
            }
        ),
    'MeanVarianceNormalization':
        Transform(
            type='nn.mean_variance_norm',
            defaults={
                'axes': [0, 2, 3],
            },
            inputs=(
                '!I[0]',
            ),
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
                'epsilon': 0.0,
            }
        ),
    'InstanceNormalization':
        Transform(
            type='nn.mean_variance_norm',
            defaults={
                'epsilon': 1e-5,
            },
            inputs=(
                '!I[0]',
                '!I[2]',
                '!I[1]',
            ),
            outputs='!O[0]',
            attribs={
                'epsilon': '!epsilon',
            }
        ),
    'Upsample':
        Transform(
            type='!"image.nearest_upsample" if mode == "nearest" else "image.linear_upsample"',
            defaults={
                'mode': "nearest",
                'scales': '!arg_as_attrib(I[1])',
            },
            cond={
                '!all(int(s) == s for s in scales)': 'scales must be integers in all dimensions',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'factor': '![int(s) for s in scales]',
                'symmetric': '!False if mode == "linear" else None',
            }
        ),
    'Resize':
        Transform(
            type='!"image.rescale" if scales is not None else "image.resize"',
            defaults={
                'mode': "nearest",
                'nearest_mode': "round_prefer_floor",
                'coordinate_transformation_mode': "half_pixel",
                'exclude_outside': 0,
                'cubic_coeff_a': -0.75,
                'antialias': 0,
                'axes': None,
            },
            using=OrderedDict([
                ('scales_idx', '!2 if _version_ >= 11 else 1'),
                ('sizes_idx', '!3 if _version_ >= 11 else 2'),
                ('roi', '!arg_as_attrib(I[1]) if _version_ >= 11 and len(I) > 1 and I[1].name != "" else None'),
                ('scales', '!arg_as_attrib(I[scales_idx]) if len(I) > scales_idx and I[scales_idx].name != "" else None'),
                ('sizes', '!arg_as_attrib(I[sizes_idx]) if len(I) > sizes_idx and I[sizes_idx].name != "" else None'),
            ]),
            cond={
                '!coordinate_transformation_mode == "half_pixel" or'
                ' coordinate_transformation_mode == "pytorch_half_pixel" or'
                ' coordinate_transformation_mode == "asymmetric" or '
                ' coordinate_transformation_mode == "align_corners"':
                    'coordinate_transformation_mode must be one of'
                    ' "half_pixel", "pytorch_half_pixel", "asymmetric", "align_corners"',
                '!exclude_outside == 0': 'exclude_outside == 1 is not supported',
                '!antialias == 0': 'antialias == 1 is not supported',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
                'size': '!sizes',
                'factor': '!scales',
                'mode': '!mode.upper()',
                'rounding_method': '!nearest_mode.upper() if mode == "nearest" else None',
                'coordinate_transform': '!coordinate_transform(coordinate_transformation_mode)',
                'antialias': '!bool(antialias) if mode == "linear" or mode == "cubic" else None',
                'cubic_coeff_a': '!cubic_coeff_a if mode == "cubic" else None',
            }
        ),
    'Gather':
        Transform(
            type='layout.gather',
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            defaults={
                'axis': 0,
            },
            attribs={
                'axis': '!axis',
            },
        ),
    'Scatter':
        Transform(
            type='layout.scatter',
            inputs=('!I[0]', '!I[1]', '!I[2]'),
            outputs='!O[0]',
            defaults={
                'axis': 0,
            },
            attribs={
                'axis': '!axis',
            },
        ),
    'GatherND':
        Transform(
            type='layout.gather_nd',
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            defaults={
                'batch_dims': 0,
            },
            attribs={
                'batch_dims': '!batch_dims',
            },
        ),
    'ScatterND':
        Transform(
            type='layout.scatter_nd',
            inputs=('!I[0]', '!I[1]', '!I[2]'),
            outputs='!O[0]',
        ),
    'Cast':
        Transform(
            type='!"layout.cast<" + ts_dtype(O[0].dtype) + ">"',
            inputs='!I[0]',
            outputs='!O[0]',
        ),
    'If':
        Transform(
            type='if',
            inputs='!tuple(I)',
            outputs='!tuple(O)',
            using={
                'then_inputs': '!range(1, _implicit_input_count_[0] + 1)',
                'else_inputs': '!range(1 + _implicit_input_count_[0], 1 + _implicit_input_count_[0] + _implicit_input_count_[1])',
            },
            attribs={
                'cond_graphs': '![I[0]]',
                'branch_graphs': '![then_branch, else_branch]',
                'cond_inputs': [0],
                'branch_inputs': '![*then_inputs, *else_inputs]',
            },
        ),
    'Loop':
        Transform(
            type='do',
            cond={
                '!has_count or num_scan_outputs == 0': 'loop must have a (valid) range if it has scan outputs',
            },
            using={
                'has_count': '!I[0].name != "" and not is_const_int_max(I[0])',
                'has_cond': '!I[1].name != "" and not is_const_true_cond(body.inputs[1], body.outputs[0], I[1])',
                'num_deps': '!len(I) - 2 - _implicit_input_count_',
                'num_scan_outputs': '!len(O) - num_deps',
            },
            inputs='!tupled(I[1], has_cond) + I[2:] + (I[0] if has_count else None,)',
            outputs='!tuple(O[:num_deps]) + tuple(stack_output(output) for output in O[num_deps:])',
            attribs={
                'cond_graph': '!body.inputs[1] if has_cond else None',
                'cond_inputs': '![0] if has_cond else []',
                'body_graph': '!body',
                'body_inputs': '!list(range(int(has_cond) + num_deps + 1 + _implicit_input_count_))',
                'iters': '!arg_as_attrib(I[0], none_on_failure=True) if has_count else None',
                'pretest': '!True if has_cond else None',
                'nvars': '!len(I) - (1 if has_cond else 2) - _implicit_input_count_',
                'nscans': 0,
            },
        ),
    'LSTM':
        Transform(
            cond={
                '!direction == "forward"': 'direction must be "forward"',
            },
            defaults={
                'layout': 0,
            },
            using={
                'dir_axis': '!0 if layout == 0 else 1',
            },
            type='nn.lstm',
            inputs=(
                '!I[0]',                                    # X
                '!squeeze_input(I[1], axes=[0])',           # W
                '!squeeze_input(I[2], axes=[0])',           # R
                '!squeeze_input(conflate_lstm_bias(I[3]), axes=[0])', # B
                '!squeeze_input(I[5], axes=[dir_axis])',    # h_0
                '!squeeze_input(I[6], axes=[dir_axis])',    # c_0
                '!I[4] if len(I) > 4 and len(I[4].name) else None',   # lens
            ),
            outputs=(
                '!unsqueeze_output(O[0], axes=[dir_axis + 1])',  # Y
                '!unsqueeze_output(O[1], axes=[dir_axis])',      # h_n
                '!unsqueeze_output(O[2], axes=[dir_axis])',      # c_n
            ),
        ),
    'NonZero':
        Transform(
            type='layout.nonzero',
            inputs='!I[0]',
            outputs='!O[0]',
        ),
    'TopK':
        Transform(
            type='algo.top_k',
            defaults={
                'k': '!arg_as_attrib(I[1], as_scalar=True)',
                'axis': -1,
                'largest': True,
                'sorted': True,
            },
            inputs='!I[0]',
            outputs=('!O[0]', '!O[1]'),
            attribs={
                'k': '!k',
                'axis': '!axis',
                'largest': '!bool(largest)',
                'sorted': '!bool(sorted)',
            },
        ),
    'NonMaxSuppression':
        Transform(
            type='algo.nonmax_suppress',
            defaults={
                'center_point_box': 0,
            },
            using={
                'max_output_boxes_per_class': '!(arg_as_attrib(I[2], as_scalar=True) if not is_const_int_max(I[2]) else None)'
                                              ' if len(I) > 2 and I[2].name != "" else 0',
                'iou_threshold': '!arg_as_attrib(I[3], as_scalar=True) if len(I) > 3 and I[3].name != "" else 0.0',
                'score_threshold': '!arg_as_attrib(I[4], as_scalar=True) if len(I) > 4 and I[4].name != "" else None',
            },
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            attribs={
                'box_format': '!"CENTER" if center_point_box == 1 else "CORNERS"',
                'max_outputs_per_class': '!max_output_boxes_per_class',
                'iou_threshold': '!iou_threshold',
                'score_threshold': '!score_threshold',
            },
        ),
    'ConstantOfShape':
        Transform(
            type='layout.constant',
            inputs=(),
            outputs='!O[0]',
            attribs={
                'shape': '!arg_as_attrib(I[0])',
                'value': '!value.item() if isinstance(value, np.ndarray) else value',
            }
        ),
    'Range':
        Transform(
            type='layout.range',
            inputs=(),
            outputs='!O[0]',
            attribs={
                'first': '!arg_as_attrib(I[0])',
                'last': '!arg_as_attrib(I[1])',
                'stride': '!arg_as_attrib(I[2]) if len(I) > 2 else None',
            },
        ),
    'Shape':
        Transform(
            type='layout.shape',
            inputs='!I[0]',
            outputs='!O[0]',
        ),
})
