from __future__ import division, print_function, absolute_import
from .converter import ConverterToSkriptND as _Converter, Transform, ConversionError, ShapeExpr, _INT_MAX
from ..model.utils import ensure_valid_ids, generate_missing_tensor_names_from_op_type
from ..utils import types
from collections import OrderedDict
import numpy as np


class Converter(_Converter):

    @staticmethod
    def defined_operations():
        return {
        }

    @staticmethod
    def defined_imports():
        return {'nn', 'math'}

    @staticmethod
    def shape_expr_args(op_type):
        return Converter.ShapeExprArgs.get(op_type, [])

    ShapeExprArgs = {
        'Conv2DBackpropInput': [0],
        'Conv3DBackpropInput': [0],
        'DepthwiseConv2dNativeBackpropInput': [0],
        'Concat': [0],
        'ConcatV2': [-1],
        'Split': [0],
        'SplitV': [1, 2],
        'Reshape': [1],
        'Transpose': [1],
        'ExpandDims': [1],
        'Min': [1],
        'Max': [1],
        'Mean': [1],
        'Sum': [1],
        'Any': [1],
        'All': [1],
        'Pad': [1],
        'MirrorPad': [1],
        'Slice': [1],
        'Tile': [1],
        'ArgMin': [1],
        'ArgMax': [1],
        'ResizeArea': [1],
        'ResizeBilinear': [1],
        'ResizeNearestNeighbor': [1],
        'Gather': [2],
        'GatherV2': [2],
    }

    def __init__(self, custom_transforms=None, custom_functions=None, mirror_unsupported=False):
        _Converter.__init__(self, transforms=self.merge_transforms(_Transforms, custom_transforms),
                            functions=custom_functions, mirror_unsupported=mirror_unsupported)

    def __call__(self, model):
        self._eliminate_placeholder_ops(model)
        self._eliminate_constant_ops(model)
        self._collect_shape_ops(model)
        model = _Converter.__call__(self, model)
        self._add_zero_copy_for_constant_outputs(model)
        self._eliminate_empty_subgraphs(model)
        self._remove_unused_constants(model)
        self._fix_shape_expr_args(model)
        ensure_valid_ids(model)
        generate_missing_tensor_names_from_op_type(model)
        return model

    def should_skip_conversion(self, op):
        return op in self._shape_ops

    def _eliminate_placeholder_ops(self, model):
        for graph in model.graphs:
            removed = [op for op in graph.operations if op.type == 'Placeholder']
            graph.remove_operations(removed, unlink=True)

    def _eliminate_constant_ops(self, model):
        for graph in model.graphs:
            removed = []
            for op in graph.operations:
                if op.type == 'Const':
                    value = op.attribs['value']
                    op.output.set_data(value)
                    removed.append(op)

            graph.remove_operations(removed, unlink=True)

    def _global_attribs(self):
        return {'_lite_': False}

    def _is_constant(self, tensor):
        if tensor.producer:
            return tensor.producer.type == 'Const'
        else:
            return tensor.data is not None

    def _read_constant(self, tensor, type=None):
        if tensor.producer is None:
            return types.from_numpy(tensor.data, type=type)
        elif tensor.producer.type == 'Const':
            value = tensor.producer.attribs['value']
            return types.from_numpy(value, type=type) if isinstance(value, np.ndarray) else types.cast(value, type=type)
        else:
            raise ConversionError('trying to evaluate non-constant tensor')

    def is_nxc(self, format):
        return format[0] == 'N' and format[-1] == 'C' and len(format) > 2

    def is_ncx(self, format):
        return format[0] == 'N' and format[1] == 'C' and len(format) > 2

    def is_cxn(self, format):
        return format[0] == 'C' and format[-1] == 'N' and len(format) > 2

    def is_xcn(self, format):
        return format[-2] == 'C' and format[-1] == 'N' and len(format) > 2

    def reshape_depthwise_filter(self, tensor):
        shape = tensor.shape[:-2] + (1, -1)
        if tensor.data is not None:
            tensor.data = np.reshape(tensor.data, shape)
            tensor.shape = tensor.data.shape
            return tensor
        else:
            return self._reshape(tensor, shape)

    def convert_size(self, value, format):
        return value[1:-1] if self.is_nxc(format) else value[2:]

    def convert_padding(self, padding, rank, explicit_paddings=None, format=None):
        padding = padding.upper()
        if padding == 'SAME':
            return None
        elif padding == 'VALID':
            return [0, 0] * rank
        elif padding == 'EXPLICIT':
            assert explicit_paddings is not None and format is not None
            before = explicit_paddings[0::2]
            after = explicit_paddings[1::2]
            return before[1:-1] + after[1:-1] if self.is_nxc(format) else before[2:] + after[2:]
        else:
            assert False, "unknown padding type '{}'".format(padding)

    def squeeze_input(self, tensor, axes, keep_dims=False):
        return self._pre_squeeze(tensor, axes=axes) if not keep_dims else tensor

    def squeeze_output(self, tensor, axes, keep_dims=False):
        return self._post_squeeze(tensor, axes=axes) if not keep_dims else tensor

    def unsqueeze_input(self, tensor, axes, keep_dims=False):
        return self._pre_unsqueeze(tensor, axes=axes) if not keep_dims else tensor

    def unsqueeze_output(self, tensor, axes, keep_dims=False):
        return self._post_unsqueeze(tensor, axes=axes) if not keep_dims else tensor

    def unsqueeze_vector(self, tensor):
        original = self._tensor_map[tensor]
        if self._is_constant(original) and len(original.consumers) == 1:
            self._transform_constant(tensor, lambda data: np.expand_dims(data, 0))
            return tensor
        else:
            return self.unsqueeze_input(tensor, axes=[0])

    def ensure_list(self, value):
        return value if isinstance(value, list) else list(value) if isinstance(value, tuple) else [value]

    def is_bit_set(self, mask, idx):
        return mask & (1 << idx) != 0

    def bit_count(self, mask):
        count = 0
        for i in range(mask.bit_length()):
            if self.is_bit_set(mask, i):
                count += 1
        return count

    def replace_item_with(self, items, index, count, value):
        return items[:index] + [value] * count + items[index+1:]

    def replace_bit_with(self, mask, index, count, value):
        value_bits = (((1 << count) - 1) << index) if value else 0
        low_bits = mask & ((1 << index) - 1)
        high_bits = (mask & ~((1 << (index + 1)) - 1)) << (count - 1)
        return low_bits | value_bits | high_bits

    def beg_index(self, stride):
        return _INT_MAX if stride < 0 else 0

    def end_index(self, stride):
        return _INT_MAX if stride > 0 else -_INT_MAX

    def is_shape_expr(self, arg):
        return isinstance(arg, ShapeExpr)

    def _eval_symbolic_shape(self, tensor):
        op = tensor.producer
        if tensor.data is not None:
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
        else:
            inputs = [self._eval_symbolic_shape(input) for input in op.inputs]
            if op.type == 'Add' or op.type == 'AddV2':
                return ShapeExpr(ShapeExpr.Op.Add, args=inputs)
            elif op.type == 'Sub':
                return ShapeExpr(ShapeExpr.Op.Sub, args=inputs)
            elif op.type == 'Mul':
                return ShapeExpr(ShapeExpr.Op.Mul, args=inputs)
            elif op.type == 'RealDiv':
                return ShapeExpr(ShapeExpr.Op.Div, args=inputs)
            elif op.type == 'Minimum':
                return ShapeExpr(ShapeExpr.Op.Min, args=inputs)
            elif op.type == 'Maximum':
                return ShapeExpr(ShapeExpr.Op.Max, args=inputs)
            elif op.type == 'Less':
                return ShapeExpr(ShapeExpr.Op.Less, args=inputs)
            elif op.type == 'Greater':
                return ShapeExpr(ShapeExpr.Op.Greater, args=inputs)
            elif op.type == 'LessEqual':
                return ShapeExpr(ShapeExpr.Op.LessEqual, args=inputs)
            elif op.type == 'GreaterEqual':
                return ShapeExpr(ShapeExpr.Op.GreaterEqual, args=inputs)
            elif op.type == 'Equal':
                return ShapeExpr(ShapeExpr.Op.Equal, args=inputs)
            elif op.type == 'NotEqual':
                return ShapeExpr(ShapeExpr.Op.NotEqual, args=inputs)
            elif op.type == 'LogicalAnd':
                return ShapeExpr(ShapeExpr.Op.And, args=inputs)
            elif op.type == 'LogicalOr':
                return ShapeExpr(ShapeExpr.Op.Or, args=inputs)
            elif op.type == 'LogicalXor':
                return ShapeExpr(ShapeExpr.Op.Xor, args=inputs)
            elif op.type == 'Ceil':
                assert inputs[0].op == ShapeExpr.Op.Div, (f"'Ceil' op can only be converted when preceded by 'Div'"
                                                          f" op in shape expression; found operation '{op.name}' preceded"
                                                          f" by operation '{inputs[0].op.name}' of type '{inputs[0].op.type}'")
                inputs[0].op = ShapeExpr.Op.CeilDiv
                return inputs[0]
            elif op.type == 'Gather' or op.type == 'GatherV2':
                assert inputs[1].rank <= 1, f"Operation '{op.name}' of type '{op.type}' must have index of rank <= 1 in shape expression"
                return ShapeExpr(ShapeExpr.Op.Subscript, args=[inputs[0], inputs[1]])
            elif op.type == 'Concat':
                return ShapeExpr(ShapeExpr.Op.Concat, args=inputs[1:])
            elif op.type == 'Pack':
                return ShapeExpr(ShapeExpr.Op.Pack, args=inputs)
            elif op.type == 'StridedSlice':
                assert inputs[1].effective_rank == 0, f"Operation '{op.name}' of type 'StridedSlice' must have 'starts' of length 1"
                assert inputs[2].effective_rank == 0, f"Operation '{op.name}' of type 'StridedSlice' must have 'ends' of length 1"
                assert inputs[3].effective_rank == 0, f"Operation '{op.name}' of type 'StridedSlice' must have 'strides' of length 1"
                assert self._ensure_zero_rank(inputs[3]).is_const(1)
                beg = self._ensure_zero_rank(inputs[1])
                end = self._ensure_zero_rank(inputs[2])
                squeeze = op.attribs.get('shrink_axis_mask', 0) != 0
                expr = ShapeExpr(ShapeExpr.Op.Slice, args=[inputs[0], beg, end])
                return ShapeExpr(ShapeExpr.Op.DownRank, args=[inputs[0], 1]) if squeeze else expr
            elif op.type == 'Squeeze':
                axes = op.attribs.get('squeeze_dims')
                if axes is None:
                    axes = [i for i, x in enumerate(op.inputs[0].shape) if x == 1]
                return ShapeExpr(ShapeExpr.Op.DownRank, args=[inputs[0], len(axes)])
            elif op.type == 'ExpandDims':
                axes = self.as_list(self.as_const(op.inputs[1]))
                return ShapeExpr(ShapeExpr.Op.UpRank, args=[inputs[0], len(axes)])
            elif op.type == 'Cast':
                output_type = self.sknd_dtype(op.output.dtype)
                return ShapeExpr(ShapeExpr.Op.Cast, args=[inputs[0], output_type])
            elif op.type == 'Tile':
                assert inputs[0].effective_rank == 0, f"Operation '{op.name}' of type 'Tile' must have input of singular dimensions"
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
            elif op.type == 'Sum':
                expr = ShapeExpr(ShapeExpr.Op.Sum, args=[inputs[0]])
                return expr if op.attribs.get('keepdims') == 0 else \
                    ShapeExpr(ShapeExpr.Op.UpRank, args=[expr, 1])
            elif op.type == 'Prod':
                expr = ShapeExpr(ShapeExpr.Op.Prod, args=[inputs[0]])
                return expr if op.attribs.get('keepdims') == 0 else \
                    ShapeExpr(ShapeExpr.Op.UpRank, args=[expr, 1])
            elif op.type == 'Min':
                expr = ShapeExpr(ShapeExpr.Op.Minimize, args=[inputs[0]])
                return expr if op.attribs.get('keepdims') == 0 else \
                    ShapeExpr(ShapeExpr.Op.UpRank, args=[expr, 1])
            elif op.type == 'Max':
                expr = ShapeExpr(ShapeExpr.Op.Maximize, args=[inputs[0]])
                return expr if op.attribs.get('keepdims') == 0 else \
                    ShapeExpr(ShapeExpr.Op.UpRank, args=[expr, 1])
            else:
                raise ConversionError(f"Conversion of operator '{op.name}' of type '{op.type}' "
                                      f"as a symbolic shape expression is not implemented")


_Transforms = Converter.unpack_transforms({
    ('Conv1D', 'Conv2D', 'Conv3D', 'DepthwiseConv2dNative'):
        Transform(
            type='nn.conv',
            using={
                'depthwise': '!_type_ == "DepthwiseConv2dNative"',
                'groups': '!I[1].shape[-2] if depthwise else None',
            },
            defaults={
                'explicit_paddings': [],
            },
            inputs=(
                '!I[0]',
                '!reshape_depthwise_filter(I[1]) if depthwise else I[1]',
            ),
            outputs=(
                '!O[0]',
            ),
            attribs={
                'stride': '!convert_size(strides, data_format)',
                'dilation': '!convert_size(dilations, data_format)',
                'padding': '!convert_padding(padding, I[0].rank - 2, explicit_paddings, data_format)',
                'groups': '!groups',
                'data_format': '!"NXC" if is_nxc(data_format) else "NCX"',
                'filter_format': "XCN",
                'ceil_mode': '!padding == "SAME"',
            }
        ),
    ('Conv2DBackpropInput', 'Conv3DBackpropInput', 'DepthwiseConv2dNativeBackpropInput'):
        Transform(
            type='nn.deconv',
            using={
                'depthwise': '!_type_ == "DepthwiseConv2dNativeBackpropInput"',
                'groups': '!I[1].shape[-2] if depthwise else None',
            },
            defaults={
                'explicit_paddings': [],
                'output_shape': '!arg_as_attrib(I[0])',
            },
            inputs=(
                '!I[2]',
                '!reshape_depthwise_filter(I[1]) if depthwise else I[1]',
            ),
            outputs=(
                '!O[0]',
            ),
            attribs={
                'stride': '!convert_size(strides, data_format)',
                'dilation': '!convert_size(dilations, data_format)',
                'padding': '!convert_padding(padding, I[2].rank - 2, explicit_paddings, data_format)',
                'output_size': '!output_shape[1:-1] if is_nxc(data_format) else output_shape[2:]',
                'groups': '!groups',
                'data_format': '!"NXC" if is_nxc(data_format) else "NCX"',
                'filter_format': "XCN",
            }
        ),
    ('MaxPool', 'AvgPool'):
        Transform(
            type=('nn.max_pool', 'nn.avg_pool'),
            defaults={
                'explicit_paddings': [],
                'data_format': 'NHWC',
            },
            inputs=(
                '!I[0]',
            ),
            outputs=(
                '!O[0]',
            ),
            attribs={
                'axes': '!list(range(1, I[0].rank - 1)) if is_nxc(data_format) else None',
                'size': '!ksize[1:-1] if is_nxc(data_format) else ksize[2:]',
                'stride': '!strides[1:-1] if is_nxc(data_format) else strides[2:]',
                'padding': '!convert_padding(padding, I[0].rank - 2, explicit_paddings, data_format)',
                'ceil_mode': '!padding == "SAME"',
            }
        ),
    'Concat':
        Transform(
            type='layout.concat',
            inputs='!list(I[1:])',
            outputs='!O[0]',
            attribs={
                'axis': '!arg_as_attrib(I[0])',
            }
        ),
    'ConcatV2':
        Transform(
            type='layout.concat',
            inputs='!list(I[:-1])',
            outputs='!O[0]',
            attribs={
                'axis': '!arg_as_attrib(I[-1])',
            }
        ),
    'Split':
        Transform(
            type='layout.split',
            inputs='!I[1]',
            outputs='!list(O)',
            attribs={
                'axis': '!arg_as_attrib(I[0])',
                'count': '!num_split if not _lite_ else num_splits',
            }
        ),
    'SplitV':
        Transform(
            type='layout.split',
            inputs='!I[0]',
            outputs='!list(O)',
            attribs={
                'axis': '!arg_as_attrib(I[2])',
                'sizes': '!arg_as_attrib(I[1])',
            }
        ),
    'Reshape':
        Transform(
            type='layout.reshape',
            inputs='!I[0]',
            outputs='!O[0]',
            using={
                'shape': '!arg_as_attrib(I[1])',
                'axis': '!leading_zeros(shape) if not is_shape_expr(shape) else 0',
            },
            attribs={
                'shape': '!shape[axis:] if axis != 0 else shape',
                'axis': '!axis if axis != 0 else None',
            }
        ),
    'Transpose':
        Transform(
            type='layout.transpose',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'perm': '!arg_as_attrib(I[1])',
            }
        ),
    'Squeeze':
        Transform(
            type='layout.squeeze',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!ensure_list(squeeze_dims) if len(squeeze_dims) != 0 else'
                        ' [i for i, x in enumerate(I[0].shape) if x == 1]',
            }
        ),
    'ExpandDims':
        Transform(
            type='layout.unsqueeze',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!ensure_list(arg_as_attrib(I[1]))',
            }
        ),
    'Pack':
        Transform(
            type='layout.stack',
            inputs='!list(I)',
            outputs='!O[0]',
            attribs={
                'axis': '!axis',
            }
        ),
    'Unpack':
        Transform(
            type='layout.unstack',
            inputs='!I[0]',
            outputs='!list(O)',
            attribs={
                'axis': '!axis',
            }
        ),
    ('Min', 'Max', 'Mean', 'Sum', 'Prod', 'Any', 'All'):
        Transform(
            type=('math.min_reduce', 'math.max_reduce', 'math.mean_reduce', 'math.sum_reduce', 'math.prod_reduce',
                  'math.any_reduce', 'math.all_reduce'),
            using={
                'axes': '!ensure_list(arg_as_attrib(I[1]))'
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!axes',
                'squeeze': '!not keep_dims',
            }
        ),
    ('Add', 'AddV2', 'Sub', 'Mul', 'RealDiv', 'Pow', 'Minimum', 'Maximum',
     'LogicalAnd', 'LogicalOr', 'LogicalXor', 'Less', 'Greater', 'LessEqual', 'GreaterEqual', 'Equal', 'NotEqual'):
        Transform(
            type=('math.add', 'math.add', 'math.sub', 'math.mul', 'math.div', 'math.pow', 'math.min', 'math.max',
                  'math.and', 'math.or', 'math.xor', 'math.lt', 'math.gt', 'math.le', 'math.ge', 'math.eq', 'math.ne'),
            inputs=(
                '!I[0]',
                '!I[1]',
            ),
            outputs='!O[0]',
        ),
    ('Identity', 'Relu', 'Elu', 'Selu', 'Gelu', 'Silu', 'Swish', 'Sigmoid', 'Softplus', 'Exp', 'Log',
     'Sin', 'Cos', 'Tan', 'Asin', 'Acos', 'Atan', 'Sinh', 'Cosh', 'Tanh', 'Asinh', 'Acosh', 'Atanh',
     'Neg', 'Reciprocal', 'Sign', 'Abs', 'Floor', 'Ceil', 'Round', 'Square', 'Sqrt', 'Rsqrt', 'LogicalNot'):
        Transform(
            type=('math.iden', 'nn.relu', 'nn.elu', 'nn.selu', 'nn.gelu', 'nn.silu', 'nn.silu', 'nn.sigmoid', 'nn.softplus', 'math.exp', 'math.log',
                  'math.sin', 'math.cos', 'math.tan', 'math.asin', 'math.acos', 'math.atan',
                  'math.sinh', 'math.cosh', 'math.tanh', 'math.asinh', 'math.acosh', 'math.atanh',
                  'math.neg', 'math.rcp', 'math.sign', 'math.abs', 'math.floor', 'math.ceil', 'math.round',
                  'math.sqr', 'math.sqrt', 'math.rsqrt', 'math.not'),
            inputs='!I[0]',
            outputs='!O[0]',
        ),
    'Relu6':
        Transform(
            type='nn.relu',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'max': '!6.0',
            }
        ),
    'LeakyRelu':
        Transform(
            type='nn.relu',
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'alpha': '!alpha',
            }
        ),
    ('FusedBatchNorm', 'FusedBatchNormV2', 'FusedBatchNormV3'):
        Transform(
            type='nn.batch_norm',
            inputs=(
                '!I[0]',
                '!I[3]',
                '!I[4]',
                '!I[2]',
                '!I[1]',
            ),
            outputs=(
                '!O[0]',
            ),
            attribs={
                'epsilon': '!epsilon',
                'channel_axis': -1,
            }
        ),
    'BiasAdd':
        Transform(
            type='math.add',
            inputs=(
                '!I[0]',
                '!I[1]',
            ),
            outputs='!O[0]',
            attribs={
                'rhs_align': '!1 if is_ncx(data_format) else None',
            },
        ),
    'Softmax':
        Transform(
            type='nn.softmax',
            cond={
                '!beta == 1 if _lite_ else True': 'beta must be 1',
            },
            defaults={
                'axis': -1,
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '![axis]',
            }
        ),
    'MatMul':
        Transform(
            type='linalg.matmul',
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            attribs={
                'transA': '!transpose_a',
                'transB': '!transpose_b',
            },
        ),
    'ClipByValue':
        Transform(
            type='math.clamp',
            inputs=('!I[0]', '!I[1]', '!I[2]'),
            outputs='!O[0]',
        ),
    ('Pad', 'MirrorPad'):
        Transform(
            type='layout.pad',
            cond={
                '!mode in ["CONSTANT", "REFLECT", "SYMMETRIC"]':
                    'mode must be one of "CONSTANT", "REFLECT" or "SYMMETRIC"',
            },
            defaults={
                'mode': 'CONSTANT',
            },
            using={
                'paddings': '!arg_as_attrib(I[1])',
                'before': '![p for p, q in paddings]',
                'after': '![q for p, q in paddings]',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'padding': '!before + after',
                'method': '!mode',
            }
        ),
    'Slice':
        Transform(
            type='layout.slice',
            using={
                'beg': '!arg_as_attrib(I[1])',
                'end': '![x if s == -1 else b + s for b, s, x in zip(as_const(I[1]), as_const(I[2]), O[0].shape)]',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'begin': '!beg',
                'end': '!end',
            }
        ),
    'StridedSlice':
        Transform(
            type='layout.slice',
            using=OrderedDict([
                ('rank', '!I[0].rank + bit_count(new_axis_mask)'),
                ('stride', '!as_const(I[3])'),
                ('ellipsis_index', '!int(math.log2(ellipsis_mask)) if ellipsis_mask != 0 else None'),
                ('ellipsis_count', '!rank - (len(stride) - 1) if ellipsis_mask != 0 else None'),
                ('beg', '!replace_item_with(as_const(I[1]), ellipsis_index, ellipsis_count, 0) '
                        'if ellipsis_index is not None else as_const(I[1])'),
                ('end', '!replace_item_with(as_const(I[2]), ellipsis_index, ellipsis_count, 0) '
                        'if ellipsis_index is not None else as_const(I[2])'),
                ('begin_mask', '!replace_bit_with(begin_mask, ellipsis_index, ellipsis_count, 1) '
                               'if ellipsis_index is not None else begin_mask'),
                ('end_mask', '!replace_bit_with(end_mask, ellipsis_index, ellipsis_count, 1) '
                             'if ellipsis_index is not None else end_mask'),
                ('new_axis_mask', '!replace_bit_with(new_axis_mask, ellipsis_index, ellipsis_count, 0) '
                                  'if ellipsis_index is not None else new_axis_mask'),
                ('shrink_axis_mask', '!replace_bit_with(shrink_axis_mask, ellipsis_index, ellipsis_count, 0) '
                                     'if ellipsis_index is not None else shrink_axis_mask'),
                ('masked_beg', '![beg_index(s) if is_bit_set(begin_mask,i) else b '
                               'for i, (b, s) in enumerate(zip(beg,stride))]'),
                ('masked_end', '![end_index(s) if is_bit_set(end_mask,i) else b + 1 '
                               'if is_bit_set(shrink_axis_mask,i) else e '
                               'for i, (b, e, s) in enumerate(zip(beg,end,stride))]'),
                ('axes', '![i for i in range(rank) if not (is_bit_set(begin_mask,i) and is_bit_set(end_mask,i))'
                         ' and not is_bit_set(new_axis_mask,i)]'),
                ('new_axes', '![i for i in range(rank) if is_bit_set(new_axis_mask,i)]'),
                ('del_axes', '![i for i in range(rank) if is_bit_set(shrink_axis_mask,i)]'),
            ]),
            inputs='!unsqueeze_input(I[0], new_axes) if len(new_axes) else I[0]',
            outputs='!squeeze_output(O[0], del_axes) if len(del_axes) else O[0]',
            attribs={
                'axes': '!axes',
                'begin': '![b for i, b in enumerate(masked_beg) if i in axes]',
                'end': '![e for i, e in enumerate(masked_end) if i in axes]',
                'stride': '![s for i, s in enumerate(stride) if i in axes]',
            }
        ),
    ('ArgMin', 'ArgMax'):
        Transform(
            type=('math.argmin', 'math.argmax'),
            using={
                'axis': '!arg_as_attrib(I[1])'
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axis': '!axis',
                'squeeze': True,
            }
        ),
    'Select':
        Transform(
            type='math.select',
            inputs=(
                '!I[0]',
                '!I[1]',
                '!I[2]',
            ),
            outputs='!O[0]',
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
    'ResizeArea':
        Transform(
            type='image.area_downsample',
            using={
                'old_size': '!I[0].shape[-3:-1]',
                'new_size': '!arg_as_attrib(I[1])',
            },
            cond={
                '!is_integer_downsample(old_size, new_size)': 'area resize must be integer down-sample',
                '!not align_corners': 'align_corners is not supported',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!list(range(I[0].rank - 3, I[0].rank - 1))',
                'factor': '!downsample_factor(old_size, new_size)',
            }
        ),
    ('ResizeBilinear', 'ResizeNearestNeighbor'):
        Transform(
            type=('image.linear_resize', 'image.nearest_resize'),
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '!list(range(I[0].rank - 3, I[0].rank - 1))',
                'size': '!arg_as_attrib(I[1])',
                'coordinate_transform': '!"ALIGNED" if align_corners else "SYMMETRIC" if half_pixel_centers else "ASYMMETRIC"',
            }
        ),
    'LRN':
        Transform(
            type='nn.local_response_norm',
            using={
                'size': '!(radius if _lite_ else depth_radius) * 2 + 1'
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axes': '![-1]',
                'size': '!size',
                'alpha': '!alpha * size',
                'beta': '!beta',
                'bias': '!bias',
            }
        ),
    'Cast':
        Transform(
            using={
                'same_type': '!sknd_dtype(O[0].dtype) == sknd_dtype(I[0].dtype)',
            },
            type='!"" if same_type else "layout.cast"',
            inputs='!I[0]',
            outputs='!O[0]',
            dtypes={
                'R': '!O[0].dtype if not same_type else None',
            },
        ),
    ('Gather', 'GatherV2'):
        Transform(
            type='layout.gather',
            using={
                'axis': '!arg_as_attrib(I[2])'
            },
            inputs=('!I[0]', '!I[1]'),
            outputs='!O[0]',
            attribs={
                'axis': '!axis',
            },
        ),
    'AddN':
        Transform(
            type='math.sum_n',
            inputs='!list(I)',
            outputs='!O[0]'
        ),
})
