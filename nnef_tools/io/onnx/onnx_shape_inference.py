from __future__ import division, print_function, absolute_import

import math
import typing
from functools import partial

import numpy as np

from nnef_tools.core import utils
from nnef_tools.io.onnx.onnx_graph import *
from nnef_tools.shape_inference import shape_inference as infer

_ConstValueByTensorT = typing.Dict[ONNXTensor, np.ndarray]


def propagate(graph, source_shapes=None, source_dtypes=None):
    # type: (ONNXGraph, typing.Dict[str, typing.List[int]], typing.Dict[str, str])->None

    if source_shapes is None:
        source_shapes = {}
    if source_dtypes is None:
        source_dtypes = {}

    graph.sort()

    for tensor in graph.tensors:
        if tensor.name and tensor.name in source_shapes:
            if not _is_compatible(tensor.shape, source_shapes[tensor.name]):
                raise utils.NNEFToolsException(
                    "The specified shape is incompatible with the original shape for {}. {} vs. {}".format(
                        tensor.name, source_shapes[tensor.name], tensor.shape))
            tensor.shape = source_shapes[tensor.name]
        if tensor.name and tensor.name in source_dtypes:
            if not (tensor.dtype is None or tensor.dtype == source_dtypes[tensor.name]):
                raise utils.NNEFToolsException(
                    "The specified dtype is incompatible with the original dtype for {}. {} vs. {}".format(
                        tensor.name, source_dtypes[tensor.name], tensor.dtype))

            tensor.dtype = source_dtypes[tensor.name]
        if len(tensor.producers) == 0 and (tensor.shape is None or -1 in tensor.shape or tensor.dtype is None):
            raise utils.NNEFToolsException(
                "Source tensor '{}' has incomplete dtype or shape: {} {}\n"
                "Please specify it in --input-shape or through the corresponding API.".format(
                    tensor.name, tensor.dtype, tensor.shape))

    for op in graph.operations:
        # Shape prop
        assert op.name in _DefaultPropagators, "No shape propagator for {}".format(op.name)
        propagated_shapes, propagated_dtypes = _DefaultPropagators[op.name](op)
        assert not utils.has_le_0(propagated_shapes)
        assert len(propagated_shapes) == len(propagated_dtypes) == len(op.outputs)
        for new_shape, new_dtype, tensor in zip(propagated_shapes, propagated_dtypes, op.outputs):
            assert _is_compatible(tensor.shape, new_shape)
            tensor.shape = new_shape
            assert tensor.dtype is None or tensor.dtype == new_dtype
            tensor.dtype = new_dtype


def _is_compatible(orig_shape, shape):
    if orig_shape is None:
        return True
    if len(orig_shape) != len(shape):
        return False
    for o, s in zip(orig_shape, shape):
        if o != s and o != -1:
            return False
    return True


def to_nnef_padding(onnx_padding):
    half = len(onnx_padding) // 2
    return list(zip(onnx_padding[:half], onnx_padding[half:]))


def get_concrete_padding(auto_padding, custom_padding, upscaled_shape, filter_shape, stride, dilation):
    if auto_padding in [None, '', 'NOTSET']:
        if custom_padding is None:
            return [(0, 0)] * len(upscaled_shape)
        return to_nnef_padding(custom_padding)
    else:
        assert custom_padding is None
        if auto_padding == 'SAME_UPPER':
            return infer.same_padding(upscaled_input=upscaled_shape,
                                      filter=filter_shape,
                                      stride=stride,
                                      dilation=dilation,
                                      left_bigger=False)
        elif auto_padding == 'SAME_LOWER':
            return infer.same_padding(upscaled_input=upscaled_shape,
                                      filter=filter_shape,
                                      stride=stride,
                                      dilation=dilation,
                                      left_bigger=True)
        elif auto_padding == 'VALID':
            return infer.valid_padding(rank=len(upscaled_shape))
        else:
            assert False, "Unexpected padding type: {}".format(auto_padding)


def evaluate_shape_tensor_simple(tensor):
    # type:(ONNXTensor)->typing.List[int]
    if tensor.data is not None:
        return [utils.anyint_to_int(i) for i in tensor.data.tolist()]
    elif tensor.producer is not None and tensor.producer.name == 'Shape':
        return list(tensor.producer.input.shape)
    else:
        assert False, "Shape tensors must be constant tensors or results of Shape for now."


def evaluate_scalar_int_tensor_simple(tensor):
    # type: (ONNXTensor)->int
    if tensor.data is not None:
        return utils.anyint_to_int(tensor.data.item())
    elif tensor.producer is not None and tensor.producer.name == 'Size':
        return tensor.producer.input.count
    else:
        assert False, "Scalar int tensors must be constant or results of Size to be evaluable for now."


def evaluate_float_list_tensor_simple(tensor):
    # type: (ONNXTensor)->int
    if tensor.data is not None:
        return tensor.data.tolist()
    else:
        assert False, "Float list tensors must be constant to be evaluable for now."


def propagate_first(op, dtype=None):
    # type: (ONNXOperation, typing.Optional[str])->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    return [infer.copy(op.inputs[0].shape)], [op.inputs[0].dtype if dtype is None else dtype]


def propagate_broadcast(op, dtype_from_index=0):
    # type: (ONNXOperation, int)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    return ([infer.elementwise(inputs=[input.shape for input in op.inputs], broadcast=infer.Broadcast.FROM_RIGHT)],
            [op.inputs[dtype_from_index].dtype])


def propagate_broadcast_with_axis(op, dtype=None):
    # type: (ONNXOperation, typing.Optional[str])->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    x, y = op.inputs

    if 'axis' in op.attribs:
        axis = op.attribs['axis']
        if axis < 0:
            axis += x.rank
        return ([infer.elementwise(inputs=[x.shape, axis * [1] + y.shape],
                                   broadcast=infer.Broadcast.FROM_LEFT)],
                [x.dtype])
    else:
        return ([infer.elementwise(inputs=[x.shape, y.shape], broadcast=infer.Broadcast.FROM_RIGHT)],
                [x.dtype if dtype is None else dtype])


def propagate_conv(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    input, filter = op.inputs[:2]

    filter_size = filter.shape[2:]
    stride = op.attribs.get('strides', [1] * len(filter_size))
    dilation = op.attribs.get('dilations', [1] * len(filter_size))
    padding = get_concrete_padding(auto_padding=op.attribs.get('auto_pad'),
                                   custom_padding=op.attribs.get('pads'),
                                   upscaled_shape=input.shape[2:],
                                   filter_shape=filter_size,
                                   stride=stride,
                                   dilation=dilation)

    return [infer.conv(input=input.shape,
                       filter=filter_size,
                       padding=padding,
                       stride=stride,
                       dilation=dilation,
                       groups=op.attribs.get('group', 1),
                       output_channels=filter.shape[0],
                       format=infer.Format.NCHW)], [input.dtype]


def propagate_conv_transpose(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    input, filter = op.inputs[:2]

    if 'output_shape' in op.attribs:
        return [infer.copy(op.attribs['output_shape'])], [input.dtype]

    filter_size = filter.shape[2:]
    stride = op.attribs.get('strides', [1] * len(filter_size))
    dilation = op.attribs.get('dilations', [1] * len(filter_size))
    padding = get_concrete_padding(auto_padding=op.attribs.get('auto_pad'),
                                   custom_padding=op.attribs.get('pads'),
                                   upscaled_shape=input.shape[2:],
                                   filter_shape=filter_size,
                                   stride=stride,
                                   dilation=dilation)
    groups = op.attribs.get('group', 1)
    output_padding = op.attribs.get('output_padding', [0] * len(filter_size))

    return [infer.conv(input=input.shape,
                       filter=filter_size,
                       padding=padding,
                       stride=stride,
                       dilation=dilation,
                       groups=groups,
                       output_channels=filter.shape[1] * groups,
                       format=infer.Format.NCHW,
                       output_padding=list(zip([0] * len(filter_size), output_padding)),
                       deconv=True)], [input.dtype]


def propagate_max_unpool(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    input, index = op.inputs[:2]
    output_shape = (evaluate_shape_tensor_simple(op.inputs[2])
                    if len(op.inputs) >= 3 and not op.inputs[2].is_null else None)

    if output_shape is not None:
        return [infer.copy(output_shape)], [input.dtype]

    filter_size = op.attribs['kernel_shape']
    stride = op.attribs.get('strides', [1] * len(filter_size))
    dilation = [1] * len(filter_size)
    padding = to_nnef_padding(op.attribs.get('pads', [0] * 2 * len(filter_size)))
    output_shape = infer.sliding_window(input=input.shape,
                                        filter=[1, 1] + filter_size,
                                        padding=[(0, 0), (0, 0)] + padding,
                                        stride=[1, 1] + stride,
                                        dilation=[1, 1] + dilation,
                                        upscale=True)

    return [output_shape], [input.dtype]


def propagate_batch_normalization(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    input, scale, bias, mean, variance = op.inputs

    return ([infer.copy(input.shape)] + [[0] if output.shape == [0] else infer.copy(mean.shape)
                                         for output in op.outputs[1:]],
            [input.dtype] * len(op.outputs))


def propagate_pool(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    filter_size = op.attribs['kernel_shape']
    stride = op.attribs.get('strides', [1] * len(filter_size))
    dilation = [1] * len(filter_size)
    padding = get_concrete_padding(auto_padding=op.attribs.get('auto_pad'),
                                   custom_padding=op.attribs.get('pads'),
                                   upscaled_shape=op.input.shape[2:],
                                   filter_shape=filter_size,
                                   stride=stride,
                                   dilation=dilation)

    output_shape = infer.sliding_window(input=op.input.shape,
                                        filter=[1, 1] + filter_size,
                                        padding=[(0, 0), (0, 0)] + padding,
                                        stride=[1, 1] + stride,
                                        dilation=[1, 1] + dilation)

    if len(op.outputs) == 1:
        return [output_shape], [op.input.dtype]
    elif len(op.outputs) == 2:  # for max pool
        return [output_shape, infer.copy(output_shape)], [op.input.dtype, 'INT64']
    else:
        assert False, 'Pooling only supported with 1 or 2 outputs'


def propagate_global_pool(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    return [op.input.shape[:2] + [1, 1]], [op.input.dtype]


def propagate_reshape(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    if 'shape' in op.attribs:
        return [infer.reshape(input=op.input.shape, shape=op.attribs['shape'], zero_means_same=True)], [op.input.dtype]
    else:
        input, shape = op.inputs

        return [infer.reshape(input=input.shape,
                              shape=evaluate_shape_tensor_simple(shape),
                              zero_means_same=True)], [input.dtype]


def propagate_gemm(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    A, B = op.inputs[:2]
    assert all(s == 1 for s in A.shape[2:]) and all(s == 1 for s in B.shape[2:])
    return [infer.matmul(a=A.shape[:2],
                         b=B.shape[:2],
                         transpose_a=bool(op.attribs.get('transA', False)),
                         transpose_b=bool(op.attribs.get('transB', False)))], [A.dtype]


def propagate_dropout(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    return [infer.copy(op.inputs[0].shape) for _ in op.outputs], [op.inputs[0].dtype for _ in op.outputs]


def propagate_concat(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    return ([infer.concat(inputs=[input.shape for input in op.inputs], axis=op.attribs.get('axis', 1))],
            [op.inputs[0].dtype])


def propagate_reduce(op,  # type: ONNXOperation
                     multi_axis,  # type: bool
                     default=None,  # type: typing.Any
                     dtype=None  # type: typing.Optional[str]
                     ):
    # type: (...)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    if default is None:
        default = list(range(op.input.rank))

    if multi_axis:
        if 'axes' in op.attribs:
            axes = list(op.attribs['axes'])
        else:
            axes = list(default)
    else:
        if 'axis' in op.attribs:
            axes = [op.attribs['axis']]
        else:
            axes = list(default)

    return [infer.reduce(input=op.input.shape, axes=axes,
                         squeeze=not op.attribs.get('keepdims', 1))], [dtype if dtype is not None else op.input.dtype]


def propagate_cast(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    types = ['UNDEFINED', 'FLOAT', 'UINT8', 'INT8', 'UINT16', 'INT16', 'INT32', 'INT64', 'STRING', 'BOOL', 'FLOAT16',
             'DOUBLE', 'UINT32', 'UINT64', 'COMPLEX64 ', 'COMPLEX128', 'BFLOAT16']  # TODO move out maybe
    return [infer.copy(op.input.shape)], [op.attribs['to']
                                         if isinstance(op.attribs['to'], str)
                                         else types[op.attribs['to']]]


def propagate_constant_of_shape(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    # value has been changed to int/bool/float type by the io module
    # dtype has been added to attribs by the io module
    return [evaluate_shape_tensor_simple(op.input)], [op.attribs['dtype']]


def propagate_size(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    return [[]], ['INT64']


def propagate_expand(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    input, shape = op.inputs

    return [infer.elementwise(inputs=[input.shape, evaluate_shape_tensor_simple(shape)],
                              broadcast=infer.Broadcast.FROM_RIGHT)], [input.dtype]


def propagate_flatten(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    axis = op.attribs.get('axis', 1)
    if axis < 0:
        axis += op.input.rank
    return [[infer.volume(op.input.shape[:axis]), infer.volume(op.input.shape[axis:])]], [op.input.dtype]


def propagate_gather(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    input, indices = op.inputs

    axis = op.attribs.get('axis', 0)
    if axis < 0:
        axis += input.rank

    output_rank = input.rank + indices.rank - 1

    output_shape = []

    for i in range(output_rank):
        if i < axis:
            output_shape.append(input.shape[i])
        elif i < axis + indices.rank:
            output_shape.append(indices.shape[i - axis])
        else:
            output_shape.append(input.shape[i - indices.rank + 1])

    return [output_shape], [input.dtype]


def propagate_max_roi_pool(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    input, rois = op.inputs

    num_rois = rois.shape[0]
    channels = input.shape[1]
    pooled_shape = op.attribs['pooled_shape']

    return [[num_rois, channels, pooled_shape[0], pooled_shape[1]]], [input.dtype]


def propagate_pad(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    if 'paddings' in op.attribs:
        paddings = op.attribs['paddings']
    elif 'pads' in op.attribs:
        paddings = op.attribs['pads']
    else:
        assert False

    return [infer.pad(input=op.input.shape, padding=to_nnef_padding(paddings))], [op.input.dtype]


def propagate_shape(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    return [[op.input.rank]], ['INT64']


def propagate_slice(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    starts = op.attribs['starts']
    ends = op.attribs['ends']
    axes = op.attribs.get('axes', list(range(len(starts))))

    return [infer.slice(input=op.input.shape,
                        axes=axes,
                        begin=starts,
                        end=ends)], [op.input.dtype]


def propagate_dynamic_slice(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    data, starts, ends, axes = op.inputs
    return [infer.slice(input=data.shape,
                        axes=evaluate_shape_tensor_simple(axes),
                        begin=evaluate_shape_tensor_simple(starts),
                        end=evaluate_shape_tensor_simple(ends))], [data.dtype]


def propagate_split(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    if 'split' in op.attribs:
        return (infer.split(input=op.input.shape, axis=op.attribs.get('axis', 0), sizes=op.attribs['split']),
                [op.input.dtype] * len(op.outputs))
    else:
        return (infer.split(input=op.input.shape, axis=op.attribs.get('axis', 0), num=len(op.outputs)),
                [op.input.dtype] * len(op.outputs))


def propagate_squeeze(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    return [infer.squeeze(input=op.input.shape, axes=op.attribs.get('axes', None))], [op.input.dtype]


def propagate_tile(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    if len(op.inputs) == 3:
        input, tiles, axis = op.inputs

        tiles = evaluate_scalar_int_tensor_simple(tiles)
        axis = evaluate_scalar_int_tensor_simple(axis)

        output_shape = list(input.shape)
        output_shape[axis] *= tiles

        return [output_shape], [op.input.dtype]

    elif len(op.inputs) == 2:
        input, repeats = op.inputs
        return [infer.tile(input=input.shape, repeat=evaluate_shape_tensor_simple(repeats))], [input.dtype]
    else:
        assert False, 'Tile must have 2 or 3 inputs'


def propagate_transpose(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    return [infer.transpose(input=op.input.shape, axes=op.attribs.get('perm', None))], [op.input.dtype]


def propagate_unsqueeze(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    return [infer.unsqueeze(input=op.input.shape, axes=op.attribs['axes'])], [op.input.dtype]


def propagate_upsample(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    if 'scales' in op.attribs:
        scales = op.attribs['scales']
    else:
        scales = evaluate_float_list_tensor_simple(op.inputs[1])

    return [[utils.anyint_to_int(math.floor(i * s)) for i, s in zip(op.inputs[0].shape, scales)]], [op.inputs[0].dtype]


def propagate_crop(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    if 'scale' in op.attribs:
        return [op.input.shape[:2] + op.attribs['scale']], [op.input.dtype]
    else:
        n, c, h, w = op.input.shape
        l, t, r, b = op.attribs['border']
        return [[n, c, b - t, r - l]], [op.input.dtype]


def propagate_depth_to_space(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]

    n, c, h, w = op.input.shape
    block_size = op.attribs['blocksize']
    return [[n, c // (block_size ** 2), h * block_size, w * block_size]], [op.input.dtype]


def UNSUPPORTED(op):
    # type: (ONNXOperation)->typing.Tuple[typing.List[typing.List[int]], typing.List[str]]
    raise utils.NNEFToolsException('ONNX shape prop: Unsupported op: {}'.format(op.name))


_DefaultPropagators = {
    'Abs': propagate_first,
    'Acos': propagate_first,
    'Acosh': propagate_first,
    'Add': propagate_broadcast_with_axis,
    'And': propagate_broadcast_with_axis,
    'ArgMax': partial(propagate_reduce, dtype='INT64', multi_axis=False, default=[0]),
    'ArgMin': partial(propagate_reduce, dtype='INT64', multi_axis=False, default=[0]),
    'Asin': propagate_first,
    'Asinh': propagate_first,
    'Atan': propagate_first,
    'Atanh': propagate_first,
    'AveragePool': propagate_pool,
    'BatchNormalization': propagate_batch_normalization,
    'Cast': propagate_cast,
    'Ceil': propagate_first,
    'Clip': propagate_first,
    'Compress': UNSUPPORTED,
    'Concat': propagate_concat,
    # Constant: no operation is made for this, so no need to propagate
    'ConstantOfShape': propagate_constant_of_shape,
    'Conv': propagate_conv,
    'ConvTranspose': propagate_conv_transpose,
    'Cos': propagate_first,
    'Cosh': propagate_first,
    'DepthToSpace': propagate_depth_to_space,
    'Div': propagate_broadcast_with_axis,
    'Dropout': propagate_dropout,
    'Elu': propagate_first,
    'Equal': partial(propagate_broadcast_with_axis, dtype='BOOL'),
    'Erf': propagate_first,
    'Exp': propagate_first,
    'Expand': propagate_expand,
    'EyeLike': UNSUPPORTED,
    'Flatten': propagate_flatten,
    'Floor': propagate_first,
    'GRU': UNSUPPORTED,
    'Gather': propagate_gather,
    'Gemm': propagate_gemm,
    'GlobalAveragePool': propagate_global_pool,
    'GlobalLpPool': propagate_global_pool,
    'GlobalMaxPool': propagate_global_pool,
    'Greater': partial(propagate_broadcast_with_axis, dtype='BOOL'),
    'HardSigmoid': propagate_first,
    'HardMax': propagate_first,
    'Identity': propagate_first,
    'If': UNSUPPORTED,
    'InstanceNormalization': propagate_first,
    'IsNan': partial(propagate_first, dtype='BOOL'),
    'LRN': propagate_first,
    'LSTM': UNSUPPORTED,
    'LeakyRelu': propagate_first,
    'Less': partial(propagate_broadcast_with_axis, dtype='BOOL'),
    'Log': propagate_first,
    'LogSoftmax': propagate_first,
    'Loop': UNSUPPORTED,
    'LpNormalization': propagate_first,
    'LpPool': propagate_pool,
    'MatMul': propagate_gemm,
    'Max': propagate_broadcast,
    'MaxPool': propagate_pool,
    'MaxRoiPool': propagate_max_roi_pool,
    'MaxUnpool': propagate_max_unpool,
    'Mean': propagate_broadcast,
    'Min': propagate_broadcast,
    'Mul': propagate_broadcast_with_axis,
    'Multinomial': UNSUPPORTED,
    'Neg': propagate_first,
    'Not': propagate_first,
    'OneHot': UNSUPPORTED,
    'Or': propagate_broadcast_with_axis,
    'PRelu': propagate_first,
    'Pad': propagate_pad,
    'Pow': propagate_broadcast_with_axis,
    'RNN': UNSUPPORTED,
    'RandomNormal': UNSUPPORTED,
    'RandomNormalLike': UNSUPPORTED,
    'RandomUniform': UNSUPPORTED,
    'RandomUniformLike': UNSUPPORTED,
    'Reciprocal': propagate_first,
    'ReduceL1': partial(propagate_reduce, multi_axis=True, default=None),
    'ReduceL2': partial(propagate_reduce, multi_axis=True, default=None),
    'ReduceLogSum': partial(propagate_reduce, multi_axis=True, default=None),
    'ReduceLogSumExp': partial(propagate_reduce, multi_axis=True, default=None),
    'ReduceMax': partial(propagate_reduce, multi_axis=True, default=None),
    'ReduceMean': partial(propagate_reduce, multi_axis=True, default=None),
    'ReduceMin': partial(propagate_reduce, multi_axis=True, default=None),
    'ReduceProd': partial(propagate_reduce, multi_axis=True, default=None),
    'ReduceSum': partial(propagate_reduce, multi_axis=True, default=None),
    'ReduceSumSquare': partial(propagate_reduce, multi_axis=True, default=None),
    "Relu": propagate_first,
    'Reshape': propagate_reshape,
    'Scan': UNSUPPORTED,
    'Scatter': UNSUPPORTED,  # consider support
    'Selu': propagate_first,
    'Shape': propagate_shape,
    'Shrink': propagate_first,
    'Sigmoid': propagate_first,
    'Sign': propagate_first,
    'Sin': propagate_first,
    'Sinh': propagate_first,
    'Size': propagate_size,
    'Slice': propagate_slice,
    'Softmax': propagate_first,
    'Softplus': propagate_first,
    'Softsign': propagate_first,
    'SpaceToDepth': UNSUPPORTED,
    'Split': propagate_split,
    'Sqrt': propagate_first,
    'Squeeze': propagate_squeeze,
    'Sub': propagate_broadcast_with_axis,
    'Sum': propagate_broadcast,
    'Tan': propagate_first,
    'Tanh': propagate_first,
    'Tile': propagate_tile,
    'TopK': UNSUPPORTED,
    'Transpose': propagate_transpose,
    'Unsqueeze': propagate_unsqueeze,
    'Upsample': propagate_upsample,
    'Where': partial(propagate_broadcast, dtype_from_index=1),
    'Xor': propagate_broadcast_with_axis,

    # Experimental ONNX ops
    'ATen': UNSUPPORTED,
    'Affine': propagate_first,
    'ConstantFill': UNSUPPORTED,
    'Crop': propagate_crop,
    'DynamicSlice': propagate_dynamic_slice,
    'GRUUnit': UNSUPPORTED,
    'GivenTensorFill': UNSUPPORTED,
    'ImageScaler': propagate_first,
    'ParametricSoftplus': propagate_first,
    'Scale': propagate_first,
    'ScaledTanh': propagate_first,
    'ThresholdedRelu': propagate_first,
}
