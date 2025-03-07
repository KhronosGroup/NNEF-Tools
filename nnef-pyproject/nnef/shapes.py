# Copyright (c) 2017 The Khronos Group Inc.
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

import nnef
import numpy as np


def _ceil_div(x, y):
    return (x + y - 1) // y if y > 0 else (x + y + 1) // y


def _clamp(x, a, b):
    return max(a, min(b, x))


def _ensure_rank(array, rank, value=1):
    return array if len(array) == rank else array + [value] * rank


def _volume(shape):
    volume = 1
    for s in shape:
        volume *= s
    return volume


def _broadcast_compatible(x,y):
    return all(xi == yi or xi == 1 or yi == 1 for xi, yi in zip(x, y))


def _broadcastable(x,y):
    return all(xi == yi or xi == 1 for xi, yi in zip(x, y))


def _broadcast_shape(x, y):
    assert _broadcast_compatible(x, y), "arguments are not broadcast compatible ({} vs {})".format(x, y)

    rank = max(len(x), len(y))
    return [max(xi,yi) for (xi, yi) in zip(_ensure_rank(x, rank), _ensure_rank(y, rank))]


def _downsize_shape(input, kernel, padding, stride, dilation):
    return [(i + p + q - (k - 1) * d - 1) // s + 1 for i, k, (p, q), s, d in
            zip(input, kernel, padding, stride, dilation)] \
        if padding else [(i + s - 1) // s for i, s in zip(input, stride)]


def _upsize_shape(input, kernel, padding, stride, dilation):
    return [(i - 1) * s + (k - 1) * d + 1 - p - q for i, k, (p, q), s, d in
            zip(input, kernel, padding, stride, dilation)] \
        if padding else [i * s for i, s in zip(input, stride)]


def nullary_shape(shape, **kwargs):
    return shape


def unary_shape(arg, **kwargs):
    return arg


def binary_shape(left, right, **kwargs):
    return _broadcast_shape(left, right)


def asymmetric_binary_shape(left, right, **kwargs):
    assert _broadcastable(right, left), \
        "second argument shape ({}) cannot be broadcast to first argument shape ({})".format(right, left)
    return left


def ternary_shape(cond, left, right, **kwargs):
    value = _broadcast_shape(left, right)
    return _broadcast_shape(cond, value)


def pool_shape(input, size, border=None, padding=[], stride=[], dilation=[], output_shape=None, transposed=False, **kwargs):
    rank = len(input)

    assert len(size) == rank, "expected kernel shape of rank {}, found {}".format(rank, size)
    assert not padding or len(padding) == rank, "expected 'padding' of length {}, found {}".format(rank, padding)
    assert not stride or len(stride) == rank, "expected 'stride' of length {}, found {}".format(rank, stride)
    assert not dilation or len(dilation) == rank, "expected 'dilation' of length {}, found {}".format(rank, dilation)
    assert all(s > 0 for s in stride), "'stride' must be positive, found {}".format(stride)
    assert all(d > 0 for d in dilation), "'dilation' must be positive, found {}".format(dilation)

    stride = _ensure_rank(stride, rank)
    dilation = _ensure_rank(dilation, rank)

    if output_shape:
        assert len(output_shape) == rank, "expected 'output_shape' of length {}, found {}".format(rank, output_shape)
        assert all(s > 0 for s in output_shape), "'output_shape' must be positive, found {}".format(output_shape)

        expected_shape = _downsize_shape(output_shape, size, padding, stride, dilation)
        assert input == expected_shape, \
            "expected input shape {} derived from 'output_shape' is incompatible with actual input shape {}".\
                format(expected_shape, input)

        return output_shape

    if transposed:
        return _upsize_shape(input, size, padding, stride, dilation)
    else:
        return _downsize_shape(input, size, padding, stride, dilation)


def pool_with_index_shape(input, size, border=None, padding=[], stride=[], dilation=[]):
    shape = pool_shape(input, size, border, padding, stride, dilation)
    return (shape, shape)


def unpool_shape(input, size, border=None, padding=[], stride=[], dilation=[], output_shape=None, **kwargs):
    return pool_shape(input, size, border, padding, stride, dilation, output_shape, transposed=True, **kwargs)


def sample_shape(input, index, size, border=None, padding=[], stride=[], dilation=[], output_shape=None, transposed=False):
    assert index == input, "'index' shape {} does not match 'input' shape {}".format(input, index)
    return pool_shape(input, size, border, padding, stride, dilation, output_shape, transposed)


def desample_shape(input, index, size, border=None, padding=[], stride=[], dilation=[], output_shape=None):
    return sample_shape(input, index, size, border, padding, stride, dilation, output_shape, transposed=True)


def conv_shape(input, filter, bias=[], border=None, padding=[], stride=[], dilation=[], groups=1, output_shape=None, transposed=False):
    rank = len(input)

    assert len(filter) == rank, "expected filter shape of rank {}, found {}".format(rank, filter)
    assert not padding or len(padding) == rank - 2, "expected 'padding' of length {}, found {}".format(rank - 2, padding)
    assert not stride or len(stride) == rank - 2, "expected 'stride' of length {}, found {}".format(rank - 2, stride)
    assert not dilation or len(dilation) == rank - 2, "expected 'dilation' of rank {}, found {}".format(rank - 2, dilation)
    assert all(s > 0 for s in stride), "'stride' must be positive, found {}".format(stride)
    assert all(d > 0 for d in dilation), "'dilation' must be positive, found {}".format(dilation)
    assert groups >= 0, "'groups' must be non-negative, found {}".format(groups)

    if groups == 0:
        groups = output_shape[1] if transposed and output_shape else input[1]

    if transposed:
        assert filter[0] == input[1], "filter batch ({}) does not match input channels ({})".format(filter[0], input[1])
    else:
        assert filter[1] * groups == input[1], \
            "filter channels ({}) times groups ({}) does not match input channels ({})".format(filter[1], groups, input[1])
    assert filter[0] % groups == 0, "'groups' ({}) does not divide filter batch ({})".format(groups, filter[0])

    assert len(bias) <= 2, "expected bias shape of rank at most 2, found {}".format(bias)
    if len(bias) == 2:
        assert bias[0] == 1, "'bias' batch dimension must be singular"
    if len(bias):
        channels = filter[1] * groups if transposed else filter[0]
        assert bias[-1] == channels, "'bias' channels ({}) does not match filter batch ({})".format(bias[-1], channels)

    stride = _ensure_rank(stride, rank - 2)
    dilation = _ensure_rank(dilation, rank - 2)

    if output_shape:
        assert len(output_shape) == rank, "expected 'output_shape' of length {}, found {}".format(rank, output_shape)
        assert all(s > 0 for s in output_shape), "'output_shape' must be positive, found {}".format(output_shape)
        assert output_shape[0] == input[0], \
            "output batch ({}) does not match input batch ({})".format(output_shape[0], input[0])
        assert output_shape[1] == filter[1] * groups, \
            "output channels ({}) does not match input channels ({}) times groups ({})".format(output_shape[1], input[1], groups)

        expected_shape = [input[0], filter[0]] + _downsize_shape(output_shape[2:], filter[2:], padding, stride, dilation)
        assert input == expected_shape, \
            "expected input shape {} derived from 'output_shape' is incompatible with actual input shape {}". \
                format(expected_shape, input)

        return output_shape

    if transposed:
        return [input[0], filter[1] * groups] + _upsize_shape(input[2:], filter[2:], padding, stride, dilation)
    else:
        return [input[0], filter[0]] + _downsize_shape(input[2:], filter[2:], padding, stride, dilation)


def separable_conv_shape(input, plane_filter, point_filter, bias=[], border=None, padding=[], stride=[], dilation=[], groups=1,
                         output_shape=None, transposed=False):
    assert all(x == 1 for x in point_filter[2:]), \
        "point-wise filter must be singular in spatial dimensions, found {}".format(point_filter)
    assert point_filter[1] == plane_filter[0], \
        "channel dimension of point-wise filter ({}) does not equal batch dimension of depth-wise filter ({})".\
            format(point_filter[1], plane_filter[0])
    assert plane_filter[1] == 1, "channel dimension of plane-wise filter must be singular, found {}".format(plane_filter)

    channels = point_filter[1] if transposed else input[1]
    filter = [point_filter[0], channels] + plane_filter[2:]
    return conv_shape(input, filter, bias, border, padding, stride, dilation, groups, output_shape, transposed)


def separable_deconv_shape(input, plane_filter, point_filter, bias=[], border=None, padding=[], stride=[], dilation=[],
                           groups=1, output_shape=None):
    return separable_conv_shape(input, plane_filter, point_filter, bias, border, padding, stride, dilation, groups,
                                output_shape, transposed=True)


def deconv_shape(input, filter, bias=[], border=None, padding=[], stride=[], dilation=[], groups=1, output_shape=None):
    return conv_shape(input, filter, bias, border, padding, stride, dilation, groups, output_shape, transposed=True)


def reduce_shape(input, axes, **kwargs):
    rank = len(input)
    assert all(0 <= axis < rank for axis in axes), "axes must be in range [0,{}), found {}".format(rank, axes)
    return [1 if i in axes else input[i] for i in range(rank)]


def normalize_shape(input, **kwargs):
    rank = len(input)
    axes = kwargs.get('axes')
    size = kwargs.get('size')
    if axes:
        assert all(0 <= axis < rank for axis in axes), "axes must be in range [0,{}), found {}".format(rank, axes)
    if size:
        assert len(size) == rank, "expected 'size' of length {}, found {}".format(rank, size)
        assert all(s >= 1 for s in size), "'size' must be positive, found {}".format(size)

    return input


def moments_shape(input, axes):
    shape = reduce_shape(input, axes=axes)
    return shape, list(shape)


def downsample_shape(input, factor, **kwargs):
    rank = len(input)
    assert len(factor) == rank - 2, "expected 'factor' of length {}, found {}".format(rank, factor)
    assert all(i % f == 0 for i, f in zip(input[2:], factor)), \
        "'factor' {} does not divide spatial input shape {}".format(factor, input[2:])

    return input[:2] + [i // f for i, f in zip(input[2:], factor)]


def upsample_shape(input, factor, **kwargs):
    rank = len(input)
    assert len(factor) == rank - 2, "expected 'factor' of length {}, found {}".format(rank, factor)

    return input[:2] + [i * f for i, f in zip(input[2:], factor)]


def reshape_shape(input, shape, axis_start=0, axis_count=-1):
    rank = len(input)
    assert all(s >= -1 for s in shape), "items in 'shape' must be >= -1, found {}".format(shape)
    assert sum(1 for s in shape if s == -1) <= 1, "at most one item may be -1 in 'shape', found {}".format(shape)
    assert 0 <= axis_start <= rank, "'axis_start' must be in range [0,{}], found {}".format(rank, axis_start)
    assert axis_count >= -1, "'axis_count' must be non-negative or -1, found {}".format(axis_count)

    if axis_count == -1:
        axis_count = rank - axis_start

    axis_end = axis_start + axis_count

    assert axis_end <= rank, "'axis_start' + 'axis_count' ({}) must be in range [0,{}]".format(axis_end, rank)

    shape = list(shape)  # don't modify original list

    for i in range(len(shape)):
        if shape[i] == 0:
            shape[i] = input[i + axis_start]

    input_range = input[axis_start:axis_end]

    if -1 in shape:
        idx = shape.index(-1)
        assert _volume(input_range) % _volume(shape) == 0, \
            "volume of 'shape' ({}) does not divide volume of 'input[{}:{}]' ({})".format(shape, axis_start, axis_end, input_range)
        shape[idx] = _volume(input_range) // -_volume(shape)
    else:
        assert _volume(shape) == _volume(input_range), \
            "volume of 'shape' ({}) does not equal volume of 'input[{}:{}]' ({})".format(shape, axis_start, axis_end, input_range)
    return input[:axis_start] + shape + input[axis_end:]


def transpose_shape(input, axes):
    rank = len(axes)
    assert sorted(axes) == list(range(rank)), "axes must be a permutation of [0..{}], found {}".format(rank-1, axes)
    return [input[axis] for axis in axes] + input[rank:]


def squeeze_shape(input, axes):
    rank = len(input)
    assert all(0 <= axis < rank for axis in axes), "axes must be in range [0,{}), found {}".format(rank, axes)
    return [input[i] for i in range(rank) if not i in axes]


def unsqueeze_shape(input, axes):
    rank = len(input) + len(axes)
    assert all(0 <= axis < rank for axis in axes), "axes must be in range [0,{}), found {}".format(rank, axes)

    output = list(input)
    for axis in axes:
        output = output[:axis] + [1] + output[axis:]
    return output


def concat_shape(values, axis):
    assert len(values) != 0, "'values' must be non-empty"

    shape = list(values[0])
    rank = len(shape)

    assert 0 <= axis < rank, "'axis' must be in range [0,{}), found {}".format(rank, axis)

    for value in values:
        assert len(value) == len(shape), "'values' must have the same rank, found {}".format(values)
        assert all(value[i] == shape[i] for i in range(rank) if i != axis), \
            "shapes of 'values' must be identical for all dimensions other than 'axis' ({}), found {}".format(axis, values)

    shape[axis] = sum(value[axis] for value in values)
    return shape


def split_shape(value, axis, ratios):
    rank = len(value)
    assert 0 <= axis < rank, "axis must be in range [0,{}), found {}".format(rank, axis)
    assert all(r > 0 for r in ratios), "'ratios' must be positive, found {}".format(ratios)
    total = sum(ratios)
    assert value[axis] % total == 0, \
        "sum of 'ratios' ({}) does not divide input shape along dimension 'axis' ({})".format(total, value[axis])

    unit = value[axis] // total
    return [[unit * r if i == axis else value[i] for i in range(rank)] for r in ratios]


def stack_shape(values, axis):
    assert len(values) != 0, "'values' must be non-empty"

    shape = values[0]
    rank = len(shape) + 1

    assert 0 <= axis < rank, "'axis' must be in range [0,{}), found {}".format(rank, axis)
    assert all(value == shape for value in values), "shapes of 'values' must be identical, found {}".format(values)

    return shape[:axis] + [len(values)] + shape[axis:]


def unstack_shape(value, axis):
    rank = len(value)
    assert 0 <= axis < rank, "'axis' must be in range [0,{}), found {}".format(rank, axis)

    return [value[:axis] + value[axis+1:]] * value[axis]


def slice_shape(input, axes, begin, end, stride=[]):
    rank = len(input)

    if len(stride) == 0:
        stride = [1] * len(axes)

    if all(s == 1 for s in stride):
        end = [input[axis] if offs == 0 else offs for axis, offs in zip(axes, end)]

    assert len(begin) == len(axes), \
        "length of 'begin' ({}) does not equal length of 'axes' ({})".format(len(begin), len(axes))
    assert len(end) == len(axes), \
        "length of 'end' ({}) does not equal length of 'axes' ({})".format(len(end), len(axes))
    assert len(stride) == len(axes), \
        "length of 'stride' ({}) does not equal length of 'axes' ({})".format(len(begin), len(axes))
    assert all(0 <= axis < rank for axis in axes), "'axes' must be in range [0,{}), found {}".format(rank, axes)

    begin = [_clamp(offs + input[axis] if offs < 0 else offs, -1, input[axis]) for axis, offs in zip(axes, begin)]
    end = [_clamp(offs + input[axis] if offs < 0 else offs, -1, input[axis]) for axis, offs in zip(axes, end)]

    assert all(s != 0 for s in stride), "'stride' must be non-zero"

    assert all(0 <= first <= last if str > 0 else last <= first < input[axis]
               for axis, first, last, str in zip(axes, begin, end, stride)), \
        "slice range ({}:{}:{}) is invalid".format(begin, end, stride)

    output = list(input)
    for axis, first, last, str in zip(axes, begin, end, stride):
        output[axis] = _ceil_div(last - first, str)
    return output


def tile_shape(input, repeats):
    rank = len(input)
    assert len(repeats) == rank, "expected 'repeats' of length {}, found {}".format(rank, repeats)

    return [i * r for i, r in zip(input, repeats)]


def pad_shape(input, padding, **kwargs):
    rank = len(input)
    assert len(padding) == rank, "expected 'padding' of length {}, found {}".format(rank, padding)

    return [p + i + q for i, (p, q) in zip(input, padding)]


def gather_shape(input, indices, axis=0):
    rank = len(input)
    assert 0 <= axis < rank, "'axis' must be in range [0,{}), found {}".format(rank, axis)

    return input[:axis] + indices + input[axis+1:]


def matmul_shape(A, B, transposeA=False, transposeB=False):
    assert len(A) == len(B), "argument rank mismatch ({} vs {})".format(len(A), len(B))
    assert len(A) >= 2, "rank of arguments must be at least 2, found {}".format(len(A))

    m = A[-1] if transposeA else A[-2]
    n = B[-2] if transposeB else B[-1]
    kA = A[-2] if transposeA else A[-1]
    kB = B[-1] if transposeB else B[-2]

    assert kA == kB, "inner dimensions must agree ({} vs {})".format(kA, kB)
    return _broadcast_shape(A[:-2], B[:-2]) + [m,n]


def linear_shape(input, filter, bias=[]):
    assert len(input) == 2, "rank of input must be 2, found {}".format(len(input))
    assert len(filter) == 2, "rank of filter must be 2, found {}".format(len(filter))
    assert len(bias) <= 2, "rank of bias must be at most 2, found {}".format(len(bias))
    assert input[1] == filter[1], "input channels ({}) does not match filter channels ({})".format(input[1], filter[1])

    if len(bias) == 2:
        assert bias[0] == 1, "'bias' batch dimension must be singular"
    if len(bias):
        c = len(bias) - 1
        assert bias[c] == filter[0], "'bias' channels ({}) does not match filter batch ({})".format(bias[c], filter[0])

    return [input[0], filter[0]]


def softmax_shape(input, axes=[1]):
    rank = len(input)
    assert all(0 <= axis < rank for axis in axes), "axes must be in range [0,{}), found {}".format(rank, axes)

    return input


def batchnorm_shape(input, mean, variance, offset, scale, epsilon=0):
    assert epsilon >= 0, "'epsilon' must be non-negative, found {}".format(epsilon)

    assert _broadcastable(mean, input), \
        "'mean' shape {} cannot be broadcast to 'input' shape {}".format(mean, input)
    assert _broadcastable(variance, input), \
        "'variance' shape {} cannot be broadcast to 'input' shape {}".format(variance, input)
    assert _broadcastable(offset, input), \
        "'offset' shape {} cannot be broadcast to 'input' shape {}".format(offset, input)
    assert _broadcastable(scale, input), \
        "'scale' shape {} cannot be broadcast to 'input' shape {}".format(scale, input)

    return input


def roi_shape(input, rois, batch_index, output_size, **kwargs):
    rank = len(input)

    assert len(output_size) == rank - 2, "expected 'output_size' of length {}, found {}".format(rank - 2, output_size)
    assert all(s > 0 for s in output_size), "'output_size' must be positive, found {}".format(output_size)

    assert len(rois) == 2, "'rois' must be of rank 2, found {}".format(rois)
    assert rois[1] == 4, "'rois' must be of extent 4 along dimension 1, found {}".format(rois)

    assert len(batch_index) == 1, "'batch_index' must be of rank 1, found {}".format(batch_index)
    assert batch_index[0] == rois[0], \
        "'batch_index' must be of same length as dimension 0 of rois; found {} vs {}".format(batch_index, rois)

    rate = kwargs.get('sampling_rate')
    if rate:
        assert len(rate) == rank - 2, "expected 'sampling_rate' of length {}, found {}".format(rank - 2, rate)
        assert all(r > 0 for r in rate), "'rate' must be positive, found {}".format(rate)

    return [rois[0], input[1]] + output_size


def quantize_shape(input, *args, **kwargs):
    for arg in args:
        assert _broadcastable(arg, input), \
            "'min/max' shape {} cannot be broadcast to 'input' shape {}".format(arg, input)

    bits = kwargs.get('bits')
    if bits is not None:
        assert bits > 0, "'bits' must be positive, found {}".format(bits)

    return input


def update_shape(variable, value):
    assert value == variable, "shape of update value {} does not match shape of variable {}".format(value, variable)
    return variable


def copy_n_shape(value, times):
    assert times > 0, "'times' must be positive, found {}".format(times)
    return [value] * times


def add_n_shape(values):
    assert len(values) != 0, "values must be non-empty"

    shape = values[0]
    assert all(value == shape for value in values), "shapes of values must be identical, found {}".format(values)

    return shape


def _get_shape(graph, value):
    if isinstance(value, nnef.Identifier):
        return graph.tensors[value].shape
    elif isinstance(value, np.ndarray):
        return list(value.shape)
    elif isinstance(value, list):
        return [_get_shape(graph, v) for v in value]
    else:
        return []


def _set_shape(graph, value, shape):
    if isinstance(value, nnef.Identifier):
        tensor = graph.tensors[value]
        graph.tensors[value] = nnef.Tensor(tensor.name, tensor.dtype, shape, tensor.data, tensor.quantization)
    elif isinstance(value, list):
        for v, s in zip(value, shape):
            _set_shape(graph, v, s)


def infer_shapes(graph, external_shapes={}, custom_shapes={}):
    # type: (nnef.Graph, dict)->None
    for op in graph.operations:
        func = _StandardShapeFuncs.get(op.name)
        if func is None:
            func = custom_shapes.get(op.name)
        if func is None:
            raise nnef.Error("shape inference function is not defined for operation '{}'".format(op.name))

        if op.name == 'external':
            id = op.outputs['output']
            override = external_shapes.get(id)
            if override is not None:
                override = list(override)
                original = op.attribs['shape']
                assert len(override) == len(original), \
                    "overridden external shape rank ({}) does not match original rank ({})".format(len(override), len(original))
                _set_shape(graph, id, override)
                continue

        input_shapes = [_get_shape(graph, input) for input in op.inputs.values()]

        try:
            output_shapes = func(*input_shapes, **op.attribs)
            if not isinstance(output_shapes, tuple):
                output_shapes = (output_shapes,)

            outputs = op.outputs.values()
            assert len(outputs) == len(output_shapes), \
                "number of shapes ({}) does not match number of outputs ({})".format(len(outputs), len(output_shapes))

            for output, shape in zip(outputs, output_shapes):
                if isinstance(output, list):
                    assert isinstance(shape, list), "expected list of shapes"
                    assert len(output) == len(shape), \
                        "number of shapes ({}) does not match number of outputs ({})".format(len(output), len(shape))
                _set_shape(graph, output, shape)

        except AssertionError as e:
            raise nnef.Error("while inferring shape of tensor(s) '{}' (operation '{}'): {}".
                             format(', '.join(op.outputs.values()), op.name, e))

    for tensor in graph.tensors.values():
        if tensor.quantization:
            for key, value in tensor.quantization.items():
                if isinstance(value, np.ndarray):
                    assert _broadcastable(value.shape, tensor.shape)


def _infer_op_shapes(op_name, attribs, input_shapes, output_counts, custom_shapes={}):
    func = _StandardShapeFuncs.get(op_name)
    if func is None:
        func = custom_shapes.get(op_name)
    if func is None:
        raise nnef.Error("shape inference function is not defined for operation '{}'".format(op_name))

    try:
        output_shapes = func(*input_shapes, **attribs)
        if not isinstance(output_shapes, tuple):
            output_shapes = (output_shapes,)

        assert len(output_counts) == len(output_shapes), \
            "number of shapes ({}) does not match number of outputs ({})".format(len(output_counts), len(output_shapes))

        for count, shape in zip(output_counts, output_shapes):
            if isinstance(count, list):
                assert isinstance(shape, list), "expected list of shapes"
                assert count == len(shape), \
                    "number of shapes ({}) does not match number of outputs ({})".format(count, len(shape))

        return output_shapes
    except AssertionError as e:
        raise nnef.Error("while inferring output shape of operation '{}': {}".format(op_name, e))


_StandardShapeFuncs = {
    'external': nullary_shape,
    'variable': nullary_shape,
    'constant': nullary_shape,
    'copy': unary_shape,
    'neg': unary_shape,
    'not': unary_shape,
    'rcp': unary_shape,
    'exp': unary_shape,
    'log': unary_shape,
    'sin': unary_shape,
    'cos': unary_shape,
    'tan': unary_shape,
    'asin': unary_shape,
    'acos': unary_shape,
    'atan': unary_shape,
    'sinh': unary_shape,
    'cosh': unary_shape,
    'tanh': unary_shape,
    'asinh': unary_shape,
    'acosh': unary_shape,
    'atanh': unary_shape,
    'abs': unary_shape,
    'sign': unary_shape,
    'floor': unary_shape,
    'ceil': unary_shape,
    'round': unary_shape,
    'sqr': unary_shape,
    'sqrt': unary_shape,
    'rsqr': unary_shape,
    'rsqrt': unary_shape,
    'log2': unary_shape,
    'sigmoid': unary_shape,
    'relu': unary_shape,
    'elu': unary_shape,
    'selu': unary_shape,
    'gelu': unary_shape,
    'silu': unary_shape,
    'softabs': unary_shape,
    'softplus': unary_shape,
    'leaky_relu': unary_shape,
    'prelu': asymmetric_binary_shape,
    'add': binary_shape,
    'sub': binary_shape,
    'mul': binary_shape,
    'div': binary_shape,
    'pow': binary_shape,
    'min': binary_shape,
    'max': binary_shape,
    'lt': binary_shape,
    'le': binary_shape,
    'gt': binary_shape,
    'ge': binary_shape,
    'eq': binary_shape,
    'ne': binary_shape,
    'and': binary_shape,
    'or': binary_shape,
    'select': ternary_shape,
    'clamp': ternary_shape,
    'conv': conv_shape,
    'deconv': deconv_shape,
    'separable_conv': separable_conv_shape,
    'separable_deconv': separable_deconv_shape,
    'box': pool_shape,
    'debox': unpool_shape,
    'sample': sample_shape,
    'desample': desample_shape,
    'avg_pool': pool_shape,
    'max_pool': pool_shape,
    'argmax_pool': pool_shape,
    'rms_pool': pool_shape,
    'max_pool_with_index': pool_with_index_shape,
    'max_unpool': unpool_shape,
    'avg_unpool': unpool_shape,
    'sum_reduce': reduce_shape,
    'min_reduce': reduce_shape,
    'max_reduce': reduce_shape,
    'mean_reduce': reduce_shape,
    'argmin_reduce': reduce_shape,
    'argmax_reduce': reduce_shape,
    'any_reduce': reduce_shape,
    'all_reduce': reduce_shape,
    'local_response_normalization': normalize_shape,
    'local_mean_normalization': normalize_shape,
    'local_variance_normalization': normalize_shape,
    'local_contrast_normalization': normalize_shape,
    'l1_normalization': normalize_shape,
    'l2_normalization': normalize_shape,
    'moments': moments_shape,
    'batch_normalization': batchnorm_shape,
    'nearest_downsample': downsample_shape,
    'area_downsample': downsample_shape,
    'nearest_upsample': upsample_shape,
    'multilinear_upsample': upsample_shape,
    'reshape': reshape_shape,
    'transpose': transpose_shape,
    'squeeze': squeeze_shape,
    'unsqueeze': unsqueeze_shape,
    'stack': stack_shape,
    'unstack': unstack_shape,
    'split': split_shape,
    'concat': concat_shape,
    'slice': slice_shape,
    'tile': tile_shape,
    'pad': pad_shape,
    'cast': unary_shape,
    'gather': gather_shape,
    'matmul': matmul_shape,
    'linear': linear_shape,
    'softmax': softmax_shape,
    'linear_quantize': quantize_shape,
    'logarithmic_quantize': quantize_shape,
    'min_max_linear_quantize': quantize_shape,
    'zero_point_linear_quantize': quantize_shape,
    'avg_roi_pool': roi_shape,
    'max_roi_pool': roi_shape,
    'avg_roi_align': roi_shape,
    'max_roi_align': roi_shape,
    'roi_resample': roi_shape,
    'update': update_shape,
    'copy_n': copy_n_shape,
    'add_n': add_n_shape,
}
