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

from __future__ import division, print_function, absolute_import

import typing
from collections import namedtuple


class Padding(object):
    VALID = 0
    SAME_UPPER = 1
    SAME_LOWER = 2

    @staticmethod
    def allowed(value):
        return value in [Padding.VALID, Padding.SAME_UPPER, Padding.SAME_LOWER]


class Broadcast(object):
    NONE = 0
    SAME_RANK = 1  # (1, 2, 3), (2, 2, 3) is broadcastable
    FROM_LEFT = 2  # (1, 2, 3), (1, 2) is broadcastable
    FROM_RIGHT = 3  # (1, 2, 3), (2, 3) is broadcastable

    @staticmethod
    def allowed(value):
        return value in [Broadcast.NONE, Broadcast.SAME_RANK, Broadcast.FROM_LEFT, Broadcast.FROM_RIGHT]


class Format(object):
    NCHW = 0
    NHWC = 1

    @staticmethod
    def allowed(value):
        return value in [Format.NCHW, Format.NHWC]


DecomposedStridedSlice = namedtuple('DecomposedStridedSlice',
                                    ['ssl_begin', 'ssl_end', 'ssl_stride', 'ssl_shape', 'reshape_shape'])

RankType = int
AxisType = int
DimType = int
ShapeType = typing.List[DimType]
ConcretePaddingType = typing.List[typing.Tuple[DimType, DimType]]
PaddingType = typing.Union[ConcretePaddingType, Padding]
AxisListType = typing.List[AxisType]
ShapeListType = typing.List[ShapeType]
ZeroOneListType = typing.List[int]


def spatial_begin(format):
    # type: (Format)->int

    if format == Format.NCHW:
        return 2
    elif format == Format.NHWC:
        return 1
    else:
        assert False


_spatial_begin = spatial_begin


def channel_axis(format):
    # type: (Format)->int

    if format == Format.NCHW:
        return 1
    elif format == Format.NHWC:
        return -1
    else:
        assert False


_channel_axis = channel_axis


def spatial(shape, format):
    # type: (ShapeType, Format)->ShapeType
    if format == Format.NCHW:
        return shape[2:]
    elif format == Format.NHWC:
        return shape[1:-1]
    else:
        assert False


def singleton(rank):
    # type: (RankType) -> ShapeType

    return [1] * rank


def copy(x):
    # type: (ShapeType)->ShapeType
    return list(x)


def elementwise(inputs, broadcast):
    # type: (ShapeListType, Broadcast)->ShapeType

    assert len(inputs) > 0

    if broadcast == Broadcast.NONE:
        assert all(inputs[i] == inputs[0] for i in range(1, len(inputs)))
        return list(inputs[0])

    max_rank = max(len(input) for input in inputs)

    if broadcast == Broadcast.FROM_LEFT:
        inputs = list(inputs)
        for i in range(len(inputs)):
            if len(inputs[i]) < max_rank:
                inputs[i] = inputs[i] + [1] * (max_rank - len(inputs[i]))
    elif broadcast == Broadcast.FROM_RIGHT:
        inputs = list(inputs)
        for i in range(len(inputs)):
            if len(inputs[i]) < max_rank:
                inputs[i] = [1] * (max_rank - len(inputs[i])) + inputs[i]

    assert all(len(inputs[i]) == len(inputs[0]) for i in range(1, len(inputs)))
    assert all(inputs[i][j] == inputs[0][j] or inputs[i][j] == 1 or inputs[0][j] == 1
               for i in range(1, len(inputs))
               for j in range(len(inputs[0])))
    return [max(inputs[i][j] for i in range(len(inputs))) for j in range(len(inputs[0]))]


def sliding_window(input,  # type: ShapeType
                   filter,  # type: ShapeType
                   padding,  # type: PaddingType
                   stride,  # type: ShapeType
                   dilation,  # type: ShapeType
                   upscale=False,  # type: bool
                   ceil=False,  # type: bool
                   output_padding=None  # type: typing.Optional[ConcretePaddingType]
                   ):
    # type: (...)->ShapeType
    assert len(input) == len(filter) == len(stride) == len(dilation)
    assert output_padding is None or len(output_padding) == len(filter)
    assert Padding.allowed(padding) or len(padding) == len(filter)

    if output_padding is None:
        output_padding = len(filter) * [(0, 0)]

    if Padding.allowed(padding):
        if padding == Padding.VALID:
            padding = valid_padding(len(input))
        elif padding in [Padding.SAME_LOWER, Padding.SAME_UPPER]:
            if upscale:
                upscaled_input = [i * s for i, s in zip(input, stride)]
                padding = same_padding(upscaled_input=upscaled_input,
                                       filter=filter,
                                       stride=stride,
                                       dilation=dilation,
                                       left_bigger=(padding == Padding.SAME_LOWER))
            else:
                padding = same_padding(upscaled_input=input,
                                       filter=filter,
                                       stride=stride,
                                       dilation=dilation,
                                       left_bigger=(padding == Padding.SAME_LOWER))

    dilated_filter = [(f - 1) * d + 1 for f, d in zip(filter, dilation)]

    if upscale:
        return [(i - 1) * s + df - (p + q) + op + oq
                for (i, s, df, (p, q), (op, oq))
                in zip(input, stride, dilated_filter, padding, output_padding)]
    else:
        return [((p + i + q - df + ((s - 1) if ceil else 0)) // s) + 1 + op + oq
                for (i, s, df, (p, q), (op, oq))
                in zip(input, stride, dilated_filter, padding, output_padding)]


def valid_padding(rank):
    # type: (RankType)->ConcretePaddingType

    return rank * [(0, 0)]


def same_padding(upscaled_input, filter, stride, dilation, left_bigger=False):
    # type:(ShapeType, ShapeType, ShapeType, ShapeType, bool)->ConcretePaddingType

    assert len(upscaled_input) == len(filter) == len(stride) == len(dilation)

    downscaled_input = [(ui + (s - 1)) // s for ui, s in zip(upscaled_input, stride)]
    dilated_filter = [(f - 1) * d + 1 for f, d in zip(filter, dilation)]

    total_padding = [(di - 1) * s + df - ui
                     for di, s, df, ui
                     in zip(downscaled_input, stride, dilated_filter, upscaled_input)]

    return [((tp + 1) // 2, tp // 2) if left_bigger else (tp // 2, (tp + 1) // 2) for tp in total_padding]


def concat(inputs, axis):
    # type: (ShapeListType, AxisType)->ShapeType
    assert len(inputs) > 0
    assert len(set(len(input) for input in inputs)) == 1
    assert -len(inputs[0]) <= axis < len(inputs[0])

    if axis < 0:
        axis += len(inputs[0])

    for i in range(len(inputs[0])):
        for input in inputs[1:]:
            assert i == axis or input[i] == inputs[0][i]

    return [sum(input[i] for input in inputs) if i == axis else inputs[0][i] for i in range(len(inputs[0]))]


def split(input,  # type: ShapeType
          axis,  # type: AxisType
          num=None,  # type: typing.Optional[int]
          sizes=None,  # type: typing.Optional[typing.List[DimType]]
          ratios=None  # type: typing.Optional[typing.List[int]]
          ):
    # type: (...)->ShapeListType

    assert sum(arg is not None for arg in (num, sizes, ratios)) == 1

    assert -len(input) <= axis < len(input)
    if axis < 0:
        axis += len(input)

    if num is not None:
        assert input[axis] % num == 0
        return [[dim // num if i == axis else dim for i, dim in enumerate(input)] for _ in range(num)]
    elif sizes is not None:
        assert input[axis] == sum(sizes)
        return [[size if i == axis else dim for i, dim in enumerate(input)] for size in sizes]
    elif ratios is not None:
        assert input[axis] % sum(ratios) == 0
        unit_size = input[axis] // sum(ratios)
        return [[unit_size * ratio if i == axis else dim for i, dim in enumerate(input)] for ratio in ratios]
    else:
        assert False


def conv(input,  # type: ShapeType
         filter,  # type: ShapeType
         padding,  # type: PaddingType
         stride,  # type: ShapeType
         dilation,  # type: ShapeType
         groups,  # type: int # just for check
         output_channels,  # type: DimType
         format=None,  # type: typing.Optional[Format]
         spatial_begin=None,  # type: typing.Optional[AxisType]
         channel_axis=None,  # type: typing.Optional[AxisType]
         deconv=False,  # type: bool
         ceil=False,  # type: bool
         output_padding=None,  # type: typing.Optional[ConcretePaddingType] # (left, right) pairs
         ):
    # type: (...)->ShapeType

    assert ((format is None and spatial_begin is not None and channel_axis is not None)
            or (format is not None and spatial_begin is None and channel_axis is None))

    if format is not None:
        spatial_begin = _spatial_begin(format)
        channel_axis = _channel_axis(format)

    assert -len(input) <= spatial_begin < len(input)
    assert -len(input) <= channel_axis < len(input)

    if spatial_begin < 0:
        spatial_begin += len(input)
    if channel_axis < 0:
        channel_axis += len(input)
    if groups == 0:
        groups = input[channel_axis]

    assert input[channel_axis] % groups == 0
    assert output_channels % groups == 0

    assert len(filter) == len(stride) == len(dilation)
    assert Padding.allowed(padding) or len(padding) == len(filter)
    assert output_padding is None or len(output_padding) == len(filter)
    assert spatial_begin + len(filter) <= len(input)
    assert channel_axis < spatial_begin or channel_axis >= spatial_begin + len(filter)

    out_spatial = sliding_window(input=input[spatial_begin:spatial_begin + len(filter)],
                                 filter=filter,
                                 padding=padding,
                                 stride=stride,
                                 dilation=dilation,
                                 upscale=deconv,
                                 ceil=ceil,
                                 output_padding=output_padding)
    output = input[:spatial_begin] + out_spatial + input[spatial_begin + len(filter):]
    output[channel_axis] = output_channels

    return output


def squeeze(input, axes=None):
    # type: (ShapeType, typing.Optional[AxisListType])->ShapeType
    if axes is None:
        axes = [i for i, dim in enumerate(input) if dim == 1]
    assert all(-len(input) <= axis < len(input) for axis in axes)
    axes = [axis if axis >= 0 else axis + len(input) for axis in axes]
    assert all(dim == 1 for i, dim in enumerate(input) if i in axes)
    return [dim for i, dim in enumerate(input) if i not in axes]


_squeeze = squeeze


def unsqueeze(input, axes):
    # type: (ShapeType, AxisListType)->ShapeType
    output_rank = len(input) + len(axes)
    assert all(-output_rank <= axis < output_rank for axis in axes)
    axes = sorted([axis if axis >= 0 else axis + output_rank for axis in axes])
    output = list(input)
    for axis in axes:
        output.insert(axis, 1)
    return output


def matmul(a, b, transpose_a=False, transpose_b=False):
    # type: (ShapeType, ShapeType, bool, bool)->ShapeType
    assert len(a) >= 2 and len(b) >= 2
    assert len(a) == len(b)
    assert a[:-2] == b[:-2]

    if transpose_a:
        a = a[:-2] + [a[-1], a[-2]]
    if transpose_b:
        b = b[:-2] + [b[-1], b[-2]]

    assert a[-1] == b[-2]

    return a[:-2] + [a[-2], b[-1]]


def reduce(input, axes, squeeze=False):
    # type: (ShapeType, AxisListType, bool)->ShapeType

    assert all(-len(input) <= axis < len(input) for axis in axes)
    axes = [axis if axis >= 0 else axis + len(input) for axis in axes]
    output = [1 if i in axes else dim for i, dim in enumerate(input)]
    return _squeeze(output, axes) if squeeze else output


def stack(inputs, axis):
    # type: (ShapeListType, AxisType)->ShapeType

    assert all(inputs[i] == inputs[0] for i in range(1, len(inputs)))
    output_rank = len(inputs[0]) + 1
    assert -output_rank <= axis < output_rank
    if axis < 0:
        axis += output_rank
    output = list(inputs[0])
    output.insert(axis, len(inputs))
    return output


def unstack(input, axis):
    # type: (ShapeType, AxisType)->ShapeListType

    assert -len(input) <= axis < len(input)
    if axis < 0:
        axis += len(input)
    return [[dim for i, dim in enumerate(input) if i != axis] for _ in range(input[axis])]


def pad(input, padding):
    # type: (ShapeType, ConcretePaddingType)->ShapeType
    assert len(input) == len(padding)
    return [p + dim + q for dim, (p, q) in zip(input, padding)]


def volume(shape):
    p = 1
    for dim in shape:
        p *= dim
    return p


def reshape(input, shape, zero_means_same=False):
    # type: (ShapeType, ShapeType, bool)->ShapeType
    if zero_means_same:
        assert all(i < len(input) for i, dim in enumerate(shape) if dim == 0)
    shape = [input[i] if shape[i] == 0 and zero_means_same else shape[i]
             for i in range(len(shape))]
    assert sum(dim == -1 for dim in shape) <= 1
    if -1 in shape:
        idx = shape.index(-1)
        shape[idx] = 1
        volume_shape = volume(shape)
        assert volume_shape != 0
        assert volume(input) % volume_shape == 0
        shape[idx] = volume(input) // volume_shape
    assert volume(input) == volume(shape)
    return shape


def flatten(input):
    # type: (ShapeType)->ShapeType
    if len(input) == 0:
        return [1, 1]
    else:
        return [input[0], volume(input[1:])]


def resize(input, size, format=None, spatial_begin=None):
    # type: (ShapeType, ShapeType, typing.Optional[Format], typing.Optional[AxisType])->ShapeType

    assert (format is None and spatial_begin is not None) or (format is not None and spatial_begin is None)

    if format is not None:
        spatial_begin = _spatial_begin(format)

    assert -len(input) <= spatial_begin < len(input)
    if spatial_begin < 0:
        spatial_begin += len(input)
    assert spatial_begin + len(size) <= len(input)

    return input[:spatial_begin] + size + input[spatial_begin + len(size):]


def upsample(input, factor, format=None, spatial_begin=None):
    # type: (ShapeType, typing.List[int], typing.Optional[Format], typing.Optional[AxisType])->ShapeType

    assert (format is None and spatial_begin is not None) or (format is not None and spatial_begin is None)

    if format is not None:
        spatial_begin = _spatial_begin(format)

    assert -len(input) <= spatial_begin < len(input)
    if spatial_begin < 0:
        spatial_begin += len(input)
    assert spatial_begin + len(factor) <= len(input)

    old_size = input[spatial_begin:spatial_begin + len(factor)]
    new_size = [dim * f for dim, f in zip(old_size, factor)]

    return input[:spatial_begin] + new_size + input[spatial_begin + len(factor):]


def downsample(input, factor, format=None, spatial_begin=None):
    # type: (ShapeType, typing.List[int], typing.Optional[Format], typing.Optional[AxisType])->ShapeType

    assert (format is None and spatial_begin is not None) or (format is not None and spatial_begin is None)

    if format is not None:
        spatial_begin = _spatial_begin(format)

    assert -len(input) <= spatial_begin < len(input)
    if spatial_begin < 0:
        spatial_begin += len(input)
    assert spatial_begin + len(factor) <= len(input)

    old_size = input[spatial_begin:spatial_begin + len(factor)]
    assert all(dim % f == 0 for dim, f in zip(old_size, factor))
    new_size = [dim // f for dim, f in zip(old_size, factor)]

    return input[:spatial_begin] + new_size + input[spatial_begin + len(factor):]


def _apply_permutation(list, perm):
    return [list[ind] for ind in perm]


def transpose(input, axes=None):
    # type: (ShapeType, typing.Optional[AxisListType])->ShapeType
    assert axes is None or len(axes) == len(input)
    if axes is None:
        return input[::-1]
    else:
        assert all(-len(input) <= axis < len(input) for axis in axes)
        axes = [axis if axis >= 0 else axis + len(input) for axis in axes]
        return _apply_permutation(input, axes)


def slice(input,  # type: ShapeType
          begin,  # type: typing.List[int]
          end=None,  # type: typing.Optional[typing.List[int]]
          size=None,  # type: typing.Optional[typing.List[int]]
          axes=None,  # type: typing.Optional[AxisListType]
          stride=None,  # type: typing.Optional[typing.List[int]]
          zero_means_all=False  # type: bool
          ):
    # type: (...)->ShapeType
    assert sum(arg is not None for arg in (end, size)) == 1

    if stride is None:
        stride = [1] * len(begin)

    if axes is not None:
        assert all(-len(input) <= axis < len(input) for axis in axes)
        axes = [axis + len(input) if axis < 0 else axis for axis in axes]
        begin = [begin[axes.index(i)] if i in axes else 0 for i, dim in enumerate(input)]
        stride = [stride[axes.index(i)] if i in axes else 1 for i, dim in enumerate(input)]

        if size is not None:
            size = [size[axes.index(i)] if i in axes else dim for i, dim in enumerate(input)]
        elif end is not None:
            end = [end[axes.index(i)] if i in axes else dim for i, dim in enumerate(input)]
        else:
            assert False

    begin = [b + dim if b < 0 else b for b, dim in zip(begin, input)]
    if size is not None:
        res = [s if s != -1 and not (s == 0 and zero_means_all) else dim - b for b, s, dim in zip(begin, size, input)]
        return [r // abs(st) for r, st in zip(res, stride)]
    elif end is not None:
        if zero_means_all:
            end = [dim if e == 0 else e for e, dim in zip(end, input)]
        end = [e + dim if e < 0 else e for e, dim in zip(end, input)]
        res = [min(e, dim) - b for dim, b, e in zip(input, begin, end)]
        return [r // abs(st) for r, st in zip(res, stride)]
    else:
        assert False


# 0-1 array: [lsb, ...]
def bit_mask_to_array(mask, rank):
    arr = list(reversed([int(d) for d in bin(mask)[2:]]))
    if rank == 0 and arr == [0]:
        arr = []
    assert len(arr) <= rank, "Invalid mask: {}, rank: {}".format(mask, rank)
    return arr + [0] * (rank - len(arr))


def decompose_strided_slice(input,  # type: ShapeType
                            begin,  # type: typing.List[int]
                            end,  # type: typing.List[int]
                            stride,  # type: typing.List[int]
                            ellipsis_mask,  # type: typing.Union[ZeroOneListType, int]
                            new_axis_mask,  # type: typing.Union[ZeroOneListType, int]
                            shrink_axis_mask,  # type: typing.Union[ZeroOneListType, int]
                            begin_mask,  # type: typing.Union[ZeroOneListType, int]
                            end_mask  # type: typing.Union[ZeroOneListType, int]
                            ):
    # type: (...)->DecomposedStridedSlice

    if not isinstance(ellipsis_mask, (list, tuple)):
        ellipsis_mask = bit_mask_to_array(ellipsis_mask, len(begin))
    if not isinstance(new_axis_mask, (list, tuple)):
        new_axis_mask = bit_mask_to_array(new_axis_mask, len(begin))
    if not isinstance(shrink_axis_mask, (list, tuple)):
        shrink_axis_mask = bit_mask_to_array(shrink_axis_mask, len(begin))
    if not isinstance(begin_mask, (list, tuple)):
        begin_mask = bit_mask_to_array(begin_mask, len(begin))
    if not isinstance(end_mask, (list, tuple)):
        end_mask = bit_mask_to_array(end_mask, len(begin))

    assert (len(begin) == len(end) == len(stride) == len(ellipsis_mask) == len(new_axis_mask)
            == len(shrink_axis_mask) == len(begin_mask) == len(end_mask))

    begin = [int(b) for b in begin]
    end = [int(e) for e in end]
    stride = [int(s) for s in stride]
    input_rank = len(input)
    mask_rank = len(begin)
    assert sum(ellipsis_mask) <= 1
    ellipsis_mask = ellipsis_mask.index(1) if 1 in ellipsis_mask else -1
    output_rank_before_shrink = input_rank + sum(new_axis_mask)

    arrays = [begin, end, stride, new_axis_mask, shrink_axis_mask, begin_mask, end_mask]
    if ellipsis_mask >= 0:
        if output_rank_before_shrink == mask_rank - 1:
            for arr in arrays:
                del arr[ellipsis_mask]
        elif output_rank_before_shrink == mask_rank:
            for arr in arrays:
                arr[ellipsis_mask] = 0
            begin_mask[ellipsis_mask] = 1
            end_mask[ellipsis_mask] = 1
            stride[ellipsis_mask] = 1
        elif output_rank_before_shrink > mask_rank:
            d = output_rank_before_shrink - mask_rank
            for arr in arrays:
                arr[ellipsis_mask] = 0
                for _ in range(d):
                    arr.insert(ellipsis_mask, 0)
            for i in range(ellipsis_mask, ellipsis_mask + d + 1):
                begin_mask[i] = 1
                end_mask[i] = 1
                stride[i] = 1
    elif mask_rank < output_rank_before_shrink:
        for i in range(mask_rank, output_rank_before_shrink):
            for arr in arrays:
                arr.append(0)
            begin_mask[i] = 1
            end_mask[i] = 1
            stride[i] = 1

    for arr in arrays:
        assert len(arr) == output_rank_before_shrink

    ssl_shape = [0] * input_rank
    shape_before_shrink = [0] * output_rank_before_shrink

    shape_idx = 0
    strided_slice_result_shape_index = 0
    for i in range(output_rank_before_shrink):
        if begin[i] < 0:
            begin[i] += input[shape_idx]
        if end[i] < 0:
            end[i] += input[shape_idx]

        if begin_mask[i] == 1:
            begin[i] = 0
        if end_mask[i] == 1:
            end[i] = input[shape_idx]

        if new_axis_mask[i] == 1:
            shape_before_shrink[i] = 1
        else:
            shape_before_shrink[i] = (end[i] - begin[i]) // abs(stride[i])
            ssl_shape[strided_slice_result_shape_index] = (end[i] - begin[i]) // abs(stride[i])
            strided_slice_result_shape_index += 1

        if new_axis_mask[i] == 0:
            shape_idx += 1

    if any(m > 0 for m in new_axis_mask) or any(m > 0 for m in shrink_axis_mask):
        reshape_shape = []
        for i in range(output_rank_before_shrink):
            if shrink_axis_mask[i] == 0:
                reshape_shape.append(shape_before_shrink[i])
    else:
        reshape_shape = list(ssl_shape)

    ssl_begin = [b for b, nm in zip(begin, new_axis_mask) if not nm]
    ssl_end = [e for e, nm in zip(end, new_axis_mask) if not nm]
    ssl_stride = [s for s, nm in zip(stride, new_axis_mask) if not nm]

    return DecomposedStridedSlice(ssl_begin, ssl_end, ssl_stride, ssl_shape, reshape_shape)


def strided_slice(input, begin, end, stride, ellipsis_mask, new_axis_mask, shrink_axis_mask, begin_mask, end_mask):
    return decompose_strided_slice(
        input, begin, end, stride, ellipsis_mask, new_axis_mask, shrink_axis_mask, begin_mask, end_mask).reshape_shape


def tile(input, repeat):
    assert len(input) == len(repeat)
    return [i * r for i, r in zip(input, repeat)]
