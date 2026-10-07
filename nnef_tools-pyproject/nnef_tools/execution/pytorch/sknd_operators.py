# Copyright (c) 2017-2025 The Khronos Group Inc.
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

from typing import Optional, List, Tuple, Callable, Any
from functools import reduce
import numpy as np
import functools
import torch
import torch.nn.functional as F
import math


def _expand_to_rank(input, rank, align=None):
    # type: (torch.Tensor, int, int)->torch.Tensor
    rank_diff = rank - len(input.shape)
    if align is None:
        return input.reshape(rank_diff * (1,) + tuple(input.shape))
    elif align < 0:
        return input.reshape((rank_diff + align) * (1,) + tuple(input.shape) + -align * (1,))
    else:
        return input.reshape(align * (1,) + tuple(input.shape) + (rank_diff - align) * (1,))


def _binary(f):
    def func(lhs, rhs, lhs_align=None, rhs_align=None):
        rank = max(len(lhs.shape), len(rhs.shape))
        return f(_expand_to_rank(lhs, rank, lhs_align), _expand_to_rank(rhs, rank, rhs_align))
    return func


def math_select(cond, lhs, rhs, cond_align, lhs_align, rhs_align):
    rank = max(len(cond.shape), len(lhs.shape), len(rhs.shape))
    return torch.where(_expand_to_rank(cond, rank, cond_align),
                       _expand_to_rank(lhs, rank, lhs_align),
                       _expand_to_rank(rhs, rank, rhs_align))


def _reduce(input, f, axes, squeeze=False):
    # type:(torch.Tensor, Callable, List[int], bool)->torch.Tensor
    if not axes:
        return input
    for axis in reversed(sorted(axes)):
        result = f(input=input, dim=axis, keepdim=not squeeze)
        input = result[0] if isinstance(result, tuple) else result
    return input


def _apply_n(inputs, binary):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return binary(inputs[0], _apply_n(inputs[1:], binary))


def nn_softmax(x, axes=None):
    # type: (torch.Tensor, Optional[List[int]])->torch.Tensor

    axes = [1] if axes is None else axes

    if len(axes) == 0:
        return x
    elif len(axes) == 1:
        return F.softmax(x, dim=axes[0])
    else:
        m = _reduce(x, torch.max, axes=axes)
        e = torch.exp(x - m)
        return e / _reduce(x, torch.sum, axes=axes)


def layout_tile(input, axes, repeats):
    reps = [1] * len(input.shape)
    for axis, repeat in zip(axes, repeats):
        reps[axis] = repeat
    return input.repeat(*reps)


def _axes_to_ncx(rank, layout):
    if layout == 'NCX':
        return list(range(rank))
    elif layout == 'NXC':
        return [0, rank - 1] + list(range(1, rank - 1))
    elif layout == 'XCN':
        return [rank - 1, rank - 2] + list(range(rank - 2))
    elif layout == 'CXN':
        return [rank - 1, 0] + list(range(1, rank - 1))
    else:
        raise ValueError("layout '{}' is unsupported".format(layout))


def _inverse_axes(axes):
    inverse = [0] * len(axes)
    for index, axis in enumerate(axes):
        inverse[axis] = index
    return inverse


def _permute(tensor, axes):
    if axes == list(range(len(axes))):
        return tensor
    return tensor.permute(axes).contiguous()


def _paddings(input_size, kernel, stride, dilation, padding, padding_align, ceil_mode, transposed, output_size=None):
    spatial = len(input_size)
    if padding is None:
        before = []
        after = []
        for index, (size, width, step, rate) in enumerate(zip(input_size, kernel, stride, dilation)):
            span = (width - 1) * rate + 1
            if transposed:
                out = size * step if output_size is None else output_size[index]
                total = (size - 1) * step + span - out
            else:
                divided = -(size // -step) if ceil_mode else size // step
                total = (divided - 1) * step + span - size
            lead = total // 2 if padding_align == 'UPPER' else -(total // -2)
            before.append(lead)
            after.append(total - lead)
    else:
        before = padding[:spatial]
        after = padding[spatial:]
    return before, after


def _pad_input(tensor, before, after, value=0):
    pad = []
    for left, right in zip(reversed(before), reversed(after)):
        pad.extend((left, right))
    if any(pad):
        tensor = F.pad(tensor, pad, value=value)
    return tensor


def _crop_spatial(tensor, before, after):
    slices = [slice(None), slice(None)]
    for lead, trail, size in zip(before, after, tensor.shape[2:]):
        stop = None if trail == 0 else size - trail
        start = None if lead == 0 else lead
        slices.append(slice(start, stop))
    return tensor[tuple(slices)]


def _as_ncx(tensor, layout):
    axes = _axes_to_ncx(len(tensor.shape), layout)
    return _permute(tensor, axes), axes


def nn_conv(input, filter, bias, stride, dilation, padding, padding_align, ceil_mode, groups, data_format, filter_format):
    spatial = len(input.shape) - 2
    assert spatial in (1, 2, 3), "nn.conv is only implemented for 1D, 2D and 3D, given: {}D.".format(spatial)

    input, data_axes = _as_ncx(input, data_format)
    filter, _ = _as_ncx(filter, filter_format)
    if groups == 0:
        groups = input.shape[1]

    before, after = _paddings(input.shape[2:], filter.shape[2:], stride, dilation, padding, padding_align, ceil_mode,
                               False)
    if before == after:
        conv_padding = tuple(before)
    else:
        input = _pad_input(input, before, after)
        conv_padding = 0
    conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[spatial]
    output = conv(input, filter, bias, stride=tuple(stride), padding=conv_padding, dilation=tuple(dilation),
                  groups=groups)
    return _permute(output, _inverse_axes(data_axes))


def nn_deconv(input, filter, bias, stride, dilation, padding, padding_align, output_size, groups, data_format,
              filter_format):
    spatial = len(input.shape) - 2
    assert spatial in (1, 2, 3), "nn.deconv is only implemented for 1D, 2D and 3D, given: {}D.".format(spatial)

    input, data_axes = _as_ncx(input, data_format)
    filter, _ = _as_ncx(filter, filter_format)
    if groups == 0:
        groups = input.shape[1]

    before, after = _paddings(input.shape[2:], filter.shape[2:], stride, dilation, padding, padding_align, False, True,
                               output_size)
    deconv = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}[spatial]
    if before == after:
        output = deconv(input, filter, bias, stride=tuple(stride), padding=tuple(before), dilation=tuple(dilation),
                        groups=groups)
    else:
        output = deconv(input, filter, bias, stride=tuple(stride), padding=0, dilation=tuple(dilation), groups=groups)
        output = _crop_spatial(output, before, after)
    return _permute(output, _inverse_axes(data_axes))


def _pool(input, axes, size, stride, dilation, padding, padding_align, ceil_mode, pools,
          pad_value=0, count_include_pad=None):
    rank = len(input.shape)
    axes = [axis + rank if axis < 0 else axis for axis in axes]
    spatial = len(axes)
    assert spatial in (1, 2, 3), "pooling is only implemented for 1D, 2D and 3D windows, given: {}D.".format(spatial)

    rest = [axis for axis in range(rank) if axis not in axes]
    order = rest + axes
    input = _permute(input, order)
    before, after = _paddings(input.shape[-spatial:], size, stride, dilation, padding, padding_align, ceil_mode, False)
    if before == after:
        pool_padding = tuple(before)
    else:
        input = _pad_input(input, before, after, pad_value)
        pool_padding = 0
    kwargs = {}
    if not all(rate == 1 for rate in dilation):
        kwargs['dilation'] = tuple(dilation)
    if count_include_pad is not None:
        kwargs['count_include_pad'] = count_include_pad
    output = pools[spatial](input, tuple(size), stride=tuple(stride), padding=pool_padding, **kwargs)
    return _permute(output, _inverse_axes(order))


def nn_max_pool(input, axes, size, stride, dilation, padding, padding_align, ceil_mode):
    pools = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}
    return _pool(input, axes, size, stride, dilation, padding, padding_align, ceil_mode, pools,
                 pad_value=float('-inf'))


def nn_sum_pool(input, axes, size, stride, dilation, padding, padding_align, ceil_mode):
    assert all(rate == 1 for rate in dilation), "nn.sum_pool is only implemented for dilation 1, given: {}.".format(dilation)
    pools = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}
    averaged = _pool(input, axes, size, stride, dilation, padding, padding_align, ceil_mode, pools,
                     count_include_pad=True)
    return averaged * math.prod(size)


def nn_avg_pool(input, axes, size, stride, dilation, padding, padding_align, ignore_border, ceil_mode):
    assert all(rate == 1 for rate in dilation), "nn.avg_pool is only implemented for dilation 1, given: {}.".format(dilation)
    rank = len(input.shape)
    spatial_size = [input.shape[axis + rank if axis < 0 else axis] for axis in axes]
    before, after = _paddings(spatial_size, size, stride, dilation, padding, padding_align, ceil_mode, False)
    if ignore_border and before != after:
        summed = nn_sum_pool(input, axes, size, stride, dilation, padding, padding_align, ceil_mode)
        counts = nn_sum_pool(torch.ones_like(input), axes, size, stride, dilation, padding, padding_align, ceil_mode)
        return summed / counts
    pools = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}
    return _pool(input, axes, size, stride, dilation, padding, padding_align, ceil_mode, pools,
                 count_include_pad=not ignore_border)


"""
Supported primitive and atomic operators (all other compounds are inlined)
"""
Operators = {
    'math.add': _binary(lambda x, y: x + y),
    'math.sub': _binary(lambda x, y: x - y),
    'math.mul': _binary(lambda x, y: x * y),
    'math.div': _binary(lambda x, y: x / y),
    'math.mod': _binary(torch.remainder),
    'math.pow': _binary(torch.pow),
    'math.min': _binary(torch.min),
    'math.max': _binary(torch.max),
    'math.lt': _binary(lambda x, y: x < y),
    'math.gt': _binary(lambda x, y: x > y),
    'math.le': _binary(lambda x, y: x <= y),
    'math.ge': _binary(lambda x, y: x >= y),
    'math.eq': _binary(torch.eq),
    'math.ne': _binary(torch.ne),
    'math.and': _binary(lambda x, y: x & y),
    'math.or': _binary(lambda x, y: x | y),
    'math.exp': torch.exp,
    'math.log': torch.log,
    'math.abs': torch.abs,
    'math.sign': torch.sign,
    'math.rcp': torch.reciprocal,
    'math.neg': torch.neg,
    'math.iden': torch.clone,
    'math.not': lambda x: ~x,
    'math.floor': torch.floor,
    'math.ceil': torch.ceil,
    'math.round': torch.round,
    'math.select': math_select,
    'math.sqr': lambda x: torch.pow(x, 2.0),
    'math.sqrt': torch.sqrt,
    'math.rsqr': lambda x: torch.pow(x, -2.0),
    'math.rsqrt': torch.rsqrt,
    'math.log2': torch.log2,
    'math.sin': lambda x: torch.sin(x),
    'math.cos': lambda x: torch.cos(x),
    'math.tan': lambda x: torch.tan(x),
    'math.asin': lambda x: torch.asin(x),
    'math.acos': lambda x: torch.acos(x),
    'math.atan': lambda x: torch.atan(x),
    'math.sinh': lambda x: torch.sinh(x),
    'math.cosh': lambda x: torch.cosh(x),
    'math.tanh': lambda x: torch.tanh(x),
    'math.asinh': lambda x: torch.asinh(x),
    'math.acosh': lambda x: torch.acosh(x),
    'math.atanh': lambda x: torch.atanh(x),
    'math.sum_reduce': lambda input, axes, squeeze: _reduce(input, torch.sum, axes=axes, squeeze=squeeze),
    'math.mean_reduce': lambda input, axes, squeeze: _reduce(input, torch.mean, axes=axes, squeeze=squeeze),
    'math.prod_reduce': lambda input, axes, squeeze: _reduce(input, torch.prod, axes=axes, squeeze=squeeze),
    'math.min_reduce': lambda input, axes, squeeze: _reduce(input, torch.min, axes=axes, squeeze=squeeze),
    'math.max_reduce': lambda input, axes, squeeze: _reduce(input, torch.max, axes=axes, squeeze=squeeze),
    'math.any_reduce': lambda input, axes, squeeze: _reduce(input, torch.any, axes=axes, squeeze=squeeze),
    'math.all_reduce': lambda input, axes, squeeze: _reduce(input, torch.all, axes=axes, squeeze=squeeze),
    'math.sum_n': lambda inputs: _apply_n(inputs, torch.add),
    'math.prod_n': lambda inputs: _apply_n(inputs, torch.mul),
    'math.min_n': lambda inputs: _apply_n(inputs, torch.minimum),
    'math.max_n': lambda inputs: _apply_n(inputs, torch.maximum),
    'math.any_n': lambda inputs: _apply_n(inputs, torch.logical_or),
    'math.all_n': lambda inputs: _apply_n(inputs, torch.logical_and),
    'nn.relu': F.relu,
    'nn.sigmoid': torch.sigmoid,
    'nn.softabs': lambda x, epsilon: torch.sqrt(torch.pow(x, 2.0) + epsilon),
    'nn.softmax': nn_softmax,
    'nn.softplus': lambda x: torch.log(torch.exp(x) + 1.0),
    'nn.elu': F.elu,
    'nn.selu': lambda x, alpha, _lambda_: F.selu(x),
    'nn.gelu': F.gelu,
    'nn.silu': lambda x: x * torch.sigmoid(x),
    'nn.prelu': lambda x, alpha: F.prelu(x, alpha),
    'nn.leaky_relu': lambda x, alpha: F.leaky_relu(x, alpha),
    'nn.conv': nn_conv,
    'nn.deconv': nn_deconv,
    'nn.max_pool': nn_max_pool,
    'nn.sum_pool': nn_sum_pool,
    'nn.avg_pool': nn_avg_pool,
    'layout.tile': layout_tile,
}
