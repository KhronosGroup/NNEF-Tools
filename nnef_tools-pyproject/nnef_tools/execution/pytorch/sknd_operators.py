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
from skriptnd import Dtype, DtypeToNumpy as _sknd_dtype_to_numpy
import numpy as np
import builtins
import torch
import torchvision
import torch.nn.functional as F
import math


_numpy_dtype_to_torch = {
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.uint8: torch.uint8,
    np.double: torch.double,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.short: torch.short,
    np.longlong: torch.long,
    np.bool: torch.bool,
    int: torch.int,
    bool: torch.bool,
    float: torch.float,
}


IntType = _numpy_dtype_to_torch[_sknd_dtype_to_numpy[Dtype.Int]]
BoolType = _numpy_dtype_to_torch[_sknd_dtype_to_numpy[Dtype.Bool]]
RealType = _numpy_dtype_to_torch[_sknd_dtype_to_numpy[Dtype.Real]]


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


def math_clamp(val, min, max, val_align, min_align, max_align):
    rank = builtins.max(len(val.shape), len(min.shape), len(max.shape))
    return torch.clamp(_expand_to_rank(val, rank, val_align),
                       _expand_to_rank(min, rank, min_align),
                       _expand_to_rank(max, rank, max_align))


def _reduce(input, f, axes, squeeze=False):
    # type:(torch.Tensor, Callable, List[int], bool)->torch.Tensor
    if not axes:
        return input
    for axis in reversed(sorted(axes)):
        result = f(input=input, dim=axis, keepdim=not squeeze)
        input = result[0] if isinstance(result, tuple) else result
    return input


def _arg_nd(input, axes, squeeze, fn):
    rank = input.dim()
    axes = [axis + rank if axis < 0 else axis for axis in axes]
    kept = [axis for axis in range(rank) if axis not in axes]
    order = sorted(axes)
    reduced = _permute(input, kept + order).reshape(*[input.shape[axis] for axis in kept], -1)
    flat = fn(reduced, dim=-1)
    coords = []
    for size in reversed([input.shape[axis] for axis in order]):
        coords.append(flat % size)
        flat = flat // size
    coords.reverse()
    index = dict(zip(order, coords))
    result = torch.stack([index[axis] for axis in axes], dim=-1)
    if not squeeze:
        for axis in order:
            result = result.unsqueeze(axis)
    return result


def _apply_n(inputs, binary):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return binary(inputs[0], _apply_n(inputs[1:], binary))


def nn_relu(x, alpha, max):
    x = F.leaky_relu(x, alpha) if alpha is not None else F.relu(x)
    return x if max is None else torch.minimum(x, max)


def nn_prelu(x, alpha, axis):
    rank = len(x.shape)
    if axis != 1:
        order = [index for index in range(rank) if index != axis]
        order.insert(1, axis)
        x = _permute(x, order)

    y = F.prelu(x, alpha)

    if axis != 1:
        y = _permute(y, _inverse_axes(order))
    return y


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


def layout_tensor(shape, value, T):
    return torch.tensor(np.array(value, dtype=T).reshape(shape))


def layout_reshape(input, axis, rank, shape):
    if axis < 0:
        axis += len(input.shape)
    shape = tuple(shape)
    if rank != len(input.shape):
        shape = input.shape[:axis] + shape + input.shape[axis + rank:]
    return torch.reshape(input, shape)


def layout_flatten(input, axis, rank):
    if axis < 0:
        axis += len(input.shape)
    if rank == 0:
        return input
    return torch.flatten(input, start_dim=axis, end_dim=axis + rank - 1)


def layout_unflatten(input, axis, shape):
    if axis < 0:
        axis += len(input.shape)
    shape = input.shape[:axis] + tuple(shape) + input.shape[axis + 1:]
    return torch.reshape(input, shape)


def layout_transpose(input, axis, perm):
    if axis < 0:
        axis += len(input.shape)
    perm = tuple(perm)
    if axis != 0:
        perm = tuple(range(0, axis)) + perm
    return input.permute(perm)


def layout_tile(input, axes, repeats):
    reps = [1] * len(input.shape)
    for axis, repeat in zip(axes, repeats):
        reps[axis] = repeat
    return input.repeat(*reps)


def layout_broadcast(input, axes, shape):
    rank = len(input.shape)
    out_shape = list(input.shape)
    for axis, extent in zip(axes, shape):
        if axis < 0:
            axis += rank
        if extent != 1:
            out_shape[axis] = extent
    return input.broadcast_to(out_shape)


def layout_pad(input, value, axes, padding, method):
    rank = input.dim()
    before = [0] * rank
    after = [0] * rank
    count = len(axes)
    for axis, lead, trail in zip(axes, padding[:count], padding[count:]):
        if axis < 0:
            axis += rank
        before[axis] = lead
        after[axis] = trail
    return _pad(input, before, after, method, 0 if value is None else value.item())


def layout_slice(input, axes, begin, end, stride):
    rank = len(input.shape)
    slices = [slice(None)] * rank
    reverse = []
    for axis, start, stop, step in zip(axes, begin, end, stride):
        if axis < 0:
            axis += rank
        stop = min(max(stop, -1), input.shape[axis])
        if step < 0:
            reverse.append(axis)
            step = -step
            start, stop = stop + 1, start + 1
            start += (stop - start - 1) % step
        slices[axis] = slice(start, None if stop == -1 else stop, step)
    input = input[tuple(slices)]
    if reverse:
        input = input.flip(reverse)
    return input


def layout_squeeze(input, axes):
    return input.squeeze(tuple(axes))


def layout_unsqueeze(input, axes):
    rank = len(input.shape) + len(axes)
    for axis in sorted(axis + rank if axis < 0 else axis for axis in axes):
        input = input.unsqueeze(axis)
    return input


def layout_concat(inputs, axis):
    return torch.cat(inputs, dim=axis)


def layout_split(input, axis, count, sizes):
    return torch.split(input, tuple(sizes), dim=axis)


def layout_gather(data, index, axis):
    selector = [slice(None)] * data.dim()
    selector[axis] = index
    return data[tuple(selector)]


def layout_scatter(data, indices, updates, axis):
    return data.scatter(axis, indices.to(torch.int64), updates)


def _nd_indices(data, indices, batch_dims):
    indices = indices.to(torch.int64)
    batch_idx = []
    for axis, size in enumerate(data.shape[:batch_dims]):
        view = [1] * (indices.dim() - 1)
        view[axis] = size
        batch_idx.append(torch.arange(size, device=indices.device).view(view))
    return tuple(batch_idx) + tuple(indices.unbind(-1))


def layout_gather_nd(data, indices, batch_dims):
    return data[_nd_indices(data, indices, batch_dims)]


def layout_scatter_nd(data, indices, updates, batch_dims):
    return data.index_put(_nd_indices(data, indices, batch_dims), updates)


def image_resize(input, axes, size, mode, coordinate_transform, rounding_method, antialias, cubic_coeff_a):
    rank = input.dim()
    axes = [axis + rank if axis < 0 else axis for axis in axes]
    resized = [(axis, extent) for axis, extent in zip(axes, size) if extent != input.shape[axis]]
    assert len(resized) <= 3, "image.resize supports at most 3 resized axes, got {}".format(len(resized))
    axes = [axis for axis, extent in resized]
    size = [extent for axis, extent in resized]
    if not axes:
        return input

    spatial = len(axes)
    if mode == 'NEAREST':
        assert coordinate_transform == 'ASYMMETRIC', \
            "image.resize nearest only supports coordinate_transform 'ASYMMETRIC', got '{}''".format(
                coordinate_transform)
        assert rounding_method == 'FLOOR', \
            "image.resize nearest only supports rounding_method 'FLOOR', got '{}'".format(rounding_method)
        mode = 'nearest'
        align_corners = None
    elif mode == 'LINEAR':
        assert coordinate_transform in ('SYMMETRIC', 'ALIGNED'), \
            "image.resize linear only supports coordinate_transform 'SYMMETRIC' and 'ALIGNED', got '{}'".format(
                coordinate_transform)
        mode = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}[spatial]
        align_corners = coordinate_transform == 'ALIGNED'
    elif mode == 'CUBIC':
        assert spatial == 2, "image.resize cubic only supports 2 resized axes, got {}".format(spatial)
        assert cubic_coeff_a == -0.75, \
            "image.resize cubic only supports cubic_coeff_a -0.75, got {}".format(cubic_coeff_a)
        assert coordinate_transform in ('SYMMETRIC', 'ALIGNED'), \
            "image.resize cubic only supports coordinate_transform 'SYMMETRIC' and 'ALIGNED', got '{}'".format(
                coordinate_transform)
        mode = 'bicubic'
        align_corners = coordinate_transform == 'ALIGNED'
    else:
        assert False, "image.resize mode '{}' is unsupported".format(mode)

    kwargs = {'antialias': False, 'align_corners': align_corners}

    kept = [axis for axis in range(rank) if axis not in axes]
    tensor = _permute(input, kept + axes)
    leading = len(kept)
    lead_shape = tensor.shape[:leading]
    if leading < 2:
        tensor = tensor.reshape((1,) * (2 - leading) + tuple(tensor.shape))
    elif leading > 2:
        tensor = tensor.reshape(-1, lead_shape[-1], *tensor.shape[leading:])
    tensor = F.interpolate(tensor, size=tuple(size), mode=mode, **kwargs)
    if leading < 2:
        tensor = tensor.reshape(tensor.shape[2 - leading:])
    elif leading > 2:
        tensor = tensor.reshape(*lead_shape[:-1], tensor.shape[1], *tensor.shape[2:])
    return _permute(tensor, _inverse_axes(kept + axes))


def image_rescale(input, axes, factor, mode, coordinate_transform, rounding_method=None, antialias=None, cubic_coeff_a=None):
    is_integer_upscale = all(int(f) == f for f in factor)
    is_integer_downsample = all(f <= 1 and (1 / round(1 / f) == f) for f in factor)
    if is_integer_upscale and mode == 'LINEAR' and coordinate_transform == 'ASYMMETRIC':
        axes = [axis for axis, f in zip(axes, factor) if f != 1]
        factor = [int(f) for f in factor if f != 1]
        return image_linear_upsample(input, axes, factor, symmetric=False, replicate_border=True)
    elif is_integer_downsample and mode == 'NEAREST' and coordinate_transform == 'SYMMETRIC' and rounding_method == 'ROUND_PREFER_FLOOR':
        axes = [axis for axis, f in zip(axes, factor) if f != 1]
        factor = [int(round(1 / f))for f in factor if f != 1]
        return image_nearest_downsample(input, axes, factor)
    else:
        size = [int(round(input.shape[axis] * scale)) for axis, scale in zip(axes, factor)]
        return image_resize(input, axes, size, mode, coordinate_transform, rounding_method, antialias, cubic_coeff_a)


def image_nearest_downsample(input, axes, factor):
    rank = len(axes)
    return nn_sum_pool(input, axes, size=[1] * rank, stride=factor,
                       dilation=[1] * rank, padding=[0] * (rank * 2))


def image_nearest_upsample(input, axes, factor):
    return image_rescale(input, axes, factor, mode='NEAREST', coordinate_transform='ASYMMETRIC',
                         rounding_method='FLOOR')


def image_area_downsample(input, axes, factor):
    rank = len(axes)
    return nn_avg_pool(input, axes, size=factor, stride=factor, dilation=[1] * rank, padding=[0] * (rank * 2))


def _upsample_weights_1d(factor, symmetric):
    size = 2 * factor - factor % 2 if symmetric else 2 * factor - 1
    offset = 0.5 if symmetric and factor % 2 == 0 else 1.0
    weights = [1.0 - abs(i - factor + offset) / factor for i in range(size)]
    return np.array(weights)


def _upsample_weights_2d(factor, symmetric):
    w0 = _upsample_weights_1d(factor[0], symmetric)
    w1 = _upsample_weights_1d(factor[1], symmetric)
    return np.outer(w0, w1)


def _upsample_weights_nd(factor, symmetric):
    ws = [_upsample_weights_1d(f, symmetric) for f in factor]
    return reduce(np.multiply, np.ix_(*ws))


def _upsample_filter_and_bias(factor, symmetric, channels, dtype, device):
    rank = len(factor)
    weights = _upsample_weights_nd(factor, symmetric)
    weights = np.tile(np.reshape(weights, newshape=(1, 1) + weights.shape), reps=(channels, 1) + (1,) * rank)
    filter = torch.from_numpy(weights).to(device=device, dtype=dtype)
    bias = torch.zeros(size=(channels,), device=device, dtype=dtype)
    return filter, bias


def image_linear_upsample(input, axes, factor, symmetric, replicate_border):
    axes = [axis for axis, f in zip(axes, factor) if f != 1]
    factor = [f for f in factor if f != 1]

    kept = [axis for axis in range(len(input.shape)) if axis not in axes]
    input = _permute(input, kept + axes)

    rank = len(axes)
    channels = input.shape[1]
    output_size = [f * s for f, s in zip(factor, input.shape[2:])]

    if replicate_border:
        input = _pad(input, before=[0, 0] + [1] * rank, after=[0, 0] + [1] * rank, method='REPLICATE')

    filter, bias = _upsample_filter_and_bias(factor, symmetric, channels, input.dtype, input.device)
    deconv = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}[rank]
    output = deconv(input, filter, bias, stride=factor, padding=0, dilation=1, groups=channels)

    size = output.shape[2:]
    before = [(f // 2 if symmetric else f - 1) + int(replicate_border) * f for f in factor]
    after = [size[i] - before[i] - output_size[i] for i in range(rank)]

    output = _crop_spatial(output, before=before, after=after)
    return _permute(output, _inverse_axes(kept + axes))


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


def image_grid_sample(input, grid, mode, padding, aligned):
    mode = 'bilinear' if mode == 'LINEAR' else 'bicubic' if mode == 'CUBIC' else 'nearest'
    padding = 'reflection' if padding == 'REFLECT' else 'border' if padding == 'REPLICATE' else 'zeros'
    return F.grid_sample(input, grid, mode=mode, padding_mode=padding, align_corners=aligned)


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


def _pad_constant(tensor, before, after, value=0):
    pad = []
    for left, right in zip(reversed(before), reversed(after)):
        pad.extend((left, right))
    if any(pad):
        tensor = F.pad(tensor, pad, value=value)
    return tensor


def _pad_symmetric(tensor, before, after):
    for axis, (lead, trail) in enumerate(zip(before, after)):
        if lead == 0 and trail == 0:
            continue
        size = tensor.shape[axis]
        parts = []
        if lead:
            parts.append(tensor.narrow(axis, 0, lead).flip(axis))
        parts.append(tensor)
        if trail:
            parts.append(tensor.narrow(axis, size - trail, trail).flip(axis))
        tensor = torch.cat(parts, dim=axis)
    return tensor


def _pad(tensor, before, after, method, value=0):
    if method == 'CONSTANT':
        return _pad_constant(tensor, before, after, value)
    if method == 'SYMMETRIC':
        return _pad_symmetric(tensor, before, after)

    rank = tensor.dim()
    axes = [axis for axis in range(rank) if before[axis] or after[axis]]
    if not axes:
        return tensor
    order = [axis for axis in range(rank) if axis not in axes] + axes
    tensor = _permute(tensor, order)
    leading = rank - len(axes)
    shape = tensor.shape[:leading]
    if leading == 0:
        tensor = tensor.unsqueeze(0)
    elif leading > 2:
        tensor = tensor.reshape(-1, *tensor.shape[leading:])
    pad = []
    for axis in reversed(axes):
        pad.extend((before[axis], after[axis]))
    tensor = F.pad(tensor, tuple(pad), mode=method.lower())
    if leading == 0:
        tensor = tensor.squeeze(0)
    elif leading > 2:
        tensor = tensor.reshape(*shape, *tensor.shape[-len(axes):])
    return _permute(tensor, _inverse_axes(order))


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


def nn_conv(input, filter, bias, stride, dilation, padding, padding_align, ceil_mode, groups,
            data_format, filter_format):
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
        input = _pad_constant(input, before, after)
        conv_padding = 0
    conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[spatial]
    output = conv(input, filter, bias, stride=tuple(stride), padding=conv_padding, dilation=tuple(dilation),
                  groups=groups)
    return _permute(output, _inverse_axes(data_axes))


def nn_deconv(input, filter, bias, stride, dilation, padding, padding_align, output_size, groups,
              data_format, filter_format):
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
        input = _pad_constant(input, before, after, pad_value)
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


def nn_sum_pool(input, axes, size, stride, dilation, padding, padding_align=None, ceil_mode=None):
    assert all(rate == 1 for rate in dilation), "nn.sum_pool is only implemented for dilation 1, given: {}.".format(dilation)
    pools = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}
    averaged = _pool(input, axes, size, stride, dilation, padding, padding_align, ceil_mode, pools,
                     count_include_pad=True)
    return averaged * math.prod(size)


def nn_avg_pool(input, axes, size, stride, dilation, padding, padding_align=None, ignore_border=True, ceil_mode=None):
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


def linalg_dot(x, y, b):
    z = torch.dot(x, y)
    return z if b is None else z + b


def linalg_matvec(A, x, b, transA):
    y = torch.mv(A.transpose(0, 1) if transA else A, x)
    return y if b is None else y + b


def linalg_matmul(A, B, C, transA, transB):
    if transA:
        A = A.transpose(-2, -1)
    if transB:
        B = B.transpose(-2, -1)
    Z = torch.matmul(A, B)
    return Z if C is None else Z + C


def linalg_outer(x, y):
    if len(x.shape) == 1 and len(y.shape) == 1:
        return torch.outer(x, y)
    return x.reshape(x.shape + (1,) * len(y.shape)) * y


def nn_local_response_norm(input, axes, size, alpha, beta, bias):
    rank = len(input.shape)
    axis = axes[0] + rank if axes[0] < 0 else axes[0]

    if axis != 1:
        order = [index for index in range(rank) if index != axis]
        order.insert(1, axis)
        input = _permute(input, order)

    new_axes = 3 - len(input.shape)
    if new_axes > 0:
        input = input.reshape(input.shape + (1,) * new_axes)

    output = F.local_response_norm(input, size[0], alpha=alpha, beta=beta, k=bias)

    if new_axes > 0:
        output = output.reshape(output.shape[:-new_axes])
    if axis != 1:
        output = _permute(output, _inverse_axes(order))
    return output


def nn_batch_norm(input, mean, variance, bias, scale, epsilon, channel_axis):
    rank = len(input.shape)
    axis = channel_axis + rank if channel_axis < 0 else channel_axis

    if axis != 1:
        order = [index for index in range(rank) if index != axis]
        order.insert(1, axis)
        input = _permute(input, order)

    output = F.batch_norm(input, mean, variance, scale, bias, training=False, eps=epsilon)

    if axis != 1:
        output = _permute(output, _inverse_axes(order))
    return output


def nn_lstm(X, W, R, B, h0, c0, steps):
    lstm = torch.nn.LSTM(X.shape[-1], W.shape[0] // 4, batch_first=False)
    lstm = lstm.to(device=X.device, dtype=X.dtype)
    with torch.no_grad():
        lstm.weight_ih_l0.copy_(W)
        lstm.weight_hh_l0.copy_(R)
        lstm.bias_ih_l0.copy_(B)
        lstm.bias_hh_l0.zero_()

    state = (h0.unsqueeze(0), c0.unsqueeze(0))
    if steps is None:
        Y, (hN, cN) = lstm(X, state)
    else:
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            X, steps.to(device='cpu', dtype=torch.int64), batch_first=False, enforce_sorted=False)
        packed_y, (hN, cN) = lstm(packed, state)
        Y, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_y, batch_first=False, total_length=X.shape[0])
    return Y, hN.squeeze(0), cN.squeeze(0)


def algo_nonmax_suppress(boxes, scores, box_format, max_outputs_per_class, iou_threshold, score_threshold):
    if box_format == 'CORNERS':
        x1 = torch.minimum(boxes[..., 1], boxes[..., 3])
        y1 = torch.minimum(boxes[..., 0], boxes[..., 2])
        x2 = torch.maximum(boxes[..., 1], boxes[..., 3])
        y2 = torch.maximum(boxes[..., 0], boxes[..., 2])
    elif box_format == 'CENTER':
        half_width = boxes[..., 2] / 2
        half_height = boxes[..., 3] / 2
        x1 = boxes[..., 0] - half_width
        y1 = boxes[..., 1] - half_height
        x2 = boxes[..., 0] + half_width
        y2 = boxes[..., 1] + half_height
    else:
        assert False, f"algo.nonmax_suppress only supports box_format 'CORNERS' and 'CENTER', got '{box_format}'"
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    selected = []
    for batch in range(scores.shape[0]):
        for cls in range(scores.shape[1]):
            class_scores = scores[batch, cls]
            indices = torchvision.ops.nms(boxes[batch], class_scores, iou_threshold)
            if score_threshold is not None:
                indices = indices[class_scores[indices] > score_threshold]
            if max_outputs_per_class is not None:
                indices = indices[:max_outputs_per_class]
            if indices.numel() != 0:
                selected.append(torch.stack([
                    torch.full_like(indices, batch),
                    torch.full_like(indices, cls),
                    indices,
                ], dim=1))

    return torch.cat(selected, dim=0) if selected else torch.empty((0, 3), dtype=IntType, device=boxes.device)


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
    'math.xor': _binary(lambda x, y: x ^ y),
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
    'math.argmin': lambda input, axis, squeeze: torch.argmin(input, dim=axis, keepdim=not squeeze),
    'math.argmax': lambda input, axis, squeeze: torch.argmax(input, dim=axis, keepdim=not squeeze),
    'math.argmin_nd': lambda input, axes, squeeze: _arg_nd(input, axes, squeeze, torch.argmin),
    'math.argmax_nd': lambda input, axes, squeeze: _arg_nd(input, axes, squeeze, torch.argmax),
    'math.sum_n': lambda inputs: _apply_n(inputs, torch.add),
    'math.prod_n': lambda inputs: _apply_n(inputs, torch.mul),
    'math.min_n': lambda inputs: _apply_n(inputs, torch.minimum),
    'math.max_n': lambda inputs: _apply_n(inputs, torch.maximum),
    'math.any_n': lambda inputs: _apply_n(inputs, torch.logical_or),
    'math.all_n': lambda inputs: _apply_n(inputs, torch.logical_and),
    'math.cumsum': lambda input, axis, exclusive, reverse: torch.cumsum(input, dim=axis),
    'math.clamp': math_clamp,
    'nn.relu': nn_relu,
    'nn.sigmoid': torch.sigmoid,
    'nn.softabs': lambda x, epsilon: torch.sqrt(torch.pow(x, 2.0) + epsilon),
    'nn.softmax': nn_softmax,
    'nn.softplus': lambda x: torch.log(torch.exp(x) + 1.0),
    'nn.elu': F.elu,
    'nn.selu': lambda x, alpha, _lambda_: F.selu(x),
    'nn.gelu': lambda x, approximate: F.gelu(x, approximate=(approximate or 'none').lower()),
    'nn.silu': lambda x: x * torch.sigmoid(x),
    'nn.prelu': nn_prelu,
    'nn.leaky_relu': lambda x, alpha: F.leaky_relu(x, alpha),
    'nn.thresholded_relu': lambda x, theta: torch.where(torch.gt(x, theta), x, 0.0),
    'nn.hard_sigmoid': lambda x, alpha, beta: torch.clamp(alpha * x + beta, 0.0, 1.0),
    'nn.erf': torch.erf,
    'nn.linear': F.linear,
    'nn.conv': nn_conv,
    'nn.deconv': nn_deconv,
    'nn.max_pool': nn_max_pool,
    'nn.sum_pool': nn_sum_pool,
    'nn.avg_pool': nn_avg_pool,
    'nn.local_response_norm': nn_local_response_norm,
    'nn.batch_norm': nn_batch_norm,
    'nn.lstm': nn_lstm,
    'linalg.dot': linalg_dot,
    'linalg.matvec': linalg_matvec,
    'linalg.matmul': linalg_matmul,
    'linalg.outer': linalg_outer,
    'layout.tensor': layout_tensor,
    'layout.reshape': layout_reshape,
    'layout.flatten': layout_flatten,
    'layout.unflatten': layout_unflatten,
    'layout.transpose': layout_transpose,
    'layout.tile': layout_tile,
    'layout.broadcast': layout_broadcast,
    'layout.pad': layout_pad,
    'layout.slice': layout_slice,
    'layout.squeeze': layout_squeeze,
    'layout.unsqueeze': layout_unsqueeze,
    'layout.concat': layout_concat,
    'layout.split': layout_split,
    'layout.gather': layout_gather,
    'layout.gather_nd': layout_gather_nd,
    'layout.scatter': layout_scatter,
    'layout.scatter_nd': layout_scatter_nd,
    'layout.cast': lambda x, R: x.to(_numpy_dtype_to_torch[R]),
    'layout.shape': lambda x: torch.tensor(x.shape, dtype=IntType),
    'layout.range': lambda first, last, stride: torch.range(first, last, stride, dtype=IntType),
    'layout.uniform': lambda value, shape: torch.full(shape, value),
    'layout.nonzero': lambda x: torch.nonzero(x).transpose(0,1),
    'image.resize': image_resize,
    'image.rescale': image_rescale,
    'image.nearest_downsample': image_nearest_downsample,
    'image.nearest_upsample': image_nearest_upsample,
    'image.area_downsample': image_area_downsample,
    'image.linear_upsample': image_linear_upsample,
    'image.grid_sample': image_grid_sample,
    'algo.top_k': lambda x, k, axis, largest, sorted: torch.topk(x, k, dim=axis, largest=largest, sorted=sorted),
    'algo.nonmax_suppress': algo_nonmax_suppress,
    '=': torch.clone,
}
