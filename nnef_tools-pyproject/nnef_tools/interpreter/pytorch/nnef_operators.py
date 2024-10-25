# Copyright (c) 2020 The Khronos Group Inc.
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
import nnef
import math


# Helpers


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
    int: torch.int,
    bool: torch.bool,
    float: torch.float,
}


def _clamp(x, a, b):
    return max(a, min(b, x))


def _expand_to_rank(input, rank):
    # type: (torch.Tensor, int)->torch.Tensor
    rank_diff = rank - len(input.shape)
    return input.reshape(tuple(input.shape) + rank_diff * (1,))


def _expand_binary(input1, input2):
    # type: (torch.Tensor, torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]
    rank = max(len(input1.shape), len(input2.shape))
    return _expand_to_rank(input1, rank), _expand_to_rank(input2, rank)


def _binary(f):
    def g(x, y):
        x, y = _expand_binary(x, y)
        return f(x, y)

    return g


def _prod(items):
    return functools.reduce(lambda x, y: x * y, items, 1)


def _same_padding(input, filter, stride, dilation):
    assert len(input) == len(filter) == len(stride) == len(dilation)

    output = [(ui + (s - 1)) // s for ui, s in zip(input, stride)]
    dilated = [(f - 1) * d + 1 for f, d in zip(filter, dilation)]
    total = [max(0, (di - 1) * s + df - ui) for di, s, df, ui in zip(output, stride, dilated, input)]

    return [(pad // 2, (pad + 1) // 2) for pad in total]


def _inverse_permutation(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def _apply_permutation(items, perm):
    return [items[ind] for ind in perm]


# Operations

def _positive_pad(input, padding, border='constant', value=0.0):
    # type: (torch.Tensor, List[Tuple[int, int]], str, float)->torch.Tensor

    assert all(p >= 0 and q >= 0 for p, q in padding), "Negative padding is not supported "

    assert padding
    assert len(input.shape) in (3, 4, 5)
    assert padding[:2] == [(0, 0), (0, 0)] or (padding[0] == (0, 0) and padding[-1] == (0, 0))
    assert border in ("constant", "reflect", "replicate")

    rank = len(input.shape)
    needs_transpose = padding[0] == (0, 0) and padding[1] != (0, 0) and padding[-1] == (0, 0)
    if needs_transpose:
        padding = padding[1:-1]
        input = input.permute([0, rank - 1] + list(range(1, rank - 1)))
    else:
        padding = padding[2:]

    pad = []
    for p, q in reversed(padding):
        pad += [p, q]

    padded = F.pad(input=input, pad=pad, mode=border, value=value) if not all(p == 0 for p in pad) else input

    if needs_transpose:
        padded = padded.permute([0] + list(range(2, rank)) + [1])

    return padded


def nnef_pad(input, padding, border='constant', value=0.0):
    # type: (torch.Tensor, List[Tuple[int, int]], str, float)->torch.Tensor

    assert padding, \
        "nnef.pad does not support empty list as padding"
    assert len(input.shape) in (3, 4, 5), \
        "nnef.pad is only implemented for 3D, 4D, 5D tensors; got: {}D.".format(len(input.shape))
    assert padding[:2] == [(0, 0), (0, 0)] or (padding[0] == (0, 0) and padding[-1] == (0, 0)), \
        "nnef.pad is not implemented in N, C dimensions; got: {}.".format(padding)

    if all(p <= 1 and q <= 1 for p, q in padding) and border == "reflect-even":
        border = "replicate"

    assert border in ("constant", "reflect", "replicate"), \
        "nnef.pad is only implemented with 'constant', 'reflect' and 'replicate' border; got: {}.".format(border)

    input = _positive_pad(input,
                          padding=[(p if p > 0 else 0, q if q > 0 else 0) for p, q in padding],
                          border=border,
                          value=value)

    return nnef_slice(input,
                      axes=list(range(len(input.shape))),
                      begin=[-p if p < 0 else 0 for p, _q in padding],
                      end=[q if q < 0 else 0 for _p, q in padding])


nnef_add = _binary(lambda x, y: x + y)


def nnef_add_n(values):
    return nnef_add(values[0], nnef_add_n(values[1:])) if len(values) > 1 else values[0]


def nnef_conv(input,  # type: torch.Tensor
              filter,  # type: torch.Tensor
              bias,  # type: torch.Tensor
              border='constant',  # type: str
              padding=None,  # type: Optional[List[Tuple[int, int]]]
              stride=None,  # type: Optional[List[int]]
              dilation=None,  # type: Optional[List[int]]
              groups=1,  # type: int
              ):
    # type: (...)->torch.Tensor

    assert len(input.shape) in (3, 4, 5), "nnef.conv is only implemented for 3D, 4D, 5D tensors, given: {}D.".format(len(input.shape))

    bias = bias.reshape(1, 1).expand((1, filter.shape[0])) if _prod(bias.size()) == 1 else bias

    spatial_dims = len(input.shape[2:])
    groups = input.shape[1] if groups == 0 else groups
    stride = [1] * spatial_dims if not stride else stride
    dilation = [1] * spatial_dims if not dilation else dilation
    if not padding:
        padding = _same_padding(input=input.shape[2:],
                                filter=filter.shape[2:],
                                stride=stride,
                                dilation=dilation)

    pad = nnef_pad(input=input, padding=[(0, 0)] * 2 + padding, border=border)
    conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[spatial_dims](input=pad,
                                                                 weight=filter,
                                                                 bias=bias.squeeze(dim=0).contiguous(),
                                                                 stride=tuple(stride),
                                                                 padding=0,
                                                                 dilation=tuple(dilation),
                                                                 groups=groups)

    return conv


def nnef_deconv(input,  # type: torch.Tensor
                filter,  # type: torch.Tensor
                bias,  # type: torch.Tensor
                border='constant',  # type: str
                padding=None,  # type: Optional[List[Tuple[int, int]]]
                stride=None,  # type: Optional[List[int]]
                dilation=None,  # type: Optional[List[int]]
                output_shape=None,  # type: Optional[List[int]]
                groups=1,  # type: int
                ):
    # type: (...)->torch.Tensor

    assert border == 'constant' or border == 'replicate', "nnef.deconv: '{}' border unsupported.".format(border)

    if output_shape and output_shape[0] != input.shape[0]:
        output_shape = list(output_shape)
        output_shape[0] = input.shape[0]

    rank = len(input.shape)
    assert rank in (3, 4, 5), "nnef.deconv is only implemented for 3D, 4D, 5D tensors, given: {}D.".format(len(input.shape))

    spatial_dims = len(input.shape[2:])
    stride = [1] * spatial_dims if not stride else stride
    dilation = [1] * spatial_dims if not dilation else dilation

    if groups == 0:
        if output_shape:
            groups = output_shape[1]
        else:
            # Planewise deconvolution without output_size, assuming that #(input channels) = #(output channels)
            groups = filter.shape[0]

    output_channels = filter.shape[1] * groups
    if output_shape:
        assert output_shape[1] == output_channels
    else:
        output_shape = nnef.shapes.deconv_shape(input=list(input.shape),
                                                filter=filter.shape,
                                                bias=bias.shape,
                                                border=border,
                                                padding=padding,
                                                stride=stride,
                                                dilation=dilation,
                                                groups=groups,
                                                output_shape=None)
    if not padding:
        padding = _same_padding(input=output_shape[2:],
                                filter=filter.shape[2:],
                                stride=stride,
                                dilation=dilation)

    if border == 'replicate':
        input = F.pad(input=input, pad=(1,) * 2 * spatial_dims, mode='replicate')
        padding = [(p + s, q + s) for (p, q), s in zip(padding, stride)]

    uncropped_output_shape = nnef.shapes.deconv_shape(input=list(input.shape),
                                                      filter=filter.shape,
                                                      bias=bias.shape,
                                                      border=border,
                                                      padding=[(0, 0)] * (rank - 2),
                                                      stride=stride,
                                                      dilation=dilation,
                                                      groups=groups,
                                                      output_shape=None)

    crop_before = [p for p, _q in padding]
    crop_after = [uncropped - out - before
                  for uncropped, out, before
                  in zip(uncropped_output_shape[2:], output_shape[2:], crop_before)]

    bias = bias.reshape(1, 1).expand((1, output_channels)) if _prod(bias.size()) == 1 else bias

    deconv = {1: F.conv_transpose1d,
              2: F.conv_transpose2d,
              3: F.conv_transpose3d}[spatial_dims](input=input,
                                                   weight=filter,
                                                   bias=bias.squeeze(dim=0).contiguous(),
                                                   stride=tuple(stride),
                                                   padding=0,
                                                   output_padding=0,
                                                   groups=groups,
                                                   dilation=tuple(dilation))

    return nnef_pad(deconv, padding=[(0, 0), (0, 0)] + [(-cb, -ca) for cb, ca in zip(crop_before, crop_after)])


def _evaluate_max_pool_or_box_params(input_shape, size, padding, stride, dilation):
    rank = len(input_shape)
    stride = [1] * rank if not stride else stride
    dilation = [1] * rank if not dilation else dilation
    padding = _same_padding(input=input_shape,
                            filter=size,
                            stride=stride,
                            dilation=dilation) if not padding else padding
    return padding, stride, dilation


def _max_pool_impl(input,  # type: torch.Tensor
                   size,  # type: List[int]
                   border='constant',  # type: str
                   padding=None,  # type: Optional[List[Tuple[int, int]]]
                   stride=None,  # type: Optional[List[int]]
                   dilation=None,  # type: Optional[List[int]]
                   with_index=False,  # type: bool
                   ):
    # type: (...)->torch.Tensor

    spatial_dims = len(input.shape) - 2
    value = float('-inf') if border == 'ignore' else 0.0
    border = 'constant' if border == 'ignore' else border

    pad = nnef_pad(input=input, padding=padding, border=border, value=value)

    result = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}[spatial_dims](input=pad,
                                                                               kernel_size=size[2:],
                                                                               stride=stride[2:],
                                                                               padding=0,
                                                                               dilation=dilation[2:],
                                                                               return_indices=with_index)
    return result


def _box_impl(input,  # type: torch.Tensor
              size,  # type: List[int]
              border,  # type: str
              padding,  # type: List[Tuple[int, int]]
              stride,  # type: List[int]
              dilation,  # type: List[int]
              normalize,  # type: bool
              ):
    # type: (...)->torch.Tensor

    assert 3 <= len(input.shape) <= 5
    assert len(input.shape) == len(size) == len(padding) == len(stride) == len(dilation)
    assert padding[:2] == [(0, 0), (0, 0)]
    assert size[:2] == stride[:2] == dilation[:2]

    assert not dilation or all(d == 1 for d in dilation), \
        "nnef.box (avg or sum pooling) is only implemented for dilation = 1."

    spatial_dims = len(input.shape) - 2

    pad = nnef_pad(input=input, padding=padding, border='constant' if border == 'ignore' else border)

    avg_pool = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[spatial_dims](
        input=pad,
        kernel_size=size[2:],
        stride=stride[2:],
        padding=0)

    if border == 'ignore' and normalize:
        ones = torch.ones_like(input)
        padded_ones = nnef_pad(input=ones, padding=padding, border='constant')
        avg_pool_ones = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[spatial_dims](
            input=padded_ones,
            kernel_size=size[2:],
            stride=stride[2:],
            padding=0)
        # If padding is big, zero averages can happen on the border, don't divide by zero
        avg_pool_ones = nnef_select(avg_pool_ones > 0, avg_pool_ones, torch.ones_like(avg_pool_ones))
        avg_pool /= avg_pool_ones

    if normalize:
        return avg_pool
    else:
        return avg_pool * _prod(size)


def _get_transform_for_box_or_max_pool(input_shape, active):
    # type: (List[int], List[bool])->Any
    assert len(input_shape) >= 3
    assert len(input_shape) == len(active)
    assert sum(active) <= 3, \
        "Sliding window operations are not supported if they have more than 3 'active' dimensions; got {}".format(sum(active))

    if 3 <= len(input_shape) <= 5 and not active[0] and not active[1]:  # Direct support
        return None, None, None, None
    else:
        inactive_dims = [i for i, a in enumerate(active) if not a]
        active_dims = [i for i, a in enumerate(active) if a]
        inactive_shape = [s for i, s in enumerate(input_shape) if i not in active_dims]
        active_shape = [s for i, s in enumerate(input_shape) if i in active_dims]
        perm = inactive_dims + active_dims
        perm_inv = _inverse_permutation(perm)
    return perm, perm_inv, inactive_shape, active_shape


def _box_or_max_pool(input,  # type: torch.Tensor
                     size,  # type: List[int]
                     border='constant',  # type: str
                     padding=None,  # type: Optional[List[Tuple[int, int]]],
                     stride=None,  # type: Optional[List[int]],
                     dilation=None,  # type: Optional[List[int]]
                     normalize=False,  # type: bool
                     is_max_pool=False,  # type: bool
                     ):
    assert not (normalize and is_max_pool)

    rank = len(input.shape)
    padding, stride, dilation = _evaluate_max_pool_or_box_params(input_shape=list(input.shape),
                                                                 size=size,
                                                                 padding=padding,
                                                                 stride=stride,
                                                                 dilation=dilation)
    active = [size_ != 1 or padding_ != (0, 0) or stride_ != 1 or dilation_ != 1
              for size_, padding_, stride_, dilation_
              in zip(size, padding, stride, dilation)]

    if sum(active) == 0:
        return input

    if rank < 3:
        perm, perm_inv, inactive_shape, active_shape = None, None, None, None
    else:
        perm, perm_inv, inactive_shape, active_shape = _get_transform_for_box_or_max_pool(list(input.shape), active)

    if rank < 3:
        input = input.unsqueeze(0).unsqueeze(0)
        size = [1, 1] + size
        padding = [(0, 0), (0, 0)] + padding
        stride = [1, 1] + stride
        dilation = [1, 1] + dilation
    elif perm is not None:
        input = input.permute(*perm)
        size = _apply_permutation(size, perm)
        padding = _apply_permutation(padding, perm)
        stride = _apply_permutation(stride, perm)
        dilation = _apply_permutation(dilation, perm)

        active_rank = len(active_shape)
        input = input.reshape(*[_prod(inactive_shape), 1] + active_shape)
        size = [1, 1] + size[-active_rank:]
        padding = [(0, 0), (0, 0)] + padding[-active_rank:]
        stride = [1, 1] + stride[-active_rank:]
        dilation = [1, 1] + dilation[-active_rank:]

    if is_max_pool:
        output = _max_pool_impl(
            input=input, size=size, border=border, padding=padding, stride=stride, dilation=dilation, with_index=False)
    else:
        output = _box_impl(input=input,
                           size=size,
                           border=border,
                           padding=padding,
                           stride=stride,
                           dilation=dilation,
                           normalize=normalize)

    if rank < 3:
        output = output.squeeze(0).squeeze(0)
    elif perm is not None:
        active_rank = len(active_shape)
        output = output.reshape(inactive_shape + list(output.shape)[-active_rank:])
        output = output.permute(*perm_inv)

    return output


def nnef_max_pool(input,  # type: torch.Tensor
                  size,  # type: List[int]
                  border='constant',  # type: str
                  padding=None,  # type: Optional[List[Tuple[int, int]]]
                  stride=None,  # type: Optional[List[int]]
                  dilation=None,  # type: Optional[List[int]]
                  ):
    # type: (...)->torch.Tensor
    return _box_or_max_pool(
        input, size=size, border=border, padding=padding, stride=stride, dilation=dilation, is_max_pool=True)


def nnef_max_pool_with_index(input,  # type: torch.Tensor
                             size,  # type: List[int]
                             border='constant',  # type: str
                             padding=None,  # type: Optional[List[Tuple[int, int]]]
                             stride=None,  # type: Optional[List[int]]
                             dilation=None,  # type: Optional[List[int]]
                             ):
    # type: (...)->torch.Tensor

    input_shape = list(input.shape)
    padding, stride, dilation = _evaluate_max_pool_or_box_params(input_shape=input_shape,
                                                                 size=size,
                                                                 padding=padding,
                                                                 stride=stride,
                                                                 dilation=dilation)

    assert len(input_shape) in (3, 4, 5), \
        "nnef.max_pool_with_index is only implemented for 3D, 4D, 5D tensors, given: {}D".format(len(input_shape))
    assert size[:2] == [1, 1], \
        "nnef.max_pool_with_index is only implemented for size = 1 in N and C dimensions"
    assert padding[:2] == [(0, 0), (0, 0)],\
        "nnef.max_pool_with_index is only implemented for padding = (0, 0) in N and C dimensions."
    assert stride[:2] == [1, 1], \
        "nnef.max_pool_with_index is only implemented for stride = 1 in N and C dimensions"
    assert dilation[:2] == [1, 1], \
        "nnef.max_pool_with_index is only implemented for dilation = 1 in N and C dimensions"

    return _max_pool_impl(input, size=size, border=border, padding=padding, stride=stride, dilation=dilation,
                          with_index=True)


def nnef_argmax_pool(input,  # type: torch.Tensor
                     size,  # type: List[int]
                     border='constant',  # type: str
                     padding=None,  # type: Optional[List[Tuple[int, int]]]
                     stride=None,  # type: Optional[List[int]]
                     dilation=None,  # type: Optional[List[int]]
                     ):
    # type: (...)->torch.Tensor
    _, index = nnef_max_pool_with_index(
        input, size=size, border=border, padding=padding, stride=stride, dilation=dilation)
    return index


def nnef_box(input,  # type: torch.Tensor
             size,  # type: List[int]
             border='constant',  # type: str
             padding=None,  # type: Optional[List[Tuple[int, int]]]
             stride=None,  # type: Optional[List[int]]
             dilation=None,  # type: Optional[List[int]]
             normalize=False,  # type: bool
             ):
    # type: (...)->torch.Tensor
    return _box_or_max_pool(
        input, size=size, border=border, padding=padding, stride=stride, dilation=dilation, normalize=normalize)


def nnef_debox(input,  # type: torch.Tensor
               size,  # type: List[int]
               border='constant',  # type: str
               padding=None,  # type: Optional[List[Tuple[int, int]]]
               stride=None,  # type: Optional[List[int]]
               dilation=None,  # type: Optional[List[int]]
               output_shape=None,  # type: Optional[List[int]]
               normalize=False,  # type: bool
               ):
    assert border in ('constant', 'ignore'), \
        "nnef.debox: '{}' border unsupported".format(border)
    assert len(size) in (3, 4, 5), \
        "nnef.debox is only implemented for 3D, 4D, 5D tensors, given: {}D".format(len(size))
    assert size[:2] == [1, 1], \
        "nnef.debox is only implemented for size = 1 in N and C dimensions"
    assert not padding or padding[:2] == [(0, 0), (0, 0)], \
        "nnef.debox is only implemented for padding = (0, 0) in N and C dimensions"
    assert not stride or stride[:2] == [1, 1], \
        "nnef.debox is only implemented for stride = 1 in N and C dimensions"
    assert not dilation or dilation[:2] == [1, 1], \
        "nnef.debox is only implemented for dilation = 1 in N and C dimensions"

    filter = torch.full(size=[input.shape[1], 1] + list(size)[2:],
                        fill_value=(1.0 / _prod(size) if normalize else 1.0),
                        device=input.device,
                        dtype=input.dtype)
    bias = torch.zeros(size=tuple(), device=input.device, dtype=input.dtype)

    return nnef_deconv(input=input,
                       filter=filter,
                       bias=bias,
                       border='constant',
                       padding=padding[2:] if padding else padding,
                       stride=stride[2:] if stride else stride,
                       dilation=dilation[2:] if dilation else dilation,
                       output_shape=output_shape,
                       groups=input.shape[1])


def nnef_avg_pool(input,  # type: torch.Tensor
                  size,  # type: List[int]
                  border='constant',  # type: str
                  padding=None,  # type: Optional[List[Tuple[int, int]]],
                  stride=None,  # type: Optional[List[int]],
                  dilation=None,  # type: Optional[List[int]]
                  ):
    # type: (...)->torch.Tensor
    return nnef_box(input, size=size, border=border, padding=padding, stride=stride, dilation=dilation, normalize=True)


def nnef_rms_pool(input,  # type: torch.Tensor
                  size,  # type: List[int]
                  border='constant',  # type: str
                  padding=None,  # type: Optional[List[Tuple[int, int]]],
                  stride=None,  # type: Optional[List[int]],
                  dilation=None,  # type: Optional[List[int]]
                  ):
    # type: (...)->torch.Tensor
    return torch.sqrt(nnef_avg_pool(torch.pow(input, 2.0),
                                    size=size,
                                    border=border,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation))


def nnef_desample(input,  # type: torch.Tensor
                  index,  # type: torch.Tensor
                  size,  # type: List[int]
                  border='constant',  # type: str
                  padding=None,  # type: Optional[List[Tuple[int, int]]]
                  stride=None,  # type: Optional[List[int]]
                  dilation=None,  # type: Optional[List[int]]
                  output_shape=None,  # type: Optional[List[int]]
                  ):
    # type: (...)->torch.Tensor

    if output_shape and output_shape[0] != input.shape[0]:
        output_shape = list(output_shape)
        output_shape[0] = input.shape[0]

    input_shape = list(input.shape)
    index_shape = list(index.shape)
    rank = len(input_shape)
    spatial_dims = len(input_shape[2:])

    assert len(input_shape) in (3, 4, 5), \
        "nnef.desample is only implemented for 3D, 4D, 5D tensors, given: {}D".format(len(input_shape))
    assert not size or size[:2] == [1, 1], \
        "nnef.desample is only implemented for size = 1 in N and C dimensions"
    assert not padding or padding[:2] == [(0, 0), (0, 0)], \
        "nnef.desample is only implemented for padding = (0, 0) in N and C dimensions"
    assert not stride or stride[:2] == [1, 1], \
        "nnef.desample is only implemented for stride = 1 in N and C dimensions"
    assert not dilation or all(d == 1 for d in dilation), \
        "nnef.desample is only implemented for dilation = 1"

    stride = [1] * rank if not stride else stride
    dilation = [1] * rank if not dilation else dilation

    if not padding:
        calculated_output_shape = [i * s for i, s in zip(input_shape, stride)]
        padding = _same_padding(input=calculated_output_shape,
                                filter=size,
                                stride=stride,
                                dilation=dilation)
    else:
        calculated_output_shape = nnef.shapes.desample_shape(input_shape, index_shape,
                                                             size=size, border=border, padding=padding,
                                                             stride=stride, dilation=dilation, output_shape=None)

    output_shape = output_shape if output_shape else calculated_output_shape
    padded_output_shape = [s + p + q for s, (p, q) in zip(output_shape, padding)]
    unpooled = {1: F.max_unpool1d, 2: F.max_unpool2d, 3: F.max_unpool3d}[spatial_dims](
        input=input, indices=index, kernel_size=size[2:], stride=stride[2:], padding=0, output_size=padded_output_shape)
    return nnef_slice(unpooled,
                      axes=list(range(rank)),
                      begin=[p for p, _q in padding],
                      end=[p + s for (p, _q), s in zip(padding, output_shape)])


def nnef_batch_normalization(input,  # type: torch.Tensor
                             mean,  # type: torch.Tensor
                             variance,  # type: torch.Tensor
                             offset,  # type: torch.Tensor
                             scale,  # type: torch.Tensor
                             epsilon,  # type: float
                             is_training=False,  # type: bool
                             momentum=0.1,  # type: float
                             ):
    # type: (...)->torch.Tensor

    if isinstance(mean, torch.nn.Parameter):
        mean.requires_grad = False
    if isinstance(variance, torch.nn.Parameter):
        variance.requires_grad = False

    return F.batch_norm(input=input,
                        running_mean=nnef_squeeze(mean, axes=[0]),
                        running_var=nnef_squeeze(variance, axes=[0]),
                        weight=nnef_squeeze(scale, axes=[0]),
                        bias=nnef_squeeze(offset, axes=[0]),
                        training=is_training,
                        momentum=momentum,
                        eps=epsilon)


def _upsample_weights_1d(factor, symmetric):
    if symmetric:
        weights = [1 - (i + 0.5) / factor for i in range(factor)]
        weights = list(reversed(weights)) + weights
    else:
        weights = [1 - abs(i) / float(factor) for i in range(-factor + 1, factor)]
    return np.array(weights)


def _upsample_weights_2d(factor, symmetric):
    w0 = _upsample_weights_1d(factor[0], symmetric)
    w1 = _upsample_weights_1d(factor[1], symmetric)
    return np.outer(w0, w1)


def _upsample_weights_nd(factor, symmetric):
    ws = [_upsample_weights_1d(f, symmetric) for f in factor]
    return reduce(np.multiply, np.ix_(*ws))


def nnef_multilinear_upsample(input, factor, method='symmetric', border='replicate'):
    # type: (torch.Tensor, List[int], str, str)->torch.Tensor

    rank = len(factor)
    assert len(input.shape) == rank + 2

    mode = 'linear' if rank == 1 else 'bilinear'

    if method == 'aligned':
        return F.interpolate(input=input, scale_factor=tuple(factor), mode=mode, align_corners=True)
    elif method == 'symmetric' and border == 'replicate':
        return F.interpolate(input=input, scale_factor=tuple(factor), mode=mode, align_corners=False)

    n, c, = input.shape[:2]

    symmetric = method == 'symmetric'
    replicate = border == 'replicate'
    weights = _upsample_weights_nd(factor, symmetric)
    weights = np.tile(np.reshape(weights, newshape=(1, 1) + weights.shape), reps=(c, 1) + (1,) * rank)
    filter = torch.from_numpy(weights).to(device=input.device, dtype=input.dtype)
    bias = torch.zeros(size=tuple(), device=input.device, dtype=input.dtype)

    output_shape = [n, c] + [f * s for f, s in zip(factor, input.shape[2:])]

    if symmetric:
        return nnef_deconv(input, filter, bias, stride=factor, padding=[(f - 1, f - 1) for f in factor],
                           border='constant', groups=c, output_shape=output_shape)
    else:
        if replicate:
            input = nnef_pad(input, padding=[(0, 0), (0, 0)] + [(1, 0)] * rank, border=border)

        padding = factor if replicate else [f // 2 for f in factor]
        return nnef_deconv(input, filter, bias, stride=factor, padding=[(p, p) for p in padding],
                           border='constant', groups=c, output_shape=output_shape)


def nnef_nearest_upsample(input, factor):
    # type: (torch.Tensor, List[int])->torch.Tensor

    assert len(input.shape) in (3, 4, 5), \
        "nnef.nearest_upsample is only implemented for 3D, 4D, 5D tensors, given: {}D.".format(len(input.shape))

    return F.interpolate(input=input, scale_factor=tuple(factor), mode='nearest')


def nnef_softmax(x, axes=None):
    # type: (torch.Tensor, Optional[List[int]])->torch.Tensor

    axes = [1] if axes is None else axes

    if len(axes) == 0:
        return x
    elif len(axes) == 1:
        return F.softmax(x, dim=axes[0])
    else:
        m = nnef_max_reduce(x, axes=axes)
        e = torch.exp(x - m)
        return e / nnef_sum_reduce(x, axes=axes)


def nnef_local_response_normalization(input, size, alpha=1.0, beta=0.5, bias=1.0):
    # type: (torch.Tensor, List[int], float, float, float)->torch.Tensor

    sigma = bias + alpha * nnef_box(torch.pow(input, 2.0), size=size, normalize=True)
    return input / torch.pow(sigma, beta)


def nnef_local_mean_normalization(input, size):
    # type: (torch.Tensor, List[int])->torch.Tensor
    mean = nnef_box(input, size=size, normalize=True)
    return input - mean


def nnef_local_variance_normalization(input, size, bias=0.0, epsilon=0.0):
    # type: (torch.Tensor, List[int], float, float)->torch.Tensor
    sigma = torch.sqrt(nnef_box(torch.pow(input, 2.0), size=size, normalize=True))
    return input / torch.max(sigma + bias,
                             torch.full(size=[], fill_value=epsilon, device=input.device, dtype=input.dtype))


def nnef_local_contrast_normalization(input, size, bias=0.0, epsilon=0.0):
    # type: (torch.Tensor, List[int], float, float)->torch.Tensor
    centered = nnef_local_mean_normalization(input, size=size)
    return nnef_local_variance_normalization(centered, size=size, bias=bias, epsilon=epsilon)


def nnef_l1_normalization(input, axes, bias=0.0, epsilon=0.0):
    # type: (torch.Tensor, List[int], float, float)->torch.Tensor
    sigma = nnef_sum_reduce(torch.abs(input), axes=axes)
    return input / torch.max(sigma + bias,
                             torch.full(size=[], fill_value=epsilon, device=input.device, dtype=input.dtype))


def nnef_l2_normalization(input, axes, bias=0.0, epsilon=0.0):
    # type: (torch.Tensor, List[int], float, float)->torch.Tensor
    sigma = torch.sqrt(nnef_sum_reduce(torch.pow(input, 2.0), axes=axes))
    return input / torch.max(sigma + bias,
                             torch.full(size=[], fill_value=epsilon, device=input.device, dtype=input.dtype))


def nnef_matmul(A, B, transposeA=False, transposeB=False):
    # type:(torch.Tensor, torch.Tensor, bool, bool)->torch.Tensor

    return torch.matmul(torch.transpose(A, len(A.shape) - 2, len(A.shape) - 1) if transposeA else A,
                        torch.transpose(B, len(B.shape) - 2, len(B.shape) - 1) if transposeB else B)


def nnef_split(value, axis, ratios):
    # type:(torch.Tensor, int, List[int])->torch.Tensor
    assert value.shape[axis] % sum(ratios) == 0

    multiplier = value.shape[axis] // sum(ratios)
    sections = [ratio * multiplier for ratio in ratios]
    return torch.split(value, split_size_or_sections=sections, dim=axis)


def nnef_slice(input, axes, begin, end, stride=None):
    # type:(torch.Tensor, List[int], List[int], List[int], List[int])->torch.Tensor

    if stride is None:
        stride = [1] * len(axes)

    shape = list(input.shape)
    slices = [slice(None)] * len(shape)

    for axis, b, e, s in zip(axes, begin, end, stride):
        if b < 0:
            b += shape[axis]
        if e < 0:
            e += shape[axis]
        elif e == 0 and s == 1:
            e = shape[axis]

        b = _clamp(b, -1, shape[axis])
        e = _clamp(e, -1, shape[axis])

        if s > 0:
            slices[axis] = slice(b, e, s)
        else:
            offs = (b - e - 1) % (-s) + 1 if b != e else 1
            slices[axis] = slice(e+offs, b+1, -s)

    input = input[slices]

    flip_axes = [axis for axis, s in zip(axes, stride) if s < 0]
    if len(flip_axes) != 0:
        input = torch.flip(input, dims=flip_axes)

    return input


def nnef_select(condition, true_value, false_value):
    # type:(torch.Tensor, torch.Tensor, torch.Tensor)->torch.Tensor
    rank = max(len(condition.shape), len(true_value.shape), len(false_value.shape))
    return torch.where(_expand_to_rank(condition, rank),
                       _expand_to_rank(true_value, rank),
                       _expand_to_rank(false_value, rank))


def _nnef_generic_reduce(input, axes, f):
    # type:(torch.Tensor, List[int], Callable)->torch.Tensor
    if not axes:
        return input
    for axis in reversed(sorted(axes)):
        input = f(input=input, dim=axis, keepdim=True)
    return input


def nnef_sum_reduce(input, axes, normalize=False):
    # type:(torch.Tensor, List[int], bool)->torch.Tensor
    return _nnef_generic_reduce(input=input, axes=axes, f=torch.mean if normalize else torch.sum)


def nnef_max_reduce(input, axes):
    # type:(torch.Tensor, List[int])->torch.Tensor
    return _nnef_generic_reduce(input=input, axes=axes,
                                f=lambda input, dim, keepdim: torch.max(input, dim=dim, keepdim=keepdim)[0])


def nnef_min_reduce(input, axes):
    # type:(torch.Tensor, List[int])->torch.Tensor
    return _nnef_generic_reduce(input=input, axes=axes,
                                f=lambda input, dim, keepdim: torch.min(input, dim=dim, keepdim=keepdim)[0])


def nnef_mean_reduce(input, axes):
    # type:(torch.Tensor, List[int])->torch.Tensor
    return _nnef_generic_reduce(input=input, axes=axes, f=torch.mean)


def _nnef_argminmax_reduce(input, axes, argmin=False):
    # type:(torch.Tensor, List[int], bool)->torch.Tensor
    if len(axes) == 1:
        return _nnef_generic_reduce(input=input, axes=axes, f=torch.argmin if argmin else torch.argmax)
    else:
        axes = sorted(axes)
        consecutive_axes = list(range(axes[0], axes[0] + len(axes)))

        assert axes == consecutive_axes, \
            "{} is only implemented for consecutive axes.".format("argmin_reduce" if argmin else "argmax_reduce")

        reshaped = nnef_reshape(input,
                                shape=(list(input.shape)[:axes[0]]
                                       + [-1]
                                       + list(input.shape[axes[0] + len(axes):])))
        reduced = _nnef_generic_reduce(input=reshaped, axes=[axes[0]], f=torch.argmin if argmin else torch.argmax)
        reshaped = nnef_reshape(reduced, shape=list(dim if axis not in axes else 1
                                                    for axis, dim in enumerate(input.shape)))
        return reshaped


def nnef_argmax_reduce(input, axes):
    # type:(torch.Tensor, List[int])->torch.Tensor
    return _nnef_argminmax_reduce(input, axes, argmin=False)


def nnef_argmin_reduce(input, axes):
    # type:(torch.Tensor, List[int])->torch.Tensor
    return _nnef_argminmax_reduce(input, axes, argmin=True)


def nnef_clamp(x, a, b):
    # type:(torch.Tensor, torch.Tensor, torch.Tensor)->torch.Tensor
    rank = max(len(x.shape), len(a.shape), len(b.shape))
    x = _expand_to_rank(x, rank)
    a = _expand_to_rank(a, rank)
    b = _expand_to_rank(b, rank)
    return torch.max(torch.min(x, b), a)


def nnef_nearest_downsample(input, factor):
    # type: (torch.Tensor, List[int])->torch.Tensor
    dims = len(input.shape)
    return nnef_box(input, size=[1] * dims, stride=[1, 1] + factor, padding=[(0, 0)] * dims)


def nnef_area_downsample(input, factor):
    # type: (torch.Tensor, List[int])->torch.Tensor
    dims = len(input.shape)
    return nnef_box(input, size=[1, 1] + factor, stride=[1, 1] + factor, padding=[(0, 0)] * dims, normalize=True)


def nnef_moments(input, axes):
    # type: (torch.Tensor, List[int])->Tuple[torch.Tensor, torch.Tensor]
    mean = nnef_mean_reduce(input, axes=axes)
    variance = nnef_mean_reduce(torch.pow(input - mean, 2.0), axes=axes)
    return mean, variance


def nnef_linear(input, filter, bias):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor)->torch.Tensor
    matmul = nnef_matmul(A=input, B=filter, transposeB=True)
    matmul, bias = _expand_binary(matmul, bias)
    return matmul + bias


def nnef_separable_conv(input,  # type: torch.Tensor
                        plane_filter,  # type: torch.Tensor
                        point_filter,  # type: torch.Tensor
                        bias,  # type: torch.Tensor
                        border='constant',  # type: str
                        padding=None,  # type: Optional[List[Tuple[int, int]]]
                        stride=None,  # type: Optional[List[int]]
                        dilation=None,  # type: Optional[List[int]]
                        groups=1,  # type: int
                        ):
    # type: (...)->torch.Tensor
    filtered = nnef_conv(input, plane_filter,
                         bias=torch.zeros(size=tuple(), device=input.device, dtype=input.dtype),
                         border=border,
                         padding=padding,
                         stride=stride,
                         dilation=dilation,
                         groups=0)
    return nnef_conv(filtered, point_filter, bias, groups=groups)


def nnef_separable_deconv(input,  # type: torch.Tensor
                          plane_filter,  # type: torch.Tensor
                          point_filter,  # type: torch.Tensor
                          bias,  # type: torch.Tensor
                          border='constant',  # type: str
                          padding=None,  # type: Optional[List[Tuple[int, int]]]
                          stride=None,  # type: Optional[List[int]]
                          dilation=None,  # type: Optional[List[int]]
                          output_shape=None,  # type: Optional[List[int]]
                          groups=1,  # type: int
                          ):
    # type: (...)->torch.Tensor
    filtered = nnef_deconv(input,
                           point_filter,
                           torch.zeros(size=tuple(), device=input.device, dtype=input.dtype),
                           groups=groups)
    return nnef_deconv(filtered, plane_filter, bias,
                       border=border,
                       padding=padding,
                       stride=stride,
                       dilation=dilation,
                       output_shape=output_shape,
                       groups=0)


def nnef_copy_n(x, times):
    # type: (torch.Tensor, int)->List[torch.Tensor]
    return [x.clone() for _ in range(times)]


def nnef_zero_point_linear_quantize(x, zero_point, scale, bits, signed, symmetric):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, int, bool, bool)->torch.Tensor

    z = torch.round(x / scale) + zero_point
    r = 2 ** (bits - 1) - 1 if signed else 2 ** bits - 1
    q = torch.clamp(z, 0 if not signed else -r if symmetric else -r - 1, r)
    y = (q - zero_point) * scale
    return y.type(x.dtype)


def nnef_min_max_linear_quantize(x, min, max, bits, signed, symmetric):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, int, bool, bool)->torch.Tensor

    r = float(2 ** bits - 1 - int(signed and symmetric))
    z = torch.clamp(x, min, max)
    q = torch.round((z - min) / (max - min) * r)
    return q * ((max - min) / r) + min


def nnef_logarithmic_quantize(x, max, bits):
    # type: (torch.Tensor, torch.Tensor, int)->torch.Tensor

    r = float(2 ** bits - 1)
    m = math.ceil(math.log2(max))
    q = torch.round(torch.clamp(torch.log2(torch.abs(x)), m - r, m))
    return torch.sign(x) * torch.pow(2.0, q)


def nnef_reshape(input, shape, axis_start=0, axis_count=-1):
    # type: (torch.Tensor, List[int], int, int)->torch.Tensor

    return input.reshape(nnef.shapes.reshape_shape(input=list(input.shape),
                                                   shape=shape,
                                                   axis_start=axis_start,
                                                   axis_count=axis_count))


def nnef_update(variable, value):
    # type: (torch.Tensor, torch.Tensor)->torch.Tensor
    return value


def nnef_transpose(input, axes):
    return input.permute(*(axes + list(range(len(axes), len(input.shape)))))


def nnef_squeeze(input, axes):
    return input.reshape(nnef.shapes.squeeze_shape(input.shape, axes))


def nnef_unsqueeze(input, axes):
    return input.reshape(nnef.shapes.unsqueeze_shape(input.shape, axes))


def nnef_cast(input, dtype):
    return input.to(_numpy_dtype_to_torch[dtype])


def nnef_gather(input, indices, axis):
    shape = tuple(indices.shape)
    if len(shape) != 1:
        indices = torch.flatten(indices)
    result = input.index_select(dim=axis, index=indices.to(torch.int64))
    if len(shape) != 1:
        result = torch.reshape(result, shape=input.shape[:axis] + shape + input.shape[axis + 1:])
    return result


"""
The supported operators
"""
Operators = {
    'update': nnef_update,
    'reshape': nnef_reshape,
    'transpose': nnef_transpose,
    'concat': lambda values, axis: torch.cat(values, axis),
    'split': nnef_split,
    'slice': nnef_slice,
    'squeeze': nnef_squeeze,
    'unsqueeze': nnef_unsqueeze,
    'stack': lambda values, axis: torch.stack(values, axis),
    'unstack': lambda value, axis: torch.unbind(value, axis),
    'add': nnef_add,
    'add_n': nnef_add_n,
    'sub': _binary(lambda x, y: x - y),
    'mul': _binary(lambda x, y: x * y),
    'div': _binary(lambda x, y: x / y),
    'pow': _binary(torch.pow),
    'exp': torch.exp,
    'log': torch.log,
    'abs': torch.abs,
    'sign': torch.sign,
    'rcp': torch.reciprocal,
    'neg': torch.neg,
    'copy': torch.clone,
    'lt': _binary(lambda x, y: x < y),
    'gt': _binary(lambda x, y: x > y),
    'le': _binary(lambda x, y: x <= y),
    'ge': _binary(lambda x, y: x >= y),
    'eq': _binary(torch.eq),
    'ne': _binary(torch.ne),
    'and': _binary(lambda x, y: x & y),
    'or': _binary(lambda x, y: x | y),
    'not': lambda x: ~x,
    'floor': torch.floor,
    'ceil': torch.ceil,
    'round': torch.round,
    'select': nnef_select,
    'sqr': lambda x: torch.pow(x, 2.0),
    'sqrt': torch.sqrt,
    'rsqr': lambda x: torch.pow(x, -2.0),
    'rsqrt': torch.rsqrt,
    'log2': torch.log2,
    'min': _binary(torch.min),
    'max': _binary(torch.max),
    'clamp': nnef_clamp,
    'matmul': nnef_matmul,
    'conv': nnef_conv,
    'deconv': nnef_deconv,
    'box': nnef_box,
    'debox': nnef_debox,
    'argmax_pool': nnef_argmax_pool,
    # 'sample': unsupported,
    'desample': nnef_desample,
    'nearest_downsample': nnef_nearest_downsample,
    'area_downsample': nnef_area_downsample,
    'nearest_upsample': nnef_nearest_upsample,
    'multilinear_upsample': nnef_multilinear_upsample,
    'sum_reduce': nnef_sum_reduce,
    'max_reduce': nnef_max_reduce,
    'min_reduce': nnef_min_reduce,
    'argmax_reduce': nnef_argmax_reduce,
    'argmin_reduce': nnef_argmin_reduce,
    'mean_reduce': nnef_mean_reduce,
    'moments': nnef_moments,
    'relu': F.relu,
    'sigmoid': torch.sigmoid,
    'softabs': lambda x, epsilon: torch.sqrt(torch.pow(x, 2.0) + epsilon),
    'softmax': nnef_softmax,
    'softplus': lambda x: torch.log(torch.exp(x) + 1.0),
    'elu': F.elu,
    'selu': lambda x, alpha, _lambda_: F.selu(x),
    'gelu': F.gelu,
    'silu': lambda x: x * torch.sigmoid(x),
    'prelu': lambda x, alpha: F.prelu(x, alpha),
    'leaky_relu': lambda x, alpha: F.leaky_relu(x, alpha),
    'max_pool_with_index': nnef_max_pool_with_index,
    'max_pool': nnef_max_pool,
    'avg_pool': nnef_avg_pool,
    'rms_pool': nnef_rms_pool,
    'linear': nnef_linear,
    'separable_conv': nnef_separable_conv,
    'separable_deconv': nnef_separable_deconv,
    'local_response_normalization': nnef_local_response_normalization,
    'local_mean_normalization': nnef_local_mean_normalization,
    'local_variance_normalization': nnef_local_variance_normalization,
    'local_contrast_normalization': nnef_local_contrast_normalization,
    'l1_normalization': nnef_l1_normalization,
    'l2_normalization': nnef_l2_normalization,
    'batch_normalization': nnef_batch_normalization,
    # 'avg_roi_pool': unsupported,
    # 'max_roi_pool': unsupported,
    # 'roi_resample': unsupported,
    # 'avg_roi_align': unsupported,
    # 'max_roi_align': unsupported,
    'linear_quantize': nnef_min_max_linear_quantize,
    'min_max_linear_quantize': nnef_min_max_linear_quantize,
    'zero_point_linear_quantize': nnef_zero_point_linear_quantize,
    'logarithmic_quantize': nnef_logarithmic_quantize,
    'copy_n': nnef_copy_n,
    'sin': lambda x: torch.sin(x),
    'cos': lambda x: torch.cos(x),
    'tan': lambda x: torch.tan(x),
    'asin': lambda x: torch.asin(x),
    'acos': lambda x: torch.acos(x),
    'atan': lambda x: torch.atan(x),
    'sinh': lambda x: torch.sinh(x),
    'cosh': lambda x: torch.cosh(x),
    'tanh': lambda x: torch.tanh(x),
    'asinh': lambda x: torch.asinh(x),
    'acosh': lambda x: torch.acosh(x),
    'atanh': lambda x: torch.atanh(x),
    'tile': lambda input, repeats: input.repeat(*repeats),
    'pad': nnef_pad,
    'cast': nnef_cast,
    'gather': nnef_gather,
    'any_reduce': lambda input, axes: _nnef_generic_reduce(input, axes=axes, f=torch.any),
    'all_reduce': lambda input, axes: _nnef_generic_reduce(input, axes=axes, f=torch.all),
}
