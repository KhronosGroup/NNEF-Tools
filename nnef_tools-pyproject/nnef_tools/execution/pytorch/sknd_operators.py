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
    def func(lhs, rhs, lhs_align, rhs_align):
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


"""
The supported operators
"""
Operators = {
    'math.add': _binary(lambda x, y: x + y),
    'math.sub': _binary(lambda x, y: x - y),
    'math.mul': _binary(lambda x, y: x * y),
    'math.div': _binary(lambda x, y: x / y),
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
    'layout.tile': layout_tile,
}
