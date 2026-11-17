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

import skriptnd as sknd
import builtins
import math
from itertools import repeat


_Unaries = {
    '!': lambda x: not x,
    '+': lambda x: x,
    '-': lambda x: -x,
    'abs': abs,
    'sign': lambda x: 1 if x > 0 else -1 if x < 0 else 0,
    'floor': math.floor,
    'ceil': math.ceil,
    'round': round,
    'frac': lambda x: math.modf(x)[0]
}

_Binaries = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': lambda x, y: x // y if isinstance(x, int) and isinstance(y, int) else x / y,
    '\\': lambda x, y: -(x // -y),
    '%': lambda x, y: x % y,
    '<?': builtins.min,
    '>?': builtins.max,
    '&&': lambda x, y: x and y,
    '||': lambda x, y: x or y,
    '^': lambda x, y: x ^ y,
    '<': lambda x, y: x < y,
    '>': lambda x, y: x > y,
    '<=': lambda x, y: x <= y,
    '>=': lambda x, y: x >= y,
    '==': lambda x, y: x == y,
    '!=': lambda x, y: x != y,
}

_Folds = {
    '+': builtins.sum,
    '*': math.prod,
    '&&': builtins.all,
    '||': builtins.any,
    '<?': builtins.min,
    '>?': builtins.max,
}

_Casts = {
    sknd.Dtype.Bool: lambda x: bool(x),
    sknd.Dtype.Int: lambda x: int(x),
    sknd.Dtype.Real: lambda x: float(x),
}


def eval_expr(expr, tensors):
    if not isinstance(expr, sknd.Expr):
        return expr
    elif isinstance(expr, sknd.ListExpr):
        return [eval_expr(item, tensors) for item in expr]
    elif isinstance(expr, sknd.UniformExpr):
        return [eval_expr(expr.value, tensors)] * eval_expr(expr.size, tensors)
    elif isinstance(expr, sknd.RangeExpr):
        first = eval_expr(expr.first, tensors)
        last = eval_expr(expr.last, tensors)
        stride = eval_expr(expr.stride, tensors)
        return list(range(first, last, stride))
    elif isinstance(expr, sknd.ShapeAccess):
        tensor = tensors[expr.tensor]
        return tensor.shape if expr.dim is None else tensor.shape[expr.dim]
    elif isinstance(expr, sknd.SizeAccess):
        pack = tensors[expr.pack]
        return len(pack)
    elif isinstance(expr, sknd.CastExpr):
        cast = _Casts[expr.dtype]
        arg = eval_expr(expr.arg, tensors)
        return [cast(item) for item in arg] if isinstance(arg, list) else cast(arg)
    elif isinstance(expr, sknd.UnaryExpr):
        op = _Unaries[expr.op]
        arg = eval_expr(expr.arg, tensors)
        return [op(item) for item in arg] if isinstance(arg, list) else op(arg)
    elif isinstance(expr, sknd.BinaryExpr):
        op = _Binaries[expr.op]
        lhs = eval_expr(expr.left, tensors)
        rhs = eval_expr(expr.right, tensors)
        if not expr.packed:
            return op(lhs, rhs)
        elif not isinstance(rhs, list):
            return [op(l, rhs) for l in lhs]
        elif not isinstance(lhs, list):
            return [op(lhs, r) for r in rhs]
        else:
            return [op(l,r) for l, r in zip(lhs, rhs)]
    elif isinstance(expr, sknd.SelectExpr):
        cond = eval_expr(expr.cond, tensors)
        if not isinstance(cond, list):
            return eval_expr(expr.left if cond else expr.right, tensors)
        else:
            lhs = eval_expr(expr.left, tensors)
            if not isinstance(lhs, list):
                lhs = repeat(lhs)
            rhs = eval_expr(expr.right, tensors)
            if not isinstance(rhs, list):
                rhs = repeat(rhs)
            return [l if c else r for c, l, r in zip(cond, lhs, rhs)]
    elif isinstance(expr, sknd.ReferenceExpr):
        return eval_expr(expr.target, tensors)
    elif isinstance(expr, sknd.FoldExpr):
        op = _Folds[expr.op]
        return op(eval_expr(item, tensors) for item in expr.pack)
    elif isinstance(expr, sknd.ConcatExpr):
        items = []
        for item in expr.items:
            value = eval_expr(item, tensors)
            if isinstance(value, list):
                items.extend(value)
            else:
                items.append(value)
        return items
    elif isinstance(expr, sknd.SliceExpr):
        pack = eval_expr(expr.pack, tensors)
        first = eval_expr(expr.first, tensors)
        last = eval_expr(expr.last, tensors)
        stride = eval_expr(expr.stride, tensors)
        return pack[slice(first, last, stride)]
    elif isinstance(expr, sknd.SubscriptExpr):
        pack = eval_expr(expr.pack, tensors)
        index = eval_expr(expr.index, tensors)
        return pack[index]
    else:
        assert False, f"Unhandled shape expr {type(expr)}"
