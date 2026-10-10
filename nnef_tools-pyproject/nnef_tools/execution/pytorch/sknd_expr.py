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


_Unaries = {
    '!': lambda x: not x,
    '+': lambda x: x,
    '-': lambda x: -x,
}

_Binaries = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': lambda x, y: x // y if isinstance(x, int) else x / y,
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
        if expr.item is not None:
            item = eval_expr(expr.item, tensors)
            tensor = tensor[item]
        return tensor.shape if expr.dim is None else tensor.shape[expr.dim]
    elif isinstance(expr, sknd.SizeAccess):
        pack = tensors[expr.pack]
        return len(pack)
    elif isinstance(expr, sknd.CastExpr):
        cast = _Casts[expr.dtype]
        return cast(eval_expr(expr.arg, tensors))
    elif isinstance(expr, sknd.UnaryExpr):
        op = _Unaries[expr.op]
        return op(eval_expr(expr.arg, tensors))
    elif isinstance(expr, sknd.BinaryExpr):
        op = _Binaries[expr.op]
        return op(eval_expr(expr.left, tensors), eval_expr(expr.right, tensors))
    elif isinstance(expr, sknd.SelectExpr):
        cond = eval_expr(expr.cond, tensors)
        return eval_expr(expr.left if cond else expr.right, tensors)
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
