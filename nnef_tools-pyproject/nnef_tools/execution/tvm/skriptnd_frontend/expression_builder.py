import math
from typing import Dict, Any
from dataclasses import dataclass, field

import numpy as np

import skriptnd as sknd
from tvm import te, relax, tir, ir

from . import convert_dtype

class DynamicError(Exception):
    pass

@dataclass
class Environment:
    """
    General environment for Expression Converter's additional data

    Main use is for TIR operation level environment, but can be used for Relax as well


    buffers: dict[str, tensor object] - TIR.Buffer or Relax.Expr, mapped to their ts name
    backward_map: dict[ts.Expr, str] - mapping from ts.Expr.__repr__() to ts name to handle some exprs
    it_vars: dict[str, tir.Var] - iteration variables

    """

    buffers: Dict[str, tir.Buffer | relax.Var] = field(default_factory=dict)
    backward_map: Dict[sknd.Expr, str] = field(default_factory=dict)
    it_vars: Dict[str, tir.Var] = field(default_factory=dict)
    it_itvars: Dict[str, tir.IterVar] = field(default_factory=dict)
    vars: Dict[str, Any] = field(default_factory=dict)

    curr_access: sknd.TensorAccess = None
    curr_idx: int = None
    curr_conds: list = field(default_factory=list)

    call_conds: list = field(default_factory=list)
    itvar_to_var = False

    def __getitem__(self, item):
        # to get the tir iteration Vars instead the IterVars use `instance.it_vars`
        if item in self.buffers:
            return self.buffers[item]
        if item in self.it_itvars:
            return self.it_itvars[item]
        if item in self.vars:
            return self.vars[item]

        raise ValueError(f"Item {item} not found in environment")

    def __contains__(self, item):
        return item in self.buffers or item in self.it_itvars or item in self.vars

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    # context manager to set current buffer and index for index expr calculation
    def __call__(self, tens, idx):
        self.curr_access = tens
        self.curr_idx = idx
        # self.call_conds = []
        return self

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.curr_access = None
        self.curr_idx = None
        self.curr_conds = []
        pass

    def reset(self):
        # reset Environment between contractions without loosing buffers
        self.it_vars = {}
        self.it_itvars = {}
        self.vars = {}
        self.curr_access = None
        self.curr_idx = None
        self.curr_conds = []
        self.call_conds = []


class ExprBuilder:
    """
    Convert SkriptND expressions to TVM expressions, using the given namespace (te for tir or relax.op for relax)
    """

    def __init__(self, env: Environment = None, namespace=te):
        """
        :param env: Environment object to store tensors and variables
        :param namespace: `te` or `relax.op` for the namespace to use
        """
        if namespace not in (te, relax.op):
            raise ValueError(f"Unknown namespace: {namespace}")
        self.env = env
        self.ns = namespace

    def __call__(self, expr: sknd.Expr):
        return self._expr_root(expr)

    def get_call_indices(self, access: sknd.TensorAccess):
        assert self.ns is te or access.indices == [], "Non-scalar tensor indexing is only supported by TE/TIR"

        indices = []
        conds = []
        for i, idx_expr in enumerate(access.indices):
            with self.env(access, i):
                try:
                    # bounded expr needs the access that was called for shape info
                    indices.append(self._expr_root(idx_expr))
                except Exception as e:
                    raise ValueError(f"Error in index {i} of tensor access {access} at [{idx_expr}]:\n{e}")
                conds.extend(self.env.curr_conds)

        return indices, conds

    def _expr_root(self, expr: sknd.Expr):
        if not isinstance(expr, sknd.Expr):
            if isinstance(expr, (bool, str)) or expr is None:
                return expr
            # py float inf is float64, but only float 32 is supported
            if math.isinf(expr):
                return tir.const(expr, "float32")
            return expr

        return {
            sknd.SizeAccess: self._size_access,
            sknd.ShapeAccess: self._shape_access,
            sknd.TensorAccess: self._tensor_access,
            sknd.IdentifierExpr: self._identifier_expr,
            sknd.ReferenceExpr: self._reference_expr,
            sknd.UnaryExpr: self._unary_expr,
            sknd.BinaryExpr: self._binary_expr,
            sknd.SelectExpr: self._select_expr,
            sknd.FoldExpr: self._fold_expr,
            sknd.ListExpr: self._list_exprs,
            sknd.CastExpr: self._cast_expr,
            sknd.BoundedExpr: self._bounded_expr,
            sknd.ConcatExpr: self._concat_expr,
            sknd.SliceExpr: self._slice_expr,
            sknd.SubscriptExpr: self._subscript_expr,
            sknd.UniformExpr: self._uniform_expr,
            sknd.RangeExpr: self._range_expr,
        }[type(expr)](expr)

    def _size_access(self, expr: sknd.SizeAccess):
        obj = expr.pack
        return len(obj)

    def _shape_access(self, expr: sknd.ShapeAccess):
        if self.ns == te:
            tens = self.env[expr.tensor.name]  # env.buffers[expr.tensor.name]
            shape = tens.shape

            return shape[expr.dim]
        else:
            # relax
            tens = self.env[expr.tensor.name]
            shape = tens.struct_info.shape
            return shape[expr.dim]

    def _tensor_access(self, expr: sknd.TensorAccess):
        # assert self.env.indices is not None
        if isinstance(expr.tensor, sknd.Tensor):
            # tens = self.env.instance_tensors[expr.tensor.name]
            tens = self.env[expr.tensor.name]
        elif isinstance(expr.tensor, sknd.TensorPack):
            raise DynamicError("TensorPack can't be indexed, the operation is dynamic")

        # in case of relax:
        if isinstance(tens, relax.Expr):
            if expr.indices == []:
                return tens

            raise ValueError("Relax tensor access with indices not implemented")

        indices, conds = self.get_call_indices(expr)
        self.env.call_conds.extend(conds)

        return tens[tuple(indices)]

    def _identifier_expr(self, expr: sknd.IdentifierExpr):
        elem = self.env[expr.name]
        if isinstance(elem, tir.IterVar):
            return elem.var
        return elem

    def _reference_expr(self, expr: sknd.ReferenceExpr):
        raise ValueError("ReferenceExpr not implemented")

    def _unary_expr(self, expr: sknd.UnaryExpr):
        op_dict = {
            "+": lambda x: x,
            "-": lambda x: -x,
            "!": tir.bitwise_not if self.ns == te else lambda x: not x,
            "sqrt": self.ns.sqrt,
            "exp": self.ns.exp,
            "log": self.ns.log,
            "sin": self.ns.sin,
            "cos": self.ns.cos,
            "tan": self.ns.tan,
            "asin": self.ns.asin,
            "acos": self.ns.acos,
            "atan": self.ns.atan,
            "sinh": self.ns.sinh,
            "cosh": self.ns.cosh,
            "tanh": self.ns.tanh,
            "asinh": self.ns.asinh,
            "acosh": self.ns.acosh,
            "atanh": self.ns.atanh,
            "round": self.ns.round,
            "floor": self.ns.floor,
            "ceil": self.ns.ceil,
            "abs": self.ns.abs,
            "frac": lambda x: x - self.ns.floor(x),  # tir.fmod(x, tir.const(1, "float32")),
            "erf": self.ns.erf,
        }
        if self.ns == te:
            op_dict["sign"] = lambda x: tir.if_then_else(x > 0, 1, tir.if_then_else(x < 0, -1, 0))
        else:
            op_dict["sign"] = relax.op.sign

        return op_dict[expr.op](self._expr_root(expr.arg))

    def _binary_expr(self, expr: sknd.BinaryExpr):
        dtype = expr.dtype
        op_dict = {
            "+": self.ns.add,
            "-": self.ns.subtract,
            "*": self.ns.multiply,
            "/": lambda x, y: x / y,
            "\\": lambda x, y: (x + y - 1) / y if dtype == sknd.Dtype.Int else self.ns.ceil(x / y),
            "**": self.ns.power,
            "<": lambda x, y: x < y,
            "<=": lambda x, y: x <= y,
            ">": lambda x, y: x > y,
            ">=": lambda x, y: x >= y,
            "==": lambda x, y: (x == y).asobject() if self.ns == te else relax.op.equal(x, y),
            "!=": lambda x, y: (x != y).asobject() if self.ns == te else relax.op.not_equal(x, y),
            "<<": lambda x, y: tir.if_then_else(x < y, x, y) if self.ns == te else relax.op.minimum(x, y),
            ">>": lambda x, y: tir.if_then_else(x > y, x, y) if self.ns == te else relax.op.maximum(x, y),
            "%": (tir.fmod if dtype == sknd.Dtype.Real else lambda x, y: x % y) if self.ns == te else relax.op.mod,
            "&&": tir.all if self.ns == te else relax.op.logical_and,
            "||": tir.any if self.ns == te else relax.op.logical_or,
            "^": lambda x, y: x ^ y,
            "=>": lambda x, y: not x or y,
        }
        return op_dict[expr.op](self._expr_root(expr.left),
                                self._expr_root(expr.right))

    def _select_expr(self, expr: sknd.SelectExpr):
        assert self.ns == te, "SelectExpr is only supported by TE/TIR"

        cond = self._expr_root(expr.cond)
        true = self._expr_root(expr.left)
        false = self._expr_root(expr.right)

        return self.ns.if_then_else(cond, true, false)

    def _fold_expr(self, expr: sknd.FoldExpr):
        raise ValueError("FoldExpr not implemented")

    def _list_exprs(self, expr: sknd.ListExpr):
        return [self._expr_root(e) for e in expr.items]

    def _cast_expr(self, expr: sknd.CastExpr):
        dtype = convert_dtype(expr.dtype)
        return self._expr_root(expr.arg).astype(dtype)

    def _bounded_expr(self, expr: sknd.BoundedExpr):
        raw_idx = self._expr_root(expr.arg)
        shape_limit = self.env.curr_access.tensor.shape[self.env.curr_idx]

        if expr.lower is not None or expr.upper is not None:
            assert expr.lower is not None and expr.upper is not None, "Both lower and upper bounds must be set"
            idx = tir.Max(self._expr_root(expr.lower), tir.Min(self._expr_root(expr.upper), raw_idx))
            # idx = tir.if_then_else(raw_idx < 0,
            #                              self._expr_root(expr.lower),
            #                              tir.if_then_else(raw_idx >= shape_limit,
            #                                               self._expr_root(expr.upper),
            #                                               raw_idx))
            cond = True  # tir.all(raw_idx >= 0, raw_idx < shape_limit) # TODO condition works?
        else:
            # idx = tir.all(raw_idx < shape_limit)
            idx = raw_idx
            cond = tir.reinterpret("uint32", raw_idx) < shape_limit

        self.env.curr_conds.append(cond)
        return idx

    def _concat_expr(self, expr: sknd.ConcatExpr):
        raise ValueError("ConcatExpr not implemented")

    def _slice_expr(self, expr: sknd.SliceExpr):
        raise ValueError("SliceExpr not implemented")

    def _subscript_expr(self, expr: sknd.SubscriptExpr):
        # todo check for tuples (tensor pack as attr??? or sth)
        if isinstance(expr.pack, sknd.ListExpr):
            name = self.env.backward_map[expr.__repr__()]
            tens = self.env.buffers[name]
            return tens[self._expr_root(expr.index)]

        raise ValueError("SubscriptExpr not implemented")

    def _uniform_expr(self, expr: sknd.UniformExpr):
        if expr.max_size != expr.size:
            if self.ns == relax.op:
                return self.ns.full((self._expr_root(expr.size),), relax.const(expr.value))
                relax.op.full()
            raise ValueError("UniformExpr not implemented with dynamic size")

            # max_size = self._expr_root(expr.max_size)
        dtype = convert_dtype(expr.dtype)
        if isinstance(expr.value, sknd.Expr):
            return self._expr_root(expr.value)

        return np.full(expr.size, expr.value,
                       dtype=dtype).tolist()  # easier to calc with numpy, but not recognized in tvm


    def _range_expr(self, expr: sknd.RangeExpr):
        raise ValueError("RangeExpr not implemented")
