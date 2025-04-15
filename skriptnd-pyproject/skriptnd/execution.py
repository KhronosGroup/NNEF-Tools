from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import skriptnd as sknd
import numpy as np
import itertools
import importlib
import shutil
import math
import os
import re


ModelTemplate = \
"""
#include <limits>
#include <algorithm>
#include <stdexcept>
#include "intrinsics.h"

struct {name}_model
{{
{tensors}

    {name}_model() {init}

    std::vector<sknd::rt::TensorRef> inputs()
    {{
        return {inputs};
    }}
    
    std::vector<sknd::rt::TensorRef> outputs()
    {{
        return {outputs};
    }}

    std::map<const char*,sknd::rt::TensorRef> variables()
    {{
        return {variables};
    }}
    
    {checks}
    
    {blocks}
}};
"""


WrapperSource = \
"""
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "runtime.h"
using sknd::rt::str;


#include HEADER_NAME


py::object make_namedtuple( const std::string& name, std::initializer_list<std::string> fields )
{
    py::object collections = py::module_::import("collections");
    py::object namedtuple = collections.attr("namedtuple");
    py::list items(fields.size());
    size_t i = 0;
    for ( auto& field : fields )
    {
        items[i++] = py::str(field);
    }
    return namedtuple(py::str(name), items);
}

py::object TensorInfo = make_namedtuple("TensorInfo", { "shape", "dtype" });


py::dtype py_dtype( sknd::rt::Dtype dtype )
{
    static const py::dtype py_dtypes[] =
    {
        py::dtype::of<sknd::real_t>(),
        py::dtype::of<sknd::int_t>(),
        py::dtype::of<sknd::bool_t>(),
    };
    return py_dtypes[(int)dtype];
}

std::string str( const py::dtype& dtype )
{
    return std::string(1, dtype.char_());
}

void fill_tensor( sknd::rt::TensorView& tensor, const py::array& array )
{
    auto dtype = py_dtype(tensor.dtype());
    if ( !array.dtype().is(dtype) )
    {
        throw std::invalid_argument("unexpected dtype '" + str(array.dtype()) + "'; expected dtype '" + str(dtype) + "'");
    }
    
    tensor.reshape(array.ndim(), array.shape());
    
    std::copy_n((char*)array.data(), array.nbytes(), tensor.bytes());
}

py::array fetch_tensor( const sknd::rt::TensorView& tensor )
{
    std::vector<py::ssize_t> shape(tensor.rank());
    std::copy_n(tensor.shape(), tensor.rank(), shape.data());
    
    py::array array = py::array(py_dtype(tensor.dtype()), shape);
    std::copy_n(tensor.bytes(), array.nbytes(), (char*)array.data());
    
    return array;
}

void load( MODEL_TYPE& model, const std::map<std::string,py::object>& variables )
{
    for ( auto& [name, value] : model.variables() )
    {
        auto it = variables.find(name);
        if ( it == variables.end() )
        {
            throw std::invalid_argument("could not fetch variable '" + std::string(name) + "'");
        }
        
        if ( value.is<sknd::rt::TensorPackView*>() )
        {
            auto& items = *value.as<sknd::rt::TensorPackView*>();
            auto& list = (const py::list&)it->second;
            
            for ( int k = 0; k < items.size(); ++k )
            {
                fill_tensor(items[k], (const py::array&)list[k]);             
            }
        }
        else
        {
            auto& tensor = *value.as<sknd::rt::TensorView*>();
            fill_tensor(tensor, (const py::array&)it->second);
        }
    }
}

py::tuple execute( MODEL_TYPE& model, py::args args )
{
    auto inputs = model.inputs();
    if ( args.size() != inputs.size() )
    {
        throw std::length_error("expected " + std::to_string(inputs.size()) + " argument(s); got " + std::to_string(args.size()));
    }
    
    for ( size_t i = 0; i < args.size(); ++i )
    {
        auto& input = inputs[i];
        if ( input.is<sknd::rt::TensorPackView*>() )
        {
            auto& items = *input.as<sknd::rt::TensorPackView*>();
            auto& arg = (const py::list&)args[i];
            
            items.resize(arg.size());
            for ( int k = 0; k < items.size(); ++k )
            {
                fill_tensor(items[k], (const py::array&)arg[k]);
            }
        }
        else
        {
            auto& item = *input.as<sknd::rt::TensorView*>();
            auto& arg = (const py::array&)args[i];
            fill_tensor(item, arg);
        }
    }
    
    model.check();
    model.execute();
    
    auto outputs = model.outputs();
    
    py::tuple results(outputs.size());
    for ( size_t i = 0; i < outputs.size(); ++i )
    {
        auto& output = outputs[i];
        if ( output.is<sknd::rt::TensorPackView*>() )
        {
            auto& items = *output.as<sknd::rt::TensorPackView*>();
            auto result = py::list(items.size());
            for ( int k = 0; k < items.size(); ++k )
            {
                result[k] = fetch_tensor(items[k]);
            }
            results[i] = result;
        }
        else
        {
            auto& item = *output.as<sknd::rt::TensorView*>();
            results[i] = fetch_tensor(item);
        }
    }
    
    return results;
}

py::tuple make_shape( const size_t rank, const int* extents )
{
    py::tuple shape(rank);
    for ( size_t i = 0; i < rank; ++i )
    {
        shape[i] = py::int_(extents[i]);
    }
    return shape;
}

py::tuple tensor_info( const std::vector<sknd::rt::TensorRef>& views )
{
    py::tuple infos(views.size());
    for ( size_t i = 0; i < views.size(); ++i )
    {
        auto& view = views[i];
        if ( view.is<sknd::rt::TensorPackView*>() )
        {
            auto& pack = *view.as<sknd::rt::TensorPackView*>();
            py::list items(pack.size());
            for ( int k = 0; k < pack.size(); ++k )
            {
                items[k] = TensorInfo(make_shape(pack[i].rank(), pack[i].shape()), py_dtype(pack[i].dtype()));
            }
            infos[i] = items;
        }
        else
        {
            auto& item = *view.as<sknd::rt::TensorView*>();
            infos[i] = TensorInfo(make_shape(item.rank(), item.shape()), py_dtype(item.dtype()));
        }
    }
    return infos;
}

py::tuple input_info( MODEL_TYPE& model )
{
    return tensor_info(model.inputs());
}

py::tuple output_info( MODEL_TYPE& model )
{
    return tensor_info(model.outputs());
}


PYBIND11_MODULE(MODULE_NAME, m) {
    py::class_<MODEL_TYPE>(m, "Model")
        .def(py::init())
        .def("load", &load)
        .def("input_info", &input_info)
        .def("output_info", &output_info)
        .def("__call__", &execute);
}
"""


def _recursive_itemize(arg):
    if type(arg) is list or type(arg) is tuple or type(arg) is sknd.TensorPack:
        for item in arg:
            yield from _recursive_itemize(item)
    else:
        yield arg


def _can_inline_tensor(tensor):
    return tensor.value is not None and tensor.shape == () and not isinstance(tensor.value, (list, tuple, np.ndarray))


def _ensure_positive_axis(axis, rank):
    return axis + rank if axis < 0 else axis


def _format_tensor_access(access, idx=None):
    packed = isinstance(access.tensor, sknd.TensorPack)
    if not packed and _can_inline_tensor(access.tensor):
        return _format_value_expr(access.tensor.value)

    tensor = access.tensor[idx] if idx is not None else access.tensor
    name = _valid_id(tensor.name)
    subscript = "[" + _format_value_expr(access.item) + "]" if access.item is not None else ""
    indices = "(" + ", ".join(_format_value_expr(index, extent=tensor.shape[dim])
                              for dim, index in enumerate(access.indices)) + ")"

    return name + subscript + indices


def _has_tensor_access(expr):
    return any(isinstance(x, sknd.TensorAccess) for x in sknd.recursive_enumerate_expr(expr))


def _format_nested_loops(contraction, indent):
    tuple_assign = isinstance(contraction.left.tensor, sknd.TensorPack) and contraction.left.item is None

    if isinstance(contraction.right, (int, float, bool)) and len(contraction.locals) == 0:
        value = _format_value_expr(contraction.right)
        if tuple_assign:
            return "\n".join(indent + f"{_valid_id(tensor.name)} = {value};"
                             for tensor in contraction.left.tensor.items)
        else:
            tensor = _valid_id(contraction.left.tensor.name)
            return indent + f"{tensor} = {value};"

    loops = "".join(indent + k * "\t" + "for ( int {i} = 0; {i} < {n}; ++{i} )\n".format(i=id, n=_format_value_expr(bound))
                    for k, (id, bound) in enumerate(contraction.bounds))

    indent += len(contraction.bounds) * "\t"

    index_guards = sknd.collect_index_guards(contraction)

    if index_guards:
        index_locals = list()
        value_locals = list()
        for local in contraction.locals:
            id, expr = local
            if _has_tensor_access(expr):
                value_locals.append(local)
            else:
                index_locals.append(local)
    else:
        index_locals = []
        value_locals = contraction.locals

    if index_locals:
        loops += indent[:-1] + "{\n"

    for id, expr in index_locals:
        expr = _format_value_expr(expr, bracket=False)
        loops += indent + f"auto {id} = {expr};\n"

    if index_guards:
        loops += indent + "if ( {} )\n".format(" && ".join(_format_guard(access.indices[dim], access.tensor, dim)
                                                           for access, dim in index_guards))
        indent += "\t"

    if value_locals:
        loops += indent[:-1] + "{\n"

    for id, expr in value_locals:
        expr = _format_value_expr(expr, bracket=False)
        loops += indent + f"auto {id} = {expr};\n"

    if contraction.condition:
        loops += indent + "if ( {} )\n".format(_format_value_expr(contraction.condition))
        indent += "\t"

    if tuple_assign:
        loops += indent[:-1] + "{\n"
        for i in range(contraction.left.tensor.max_size):
            loops += _format_contraction_assignment(contraction, indent, i)
        loops += indent[:-1] + "}\n"
    else:
        loops += _format_contraction_assignment(contraction, indent)

    if contraction.condition:
        indent = indent[:-1]

    if value_locals:
        loops += indent[:-1] + "}\n"

    if index_locals:
        loops += indent[:-2] + "}\n"

    return loops


def _format_contraction_assignment(contraction, indent, idx=None):
    lhs = _format_tensor_access(contraction.left, idx)
    rhs = _format_value_expr(contraction.right[idx] if idx is not None else contraction.right, bracket=False)

    if contraction.assignment == '>=':
        return indent + f"{lhs} = std::max({lhs}, {rhs});\n"
    elif contraction.assignment == '<=':
        return indent + f"{lhs} = std::min({lhs}, {rhs});\n"
    elif contraction.assignment == '>!' or contraction.assignment == '<!':
        axis = contraction.axes[0]
        idx = contraction.bounds[axis][0]
        op = contraction.assignment[0]
        return indent + f"if ( {rhs} {op} ${lhs} ) {{ {lhs} = {idx}; ${lhs} = {rhs}; }}\n"
    else:
        op = '=' if contraction.assignment == ':=' else contraction.assignment
        return indent + f"{lhs} {op} {rhs};\n"


def _format_guard(index, tensor, dim):
    lhs = _format_value_expr(index)
    rhs = f"(unsigned){_valid_id(tensor.name)}.shape({dim})" if isinstance(tensor.shape[dim], sknd.Expr) \
        else _format_value_expr(tensor.shape[dim])
    return f"(unsigned)({lhs}) < {rhs}"


def _format_dtype(dtype):
    return f"sknd::{dtype.name.lower()}_t"


def _format_decl_type(tensor):
    rank = len(tensor.shape)
    dtype = _format_dtype(tensor.dtype)
    if isinstance(tensor, sknd.TensorPack):
        return f"sknd::rt::TensorPack<{rank},{dtype},{tensor.max_size}>"
    else:
        return f"sknd::rt::Tensor<{rank},{dtype}>"


def _format_dynamic_mask(tensor):
    return '0b' + ''.join('1' if isinstance(s, sknd.Expr) else '0' for s in reversed(tensor.shape))


def _format_tensor_declaration(tensor, indent):
    type = _format_decl_type(tensor)
    name = _valid_id(tensor.name)
    shape = ", ".join(str(s) for s in tensor.max_shape)
    if len(tensor.shape):
        shape = _format_dynamic_mask(tensor) + ", " + shape
    return indent + f"{type} {name}{{{shape}}};"


def _format_pack_population(pack, indent):
    name = _valid_id(pack.name)
    items = ", ".join(_valid_id(item.name) for item in pack)
    return indent + f"{name}.populate({items});\n"


def _format_pack_shape_update(pack, indent):
    name = _valid_id(pack.name)
    return indent + f"{name}.update_shape();\n"


def _format_const_initializer(tensor, indent):
    name = _valid_id(tensor.name)
    if isinstance(tensor.value, np.ndarray):
        values = ", ".join(_format_value_expr(value.item()) for value in tensor.value.flat)
        return indent + f"{name} = {{ {values} }};\n"
    else:
        value = _format_value_expr(tensor.value)
        return indent + f"{name} = {value};\n"


def _format_tensor_initializers(model, indent, context):
    deferred_packs = context['deferred_packs']
    const_inits = "".join(_format_const_initializer(tensor, indent) for tensor in context['tensors']
                          if tensor.is_constant)
    pack_inits = "".join(_format_pack_population(pack, indent) for pack in context['packs']
                         if pack.name not in deferred_packs)
    separator = "\n" if len(const_inits) and len(pack_inits) else ""

    return const_inits + separator + pack_inits


def _format_uniform(arg, max_size):
    return "sknd::rt::uniform<{max_size}>({arg})".format(arg=arg, max_size=max_size)


def _format_value_expr(expr, bracket=True, extent=None):
    lb = "(" if bracket else ""
    rb = ")" if bracket else ""
    if expr is None:
        return "std::nullopt"
    elif isinstance(expr, bool):
        return "true" if expr else "false"
    elif isinstance(expr, int):
        return str(expr)
    elif isinstance(expr, float):
        if expr == math.inf:
            return "std::numeric_limits<sknd::real_t>::infinity()"
        elif expr == -math.inf:
            return "-std::numeric_limits<sknd::real_t>::infinity()"
        return f"{expr}f"
    elif isinstance(expr, str):
        return f'"{expr}"'
    elif isinstance(expr, sknd.IdentifierExpr):
        return _valid_id(expr.name)
    elif isinstance(expr, sknd.ReferenceExpr):
        return _valid_id(expr.name)
    elif isinstance(expr, sknd.SizeAccess):
        iden = _valid_id(expr.pack.name)
        return f"{iden}.size()"
    elif isinstance(expr, sknd.ShapeAccess):
        name = _valid_id(expr.tensor.name)
        dim = _format_value_expr(expr.dim)
        if expr.item is not None:
            item = _format_value_expr(expr.item)
            return f"{name}[{item}].shape({dim})"
        elif isinstance(expr.tensor, sknd.TensorPack) and expr.packed:
            return f"{name}.shapes({dim})"
        else:
            return f"{name}.shape({dim})"
    elif isinstance(expr, sknd.TensorAccess):
        return _format_tensor_access(expr)
    elif isinstance(expr, sknd.CastExpr):
        if not expr.packed:
            if expr.dtype == sknd.Dtype.Int:
                if expr.arg == float('inf'):
                    return "std::numeric_limits<sknd::int_t>::max()"
                elif expr.arg == float('-inf'):
                    return "std::numeric_limits<sknd::int_t>::min()"
            arg = _format_value_expr(expr.arg)
            dtype = _format_dtype(expr.dtype)
            return f"({dtype}){arg}"
        else:
            arg = _format_value_expr(expr.arg, bracket=False)
            dtype = expr.dtype.name.lower()
            return f"sknd::rt::unary<sknd::rt::to_{dtype}>({arg})"
    elif isinstance(expr, sknd.UnaryExpr):
        if not expr.packed:
            if len(expr.op) < 3:
                return expr.op + _format_value_expr(expr.arg)
            else:
                ns = "sknd" if expr.op == "sign" or expr.op == "frac" else "std"
                arg = _format_value_expr(expr.arg, bracket=False)
                return f"{ns}::{expr.op}({arg})"
        else:
            if expr.op == "+":
                op = "std::plus"
            elif expr.op == "!":
                op = "std::logical_not"
            else:
                op = expr.op

            arg = _format_value_expr(expr.arg, bracket=False)
            return "sknd::rt::unary<{op}>({arg})".format(op=op, arg=arg)
    elif isinstance(expr, sknd.BinaryExpr):
        if not expr.packed:
            left = _format_value_expr(expr.left)
            right = _format_value_expr(expr.right)
            if expr.op == "**":
                return f"std::pow({left}, {right})"
            elif expr.op == "%" and sknd.expr_dtype(expr) == sknd.Dtype.Real:
                return f"std::fmod({left}, {right})"
            elif expr.op == "<<":
                return f"std::min({left}, {right})"
            elif expr.op == ">>":
                return f"std::max({left}, {right})"
            elif expr.op == "->":
                return f"!{left} || {right}"
            elif expr.op == "\\":
                return f"sknd::rt::ceil_div({left}, {right})"
            else:
                return lb + f"{left} {expr.op} {right}" + rb
        else:
            if expr.op == "+":
                op = "std::plus"
            elif expr.op == "-":
                op = "std::minus"
            elif expr.op == "*":
                op = "std::multiplies"
            elif expr.op == "/":
                op = "std::divides"
            elif expr.op == "\\":
                op = "sknd::rt::ceil_divides"
            elif expr.op == "%":
                op = "std::modulus"
            elif expr.op == "<<":
                op = "sknd::rt::minimize"
            elif expr.op == ">>":
                op = "sknd::rt::maximize"
            elif expr.op == "&&":
                op = "std::logical_and"
            elif expr.op == "||":
                op = "std::logical_or"
            elif expr.op == "==":
                op = "std::equal_to"
            elif expr.op == "!=":
                op = "std::not_equal_to"
            elif expr.op == "<":
                op = "std::less"
            elif expr.op == ">":
                op = "std::greater"
            elif expr.op == "<=":
                op = "std::less_equal"
            elif expr.op == ">=":
                op = "std::greater_equal"
            else:
                op = expr.op

            left = _format_value_expr(expr.left, bracket=False)
            if not sknd.expr_is_packed(expr.left):
                left = _format_uniform(left, expr.max_size)
            right = _format_value_expr(expr.right, bracket=False)
            if not sknd.expr_is_packed(expr.right):
                right = _format_uniform(right, expr.max_size)
            return f"sknd::rt::binary<{op}>({left}, {right})"
    elif isinstance(expr, sknd.SelectExpr):
        cond = _format_value_expr(expr.cond, bracket=False)
        left = _format_value_expr(expr.left, bracket=False)
        right = _format_value_expr(expr.right, bracket=False)
        if not expr.packed or isinstance(expr.cond, bool):
            return lb + cond + " ? " + left + " : " + right + rb
        else:
            if not sknd.expr_is_packed(expr.left):
                left = _format_uniform(left, expr.max_size)
            if not sknd.expr_is_packed(expr.right):
                right = _format_uniform(right, expr.max_size)
            return f"sknd::rt::select({cond}, {left}, {right})"
    elif isinstance(expr, sknd.FoldExpr):
        if expr.op == "+":
            op = "std::plus"
        elif expr.op == "*":
            op = "std::multiplies"
        elif expr.op == "<<":
            op = "sknd::rt::minimize"
        elif expr.op == ">>":
            op = "sknd::rt::maximize"
        elif expr.op == "||":
            op = "std::logical_or"
        elif expr.op == "&&":
            op = "std::logical_and"
        else:
            raise TypeError("Invalid fold operator: " + str(expr.op))

        arg = _format_value_expr(expr.pack)
        if not expr.packed:
            return f"sknd::rt::reduce<{op}>({arg})"
        else:
            return f"sknd::rt::accum<{op}>({arg})"
    elif isinstance(expr, sknd.BoundedExpr):
        arg = _format_value_expr(expr.arg, bracket=False)
        if expr.lower is not None and expr.upper is not None:
            lower = _format_value_expr(expr.lower, bracket=False)
            upper = _format_value_expr(expr.upper, bracket=False)
            return arg + " < 0 ? " + lower + " : " + arg + " >= " + str(extent) + " ? " + upper + " : " + arg
        else:
            return arg
    elif isinstance(expr, sknd.ListExpr):
        return "sknd::rt::list({})".format(", ".join(_format_value_expr(item, bracket=False) for item in expr))
    elif isinstance(expr, sknd.ConcatExpr):
        return "sknd::rt::concat({})".format(", ".join(_format_value_expr(item, bracket=False) for item in expr.items))
    elif isinstance(expr, sknd.SliceExpr):
        pack = _format_value_expr(expr.pack)
        first = _format_value_expr(expr.first)
        last = _format_value_expr(expr.last)
        if expr.stride == 1:
            return f"sknd::rt::slice({pack}, {first}, {last})"
        else:
            stride = _format_value_expr(expr.stride)
            return f"sknd::rt::slice({pack}, {first}, {last}, {stride})"
    elif isinstance(expr, sknd.SubscriptExpr):
        pack = _format_value_expr(expr.pack)
        index = _format_value_expr(expr.index)
        return f"{pack}[{index}]"
    elif isinstance(expr, sknd.UniformExpr):
        value = _format_value_expr(expr.value)
        size = _format_value_expr(expr.size)
        return f"sknd::rt::uniform<{expr.max_size}>({value}, {size})"
    elif isinstance(expr, sknd.RangeExpr):
        first = _format_value_expr(expr.first)
        last = _format_value_expr(expr.last)
        if expr.stride == 1:
            return f"sknd::rt::range<{expr.max_size}>({first}, {last})"
        else:
            stride = _format_value_expr(expr.stride)
            return f"sknd::rt::range<{expr.max_size}>({first}, {last}, {stride})"
    else:
        raise TypeError("Invalid value expression: " + str(type(expr)))


def _format_value_exprs(expr, bracket=True):
    if isinstance(expr, list):
        return "{" + ", ".join(_format_value_expr(item, bracket=False) for item in expr) + "}"
    else:
        return _format_value_expr(expr, bracket=bracket)


def _format_shape_propagation(output, indent):
    text = ""
    if isinstance(output, sknd.TensorPack) and _is_dynamic_size(output.size) and not _is_placeholder(output.size):
        length = _format_value_expr(output.size, bracket=False)
        text += indent + f"{_valid_id(output.name)}.resize({length});\n"
    if _is_dynamic_shape(output.shape) and all(not _is_placeholder(item) for item in output.shape):
        shape = ", ".join(_format_value_expr(expr, bracket=False) if not sknd.expr_is_packed(expr) else "-1"
                          for expr in output.shape)
        text += indent + f"{_valid_id(output.name)}.reshape({shape});\n"
    if isinstance(output, sknd.TensorPack):
        text += "".join(_format_shape_propagation(item, indent) for item in output)
    return text


def _is_dynamic_size(expr):
    return not isinstance(expr, int)


def _is_dynamic_shape(shape):
    return any(_is_dynamic_size(item) for item in shape)


def _is_placeholder(expr):
    return isinstance(expr, sknd.PlaceholderExpr)


def _format_packable_tensor_in_comment(tensor):
    if tensor is None:
        return "~"
    elif not isinstance(tensor, sknd.TensorPack) and _can_inline_tensor(tensor):
        return _format_value_in_comment(tensor.value)
    else:
        return tensor.name


def _format_attrib_in_comment(attrib):
    if isinstance(attrib, sknd.ListExpr):
        return "[" + ",".join(_format_attrib_in_comment(item) for item in attrib) + "]"
    elif isinstance(attrib, sknd.UniformExpr):
        return _format_attrib_in_comment(attrib.value)
    elif isinstance(attrib, sknd.Graph):
        return attrib.name
    elif isinstance(attrib, list) and all(isinstance(item, sknd.Graph) for item in attrib):
        return "[" + ",".join(item.name for item in attrib) + "]"
    else:
        return _format_value_expr(attrib)


def _format_value_in_comment(value):
    if isinstance(value, sknd.ListExpr):
        return "[" + ",".join(_format_value_in_comment(item) for item in value) + "]"
    elif isinstance(value, sknd.UniformExpr):
        return _format_value_in_comment(value.value)
    elif isinstance(value, bool):
        return "true" if value else "false"
    else:
        return str(value)


def _format_invocation_in_comment(name, attribs, inputs, outputs):
    text = ", ".join(_format_packable_tensor_in_comment(output) for output in outputs)
    text += " = "
    text += name
    if attribs:
        values = ", ".join(k + "=" + _format_attrib_in_comment(v) for k, v in attribs.items() if v is not None)
        if values:
            text += "{" + values + "}"
    text += "(" + ", ".join(_format_packable_tensor_in_comment(input) for input in inputs) + ")"
    return text


def _format_operation(op, indent, context):
    deferred_packs = context['deferred_packs']

    text = "".join(_format_pack_shape_update(input, indent) for input in op.inputs
                   if isinstance(input, sknd.TensorPack) and _is_dynamic_shape(input.shape))

    text += indent + "// " + _format_invocation_in_comment(op.name, op.attribs, op.inputs, op.outputs) + "\n"

    for iden, expr in op.subexprs.items():
        iden = _valid_id(iden)
        size = sknd.expr_max_size(expr)
        dtype = _format_dtype(sknd.expr_dtype(expr))
        value = _format_value_expr(expr, bracket=False) if not isinstance(expr, sknd.ListExpr) \
            else "{ " + ", ".join(_format_value_expr(item, bracket=False) for item in expr) + " }"
        if size is not None:
            text += indent + f"const sknd::rt::ValuePack<{dtype},{size}> {iden} = {value};\n"
        else:
            text += indent + f"const {dtype} {iden} = {value};\n"

    text += "".join(_format_pack_population(output, indent) for output in op.outputs
                    if output.name in deferred_packs)
    text += "".join(_format_shape_propagation(output, indent) for output in op.outputs)
    text += _format_contractions(op, indent) if len(op.contractions) else _format_intrinsic(op, indent, context)

    return text


def _format_contractions(op, indent):
    return "\n".join(_format_nested_loops(contraction, indent) for contraction in op.contractions)


def _format_execution_code(operations, indent, context):
    return "\n".join(_format_operation(op, indent, context) for op in operations)


def _is_trivial_block(block):
    return len(block.operations) == 1 and block.operations[0].name == ''


def _format_block_params(inputs, outputs, context):
    input_params = ", ".join("const {type}& {name}".format(type=_format_decl_type(input),
                                                           name=_valid_id(input.name))
                             for input in inputs)
    output_params = ", ".join("{type}& {name}".format(type=_format_decl_type(output),
                                                      name=_valid_id(output.name))
                              for output in outputs)
    return input_params + ", " + output_params if len(input_params) and len(output_params) \
        else input_params if len(input_params) else output_params if len(output_params) else ""


def _format_checks_code(graphs, indent, context):
    main = graphs[0]

    placeholders = {}
    for arg, input in enumerate(main.inputs):
        for dim, s in enumerate(input.shape):
            if isinstance(s, sknd.PlaceholderExpr):
                if s.id not in placeholders:
                    placeholders[s.id] = []
                placeholders[s.id].append((arg, dim))
        if isinstance(input, sknd.TensorPack) and isinstance(input.size, sknd.PlaceholderExpr):
            if input.size.id not in placeholders:
                placeholders[input.size.id] = []
            placeholders[input.size.id].append((arg, None))

    code = "".join(_format_placeholder_check_code(items, main.inputs, indent)
                   for items in placeholders.values() if len(items) > 1)
    code += "".join(_format_assert_check_code(assertion, indent)
                    for assertion in main.asserts)

    return f"void check()\n\t{{\n{code}\t}}"


def _format_placeholder_check_code(items, inputs, indent):
    def _shape_or_size_cond(name, dim):
        iden = _valid_id(name)
        return f"{iden}.shape({dim})" if dim is not None else f"{iden}.size()"

    def _shape_or_size_msg(arg, dim):
        return f"dimension {dim} of argument {arg}" if dim is not None else f"size of argument {arg}"

    arg0, dim0 = items[0]
    cond = " || ".join(f"{_shape_or_size_cond(inputs[arg0].name, dim0)} != {_shape_or_size_cond(inputs[arg].name, dim)}"
                       for arg, dim in items[1:])
    msg = '"' + " and ".join(_shape_or_size_msg(arg, dim) for arg, dim in items) + ' must match"'
    return indent + f"if ( {cond} )\n" + indent + f"\tthrow std::invalid_argument({msg});\n"


def _format_assert_check_code(assertion, indent):
    condition = _format_value_expr(assertion.condition)
    message = assertion.message.replace("{}", "%s")
    args = ", ".join(_format_value_expr(arg) for arg in assertion.args)
    return (indent + f"if ( {condition} )\n" + indent +
            f"\tthrow std::runtime_error(sknd::string_format(\"{message}\", {args}));\n")


def _format_block_code(block, idx, indent, context, condition):
    type = "bool" if condition else "void"
    name = _valid_id(block.name) if idx else "execute"
    params = _format_block_params(block.inputs, block.outputs, context) if idx else ""
    code = _format_execution_code(block.operations, indent, context)
    if condition:
        output = block.outputs[0]
        params += " = sknd::rt::condition_result<{rank}>()".format(rank=len(output.shape))
        code += "\n" + indent + "return " + _valid_id(output.name) + "(" + ",".join("0" for _ in output.shape) + ");"
    return "{type} {name}( {params} ) {code}".format(type=type,
                                                     name=name,
                                                     params=params,
                                                     code=_wrap_brackets(code))


def _format_blocks_code(graphs, indent, context):
    cond_graphs = set()
    body_graphs = set()
    for graph in graphs:
        for op in graph.operations:
            if op.name == 'if':
                for subgraph in op.attribs['cond_graphs']:
                    cond_graphs.add(subgraph.name)
                for subgraph in op.attribs['branch_graphs']:
                    body_graphs.add(subgraph.name)
            elif op.name == 'do':
                subgraph = op.attribs.get('cond_graph')
                if subgraph:
                    cond_graphs.add(subgraph.name)
                subgraph = op.attribs.get('body_graph')
                body_graphs.add(subgraph.name)

    return "\n\n\t".join(_format_block_code(block, i, indent, context, block.name in cond_graphs)
                         for i, block in enumerate(graphs)
                         if not (block.name in cond_graphs and block.name not in body_graphs and _is_trivial_block(block)))


def _format_intrinsic(op, indent, context):
    if op.name == 'layout.reshape':
        return indent + "std::copy_n({input}.data(), {input}.volume(), {output}.data());".format(
            input=_valid_id(op.inputs[0].name),
            output=_valid_id(op.outputs[0].name))
    elif op.name == 'if':
        return _format_if(op, indent)
    elif op.name == 'do':
        return _format_do(op, indent, context)
    elif op.name == '':
        return _format_copy(op, indent)
    elif op.name == 'layout.nonzero':
        return _format_nonzero(op, indent)
    elif op.name == 'algo.top_k':
        return _format_topk(op, indent)
    elif op.name == 'algo.nonmax_suppress':
        return _format_nms(op, indent)
    else:
        raise ValueError("Unhandled intrinsic operation '{}'".format(op.name))


def _format_call(block, args, is_condition=False):
    if is_condition and _is_trivial_block(block):
        cond = args[0]
        iden = _valid_id(cond.name)
        return iden + "({})".format(",".join("0" for _ in cond.shape))
    else:
        return _valid_id(block.name) + "({})".format(", ".join(_valid_id(arg.name) + '[$]' if isinstance(arg, sknd.TensorPack) else
                                                     _format_value_expr(arg.value) if _can_inline_tensor(arg) else
                                                     _valid_id(arg.name) for arg in args))


def _format_copy(op, indent):
    lhs = _valid_id(op.outputs[0].name)
    if len(op.inputs):
        arg = op.inputs[0]
        rhs = _format_value_expr(arg.value) if _can_inline_tensor(arg) else _valid_id(arg.name)
    else:
        rhs = _format_value_expr(op.attribs[""])
    return indent + f"{lhs} = {rhs};"


def _format_if(op, indent):
    conditions = op.attribs['cond_graphs']
    branches = op.attribs['branch_graphs']
    cond_input_indices = op.attribs['cond_inputs']
    branch_input_indices = op.attribs['branch_inputs']
    cond_input_offset = 0
    branch_input_offset = 0

    text = indent
    for condition, branch in zip(conditions, branches):
        cond_inputs = tuple(op.inputs[idx] for idx in cond_input_indices[cond_input_offset:cond_input_offset+len(condition.inputs)])
        branch_inputs = tuple(op.inputs[idx] for idx in branch_input_indices[branch_input_offset:branch_input_offset+len(branch.inputs)])
        text += "if ( {cond} ) {branch}; else ".format(cond=_format_call(condition, cond_inputs, is_condition=True),
                                                       branch=_format_call(branch, branch_inputs + op.outputs))
        cond_input_offset += len(condition.inputs)
        branch_input_offset += len(branch.inputs)

    branch_inputs = tuple(op.inputs[idx] for idx in branch_input_indices[branch_input_offset:])
    text += _format_call(branches[-1], branch_inputs + op.outputs) + ";"
    return text


def _format_tensor_ref(tensor, braces=False):
    return _format_value_expr(tensor.value) if _can_inline_tensor(tensor) else _valid_id(tensor.name) + ("()" if braces else "")


def _format_do(op, indent, context):
    condition = op.attribs.get('cond_graph')
    pretest = op.attribs.get('pretest')
    body = op.attribs['body_graph']
    nvars = op.attribs['nvars']
    nscans = op.attribs['nscans']
    static_iters = op.attribs.get('iters')
    dynamic_iters = op.inputs[nvars+nscans]

    auxiliaries = context['auxiliaries']
    index = sknd.Tensor(name='$', dtype=sknd.Dtype.Int, shape=(), max_shape=())
    vars = tuple(auxiliaries[output] for output in op.outputs[:nvars])
    subgraph_inputs = vars + op.inputs[nvars:nvars+nscans] + (index,) + op.inputs[nvars+nscans+1:]
    body_inputs = tuple(subgraph_inputs[idx] for idx in op.attribs['body_inputs'])

    text = ""

    for i in range(nvars):
        text += indent + "{lhs} = {rhs};\n".format(lhs=_format_tensor_ref(op.outputs[i]), rhs=_format_tensor_ref(op.inputs[i]))

    if condition:
        subgraph_inputs = op.outputs[:nvars] + op.inputs[nvars:nvars+nscans] + (index,) + op.inputs[nvars+nscans+1:]
        cond_inputs = tuple(subgraph_inputs[idx] for idx in op.attribs['cond_inputs'])
        cond_text = (indent + "\tif ( !{cond} ) break;\n"
                     .format(cond=_format_call(condition, cond_inputs, is_condition=True)))

    text += indent + "for ( int $ = 0; {bound}; ++$ )\n".format(
        bound=("$ < " + _format_tensor_ref(dynamic_iters, braces=True)) if dynamic_iters else
              ("$ < " + _format_value_expr(static_iters)) if static_iters else "")

    text += indent + "{\n"

    if condition and pretest:
        text += cond_text

    for i in range(nvars):
        text += indent + "\tstd::swap({lhs}, {rhs});\n".format(lhs=_valid_id(vars[i].name),
                                                               rhs=_valid_id(op.outputs[i].name))

    text += indent + "\t" + _format_call(body, body_inputs + op.outputs) + ";\n"

    if condition or dynamic_iters:
        for i in range(nvars, len(op.outputs)):
            text += indent + "\t" + _valid_id(op.outputs[i].name) + ".resize($+1);\n"

    if condition and not pretest:
        text += cond_text

    text += indent + "}"

    return text


def _format_nonzero(op, indent):
    return indent + "sknd::rt::nonzero({input}, {indices});".format(
        input=_valid_id(op.inputs[0].name),
        indices=_valid_id(op.outputs[0].name),
    )


def _format_topk(op, indent):
    return indent + "sknd::rt::top_k({input}, {values}, {indices}, {k}, {axis}, {largest}, {sorted});".format(
        input=_valid_id(op.inputs[0].name),
        values=_valid_id(op.outputs[0].name),
        indices=_valid_id(op.outputs[1].name),
        k=_format_value_expr(op.attribs['k'], bracket=False),
        axis=_format_value_expr(_ensure_positive_axis(op.attribs['axis'], len(op.inputs[0].shape))),
        largest=_format_value_expr(op.attribs['largest']),
        sorted=_format_value_expr(op.attribs['sorted']),
    )


def _format_nms(op, indent):
    return indent + "sknd::rt::bbox_nms({boxes}, {scores}, {indices}, {box_format_centered}, {max_outputs_per_class}, " \
                    "{iou_threshold}, {score_threshold});".format(
        boxes=_valid_id(op.inputs[0].name),
        scores=_valid_id(op.inputs[1].name),
        indices=_valid_id(op.outputs[0].name),
        box_format_centered=_format_value_expr(op.attribs['box_format'] == "CENTER"),
        max_outputs_per_class=_format_value_expr(op.attribs['max_outputs_per_class'] or 0),
        iou_threshold=_format_value_expr(op.attribs['iou_threshold']),
        score_threshold=_format_value_expr(op.attribs['score_threshold']),
    )


def _format_tensor_declarations(model, indent, context):
    auxiliaries = {contraction.left.tensor: _make_auxiliary_tensor(contraction.left.tensor, sknd.expr_dtype(contraction.right))
                   for op in model.graphs[0].operations for contraction in op.contractions
                   if contraction.assignment == '<!' or contraction.assignment == '>!'}

    subgraph_io = {tensor.name for graph in model.graphs[1:] if len(graph.operations)
                   for tensor in itertools.chain(graph.inputs, graph.outputs)}

    for block in model.graphs:
        for op in block.operations:
            if op.name == 'do':
                nvars = op.attribs['nvars']
                auxiliaries.update({tensor: _make_auxiliary_tensor(tensor) for tensor in op.outputs[:nvars]})

    shaped_tensors = [tensor for graph in reversed(model.graphs) for tensor in graph.tensors]
    shaped_tensors.extend(auxiliaries.values())

    declared_tensors = [tensor for tensor in model.tensors
                        if tensor.name not in subgraph_io and not _can_inline_tensor(tensor)]
    declared_packs = [pack for pack in model.packs if pack.name not in subgraph_io]
    deferred_packs = {pack.name for pack in declared_packs if any(item.name in subgraph_io for item in pack)}

    text = "\n".join(_format_tensor_declaration(tensor, indent)
                     for tensor in itertools.chain(declared_tensors, auxiliaries.values(), declared_packs))

    context['auxiliaries'] = auxiliaries
    context['tensors'] = declared_tensors
    context['packs'] = declared_packs
    context['deferred_packs'] = deferred_packs

    return text


def _format_tensor_views(tensors):
    return "\n".join(f"\t\t\t&{_valid_id(tensor.name) }," for tensor in tensors)


def _format_variable_views(tensors):
    return "\n".join(f"\t\t\t{{ \"{tensor.name}\", &{_valid_id(tensor.name)} }},"
                     for tensor in tensors if isinstance(tensor.value, np.ndarray))


def _wrap_brackets(text, inner=False):
    if len(text) == 0:
        return "{}"

    if inner:
        return "{\n" + text + "\n\t\t}"
    else:
        return "\n\t{\n" + text + "\n\t}"


_id_pattern = re.compile('[^~_$0-9a-zA-Z]')


def _valid_id(name):
    return _id_pattern.sub('$', name)


def _make_auxiliary_tensor(tensor, dtype=None):
    return sknd.Tensor(name='$'+tensor.name, dtype=dtype or tensor.dtype, shape=tensor.shape, max_shape=tensor.max_shape)


def _generate_model_source(model):
    context = {}
    inputs = model.graphs[0].inputs
    outputs = model.graphs[0].outputs
    return ModelTemplate.format(name=_valid_id(model.name),
                                tensors=_format_tensor_declarations(model, indent='\t', context=context),
                                init=_wrap_brackets(_format_tensor_initializers(model, indent='\t\t', context=context)),
                                checks=_format_checks_code(model.graphs, indent='\t\t', context=context),
                                blocks=_format_blocks_code(model.graphs, indent='\t\t', context=context),
                                inputs=_wrap_brackets(_format_tensor_views(inputs), inner=True),
                                outputs=_wrap_brackets(_format_tensor_views(outputs), inner=True),
                                variables=_wrap_brackets(_format_variable_views(model.variables), inner=True))


def _save_to_file(text, filename):
    with open(filename, "w") as file:
        file.write(text)
        file.close()


def _normalize_dtype(array):
    if array.dtype == np.int64:
        return array.astype(np.int32)
    if array.dtype == np.float64:
        return array.astype(np.float32)
    return array


def compile_model(model, keep_generated_code=False):
    model_name = _valid_id(model.name)
    model_fn = model_name.replace('$', '_')
    hdr_name = model_fn + "_model.h"
    cpp_name = model_fn + "_pybind.cpp"
    module_name = model_fn + "_module"

    _save_to_file(_generate_model_source(model), hdr_name)
    _save_to_file(WrapperSource, cpp_name)

    base = os.path.normpath(os.path.join(__file__, '../cpp/include'))

    ext_modules = [
        Pybind11Extension(module_name,
                          [cpp_name],
                          include_dirs=[base, os.path.join(base, 'core'), os.path.join(base, 'runtime')],
                          define_macros=[
                              ('MODEL_TYPE', model_name + '_model'),
                              ('MODULE_NAME', module_name),
                              ('HEADER_NAME', '"' + hdr_name + '"'),
                          ],
                          ),
    ]

    import distutils.command.build

    parent_dir = os.path.normpath(os.path.join(__file__, '../'))
    build_dir = os.path.join(parent_dir, 'build_' + model_fn)

    # Override build command
    class BuildCommand(distutils.command.build.build):
        def initialize_options(self):
            distutils.command.build.build.initialize_options(self)
            self.build_base = build_dir

    setup(
        name=module_name,
        version="1.0",
        ext_modules=ext_modules,
        packages=[],
        cmdclass={"build_ext": build_ext, "build": BuildCommand},
        zip_safe=False,
        python_requires=">=3.6",
        script_name='setup.py',
        script_args=['--quiet', 'develop'],
        include_dirs=["skriptnd/cpp/include", "skriptnd/cpp/include/core",
                      "skriptnd/cpp/include/frontend", "skriptnd/cpp/include/composer", "skriptnd/cpp/include/runtime"],
    )

    os.remove(cpp_name)
    if not keep_generated_code:
        os.remove(hdr_name)

    shutil.rmtree(build_dir)
    if os.path.exists(module_name + '.egg-info'):
        shutil.rmtree(module_name + '.egg-info')

    import site
    importlib.reload(site)
    module = importlib.import_module(module_name, package=".")
    os.remove(module.__file__)

    compiled_model = module.Model()
    compiled_model.load({tensor.name: _normalize_dtype(tensor.value) for tensor in model.variables})
    return compiled_model
