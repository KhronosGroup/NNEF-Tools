import _skriptnd as _sknd
import collections.abc

from .parser import *
from .printer import *
from .binary import *
from .execution import *
import copy
import os


Dtype = _sknd.Dtype
Position = _sknd.Position


Model = _sknd.Model             # dataclass('Model', {
                                #   'name': str,
                                #   'graphs': List[Graph],
                                # }),
Graph = _sknd.Graph             # dataclass('Graph', {
                                #   'name': str,
                                #   'operations': List[Operation],
                                #   'inputs': Tuple[Tensor],
                                #   'outputs': Tuple[Tensor],
                                #   'tensors': List[Tensor],
                                #   'packs': List[TensorPack],
                                #   'asserts': List[Assertion],
                                # })
Tensor = _sknd.Tensor           # dataclass('Tensor', {
                                #   'name': str,
                                #   'dtype': Dtype,
                                #   'shape': Tuple[Expr],
                                #   'max_shape': Tuple[int],
                                #   'quant': Dict[str, object]},
                                #   'value': object,
                                # })
TensorPack = _sknd.TensorPack   # dataclass('TensorPack', {
                                #   'name': str,
                                #   'dtype': Dtype,
                                #   'shape': Tuple[Expr],
                                #   'max_shape': typing.Tuple[int],
                                #   'length': Expr,
                                #   'items': List[Tensor],
                                # })
Operation = _sknd.Operation     # dataclass('Operation', {
                                #   'name': str,
                                #   'dtypes': OrderedDict[str, Dtype],
                                #   'attribs': OrderedDict[str, Expr],
                                #   'inputs': Tuple[Tensor],
                                #   'outputs': Tuple[Tensor],
                                #   'contractions': List[Contraction],
                                #   'asserts': List[Assertion],
                                #   'subexprs': Dict[Expr],
                                # }))

Contraction = _sknd.Contraction # dataclass('Contraction', {
                                #   'left': TensorAccess,
                                #   'right': Expr,
                                #   'assignment': str,
                                #   'locals': List[Tuple[str,Expr]],
                                #   'bounds': Tuple[Tuple[str,Expr]],
                                #   'subscripts': List[int],
                                #   'axes': List[int],
                                # }))

Assertion = _sknd.Assertion     # dataclass('Assertion', {
                                #   'condition': Expr,
                                #   'message': str,
                                #   'args': List[Expr],
                                # }))

# expression sub-types (Expr)
Expr = _sknd.Expr
SizeAccess = _sknd.SizeAccess
ShapeAccess = _sknd.ShapeAccess
TensorAccess = _sknd.TensorAccess
PlaceholderExpr = _sknd.PlaceholderExpr
IdentifierExpr = _sknd.IdentifierExpr
ReferenceExpr = _sknd.ReferenceExpr
UnaryExpr = _sknd.UnaryExpr
BinaryExpr = _sknd.BinaryExpr
SelectExpr = _sknd.SelectExpr
FoldExpr = _sknd.FoldExpr
ListExpr = _sknd.ListExpr
CastExpr = _sknd.CastExpr
BoundedExpr = _sknd.BoundedExpr
ConcatExpr = _sknd.ConcatExpr
SliceExpr = _sknd.SliceExpr
SubscriptExpr = _sknd.SubscriptExpr
UniformExpr = _sknd.UniformExpr
RangeExpr = _sknd.RangeExpr


SizeAccess.dtype = property(lambda access: Dtype.Int)
ShapeAccess.dtype = property(lambda access: Dtype.Int)
SizeAccess.dtype = property(lambda access: Dtype.Int)
TensorAccess.dtype = property(lambda access: access.tensor.dtype)
PlaceholderExpr.dtype = property(lambda access: Dtype.Int)

SizeAccess.size = property(lambda access: None)
TensorAccess.size = property(lambda access: None)
PlaceholderExpr.size = property(lambda expr: None)
IdentifierExpr.size = property(lambda expr: None)
BoundedExpr.size = property(lambda expr: None)
ReferenceExpr.size = property(lambda expr: expr.target.size)
ShapeAccess.size = property(lambda access: access.tensor.length if isinstance(access.tensor, sknd.TensorPack) else None)
CastExpr.size = property(lambda expr: expr.arg.size)
UnaryExpr.size = property(lambda expr: expr.arg.size)
BinaryExpr.size = property(lambda expr: expr.left.size or expr.right.size)
SelectExpr.size = property(lambda expr: expr.cond.size or expr.left.size or expr.right.size)
FoldExpr.size = property(lambda expr: expr.pack.size if expr.packed else None)
ListExpr.size = property(lambda expr: len(expr.items))
SubscriptExpr.size = property(lambda expr: expr.index.size)

SizeAccess.max_size = property(lambda access: None)
TensorAccess.max_size = property(lambda access: None)
PlaceholderExpr.max_size = property(lambda expr: None)
IdentifierExpr.max_size = property(lambda expr: None)
BoundedExpr.max_size = property(lambda expr: None)
ReferenceExpr.max_size = property(lambda expr: expr.target.max_size)
ListExpr.max_size = property(lambda expr: len(expr.items))

SizeAccess.packed = property(lambda access: False)
ShapeAccess.packed = property(lambda access: access.max_size is not None)
TensorAccess.packed = property(lambda access: False)
PlaceholderExpr.packed = property(lambda expr: False)
IdentifierExpr.packed = property(lambda expr: False)
BoundedExpr.packed = property(lambda expr: False)
ReferenceExpr.packed = property(lambda expr: expr.max_size is not None)
CastExpr.packed = property(lambda expr: expr.max_size is not None)
UnaryExpr.packed = property(lambda expr: expr.max_size is not None)
BinaryExpr.packed = property(lambda expr: expr.max_size is not None)
SelectExpr.packed = property(lambda expr: expr.max_size is not None)
FoldExpr.packed = property(lambda expr: expr.max_size is not None)
ListExpr.packed = property(lambda expr: True)
SubscriptExpr.packed = property(lambda expr: expr.max_size is not None)
ConcatExpr.packed = property(lambda expr: True)
SliceExpr.packed = property(lambda expr: True)
UniformExpr.packed = property(lambda expr: True)
RangeExpr.packed = property(lambda expr: True)


def _local_name(name):
    return name[name.rfind('.')+1:]


def _shape_access_str(x):
    s = _local_name(x.tensor.name)
    if x.item is not None:
        s += '[' + str(x.item) + ']'
    s += '.shape'
    if x.dim is not None:
        s += '[' + str(x.dim) + ']'
    return s


def _tensor_access_str(x):
    s = _local_name(x.tensor.name)
    if x.item is not None:
        s += '[' + str(x.item) + ']'
    s += '[' + ','.join(str(i) for i in x.indices) + ']'
    return s


PlaceholderExpr.__str__ = lambda x: '~|' + str(x.max_value)
IdentifierExpr.__str__ = lambda x: _local_name(x.name)
ReferenceExpr.__str__ = lambda x: _local_name(x.name)
SizeAccess.__str__ = lambda x: _local_name(x.pack.name) + '.size'
ShapeAccess.__str__ = _shape_access_str
TensorAccess.__str__ = _tensor_access_str
CastExpr.__str__ = lambda x: f"{x.dtype}({x.arg})"
UnaryExpr.__str__ = lambda x: f"{x.op}({x.arg})"
BinaryExpr.__str__ = lambda x: f"({x.left} {x.op} {x.right})"
SelectExpr.__str__ = lambda x: f"({x.cond} ? {x.left} : {x.right})"
FoldExpr.__str__ = lambda x: f"( {x.pack} {x.op} {'...' if x.packed else '..'} )"
ListExpr.__str__ = lambda x: f"[{', '.join(str(item) for item in x.items)}]"
BoundedExpr.__str__ = lambda x: f"|{x.value} <> {x.lower} : {x.upper}|" if x.lower is not None and x.upper is not None else f"|{x.value}|"
ConcatExpr.__str__ = lambda x: f"[{', '.join(f'{item}..' for item in x.items)}]"
SliceExpr.__str__ = lambda x: f"{x.pack}[{x.first}:{x.last}{':'+str(x.stride) if x.stride != 1 else ''}]"
SubscriptExpr.__str__ = lambda x: f"{x.pack}[{x.index}]"
UniformExpr.__str__ = lambda x: f"[{x.value} ..({x.size})]"
RangeExpr.__str__ = lambda x: f"({x.first}:{x.last}{':'+str(x.stride) if x.stride != 1 else ''})"

Tensor.__hash__ = lambda tensor: hash(tensor.name)
Tensor.is_activation = property(lambda tensor: tensor.value is None and not tensor.variable)
Tensor.is_variable = property(lambda tensor: tensor.variable)
Tensor.is_constant = property(lambda tensor: tensor.value is not None and not tensor.variable)

TensorPack.__len__ = lambda pack: len(pack.items)
TensorPack.__getitem__ = lambda pack, i: pack.items[i]
TensorPack.__iter__ = lambda pack: iter(pack.items)
TensorPack.max_size = property(lambda pack: len(pack.items))
TensorPack.packed = property(lambda expr: True)

Operation.constants = property(lambda op: (tensor for tensor in op.internals if tensor.is_constant))
Operation.variables = property(lambda op: (tensor for tensor in op.internals if tensor.is_variable))

Graph.variables = property(lambda block: (tensor for tensor in block.tensors if tensor.is_variable))
Graph.constants = property(lambda block: (tensor for tensor in block.tensors if tensor.is_constant))
Graph.activations = property(lambda block: (tensor for tensor in block.tensors if tensor.is_activation))
Graph.intermediates = property(lambda block: (tensor for op in block.operations for tensor in _itemize(op.outputs)))

Model.tensors = property(lambda model: (tensor for graph in model.graphs for tensor in graph.tensors))
Model.packs = property(lambda model: (pack for graphs in model.graphs for pack in graphs.packs))
Model.variables = property(lambda model: (tensor for tensor in model.tensors if tensor.is_variable))
Model.constants = property(lambda model: (tensor for tensor in model.tensors if tensor.is_constant))
Model.activations = property(lambda model: (tensor for tensor in model.tensors if tensor.is_activation))


ListExpr.__len__ = lambda expr: len(expr.items)
ListExpr.__getitem__ = lambda expr, i: expr.items[i]
ListExpr.__iter__ = lambda expr: iter(expr.items)
ListExpr.__contains__ = lambda expr, value: value in expr.items
ListExpr.__reversed__ = lambda expr: ListExpr(reversed(expr.items), expr.dtype)

UniformExpr.__len__ = lambda expr: expr.max_size
UniformExpr.__getitem__ = lambda expr, i: expr.value
UniformExpr.__iter__ = lambda expr: (expr.value for _ in range(expr.max_size))
UniformExpr.__contains__ = lambda expr, value: expr.value == value
UniformExpr.__reversed__ = lambda expr: expr

RangeExpr.__len__ = lambda expr: expr.max_size
RangeExpr.__getitem__ = lambda expr, i: expr.first + i * expr.stride
RangeExpr.__iter__ = lambda expr: (expr.first + i * expr.stride for i in range(expr.max_size))

collections.abc.Sequence.register(TensorPack)
collections.abc.Sequence.register(ListExpr)
collections.abc.Sequence.register(UniformExpr)
collections.abc.Sequence.register(RangeExpr)


def expr_dtype(expr):
    if expr is None:
        return sknd.Dtype.Type
    elif isinstance(expr, bool):
        return sknd.Dtype.Bool
    elif isinstance(expr, int):
        return sknd.Dtype.Int
    elif isinstance(expr, float):
        return sknd.Dtype.Real
    elif isinstance(expr, str):
        return sknd.Dtype.Str
    else:
        return expr.dtype


def expr_size(expr):
    return expr.size if isinstance(expr, Expr) else None


def expr_max_size(expr):
    return expr.max_size if isinstance(expr, Expr) else None


def expr_is_packed(expr):
    return expr.packed if isinstance(expr, Expr) else False


def expr_is_dynamic(expr):
    if isinstance(expr, sknd.ListExpr):
        return any(isinstance(item, Expr) for item in expr)
    elif isinstance(expr, sknd.UniformExpr):
        return isinstance(expr.value, Expr) or isinstance(expr.size, Expr)
    elif isinstance(expr, sknd.RangeExpr):
        return not isinstance(expr.size, int)
    else:
        return isinstance(expr, Expr)


def expr_has_dynamic_size(expr):
    return isinstance(expr_size(expr), Expr)


Tensor.is_dynamic = lambda tensor: any(expr_is_dynamic(item) for item in tensor.shape)
Operation.is_dynamic = lambda op: (any(tensor.is_dynamic for tensor in op.inputs) or
                                   any(tensor.is_dynamic for tensor in op.outputs) or
                                   any(expr_is_dynamic(value) for key, value in op.attribs))
Contraction.is_dynamic = lambda contraction: (any(expr_is_dynamic(expr) for iden, expr in contraction.bounds) or
                                              any(expr_is_dynamic(expr) for iden, expr in contraction.locals) or
                                              expr_is_dynamic(contraction.left) or expr_is_dynamic(contraction.right))
Graph.is_dynamic = lambda graph: any(op.is_dynamic for op in graph.operations)
Model.is_dynamic = lambda model: any(graph.is_dynamic for graph in model.graphs)


DtypeToNumpy = {
    Dtype.Type: np.void,
    Dtype.Real: np.float32,
    Dtype.Int: np.int32,
    Dtype.Bool: np.bool_,
    Dtype.Str: np.str_,
}

DtypeFromNumpy = {
    np.float16: Dtype.Real,
    np.float32: Dtype.Real,
    np.float64: Dtype.Real,
    np.int8: Dtype.Int,
    np.uint8: Dtype.Int,
    np.int16: Dtype.Int,
    np.uint16: Dtype.Int,
    np.int32: Dtype.Int,
    np.uint32: Dtype.Int,
    np.int64: Dtype.Int,
    np.uint64: Dtype.Int,
    np.bool_: Dtype.Bool,
    np.str_: Dtype.Str,
    np.void: Dtype.Type,
}


def read_model(path, attribs=None, atomic=None, unroll=None):
    if not os.path.isfile(path) and not os.path.isdir(path):
        raise FileNotFoundError("Path '{}' does not exist".format(path))

    isdir = os.path.isdir(path)
    if isdir and path[-1] != '/' and path[-1] != '\\':
        path += '/'

    model = parse_file(path, attribs=attribs, atomic=atomic, unroll=unroll)
    if model:
        _init_tensor_data(model.tensors, path if isdir else None)

    return model


def scan_model(source, attribs=None, atomic=None, unroll=None):
    model = parse_string(source, attribs=attribs, atomic=atomic, unroll=unroll)
    if model:
        _init_tensor_data(model.tensors, None)

    return model


def write_model(model, path, operators=None, imports=None, inline_subgraphs=False):
    imported = {_split_module_name(op.name) for graph in model.graphs for op in graph.operations}
    if imports:
        imported.update(imports)

    if not os.path.exists(path):
        os.makedirs(path)

    filename = os.path.join(path, 'main.sknd')
    with open(filename, 'w') as file:
        for module in imported:
            if module != 'main':
                print('import {};'.format(module), file=file)

        print('', file=file)

        if operators:
            for op in operators:
                print(op, file=file)
                print('', file=file)

        print_model(model, file, inline_subgraphs=inline_subgraphs, module='main')

    module_scope = 'main.'
    for i, graph in enumerate(model.graphs):
        graph_name = graph.name if graph.name.startswith(module_scope) else module_scope + graph.name
        block_scope = graph_name + '.'
        for tensor in graph.variables:
            name = tensor.name if tensor.name.startswith(block_scope) else block_scope + tensor.name
            variable_path = os.path.join(path, name + '.dat')
            with open(variable_path, 'wb') as variable_file:
                write_tensor(variable_file, tensor.value)


def _init_tensor_data(tensors, path):
    for tensor in tensors:
        if tensor.is_variable:
            if path:
                variable_path = os.path.join(path, tensor.name + '.dat')
                with open(variable_path, 'rb') as variable_file:
                    data = read_tensor(variable_file)

                if data.shape != tensor.shape:
                    raise ValueError('shape {} in variable file does not match shape {} defined in graph structure'
                                     .format(data.shape, tensor.shape))

                tensor.value = data
            else:
                tensor.value = np.full(tensor.shape, dtype=DtypeToNumpy[tensor.dtype],
                                       fill_value='' if tensor.dtype == Dtype.Str else 0, )
        elif tensor.is_constant:
            if isinstance(tensor.value, sknd.ListExpr):
                tensor.value = np.array(tensor.value.items, dtype=DtypeToNumpy[tensor.dtype]).reshape(tensor.shape)


def _split_module_name(op_name):
    idx = op_name.find('.')
    return op_name[:idx] if idx != -1 else 'main'


def _itemize(arg):
    if type(arg) is list or type(arg) is tuple:
        for item in arg:
            yield from _itemize(item)
    else:
        yield arg


def recursive_enumerate_expr(expr, preorder=True, follow_references=False):
    if preorder:
        yield expr

    if isinstance(expr, ReferenceExpr):
        if follow_references:
            yield from recursive_enumerate_expr(expr.target)
    elif isinstance(expr, CastExpr):
        yield from recursive_enumerate_expr(expr.arg)
    elif isinstance(expr, UnaryExpr):
        yield from recursive_enumerate_expr(expr.arg)
    elif isinstance(expr, BinaryExpr):
        yield from recursive_enumerate_expr(expr.left)
        yield from recursive_enumerate_expr(expr.right)
    elif isinstance(expr, SelectExpr):
        yield from recursive_enumerate_expr(expr.cond)
        yield from recursive_enumerate_expr(expr.left)
        yield from recursive_enumerate_expr(expr.right)
    elif isinstance(expr, BoundedExpr):
        yield from recursive_enumerate_expr(expr.arg)
        yield from recursive_enumerate_expr(expr.lower)
        yield from recursive_enumerate_expr(expr.upper)
    elif isinstance(expr, FoldExpr):
        yield from recursive_enumerate_expr(expr.pack)
    elif isinstance(expr, ListExpr):
        for item in expr:
            yield from recursive_enumerate_expr(item)
    elif isinstance(expr, ConcatExpr):
        for item in expr.items:
            yield from recursive_enumerate_expr(item)
        yield from recursive_enumerate_expr(expr.size)
    elif isinstance(expr, SliceExpr):
        yield from recursive_enumerate_expr(expr.pack)
        yield from recursive_enumerate_expr(expr.first)
        yield from recursive_enumerate_expr(expr.last)
        yield from recursive_enumerate_expr(expr.stride)
        yield from recursive_enumerate_expr(expr.size)
    elif isinstance(expr, SubscriptExpr):
        yield from recursive_enumerate_expr(expr.pack)
        yield from recursive_enumerate_expr(expr.index)
    elif isinstance(expr, UniformExpr):
        yield from recursive_enumerate_expr(expr.value)
        yield from recursive_enumerate_expr(expr.size)
    elif isinstance(expr, RangeExpr):
        yield from recursive_enumerate_expr(expr.first)
        yield from recursive_enumerate_expr(expr.last)
        yield from recursive_enumerate_expr(expr.stride)
        yield from recursive_enumerate_expr(expr.size)
    elif isinstance(expr, ShapeAccess):
        yield from recursive_enumerate_expr(expr.item)
    elif isinstance(expr, TensorAccess):
        yield from recursive_enumerate_expr(expr.item)
        for item in expr.indices:
            yield from recursive_enumerate_expr(item)

    if not preorder:
        yield expr


def transform_expr(expr, func, preorder=True):
    if preorder:
        transformed = func(expr)
        if transformed is not None:
            return transformed

    if isinstance(expr, PlaceholderExpr):
        expr = PlaceholderExpr(expr.id, expr.max_value)
    elif isinstance(expr, IdentifierExpr):
        expr = IdentifierExpr(expr.name, expr.kind, expr.dtype)
    elif isinstance(expr, ReferenceExpr):
        expr = ReferenceExpr(expr.name, expr.target, expr.dtype)
    elif isinstance(expr, CastExpr):
        arg = transform_expr(expr.arg, func)
        expr = CastExpr(arg, expr.dtype, expr.max_size)
    elif isinstance(expr, UnaryExpr):
        arg = transform_expr(expr.arg, func)
        expr = UnaryExpr(expr.op, arg, expr.dtype, expr.max_size)
    elif isinstance(expr, BinaryExpr):
        left = transform_expr(expr.left, func)
        right = transform_expr(expr.right, func)
        expr = BinaryExpr(expr.op, left, right, expr.dtype, expr.max_size)
    elif isinstance(expr, SelectExpr):
        cond = transform_expr(expr.cond, func)
        left = transform_expr(expr.left, func)
        right = transform_expr(expr.right, func)
        expr = SelectExpr(cond, left, right, expr.dtype, expr.max_size)
    elif isinstance(expr, BoundedExpr):
        arg = transform_expr(expr.arg, func)
        lower = transform_expr(expr.lower, func)
        upper = transform_expr(expr.upper, func)
        expr = BoundedExpr(arg, lower, upper, expr.dtype)
    elif isinstance(expr, FoldExpr):
        pack = transform_expr(expr.pack, func)
        expr = FoldExpr(expr.op, pack, expr.dtype, expr.max_size)
    elif isinstance(expr, ListExpr):
        items = [transform_expr(item, func) for item in expr]
        expr = ListExpr(items, expr.dtype)
    elif isinstance(expr, ConcatExpr):
        items = [transform_expr(item, func) for item in expr.items]
        size = transform_expr(expr.size, func)
        return ConcatExpr(items, expr.dtype, size, expr.max_size)
    elif isinstance(expr, SliceExpr):
        pack = transform_expr(expr.pack, func)
        first = transform_expr(expr.first, func)
        last = transform_expr(expr.last, func)
        stride = transform_expr(expr.stride, func)
        size = transform_expr(expr.size, func)
        expr = SliceExpr(pack, first, last, stride, expr.dtype, size, expr.max_size)
    elif isinstance(expr, SubscriptExpr):
        pack = transform_expr(expr.pack, func)
        index = transform_expr(expr.index, func)
        expr = SubscriptExpr(pack, index, expr.dtype, expr.max_size)
    elif isinstance(expr, UniformExpr):
        value = transform_expr(expr.value, func)
        size = transform_expr(expr.size, func)
        expr = UniformExpr(value, size, expr.dtype, expr.max_size)
    elif isinstance(expr, RangeExpr):
        first = transform_expr(expr.first, func)
        last = transform_expr(expr.last, func)
        stride = transform_expr(expr.stride, func)
        size = transform_expr(expr.size, func)
        expr = RangeExpr(first, last, stride, expr.dtype, size, expr.max_size)
    elif isinstance(expr, SizeAccess):
        expr = SizeAccess(expr.pack)
    elif isinstance(expr, ShapeAccess):
        item = transform_expr(expr.item, func)
        expr = ShapeAccess(expr.tensor, expr.dim, item, expr.max_size)
    elif isinstance(expr, TensorAccess):
        item = transform_expr(expr.item, func)
        indices = [transform_expr(item, func) for item in expr.indices]
        expr = TensorAccess(expr.tensor, indices, item)
    else:
        expr = copy.copy(expr)

    if not preorder:
        transformed = func(expr)
        if transformed is not None:
            return transformed

    return expr


def collect_index_guards(contraction):
    guards = []
    locals = {name: value for name, value in contraction.locals}
    _collect_index_guards_access(contraction.left, locals, guards)
    _collect_index_guards_expr(contraction.right, locals, guards)
    return guards


def _collect_index_guards_expr(expr, locals, guards):
    for item in sknd.recursive_enumerate_expr(expr):
        if isinstance(item, sknd.TensorAccess):
            _collect_index_guards_access(item, locals, guards)


def _collect_index_guards_access(access, locals, guards):
    for dim, index in enumerate(access.indices):
        if isinstance(index, sknd.IdentifierExpr) and index.name in locals:
            index = locals[index.name]
        if isinstance(index, sknd.BoundedExpr) and index.lower is None and index.upper is None:
            guards.append((access, dim))
