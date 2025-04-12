import _skriptnd as _sknd
from collections.abc import Callable


_StdlibPath = __file__[:-9] + "stdlib/"


def _default_error_callback(position, message, stack, warning):
    print("{} in module '{}' [{}:{}]: {}".format("⚠️Warning" if warning else "🛑Error",
                                                 position.module, position.line, position.column, message))
    for op, pos in reversed(stack):
        print("\t...while calling operator {} in module '{}' [{}:{}]".format(op, pos.module, pos.line, pos.column))


def _resolve_control_flow_attribs(model):
    for graph in model.graphs:
        for op in graph.operations:
            if op.name == 'if':
                op.attribs['cond_graphs'] = [model.graphs[cond] for cond in op.attribs['cond_graphs']]
                op.attribs['branch_graphs'] = [model.graphs[branch] for branch in op.attribs['branch_graphs']]
            elif op.name == 'do':
                op.attribs['body_graph'] = model.graphs[op.attribs['body_graph']]
                condition = op.attribs.get('cond_graph')
                if condition:
                    op.attribs['cond_graph'] = model.graphs[condition]


def _check_operation_callback(obj, param):
    if obj is None:
        return
    elif isinstance(obj, (bool, Callable)):
        return
    elif isinstance(obj, (list, tuple, set)):
        if all(isinstance(item, str) for item in obj):
            return
    elif isinstance(obj, dict):
        if all(isinstance(key, str) and isinstance(value, (bool, Callable)) for key, value in obj.items()):
            return

    raise TypeError(f"Argument '{param}' must be of type bool, list[str], tuple[str], set[str], dict[str,Callable] "
                    f"or Callable")


def parse_file(path, attribs=None, atomic=None, unroll=None,  error=None):
    _check_operation_callback(atomic, 'atomic_callback')
    _check_operation_callback(atomic, 'unroll_callback')

    model = _sknd.parse_file(path, stdlib=_StdlibPath, attribs=attribs or {},
                             error_callback=error or _default_error_callback,
                             atomic_callback=atomic,
                             unroll_callback=unroll)
    if model is not None:
        _resolve_control_flow_attribs(model)
    return model


def parse_string(text, attribs=None, atomic=None, unroll=None,  error=None):
    _check_operation_callback(atomic, 'atomic_callback')
    _check_operation_callback(atomic, 'unroll_callback')

    model = _sknd.parse_string(text, stdlib=_StdlibPath, attribs=attribs or {},
                               error_callback=error or _default_error_callback,
                               atomic_callback=atomic,
                               unroll_callback=unroll)
    if model is not None:
        _resolve_control_flow_attribs(model)
    return model
