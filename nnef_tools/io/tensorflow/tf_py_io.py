# Copyright (c) 2017 The Khronos Group Inc.
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

import importlib
import inspect
import os
import sys
import traceback
import typing
from collections import OrderedDict

import numpy as np
import six
import tensorflow as tf

from nnef_tools.conversion import conversion_info
from nnef_tools.core import graph_utils
from nnef_tools.core import utils
from nnef_tools.io.tensorflow.tf_graph import *
from nnef_tools.io.tensorflow.tf_py import tf_py_shape_inference, tf_py_eval
from nnef_tools.io.tensorflow.tf_py.tf_py_definitions import *


def trace(network_function,  # type: typing.Callable[[], typing.Any]
          checkpoint_path=None,  # type: typing.Optional[str]
          raise_on_missing_weight=True,  # type: bool
          expand_gradients=False,  # type: bool
          custom_traceable_functions=None  # type: typing.Optional[typing.List[TraceableFunction]]
          ):
    # type: (...)->TFGraph

    if custom_traceable_functions is None:
        custom_traceable_functions = []

    traceable_functions = DefaultTraceableFunctions + custom_traceable_functions
    for trf in traceable_functions:
        trf.eval_functions()

    functions_by_name = {trf.op_proto.op_name: trf.functions for trf in traceable_functions}
    if expand_gradients:
        del functions_by_name["tf.gradients"]

    tracer = _InvocationTracer(functions_by_name)

    result = tracer.trace(network_function, allow_nesting=expand_gradients)
    result = _eliminate_named_tuples(result)
    invocations = tracer.invocations

    for invocation in invocations:
        invocation.args = _eliminate_named_tuples(invocation.args)
        invocation.result = _eliminate_named_tuples(invocation.result)
    invocations = _eliminate_identities(invocations)

    # _print_invocations(invocations)
    if expand_gradients:
        _fix_strange_grad_functions(invocations)
        invocations = _eliminate_nesting(invocations)
        # _print_invocations(invocations)

    assert not _check_has_untraced_ops(invocations, result), \
        "There were untraced operations. " \
        "Add the untraced operations to custom_functions_to_trace or use only supported operations."

    op_proto_by_name = {trf.op_proto.op_name: trf.op_proto for trf in traceable_functions}
    tf_graph = _to_tf_graph(network_function.__name__, invocations, result, op_proto_by_name)
    graph_utils.remove_unreachable(tf_graph)

    if checkpoint_path:
        checkpoint_reader = tf.contrib.framework.load_checkpoint(checkpoint_path)
        for tensor in tf_graph.list_variables():
            assert tensor.name.endswith("/read:0"), "Strange variable name: {}".format(tensor.name)
            var_name = tensor.name[:-len("/read:0")]
            if checkpoint_reader.has_tensor(var_name):
                tensor.data = checkpoint_reader.get_tensor(var_name)
                if not isinstance(tensor.data, np.ndarray):
                    tensor.data = np.array(tensor.data).reshape(tensor.shape)
            elif raise_on_missing_weight:
                assert False, "Checkpoint {} does not have var {}".format(checkpoint_path, var_name)

    return tf_graph


def write(tf_graph,  # type: TFGraph
          file_path,  # type: str
          write_weights=True,  # type: bool
          custom_op_protos=None,  # type: typing.Optional[typing.List[OpProto]]
          custom_imports=None  # type: str
          ):
    # type: (...) -> typing.Optional[conversion_info.ConversionInfo]

    generate_source_operations(tf_graph)
    tf_graph.sort()
    try:

        names_to_write = _generate_names(tf_graph=tf_graph,
                                         custom_imports=custom_imports,
                                         custom_op_protos=custom_op_protos)

        old_names = {}
        for tensor in tf_graph.tensors:
            old_names[tensor] = tensor.name
            tensor.name = utils.anystr_to_str(names_to_write[tensor])

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        with open(file_path, "w") as f:
            _print(tf_graph, file_handle=f, custom_op_protos=custom_op_protos, custom_imports=custom_imports)

        with open(file_path, "r") as f:
            tf_source = f.read()

        if tf_graph.list_variables() and write_weights:
            checkpoint_dir = file_path + ".checkpoint"
            checkpoint_path = os.path.join(checkpoint_dir, os.path.basename(file_path) + ".ckpt")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            _create_checkpoint_with_values(
                net_fun=_tfsource_to_function(tf_source, tf_graph.name),
                file_name=checkpoint_path,
                variable_value_by_name={t.name: t.data for t in tf_graph.tensors if t.is_variable and t.name})

        for tensor in tf_graph.tensors:
            tensor.name = old_names[tensor]

        return _get_rename_info(tf_graph, names_to_write)
    finally:
        remove_source_operations(tf_graph)


def _tfsource_to_function(src, function_name):
    globals = {}
    exec(src, globals)
    return globals[function_name]


def _create_checkpoint_with_values(net_fun, file_name, variable_value_by_name):
    assert all(value.size != 0 for value in six.itervalues(variable_value_by_name)), \
        "Can not export weights when we have dummy weights. Did you read your graph with weights?"

    def get_variables():
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    tf.reset_default_graph()
    tf.set_random_seed(0)

    net_fun()
    variables = get_variables()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        assigns = []
        for variable in variables:
            value = variable_value_by_name[variable.name]  # dtype: np.ndarray
            assigns.append(tf.assign(variable, value))
        sess.run(assigns)
        saver.save(sess, os.path.relpath(file_name))


def _format_tensor_name(name):
    # type:(str)->str
    name = "t_" + name
    if name.endswith(':0'):
        name = name[:-2]
    else:
        name = name.replace(':', '_')
    return name.replace('/', '_').replace('.', '_')


def _format_result_names(result):
    # type: (typing.Any)->str
    return ", ".join(_format_tensor_name(r.name) for r in result)


def _transform_arg(arg):
    if isinstance(arg, TFTensor):
        if arg.is_constant and arg.rank == 0:
            return arg.data[0]
        return _format_tensor_name(arg.name)
    elif isinstance(arg, str):
        return "'{}'".format(arg)
    elif isinstance(arg, tf.DType):
        return "tf." + arg.name
    elif isinstance(arg, np.ndarray):
        return arg.tolist()
    elif isinstance(arg, float):
        s = "{:.12f}".format(arg)
        if "." in s:
            s = s.rstrip('0')
            if s[-1] == '.':
                return s + '0'
            else:
                return s
        return s
    else:
        return arg


def _format_rec(arg):
    if isinstance(arg, (list, tuple)):
        parts = []
        for a in arg:
            parts.append(_format_rec(a))
        s = ", ".join(parts)
        if isinstance(arg, list):
            s = "[" + s + "]"
        if isinstance(arg, tuple):
            s = "(" + s + ")"
    else:
        s = str(arg)
    return s


def _format_args(args):
    parts = []
    for k, v in args.items():
        arg = utils.recursive_transform(v, _transform_arg)
        if k == "dtype":  # TODO less hack
            arg_str = arg if arg is None else "tf.{}".format(arg[1:-1])
        else:
            arg_str = _format_rec(arg)
        parts.append("{}={}".format(k, arg_str))
    return ", ".join(parts)


def _print(tf_graph, file_handle, custom_op_protos=None, custom_imports=None, with_name_dict=False):
    # type: (TFGraph, typing.TextIO, typing.Optional[typing.List[OpProto]], str, bool)->None

    op_proto_by_name = {trf.op_proto.op_name: trf.op_proto for trf in DefaultTraceableFunctions}
    if custom_op_protos:
        op_proto_by_name.update({op_proto.op_name: op_proto for op_proto in custom_op_protos})

    printed_tensors = set()  # we need this because 0d constants are not printed as tensors
    f = file_handle
    print("from __future__ import division, print_function, absolute_import", file=f)
    print("from collections import OrderedDict", file=f)
    print("import tensorflow as tf", file=f)
    if any(op.name.startswith("_tf.") for op in tf_graph.operations):
        print("from nnef_tools.io.tensorflow.tf_py.tf_py_compat import tf_internal as _tf", file=f)
    if custom_imports:
        print(custom_imports, file=f)
    print(file=f)
    print(file=f)
    assert utils.is_identifier(tf_graph.name), "Graph name '{}' is not an identifier".format(tf_graph.name)

    print("def {}():".format(tf_graph.name), file=f)
    indent = " " * 4
    for op in list(tf_graph.operations):
        assert op.name in op_proto_by_name, 'We have no op_proto for op: {}'.format(op.name)
        args = args_from_tfop(op, op_proto_by_name[op.name], allow_missing=True)
        for t in op.outputs:
            printed_tensors.add(t)
        print("{}{} = {}({})".format(
            indent, _format_result_names(op.outputs), op.name, _format_args(args)), file=f)
    print(file=f)
    if with_name_dict:
        inputs = ",\n{}{}".format(indent, indent).join('("{}", {})'.format(name, _format_tensor_name(t.name))
                                                       for name, t in zip(tf_graph.input_ids, tf_graph.inputs))
        print("{}__inputs = OrderedDict([\n{}{}{}\n{}])".format(indent, indent, indent, inputs, indent), file=f)
        outputs = ",\n{}{}".format(indent, indent).join('("{}", {})'.format(name, _format_tensor_name(t.name))
                                                        for name, t in zip(tf_graph.output_ids, tf_graph.outputs))
        print("{}__outputs = OrderedDict([\n{}{}{}\n{}])".format(indent, indent, indent, outputs, indent), file=f)
        tensors = ",\n{}{}".format(indent, indent).join('("{}", {})'.format(t.name, _format_tensor_name(t.name))
                                                        for t in sorted(tf_graph.tensors, key=lambda t: t.name)
                                                        if t in printed_tensors)
        print("{}__tensors = OrderedDict([\n{}{}{}\n{}])".format(indent, indent, indent, tensors, indent), file=f)
        print(file=f)
        print("{}return __inputs, __outputs, __tensors".format(indent), file=f)
    else:
        outputs = ",\n{}{}".format(indent, indent).join('("{}", {})'.format(name, _format_tensor_name(t.name))
                                                        for name, t in zip(tf_graph.output_ids, tf_graph.outputs))
        print("{}return OrderedDict([\n{}{}{}\n{}])".format(indent, indent, indent, outputs, indent), file=f)
    print(file=f)


def generate_source_operations(tf_graph):
    # type: (TFGraph) -> None
    # assert tf_graph.is_unique

    def get_input_name(tensor):
        if tensor in tf_graph.inputs and tf_graph.input_ids:
            return tf_graph.input_ids[tf_graph.inputs.index(tensor)]
        else:
            return tensor.name.split(':')[0]

    for t in list(tf_graph.tensors):
        if t.producer is None:
            if t.is_constant:
                if t.rank > 0 or t in tf_graph.outputs:
                    TFOperation(graph=tf_graph,
                                name="tf.constant",
                                attribs=dict(shape=t.shape, dtype=t.dtype, value=t.data),
                                inputs=tuple(),
                                outputs=t)
            elif t.is_variable:
                TFOperation(graph=tf_graph,
                            name="tf.get_variable",
                            attribs=dict(shape=t.shape, dtype=t.dtype, name=t.label),
                            inputs=tuple(),
                            outputs=t)
            elif t.producer is None:
                TFOperation(graph=tf_graph,
                            name="tf.placeholder",
                            attribs=dict(shape=t.shape, dtype=t.dtype, name=get_input_name(t)),
                            inputs=tuple(),
                            outputs=t)
            else:
                assert False, "All non-source tensors must have a producer in a TF graph"


def remove_source_operations(tf_graph):
    # type: (TFGraph) -> None
    # assert tf_graph.is_unique

    tf_graph.remove_operations([op for op in list(tf_graph.operations)
                                if op.name in {"tf.constant", "tf.get_variable", "tf.placeholder"}],
                               unlink=True)


class _Invocation(object):
    def __init__(self):
        self.function_name = None
        self.args = None
        self.nesting_level = None
        self.result = None
        self.stack = None
        self.parent = None
        self.children = None
        self.tmp_frame = None
        self.tmp_level = None
        self.tmp_args_checked = None

    def __repr__(self):
        return "_Invocation({}, {}, {}, {})".format(self.nesting_level, self.result, self.function_name, self.args)


class _InvocationTracer(object):
    def __init__(self, functions_by_name):
        self.function_name_by_qualified_undecorated_name = {}
        self.undecorated_function_names = set()

        self.invocation_stack = []
        self.invocations = []
        self.call_count = {}
        self.allow_nesting = False

        for function_name, functions in six.iteritems(functions_by_name):
            for function_ in functions:
                if function_ is not None:
                    undecorated_function = _undecorate(function_)
                    undecorated_name = undecorated_function.__name__
                    qualified_undecorated_name = undecorated_function.__module__ + '.' + undecorated_function.__name__
                    self.function_name_by_qualified_undecorated_name[qualified_undecorated_name] = function_name
                    self.undecorated_function_names.add(undecorated_name)
                    self.call_count[function_name] = 0

    def __call__(self, frame, event, result):
        if not self.allow_nesting and self.invocation_stack and event == 'call':
            return None

        func_name = frame.f_code.co_name
        is_constructor = False
        if func_name == '__init__':
            is_constructor = True
            result = frame.f_locals.get('self')
            if result is not None:
                func_name = result.__class__.__name__

        if func_name not in self.undecorated_function_names:
            return None

        mod = inspect.getmodule(frame)
        if mod is None:
            return None

        func_name = mod.__name__ + '.' + func_name

        if func_name not in self.function_name_by_qualified_undecorated_name:
            return None

        func_name = self.function_name_by_qualified_undecorated_name[func_name]

        if event == 'call':
            self.call_count[func_name] += 1
            arg_values = inspect.getargvalues(frame)
            invocation = _Invocation()
            invocation.function_name = func_name
            invocation.args = {key: value for (key, value) in arg_values.locals.items() if key in arg_values.args}
            if is_constructor:
                del invocation.args["self"]
            invocation.nesting_level = len(self.invocation_stack)
            invocation.tmp_frame = frame
            if self.invocation_stack:
                invocation.parent = self.invocation_stack[-1]
                invocation.parent.children.append(invocation)
            invocation.children = []
            self.invocation_stack.append(invocation)
            # This allows us to trace the "return" event of the same function
            return self
        elif event == 'return':
            if self.invocation_stack:
                invocation = self.invocation_stack[-1]
                if invocation.tmp_frame == frame:
                    self.invocation_stack.pop()
                    if result is not None:
                        invocation.tmp_frame = None
                        if isinstance(result, list):
                            result = list(result)
                        elif isinstance(result, dict):
                            result = dict(result)
                        invocation.result = result
                        invocation.stack = traceback.extract_stack(frame.f_back)
                        self.invocations.append(invocation)
                    else:
                        print("Error: Traced function {} returned without result, it probably raised."
                              .format(func_name))
            return None

    def trace(self, func, allow_nesting=False):
        self.invocation_stack = []
        self.invocations = []
        self.allow_nesting = allow_nesting

        for k in six.iterkeys(self.call_count):
            self.call_count[k] = 0

        old_tracer = sys.gettrace()
        sys.settrace(self)
        result = func()
        sys.settrace(old_tracer)

        return result


def _undecorate(func, _orig_name=None):
    if _orig_name is None:
        _orig_name = func.__name__

    if not hasattr(func, "__closure__") or not func.__closure__:
        return func

    for obj in (c.cell_contents for c in func.__closure__):
        if hasattr(obj, "__name__") and obj.__name__ == _orig_name:
            return obj
        if hasattr(obj, "__closure__") and obj.__closure__:
            found = _undecorate(obj, _orig_name)
            if found:
                return found
    return None


def _get_location_summary(stack, max_path_length=32):
    def get_path_and_line_number(frame_):
        if sys.version_info[0] < 3:
            return frame_[0], frame_[1]
        else:
            return frame_.filename, frame_.lineno

    def shorten(path_):
        if len(path_) > max_path_length:
            s = path_[-max_path_length + 3:]
            if "/" in s:
                s = s[s.index("/"):]
            elif "\\" in s:
                s = s[s.index("\\"):]
            return "..." + s
        return path_

    def is_library_file(path_):
        return "/dist-packages/" in path_ or "/site-packages/" in path_

    location = "unknown location"
    if stack:
        path, line_number = get_path_and_line_number(stack[-1])
        location = "{}:{}".format(shorten(path), line_number)
        if is_library_file(path):
            for frame in reversed(stack):
                path, line_number = get_path_and_line_number(frame)
                if not is_library_file(path):
                    location = "{}:{} ({})".format(shorten(path), line_number, location)
                    break

    return location


def _eliminate_named_tuples(data):
    def transform(data_):
        if isinstance(data_, tuple):
            return tuple(data_)
        return data_

    return utils.recursive_transform(data, transform)


def _eliminate_identities(invocations):
    return [invocation for invocation in invocations
            if not (isinstance(invocation.result, tf.Tensor)
                    and utils.recursive_any(invocation.args, lambda x: x is invocation.result))]


def _check_has_untraced_ops(invocations, result):
    has_untraced = [False]
    tensors = set()
    for invocation in invocations:
        def check_args(arg):
            if isinstance(arg, tf.Variable):
                arg = arg.value()
            if isinstance(arg, tf.Tensor) and arg not in tensors:
                print("Error: Untraced tensor: {}, used near: {}".format(arg, _get_location_summary(invocation.stack)))

                has_untraced[0] = True
                tensors.add(arg)

        def add_results(result_):
            if isinstance(result_, tf.Variable):
                result_ = result_.value()
            if isinstance(result_, tf.Tensor):
                tensors.add(result_)

        utils.recursive_visit(invocation.args, check_args)
        utils.recursive_visit(invocation.result, add_results)

    def check_outputs(result_):
        if isinstance(result_, tf.Variable):
            result_ = result_.value()
        if isinstance(result_, tf.Tensor) and result_ not in tensors:
            print("Error: Untraced output tensor: {}".format(result_))

            has_untraced[0] = True
            tensors.add(result_)

    utils.recursive_visit(result, check_outputs)

    return has_untraced[0]


def _normalize_types(arg):
    if utils.is_anyint(arg):
        return utils.anyint_to_int(arg)
    elif utils.is_anystr(arg):
        return utils.anystr_to_str(arg)
    elif isinstance(arg, np.ndarray):
        return arg.tolist()
    elif isinstance(arg, tf.TensorShape):
        if arg.dims is None:
            return None
        return [None if dim is None else int(dim) for dim in arg.as_list()]
    elif isinstance(arg, tf.Dimension):
        return arg.value
    elif isinstance(arg, tf.DType):
        return arg.name
    elif isinstance(arg, (np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64,
                          np.float16, np.float32, np.float64,
                          np.bool_)):
        return arg.item()
    else:
        return arg


def _recursive_visit_with_path(arg, fun, path_prefix=""):
    # type: (typing.Any, typing.Callable[[typing.Any, str], None], str)->None

    if type(arg) is tuple or type(arg) is list:
        for i, item in enumerate(arg):
            _recursive_visit_with_path(item, fun, "{}_{}".format(path_prefix, i))
    elif type(arg) is dict or type(arg) is OrderedDict:
        for k, v in six.iteritems(arg):
            _recursive_visit_with_path(v, fun, "{}_{}".format(path_prefix, k))
    else:
        fun(arg, path_prefix)


def _get_nice_input_name(tf_name):
    pos = tf_name.find(':')
    tf_name = tf_name[:pos] if pos != -1 else tf_name
    tf_name = ''.join(c if c.isalnum() else '_' for c in tf_name)
    if not tf_name[0].isalpha() and tf_name[0] != '_':
        return 'n_' + tf_name  # TODO this might cause collision
    return tf_name


def _get_arg(op_name, args, arg_proto):
    # type: (str, typing.Dict[str, typing.Any], ArgProto)->typing.Any
    found = True
    value = None
    for arg_name in arg_proto.arg_names:
        if arg_name in args:
            found = True
            if args[arg_name] is not None:
                assert value is None
                value = args[arg_name]
    if found:
        return value
    if arg_proto.is_optional:
        return None
    assert False, "Arg '{}' not found for op '{}'".format(arg_proto.primary_arg_name, op_name)


def _get_attrs(op_name, args, op_proto):
    # type: (str, typing.Dict[str, typing.Any], OpProto)->typing.Dict[str, typing.Any]
    return {arg_proto.primary_arg_name: _get_arg(op_name, args, arg_proto)
            for arg_proto in op_proto.list_nontensor_arg_protos()}


def _unify_attrs(attrs, op_proto):
    # type: (typing.Dict[str, typing.Any], OpProto)-> typing.Dict[str, typing.Any]
    dict2 = {}
    for arg_proto in op_proto.list_nontensor_arg_protos():
        val = attrs[arg_proto.primary_arg_name]
        if arg_proto.is_array and val is not None:
            val = utils.listify(val)
        dict2[arg_proto.primary_arg_name] = val
    return dict2


def _ensure_tensor(g, value, op_name):
    # type: (TFGraph, typing.Any, str)->typing.Optional[TFTensor]
    if value is None or isinstance(value, TFTensor):
        return value
    elif isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], TFTensor):
        tensor = TFTensor(graph=g, name=None, shape=[1] + value[0].shape, dtype=value[0].dtype)
        TFOperation(graph=g, name="tf.expand_dims", inputs=value[0], attribs=dict(axis=0), outputs=tensor)
        return tensor
    else:
        assert not utils.recursive_any(value, lambda x: isinstance(x, TFTensor)), \
            "Missing [] after T in op proto of '{}'?".format(op_name)
        arr = np.array(value)
        if str(arr.dtype).startswith('int'):
            arr = np.array(value, dtype=np.float32)
        return TFTensor(graph=g, name=None, shape=list(arr.shape), dtype=str(arr.dtype), data=arr.tolist())


def _get_inputs(graph, op_name, args, op_proto):
    # type: (TFGraph, str, typing.Dict[str, typing.Any], OpProto)->typing.Union[typing.List, typing.Tuple]
    inputs = []
    is_list = False
    for arg_proto in op_proto.list_tensor_arg_protos():
        arg = _get_arg(op_name, args, arg_proto)
        if arg_proto.is_array:
            is_list = True
            inputs += arg
        else:
            if arg_proto.is_optional and arg is None:
                continue
            inputs.append(arg)
    inputs = [_ensure_tensor(graph, value, op_name) for value in inputs]
    return inputs if is_list else tuple(inputs)


def _get_outputs(result):
    # type: (typing.Any)->typing.Union[typing.List, typing.Tuple]
    is_list = isinstance(result, list)
    outputs = utils.recursive_collect(result)
    return outputs if is_list else tuple(outputs)


def _unify_shape(shape):
    if utils.recursive_any(shape, lambda x: x is None):
        return None
    assert all(isinstance(s, int) for s in shape)
    return shape


def _tolist_safe(arr):
    if arr.dtype == np.object:
        return [_tolist_safe(a) for a in arr]
    else:
        return arr.tolist()


def _to_tf_graph(name, invocations, output, op_proto_by_name):
    # type: (str, typing.List[_Invocation], typing.Any, typing.Dict[str, OpProto])->TFGraph
    g = TFGraph(name)
    tensor_by_tf_tensor = {}
    inputs = OrderedDict()
    outputs = OrderedDict()
    const_value_by_tensor = {}  # type: typing.Dict[TFTensor, np.ndarray]
    for invocation in invocations:
        if invocation.function_name == "tf.get_variable" and invocation.result.value() in tensor_by_tf_tensor:
            continue

        def arg_transform(arg):
            if isinstance(arg, tf.Variable):
                arg = arg.value()

            if isinstance(arg, tf.Tensor):
                assert arg in tensor_by_tf_tensor, "Undefined tensor: {}".format(arg)
                return tensor_by_tf_tensor[arg]
            return _normalize_types(arg)

        args = utils.recursive_transform(invocation.args, arg_transform)

        def result_transform(result_):
            if isinstance(result_, tf.Variable):
                result_ = result_.value()

            if isinstance(result_, tf.Tensor):
                if result_ in tensor_by_tf_tensor:
                    print("Warning: {} was returned multiple times.\nInvocation: {}"
                          .format(tensor_by_tf_tensor[result_], invocation))
                    t = TFTensor(graph=g,
                                 name=_normalize_types(result_.name) + ":duplicate",
                                 shape=_unify_shape(_normalize_types(result_.shape)),
                                 dtype=_normalize_types(result_.dtype))
                else:
                    t = TFTensor(graph=g,
                                 name=_normalize_types(result_.name),
                                 shape=_unify_shape(_normalize_types(result_.shape)),
                                 dtype=_normalize_types(result_.dtype))
                    tensor_by_tf_tensor[result_] = t
                return t
            return result_

        result = utils.recursive_transform(invocation.result, result_transform)

        if invocation.function_name == "tf.placeholder":
            assert isinstance(result, TFTensor)
            if result not in six.itervalues(inputs):
                input_name = _get_nice_input_name(result.name)
                inputs[input_name] = result
        elif invocation.function_name == "tf.constant":
            assert isinstance(result, TFTensor)
            result.data = np.array(args["value"], dtype=result.dtype).flatten().tolist()

            tf_py_eval.evaluate_constant(result, const_value_by_tensor)
            tf_py_shape_inference.evaluate_shape_of_constant(result, const_value_by_tensor)
        elif invocation.function_name in ["tf.Variable", "tf.get_variable"]:
            assert isinstance(result, TFTensor)
            result.data = np.array([])
            result.label = result.name
        else:
            op_proto = op_proto_by_name[invocation.function_name]
            op_attrs = _get_attrs(invocation.function_name, args, op_proto)

            def eval_tensors(x):
                if isinstance(x, TFTensor):
                    assert x in const_value_by_tensor, "Tensor could not be evaluated: {}".format(x)
                    return _tolist_safe(const_value_by_tensor[x])
                return x

            op_attrs = utils.recursive_transform(op_attrs, eval_tensors)
            op_attrs = _unify_attrs(op_attrs, op_proto)

            op_inputs = _get_inputs(g, invocation.function_name, args, op_proto)
            for input in op_inputs:  # evaluate newly generated constant tensors
                if input is not None and input not in const_value_by_tensor:
                    if input.is_constant:
                        tf_py_eval.evaluate_constant(input, const_value_by_tensor)
                        tf_py_shape_inference.evaluate_shape_of_constant(input, const_value_by_tensor)
                    elif input.producer is not None and input.producer.name == "tf.expand_dims":
                        tf_py_eval.evaluate_expand_dims(input.producer, const_value_by_tensor)
                        tf_py_shape_inference.evaluate_shape_of_expand_dims(input.producer)

            op_outputs = _get_outputs(result)
            op = TFOperation(graph=g,
                             name=invocation.function_name,
                             attribs=op_attrs,
                             inputs=op_inputs,
                             outputs=op_outputs,
                             location=_get_location_summary(invocation.stack))
            tf_py_eval.try_to_evaluate_operation(op, const_value_by_tensor)
            tf_py_shape_inference.evaluate_shape_of_operation(op, const_value_by_tensor)

    def visit(output_, path):
        # type: (typing.Any, str)->None

        if isinstance(output_, tf.Variable):
            output_ = output_.value()

        if isinstance(output_, tf.Tensor):
            assert output_ in tensor_by_tf_tensor, "Undefined tensor: {}".format(output_)
            output_tensor = tensor_by_tf_tensor[output_]
            path = path[1:]
            if not path:
                path = "output"
            elif path[0].isdigit():
                path = "output" + path
            assert utils.is_identifier(path), \
                "Bad name_override '{}' for tensor {}. " \
                "Please use valid identifiers as keys in the dict(s) " \
                "returned by your network function.".format(path, output_tensor.name)
            outputs[path] = output_tensor

    _recursive_visit_with_path(output, visit)

    g.inputs = inputs
    g.outputs = outputs
    return g


def _print_invocations(invocations):
    print("Number of invocations:", len(invocations))
    for invocation in invocations:
        print(invocation)
    print()


# Helpers just for gradient calculation:

def with_all_gradients(net_fun, name=None):
    def get_placeholders():
        tensors = []
        for op in tf.get_default_graph().get_operations():
            if "Placeholder" in op.node_def.op:
                tensors.append(op.outputs[0])
        return sorted(tensors, key=lambda t: t.name)

    def to_id(s):
        cc = []
        for c in s:
            if not c.isalnum() and c != "_":
                c = "_"
            cc.append(c)
        s = ''.join(cc)
        if s[0] != '_' and not s[0].isalpha():
            s2 = "tensor_" + s
            s = s2
        return s.lower()

    def net_fun_with_gradients():
        outputs_ = _eliminate_named_tuples(net_fun())
        outputs_dict = OrderedDict()

        def visit(output_, path):
            # type: (typing.Any, str)->None
            if isinstance(output_, tf.Variable):
                output_ = output_.value()
            if isinstance(output_, tf.Tensor):
                path = path[1:]
                if not path:
                    path = "output"
                elif path[0].isdigit():
                    path = "output" + path
                assert utils.is_identifier(path), \
                    "Bad name_override '{}' for tensor {}. " \
                    "Please use valid identifiers as keys in the dict(s) " \
                    "returned by your network function.".format(path, output_.name)
            outputs_dict[path] = output_

        _recursive_visit_with_path(outputs_, visit)

        inputs = get_placeholders() + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        grad_ys = None

        # We can test with other grad_ys too
        # grad_ys = [tf.constant(value=2.0, dtype=tf.float32, shape=output_.shape) for output_ in outputs]

        ys = [y
              for y in six.itervalues(outputs_dict)
              if y.dtype.name.startswith("float") or y.dtype.name.startswith("int") or y.dtype.name.startswith("uint")]

        gradients = [gradient
                     for gradient in tf.gradients(ys=ys, xs=inputs, grad_ys=grad_ys)
                     if gradient not in six.itervalues(outputs_dict)]

        items = [(name_, output_) for name_, output_ in six.iteritems(outputs_dict)]

        items += [("grad_{}".format(to_id(input_.name[:-2])), gradient)
                  for input_, gradient in zip(inputs, gradients)
                  if None not in [input_, gradient]]

        return OrderedDict(utils.unique(items, key=lambda item: item[1]))

    net_fun_with_gradients.__name__ = net_fun.__name__ if name is None else name
    return net_fun_with_gradients


def _fix_strange_grad_functions(invocations):
    for invocation in invocations:
        if isinstance(invocation.args.get("op"), tf.Operation):
            # print("Info: Had to fix strange invocation {}".format(invocation.function_name))
            op = invocation.args["op"]
            args = {}
            for i, t in enumerate(op.inputs):
                args["_orig_input_{}".format(i)] = t
            for i, t in enumerate(op.outputs):
                args["_orig_output_{}".format(i)] = t
            for k, v in six.iteritems(invocation.args):
                if k != "op":
                    args[k] = v
            invocation.args = args
            if isinstance(invocation.result, (list, tuple)) and all(r is None for r in invocation.result[1:]):
                invocation.result = invocation.result[0]


def _bubble_up(invocation, least_nested_producer):
    if invocation.tmp_level == -1:
        return

    print("Info: {} operation near {} has been expanded, because there is a reference to a partial result inside it. "
          "This usually happens when using expand_gradients=True."
          .format(invocation.function_name, _get_location_summary(invocation.stack)))

    for child in invocation.children:
        child.tmp_level = 0
        _check_args_for_nested_reference(child, least_nested_producer)
    invocation.tmp_level = -1
    if invocation.parent:
        _bubble_up(invocation.parent, least_nested_producer)


def _check_args_for_nested_reference(invocation, least_nested_producer):
    if invocation.tmp_args_checked:
        return
    invocation.tmp_args_checked = True

    def visit(arg):
        if isinstance(arg, tf.Variable):
            arg = arg.value()
        if isinstance(arg, tf.Tensor) and arg in least_nested_producer:
            producer = least_nested_producer[arg]
            if producer.tmp_level > 0:
                if producer.function_name == "tf.constant":
                    producer.tmp_level = 0
                else:
                    _bubble_up(producer.parent, least_nested_producer)

    utils.recursive_visit(invocation.args, visit)


def _eliminate_nesting(invocations):
    # type: (typing.List[_Invocation])->typing.List[_Invocation]
    least_nested_producer = {}  # type: typing.Dict[tf.Tensor, _Invocation]
    for invocation in invocations:
        def add_producers(result_):
            if isinstance(result_, tf.Variable):
                result_ = result_.value()
            if isinstance(result_, tf.Tensor):
                if (result_ not in least_nested_producer
                        or invocation.nesting_level < least_nested_producer[result_].nesting_level):
                    least_nested_producer[result_] = invocation

        utils.recursive_visit(invocation.result, add_producers)

    for invocation in invocations:
        invocation.tmp_level = invocation.nesting_level
        invocation.tmp_args_checked = False

    for invocation in invocations:
        if invocation.tmp_level == 0:
            _check_args_for_nested_reference(invocation, least_nested_producer)

    invocations = [invocation for invocation in invocations if invocation.tmp_level == 0]

    for invocation in invocations:
        invocation.tmp_level = None
        invocation.tmp_args_checked = None

    return invocations


def _generate_names(tf_graph, custom_imports, custom_op_protos):
    # type: (TFGraph, str, typing.Optional[typing.List[OpProto]])->typing.Dict[TFTensor, str]
    f = six.StringIO()
    try:
        _print(tf_graph=tf_graph,
               file_handle=f,
               custom_op_protos=custom_op_protos,
               custom_imports=custom_imports,
               with_name_dict=True)
        src = f.getvalue()
    finally:
        f.close()

    fun = _tfsource_to_function(src, tf_graph.name)

    tf.reset_default_graph()
    tf.set_random_seed(0)

    _input, _output, tensors = fun()

    names = {}

    for t in tf_graph.tensors:
        if t.name in tensors:
            names[t] = tensors[t.name].name
        else:
            names[t] = None

    return names


def _get_rename_info(graph, names):
    tensor_infos = []
    for tensor in graph.tensors:
        if tensor.name and names[tensor]:
            name = names[tensor]
            name = name if ':' in name else name + ':0'
            tensor_infos.append(conversion_info.TensorInfo(source_name=tensor.name,
                                                           target_name=name,
                                                           target_shape=list(tensor.shape),
                                                           target_dtype=tensor.dtype,
                                                           is_input=tensor in graph.inputs,
                                                           is_output=tensor in graph.outputs,
                                                           is_variable=tensor.is_variable))
    return conversion_info.ConversionInfo(tensor_infos)


class Reader(object):

    def __init__(self, expand_gradients=False, custom_traceable_functions=None):
        self._expand_gradients = expand_gradients
        self._custom_traceable_functions = custom_traceable_functions

    def __call__(self, function_path, checkpoint_path=None):
        package_and_module, function_name = function_path.rsplit('.', 1)

        sys.path.insert(0, '.')
        try:
            module = importlib.import_module(package_and_module)
        except ImportError:
            raise RuntimeError("Error: Can not import module {}".format(package_and_module))
        finally:
            sys.path = sys.path[1:]

        try:
            function_ = getattr(module, function_name)
            function_.__name__ = function_name
        except AttributeError:
            raise RuntimeError(
                "Error: Function {} not found in module {}".format(function_name, package_and_module))

        return trace(network_function=function_,
                     checkpoint_path=checkpoint_path,
                     expand_gradients=self._expand_gradients,
                     custom_traceable_functions=self._custom_traceable_functions)


class Writer(object):

    def __init__(self, write_weights=True, custom_imports=None, custom_op_protos=None):
        self._write_weights = write_weights
        self._custom_imports = custom_imports
        self._custom_op_protos = custom_op_protos

    def __call__(self, graph, filename):
        return write(graph, filename,
                     write_weights=self._write_weights,
                     custom_imports=self._custom_imports,
                     custom_op_protos=self._custom_op_protos)
