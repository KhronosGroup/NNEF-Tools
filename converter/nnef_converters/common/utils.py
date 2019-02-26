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

from __future__ import division, print_function

import inspect
import math
import os
import shutil
import sys
import tarfile
from collections import OrderedDict

if sys.version_info[0] < 3:
    # noinspection PyUnresolvedReferences,PyCompatibility
    from cStringIO import StringIO
else:
    # noinspection PyUnresolvedReferences,PyCompatibility
    from io import StringIO

if sys.version_info[0] < 3:
    # noinspection PyUnresolvedReferences,PyCompatibility
    from Queue import Queue
else:
    # noinspection PyUnresolvedReferences,PyCompatibility
    from queue import Queue

if sys.version_info[0] < 3:
    import unicodedata

from .types import *

_RedText = "\x1b[31m"
_ResetStyle = "\x1b[0m"

_had_error = False
error_file = sys.stderr
warning_file = sys.stderr
info_file = sys.stdout

NNEF_READ_PROVISIONAL = False


def had_error():
    return _had_error


def reset_error():
    global _had_error
    _had_error = False


class ConversionException(Exception):
    pass


def raise_if_had_error(listing=""):
    if had_error():
        reset_error()
        if listing:
            print_info("Listing code before raise:")
            print_info(listing)
        raise ConversionException("nn_lang_converter: There were errors!")


def print_error(msg):
    global _had_error

    if error_file.isatty():
        error_file.write("%sError: %s%s\n" % (_RedText, msg, _ResetStyle))
    else:
        error_file.write("Error: %s\n" % (msg,))
    error_file.flush()
    _had_error = True


def print_warning(msg):
    if warning_file.isatty():
        warning_file.write("%sWarning: %s%s\n" % (_RedText, msg, _ResetStyle))
    else:
        warning_file.write("Warning: %s\n" % (msg,))
    warning_file.flush()


def print_info(msg, end='\n'):
    info_file.write(msg + end)
    info_file.flush()


def add_line_numbers(s):
    r = ''
    for i, line in enumerate(s.split('\n')):
        r += str(i + 1) + ': ' + line + '\n'
    return r[:-1]


REMOVE = object()


def recursive_transform(data, fun):
    if type(data) is dict or type(data) is OrderedDict:
        data2 = type(data)()
        for k, v in data.items():
            w = recursive_transform(v, fun)
            if w is not REMOVE:
                data2[k] = w
    elif type(data) is list or type(data) is tuple:
        data2 = []
        for v in data:
            w = recursive_transform(v, fun)
            if w is not REMOVE:
                data2.append(w)
        data2 = type(data)(data2)
    else:
        data2 = fun(data)

    return data2


def recursive_visit(data, fun):
    if type(data) is dict or type(data) is OrderedDict:
        for _k, v in data.items():
            recursive_visit(v, fun)
    elif type(data) is list or type(data) is tuple:
        for v in data:
            recursive_visit(v, fun)
    else:
        fun(data)


def recursive_contains_by_pointer(haystack, needle):
    is_found = [False]

    def visitor(x):
        if x is needle:
            is_found[0] = True

    recursive_visit(haystack, visitor)
    return is_found[0]


def zip_inverse(output_count, arr):
    if not arr:
        return tuple([[] for _ in range(output_count)])
    return tuple([list(a) for a in zip(*arr)])


def flatten(x):
    out_arr = []
    _flatten(x, out_arr)
    return out_arr


def _flatten(x, out_arr):
    if isinstance(x, (list, tuple)):
        for y in x:
            _flatten(y, out_arr)
    else:
        out_arr.append(x)


def to_list(x):
    import numpy as np

    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    elif isinstance(x, np.ndarray):
        return x.tolist()
    else:
        raise Exception("Cannot convert {} to list".format(x.__class__.__name__))


def tuplify(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return x,


def listify(x):
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, list):
        return x
    return [x]


def unique(arr, key=None):
    if key is None:
        key = lambda item: item
    s = set()
    arr2 = []
    for a in arr:
        k = key(a)
        if k not in s:
            s.add(k)
            arr2.append(a)
    return arr2


def has_greater_than(arr, x):
    has = [False]

    def visit(y):
        if y > x:
            has[0] = True

    recursive_visit(arr, visit)

    return has[0]


def has_not_equal(arr, x):
    has = [False]

    def visit(y):
        if y != x:
            has[0] = True

    recursive_visit(arr, visit)

    return has[0]


def has_greater_than_0(arr):
    return has_greater_than(arr, 0)


def has_greater_than_1(arr):
    return has_greater_than(arr, 1)


def has_not_equal_0(arr):
    return has_not_equal(arr, 0)


def has_not_equal_1(arr):
    return has_not_equal(arr, 1)


def int_log2(i):
    if i == 0:
        return -1
    for j in range(0, 32):
        if i == (1 << j):
            return j
    return -2


def get_inverse_permutation(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def apply_permutation(list_, perm):
    assert len(list_) == len(perm)
    list2 = [None] * len(list_)
    for i in range(len(list_)):
        list2[i] = list_[perm[i]]
    return list2


def apply_permutation_to_axis(axis, perm):
    return perm.index(axis)


def apply_permutation_to_axes(axes, perm):
    return [apply_permutation_to_axis(a, perm) for a in axes]


def _unsqueeze_shape2(shape, axes, i, n):
    return ([] if i == n
            else ([1] + _unsqueeze_shape2(shape, axes, i + 1, n) if i in axes
                  else [shape[0]] + _unsqueeze_shape2(shape[1:], axes, i + 1, n)))


# as in nnef (axes correspond to output dims)
def apply_unsqueeze_shape(shape, axes):
    return _unsqueeze_shape2(shape, axes, 0, len(shape) + len(axes))


def apply_squeeze_shape(shape, axes, can_squeeze_non_one=False):
    if not can_squeeze_non_one:
        for i in range(len(shape)):
            if i in axes:
                assert shape[i] == 1

    return [shape[i] for i in range(len(shape)) if i not in axes]


def without(iterable, x):
    list_ = list(iterable)
    if x in list_:
        list_.remove(x)
    return list_


def get_functions(prefix="", module=None):
    if module is None:
        caller_frame = inspect.stack()[1]
        module = inspect.getmodule(caller_frame[0])

    return sorted(
        [
            obj
            for name, obj in inspect.getmembers(module)
            if inspect.isfunction(obj) and name.startswith(prefix)
        ],
        key=lambda f: inspect.getsourcelines(f)[1]
    )


def try_tf_dtype_to_np(tf_dtype):
    import tensorflow as tf
    tf_dtype = tf.as_dtype(tf_dtype).base_dtype
    if tf_dtype.is_numpy_compatible:
        return tf_dtype.as_numpy_dtype
    return None


_nnef_dtype_by_np_dtype = None


def _np_load_dtype_table():
    import numpy as np

    global _nnef_dtype_by_np_dtype

    if _nnef_dtype_by_np_dtype is None:
        _nnef_dtype_by_np_dtype = {
            np.dtype(np.float16): 'scalar',
            np.dtype(np.float32): 'scalar',
            np.dtype(np.float64): 'scalar',
            np.dtype(np.int8): 'integer',
            np.dtype(np.uint8): 'integer',
            np.dtype(np.int16): 'integer',
            np.dtype(np.uint16): 'integer',
            np.dtype(np.int32): 'integer',
            np.dtype(np.uint32): 'integer',
            np.dtype(np.int64): 'integer',
            np.dtype(np.uint64): 'integer',
            np.dtype(np.bool_): 'logical'
        }


def is_np_dtype_exportable_to_nnef(dtype):
    import numpy as np
    _np_load_dtype_table()

    return np.dtype(dtype) in _nnef_dtype_by_np_dtype


def try_np_dtype_to_nnef(dtype):
    import numpy as np
    _np_load_dtype_table()

    return _nnef_dtype_by_np_dtype.get(np.dtype(dtype))


def are_np_dtypes_compatible_in_nnef(t1, t2):
    t1 = try_np_dtype_to_nnef(t1)
    t2 = try_np_dtype_to_nnef(t2)
    return t1 is not None and t2 is not None and t1 == t2


def nnef_dtype_to_tf(nnef_dtype):
    import tensorflow as tf

    return {
        "integer": tf.int32,
        "scalar": tf.float32,
        "logical": tf.bool
    }[nnef_dtype]


def tf_type_of_python_scalar(x):
    import tensorflow as tf

    if isinstance(x, float):
        return tf.float32
    elif isinstance(x, int):
        return tf.int32
    elif isinstance(x, bool):
        return tf.bool
    else:
        assert False


def write_nnef_tensor(filename, tensor):
    import nnef

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "wb") as file:
        nnef.write_tensor(file, tensor, version=(1, 0))


def read_nnef_tensor(filename):
    import nnef

    with open(filename, "rb") as file:
        if NNEF_READ_PROVISIONAL:
            # noinspection PyProtectedMember
            return nnef._read_tensor_provisional(file)
        else:
            return nnef.read_tensor(file)[0]


def tf_call_silently(fun, *args):
    if 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
        old_value = os.environ['TF_CPP_MIN_LOG_LEVEL']
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        retval = fun(*args)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = old_value
        return retval
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        retval = fun(*args)
        del os.environ['TF_CPP_MIN_LOG_LEVEL']
        return retval


def tf_has_cuda_gpu():
    import tensorflow as tf
    return tf_call_silently(tf.test.is_gpu_available, True)


def _tf_is_constant(tensor):
    # for the rank operator this is not good logic, it can have a placeholder as input but it is still constant
    if "Variable" in tensor.op.node_def.op or "Placeholder" in tensor.op.node_def.op:
        return False
    for t in tensor.op.inputs:
        if not _tf_is_constant(t):
            return False
    return True


def _tf_evaluate_if_constant(tensor):
    import tensorflow as tf

    if not _tf_is_constant(tensor):
        return None

    with tf.Session() as sess:
        return sess.run(tensor)


def tf_constant_value(tensor, evaluate_if_needed=True):
    import tensorflow as tf

    res = tf.contrib.util.constant_value(tensor)
    if not evaluate_if_needed or res is not None:
        return res

    return tf_call_silently(_tf_evaluate_if_constant, tensor)


def tf_version_greater_equal(major, minor):
    import tensorflow as tf

    str = tf.__version__
    i = str.index('.')
    j = str.index('.', i + 1)
    return (int(str[:i]), int(str[i + 1:j])) >= (major, minor)


def tf_print_graph(output_tensors):
    import tensorflow as tf

    if isinstance(output_tensors, tf.Tensor):
        output_tensors = [output_tensors]
    if isinstance(output_tensors, dict):
        output_tensors = list(output_tensors.values())
    print_info("{}".format(output_tensors))
    visited = set()
    indent = ""
    _tf_print_graph2(output_tensors, visited, indent)


def _tf_print_graph2(tensors, visited, indent):
    for tensor in tensors:
        _tf_print_graph3(tensor, visited, indent)


def _tf_print_graph3(tensor, visited, indent):
    if tensor in visited:
        return
    visited.add(tensor)
    input_names = [i.name for i in tensor.op.inputs]
    const_val = tf_constant_value(tensor)
    if const_val is not None:
        if const_val.size > 5:
            const_val = "value_known"
        else:
            const_val = "value_known: {}".format(const_val)
    print_info("{}{} = {}({}) {} shape: {}".format(
        indent,
        tensor.name,
        tensor.op.node_def.op,
        ", ".join(input_names),
        "" if const_val is None else const_val,
        tensor.shape
    ))
    _tf_print_graph2(tensor.op.inputs, visited, indent + "  ")


def _tf_get_inputs():
    import tensorflow as tf

    tensors = []
    for op in tf.get_default_graph().get_operations():
        if "Variable" in op.node_def.op or "Placeholder" in op.node_def.op:
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
        s2 = "id_" + s
        s = s2
    return s


def to_id_without_number(s):
    pos = s.find(':')
    s = s[:pos] if pos != -1 else s
    return to_id(s)


def get_short_name(name):
    pos = name.find(':')
    name = name[:pos] if pos != -1 else name

    pos = name.rfind('/')
    name = name[pos + 1:] if pos != -1 else name
    return to_id(name)


def normalize_str_upper(str_or_none):
    if str_or_none is None:
        return None
    elif isinstance(str_or_none, bytes):
        return str_or_none.decode('utf-8').upper()
    else:
        return str_or_none.upper()


def tf_with_gradients(net_fun):
    import tensorflow as tf

    def f():
        outputs = net_fun()
        if isinstance(outputs, (list, tuple)):
            outputs = list(outputs)
        else:
            outputs = [outputs]
        inputs = _tf_get_inputs()

        # We can test with other grad_ys too
        # grad_ys = [tf.constant(value=2.0, dtype=tf.float32, shape=o.shape) for o in outputs]
        grad_ys = None
        ys = [y for y in outputs
              if y.dtype.name.startswith("float") or y.dtype.name.startswith("int") or y.dtype.name.startswith("uint")]
        gradients = [g for g in tf.gradients(ys=ys, xs=inputs, grad_ys=grad_ys) if g not in outputs]

        items = [("output{}".format(i), o) for i, o in enumerate(outputs)]
        items += [("grad_{}".format(to_id(i.name[:-2])), g) for i, g in zip(inputs, gradients) if None not in [i, g]]

        return dict(unique(items, key=lambda item: item[1]))

    f.__name__ = net_fun.__name__
    return f


def is_nhwc(op):
    data_format = op.args.get('data_format')
    if isinstance(data_format, bytes):
        data_format = data_format.decode('utf-8')
    return data_format is None or not data_format.upper().startswith("NC")


def shape_nhwc_to_nchw(shape):
    return shape[0:1] + shape[-1:] + shape[1:-1]


def shape_nchw_to_nhwc(shape):
    return shape[0:1] + shape[2:] + shape[1:2]


def shape_hwcn_to_nchw(shape):
    return shape[-1:] + shape[-2:-1] + shape[:-2]


def shape_hwcm_to_nchw(shape):
    return [shape[-2] * shape[-1], 1] + shape[:-2]


def shape_nchw_to_hwcn(shape):
    return shape[2:] + shape[1:2] + shape[:1]


def shape_nchw_to_hwcm(shape, input_channels):
    return shape[2:] + [input_channels, shape[0] // input_channels]


def transpose_axes_nhwc_to_nchw(rank):
    return shape_nhwc_to_nchw(list(range(rank)))


def transpose_axes_nchw_to_nhwc(rank):
    return shape_nchw_to_nhwc(list(range(rank)))


def transpose_axes_hwcn_to_nchw(rank):
    return shape_hwcn_to_nchw(list(range(rank)))


def transpose_axes_nchw_to_hwcn(rank):
    return shape_nchw_to_hwcn(list(range(rank)))


def reorder_axes(axes, perm):
    return [perm.index(a) for a in axes]


def is_list_of_consecutive_numbers(_list):
    if len(_list) <= 1:
        return True
    return _list == list(range(_list[0], _list[0] + len(_list)))


def is_reorderable_squeeze(squeeze_axes, perm):
    return is_list_of_consecutive_numbers(sorted(reorder_axes(squeeze_axes, perm)))


def np_apply_transforms(arr, trafos):
    if not trafos:
        return arr
    else:
        for trafo in trafos:
            import numpy as np
            if trafo[0] == "transpose":
                arr = np.transpose(arr, axes=trafo[1])
            elif trafo[0] == "unsqueeze":
                arr = np.reshape(arr, newshape=apply_unsqueeze_shape(list(arr.shape), trafo[1]))
            elif trafo[0] == "squeeze":
                arr = np.reshape(arr, newshape=apply_squeeze_shape(list(arr.shape), trafo[1]))
            elif trafo[0] == "reshape":
                arr = np.reshape(arr, newshape=trafo[1])
            else:
                assert False
        return arr


def starts_with(list_, prefix):
    return len(list_) >= len(prefix) and list_[:len(prefix)] == prefix


def ends_with(list_, ending):
    return len(list_) >= len(ending) and list_[-len(ending):] == ending


def can_broadcast_from_left(list_, prefix):
    if len(list_) < len(prefix):
        return False
    for i in range(len(prefix)):
        if prefix[i] not in [1, list_[i]]:
            return False
    return True


def can_broadcast_from_right(list_, ending):
    if len(list_) < len(ending):
        return False
    for i in range(len(ending)):
        if ending[-1 - i] not in [1, list_[-1 - i]]:
            return False
    return True


def ensure_dir(path, clear=False):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise Exception("{} is not a directory".format(path))
        if clear:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def without_slash(path):
    while len(path) > 1 and path[-1] == '/':
        path = path[:-1]
    return path


def without_file_name(path):
    return os.path.dirname(path)


class _OrderedDictMaker(object):
    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        assert all(isinstance(key, slice) for key in keys)

        return OrderedDict([(k.start, k.stop) for k in keys])


ordered_dict_maker = _OrderedDictMaker()


def nnef_dilated_filter_size_element(filter_size, dilation):
    return (filter_size - 1) * dilation + 1


def nnef_auto_padding_element(upscaled_size, downscaled_size, filter_size, stride, dilation):
    t = (downscaled_size - 1) * stride + nnef_dilated_filter_size_element(filter_size, dilation) - upscaled_size
    return max(0, int(math.floor(t / 2))), max(0, int(math.ceil(t / 2)))


# uses spatial sizes
def nnef_auto_padding(upscaled_size, downscaled_size, filter_size, stride, dilation):
    if len(unique([len(upscaled_size), len(downscaled_size), len(filter_size), len(stride), len(dilation)])) != 1:
        print_error("nnef_auto_padding: different ranks")
    return [
        nnef_auto_padding_element(X, x, f, s, d)
        for X, x, f, s, d
        in zip(upscaled_size, downscaled_size, filter_size, stride, dilation)
    ]


def count(iterable):
    c = 0
    for x in iterable:
        if x:
            c += 1
    return c


def silence(fun, *args, **kwargs):
    old_out = sys.stdout
    string_io = StringIO()
    sys.stdout = string_io
    try:
        return fun(*args, **kwargs)
    finally:
        output = string_io.getvalue()
        output_lower = output.lower()
        string_io.close()
        sys.stdout = old_out
        if "not implemented" in output_lower or "error" in output_lower:
            raise Exception("Error in output:\n" + output)


def compare_activation_dirs_np(dirname1, dirname2, verbose=False, tf_hack_fun_name=""):
    import numpy as np

    print_info("DIFF {} {}".format(dirname1, dirname2))
    fnlist1 = sorted(os.listdir(dirname1))
    fnlist2 = sorted(os.listdir(dirname2))
    fnset1 = set(fnlist1)
    fnset2 = set(fnlist2)

    fns = [fn for fn in fnlist1 if fn not in fnset2]
    if fns:
        print_warning("files only present in left dir: " + ", ".join(fns))

    fns = [fn for fn in fnlist2 if fn not in fnset1]
    if fns:
        print_warning("files only present in right dir: " + ", ".join(fns))

    fns = [fn for fn in fnlist1 if fn in fnset2]

    good = True
    for fn in fns:
        arr1 = read_nnef_tensor('{}/{}'.format(dirname1, fn))
        arr2 = read_nnef_tensor('{}/{}'.format(dirname2, fn))

        can_reshape = (starts_with(arr1.shape, arr2.shape) or starts_with(arr2.shape, arr1.shape))

        if can_reshape:
            arr1 = np.reshape(arr1, arr2.shape)

        if arr1.dtype != arr2.dtype:
            print_warning("different dtypes: {} {}".format(arr1.dtype.name, arr2.dtype.name))

        if not are_np_dtypes_compatible_in_nnef(arr1.dtype, arr2.dtype):
            print_error("incompatible dtypes: {} {}".format(arr1.dtype.name, arr2.dtype.name))
            good = False
        elif arr1.shape != arr2.shape:
            print_error("{} shape error {} {}".format(fn, arr1.shape, arr2.shape))
            good = False
        else:
            if arr1.dtype == np.dtype(np.bool_):
                differences = (arr1 != arr2).astype(np.int8).sum(dtype=np.int64)
                if differences > 0:  # Maybe too strict
                    print_error("bool tensors different at {} places".format(differences))
                    good = False
            else:
                max_error_rate1 = np.max(np.abs(arr2 - arr1) / np.maximum(np.abs(arr1), 1e-32))
                max_error_rate2 = np.max(np.abs(arr2 - arr1) / np.maximum(np.abs(arr2), 1e-32))
                max_error_rate = max(max_error_rate1, max_error_rate2)
                max_diff = np.max(np.abs(arr2 - arr1))
                max_value = min(np.max(np.abs(arr1)), np.max(np.abs(arr2)))
                if max_value < 1e-5:
                    print_info("{} max value: {}".format(fn, max_value))

                problematic_functions = [
                    "big_test_inception_v2",
                    "test_optimizer11_with_gradients",
                    "test_optimizer_no_io_transpose_with_gradients"
                ]

                err_thresh = 1e-4 if tf_hack_fun_name in problematic_functions else 1e-5

                if max_error_rate > err_thresh and max_diff > err_thresh:
                    print_error("{} max error rate: {} max diff: {}".format(fn, max_error_rate, max_diff))
                    # print(arr1.flat[:10], '\n', arr2.flat[:10], file=sys.stderr, flush=True)
                    good = False
                elif max_diff != 0 or verbose:
                    print_info("{} max error rate: {} max diff: {}".format(fn, max_error_rate, max_diff))
    if good:
        print_info("Activations were (almost) the same.")
    return good


def get_function_without_decorators(func, _orig_name=None):
    if _orig_name is None:
        _orig_name = func.__name__

    if not hasattr(func, "__closure__") or not func.__closure__:
        return func

    for obj in (c.cell_contents for c in func.__closure__):
        if hasattr(obj, "__name__") and obj.__name__ == _orig_name:
            return obj
        if hasattr(obj, "__closure__") and obj.__closure__:
            found = get_function_without_decorators(obj, _orig_name)
            if found:
                return found
    return None


def get_qualified_name(func, undecorate=True):
    undecorated = get_function_without_decorators(func) if undecorate else func
    return undecorated.__module__ + '.' + undecorated.__name__


def get_qualified_names(functions, undecorate=True):
    return [get_qualified_name(f, undecorate) for f in functions]


def without_nones(list_):
    return [x for x in list_ if x is not None]


def ensure_not_unicode_in_python2(s):
    # type: (AnyStr)->str
    if sys.version_info[0] < 3 and isinstance(s, unicode):
        return unicodedata.normalize('NFKD', s).encode('ascii', 'replace')
    return s


def pad_right(list_, min_rank, pad_value):
    return list_ + (max(0, min_rank - len(list_))) * [pad_value]


def tgz_compress(dir_name, file_name):
    with tarfile.open(file_name, 'w:gz') as tar:
        for file_ in os.listdir(dir_name):
            tar.add(dir_name + '/' + file_, file_)


def tgz_extract(file_name, dir_name):
    with tarfile.open(file_name, 'r:gz') as tar:
        tar.extractall(dir_name)
