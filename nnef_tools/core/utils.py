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

import fnmatch
import inspect
import itertools
import os
import re
import shlex
import shutil
import sys
import tarfile
import typing
from collections import OrderedDict

import six

INT32_MAX = 2147483647


def nice_id(obj):
    return '@' + hex(id(obj))[2:]


def recursive_visit(arg, fun):
    # type: (typing.Any, typing.Callable[[typing.Any], None])->None

    if type(arg) is tuple or type(arg) is list:
        for item in arg:
            recursive_visit(item, fun)
    elif type(arg) is dict or type(arg) is OrderedDict:
        for item in six.itervalues(arg):
            recursive_visit(item, fun)
    else:
        fun(arg)


def recursive_transform(data, fun):
    # type: (typing.Any, typing.Callable[[typing.Any], typing.Any])->typing.Any
    if type(data) is dict or type(data) is OrderedDict:
        data2 = type(data)()
        for k, v in six.iteritems(data):
            w = recursive_transform(v, fun)
            data2[k] = w
    elif type(data) is list or type(data) is tuple:
        data2 = []
        for v in data:
            w = recursive_transform(v, fun)
            data2.append(w)
        data2 = type(data)(data2)
    else:
        data2 = fun(data)

    return data2


def recursive_copy(data):
    return recursive_transform(data, lambda x: x)


def recursive_any(data, pred):
    found = [False]

    def visit(data_):
        if pred(data_):
            found[0] = True

    recursive_visit(data, visit)
    return found[0]


def recursive_collect(data, pred=None):
    items = []

    def visit(x):
        if pred is None or pred(x):
            items.append(x)

    recursive_visit(data, visit)
    return items


def has_gt_1(data):
    return recursive_any(data, lambda x: x > 1)


def has_gt_0(data):
    return recursive_any(data, lambda x: x > 0)


def has_le_0(data):
    return recursive_any(data, lambda x: x <= 0)


def unique(arr, key=None):
    if key is None:
        def identity(x):
            return x

        key = identity
    s = set()
    arr2 = []
    for a in arr:
        k = key(a)
        if k not in s:
            s.add(k)
            arr2.append(a)
    return arr2


def is_unique(arr, key=None):
    return len(list(arr)) == len(unique(arr, key=key))


def flatten(x):
    def flatten_(x_, out_arr_):
        if isinstance(x_, (list, tuple)):
            for y in x_:
                flatten_(y, out_arr_)
        else:
            out_arr_.append(x_)

    out_arr = []
    flatten_(x, out_arr)
    return out_arr


def inverse_permutation(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def apply_permutation(list_, perm):
    return [list_[ind] for ind in perm]


COPY = object()


def get_or_copy(a, b):
    return a if a is not COPY else b


def first_set(a, b):
    return a if a is not None else b


class NameGenerator(object):
    def __init__(self, used_names=None):
        # type: (typing.Iterable[str])->None

        if used_names is None:
            used_names = []
        self.used_names = set(used_names)  # type: typing.Set[str]
        self.generated_names = set()  # type: typing.Set[str]
        self.counters = dict()  # type: typing.Dict[str, int]

    def is_available(self, name):
        # type: (str)->bool
        return name not in self.used_names

    def get_new_name(self, name):
        # type: (str)->str
        if name not in self.used_names:
            self.used_names.add(name)
            return name

        if name in self.generated_names:
            counter = int(name.split("_")[-1])
            self.counters[name] = max(self.counters.get(name, 1), counter)
            name = name[:name.rfind('_')]

        if name is self.counters:
            counter = self.counters[name]
        else:
            counter = 1

        name_candidate = name + "_" + str(counter)
        counter += 1
        while name_candidate in self.used_names:
            name_candidate = name + "_" + str(counter)
            counter += 1

        self.used_names.add(name_candidate)
        self.generated_names.add(name_candidate)
        self.counters[name] = counter
        return name_candidate


def listify(x):
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, list):
        return x
    return [x]


def dict_union(*dicts):
    res = {}
    for d in dicts:
        res.update(d)
    return res


def ordered_dict_union(*dicts):
    res = OrderedDict()
    for d in dicts:
        res.update(d)
    return res


def updated_dict(_D, **_U):
    return dict_union(recursive_copy(_D), _U)


def updated_dict_(d, u):
    return dict_union(recursive_copy(d), u)


def without(iterable, x):
    result = []
    for elem in iterable:
        if elem != x:
            result.append(elem)
    return result


def without_none(iterable):
    return without(iterable, None)


def zip_inverse(output_count, iterable):
    list_ = iterable if isinstance(iterable, list) else list(iterable)
    if list_:
        return tuple([list(a) for a in zip(*list_)])
    return tuple([[] for _ in range(output_count)])


def concat_lists(lists):
    return list(itertools.chain(*lists))


def get_dict_items(dict, *keys):
    return tuple(dict[key] for key in keys)


def key_value_swapped(dict):
    return {v: k for k, v in six.iteritems(dict)}


_identifier_regex = re.compile(r"^[^\d\W]\w*\Z", re.UNICODE)


def is_identifier(s):
    # type: (str)->bool
    return _identifier_regex.match(s) is not None


# noinspection PyUnresolvedReferences
def is_anystr(s):
    if sys.version_info[0] >= 3:
        return isinstance(s, (bytes, str))
    else:
        return isinstance(s, (str, unicode))


# noinspection PyUnresolvedReferences
def anystr_to_str(s):
    if sys.version_info[0] >= 3:
        if isinstance(s, bytes):
            return s.decode('utf-8')
        else:
            return s
    else:
        if isinstance(s, unicode):
            return s.encode('utf-8')
        else:
            return s


# noinspection PyUnresolvedReferences
def is_anyint(i):
    if sys.version_info[0] >= 3:
        return isinstance(i, int)
    else:
        return isinstance(i, (int, long))


def anyint_to_int(i):
    i = int(i)
    assert isinstance(i, int), "Value cannot be converted to int: {}".format(i)
    return i


class NNEFToolsException(Exception):
    pass


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


def batches(list, size):
    for i in range(0, len(list), size):
        yield list[i:i + size]


def compatible_shapes(orig_shape, shape):
    if orig_shape is None:
        return True
    if len(orig_shape) != len(shape):
        return False
    for o, s in zip(orig_shape, shape):
        if o != s and o != -1:
            return False
    return True


def product(list_, default=1):
    prod = default
    for val in list_:
        prod *= val
    return prod


def command_to_argv(command):
    return shlex.split(command.replace('\\', ' ').replace('\n', ' ').replace('\r', ' '))


def path_without_trailing_separator(path):
    while len(path) > 1 and (path[-1] == '/' or path[-1] == '\\'):
        path = path[:-1]
    return path


def rreplace(s, old, new, count=-1):
    li = s.rsplit(old, count)
    return new.join(li)


class EnvVars(object):
    def __init__(self, **kwargs):
        self.desired = kwargs
        self.save = dict()

    def __enter__(self):
        self.save.clear()
        for key, value in six.iteritems(self.desired):
            self.save[key] = os.environ.get(key, None)
            os.environ[key] = str(value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in six.iteritems(self.save):
            if value is None:
                del os.environ[key]
            else:
                os.environ[key] = value


def split_path(path):
    return path.replace('\\', '/').split('/')


# python2 does not have exist_ok
def makedirs(path, mode=0o777, exist_ok=False):
    if os.path.exists(path):
        if not exist_ok:
            raise RuntimeError("'{}' already exists".format(path))
        if not os.path.isdir(path):
            raise RuntimeError("'{}' exists but is not a directory".format(path))
    else:
        os.makedirs(path, mode)


# python2 interoperability
def unify_int_and_str_types(value):
    def unify(val):
        if is_anyint(val):
            return anyint_to_int(val)
        elif is_anystr(val):
            return anystr_to_str(val)
        else:
            return val

    return recursive_transform(value, unify)


def rmtree(path, exist_ok=False):
    if exist_ok:
        if os.path.exists(path):
            shutil.rmtree(path)
    else:
        shutil.rmtree(path)


def prefix_sum(iterable, zero=0):
    sum = zero
    l = []
    for elem in iterable:
        sum = sum + elem
        l.append(sum)
    return l


def recursive_glob(dir, glob):
    matches = []
    for root, dir_names, file_names in os.walk(dir):
        for filename in fnmatch.filter(file_names, glob):
            matches.append(os.path.join(root, filename))
    return matches


def tgz_compress(dir_path, file_path, compression_level=0):
    target_directory = os.path.dirname(file_path)
    if target_directory and not os.path.exists(target_directory):
        os.makedirs(target_directory)

    with tarfile.open(file_path, 'w:gz', compresslevel=compression_level) as tar:
        for file_ in os.listdir(dir_path):
            tar.add(dir_path + '/' + file_, file_)


def tgz_extract(file_path, dir_path):
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(dir_path)


def get_numbered_name(name, names_so_far):
    idx = 1
    while name + str(idx) in names_so_far:
        idx += 1
    return name + str(idx)
