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

import typing

from nnef_tools.conversion.tensorflow.nnef_to_tf import Converter as NNEFToTFConverter
from nnef_tools.conversion.tensorflow.tf_to_nnef import Converter as TFToNNEFConverter
from nnef_tools.io.nnef.nnef_graph import NNEFGraph, NNEFOperation, NNEFTensor
from nnef_tools.io.tensorflow.tf_graph import TFGraph, TFOperation, TFTensor
from nnef_tools.io.tensorflow.tf_py.tf_py_definitions import OpProto, ArgProto

_StrOrStrs = typing.Union[str, typing.List[str], typing.Tuple[str, ...]]
_Type = typing.Any


class Signature(object):
    @staticmethod
    def _type_to_string(t):
        # type: (object)->str

        if t is Ellipsis:
            return '...'
        if isinstance(t, type):
            return t.__name__
        return str(t)

    @staticmethod
    def _unify_type(t):
        # type: (object)->object
        assert t in Signature._AllowedTypes or isinstance(t, Signature._CompoundType), "Type not allowed: {}".format(t)

        if t is list:
            return Signature.List[Signature.Any]
        if t is tuple:
            return Signature.Tuple[Signature.Any, ...]

        if t is Signature.Tensor:
            return Signature.Tensor[Signature.Any]
        if t is Signature.List:
            return Signature.List[Signature.Any]
        if t is Signature.Tuple:
            return Signature.Tuple[Signature.Any, ...]
        if t is Signature.Optional:
            return Signature.Optional[Signature.Any]

        if isinstance(t, Signature._CompoundType):
            return Signature._CompoundType(t.main,
                                           [Signature._unify_type(inner) for inner in t.inner])

        return t

    class _CompoundType(object):
        def __init__(self, main, inner):
            self.main = main
            self.inner = inner

        def __repr__(self):
            return "{}[{}]".format(Signature._type_to_string(self.main),
                                   ', '.join([Signature._type_to_string(t) for t in self.inner]))

    class _CompoundTypeCreator(object):
        def __init__(self, main, min_args=0, max_args=1000):
            self.main = main
            self.min_args = min_args
            self.max_args = max_args

        def __getitem__(self, item):
            items = item if isinstance(item, tuple) else (item,)
            assert self.min_args <= len(items) <= self.max_args
            return Signature._CompoundType(self.main, items)

        def __repr__(self):
            return "{}[?]".format(Signature._type_to_string(self.main))

    class _Tensor(object):
        pass

    class _Optional(object):
        pass

    class Any(object):
        pass

    Tensor = _CompoundTypeCreator(_Tensor, min_args=1, max_args=1)
    List = _CompoundTypeCreator(list, min_args=1, max_args=1)
    Tuple = _CompoundTypeCreator(tuple)
    Optional = _CompoundTypeCreator(_Optional, min_args=1, max_args=1)

    _AllowedTypes = {str, int, float, bool, list, tuple, Ellipsis,
                     _Tensor, _Optional, Any, Tensor, List, Tuple, Optional}

    def __init__(self, name, args):
        # type: (_StrOrStrs, typing.List[typing.Tuple[_StrOrStrs, _Type]])->None

        assert isinstance(args, (list, tuple))
        args = [(n, self._unify_type(t)) for n, t in args]
        names = name if isinstance(name, (list, tuple)) else [name]
        assert len(names) >= 1
        self.imports = []
        self.op_names = []
        for name in names:
            if '.' in name and not name.startswith('tf.') and not name.startswith('_tf.'):
                parts = name.split('.')
                self.imports.append("from {} import {}\n".format('.'.join(parts[:-1]), parts[-1]))
                name = parts[-1]
            else:
                self.imports.append("")
            self.op_names.append(name)

        arg_protos = []
        for arg_name, arg_type in args:
            arg_names = arg_name if isinstance(arg_name, (list, tuple)) else [arg_name]
            is_optional = isinstance(arg_type, self._CompoundType) and arg_type.main is self._Optional
            if is_optional:
                arg_type = arg_type.inner[0]
            is_array = isinstance(arg_type, self._CompoundType) and arg_type.main is list
            if is_array:
                arg_type = arg_type.inner[0]
            is_tensor = isinstance(arg_type, self._CompoundType) and arg_type.main is self._Tensor
            arg_protos.append(ArgProto(arg_names=arg_names,
                                       is_optional=is_optional,
                                       is_array=is_array,
                                       is_tensor=is_tensor))
        self.op_proto = OpProto(op_name=self.op_names[0], arg_protos=arg_protos)


__all__ = [
    'Signature',
    'TFToNNEFConverter',
    'NNEFToTFConverter',
    'NNEFGraph',
    'NNEFOperation',
    'NNEFTensor',
    'TFGraph',
    'TFOperation',
    'TFTensor',
]
