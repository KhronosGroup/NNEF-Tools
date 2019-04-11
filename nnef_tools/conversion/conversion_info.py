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

import json
import os
import typing

from nnef_tools.conversion.transforms import *


class _Empty(object):
    pass


class _CustomEncoder(json.JSONEncoder):

    def default(self, o):
        d = dict(o.__dict__)
        d["__class__"] = o.__class__.__name__
        return d


class _CustomDecoder(object):
    def __init__(self, class_by_name):
        self._class_by_name = class_by_name

    def __call__(self, d):
        if "__class__" in d:
            o = _Empty()
            d2 = dict(d)
            del d2["__class__"]
            o.__dict__.update(d2)
            o.__class__ = self._class_by_name[d["__class__"]]
            return o
        return d


def _json_dump(obj, file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    with open(file_name, 'w') as f:
        json.dump(obj, f, indent=4, sort_keys=True, cls=_CustomEncoder)


def _json_load(file_name, classes=None):
    if classes is None:
        classes = []
    with open(file_name, 'r') as f:
        return json.load(f, object_hook=_CustomDecoder({class_.__name__: class_ for class_ in classes}))


class TensorInfo(object):
    def __init__(self, source_name, target_name, target_shape, target_dtype,
                 is_input=False, is_output=False, is_variable=False, transforms=None):
        if transforms is None:
            transforms = []
        self.source_name = source_name
        self.target_name = target_name
        self.is_input = is_input
        self.is_output = is_output
        self.is_variable = is_variable
        self.transforms = transforms  # type: typing.List[Transform]
        self.target_dtype = target_dtype
        self.target_shape = target_shape

    def copy(self):
        return TensorInfo(source_name=self.source_name,
                          target_name=self.target_name,
                          is_input=self.is_input,
                          is_output=self.is_output,
                          is_variable=self.is_variable,
                          transforms=[t.copy() for t in self.transforms],
                          target_shape=self.target_shape,
                          target_dtype=self.target_dtype)


class ConversionInfo(object):
    def __init__(self, tensors):
        self.tensors = tensors  # type: typing.List[TensorInfo]

    def copy(self):
        return ConversionInfo([t.copy() for t in self.tensors])


def dump(conversion_info, file_name):
    _json_dump(conversion_info, file_name)


def load(file_name):
    return _json_load(file_name, [ConversionInfo, TensorInfo, Transpose, Squeeze, Unsqueeze, Reshape])


def _compose(info1, info2):
    # type: (ConversionInfo, ConversionInfo)->ConversionInfo
    tensor_info1_by_target_name = {info.target_name: info for info in info1.tensors}
    tensor_infos = []
    for right in info2.tensors:
        if right.source_name in tensor_info1_by_target_name:
            left = tensor_info1_by_target_name[right.source_name]  # type: TensorInfo
            del tensor_info1_by_target_name[right.source_name]
            result = right.copy()
            result.source_name = left.source_name
            result.transforms = left.transforms + right.transforms
            tensor_infos.append(result)

    return ConversionInfo(tensor_infos)


def compose(info1, info2, *more_infos):
    infos = (info1, info2) + more_infos
    infos = [info for info in infos if info is not None]

    if not infos:
        return None

    left = infos[0]
    for right in infos[1:]:
        left = _compose(left, right)

    return left
