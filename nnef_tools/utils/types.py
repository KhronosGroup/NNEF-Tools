# Copyright (c) 2020 The Khronos Group Inc.
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

import sys
import numpy as np
from collections.abc import Sequence


# noinspection PyUnresolvedReferences
def as_str(s):
    if sys.version_info[0] >= 3:
        return s.decode('utf-8') if isinstance(s, bytes) else s
    else:
        return s.encode('utf-8') if isinstance(s, unicode) else s


PyTypeFromNumpyDtype = {
    np.float16: float,
    np.float32: float,
    np.float64: float,
    np.int8: int,
    np.int16: int,
    np.int32: int,
    np.int64: int,
    np.uint8: int,
    np.uint16: int,
    np.uint32: int,
    np.uint64: int,
    np.bool_: bool,
    np.str_: str,
}

PyTypeToNumpyDtype = {
    int: np.int32,
    float: np.float32,
    bool: np.bool_,
    str: np.str_,
}


_builtin_type = type


def cast(value, type):
    return _builtin_type(value)(cast(item, type) for item in value) if isinstance(value, Sequence) else type(value)


def from_numpy(array, type=None):
    if type is None:
        type = PyTypeFromNumpyDtype[array.dtype.type]
    return cast(array.tolist(), type)


def to_numpy(value, dtype=None):
    def _item(value):
        return _item(value[0]) if isinstance(value, Sequence) else value

    if dtype is None:
        dtype = PyTypeToNumpyDtype.get(type(_item(value)))
    return np.array(value, dtype=dtype)
