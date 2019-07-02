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

import os
import sys

from nnef_tools.core import utils

try:
    import caffe2.proto.caffe2_pb2

    caffe2_is_installed = True
except ImportError:
    caffe2_is_installed = False

try:
    if not caffe2_is_installed:
        sys.path.insert(0, os.path.dirname(__file__))
    from caffe2.proto.caffe2_pb2 import TensorProto, NetDef, OperatorDef, Argument, DeviceOption


    def fixstr(s):
        return utils.anystr_to_str(s) if s is not None else None

    def dtype_name_to_id(name):
        return TensorProto.DataType.Value(name)

    def dtype_id_to_name(dtype_int):
        return fixstr(TensorProto.DataType.Name(dtype_int))


    __all__ = [
        'TensorProto',
        'NetDef',
        'OperatorDef',
        'Argument',
        'DeviceOption',
        'dtype_name_to_id',
        'dtype_id_to_name',
    ]
finally:
    if not caffe2_is_installed:
        sys.path.pop(0)
