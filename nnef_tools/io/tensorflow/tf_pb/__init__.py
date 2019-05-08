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

try:
    import tensorflow

    has_tensorflow_installed = True
except ImportError:
    has_tensorflow_installed = False

try:
    if not has_tensorflow_installed:
        sys.path.insert(0, os.path.dirname(__file__))
    from tensorflow.core.framework.graph_pb2 import GraphDef
    from tensorflow.core.framework.node_def_pb2 import NodeDef
    from tensorflow.core.framework.attr_value_pb2 import AttrValue
    from tensorflow.core.framework.types_pb2 import DataType
    from tensorflow.core.framework.tensor_pb2 import TensorProto
    from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto

    __all__ = [
        'GraphDef',
        'NodeDef',
        'AttrValue',
        'DataType',
        'TensorProto',
        'TensorShapeProto',
    ]
finally:
    if not has_tensorflow_installed:
        sys.path.pop(0)
