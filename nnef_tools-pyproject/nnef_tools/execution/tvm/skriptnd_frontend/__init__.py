# Copyright (c) 2017-2025 The Khronos Group Inc.
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

import logging

from skriptnd import Dtype

LOGGING_LEVEL = logging.WARNING


def convert_dtype(dtype: Dtype) -> str:
    """Convert a SkriptND dtype to string dtype repr"""
    dtype_map = {
        Dtype.Real: "float32",
        Dtype.Int: "int32",
        Dtype.Bool: "bool",
        Dtype.Str: "str",
    }
    return dtype_map[dtype]


class ConverterError(Exception):
    def __init__(self, op: str, *args, **kwargs):
        super().__init__(f"Error while trying to covnert '{op}'")
        self.op = op
        self.args = args
        self.kwargs = kwargs
        self.warn_msg = ""

    # keep pickling support
    def __reduce__(self):
        return (self.__class__, (self.op, *self.args), self.__dict__)


from .graph_builder import from_skriptnd, is_atomic
