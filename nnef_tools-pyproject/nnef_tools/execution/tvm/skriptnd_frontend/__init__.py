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


# atomic operations that can be handled by high level Relax ops
# Callback signature: (name: str, dtype: Dtype, attribs: dict[str, any], shape: list[tuple[int]]) -> bool

def lrn_callback(name: str, dtype: Dtype, attribs: dict[str, any], shape: list[tuple[int]]) -> bool:
    if attribs.get("axes").max_size != 1 or attribs.get("size").max_size != 1:
        logging.log(LOGGING_LEVEL, f"LRN {name} has more than 1 axis or size")
        return False
    return True


def avg_pool_callback(name: str, dtype: Dtype, attribs: dict[str, any], shape: list[tuple[int]]) -> bool:
    if any(dim in attribs.get("axes").items for dim in [0, 1]):
        logging.log(LOGGING_LEVEL, f"Average Pool {name} pools over channels or batch")
        return False
    return True


_atomics = {
    "image.area_downsample": True,
    "image.resize": True,
    "image.rescale": True,
    # "layout.space_to_batch": True, # has no real improvement over composed
    "math.mean_reduce": True,
    "nn.separable_conv": True,
    "nn.avg_pool": avg_pool_callback,
    "nn.batch_norm": True,
    "nn.local_response_norm": lrn_callback,
}

from .graph_builder import from_skriptnd

# limit import
__all__ = ["from_skriptnd", "ConverterError", "convert_dtype", "LOGGING_LEVEL", "_atomics"]
