# Copyright (c) 2018 The Khronos Group Inc.
# Copyright (c) 2018 Au-Zone Technologies Inc.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common.nnef_data import *
from tensorflow.core.framework import types_pb2

class TFDType(object):
    #https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/framework/dtypes.py
    datatype_dtype_mapping_tf_nnef = {
        'DT_HALF': nnef_float16,
        'DT_FLOAT': nnef_float32,
        'DT_DOUBLE': nnef_float64,
        'DT_INT32': nnef_int32,
        'DT_UINT8': nnef_uint8,
        'DT_UINT16': nnef_uint16,
        'DT_UINT32': nnef_uint32,
        'DT_UINT64': nnef_uint64,
        'DT_INT16': nnef_int16,
        'DT_INT8': nnef_int8,
        'DT_STRING': nnef_undefined,
        'DT_COMPLEX64': nnef_undefined,
        'DT_COMPLEX128': nnef_undefined,
        'DT_INT64': nnef_int64,
        'DT_BOOL': nnef_undefined,
        'DT_QINT8': nnef_qint8,
        'DT_QUINT8': nnef_quint8,
        'DT_QINT16': nnef_qint16,
        'DT_QUINT16': nnef_quint16,
        'DT_QINT32': nnef_qint32
    }

    #    DT_BFLOAT16: bfloat16,
    #    DT_RESOURCE: resource,
    #    DT_VARIANT: variant,
    #    DT_HALF_REF: float16_ref,
    # ...

    def __init__(self):
        return

    def _get_dtype_key(self, tensor_dtype_int):
        desc = types_pb2.DataType.DESCRIPTOR
        for (k,v) in desc.values_by_name.items():
            #print(k,v.number)
            if v.number == tensor_dtype_int:
                return k
        return None

    def get_NNEF_dtype(self, tensor_dtype):
        key = self._get_dtype_key(int(tensor_dtype))

        if key not in self.datatype_dtype_mapping_tf_nnef:
            raise TypeError("tf dtype not listed in datatype_dtype_mapping_tf_nnef: %s" % key)
        return self.datatype_dtype_mapping_tf_nnef[key]
