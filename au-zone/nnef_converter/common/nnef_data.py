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

from __future__ import division

import io
import numpy as np
import collections
from enum import Enum
import os
import os.path
import struct
import sys
import errno


MAGIC_NUMBER = b'\x4E\xEF'

VERSION_MAJOR = 1
VERSION_MINOR = 0

class NNEFDType(object):
    _nnef_base_types = {
        'FLOAT': 0,
        'QUANTIZED': 1,
        'SIGNED_INTEGER': 2,
        'UNSIGNED_INTEGER': 3,
        'UNSUPPORTED': -1
    }

    def _assign_numpy_dt(self):
        np_dtype=None
        if self._nnef_base_dtype_key is "FLOAT":
            if self._bit_width == 16:
                np_dtype = np.dtype(np.float16, align=True)
            elif self._bit_width == 32:
                np_dtype = np.dtype(np.float32, align=True)
            elif self._bit_width == 64:
                np_dtype = np.dtype(np.float64, align=True)
        elif self._nnef_base_dtype_key is "SIGNED_INTEGER":
            if self._bit_width == 8:
                np_dtype = np.dtype(np.int8, align=True)
            elif self._bit_width == 16:
                np_dtype = np.dtype(np.int16, align=True)
            elif self._bit_width == 32:
                np_dtype = np.dtype(np.int32, align=True)
            elif self._bit_width == 64:
                np_dtype = np.dtype(np.int64, align=True)
        elif self._nnef_base_dtype_key is "UNSIGNED_INTEGER":
            if self._bit_width == 8:
                np_dtype = np.dtype(np.uint8, align=True)
            elif self._bit_width == 16:
                np_dtype = np.dtype(np.uint16, align=True)
            elif self._bit_width == 32:
                np_dtype = np.dtype(np.uint32, align=True)
            elif self._bit_width == 64:
                np_dtype = np.dtype(np.uint64, align=True)
        elif self._nnef_base_dtype_key == "QUANTIZED":
            # Todo: support quantized
            #np_dtype = np.dtype(np.float32, align=True)
            np_dtype = None
        elif self._nnef_base_dtype_key == "UNSUPPORTED":
            #?
            np_dtype = None
        return np_dtype

    def __init__(self, nnef_base_dtype, bit_width):
        #if nnef_base_dtype not in self._nnef_base_dtypes:
        #    raise TypeError("nnef_base_dtype is not valid: %s" % nnef_base_dtype)

        if isinstance(nnef_base_dtype, int):
            self._nnef_base_dtype_key = None
            for k,v in self._nnef_base_types.items():
                if v == nnef_base_dtype:
                    self._nnef_base_dtype_key = k
            self._nnef_base_dtype_int = nnef_base_dtype
        elif isinstance(nnef_base_dtype, str):
            self._nnef_base_dtype_key = nnef_base_dtype
            self._nnef_base_dtype_int = self._nnef_base_types[nnef_base_dtype]
        else:
            raise TypeError("nnef_base_dtype must either be str or int (%s)" % type(nnef_base_dtype))

        self._bit_width = bit_width
        self._np_dtype = self._assign_numpy_dt()

    def get_numpy_dtype(self):
        return self._np_dtype

    def get_nnef_base_type_value(self):
        return self._nnef_base_dtype_int

    def get_nnef_bit_width(self):
        return self._bit_width

_np_qint8 = np.dtype([("qint8", np.int8, 1)])
_np_quint8 = np.dtype([("quint8", np.uint8, 1)])
_np_qint16 = np.dtype([("qint16", np.int16, 1)])
_np_quint16 = np.dtype([("quint16", np.uint16, 1)])
_np_qint32 = np.dtype([("qint32", np.int32, 1)])

nnef_float16 = NNEFDType('FLOAT', 16)
nnef_float32 = NNEFDType('FLOAT', 32)
nnef_float64 = NNEFDType('FLOAT', 64)
nnef_int32 = NNEFDType('SIGNED_INTEGER', 32)
nnef_uint8 = NNEFDType('UNSIGNED_INTEGER', 8)
nnef_uint16 = NNEFDType('UNSIGNED_INTEGER', 16)
nnef_uint32 = NNEFDType('UNSIGNED_INTEGER', 32)
nnef_uint64 = NNEFDType('UNSIGNED_INTEGER', 64)
nnef_int16 = NNEFDType('SIGNED_INTEGER', 16)
nnef_int8 = NNEFDType('SIGNED_INTEGER', 8)
nnef_int64 = NNEFDType('SIGNED_INTEGER', 64)
nnef_qint8 = NNEFDType('QUANTIZED', 8)
nnef_quint8 = NNEFDType('QUANTIZED', 8)
nnef_qint16 = NNEFDType('QUANTIZED', 16)
nnef_quint16 = NNEFDType('QUANTIZED', 16)
nnef_qint32 = NNEFDType('QUANTIZED', 32)
nnef_undefined = NNEFDType('UNSUPPORTED', -1)


class Header:
    def set_int_property(self, src, nb_bytes):
        assert isinstance(src, bytes) or isinstance(src, int) and isinstance(nb_bytes, int), "set_int_property src must be of bytes type or int type"
        if isinstance(src, bytes):
            fmt = {1: '<B', 2: '<H', 4: '<I', 8: '<Q'}
            dst_bytes = src
            dst_int = struct.unpack(fmt[len(src)], src)[0]
        else:
            fmt = {1: '<B', 2: '<H', 4: '<I', 8: '<Q'}
            assert nb_bytes in fmt, 'set_int_property called with invalid byte length: %d' % nb_bytes
            dst_bytes = struct.pack(fmt[nb_bytes], src)
            dst_int = src
        return dst_int, dst_bytes

    def set_str_property(self, src):
        assert isinstance(src, bytes) or isinstance(src,str), "set_str_property src must be of bytes type or str type"
        if isinstance(src, bytes):
            dst_bytes = src
            dst_str = src.decode('utf-8')
        else:
            dst_bytes = src.encode("utf-8")
            dst_str = src
        return dst_str, dst_bytes, len(dst_bytes)

    def get_magic_number(self):
        return self._b_magic_number

    def set_magic_number(self, value):
        assert isinstance(value, bytes), "magic_number must be of bytes type"
        self._b_magic_number = value

    def get_version_major(self):
        return self._version_major, self._b_version_major

    def set_version_major(self, value):
        self._version_major, self._b_version_major = self.set_int_property(value, self.header_field_sizes['version_major'])

    def get_version_minor(self):
        return self._version_minor, self._b_version_minor

    def set_version_minor(self, value):
        self._version_minor, self._b_version_minor = self.set_int_property(value, self.header_field_sizes['version_minor'])

    def get_data_offset(self):
        return self._data_offset, self._b_data_offset

    def set_data_offset(self, value):
        self._data_offset, self._b_data_offset = self.set_int_property(value, self.header_field_sizes['data_offset'])

    def get_tensor_rank(self):
        return self._tensor_rank, self._b_tensor_rank

    def set_tensor_rank(self, value):
        self._tensor_rank, self._b_tensor_rank = self.set_int_property(value, self.header_field_sizes['tensor_rank'])

    def get_tensor_dimensions(self):
        return self._tensor_dimensions, self._b_tensor_dimensions

    def set_tensor_dimensions(self, value):
        if not self._modified:
            self._modified = True
        # else:
        #     return
        assert isinstance(value, list) or isinstance(value,bytes), "set tensor dimensions receives list or int type values."

        if isinstance(value, list):
            self._tensor_dimensions = value
            self._b_tensor_dimensions = b''
            for dim in value:
                self._b_tensor_dimensions += struct.pack('<I', dim)
        else:
            self._tensor_dimensions.append(struct.unpack('<I', value)[0]) #int.from_bytes(value, byteorder='little', signed=False))
            self._b_tensor_dimensions += value

    def get_data_type(self):
        return self._data_type, self._b_data_type

    def set_data_type(self, value):
        self._data_type, self._b_data_type = self.set_int_property(value, self.header_field_sizes['data_type'])

    def get_bit_width(self):
        return self._bit_width, self._b_bit_width

    def set_bit_width(self, value):
        self._bit_width, self._b_bit_width = self.set_int_property(value, self.header_field_sizes['bit_width'])

    def get_len_quantize_algo_string(self):
        return self._len_quantize_algo_string, self._b_len_quantize_algo_string

    def set_len_quantize_algo_string(self, value):
        self._len_quantize_algo_string, self._b_len_quantize_algo_string = self.set_int_property(value, self.header_field_sizes['len_quantize_algo_string'])

    def get_quantize_algo_string(self):
        return self._quantize_algo_string, self._b_quantize_algo_string

    def set_quantize_algo_string(self, value):
        self._quantize_algo_string, self._b_quantize_algo_string, algo_string_len = self.set_str_property(value)
        self.set_len_quantize_algo_string(algo_string_len)

    def __init__(self):
        self._b_magic_number = b''
        self._b_version_major= b''
        self._b_version_minor= b''
        self._b_data_offset = b''
        self._b_tensor_rank = b''
        self._b_tensor_dimensions = b''
        self._b_data_type = b''
        self._b_bit_width = b''
        self._b_len_quantize_algo_string = b''
        self._b_quantize_algo_string= b''
        self._magic_number = ''
        self._version_major= ''
        self._version_minor= ''
        self._data_offset = ''
        self._tensor_rank = ''
        self._tensor_dimensions = []
        self._data_type = ''
        self._bit_width = ''
        self._len_quantize_algo_string = ''
        self._quantize_algo_string= ''
        self._modified = False

        self.header_field_sizes = collections.OrderedDict()
        self.header_field_sizes['magic_number']=2
        self.header_field_sizes['version_major']=1
        self.header_field_sizes['version_minor']=1
        self.header_field_sizes['data_offset']=4
        self.header_field_sizes['tensor_rank']=4
        #tensor dimensions
        self.header_field_sizes['data_type']=1
        self.header_field_sizes['bit_width']=1
        self.header_field_sizes['len_quantize_algo_string']=2
        #quant string

        self.set_magic_number(MAGIC_NUMBER)
        self.set_version_major(VERSION_MAJOR)
        self.set_version_minor(VERSION_MINOR)
        return

    def set_content(self, tensor_shape, nnef_dtype, quantize_algo_string=""):
        assert isinstance(tensor_shape, list), "Header requires tensor_shape to be a list"
        assert len(tensor_shape) > 0, "Header requires tensor_shape to be a list with at least one element"
        assert isinstance(nnef_dtype, NNEFDType), "Header requires nnef_dtype to be of type NNEFDType"
        assert isinstance(quantize_algo_string, str), "Header requires quantize_algo_string to be of type str"

        self.set_tensor_rank(len(tensor_shape))
        self.set_tensor_dimensions(tensor_shape)
        self.set_data_type(nnef_dtype.get_nnef_base_type_value())   #data_type.value
        self.set_bit_width(nnef_dtype.get_nnef_bit_width())         #bit_width
        self.set_quantize_algo_string(quantize_algo_string)

        # Setting the offset to '0' temporarily
        self.set_data_offset(struct.pack('<I', 0))
        # Running it a first time to get the data offset
        self.header_size, self.binary_header = self.to_binary()
        self.set_data_offset(struct.pack('<I', self.header_size))
        # Running it a second time to include proper data offset in header
        self.header_size, self.binary_header = self.to_binary()

        #print("Created header of size: ", self.header_size)

    def to_binary(self):
        # todo: handle OSError

        # print(int.from_bytes(self._b_bit_width, byteorder='little', signed=False))

        with io.BytesIO() as f:
            f.write(self._b_magic_number)
            f.write(self._b_version_major)
            f.write(self._b_version_minor)
            f.write(self._b_data_offset)
            f.write(self._b_tensor_rank)
            f.write(self._b_tensor_dimensions)
            f.write(self._b_data_type)
            f.write(self._b_bit_width)
            f.write(self._b_len_quantize_algo_string)
            f.write(self._b_quantize_algo_string)
            return len(f.getvalue()), f.getvalue()

    def read_content(self, f):
        try:
            rank=None
            for k, nb_bytes in self.header_field_sizes.items():
                #print(k, nb_bytes)
                read_value = f.read(nb_bytes)

                assert hasattr(self, k) is not None or k is not magic_number, "Header has incomple set of attributes"
                func = getattr(self, "set_"+k)
                func(read_value)

                # #debug
                # func = getattr(self, "get_"+k)
                # print("%s:"%(k),func()[0])

                if k is 'magic_number':
                    assert read_value == MAGIC_NUMBER,"File does not contain NNEF Tensor File Format MAGIC_NUMBER."
                elif k is 'version_major':
                    assert self.get_version_major()[0] <= VERSION_MAJOR, "VERSION_MAJOR of the .dat file is more recent than the one supported by the script (%s vs %s)"%(read_value, VERSION_MAJOR)
                elif k is 'version_minor':
                    assert self.get_version_minor()[0] <= VERSION_MINOR, "VERSION_MINOR of the .dat file is more recent than the one supported by the script (%s vs %s)"%(read_value, VERSION_MINOR)
                elif k is 'tensor_rank':
                    for i in range(self.get_tensor_rank()[0]):
                        read_value = f.read(4)
                        self.set_tensor_dimensions(read_value)
                elif k is 'len_quantize_algo_string' and self.get_len_quantize_algo_string()[0]>0:
                        read_value = f.read(self.get_len_quantize_algo_string()[0])
                        self.set_quantize_algo_string(read_value)
        except OSError as err:
            print("OS error: {0}".format(err))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def write_content(self, f):
        try:
            h_size, h_data = self.to_binary()
            f.write(h_data)
        except OSError as err:
            print("OS error: {0}".format(err))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

class NNEFTensor:
    def __init__(self):
        self._modified = False
        self._np_array = None
        self._data_type = None

    def get_array(self):
        return self._np_array, self._data_type

    def set_array(self, np_array, dt = np.dtype(np.float32, align=True), override = False):
        if not self._modified or override:
            dt.newbyteorder('<')
            self._np_array = np_array.astype(dt)
            self._data_type = dt
            self._modified = True
        return self._np_array

    def set_array_from_bytes(self, bytes_buffer, shape, dt = np.dtype(np.float32, align=True)):
        np_tensor_data = np.frombuffer(bytes_buffer, dtype = dt)
        self._np_array = np.reshape(np_tensor_data, shape)
        self._data_type = dt
        return self._np_array

    def read_content(self, file, offset, dt = np.dtype(np.float32, align=True)):
        try:
            dt.newbyteorder('<')
            file.seek(offset, os.SEEK_SET)
            self._np_array = np.fromfile(file, dtype=dt)
            self._data_type = dt
        except OSError as err:
            print("OS error: {0}".format(err))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def write_content(self, file, offset):
        assert self._np_array is not None, "Undefined tensor data while writing content to file"
        assert self._data_type is not None, "Undefined data type while writing content to file"

        try:
            file.seek(offset, os.SEEK_SET)
            self._np_array.tofile(file)
        except OSError as err:
            print("OS error: {0}".format(err))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    @property
    def shape(self):
        return list(self._np_array.shape)

    @property
    def datatype(self):
        return NNEFDType(self._data_type, 32) #TODO review bit width!!!!


class TensorDataFile(object):

    def __init__(self, tensor_content = None, nnef_dtype = None, nnef_shape = None):
        self.path = None
        self.header = None
        self.tensor = None
        if tensor_content == None and nnef_dtype == None and nnef_shape == None:
            return

        self.set_header(nnef_shape, nnef_dtype)
        nnef_tensor = NNEFTensor()
        nnef_tensor.set_array_from_bytes(tensor_content, nnef_shape, nnef_dtype.get_numpy_dtype())
        self.set_data(nnef_tensor)

    def set_header(self, tensor_shape, nnef_dtype, quantize_algo_string=""):
        assert isinstance(nnef_dtype, NNEFDType), "nnef_dtype must be of type NNEFDType."
        assert isinstance(tensor_shape, list), "tensor_shape must be a list."
        self.header = Header()
        self.header.set_content(tensor_shape, nnef_dtype, quantize_algo_string)

    def set_data(self, nnef_tensor):
        self.tensor = nnef_tensor

    def get_data(self):
        return self.tensor

    def get_numpy_array(self):
        return self.tensor.get_array()[0]

    def read_from_disk(self, path):

        assert isinstance(path, str), "TensorDataFile requires path of type str."
        with open(path, 'rb') as f:
            self.header = Header()
            self.header.read_content(f) # change that!

            offset = self.header.get_data_offset()[0]
            #print("Reading")
            nnef_data_type_int = self.header.get_data_type()[0]
            #print(nnef_data_type_int)
            nnef_bit_width = self.header.get_bit_width()[0]
            #print(nnef_bit_width)
            nnef_dtype = NNEFDType(nnef_data_type_int, nnef_bit_width)
            #print(nnef_dtype)
            np_dt = nnef_dtype.get_numpy_dtype()
            #print(np_dt)
            #input()

            assert np_dt is not None, "Unknown np datatype!!!"

            self.tensor = NNEFTensor()
            self.tensor.read_content(f, offset, np_dt)

            return self.tensor.get_array()[0]

    def mkdir_p(self, path):
        try:
            if(path != ''):
                os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise
    def safe_open(self, path, mode):
        ''' Open "path" for writing, creating any parent directories as needed.
        '''
        self.mkdir_p(os.path.dirname(path))
        return open(path, mode)

    def write_to_disk(self, path):
        assert isinstance(path, str), "TensorDataFile requires path of type str."
        assert isinstance(self.tensor, NNEFTensor), "TensorDataFile requires a NNEFTensor to be defined before writing to disk."
        assert isinstance(self.header, Header), "TensorDataFile requires a Header to be defined before writing to disk."

        #print("Writing")
        #nnef_data_type_int = self.header.get_data_type()[0]
        #print(nnef_data_type_int)
        #nnef_bit_width = self.header.get_bit_width()[0]
        #print(nnef_bit_width)
        #nnef_dtype = NNEFDType(nnef_data_type_int, nnef_bit_width)
        #print(nnef_dtype)
        #np_dt = nnef_dtype.get_numpy_dtype()
        #print(np_dt)
        #input()

        with self.safe_open(path, 'wb+') as f:
            self.header.write_content(f)
            self.tensor.write_content(f, self.header.get_data_offset()[0])


