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

import os
import sys
import numpy as np

QUANT_CODE_FLOAT = 0x00
QUANT_CODE_INTEGER = 0x01
QUANT_CODE_LINEAR = 0x10
QUANT_CODE_LOGARITHMIC = 0x11


def _numpy_dtype_split(dtype):
    splits = {
        np.float16: (np.float, 16),
        np.float32: (np.float, 32),
        np.float64: (np.float, 64),
        np.int8: (np.int, 8),
        np.uint8: (np.uint, 8),
        np.int16: (np.int, 16),
        np.uint16: (np.uint, 16),
        np.int32: (np.int, 32),
        np.uint32: (np.uint, 32),
        np.int64: (np.int, 64),
        np.uint64: (np.uint, 64),
        np.bool_: (np.uint, 1),
        np.bool: (np.uint, 1),
    }
    split = splits.get(dtype.type)
    if split is None:
        raise TypeError('unsupported tensor dtype: ' + str(dtype))
    return split


def _numpy_dtype_make(code, bits, signed=False):
    if not code in [QUANT_CODE_FLOAT, QUANT_CODE_INTEGER, QUANT_CODE_LINEAR, QUANT_CODE_LOGARITHMIC]:
        raise ValueError('unsupported data type code: {}'.format(code))

    if code == QUANT_CODE_FLOAT and bits == 16:
        return np.float16
    elif code == QUANT_CODE_FLOAT and bits == 32:
        return np.float32
    elif code == QUANT_CODE_FLOAT and bits == 64:
        return np.float64
    elif code == QUANT_CODE_INTEGER and bits == 1:
        return np.bool
    elif bits == 8:
        return np.int8 if code == QUANT_CODE_INTEGER and signed else np.uint8
    elif bits == 16:
        return np.int16 if code == QUANT_CODE_INTEGER and signed else np.uint16
    elif bits == 32:
        return np.int32 if code == QUANT_CODE_INTEGER and signed else np.uint32
    elif bits == 64:
        return np.int64 if code == QUANT_CODE_INTEGER and signed else np.uint64
    else:
        raise ValueError('unsupported combination of item type code ({}) and bits per item ({})'.format(code, bits))


MaxTensorRank = 8;

def _rank_of(shape):
    rank = len(shape)
    while rank > 1 and shape[rank - 1] == 1:
        rank -= 1
    return rank


_is_little_endian = sys.byteorder == 'little'


def _tofile(data, file):
    if not _is_little_endian and data.dtype != np.uint8 and data.dtype != np.int8:
        data = data.byteswap()
    data.tofile(file)


def _fromfile(file, dtype, count):
    data = np.fromfile(file, dtype, count)
    if not _is_little_endian and data.dtype != np.uint8 and data.dtype != np.int8:
        data = data.byteswap()
    return data


def write_tensor(file, tensor, version=(1,0), quantization={}):
    if isinstance(file, str):
        raise ValueError('file parameter must be a file object not a file name')

    _tofile(np.asarray([0x4E, 0xEF, version[0], version[1]], dtype=np.uint8), file)

    dtype, bits = _numpy_dtype_split(tensor.dtype)
    qbits = quantization.get('bits-per-item')
    if qbits is not None and qbits != bits:
        raise ValueError('incompatible bits per item ({}) and tensor dtype ({})'.format(qbits, tensor.dtype))

    data_length = int((np.prod(tensor.shape) * bits + 7) // 8)
    _tofile(np.asarray([data_length, tensor.ndim], dtype=np.uint32), file)

    if tensor.ndim > MaxTensorRank:
        raise ValueError('tensor rank exceeds maximum possible value of {}'.format(MaxTensorRank))

    _tofile(np.asarray(tensor.shape, dtype=np.uint32), file)
    _tofile(np.asarray([0] * (MaxTensorRank - tensor.ndim), dtype=np.uint32), file)

    code = quantization.get('code', QUANT_CODE_FLOAT if dtype == np.float else QUANT_CODE_INTEGER)
    if (code == QUANT_CODE_FLOAT and dtype != np.float) or \
        (code == QUANT_CODE_INTEGER and dtype != np.int and dtype != np.uint) or \
        ((code == QUANT_CODE_LINEAR or code == QUANT_CODE_LOGARITHMIC) and dtype != np.uint):
        raise ValueError('incompatible quantization code ({}) and tensor dtype ({})'.format(code, tensor.dtype))

    _tofile(np.asarray([bits, code], dtype=np.uint32), file)

    params = [0] * 8
    if code == QUANT_CODE_LINEAR or code == QUANT_CODE_LOGARITHMIC:
        params[0] = quantization['min']
        params[1] = quantization['max']
    elif code == QUANT_CODE_INTEGER:
        params[0] = 0 if dtype == np.uint else 1
    elif code != QUANT_CODE_FLOAT:
        raise ValueError('unsupported item type code: {}'.format(code))

    _tofile(np.asarray(params, dtype=np.float32), file)
    _tofile(np.asarray([0] * 11, dtype=np.uint32), file)

    data = np.packbits(tensor) if bits == 1 else tensor
    _tofile(data, file)


def read_tensor(file):
    if isinstance(file, str):
        raise ValueError('file parameter must be a file object not a file name')

    [magic1, magic2, major, minor] = _fromfile(file, dtype=np.uint8, count=4)
    if magic1 != 0x4E or magic2 != 0xEF:
        raise ValueError('not a valid NNEF file')
    if major > 1 or minor > 0:
        raise ValueError('unsupported file version')

    [data_length, rank] = _fromfile(file, dtype=np.uint32, count=2)

    header_size = 128
    file_size = os.fstat(file.fileno()).st_size
    if file_size != header_size + data_length:
        raise ValueError('invalid tensor file; size does not match header info')

    if rank > MaxTensorRank:
        raise ValueError('tensor rank exceeds maximum possible value of {}'.format(MaxTensorRank))

    shape = _fromfile(file, dtype=np.uint32, count=MaxTensorRank)
    shape = shape[:rank]

    [bits, code] = _fromfile(file, dtype=np.uint32, count=2)
    params = _fromfile(file, dtype=np.float32, count=8)
    _padding = _fromfile(file, dtype=np.uint32, count=11)

    signed = params[0] != 0 if code == QUANT_CODE_INTEGER else False

    count = np.prod(shape)
    if bits == 1:
        byte_count = int((count + 7) // 8)
        data = _fromfile(file, dtype=np.uint8, count=byte_count)
        if len(data) != byte_count:
            raise ValueError('could not read tensor data')
        data = np.unpackbits(data).astype(np.bool)[:count]
    else:
        data = _fromfile(file, dtype=_numpy_dtype_make(code,bits,signed), count=count)
        if len(data) != count:
            raise ValueError('could not read tensor data')

    tensor = data.reshape(shape)

    quantization = {'op-code': code, 'bits-per-item': bits}
    if code == QUANT_CODE_LINEAR or code == QUANT_CODE_LOGARITHMIC:
        quantization['min'] = params[0]
        quantization['max'] = params[1]
    elif code != QUANT_CODE_FLOAT and code != QUANT_CODE_INTEGER:
        raise ValueError('unsupported item type code: {}'.format(code))

    return tensor, quantization



def _write_tensor_provisional(file, tensor, version=(1,0)):
    _tofile(np.asarray([0x4E, 0xEF, version[0], version[1]], dtype=np.uint8), file)

    header_length = 4 + 4 + (tensor.ndim + 1) * 4 + 4
    _tofile(np.asarray([header_length], dtype=np.uint32), file)

    _tofile(np.asarray([tensor.ndim], dtype=np.uint32), file)
    _tofile(np.asarray(tensor.shape, dtype=np.uint32), file)

    dtype, bits = _numpy_dtype_split(tensor.dtype)
    code = 0 if dtype == np.float else 3
    _tofile(np.asarray([code, bits], dtype=np.uint8), file)

    _tofile(np.asarray([0], dtype=np.uint16), file)

    _tofile(tensor, file)


def _read_tensor_provisional(file):
    [magic1, magic2, major, minor] = _fromfile(file, dtype=np.uint8, count=4)
    if magic1 != 0x4E or magic2 != 0xEF:
        raise ValueError('not a valid NNEF file')
    if major > 1 or minor > 0:
        raise ValueError('unsupported file version')

    [header_length] = _fromfile(file, dtype=np.uint32, count=1)

    [rank] = _fromfile(file, dtype=np.uint32, count=1)
    shape = _fromfile(file, dtype=np.uint32, count=rank)

    [code, bits] = _fromfile(file, dtype=np.uint8, count=2)
    [qlen] = _fromfile(file, dtype=np.uint16, count=1)

    assert(code == 0)
    assert(bits == 32)
    assert(qlen == 0)

    return _fromfile(file, dtype=np.float32, count=np.prod(shape)).reshape(shape)
