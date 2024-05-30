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


class ItemType:
    FLOAT = 0
    UINT = 1
    QUINT = 2
    QINT = 3
    INT = 4
    BOOL = 5


def _numpy_dtype_split(dtype):
    splits = {
        np.float16: (ItemType.FLOAT, 16),
        np.float32: (ItemType.FLOAT, 32),
        np.float64: (ItemType.FLOAT, 64),
        np.int8: (ItemType.INT, 8),
        np.uint8: (ItemType.UINT, 8),
        np.int16: (ItemType.INT, 16),
        np.uint16: (ItemType.UINT, 16),
        np.int32: (ItemType.INT, 32),
        np.uint32: (ItemType.UINT, 32),
        np.int64: (ItemType.INT, 64),
        np.uint64: (ItemType.UINT, 64),
        np.bool_: (ItemType.BOOL, 1),
    }
    split = splits.get(dtype.type)
    if split is None:
        raise TypeError('unsupported tensor dtype: ' + str(dtype))
    return split


def _numpy_dtype_make(item_type, bits):
    dtypes = {
        (ItemType.FLOAT, 16): np.float16,
        (ItemType.FLOAT, 32): np.float32,
        (ItemType.FLOAT, 64): np.float64,
        (ItemType.INT, 8): np.int8,
        (ItemType.INT, 16): np.int16,
        (ItemType.INT, 32): np.int32,
        (ItemType.INT, 64): np.int64,
        (ItemType.UINT, 8): np.uint8,
        (ItemType.UINT, 16): np.uint16,
        (ItemType.UINT, 32): np.uint32,
        (ItemType.UINT, 64): np.uint64,
        (ItemType.QINT, 8): np.int8,
        (ItemType.QINT, 16): np.int16,
        (ItemType.QINT, 32): np.int32,
        (ItemType.QINT, 64): np.int64,
        (ItemType.QUINT, 8): np.uint8,
        (ItemType.QUINT, 16): np.uint16,
        (ItemType.QUINT, 32): np.uint32,
        (ItemType.QUINT, 64): np.uint64,
        (ItemType.BOOL, 1): np.bool_,
    }
    dtype = dtypes.get((item_type, bits))
    if dtype is None:
        raise ValueError('unsupported combination of item type ({}) and bits per item ({})'.format(item_type, bits))
    return dtype


MaxTensorRank = 8


def _rank_of(shape):
    rank = len(shape)
    while rank > 1 and shape[rank - 1] == 1:
        rank -= 1
    return rank


_is_little_endian = sys.byteorder == 'little'


def _tofile(data, file):
    if not _is_little_endian and data.dtype != np.uint8 and data.dtype != np.int8:
        data = data.byteswap()
    if file.seekable():
        data.tofile(file)
    else:
        file.write(data.tobytes())


def _fromfile(file, dtype, count):
    if file.seekable():
        data = np.fromfile(file, dtype, count)
    else:
        data = np.frombuffer(file.read(count * np.dtype(dtype).itemsize), dtype, count)

    if not _is_little_endian and data.dtype != np.uint8 and data.dtype != np.int8:
        data = data.byteswap()
    return data


def write_tensor(file, tensor, quantized=False, version=(1, 0)):
    if isinstance(file, str):
        raise ValueError('file parameter must be a file object not a file name')

    _tofile(np.asarray([0x4E, 0xEF, version[0], version[1]], dtype=np.uint8), file)

    item_type, bits = _numpy_dtype_split(tensor.dtype)
    if quantized:
        if item_type == ItemType.INT:
            item_type = ItemType.QINT
        elif item_type == ItemType.UINT:
            item_type = ItemType.QUINT
        else:
            raise ValueError("invalid tensor dtype '{}' for quantized tensor".format(tensor.dtype))

    count = int(np.prod(tensor.shape))
    data_length = (count + 7) // 8 if bits == 1 else count * (bits // 8)
    _tofile(np.asarray([data_length, tensor.ndim], dtype=np.uint32), file)

    if tensor.ndim > MaxTensorRank:
        raise ValueError('tensor rank exceeds maximum possible value of {}'.format(MaxTensorRank))

    _tofile(np.asarray(tensor.shape, dtype=np.uint32), file)
    _tofile(np.asarray([0] * (MaxTensorRank - tensor.ndim), dtype=np.uint32), file)

    _tofile(np.asarray([bits, item_type], dtype=np.uint32), file)
    _tofile(np.asarray([0] * 19, dtype=np.uint32), file)

    data = np.packbits(tensor) if bits == 1 else tensor
    _tofile(data, file)


def read_tensor(file, return_quantization=False):
    if isinstance(file, str):
        raise ValueError('file parameter must be a file object not a file name')

    [magic1, magic2, major, minor] = _fromfile(file, dtype=np.uint8, count=4)
    if magic1 != 0x4E or magic2 != 0xEF:
        raise ValueError('not a valid NNEF file')

    if major > 1 or minor > 0:
        raise ValueError('unsupported file version')

    [data_length, rank] = _fromfile(file, dtype=np.uint32, count=2)

    if file.seekable():
        header_size = 128
        file_size = os.fstat(file.fileno()).st_size
        if file_size != header_size + data_length:
            raise ValueError('invalid tensor file; size does not match header info')

    if rank > MaxTensorRank:
        raise ValueError('tensor rank exceeds maximum possible value of {}'.format(MaxTensorRank))

    shape = _fromfile(file, dtype=np.uint32, count=MaxTensorRank)
    shape = shape[:rank]

    [bits, item_type] = _fromfile(file, dtype=np.uint32, count=2)
    _reserved = _fromfile(file, dtype=np.uint32, count=19)
    if item_type == ItemType.UINT and _reserved[0] != 0:
        item_type = ItemType.INT

    quantized = item_type == ItemType.QINT or item_type == ItemType.QUINT
    count = int(np.prod(shape))
    if bits == 1:
        byte_count = int((count + 7) // 8)
        data = _fromfile(file, dtype=np.uint8, count=byte_count)
        if len(data) != byte_count:
            raise ValueError('could not read tensor data')
        data = np.unpackbits(data).astype(bool)[:count]
    else:
        data = _fromfile(file, dtype=_numpy_dtype_make(item_type, bits), count=count)
        if len(data) != count:
            raise ValueError('could not read tensor data')

    tensor = data.reshape(shape)

    return (tensor, quantized) if return_quantization else tensor


def _write_tensor_provisional(file, tensor, version=(1, 0)):
    _tofile(np.asarray([0x4E, 0xEF, version[0], version[1]], dtype=np.uint8), file)

    header_length = 4 + 4 + (tensor.ndim + 1) * 4 + 4
    _tofile(np.asarray([header_length], dtype=np.uint32), file)

    _tofile(np.asarray([tensor.ndim], dtype=np.uint32), file)
    _tofile(np.asarray(tensor.shape, dtype=np.uint32), file)

    dtype, bits = _numpy_dtype_split(tensor.dtype)
    _tofile(np.asarray([dtype, bits], dtype=np.uint8), file)

    _tofile(np.asarray([0], dtype=np.uint16), file)

    _tofile(tensor, file)


def _read_tensor_provisional(file):
    [magic1, magic2, major, minor] = _fromfile(file, dtype=np.uint8, count=4)
    if magic1 != 0x4E or magic2 != 0xEF:
        raise ValueError('not a valid NNEF file')
    if major > 1 or minor > 0:
        raise ValueError('unsupported file version')

    [_header_length] = _fromfile(file, dtype=np.uint32, count=1)

    [rank] = _fromfile(file, dtype=np.uint32, count=1)
    shape = _fromfile(file, dtype=np.uint32, count=rank)

    [code, bits] = _fromfile(file, dtype=np.uint8, count=2)
    [qlen] = _fromfile(file, dtype=np.uint16, count=1)

    assert (code == 0)
    assert (bits == 32)
    assert (qlen == 0)

    return _fromfile(file, dtype=np.float32, count=int(np.prod(shape))).reshape(shape)
