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


import _nnef
import numpy as np
from collections import OrderedDict


Tensor = _nnef.Tensor
error = _nnef.error

StandardOperations = [
    "external",
    "variable",
    "constant",
    "add",
    "sub",
    "mul",
    "div",
    "min",
    "max",
    "pow",
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "ne",
    "and",
    "or",
    "min",
    "max",
    "idn",
    "neg",
    "rcp",
    "exp",
    "log",
    "abs",
    "sign",
    "floor",
    "ceil",
    "round",
    "sqr",
    "sqrt",
    "rsqr",
    "rsqrt",
    "not",
    "log2",
    "relu",
    "sigmoid",
    "tanh",
    "elu",
    "softabs",
    "softmax",
    "softplus",
    "leaky_relu",
    "linear_quantize",
    "logarithmic_quantize",
    "binary_quantize",
    "ternary_quantize",
    "conv",
    "box",
    "sample",
    "separable_conv",
    "planewise_conv",
    "deconv",
    "debox",
    "desample",
    "separable_deconv",
    "planewise_deconv",
    "max_pool",
    "argmax_pool",
    "max_pool_with_index",
    "avg_pool",
    "rms_pool",
    "sum_reduce",
    "min_reduce",
    "max_reduce",
    "mean_reduce",
    "moments",
    "nearest_downsample",
    "area_downsample",
    "nearest_upsample",
    "multilinear_upsample",
    "local_response_normalization",
    "local_mean_normalization",
    "local_variance_normalization",
    "local_contrast_normalization",
    "l1_normalization",
    "l2_normalization",
    "layer_normalization",
    "divisive_normalization",
    "batch_normalization",
    "reshape",
    "transpose",
    "split",
    "concat",
    "select",
    "matmul",
    "linear",
    "update",
    "softmax",
    "copy_n",
    "add_n",
]


def parse_file( string, flat=False, layers=False, atomics=StandardOperations, shape_prop={} ):
    return _nnef.parse_file(string, flat, layers, atomics, shape_prop)


def parse_string( string, flat=False, layers=False, atomics=StandardOperations, shape_prop={} ):
    return _nnef.parse_string(string, flat, layers, atomics, shape_prop)


def format_argument( value ):
    if isinstance(value, (list, tuple)):
        string = '[' if isinstance(value, list) else '('
        for idx, item in enumerate(value):
            if idx != 0:
                string += ', '
            string += format_argument(item)
        string += ']' if isinstance(value, list) else ')'
        return string
    elif isinstance(value, Tensor):
        return value
    elif isinstance(value, str):
        return "'" + value + "'"
    elif isinstance(value, bool):
        return 'true' if value else 'false'
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        raise TypeError('arguments must be of type int, float, str, nnef.Tensor or list/tuple of such, found: ' + str(type(value)))


def format_invocation( name, args, kwargs, results=[] ):
    string = str()

    for (idx, result) in enumerate(results):
        if idx != 0:
            string += ', '
        string += result

    if len(results) != 0:
        string += ' = '

    string += name
    string += '('

    idx = 0

    for arg in args:
        if idx != 0:
            string += ', '
        string += format_argument(arg)
        idx += 1

    for (key, value) in kwargs.items():
        if idx != 0:
            string += ', '
        string += key
        string += ' = '
        string += format_argument(value)
        idx += 1

    string += ')'

    return string


def format_params( params ):
    string = '( '
    if isinstance(params, dict):
        for idx, (param, typespec) in enumerate(params.items()):
            if idx != 0:
                string += ', '
            string += param + ': ' + typespec
    else:
        for idx, param in enumerate(params):
            if idx != 0:
                string += ', '
            string += param
    string += ' )'
    return string


def format_declaration( name, inputs, outputs ):
    return name + format_params(inputs) + ' -> ' + format_params(outputs)


def _tensor_header_length(rank, qlen):
    return 4 + 4 + (rank + 1) * 4 + 4 + qlen


def _tensor_dtype_bits(dtype):
    bits = {
        np.float32: 32,
        np.float64: 64,
        np.int8: 8,
        np.uint8: 8,
        np.int16: 16,
        np.uint16: 16,
        np.int32: 32,
        np.uint32: 32,
        np.int64: 64,
        np.uint64: 64,
    }.get(dtype.type)
    if bits is None:
        raise TypeError('unsupported tensor dtype (bit width): ' + str(dtype))
    return bits


def _tensor_dtype_code(dtype, quantization):
    if np.issubdtype(dtype, np.float) and quantization == '':
        return 0
    elif np.issubdtype(dtype, np.signedinteger) and quantization == '':
        return 2
    elif np.issubdtype(dtype, np.unsignedinteger):
        return 1 if quantization != '' else 3
    else:
        if quantization == '':
            raise TypeError('unsupported tensor dtype: ' + str(dtype))
        else:
            raise TypeError('unsupported tensor dtype with quantization: ' + str(dtype))


def _numpy_dtype(code, bits):
    if code == 0 and bits == 32:
        return np.float32
    elif code == 0 and bits == 64:
        return np.float64
    elif code == 2 and bits == 8:
        return np.int8
    elif code == 2 and bits == 16:
        return np.int16
    elif code == 2 and bits == 32:
        return np.int32
    elif code == 2 and bits == 64:
        return np.int64
    elif bits == 8:
        return np.uint8
    elif bits == 16:
        return np.uint16
    elif bits == 32:
        return np.uint32
    elif bits == 64:
        return np.uint64
    else:
        code_name = ['float', 'quantized', 'signed', 'unsigned']
        raise ValueError('unsupported data type: {} bit {}'.format(bits, code_name[code]))


def write_tensor(file, tensor, quantization='', version_major=1, version_minor=0):
    np.asarray([0x4E, 0xEF, version_major, version_minor], dtype=np.uint8).tofile(file)

    header_length = _tensor_header_length(rank=tensor.ndim, qlen=len(quantization))
    np.asarray([header_length], dtype=np.uint32).tofile(file)

    np.asarray([tensor.ndim], dtype=np.uint32).tofile(file)
    np.asarray(tensor.shape, dtype=np.uint32).tofile(file)

    bits = _tensor_dtype_bits(tensor.dtype)
    code = _tensor_dtype_code(tensor.dtype, quantization)
    np.asarray([code, bits], dtype=np.uint8).tofile(file)

    np.asarray([len(quantization)], dtype=np.uint16).tofile(file)
    file.write(quantization.encode())

    tensor.tofile(file)


def read_tensor(file):
    [magic1, magic2, major, minor] = np.fromfile(file, dtype=np.uint8, count=4)
    if magic1 != 0x4E or magic2 != 0xEF:
        raise ValueError('not a valid NNEF file')
    if major > 1 or minor > 0:
        raise ValueError('unsupported file version')

    [header_length] = np.fromfile(file, dtype=np.uint32, count=1)

    [rank] = np.fromfile(file, dtype=np.uint32, count=1)
    shape = np.fromfile(file, dtype=np.uint32, count=rank)

    [code, bits] = np.fromfile(file, dtype=np.uint8, count=2)
    [qlen] = np.fromfile(file, dtype=np.uint16, count=1)

    if header_length != _tensor_header_length(rank, qlen):
        raise ValueError('invalid tensor header')

    quantization = file.read(qlen)
    tensor = np.fromfile(file, dtype=_numpy_dtype(code,bits), count=np.prod(shape)).reshape(shape)

    return tensor, quantization.decode()


def extract_positional_args(args):
    def isTensor(value):
        return isinstance(value, Tensor) or \
               (isinstance(value, list) and len(value) > 0 and isTensor(value[0]))

    positionals = OrderedDict()
    for (key, value) in args.items():
        if not isTensor(value):
            break
        positionals[key] = value

    for key in positionals:
        del args[key]

    return positionals, args
