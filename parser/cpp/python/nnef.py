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


Identifier = _nnef.Identifier
ShapeOf = _nnef.ShapeOf
Error = _nnef.Error
Prototype = _nnef.Prototype
TensorType = _nnef.TensorType
ArrayType = _nnef.ArrayType
TupleType = _nnef.TupleType
StandardOperations = _nnef.StandardOperations


def _register_layer_ops():
    _nnef.register_layer_ops()


def _unregister_layer_ops():
    _nnef.unregister_layer_ops()


def _register_custom_ops(key, text):
    _nnef.register_custom_ops(key, text)


def _unregister_custom_ops(key):
    _nnef.unregister_custom_ops(key)


def _register_custom_shapes(shapes):
    _nnef.register_custom_shapes(shapes)


def _register_deferred_shapes(shapes):
    _nnef.register_deferred_shapes(shapes)


def parse_file(input, quantization=None, atomics=StandardOperations):
    return _nnef.parse_file(input, quantization=quantization, atomics=atomics)


def parse_string(input, quantization=None, atomics=StandardOperations):
    return _nnef.parse_string(input, quantization=quantization, atomics=atomics)


def format_version(version):
    major, minor = version
    return 'version {}.{};'.format(major, minor)


def format_extensions(extensions):
    string = str()
    for i, ext in enumerate(extensions):
        if i != 0:
            string += '\n'
        string += 'extension {};'.format(ext)
    return string


def format_argument(value):
    if isinstance(value, Identifier):
        return value
    elif isinstance(value, ShapeOf):
        return 'shape_of(' + value.id + ')'
    elif isinstance(value, str):
        return "'" + value + "'"
    elif isinstance(value, bool):
        return 'true' if value else 'false'
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, (list, tuple)):
        string = '[' if isinstance(value, list) else '('
        for idx, item in enumerate(value):
            if idx != 0:
                string += ', '
            string += format_argument(item)
        string += ']' if isinstance(value, list) else ')'
        return string
    else:
        raise TypeError('arguments must be of type int, float, str, nnef.Identifier or list/tuple of such, found: ' + str(type(value)))


def format_result(value):
    if isinstance(value, (list, tuple)):
        string = '[' if isinstance(value, list) else '('
        for idx, item in enumerate(value):
            if idx != 0:
                string += ', '
            string += format_result(item)
        string += ']' if isinstance(value, list) else ')'
        return string
    elif isinstance(value, Identifier):
        return value
    else:
        raise TypeError('results must be of type nnef.Identifier or list/tuple of such, found: ' + str(type(value)))


def format_invocation(name, args, kwargs, results=[], dtype=None):
    string = str()

    for (idx, result) in enumerate(results):
        if idx != 0:
            string += ', '
        string += format_result(result)

    if len(results) != 0:
        string += ' = '

    string += name

    if dtype is not None:
        string += '<' + dtype + '>'

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


def format_typespec(typespec):
    _check_typespec(typespec)

    if isinstance(typespec, str):
        return typespec
    elif isinstance(typespec, TensorType):
        return 'tensor<' + format_typespec(typespec.dataType) + '>' if typespec.dataType is not None else 'tensor<>'
    elif isinstance(typespec, ArrayType):
        return format_typespec(typespec.itemType) + '[]'
    elif isinstance(typespec, TupleType):
        string = '('
        for idx, item in enumerate(typespec.itemTypes):
            if idx != 0:
                string += ','
            string += format_typespec(item)
        string += ')'
        return string


def _format_params(items):
    string = ''
    for idx, item in enumerate(items):
        if idx != 0:
            string += ', '
        string += item
    return '( ' + string + ' )'


def format_params(params, defaults={}):
    def _format_default(param, defaults):
        return ' = ' + format_argument(defaults[param]) if param in defaults else ''

    items = [param + ': ' + format_typespec(typespec) + _format_default(param, defaults) for (param, typespec) in params.items()]
    return _format_params(items)


def format_graph(name, inputs, outputs):
    if isinstance(inputs, dict):
        inputs = [input for (input, type) in inputs.items() if not _is_attribute_type(type)]

    return name + _format_params(inputs) + ' -> ' + _format_params(outputs)


def format_prototype(name, inputs, outputs, defaults={}):
    has_generic_input = any([_is_generic_type(input) for input in inputs.values()])
    has_generic_output = any([_is_generic_type(output) for output in outputs.values()])
    is_generic = has_generic_input or has_generic_output
    generic_default = defaults.get('?')
    generic = '<? = ' + generic_default + '>' if generic_default is not None else '<?>' if is_generic else ''

    return name + generic + format_params(inputs, defaults) + ' -> ' + format_params(outputs)


def format_document(attrs, ops):
    string = format_version(attrs['version']) + '\n'
    string += format_extensions(attrs['extensions']) + '\n'

    graph = attrs['graph']
    string += 'graph ' + format_graph(name=graph.name, inputs=graph.params, outputs=graph.results) + '\n'

    string += '{\n'
    for (proto, args) in ops:
        inputs, attribs, outputs, dtype = split_args(args, params=proto.params, results=proto.results, split_attribs=True)
        invocation = format_invocation(proto.name, args=inputs.values(), kwargs=attribs, results=outputs.values(), dtype=dtype)
        string += '\t' + invocation + ';\n'
    string += '}\n'
    return string


def split_args(args, params, results, split_attribs):
    inputs = OrderedDict()
    attribs = OrderedDict()
    for (name, type) in params.items():
        if split_attribs and _is_attribute_type(type):
            attribs[name] = args[name]
        else:
            inputs[name] = args[name]

    outputs = OrderedDict()
    for (name, type) in results.items():
        outputs[name] = args[name]

    dtype = args.get('?')
    return (inputs, attribs, outputs, dtype) if split_attribs else (inputs, outputs, dtype)


def _is_attribute_type(typespec):
    _check_typespec(typespec)

    if isinstance(typespec, str):
        return True
    elif isinstance(typespec, TensorType):
        return False
    elif isinstance(typespec, ArrayType):
        return _is_attribute_type(typespec.itemType)
    elif isinstance(typespec, TupleType):
        return all([_is_attribute_type(item) for item in typespec.itemTypes])


def _is_generic_type(typespec):
    _check_typespec(typespec)

    if isinstance(typespec, str):
        return typespec == '?'
    elif isinstance(typespec, TensorType):
        return _is_generic_type(typespec.dataType)
    elif isinstance(typespec, ArrayType):
        return _is_generic_type(typespec.itemType)
    elif isinstance(typespec, TupleType):
        return any([_is_generic_type(item) for item in typespec.itemTypes])


def _check_typespec(typespec):
    if isinstance(typespec, str):
        if not typespec in ['scalar', 'integer', 'logical', 'string', '?']:
            raise TypeError('invalid primitive type name: ' + typespec)
    elif not isinstance(typespec, (TensorType, ArrayType, TupleType)):
        raise TypeError('typespec must be string or one of nnef.TensorType, nnef.ArrayType, nnef.TupleType')



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


def write_tensor(file, tensor, version=(1,0), quantization={}):
    np.asarray([0x4E, 0xEF, version[0], version[1]], dtype=np.uint8).tofile(file)

    dtype, bits = _numpy_dtype_split(tensor.dtype)
    data_length = (np.prod(tensor.shape) * bits + 7) / 8
    np.asarray([data_length, tensor.ndim], dtype=np.uint32).tofile(file)

    if tensor.ndim > MaxTensorRank:
        raise ValueError('tensor rank exceeds maximum possible value of {}'.format(MaxTensorRank))

    np.asarray(tensor.shape, dtype=np.uint32).tofile(file)
    np.asarray([0] * (MaxTensorRank - tensor.ndim), dtype=np.uint32).tofile(file);

    code = quantization.get('code', QUANT_CODE_FLOAT if dtype == np.float else QUANT_CODE_INTEGER)
    if (code == QUANT_CODE_FLOAT and dtype != np.float) or \
        (code == QUANT_CODE_INTEGER and dtype != np.int and dtype != np.uint) or \
        ((code == QUANT_CODE_LINEAR or code == QUANT_CODE_LOGARITHMIC) and dtype != np.uint):
        raise ValueError('incompatible quantization code ({}) and tensor dtype ({})'.format(code, tensor.dtype))

    np.asarray([bits, code], dtype=np.uint32).tofile(file)

    params = [0] * 8
    if code == QUANT_CODE_LINEAR or code == QUANT_CODE_LOGARITHMIC:
        params[0] = quantization['min']
        params[1] = quantization['max']
    elif code == QUANT_CODE_INTEGER:
        params[0] = 0 if dtype == np.uint else 1
    elif code != QUANT_CODE_FLOAT:
        raise ValueError('unsupported item type code: {}'.format(code))

    np.asarray(params, dtype=np.float32).tofile(file)
    np.asarray([0] * 12, dtype=np.uint32).tofile(file)

    tensor.tofile(file)


def read_tensor(file):
    [magic1, magic2, major, minor] = np.fromfile(file, dtype=np.uint8, count=4)
    if magic1 != 0x4E or magic2 != 0xEF:
        raise ValueError('not a valid NNEF file')
    if major > 1 or minor > 0:
        raise ValueError('unsupported file version')

    [data_length, rank] = np.fromfile(file, dtype=np.uint32, count=2)

    if rank > MaxTensorRank:
        raise ValueError('tensor rank exceeds maximum possible value of {}'.format(MaxTensorRank))

    shape = np.fromfile(file, dtype=np.uint32, count=MaxTensorRank)
    shape = shape[:rank]

    [bits, code] = np.fromfile(file, dtype=np.uint32, count=2)
    params = np.fromfile(file, dtype=np.float32, count=8)
    padding = np.fromfile(file, dtype=np.uint32, count=12)

    signed = params[0] != 0 if code == QUANT_CODE_INTEGER else False

    count = np.prod(shape)
    data = np.fromfile(file, dtype=_numpy_dtype_make(code,bits,signed), count=count)
    if len(data) != count:
        raise ValueError('could not read tensor data')

    tensor = data.reshape(shape)

    quantization = {'code': code}
    if code == QUANT_CODE_LINEAR or code == QUANT_CODE_LOGARITHMIC:
        quantization['min'] = params[0]
        quantization['max'] = params[1]
    elif code != QUANT_CODE_FLOAT and code != QUANT_CODE_INTEGER:
        raise ValueError('unsupported item type code: {}'.format(code))

    return tensor, quantization



def _write_tensor_provisional(file, tensor, version=(1,0)):
    np.asarray([0x4E, 0xEF, version[0], version[1]], dtype=np.uint8).tofile(file)

    header_length = 4 + 4 + (tensor.ndim + 1) * 4 + 4
    np.asarray([header_length], dtype=np.uint32).tofile(file)

    np.asarray([tensor.ndim], dtype=np.uint32).tofile(file)
    np.asarray(tensor.shape, dtype=np.uint32).tofile(file)

    dtype, bits = _numpy_dtype_split(tensor.dtype)
    code = 0 if dtype == np.float else 3
    np.asarray([code, bits], dtype=np.uint8).tofile(file)

    np.asarray([0], dtype=np.uint16).tofile(file)

    tensor.tofile(file)


def _read_tensor_provisional(file):
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

    assert(code == 0)
    assert(bits == 32)
    assert(qlen == 0)

    return np.fromfile(file, dtype=np.float32, count=np.prod(shape)).reshape(shape)
