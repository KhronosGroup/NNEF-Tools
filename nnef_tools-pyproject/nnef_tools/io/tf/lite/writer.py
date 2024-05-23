# Copyright (c) 2020 The Khronos Group Inc.
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

from .helpers import *
import flatbuffers
try:
    from flatbuffers import flexbuffers
    has_flexbuffers = True
except ImportError:
    has_flexbuffers = False
import numpy as np


def _CreateNumpyVector(builder, x):
    if not isinstance(x, np.ndarray):
        raise TypeError("non-numpy-ndarray passed to CreateNumpyVector")

    if x.dtype.kind not in ['b', 'i', 'u', 'f']:
        raise TypeError("numpy-ndarray holds elements of unsupported datatype")

    if x.ndim > 1:
        raise TypeError("multidimensional-ndarray passed to CreateNumpyVector")

    builder.StartVector(x.itemsize, x.size, x.dtype.alignment)

    # Ensure little endian byte ordering
    if x.dtype.str[0] != "<":
        x = x.byteswap()

    length = x.itemsize * x.size
    builder.head -= length
    builder.Bytes[builder.head: builder.head + length] = x.tobytes()

    return builder.EndVector(x.size)


def _build_buffer(builder, bytes):
    data = _CreateNumpyVector(builder, bytes)
    fb.BufferStart(builder)
    fb.BufferAddData(builder, data)
    return fb.BufferEnd(builder)


def _build_tensor(builder, tensor, buffer_index):
    name = builder.CreateString(tensor.name)
    type = DtypeFromNumpy[tensor.data.dtype.type if isinstance(tensor.data, np.ndarray) else tensor.dtype]

    fb.TensorStartShapeVector(builder, len(tensor.shape))
    for s in reversed(tensor.shape):
        builder.PrependInt32(s)
    shape = builder.EndVector(len(tensor.shape))

    buffer = buffer_index if tensor.data is not None else 0

    quant = _build_quantization(builder, tensor.quant, type)

    fb.TensorStart(builder)
    fb.TensorAddName(builder, name)
    fb.TensorAddShape(builder, shape)
    fb.TensorAddType(builder, type)
    fb.TensorAddBuffer(builder, buffer)
    if quant is not None:
        fb.TensorAddQuantization(builder, quant)
    return fb.TensorEnd(builder)


def _ensure_numpy_array(x, dtype):
    if isinstance(x, np.ndarray):
        assert x.dtype == dtype
        return x
    else:
        return np.array(x, dtype=dtype)


def _build_quantization(builder, quant, dtype):
    if quant is None:
        return None

    min = quant.get('min')
    max = quant.get('max')
    zero_point = quant.get('zero_point')
    scale = quant.get('scale')

    if all(item is None or item == 0 for item in [min, max, zero_point, scale]):
        return None

    min = _CreateNumpyVector(builder, _ensure_numpy_array(min, dtype=np.float32)) if min is not None else None
    max = _CreateNumpyVector(builder, _ensure_numpy_array(max, dtype=np.float32)) if max is not None else None
    scale = _CreateNumpyVector(builder, _ensure_numpy_array(scale, dtype=np.float32)) if scale is not None else None
    zero_point = _CreateNumpyVector(builder, _ensure_numpy_array(zero_point, dtype=np.int64)) if zero_point is not None else None

    fb.QuantizationParametersStart(builder)
    if dtype != fb.TensorType.INT32:
        if min is not None:
            fb.QuantizationParametersAddMin(builder, min)
        if max is not None:
            fb.QuantizationParametersAddMax(builder, max)
    if scale is not None:
        fb.QuantizationParametersAddScale(builder, scale)
    if zero_point is not None:
        fb.QuantizationParametersAddZeroPoint(builder, zero_point)
    return fb.QuantizationParametersEnd(builder)


def _build_operator_code(builder, operation):
    builtinCode = BuiltinOperatorValueByType[operation.type] if not operation.custom else fb.BuiltinOperator.CUSTOM
    customCode = builder.CreateString(operation.type) if operation.custom else None

    fb.OperatorCodeStart(builder)
    fb.OperatorCodeAddBuiltinCode(builder, builtinCode)
    if customCode:
        fb.OperatorCodeAddCustomCode(builder, customCode)
    return fb.OperatorCodeEnd(builder)


def _build_operator_options(builder, attribs, optionsClass):
    starter, ender = get_options_starter_ender(optionsClass)
    adders = enumerate_options_adders(optionsClass)
    vector_starters = enumerate_options_vector_starters(optionsClass)

    vector_values = {}
    for name, vector_starter in vector_starters.items():
        value = attribs[name]
        assert isinstance(value, (list, tuple)) and (len(value) == 0 or isinstance(value[0], int))
        vector_starter(builder, len(value))
        for i in reversed(value):
            builder.PrependInt32(i)
        vector_values[name] = builder.EndVector(len(value))

    starter(builder)
    for name, adder in adders.items():
        if name == 'fused_activation_function' and name not in attribs:
            value = 'NONE'
        else:
            value = attribs[name]
            if isinstance(value, type):
                value = DtypeFromNumpy[value]

        value = vector_values.get(name, value)
        value = substitute_enum_name_with_value(name, value, optionsClass)

        adder(builder, value)

    return ender(builder)


def _encode_custom_options(attribs):
    builder = flexbuffers.Builder()
    builder.MapFromElements(attribs)
    return builder.Finish()


def _build_operator_custom_options(builder, attribs):
    value = _encode_custom_options(attribs) if has_flexbuffers else attribs[CustomOptionsKey]

    fb.OperatorStartCustomOptionsVector(builder, len(value))
    for b in reversed(value):
        builder.PrependUint8(b)
    return builder.EndVector(len(value))


def _build_operator(builder, operation, op_code_index, tensor_index):
    inputs = [tensor_index[tensor] for tensor in operation.inputs]
    fb.OperatorStartInputsVector(builder, len(inputs))
    for input in reversed(inputs):
        builder.PrependInt32(input)
    inputs = builder.EndVector(len(inputs))

    outputs = [tensor_index[tensor] for tensor in operation.outputs]
    fb.OperatorStartOutputsVector(builder, len(outputs))
    for output in reversed(outputs):
        builder.PrependInt32(output)
    outputs = builder.EndVector(len(outputs))

    attribs = {name: value for name, value in operation.attribs.items()}

    optionsType = BuiltinOptionsByOperator[BuiltinOperatorValueByType.get(operation.type, fb.BuiltinOperator.CUSTOM)]
    optionsClass = BuiltinOptionsClasses[optionsType]
    options = _build_operator_options(builder, attribs, optionsClass) if optionsClass is not None else None
    custom_options = _build_operator_custom_options(builder, attribs) if operation.custom else None

    fb.OperatorStart(builder)
    fb.OperatorAddOpcodeIndex(builder, op_code_index[operation.type])
    fb.OperatorAddInputs(builder, inputs)
    fb.OperatorAddOutputs(builder, outputs)
    fb.OperatorAddBuiltinOptionsType(builder, optionsType)

    if options:
        fb.OperatorAddBuiltinOptions(builder, options)

    if custom_options:
        fb.OperatorAddCustomOptions(builder, custom_options)

    return fb.OperatorEnd(builder)


# https://github.com/google/flatbuffers/issues/4814
def FinishWithFileIdentifier(builder, rootTable, fid):
    from flatbuffers import number_types as N
    from flatbuffers import encode

    if fid is None or len(fid) != 4:
        raise Exception('fid must be 4 chars')

    flags = N.Uint8Flags
    prepSize = 4
    builder.Prep(builder.minalign, prepSize + len(fid))
    for i in range(3, -1, -1):
        builder.head = builder.head - flags.bytewidth
        encode.Write(flags.packer_type, builder.Bytes, builder.Head(), ord(fid[i]))

    return builder.Finish(rootTable)


def write_flatbuffers(graph, filename):
    graph.sort()
    builder = flatbuffers.Builder(0)

    fb.BufferStartDataVector(builder, 0)
    data = builder.EndVector(0)
    fb.BufferStart(builder)
    fb.BufferAddData(builder, data)
    buffer = fb.BufferEnd(builder)

    buffers = [buffer]
    for tensor in graph.tensors:
        if tensor.data is not None:
            tensor_data = tensor.data
            if not isinstance(tensor_data, np.ndarray):
                tensor_data = np.array(tensor_data, dtype=tensor.dtype)
            bytes = tensor_data.reshape([-1]).view(np.uint8)
            buffers.append(_build_buffer(builder, bytes))

    fb.ModelStartBuffersVector(builder, len(buffers))
    for buffer in reversed(buffers):
        builder.PrependUOffsetTRelative(buffer)
    buffers = builder.EndVector(len(buffers))

    buffer_index = 1

    tensors = []
    tensor_index = {}
    for tensor in graph.tensors:
        tensor_index[tensor] = len(tensors)
        tensors.append(_build_tensor(builder, tensor, buffer_index))
        if tensor.data is not None:
            buffer_index += 1

    fb.SubGraphStartTensorsVector(builder, len(tensors))
    for tensor in reversed(tensors):
        builder.PrependUOffsetTRelative(tensor)
    tensors = builder.EndVector(len(tensors))

    op_codes = []
    op_code_index = {}
    for operation in graph.operations:
        if operation.type not in op_code_index:
            op_code_index[operation.type] = len(op_codes)
            op_codes.append(_build_operator_code(builder, operation))

    fb.ModelStartOperatorCodesVector(builder, len(op_codes))
    for op_code in reversed(op_codes):
        builder.PrependUOffsetTRelative(op_code)
    op_codes = builder.EndVector(len(op_codes))

    operators = []
    for operation in graph.operations:
        operators.append(_build_operator(builder, operation, op_code_index, tensor_index))

    fb.SubGraphStartOperatorsVector(builder, len(operators))
    for operator in reversed(operators):
        builder.PrependUOffsetTRelative(operator)
    operators = builder.EndVector(len(operators))

    name = builder.CreateString(graph.name) if graph.name is not None else None

    inputs = graph.inputs
    fb.SubGraphStartInputsVector(builder, len(inputs))
    for input in reversed(inputs):
        builder.PrependInt32(tensor_index[input])
    inputs = builder.EndVector(len(inputs))

    outputs = graph.outputs
    fb.SubGraphStartInputsVector(builder, len(outputs))
    for output in reversed(outputs):
        builder.PrependInt32(tensor_index[output])
    outputs = builder.EndVector(len(outputs))

    fb.SubGraphStart(builder)
    if name is not None:
        fb.SubGraphAddName(builder, name)
    fb.SubGraphAddTensors(builder, tensors)
    fb.SubGraphAddOperators(builder, operators)
    fb.SubGraphAddInputs(builder, inputs)
    fb.SubGraphAddOutputs(builder, outputs)
    subgraph = fb.SubGraphEnd(builder)

    fb.ModelStartSubgraphsVector(builder, 1)
    builder.PrependUOffsetTRelative(subgraph)
    subgraphs = builder.EndVector(1)

    fb.ModelStart(builder)
    fb.ModelAddVersion(builder, OUTPUT_SCHEMA_VERSION)
    fb.ModelAddBuffers(builder, buffers)
    fb.ModelAddOperatorCodes(builder, op_codes)
    fb.ModelAddSubgraphs(builder, subgraphs)
    model = fb.ModelEnd(builder)

    FinishWithFileIdentifier(builder, model, OUTPUT_FILE_IDENTIFIER)

    with open(filename, 'wb') as file:
        file.write(builder.Output())


class Writer(object):

    def __call__(self, graph, filename):
        write_flatbuffers(graph, filename)
