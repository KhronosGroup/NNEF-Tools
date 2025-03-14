from __future__ import division, print_function, absolute_import

from .helpers import *
from ....model import *
try:
    from flatbuffers import flexbuffers
    has_flexbuffers = True
except ImportError:
    has_flexbuffers = False
import sys
import six


def _get_quantization(tensor):
    quant = tensor.Quantization()
    if quant is None:
        return None

    if quant.MinLength() == 0:
        min = None
    elif quant.MinLength() == 1:
        min = float(quant.Min(0))
    else:
        min = quant.MinAsNumpy()

    if quant.MaxLength() == 0:
        max = None
    elif quant.MaxLength() == 1:
        max = float(quant.Max(0))
    else:
        max = quant.MaxAsNumpy()

    if quant.ScaleLength() == 0:
        scale = None
    elif quant.ScaleLength() == 1:
        scale = float(quant.Scale(0))
    else:
        scale = quant.ScaleAsNumpy()

    if quant.ZeroPointLength() == 0:
        zero_point = None
    elif quant.ZeroPointLength() == 1:
        zero_point = int(quant.ZeroPoint(0))
    else:
        zero_point = quant.ZeroPointAsNumpy()

    if all(x is None for x in [min, max, scale, zero_point]):
        return None
    else:
        return dict(min=min, max=max, zero_point=zero_point, scale=scale)


def _get_data_as_ndarray(buffer, dtype, shape):
    return buffer.DataAsNumpy().view(dtype).reshape(shape) if buffer.DataLength() != 0 else None


def _get_options_starter_ender(optionsClass):
    className = optionsClass.__name__
    optionsModule = sys.modules[optionsClass.__module__]
    moduleDict = optionsModule.__dict__
    return moduleDict[className + 'Start'], moduleDict[className + 'End']


def _enumerate_attributes(optionsClass, optionsObject):
    getters = enumerate_options_getters(optionsClass)
    length_getters = enumerate_options_length_getters(optionsClass)

    attribs = {}
    for name, getter in getters.items():
        length_getter = length_getters.get(name)

        value = getter(optionsObject) if length_getter is None else \
            [getter(optionsObject, i) for i in range(length_getter(optionsObject))]

        attribs[name] = substitute_enum_value_with_name(name, value, optionsClass)

    return attribs


def _decode_custom_options(bytes):
    root = flexbuffers.GetRoot(bytes)
    assert root.IsMap
    return {key: value for key, value in six.iteritems(root.AsMap.Value) if not key.startswith('_')}


def read_flatbuffers(filename):
    with open(filename, 'rb') as file:
        bytes = bytearray(file.read())

    fbmodel = fb.Model.GetRootAsModel(bytes, 0)

    if fbmodel.SubgraphsLength() != 1:
        raise NotImplementedError('graphs with multiple sub-graphs are not supported')

    subgraph = fbmodel.Subgraphs(0)
    name = subgraph.Name()
    name = name.decode() if name is not None else None

    model = Model(name)
    graph = Graph(model, name)

    tensors = []
    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        name = tensor.Name().decode()
        shape = tuple(tensor.Shape(i) for i in range(tensor.ShapeLength()))
        dtype = DtypeToNumpy[tensor.Type()]
        buffer = fbmodel.Buffers(tensor.Buffer())
        data = _get_data_as_ndarray(buffer, dtype, shape)
        quant = _get_quantization(tensor)
        tensors.append(Tensor(graph, name, shape, dtype, data, quant))

    for i in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(i)
        operatorCode = fbmodel.OperatorCodes(operator.OpcodeIndex())
        builtinCode = operatorCode.BuiltinCode()
        opType = BuiltinOperatorTypeByValue[builtinCode] if builtinCode != fb.BuiltinOperator.CUSTOM else \
            operatorCode.CustomCode().decode('ascii')
        custom = builtinCode == fb.BuiltinOperator.CUSTOM

        options = operator.BuiltinOptions()
        optionsClass = BuiltinOptionsClasses[operator.BuiltinOptionsType()]

        inputs = tuple(tensors[operator.Inputs(i)] for i in range(operator.InputsLength()) if operator.Inputs(i) != -1)
        outputs = tuple(tensors[operator.Outputs(i)] for i in range(operator.OutputsLength()) if operator.Outputs(i) != -1)

        if options is not None and optionsClass is not None:
            optionsObject = optionsClass()
            optionsObject.Init(options.Bytes, options.Pos)
            attribs = _enumerate_attributes(optionsClass, optionsObject)
        elif custom:
            bytes = operator.CustomOptionsAsNumpy().tobytes()
            attribs = _decode_custom_options(bytes) if has_flexbuffers else {CustomOptionsKey: bytes}
        else:
            attribs = {}

        Operation(graph, type=opType, custom=custom, attribs=attribs, inputs=inputs, outputs=outputs)

    graph.inputs = tuple(tensors[subgraph.Inputs(i)] for i in range(subgraph.InputsLength()))
    graph.outputs = tuple(tensors[subgraph.Outputs(i)] for i in range(subgraph.OutputsLength()))

    return model


class Reader(object):

    def __call__(self, filename):
        return read_flatbuffers(filename)
