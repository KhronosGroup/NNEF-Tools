from __future__ import division, print_function, absolute_import

import nnef
import torch
import keyword

from . import nnef_operators
from ...io import nnef as nnef_io
from ...io.nnef.reader import _build_model
from ...model import *
from ...model.utils import recursive_itemize


class NNEFModule(torch.nn.Module):

    """
    A torch.nn.Module that interprets the given NNEF model
    """

    def __init__(self,
                 model,  # type: str
                 decomposed=None,  # type: typing.Optional[typing.List[str]]
                 custom_operators=None,  # type: typing.Optional[typing.Dict[str, typing.Callable]]
                 activation_callback=None,  # type: typing.Optional[typing.Callable[[str, torch.Tensor], None]]
                 training_attributes=None,  # type: typing.Optional[typing.Dict[str, typing.Dict[str, typing.Any]]]
                 ):
        # type: (...)->None
        """
            nnef_graph might be modified by this class if training and write_nnef is used
        """
        super(NNEFModule, self).__init__()
        if isinstance(model, nnef.Graph):
            self._nnef_model = _build_model(model)
        else:
            reader = nnef_io.Reader(decomposed=decomposed, infer_shapes=False)
            self._nnef_model = reader(model)

        self._name_inline_constants(self._nnef_model)

        graph = self._nnef_model.main
        for nnef_tensor in graph.tensors:
            if self._is_variable(nnef_tensor):
                name = self._registered_name(nnef_tensor.name)
                data = self._dequantize(nnef_tensor.data, nnef_tensor.quant, channel_axis=0) \
                    if nnef_tensor.quant else nnef_tensor.data
                data = self.normalize_dtype(data)
                self.register_parameter(name, torch.nn.Parameter(torch.tensor(data), requires_grad=data.dtype == np.float32))
            elif self._is_constant(nnef_tensor):
                name = self._registered_name(nnef_tensor.name)
                data = nnef_tensor.data if not nnef_tensor.producer else \
                    self._as_numpy(nnef_tensor.producer.attribs['value'],
                                   nnef_tensor.producer.attribs['shape'],
                                   nnef_tensor.producer.attribs['dtype'])
                data = self.normalize_dtype(data)
                self.register_buffer(name, torch.tensor(data))

        self._operators = {}
        self._operators.update(nnef_operators.Operators)
        if custom_operators:
            self._operators.update(custom_operators)
        self._activation_callback = activation_callback
        self._training_attributes = training_attributes or {}

    def forward(self, *inputs):
        graph = self._nnef_model.main
        assert len(inputs) == len(graph.inputs)
        activations = {nnef_tensor.name: torch_tensor for torch_tensor, nnef_tensor
                       in zip(inputs, graph.inputs)}

        def get_tensor(name):
            if hasattr(self, self._registered_name(name)):
                return getattr(self, self._registered_name(name))
            else:
                return activations[name]

        def get_tensors(query):
            return [get_tensor(item.name) for item in query] if isinstance(query, list) else get_tensor(query.name)

        def has_tensor(name):
            return hasattr(self, self._registered_name(name)) or name in activations

        def has_tensors(query):
            return all(has_tensor(item.name) for item in query) if isinstance(query, list) else has_tensor(query.name)

        for op in graph.operations:
            if op.type == 'external' or op.type == 'variable' or op.type == 'constant':
                output = get_tensor(op.output.name)
                if self._activation_callback:
                    self._activation_callback(op.output.name, output)
            else:
                assert op.type in self._operators, "Unsupported operation: {}".format(op.type)
                func = self._operators[op.type]

                assert all(has_tensors(input) for input in op.inputs),\
                    "could not fetch input tensor(s) {} for operation {}"\
                        .format({[item.name for item in input] if isinstance(input, list) else input.name
                                 for input in op.inputs}, op.type)

                training_attribs = self._training_attributes.get(op.type, {})
                attribs = {**op.attribs, **training_attribs}
                attribs = {self._escape_keyword(name): value for name, value in six.iteritems(attribs)}

                if 'dtype' in attribs and op.type != 'constant' and op.type != 'cast':
                    del attribs['dtype']

                inputs = [get_tensors(input) for input in op.inputs]
                outputs = func(*inputs, **attribs)

                if not isinstance(outputs, tuple):
                    outputs = (outputs,)

                for nnef_tensor, output in zip(recursive_itemize(op.outputs), recursive_itemize(outputs)):
                    if nnef_tensor.quant and not self._is_variable(nnef_tensor):
                        output = self._fake_quantize(output, nnef_tensor.quant, channel_axis=0)

                    activations[nnef_tensor.name] = output
                    if self._activation_callback:
                        self._activation_callback(nnef_tensor.name, output)

                for nnef_tensor in recursive_itemize(op.inputs):
                    if nnef_tensor.name in activations and op is nnef_tensor.consumers[-1] and \
                            nnef_tensor not in graph.outputs:
                        del activations[nnef_tensor.name]

        return tuple(get_tensors(output) for output in graph.outputs)

    def save_nnef(self, path):
        graph = self._nnef_model.main
        for nnef_tensor in graph.tensors:
            if self._is_variable(nnef_tensor.name):
                torch_tensor = getattr(self, self._registered_name(nnef_tensor.name))
                nnef_tensor.data = torch_tensor.detach().cpu().numpy().astype(nnef_tensor.dtype)

        writer = nnef_io.Writer()
        writer(self._nnef_model, path)

    @property
    def activation_callback(self):
        return self._activation_callback

    @activation_callback.setter
    def activation_callback(self, callback):
        self._activation_callback = callback

    @staticmethod
    def _is_variable(tensor):
        return tensor.producer and tensor.producer.type == 'variable'

    @staticmethod
    def _is_constant(tensor):
        return not tensor.producer or tensor.producer.type == 'constant'

    @staticmethod
    def _as_numpy(value, shape, dtype):
        if isinstance(value, list):
            if len(value) == 1 and int(np.prod(shape)) != 1:
                return np.full(shape, value[0], dtype=dtype)
            else:
                return np.array(value, dtype=dtype).reshape(shape)
        else:
            return np.full(shape, value, dtype=dtype)

    @staticmethod
    def _escape_keyword(name):
        return name if not keyword.iskeyword(name) else '_' + name + '_'

    @staticmethod
    def _name_inline_constants(model):
        constants = 0
        graph = model.main
        for tensor in graph.tensors:
            if not tensor.name:
                assert not tensor.producer
                tensor.name = '$' + str(constants)
                constants += 1

    @staticmethod
    def _registered_name(name):
        return '_nnef_' + name

    @staticmethod
    def normalize_dtype(data):
        dtype = NNEFModule._dtypeRemap.get(data.dtype.type)
        return data.astype(dtype) if dtype is not None else data

    @staticmethod
    def _dequantize(data, quant, channel_axis):
        op_name = quant['op-name']
        rank = len(data.shape)
        if op_name == 'zero_point_linear_quantize':
            return NNEFModule._dequantize_zero_point(data,
                                                     NNEFModule._ensure_rank(quant['zero_point'], rank, channel_axis),
                                                     NNEFModule._ensure_rank(quant['scale'], rank, channel_axis))
        elif op_name == 'min_max_linear_quantize' or op_name == 'linear_quantize':
            return NNEFModule._dequantize_min_max(data,
                                                  NNEFModule._ensure_rank(quant['min'], rank, channel_axis),
                                                  NNEFModule._ensure_rank(quant['max'], rank, channel_axis),
                                                  quant['signed'], quant['symmetric'], quant['bits'])
        else:
            raise ValueError("Quantization operation '{}' not implemented".format(op_name))

    @staticmethod
    def _dequantize_zero_point(data, zero_point, scale):
        return (data - zero_point) * scale

    @staticmethod
    def _dequantize_min_max(data, min, max, signed, symmetric, bits):
        if signed:
            data += 2 ** (bits - 1) - int(symmetric)
        r = 2 ** bits - 1 - int(signed and symmetric)
        return data * ((max - min) / r) + min

    def _fake_quantize(self, tensor, quant, channel_axis):
        op_type = quant['op-name']
        rank = len(tensor.shape)
        attribs = {key: NNEFModule._ensure_rank(value, rank, channel_axis) if isinstance(value, np.ndarray) else value
                   for key, value in six.iteritems(quant) if key != 'op-name'}

        assert op_type in self._operators, "Unsupported quantization operation: {}".format(op_type)
        func = self._operators[op_type]
        return func(tensor, **attribs)

    @staticmethod
    def _ensure_rank(value, rank, offset=0):
        array = np.array(value)
        return np.reshape(array, newshape=(1,) * offset + array.shape + (1,) * (rank - offset - len(array.shape)))

    _dtypeRemap = {
        np.float16: np.float32,
        np.float64: np.float32,
        np.int8: np.int64,
        np.uint8: np.int64,
        np.int16: np.int64,
        np.uint16: np.int64,
        np.int32: np.int64,
        np.uint32: np.int64,
        np.uint64: np.int64,
    }
