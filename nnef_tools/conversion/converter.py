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

from ..model import *
from ..utils import types
import numpy as np
import functools
import inspect
import math
import copy
import six
import re


class Transform:

    def __init__(self, type, name=None, inputs=None, outputs=None, attribs=None,
                 defaults=None, using=None, cond=None, custom=False):
        self.type = type
        self.name = name or '!_name_'
        self.inputs = inputs or ()
        self.outputs = outputs or ()
        self.attribs = attribs or {}
        self.defaults = defaults
        self.using = using or {}
        self.cond = cond
        self.custom = custom

    def with_type(self, type):
        return Transform(type=type, name=self.name, inputs=self.inputs, outputs=self.outputs, attribs=self.attribs,
                         defaults=self.defaults, using=self.using, cond=self.cond, custom=self.custom)


class ConversionError(Exception):

    def __init__(self, message, details=None):
        Exception.__init__(self, message)
        self.details = details


class Converter:

    @staticmethod
    def find_public_methods(obj):
        methods = inspect.getmembers(obj, predicate=inspect.ismethod)
        return {name: func for name, func in methods if not name.startswith('_')}

    @staticmethod
    def find_public_functions(obj):
        methods = inspect.getmembers(obj, predicate=inspect.isfunction)
        return {name: func for name, func in methods if not name.startswith('_')}

    @staticmethod
    def decomposed_operations():
        return []       # return list of decomposed NNEF ops in subclass if converting from NNEF

    @staticmethod
    def defined_operations():
        return {}       # return dictionary of NNEF operator (fragment) definitions in subclass if converting to NNEF

    @staticmethod
    def defined_operation_dependencies():
        return {}  # return dictionary of NNEF operator (fragment) dependencies in subclass if converting to NNEF

    @staticmethod
    def defined_shapes():
        return {}       # return dictionary of shape functions for NNEF fragments defined by the converter

    @staticmethod
    def unpack_transforms(transforms):
        unpacked = {}
        for key, value in six.iteritems(transforms):
            assert isinstance(value, Transform)

            if isinstance(key, tuple):
                if isinstance(value, Transform) and isinstance(value.type, tuple):
                    for key_item, type_item in zip(key, value.type):
                        value_item = copy.deepcopy(value)
                        value_item.type = type_item
                        unpacked[key_item] = value_item
                else:
                    for item in key:
                        unpacked[item] = value
            else:
                unpacked[key] = value

        return unpacked

    @staticmethod
    def merge_transforms(default_transforms, custom_transforms):
        if custom_transforms is None:
            return default_transforms

        transforms = dict(default_transforms)
        transforms.update(custom_transforms)
        return transforms

    def __init__(self, transforms, functions=None, mirror_unsupported=False, infer_shapes=False, custom_shapes=None):
        self._graph = None
        self._transforms = transforms
        self._callables = self.find_public_methods(self)
        if functions:
            self._callables.update({name: functools.partial(func, self) for name, func in six.iteritems(functions)})
        self._mirror_unsupported = mirror_unsupported
        self._infer_shapes = infer_shapes
        self._custom_shapes = custom_shapes or {}

    def __call__(self, graph):
        if not self._infer_shapes:
            unknown_tensors = [tensor for tensor in graph.tensors if (tensor.shape is None or any(s is None for s in tensor.shape))
                               and len(tensor.consumers)]
            if len(unknown_tensors):
                names = ["'{}'".format(tensor.name) for tensor in unknown_tensors if tensor.name]
                raise ConversionError(("Input graph contains tensors with dynamic shape: " +
                                      ", ".join(names) if len(names) else "(no names)") +
                                      "\nTry the --fold-constants option to evaluate constant sub-graphs "
                                      "or the --static-only option to convert only the static part of the graph")

        self._graph = Graph(name=graph.name)
        self._tensor_map = {tensor: self._copy_tensor_(tensor) for tensor in graph.tensors}
        self._tensor_map.update({val: key for key, val in six.iteritems(self._tensor_map)})
        self._transposes = {}

        self._prepare(self._graph)

        if not self._infer_shapes:
            errors = []
            for op in graph.operations:
                transform = self._transforms.get(op.type)
                if transform is None and self._mirror_unsupported:
                    continue

                if isinstance(transform, Transform) and transform.type is None:
                    continue

                error = self._error_message(op, transform)
                if error is not None:
                    errors.append(error)

            if len(errors):
                raise ConversionError("Found {} operator(s) that cannot be converted\n{}".format(len(errors), "\n".join(errors)))

        for op in graph.operations:
            transform = self._transforms.get(op.type)
            if isinstance(transform, Transform) and transform.type is None:
                continue

            if self._infer_shapes:
                error = self._error_message(op, transform)
                if error is not None:
                    raise ConversionError(error)

            count = len(self._graph.operations)

            if transform is not None:
                self._convert(op, transform)
            elif self._mirror_unsupported:
                self._mirror(op)

            if self._infer_shapes:
                from nnef.shapes import _infer_op_shapes
                for op in self._graph.operations[count:]:
                    input_shapes = [list(tensor.shape) for tensor in op.inputs]
                    if not isinstance(op.inputs, tuple):
                        input_shapes = (input_shapes,)

                    output_counts = [len(op.outputs)] if not isinstance(op.outputs, tuple) else [None] * len(op.outputs)
                    output_shapes = _infer_op_shapes(op.type, op.attribs, input_shapes, output_counts,
                                                     custom_shapes=self._custom_shapes)
                    if not isinstance(op.outputs, tuple):
                        output_shapes = output_shapes[0]

                    for output, shape in zip(op.outputs, output_shapes):
                        output.shape = tuple(shape)

        for tensor, shape in self._transposes.items():
            tensor.shape = shape

        self._graph.remove_tensors([tensor for tensor in self._graph.tensors
                                    if len(tensor.producers) == 0 and len(tensor.consumers) == 0])

        self._graph.inputs = tuple(self._tensor_map[tensor] for tensor in graph.inputs if self._tensor_map[tensor].graph)
        self._graph.outputs = tuple(self._tensor_map[tensor] for tensor in graph.outputs if self._tensor_map[tensor].graph)

        return self._graph

    def tensor_mapping(self):
        return {key.name: value.name for key, value in six.iteritems(self._tensor_map)
                if value.graph == self._graph and key.name is not None and value.name is not None}

    def _global_attribs(self):
        return {}

    def _prepare(self, graph):
        pass

    def _check_conditions(self, op, transform):
        op_inputs = list(self._tensor_map[tensor] for tensor in op.inputs)
        op_outputs = list(self._tensor_map[tensor] for tensor in op.outputs)
        op_attribs = self._add_default_attribs(op.attribs, transform.defaults, op_inputs, op_outputs, op.type, op.name)

        using = {'_type_': op.type, '_name_': op.name, **self._global_attribs()}
        for key, item in six.iteritems(transform.using):
            value = self._evaluate(op_attribs, op_inputs, op_outputs, item, using)
            self._check_value(value, 'using', key, op.type, op.name)
            using[key] = value

        error = None
        if transform.cond is not None:
            for condition, message in transform.cond.items():
                if not self._evaluate(op_attribs, op_inputs, op_outputs, condition, using):
                    error = message if error is None else error + ', ' + message

        return error

    def _error_message(self, op, transform):
        if transform is None:
            return "Conversion of operator '{}' is not implemented".format(op.type)

        message = self._check_conditions(op, transform)
        if message is not None:
            attribs = {key: value for key, value in six.iteritems(op.attribs) if not key.startswith('_')}
            input_shapes = ", ".join(str(tensor.shape) for tensor in op.inputs)
            output_shapes = ", ".join(str(tensor.shape) for tensor in op.outputs)
            return "Conversion of operator '{}' is not possible: {}"\
                   "\n  attributes: {}\n  input-shapes: {}\n  output-shapes: {}"\
                .format(op.type, message, attribs, input_shapes, output_shapes)

        return None

    def _convert(self, op, transform):
        op_inputs = list(self._tensor_map[tensor] for tensor in op.inputs)
        op_outputs = list(self._tensor_map[tensor] for tensor in op.outputs)
        op_attribs = self._add_default_attribs(op.attribs, transform.defaults, op_inputs, op_outputs, op.type, op.name)

        using = {'_type_': op.type, '_name_': op.name, **self._global_attribs()}
        for key, item in six.iteritems(transform.using):
            value = self._evaluate(op_attribs, op_inputs, op_outputs, item, using)
            self._check_value(value, 'using', key, op.type, op.name)
            using[key] = value

        type = self._evaluate(op_attribs, op_inputs, op_outputs, transform.type, using)
        self._check_value(type, 'field', 'type', op.type, op.name)

        name = self._evaluate(op_attribs, op_inputs, op_outputs, transform.name, using)
        self._check_value(name, 'field', 'name', op.type, op.name)

        attribs = {}
        for key, item in six.iteritems(transform.attribs):
            value = self._evaluate(op_attribs, op_inputs, op_outputs, item, using)
            if value is not None:
                attribs[key] = value

        for key, value in six.iteritems(attribs):
            self._check_value(value, 'attribute', key, op.type, op.name)

        if isinstance(transform.inputs, list):
            inputs = self._evaluate_tensor_list(op_attribs, op_inputs, op_outputs, transform.inputs, using)
        elif isinstance(transform.inputs, tuple):
            inputs = tuple(self._filter_none(self._evaluate(op_attribs, op_inputs, op_outputs, item, using)
                                             for item in transform.inputs))
        else:
            inputs = (self._evaluate(op_attribs, op_inputs, op_outputs, transform.inputs, using),)

        for idx, item in enumerate(inputs):
            self._check_value(item, 'input', idx, op.type, op.name, tensor=True)

        offset = len(self._graph.operations)

        if isinstance(transform.outputs, list):
            outputs = self._evaluate_tensor_list(op_attribs, op_inputs, op_outputs, transform.outputs, using)
        elif isinstance(transform.outputs, tuple):
            outputs = tuple(self._filter_none(self._evaluate(op_attribs, op_inputs, op_outputs, item, using)
                                              for item in transform.outputs))
        else:
            outputs = (self._evaluate(op_attribs, op_inputs, op_outputs, transform.outputs, using),)

        for idx, item in enumerate(outputs):
            self._check_value(item, 'output', idx, op.type, op.name, tensor=True)

        op = Operation(self._graph, type=type, name=name, attribs=attribs, inputs=inputs, outputs=outputs,
                       custom=transform.custom)
        self._graph.reverse(offset)
        return op

    def _mirror(self, op):
        inputs_type = tuple if isinstance(op.inputs, tuple) else list
        outputs_type = tuple if isinstance(op.outputs, tuple) else list

        op_inputs = inputs_type(self._tensor_map[tensor] for tensor in op.inputs)
        op_outputs = outputs_type(self._tensor_map[tensor] for tensor in op.outputs)

        return Operation(self._graph, type=op.type, name=op.name, attribs=op.attribs,
                         inputs=op_inputs, outputs=op_outputs, custom=True)

    def _add_default_attribs(self, attribs, defaults, inputs, outputs, op_type, op_name):
        if defaults is None:
            return attribs

        attribs = dict(attribs)
        for key, value in six.iteritems(defaults):
            if key not in attribs:
                value = self._evaluate({}, inputs, outputs, value)
                self._check_value(value, 'default', key, op_type, op_name)
                attribs[key] = value

        return attribs

    def _evaluate(self, attribs, inputs, outputs, arg, using={}):
        if isinstance(arg, str) and arg[0] == '!':
            try:
                return eval(arg[1:], {'I': inputs, 'O': outputs, **attribs, **using, **self._callables,
                                      'np': np, 'math': math})
            except Exception as e:
                return e
        else:
            return arg

    def _evaluate_tensor_list(self, attribs, inputs, outputs, arg, using):
        values = []
        for item in arg:
            value = self._evaluate(attribs, inputs, outputs, item, using)
            if isinstance(value, Tensor) or isinstance(value, Exception):
                values.append(value)
            else:
                assert isinstance(value, (list, tuple))
                values += list(value)
        return values

    def _filter_none(self, items):
        return (item for item in items if item is not None)

    def _check_value(self, value, kind, key, op_type, op_name, tensor=False):
        if isinstance(value, Exception):
            raise ConversionError("Could not evaluate {kind} '{key}' while converting operator '{type}'; {err}: {cause}"
                                  .format(kind=kind, key=key, type=op_type, name=op_name,
                                          err=type(value).__name__, cause=str(value) or repr(value)))
        if tensor and not isinstance(value, Tensor):
            raise ConversionError("While converting operator '{op_type}', {kind} '{key}' must result in a tensor, "
                                  "but found {value_type}"
                                  .format(kind=kind, key=key, op_type=op_type, value_type=type(value)))

    def _copy_tensor_(self, tensor):
        return Tensor(self._graph, name=tensor.name, dtype=tensor.dtype, shape=tensor.shape,
                      data=tensor.data, quant=copy.deepcopy(tensor.quant))

    def _read_constant(self, tensor, type):
        raise NotImplementedError()

    def _make_constant(self, graph, dtype, value, inline):
        raise NotImplementedError()

    def _const_operation(self, output, value):
        raise NotImplementedError()

    def _transpose_operation(self, input, output, perm):
        raise NotImplementedError()

    def _reshape_operation(self, input, output, shape):
        raise NotImplementedError()

    def _squeeze_operation(self, input, output, axes):
        raise NotImplementedError()

    def _unsqueeze_operation(self, input, output, axes):
        raise NotImplementedError()

    def _scale_operation(self, input, output, scalar):
        raise NotImplementedError()

    @staticmethod
    def _permute(items, perm):
        permuted = list(items)
        for i in range(len(perm)):
            permuted[i] = items[perm[i]]
        return type(items)(permuted)

    @staticmethod
    def _inverse_permute(items, perm):
        permuted = list(items)
        for i in range(len(perm)):
            permuted[perm[i]] = items[i]
        return type(items)(permuted)

    def _working_shape(self, tensor):
        return self._transposes.get(tensor) or tensor.shape

    def _pre_transpose(self, input, perm):
        shape = self._permute(self._working_shape(input), perm)
        output = Tensor(input.graph, dtype=input.dtype, shape=shape, quant=copy.deepcopy(input.quant))
        self._transpose_operation(input, output, perm)
        return output

    def _post_transpose(self, output, perm):
        shape = self._inverse_permute(self._working_shape(output), perm)
        input = Tensor(output.graph, dtype=output.dtype, shape=shape, quant=copy.deepcopy(output.quant))
        self._transpose_operation(input, output, perm)
        return input

    def _pre_squeeze(self, input, axes):
        shape = self.squeeze_shape(self._working_shape(input), axes)
        output = Tensor(input.graph, dtype=input.dtype, shape=shape, quant=copy.deepcopy(input.quant))
        self._squeeze_operation(input, output, axes)
        return output

    def _pre_unsqueeze(self, input, axes):
        shape = self.unsqueeze_shape(self._working_shape(input), axes)
        output = Tensor(input.graph, dtype=input.dtype, shape=shape, quant=copy.deepcopy(input.quant))
        self._unsqueeze_operation(input, output, axes)
        return output

    def _post_squeeze(self, output, axes):
        shape = self.unsqueeze_shape(self._working_shape(output), axes)
        input = Tensor(output.graph, dtype=output.dtype, shape=shape, quant=copy.deepcopy(output.quant))
        self._squeeze_operation(input, output, axes)
        return input

    def _post_unsqueeze(self, output, axes):
        shape = self.squeeze_shape(self._working_shape(output), axes)
        input = Tensor(output.graph, dtype=output.dtype, shape=shape, quant=copy.deepcopy(output.quant))
        self._unsqueeze_operation(input, output, axes)
        return input

    def _reshape(self, input, shape):
        output = Tensor(input.graph, dtype=input.dtype, shape=shape, quant=copy.deepcopy(input.quant))
        self._reshape_operation(input, output, shape)
        return output

    def _shape_of(self, value):
        if isinstance(value, (list, tuple)):
            length = len(value)
            return (length,) + self._shape_of(value[0]) if length > 0 else (0,)
        elif isinstance(value, np.ndarray):
            return value.shape
        else:
            return ()

    def squeeze_shape(self, shape, axes):
        return type(shape)(shape[i] for i in range(len(shape)) if i not in axes)

    def unsqueeze_shape(self, shape, axes):
        for axis in axes:
            shape = shape[:axis] + (1,) + shape[axis:]
        return shape

    def transposing(self, tensor):
        return tensor in self._transposes

    def nxc_to_ncx(self, items, cond=True):
        return items[0:1] + items[-1:] + items[1:-1] if cond else items

    def ncx_to_nxc(self, items, cond=True):
        return items[0:1] + items[2:] + items[1:2] if cond else items

    def xcn_to_ncx(self, items, cond=True):
        return items[-1:] + items[-2:-1] + items[:-2] if cond else items

    def ncx_to_xcn(self, items, cond=True):
        return items[2:] + items[1:2] + items[0:1] if cond else items

    def cxn_to_ncx(self, items, cond=True):
        return items[-1:] + items[:-1] if cond else items

    def ncx_to_cxn(self, items, cond=True):
        return items[1:] + items[:1] if cond else items

    def nxc_to_ncx_perm(self, rank):
        return self.nxc_to_ncx(list(range(rank)))

    def ncx_to_nxc_perm(self, rank):
        return self.ncx_to_nxc(list(range(rank)))

    def xcn_to_ncx_perm(self, rank):
        return self.xcn_to_ncx(list(range(rank)))

    def ncx_to_xcn_perm(self, rank):
        return self.ncx_to_xcn(list(range(rank)))

    def cxn_to_ncx_perm(self, rank):
        return self.cxn_to_ncx(list(range(rank)))

    def ncx_to_cxn_perm(self, rank):
        return self.ncx_to_cxn(list(range(rank)))

    def axis_nxc_to_ncx(self, value, rank):
        if isinstance(value, (list, tuple)):
            return type(value)(self.axis_nxc_to_ncx(v, rank) for v in value)
        else:
            if value < 0:
                value += rank
            return 0 if value == 0 else 1 if value == rank - 1 else value + 1

    def axis_ncx_to_nxc(self, value, rank):
        if isinstance(value, (list, tuple)):
            return type(value)(self.axis_ncx_to_nxc(v, rank) for v in value)
        else:
            if value < 0:
                value += rank
            return 0 if value == 0 else rank - 1 if value == 1 else value - 1

    def ensure_positive(self, axis, rank):
        if isinstance(axis, (list, tuple)):
            return type(axis)(self.ensure_positive(item, rank) for item in axis)
        else:
            return axis + rank if axis < 0 else axis

    def as_const(self, tensor, type=None):
        return self._read_constant(self._tensor_map[tensor], type=type)

    def is_const(self, tensor, type=None):
        tensor = self._tensor_map[tensor]
        if tensor.data is not None:
            return True
        try:
            self._read_constant(tensor, type=type)
            return True
        except ConversionError:
            return False

    def is_zero(self, tensor):
        return self.is_const(tensor) and len(tensor.shape) == 0 and self.as_const(tensor) == 0

    def as_tensor(self, arg, dtype, inline=None):
        return self._make_constant(self._graph, dtype=dtype, value=arg, inline=inline)

    def new_tensor(self, shape, dtype):
        return Tensor(self._graph, dtype=dtype, shape=shape)

    def is_integer_upsample(self, input_shape, output_shape):
        return all(output % input == 0 for input, output in zip(input_shape, output_shape))

    def is_integer_downsample(self, input_shape, output_shape):
        return all(input % output == 0 for input, output in zip(input_shape, output_shape))

    def upsample_factor(self, input_shape, output_shape):
        return [output // input for input, output in zip(input_shape, output_shape)]

    def downsample_factor(self, input_shape, output_shape):
        return [input // output for input, output in zip(input_shape, output_shape)]

    def from_numpy(self, array, type=None):
        return types.from_numpy(array, type)

    def to_numpy(self, value, dtype=None):
        return types.to_numpy(value, dtype)

    def flexible_batch(self, output_shape, batch):
        return [0] + output_shape[1:] if output_shape[0] == batch else output_shape

    def fixed_batch(self, output_shape, batch):
        return [batch] + output_shape[1:] if output_shape[0] == 0 else output_shape


class ConverterToNNEF(Converter):

    _DtypeFromNumpy = {
        np.float16: 'scalar',
        np.float32: 'scalar',
        np.float64: 'scalar',
        np.int8: 'integer',
        np.uint8: 'integer',
        np.int16: 'integer',
        np.uint16: 'integer',
        np.int32: 'integer',
        np.uint32: 'integer',
        np.int64: 'integer',
        np.uint64: 'integer',
        np.bool_: 'logical',
    }

    def __init__(self, transforms, functions=None, mirror_unsupported=False, infer_shapes=False, custom_shapes=None):
        Converter.__init__(self, transforms, functions, mirror_unsupported, infer_shapes, custom_shapes)

    def _insert_externals_and_constants(self, graph):
        for tensor in graph.tensors:
            mapped = self._tensor_map[tensor]
            if mapped.producer is None and len(mapped.consumers) > 0:
                if mapped.data is None:
                    Operation(graph, type='external', inputs=(), outputs=tensor,
                              attribs={'shape': list(tensor.shape), 'dtype': tensor.dtype})
                else:
                    Operation(graph, type='constant', inputs=(), outputs=tensor,
                              attribs={'shape': list(tensor.shape), 'dtype': tensor.dtype, 'value': mapped.data})

    def _ensure_valid_ids(self, graph):
        if graph.name is not None:
            graph.name = self.ensure_valid_id(graph.name)

        for tensor in graph.tensors:
            if tensor.name is not None:
                tensor.name = self.ensure_valid_id(tensor.name)

    def _make_constant(self, graph, dtype, value, inline):
        if isinstance(value, tuple):
            value = list(value)
        shape = value.shape if isinstance(value, np.ndarray) else (len(value),) if isinstance(value, list) else ()
        isarray = isinstance(value, np.ndarray) or isinstance(value, list)

        tensor = Tensor(graph, dtype=dtype, shape=shape)
        if inline:
            tensor.data = types.to_numpy(value, dtype)
        else:
            self._const_operation(tensor, value=value if isarray else [value])
        return tensor

    def _const_operation(self, output, value):
        Operation(output.graph, type='constant', inputs=(), outputs=output,
                  attribs={'value': value, 'dtype': output.dtype, 'shape': list(output.shape)})

    def _transpose_operation(self, input, output, perm):
        Operation(input.graph, type='transpose', inputs=input, outputs=output, attribs={'axes': perm})

    def _reshape_operation(self, input, output, shape):
        Operation(input.graph, type='reshape', inputs=input, outputs=output, attribs={'shape': list(shape)})

    def _squeeze_operation(self, input, output, axes):
        Operation(input.graph, type='squeeze', inputs=input, outputs=output, attribs={'axes': axes})

    def _unsqueeze_operation(self, input, output, axes):
        Operation(input.graph, type='unsqueeze', inputs=input, outputs=output, attribs={'axes': axes})

    def _scale_operation(self, input, output, scalar):
        if not isinstance(scalar, Tensor):
            scalar = self.as_tensor(scalar, np.float32)

        Operation(input.graph, type='mul', inputs=(input, scalar), outputs=output)

    def _bias_operation(self, input, output, bias):
        if not isinstance(bias, Tensor):
            bias = self.as_tensor(bias, np.float32)

        Operation(input.graph, type='add', inputs=(input, bias), outputs=output)

    def _transform_constant(self, tensor, func):
        data = func(tensor.producer.attribs['value'])
        tensor.shape = data.shape
        tensor.producer.attribs['value'] = data
        tensor.producer.attribs['shape'] = list(data.shape)

    @staticmethod
    def remove_unused_constants(graph):
        ops = [op for op in graph.operations if op.type == 'constant' and not op.output.has_consumer]
        tensors = [op.output for op in ops]
        graph.outputs = [tensor for tensor in graph.outputs if tensor not in tensors]
        graph.remove_operations(ops, unlink=True)
        graph.remove_tensors(tensors)

    @staticmethod
    def inline_scalar_constants(graph):
        for op in graph.operations:
            if op.type == 'constant':
                value = op.attribs['value']
                if not isinstance(value, np.ndarray):
                    value = np.array(value, op.output.dtype).reshape(op.output.shape)
                if len(value.shape) == 0:
                    op.output.data = value
                    graph.remove_operation(op, unlink=True)

    @staticmethod
    def convert_constants_to_variables(graph):
        variables = 0
        for op in graph.operations:
            if op.type == 'constant':
                value = op.attribs['value']
                if isinstance(value, np.ndarray):
                    variables += 1
                    op.type = 'variable'
                    op.attribs['label'] = op.name if op.name else 'variable' + str(variables)
                    op.output.data = value
                    del op.attribs['value']

    @staticmethod
    def ensure_valid_id(name):
        return re.sub('[^_0-9a-zA-Z]+', '_', name)

    def nnef_dtype(self, dtype):
        return ConverterToNNEF._DtypeFromNumpy[dtype]


class ConverterFromNNEF(Converter):

    @staticmethod
    def decomposed_operations():
        return ['separable_conv', 'separable_deconv', 'rms_pool',
                'local_mean_normalization', 'local_variance_normalization', 'local_contrast_normalization',
                'l1_normalization', 'moments']

    def __init__(self, transforms, functions=None, mirror_unsupported=False):
        Converter.__init__(self, transforms, functions, mirror_unsupported)

    @staticmethod
    def convert_variables_to_constants(graph):
        for op in graph.operations:
            if op.type == 'variable':
                op.type = 'constant'
                op.attribs['value'] = op.output.data
                del op.attribs['label']

    @staticmethod
    def fill_data_in_constants(graph):
        for op in graph.operations:
            if op.type == 'constant':
                op.output.data = op.attribs['value']

    def _is_constant(self, tensor):
        if tensor.producer:
            return tensor.producer.type == 'constant'
        else:
            return tensor.data is not None

    def _read_constant(self, tensor, type):
        if tensor.data is not None:
            value = tensor.data
        elif tensor.producer and tensor.producer.type == 'constant':
            value = tensor.producer.attribs['value']
        else:
            raise ConversionError('trying to evaluate non-constant tensor')

        return types.from_numpy(value, type=type) if isinstance(value, np.ndarray) else \
            types.cast(value, type=type) if type is not None else value
