# Copyright (c) 2017-2025 The Khronos Group Inc.
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
from ..model.utils import remove_unused_tensors
from ..utils import types
from skriptnd import PlaceholderExpr
from enum import Enum
import numpy as np
import functools
import inspect
import math
import copy
import six
import re


class Transform:

    def __init__(self, type, name=None, inputs=None, outputs=None, dtypes=None, attribs=None,
                 defaults=None, using=None, cond=None, custom=False):
        self.type = type
        self.name = name or '!_name_'
        self.inputs = inputs or ()
        self.outputs = outputs or ()
        self.attribs = attribs or {}
        self.dtypes = dtypes or {}
        self.defaults = defaults
        self.using = using or {}
        self.cond = cond
        self.custom = custom

    def with_type(self, type):
        return Transform(type=type, name=self.name, inputs=self.inputs, outputs=self.outputs, attribs=self.attribs,
                         dtypes=self.dtypes, defaults=self.defaults, using=self.using, cond=self.cond, custom=self.custom)


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
    def atomic_operations():
        return []

    @staticmethod
    def decomposed_operations():
        return []

    @staticmethod
    def defined_operations():
        return {}

    @staticmethod
    def defined_operation_dependencies():
        return {}

    @staticmethod
    def defined_shapes():
        return {}

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

    def should_skip_conversion(self, op):
        return False

    def __init__(self, transforms, functions=None, mirror_unsupported=False, enforce_known_shapes=False):
        self._model = None
        self._graph = None
        self._transforms = transforms
        self._callables = self.find_public_methods(self)
        if functions:
            self._callables.update({name: functools.partial(func, self) for name, func in six.iteritems(functions)})
        self._mirror_unsupported = mirror_unsupported
        self._enforce_known_shapes = enforce_known_shapes

    def __call__(self, model):
        if self._enforce_known_shapes:
            unknown_tensors = [tensor for tensor in model.tensors if tensor.shape is None and len(tensor.consumers)]
            if len(unknown_tensors):
                names = ["'{}'".format(tensor.name) for tensor in unknown_tensors if tensor.name]
                raise ConversionError(("Input graph contains tensors with unknown shape: " +
                                      ", ".join(names) if len(names) else "(no names)") +
                                      "\nTry the --fold-constants option to eliminate unnecessary constant sub-graphs")

        self._model = Model(name=model.name)
        self._tensor_map = {}
        self._graph_map = {}

        for graph in model.graphs:
            self._graph_map[graph] = self._graph = Graph(self._model, graph.name)

            for tensor in graph.tensors:
                self._tensor_map[tensor] = Tensor(self._graph, name=tensor.name, dtype=tensor.dtype, shape=tensor.shape,
                                                  data=tensor.data, quant=copy.deepcopy(tensor.quant), variable=tensor.is_variable)

            self._graph.inputs = tuple(self._map_tensor(tensor) for tensor in graph.inputs)
            self._graph.outputs = tuple(self._map_tensor(tensor) for tensor in graph.outputs)

        self._tensor_map.update({val: key for key, val in six.iteritems(self._tensor_map)})
        self._transposes = {}

        self._prepare(self._model)

        errors = []
        for graph in model.graphs:
            self._graph = self._graph_map[graph]
            for op in graph.operations:
                transform = self._transforms.get(op.type)
                if transform is None and self._mirror_unsupported:
                    continue

                if isinstance(transform, Transform) and transform.type is None:
                    continue

                if self.should_skip_conversion(op):
                    continue

                error = self._error_message(op, model.version, transform)
                if error is not None:
                    errors.append(error)

        if len(errors):
            raise ConversionError("Found {} operator(s) that cannot be converted\n{}"
                                  .format(len(errors), "\n".join(e for e in errors)))

        for graph in model.graphs:
            self._graph = self._graph_map[graph]
            for op in graph.operations:
                transform = self._transforms.get(op.type)
                if isinstance(transform, Transform) and transform.type is None:
                    continue

                if self.should_skip_conversion(op):
                    continue

                if transform is not None:
                    self._convert(op, model.version, transform)
                elif self._mirror_unsupported:
                    self._mirror(op)

        for tensor, shape in self._transposes.items():
            tensor.shape = shape

        remove_unused_tensors(self._model)

        return self._model

    def tensor_mapping(self):
        return {key.name: value.name for key, value in six.iteritems(self._tensor_map)
                if value.graph == self._model and key.name is not None and value.name is not None}

    def _global_attribs(self):
        return {}

    def _prepare(self, graph):
        pass

    def _check_conditions(self, op, version, transform):
        op_inputs = list(self._map_tensor(tensor) for tensor in op.inputs)
        op_outputs = list(self._map_tensor(tensor) for tensor in op.outputs)
        op_attribs = self._remap_attribs(op.attribs, transform.defaults, op_inputs, op_outputs, op.type, op.name, version)

        using = {'_type_': op.type, '_name_': op.name, '_version_': version, **self._global_attribs()}
        for key, item in six.iteritems(transform.using):
            value = self._evaluate(op_attribs, op_inputs, op_outputs, item, using)
            self._check_value(value, 'local expression', key, op.type, op.name)
            using[key] = value

        error = None
        if transform.cond is not None:
            for condition, message in transform.cond.items():
                if not self._evaluate(op_attribs, op_inputs, op_outputs, condition, using):
                    error = message if error is None else error + ', ' + message

        return error

    def _error_message(self, op, version, transform):
        if transform is None:
            return "Conversion of operator '{}' is not implemented".format(op.type)

        try:
            message = self._check_conditions(op, version, transform)
        except ConversionError as e:
            message = str(e)

        if message is not None:
            attribs = {key: value for key, value in six.iteritems(op.attribs) if not key.startswith('_')}
            input_shapes = ", ".join(str([t.shape for t in tensor]) if isinstance(tensor, list) else str(tensor.shape)
                                     for tensor in op.inputs)
            output_shapes = ", ".join(str([t.shape for t in tensor]) if isinstance(tensor, list) else str(tensor.shape)
                                      for tensor in op.outputs)
            return "Conversion of operator '{}' is not possible: {}"\
                   "\n  attributes: {}\n  input-shapes: {}\n  output-shapes: {}"\
                .format(op.type, message, attribs, input_shapes, output_shapes)

        return None

    def _map_tensor(self, tensor):
        return [self._tensor_map[t] for t in tensor] if isinstance(tensor, list) else self._tensor_map[tensor]

    def _convert(self, op, version, transform):
        op_inputs = tuple(self._map_tensor(tensor) for tensor in op.inputs)
        op_outputs = tuple(self._map_tensor(tensor) for tensor in op.outputs)
        op_attribs = self._remap_attribs(op.attribs, transform.defaults, op_inputs, op_outputs, op.type, op.name, version)

        using = {'_type_': op.type, '_name_': op.name, '_version_': version, **self._global_attribs()}
        for key, item in six.iteritems(transform.using):
            value = self._evaluate(op_attribs, op_inputs, op_outputs, item, using)
            self._check_value(value, 'local expression', key, op.type, op.name)
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

        dtypes = {}
        for key, item in six.iteritems(transform.dtypes):
            value = self._evaluate(op_attribs, op_inputs, op_outputs, item, using)
            if value is not None:
                dtypes[key] = value

        for key, value in six.iteritems(dtypes):
            self._check_value(value, 'dtype', key, op.type, op.name)

        if isinstance(transform.inputs, tuple):
            inputs = tuple(self._filter_none(self._evaluate(op_attribs, op_inputs, op_outputs, item, using)
                                             for item in transform.inputs))
        else:
            inputs = self._evaluate(op_attribs, op_inputs, op_outputs, transform.inputs, using)
            if isinstance(inputs, (Tensor, Exception)):
                inputs = (inputs,)

        for idx, item in enumerate(inputs):
            self._check_value(item, 'input', idx, op.type, op.name, tensor=True)

        offset = len(self._graph.operations)

        if isinstance(transform.outputs, tuple):
            outputs = tuple(self._filter_none(self._evaluate(op_attribs, op_inputs, op_outputs, item, using)
                                              for item in transform.outputs))
        else:
            outputs = self._evaluate(op_attribs, op_inputs, op_outputs, transform.outputs, using)
            if isinstance(outputs, (Tensor, Exception)):
                outputs = (outputs,)

        for idx, item in enumerate(outputs):
            self._check_value(item, 'output', idx, op.type, op.name, tensor=True)

        op = Operation(self._graph, type=type, name=name, dtypes=dtypes, attribs=attribs, inputs=inputs, outputs=outputs,
                       custom=transform.custom)
        self._graph.reverse(offset)
        return op

    def _mirror(self, op):
        op_inputs = tuple(self._map_tensor(tensor) for tensor in op.inputs)
        op_outputs = tuple(self._map_tensor(tensor) for tensor in op.outputs)

        return Operation(self._graph, type=op.type, name=op.name, dtypes=op.dtypes, attribs=op.attribs,
                         inputs=op_inputs, outputs=op_outputs, custom=True)

    def _remap_attribs(self, attribs, defaults, inputs, outputs, op_type, op_name, version):
        attribs = {key: self._tensor_map[value] if isinstance(value, Tensor) else
                        self._graph_map[value] if isinstance(value, Graph) else value
                   for key, value in six.iteritems(attribs)}

        if defaults is not None:
            predefined = {'_type_': op_type, '_name_': op_name, '_version_': version}
            for key, value in six.iteritems(defaults):
                if key not in attribs:
                    value = self._evaluate(predefined, inputs, outputs, value)
                    self._check_value(value, 'expression', key, op_type, op_name)
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

    def _filter_none(self, items):
        return (item for item in items if item is not None)

    def _check_value(self, value, kind, key, op_type, op_name, tensor=False):
        if isinstance(value, Exception):
            err_type = type(value).__name__ + ": " if type(value) != ConversionError else ""
            raise ConversionError("Could not evaluate {kind} '{key}' while converting operator '{name}' of type '{type}'; {err}{cause}"
                                  .format(kind=kind, key=key, type=op_type, name=op_name or '', err=err_type, cause=str(value) or repr(value)))
        if tensor and value is not None and not isinstance(value, Tensor) and not \
                (isinstance(value, list) and all(isinstance(item, Tensor) for item in value)):
            raise ConversionError("While converting operator '{name}' of type '{op_type}', {kind} '{key}' must result in a tensor, "
                                  "but found {value_type}"
                                  .format(kind=kind, key=key, name=op_name, op_type=op_type, value_type=type(value)))

    def _read_constant(self, tensor, type, flat):
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

    def _bias_operation(self, input, output, bias):
        raise NotImplementedError()

    def _split_operation(self, input, outputs, axis, count):
        raise NotImplementedError()

    def _stack_operation(self, input, outputs, axis):
        raise NotImplementedError()

    def _unstack_operation(self, input, outputs, axis):
        raise NotImplementedError()

    def _concat_operation(self, inputs, output, axis):
        raise NotImplementedError()

    def _add_operation(self, lhs, rhs, output):
        raise NotImplementedError()

    def _mul_operation(self, lhs, rhs, output):
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

    def _split(self, input, axis, count):
        shape = list(input.shape)
        shape[axis] /= count
        outputs = [Tensor(input.graph, dtype=input.dtype, shape=tuple(shape), quant=copy.deepcopy(input.quant))]
        self._split_operation(input, outputs, axis, count)
        return outputs

    def _concat(self, inputs, axis):
        shape = list(inputs[0].shape)
        shape[axis] = sum(input.shape[axis] for input in inputs)
        output = Tensor(inputs[0].graph, dtype=inputs[0].dtype, shape=tuple(shape), quant=copy.deepcopy(inputs[0].quant))
        self._concat_operation(inputs, output, axis)
        return output

    def _add(self, lhs, rhs):
        output = Tensor(lhs.graph, dtype=lhs.dtype, shape=lhs.shape, quant=copy.deepcopy(lhs.quant))
        self._add_operation(lhs, rhs, output)
        return output

    def _mul(self, lhs, rhs):
        output = Tensor(lhs.graph, dtype=lhs.dtype, shape=lhs.shape, quant=copy.deepcopy(lhs.quant))
        self._mul_operation(lhs, rhs, output)
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

    def as_const(self, tensor, type=None, flat=False):
        return self._read_constant(self._tensor_map[tensor], type=type, flat=flat)

    def is_const(self, tensor):
        tensor = self._tensor_map[tensor]
        if tensor.data is not None:
            return True
        try:
            self._read_constant(tensor, type=None, flat=True)
            return True
        except ConversionError:
            return False

    def is_const_int_max(self, arg):
        if not self.is_const(arg):
            return False
        value = self.as_const(arg)
        if isinstance(value, list):
            value = value[0]
        return value >= _INT_MAX

    def is_zero(self, tensor):
        return self.is_const(tensor) and len(tensor.shape) == 0 and self.as_const(tensor) == 0

    def as_list(self, obj):
        return obj.tolist() if isinstance(obj, np.ndarray) else list(obj)

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
        return types.from_numpy(array, type) if isinstance(array, np.ndarray) else array

    def to_numpy(self, value, dtype=None):
        return types.to_numpy(value, dtype)

    def flexible_batch(self, output_shape, batch):
        return [0] + output_shape[1:] if output_shape[0] == batch else output_shape

    def fixed_batch(self, output_shape, batch):
        return [batch] + output_shape[1:] if output_shape[0] == 0 else output_shape

    def leading_zeros(self, shape):
        for i, s in enumerate(shape):
            if s != 0:
                return i
        return len(shape)


class ConverterToSkriptND(Converter):

    _DtypeFromNumpy = {
        np.float16: 'real',
        np.float32: 'real',
        np.float64: 'real',
        np.int8: 'int',
        np.uint8: 'int',
        np.int16: 'int',
        np.uint16: 'int',
        np.int32: 'int',
        np.uint32: 'int',
        np.int64: 'int',
        np.uint64: 'int',
        np.bool_: 'bool',
        np.str_: 'str',
    }

    @staticmethod
    def shape_expr_args(op_type):
        raise NotImplementedError()

    def __init__(self, transforms, functions=None, mirror_unsupported=False):
        Converter.__init__(self, transforms, functions, mirror_unsupported)

    def _make_constant(self, graph, dtype, value, inline):
        if isinstance(value, tuple):
            value = list(value)
        shape = value.shape if isinstance(value, np.ndarray) else (len(value),) if isinstance(value, list) else ()

        return Tensor(graph, dtype=dtype, shape=shape, data=value)

    def _const_operation(self, output, value):
        pass

    def _transpose_operation(self, input, output, perm):
        Operation(self._graph, type='layout.transpose', inputs=input, outputs=output, attribs={'perm': perm})

    def _reshape_operation(self, input, output, shape):
        Operation(self._graph, type='layout.reshape', inputs=input, outputs=output, attribs={'shape': list(shape)})

    def _squeeze_operation(self, input, output, axes):
        Operation(self._graph, type='layout.squeeze', inputs=input, outputs=output, attribs={'axes': axes})

    def _unsqueeze_operation(self, input, output, axes):
        Operation(self._graph, type='layout.unsqueeze', inputs=input, outputs=output, attribs={'axes': axes})

    def _split_operation(self, input, outputs, axis, count):
        Operation(self._graph, type='layout.split', inputs=input, outputs=outputs,
                  attribs={'axis': axis, 'ratios': [1] * count})

    def _stack_operation(self, inputs, output, axis):
        Operation(self._graph, type='layout.stack', inputs=inputs, outputs=output,
                  attribs={'axis': axis})

    def _unstack_operation(self, input, outputs, axis):
        Operation(self._graph, type='layout.unstack', inputs=input, outputs=outputs,
                  attribs={'axis': axis})

    def _concat_operation(self, inputs, output, axis):
        Operation(self._graph, type='layout.concat', inputs=inputs, outputs=output, attribs={'axis': axis})

    def _scale_operation(self, input, output, scalar):
        if not isinstance(scalar, Tensor):
            scalar = self.as_tensor(scalar, np.float32)

        Operation(self._graph, type='mul', inputs=(input, scalar), outputs=output)

    def _bias_operation(self, input, output, bias):
        if not isinstance(bias, Tensor):
            bias = self.as_tensor(bias, np.float32)

        Operation(self._graph, type='add', inputs=(input, bias), outputs=output)

    def _mul_operation(self, lhs, rhs, output):
        Operation(self._graph, type='mul', inputs=(lhs, rhs), outputs=output)

    def _add_operation(self, lhs, rhs, output):
        Operation(self._graph, type='add', inputs=(lhs, rhs), outputs=output)

    def _zero_copy_operation(self, input, output):
        Operation(input.graph, type='', inputs=input, outputs=output)

    def _transform_constant(self, tensor, func):
        tensor.data = func(tensor.data)
        tensor.shape = tensor.data.shape

    def _remove_unused_constants(self, model):
        for graph in model.graphs:
            removed = [tensor for tensor in graph.tensors
                       if tensor.data is not None and not tensor.has_consumer and not tensor in graph.outputs]
            graph.inputs = tuple(tensor for tensor in graph.inputs if tensor not in removed)
            graph.remove_tensors(removed)

    def _add_zero_copy_for_constant_outputs(self, model):
        for graph in model.graphs:
            graph.outputs = tuple(self.zero_copy(tensor) if tensor.is_constant or tensor.is_variable else tensor
                                  for tensor in graph.outputs)

    def _eliminate_empty_subgraphs(self, model):
        for graph in model.graphs:
            for op in graph.operations:
                for key, value in op.attribs.items():
                    if isinstance(value, Graph) and len(value.operations) == 0 and len(value.outputs) == 1:
                        op.attribs[key] = value = value.outputs[0]
                        op.inputs = op.inputs + (value,)
                    elif isinstance(value, list):
                        for idx, item in enumerate(value):
                            if isinstance(item, Graph) and len(item.operations) == 0 and len(item.outputs) == 1:
                                op.attribs[key][idx] = item = item.outputs[0]
                                op.inputs = op.inputs + (item,)

        removed = [graph for graph in model.graphs[1:] if len(graph.operations) == 0 and len(graph.outputs) == 1]
        model.remove_graphs(removed)

    def zero_copy(self, tensor):
        result = Tensor(tensor.graph, dtype=tensor.dtype, shape=tensor.shape, quant=copy.deepcopy(tensor.quant))
        self._zero_copy_operation(tensor, result)
        return result

    def sknd_dtype(self, dtype):
        return ConverterToSkriptND._DtypeFromNumpy[dtype]

    def _collect_shape_ops(self, model, shape_op_type='Shape'):
        self._shape_ops = set()
        for graph in model.graphs:
            for op in reversed(graph.operations):
                if op in self._shape_ops:
                    if op.type != shape_op_type:
                        self._shape_ops.update({input.producer for input in op.inputs if input.producer is not None})
                else:
                    shape_args = self.shape_expr_args(op.type)
                    for idx, input in enumerate(op.inputs):
                        neg_idx = idx - len(op.inputs)
                        if input.producer is not None and (idx in shape_args or neg_idx in shape_args):
                            self._shape_ops.add(input.producer)

    def _is_shape_expr(self, tensor):
        original = self._tensor_map.get(tensor)
        return original.producer in self._shape_ops if original else False

    def _fix_shape_expr_args(self, model):
        for graph in model.graphs:
            count = len(graph.operations)
            for i in range(count):
                op = graph.operations[i]
                for input in op.inputs:
                    if isinstance(input, list):
                        for item in input:
                            if item is not None and not item.has_producer and self._is_shape_expr(item):
                                self._fix_shape_expr_arg(item, graph)
                    else:
                        if input is not None and not input.has_producer and self._is_shape_expr(input):
                            self._fix_shape_expr_arg(input, graph)
            graph.sort()

    def _fix_shape_expr_arg(self, arg, graph):
        expr = self.arg_as_attrib(arg)
        if expr.op == ShapeExpr.Op.Const:
            arg.set_data(expr.args[0], variable=False)
        else:
            Operation(graph, type="layout.constant", attribs={"shape": list(arg.shape), "value": expr}, inputs=(), outputs=(arg,))

    @staticmethod
    def _set_max_input_shapes(model, max_input_shapes):
        for tensor in model.main.inputs:
            max_shape = max_input_shapes.get(tensor.name)
            if max_shape is not None:
                tensor.shape = tuple(s if s is not None else PlaceholderExpr(None, max_shape[i])
                                     for i, s in enumerate(tensor.shape))

    @staticmethod
    def _convert_int_inf(x):
        return _INT_POS_INF if x > _INT_MAX else _INT_NEG_INF if x < -_INT_MAX else x

    @staticmethod
    def _non_singleton_rank(shape):
        return sum(s != 1 for s in shape)

    @staticmethod
    def _ensure_zero_rank(expr):
        if expr.rank == 0:
            return expr
        elif expr.op == ShapeExpr.Op.UpRank and expr.arg.rank == 0:
            return expr.arg
        else:
            return ShapeExpr(ShapeExpr.Op.DownRank, expr.rank)

    def _eval_symbolic_shape(self, tensor):
        raise NotImplementedError()

    def eval_symbolic_shape(self, tensor, as_scalar=False):
        symbolic = self._eval_symbolic_shape(self._tensor_map[tensor])
        if as_scalar:
            subscript = ShapeExpr(ShapeExpr.Op.Const, args=[np.array(0)])
            symbolic = ShapeExpr(ShapeExpr.Op.Subscript, args=[symbolic, subscript])
        symbolic = optimize_shape_expr(symbolic)
        check_shape_expr(symbolic)
        return symbolic

    def arg_as_attrib(self, arg, as_scalar=False, convert_int_inf=False, none_on_failure=False, flat=True):
        if isinstance(arg, list):
            return [self.arg_as_attrib(item, as_scalar=as_scalar, convert_int_inf=convert_int_inf) for item in arg]

        if self.is_const(arg):
            value = self.as_const(arg, flat=flat)
            if convert_int_inf:
                value = [self._convert_int_inf(x) for x in value] if isinstance(value, list) else self._convert_int_inf(value)
            return value[0] if as_scalar and isinstance(value, list) and len(value) == 1 else value
        else:
            try:
                return self.eval_symbolic_shape(arg, as_scalar=as_scalar)
            except AssertionError as e:
                if none_on_failure:
                    return None
                raise ConversionError(f"Conversion of shape expression is not possible: " + str(e))


class ConverterFromSkriptND(Converter):

    def __init__(self, transforms, functions=None, mirror_unsupported=False):
        Converter.__init__(self, transforms, functions, mirror_unsupported)

    def _is_constant(self, tensor):
        return tensor.data is not None

    def _read_constant(self, tensor, type, flat):
        if tensor.data is not None:
            value = tensor.data
            return types.from_numpy(value, type=type, flat=flat) if isinstance(value, np.ndarray) else \
                types.cast(value, type=type) if type is not None else value
        else:
            raise ConversionError('trying to evaluate non-constant tensor')


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
        Converter.__init__(self, transforms, functions, mirror_unsupported)
        self._infer_shapes = infer_shapes
        self._custom_shapes = custom_shapes

    def _insert_externals_and_constants(self, model):
        graph = model.main
        for tensor in graph.tensors:
            mapped = self._tensor_map[tensor]
            if mapped.producer is None and len(mapped.consumers) > 0 and mapped.dtype != np.void:
                if mapped.data is None:
                    Operation(graph, type='external', inputs=(), outputs=tensor,
                              attribs={'shape': list(tensor.shape), 'dtype': tensor.dtype})
                else:
                    Operation(graph, type='constant', inputs=(), outputs=tensor,
                              attribs={'shape': list(tensor.shape), 'dtype': tensor.dtype, 'value': mapped.data})

    def _make_constant(self, graph, dtype, value, inline):
        if isinstance(value, tuple):
            value = list(value)
        shape = value.shape if isinstance(value, np.ndarray) else (len(value),) if isinstance(value, list) else ()
        isarray = isinstance(value, np.ndarray) or isinstance(value, list)

        tensor = Tensor(graph, dtype=dtype, shape=shape)
        if inline:
            tensor.set_data(types.to_numpy(value, dtype), variable=False)
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
    def remove_unused_constants(model):
        for graph in model.graphs:
            ops = [op for op in graph.operations if op.type == 'constant' and not op.output.has_consumer]
            tensors = [op.output for op in ops]
            graph.outputs = tuple(tensor for tensor in graph.outputs if tensor not in tensors)
            graph.remove_operations(ops, unlink=True)
            graph.remove_tensors(tensors)

    @staticmethod
    def inline_scalar_constants(model):
        for graph in model.graphs:
            for op in graph.operations:
                if op.type == 'constant':
                    value = op.attribs['value']
                    if not isinstance(value, np.ndarray):
                        value = np.array(value, op.output.dtype).reshape(op.output.shape)
                    if len(value.shape) == 0:
                        op.output.set_data(value, variable=False)
                        graph.remove_operation(op, unlink=True)

    @staticmethod
    def convert_constants_to_variables(model):
        for graph in model.graphs:
            variables = 0
            for op in graph.operations:
                if op.type == 'constant':
                    value = op.attribs['value']
                    if isinstance(value, np.ndarray):
                        variables += 1
                        op.type = 'variable'
                        op.attribs['label'] = op.name if op.name else 'variable' + str(variables)
                        op.output.set_data(value, variable=True)
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
    def convert_variables_to_constants(model):
        for graph in model.graphs:
            for op in graph.operations:
                if op.type == 'variable':
                    op.type = 'constant'
                    op.attribs['value'] = op.output.data
                    del op.attribs['label']

    @staticmethod
    def fill_data_in_constants(model):
        for graph in model.graphs:
            for op in graph.operations:
                if op.type == 'constant':
                    op.output.set_data(op.attribs['value'], variable=False)

    def _is_constant(self, tensor):
        if tensor.producer:
            return tensor.producer.type == 'constant'
        else:
            return tensor.data is not None

    def _read_constant(self, tensor, type, flat):
        if tensor.data is not None:
            value = tensor.data
        elif tensor.producer and tensor.producer.type == 'constant':
            value = tensor.producer.attribs['value']
        else:
            raise ConversionError('trying to evaluate non-constant tensor')

        return types.from_numpy(value, type=type, flat=flat) if isinstance(value, np.ndarray) else \
            types.cast(value, type=type) if type is not None else value


class ShapeExpr:

    Op = Enum('Op', ['Shape', 'Const', 'Add', 'Sub', 'Mul', 'Div', 'Min', 'Max', 'CeilDiv',
                     'Less', 'Greater', 'LessEqual', 'GreaterEqual', 'Equal', 'NotEqual',
                     'And', 'Or', 'Xor', 'Neg', 'Not', 'Tilde', 'Select',
                     'Sum', 'Prod', 'Minimize', 'Maximize', 'Any', 'All',
                     'Pack', 'Concat', 'Slice', 'Subscript', 'Range', 'Cast', 'UpRank', 'DownRank',
                     'Expand', 'Length'])

    def __init__(self, op, args):
        self.op = op
        self.args = args

    def __repr__(self):
        return f"{self.op}({','.join(repr(arg) if isinstance(arg, ShapeExpr) else str(arg) for arg in self.args)})"

    def __str__(self):
        if self.op == ShapeExpr.Op.Shape:
            return f"{self.args[0].name}.shape"
        elif self.op == ShapeExpr.Op.Const:
            return f"{np.array2string(self.args[0], separator=', ')}" if isinstance(self.args[0], np.ndarray) else f"{self.args[0]}"
        elif self.op == ShapeExpr.Op.Add:
            return f"({self.args[0]} + {self.args[1]})"
        elif self.op == ShapeExpr.Op.Sub:
            return f"({self.args[0]} - {self.args[1]})"
        elif self.op == ShapeExpr.Op.Mul:
            return f"({self.args[0]} * {self.args[1]})"
        elif self.op == ShapeExpr.Op.Div:
            return f"({self.args[0]} / {self.args[1]})"
        elif self.op == ShapeExpr.Op.CeilDiv:
            return f"({self.args[0]} \\ {self.args[1]})"
        elif self.op == ShapeExpr.Op.Min:
            return f"({self.args[0]} << {self.args[1]})"
        elif self.op == ShapeExpr.Op.Max:
            return f"({self.args[0]} >> {self.args[1]})"
        elif self.op == ShapeExpr.Op.Less:
            return f"({self.args[0]} < {self.args[1]})"
        elif self.op == ShapeExpr.Op.Greater:
            return f"({self.args[0]} > {self.args[1]})"
        elif self.op == ShapeExpr.Op.LessEqual:
            return f"({self.args[0]} <= {self.args[1]})"
        elif self.op == ShapeExpr.Op.GreaterEqual:
            return f"({self.args[0]} >= {self.args[1]})"
        elif self.op == ShapeExpr.Op.Equal:
            return f"({self.args[0]} == {self.args[1]})"
        elif self.op == ShapeExpr.Op.NotEqual:
            return f"({self.args[0]} != {self.args[1]})"
        elif self.op == ShapeExpr.Op.And:
            return f"({self.args[0]} && {self.args[1]})"
        elif self.op == ShapeExpr.Op.Or:
            return f"({self.args[0]} || {self.args[1]})"
        elif self.op == ShapeExpr.Op.Xor:
            return f"({self.args[0]} ^ {self.args[1]})"
        elif self.op == ShapeExpr.Op.Neg:
            return f"-{self.args[0]}"
        elif self.op == ShapeExpr.Op.Not:
            return f"!{self.args[0]}"
        elif self.op == ShapeExpr.Op.Tilde:
            return f"~|{self.args[0]}"
        elif self.op == ShapeExpr.Op.Pack:
            return "[" + ",".join(str(arg) for arg in self.args) + "]"
        elif self.op == ShapeExpr.Op.Concat:
            return "[" + ",".join(str(arg.arg) if arg.op == ShapeExpr.Op.UpRank else (str(arg) + '..') for arg in self.args) + "]"
        elif self.op == ShapeExpr.Op.Slice:
            return f"{self.args[0]}[{self.args[1]}:{self.args[2]}]"
        elif self.op == ShapeExpr.Op.Subscript:
            return f"{self.args[0]}[{self.args[1]}]"
        elif self.op == ShapeExpr.Op.Range:
            return f"[{self.args[0]}:{self.args[1]}:{self.args[2]}]"
        elif self.op == ShapeExpr.Op.Cast:
            return f"{self.args[1]}({self.args[0]})"
        elif self.op == ShapeExpr.Op.UpRank:
            return f"[{self.args[0]}]"
        elif self.op == ShapeExpr.Op.DownRank:
            return f"{self.args[0]}[0]"
        elif self.op == ShapeExpr.Op.Expand:
            return f"[{self.args[0]} ..({self.args[1]})]"
        elif self.op == ShapeExpr.Op.Length:
            return f"len({self.args[0]})"
        elif self.op == ShapeExpr.Op.Select:
            return f"{self.args[0]} ? {self.args[1]} : {self.args[2]}"
        elif self.op == ShapeExpr.Op.Sum:
            return f"({self.args[0]} + ..)"
        elif self.op == ShapeExpr.Op.Prod:
            return f"({self.args[0]} * ..)"
        elif self.op == ShapeExpr.Op.Minimize:
            return f"({self.args[0]} << ..)"
        elif self.op == ShapeExpr.Op.Maximize:
            return f"({self.args[0]} >> ..)"
        elif self.op == ShapeExpr.Op.Any:
            return f"({self.args[0]} || ..)"
        elif self.op == ShapeExpr.Op.All:
            return f"({self.args[0]} && ..)"
        else:
            assert False

    @property
    def rank(self):
        if self.op == ShapeExpr.Op.Shape:
            return 1
        elif self.op == ShapeExpr.Op.Const:
            return len(self.args[0].shape)
        elif self.op == ShapeExpr.Op.Slice:
            return self.args[0].rank
        elif self.op == ShapeExpr.Op.Subscript:
            return self.args[1].rank
        elif self.op == ShapeExpr.Op.UpRank:
            return self.args[0].rank + self.args[1]
        elif self.op == ShapeExpr.Op.DownRank:
            return self.args[0].rank - self.args[1]
        elif self.op == ShapeExpr.Op.Expand:
            return 1
        elif self.op == ShapeExpr.Op.Length:
            return 0
        elif self.op == ShapeExpr.Op.Pack:
            return 1
        elif self.op == ShapeExpr.Op.Cast:
            return self.args[0].rank
        elif self.op in [ShapeExpr.Op.Sum, ShapeExpr.Op.Prod,
                         ShapeExpr.Op.Minimize, ShapeExpr.Op.Maximize,
                         ShapeExpr.Op.Any, ShapeExpr.Op.All]:
            return 0
        else:
            return max(arg.rank for arg in self.args)

    @property
    def effective_rank(self):
        if self.op == ShapeExpr.Op.Shape:
            return 1
        elif self.op == ShapeExpr.Op.Const:
            return sum(s != 1 for s in self.args[0].shape)
        elif self.op == ShapeExpr.Op.Slice:
            return self.args[0].rank
        elif self.op == ShapeExpr.Op.Subscript:
            return self.args[1].rank
        elif self.op == ShapeExpr.Op.UpRank:
            return self.args[0].rank
        elif self.op == ShapeExpr.Op.DownRank:
            return self.args[0].rank - self.args[1]
        elif self.op == ShapeExpr.Op.Expand:
            return 1
        elif self.op == ShapeExpr.Op.Length:
            return 0
        elif self.op == ShapeExpr.Op.Pack:
            return 1
        elif self.op == ShapeExpr.Op.Cast:
            return self.args[0].rank
        elif self.op in [ShapeExpr.Op.Sum, ShapeExpr.Op.Prod,
                         ShapeExpr.Op.Minimize, ShapeExpr.Op.Maximize,
                         ShapeExpr.Op.Any, ShapeExpr.Op.All]:
            return 0
        else:
            return max(arg.rank for arg in self.args)

    @property
    def dtype(self):
        if self.op == ShapeExpr.Op.Shape:
            return int
        elif self.op == ShapeExpr.Op.Length:
            return int
        elif self.op == ShapeExpr.Op.Const:
            return self.args[0].dtype
        elif self.op == ShapeExpr.Op.Cast:
            dtype = self.args[1]
            return bool if dtype == 'bool' else int if dtype == 'int' else float if dtype == 'real' else str
        elif self.op == ShapeExpr.Op.Select:
            return self.args[1].dtype
        elif self.op in [ShapeExpr.Op.Less, ShapeExpr.Op.Greater, ShapeExpr.Op.LessEqual, ShapeExpr.Op.GreaterEqual,
                         ShapeExpr.Op.Equal, ShapeExpr.Op.NotEqual, ShapeExpr.Op.Any, ShapeExpr.Op.All,
                         ShapeExpr.Op.And, ShapeExpr.Op.Or, ShapeExpr.Op.Xor, ShapeExpr.Op.Not]:
            return bool
        else:
            return self.args[0].dtype

    @property
    def arg(self):
        return self.args[0]

    def is_const(self, value=None):
        return self.op == ShapeExpr.Op.Const and (value is None or np.array_equal(self.arg, value))


def optimize_shape_expr(expr):
    if expr.op == ShapeExpr.Op.Cast:
        if expr.dtype == int:
            while expr.arg.op == ShapeExpr.Op.Cast:
                expr.args[0] = expr.arg.arg
            if expr.arg.op in [ShapeExpr.Op.Min, ShapeExpr.Op.Max]:
                expr.arg.args[0] = ShapeExpr(ShapeExpr.Op.Cast, [expr.arg.args[0], expr.args[1]])
                expr.arg.args[1] = ShapeExpr(ShapeExpr.Op.Cast, [expr.arg.args[1], expr.args[1]])

    expr.args = [optimize_shape_expr(arg) if isinstance(arg, ShapeExpr) else arg for arg in expr.args]

    if expr.op == ShapeExpr.Op.DownRank:
        if expr.arg.op == ShapeExpr.Op.UpRank and expr.args[1] == expr.arg.args[1]:
            return expr.arg.arg
        elif expr.arg.op == ShapeExpr.Op.Slice and expr.args[1] == 1:
            return ShapeExpr(ShapeExpr.Op.Subscript, [expr.arg.args[0], expr.arg.args[1]])
    elif expr.op == ShapeExpr.Op.UpRank:
        if expr.arg.op == ShapeExpr.Op.DownRank and expr.args[1] == expr.arg.args[1]:
            return expr.arg.arg
    elif expr.op == ShapeExpr.Op.Subscript:
        if expr.args[1].is_const(0):
            if expr.arg.op == ShapeExpr.Op.UpRank and expr.arg.args[1] == 1:
                return expr.arg.arg
            elif expr.arg.op == ShapeExpr.Op.Slice:
                return ShapeExpr(ShapeExpr.Op.Subscript, [expr.arg.args[0], expr.arg.args[1]])
    elif expr.op == ShapeExpr.Op.Concat:
        if len(expr.args) == 1:
            return expr.arg
        elif any(item.op == ShapeExpr.Op.Concat for item in expr.args):
            args = list()
            for item in expr.args:
                if item.op == ShapeExpr.Op.Concat:
                    args += item.args
                else:
                    args.append(item)
            expr.args = args
            return expr
    elif expr.op == ShapeExpr.Op.Cast:
        if expr.dtype == expr.arg.dtype:
            return expr.arg
        if expr.arg.op == ShapeExpr.Op.Const:
            expr.arg.args[0] = expr.arg.arg.astype(expr.dtype)
            return expr.arg
    elif expr.op == ShapeExpr.Op.Minimize:
        return _optimize_reduce_shape_expr(expr, ShapeExpr.Op.Minimize, ShapeExpr.Op.Min)
    elif expr.op == ShapeExpr.Op.Maximize:
        return _optimize_reduce_shape_expr(expr, ShapeExpr.Op.Minimize, ShapeExpr.Op.Max)
    elif expr.op == ShapeExpr.Op.Sum:
        return _optimize_reduce_shape_expr(expr, ShapeExpr.Op.Minimize, ShapeExpr.Op.Add)
    elif expr.op == ShapeExpr.Op.Prod:
        return _optimize_reduce_shape_expr(expr, ShapeExpr.Op.Minimize, ShapeExpr.Op.Mul)
    elif expr.op == ShapeExpr.Op.Length:
        if expr.arg.op == ShapeExpr.Op.Pack:
            return ShapeExpr(ShapeExpr.Op.Const, args=[np.array(len(expr.arg.args))])

    return expr


_INT_MAX = 2 ** 31 - 1
_FLT_POS_INF = ShapeExpr(ShapeExpr.Op.Const, args=[float('inf')])
_FLT_NEG_INF = ShapeExpr(ShapeExpr.Op.Const, args=[float('-inf')])
_INT_POS_INF = ShapeExpr(ShapeExpr.Op.Cast, args=[_FLT_POS_INF, 'int'])
_INT_NEG_INF = ShapeExpr(ShapeExpr.Op.Cast, args=[_FLT_NEG_INF, 'int'])


def _optimize_reduce_shape_expr(expr, reduce_op, binary_op):
    arg = expr.arg
    if arg.op == ShapeExpr.Op.Concat or arg.op == ShapeExpr.Op.Pack:
        if arg.op == ShapeExpr.Op.Concat:
            items = [item.arg if item.op == ShapeExpr.Op.UpRank and item.args[1] == 1 else ShapeExpr(reduce_op, [item])
                     for item in arg.args]
        else:
            items = arg.args

        if len(items) == 1:
            return items[0]
        elif len(items) > 1:
            expr = ShapeExpr(binary_op, [items[0], items[1]])
            for item in items[2:]:
                expr = ShapeExpr(binary_op, [expr, item])
            return expr
    return expr


def check_shape_expr(expr):
    for arg in expr.args:
        if isinstance(arg, ShapeExpr):
            check_shape_expr(arg)

    assert expr.rank <= 1, f"shape expression rank must be <= 1, but expression contains sub-expression {expr} of rank {expr.rank}"
