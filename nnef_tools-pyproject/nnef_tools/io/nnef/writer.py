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

import nnef
import numpy as np
import tempfile
import shutil
import six
import os
from .helpers import tgz_compress
from ...model import Tensor
from ...utils.types import as_str, from_numpy


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


_DtypeFromPyType = {
    str: 'string',
    float: 'scalar',
    int: 'integer',
    bool: 'logical',
    None: 'dtype',
}


def _nnef_dtype(dtype):
    return _DtypeFromNumpy[dtype.type if isinstance(dtype, np.dtype) else dtype] if dtype is not None else None


def _print(graph, file, extensions, fragments, version_custom_ops, annotate_shapes):
    assert graph.is_sorted(), "graph must be topologically sorted"
    assert all(tensor.name is not None or (tensor.producer is None and tensor.data is not None)
               for tensor in graph.tensors), \
        "all tensors must have names"
    assert all(all(s is not None for s in op.attribs['shape'])
               for op in graph.operations if op.type == 'external'), \
        "external ops must not contain undefined shapes"

    print(nnef.format_version((1, 0)), file=file)
    if len(extensions):
        print(file=file)
        print(nnef.format_extensions(extensions), file=file)
    if fragments:
        print(file=file)
        print(fragments, file=file)
    print(file=file)

    graph_name = as_str(graph.name) if graph.name is not None else "G"
    graph_inputs = [as_str(item.name) for item in graph.inputs]
    graph_outputs = [as_str(item.name) for item in graph.outputs]

    print("graph {}({}) -> ({})".format(graph_name, ', '.join(graph_inputs), ', '.join(graph_outputs)), file=file)
    print("{", file=file)

    versions = {}
    for op in graph.operations:
        assert all(isinstance(item, Tensor) for item in op.outputs)

        inputs = ((from_numpy(item.data) if item.producer is None else nnef.Identifier(as_str(item.name)))
                  if isinstance(item, Tensor) else item for item in op.inputs)
        inputs = tuple(inputs) if isinstance(op.inputs, tuple) else (list(inputs),)

        outputs = (nnef.Identifier(as_str(item.name)) for item in op.outputs)
        outputs = tuple(outputs) if isinstance(op.outputs, tuple) else (list(outputs),)

        attribs = {as_str(key): value for key, value in six.iteritems(op.attribs)}

        name = _next_version(op.type, versions) if op.type not in nnef.StandardOperations and version_custom_ops else op.type

        dtype = attribs.get('dtype')
        if dtype is not None:
            dtype = _nnef_dtype(dtype)
            del attribs['dtype']

        for key, value in six.iteritems(attribs):
            if isinstance(value, (type, np.dtype)):
                attribs[key] = _nnef_dtype(value)

        invocation = nnef.format_invocation(name=name, dtype=dtype, attribs=attribs, inputs=inputs, outputs=outputs)
        annotation = "    # " + ", ".join(_nnef_dtype(output.dtype) + str(output.shape) for output in op.outputs) \
            if annotate_shapes else ''

        print("    {};{}".format(invocation, annotation), file=file)

    print("}", file=file)


def _write_tensor(array, filename, quantized):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "wb") as file:
        nnef.write_tensor(file=file, tensor=array, quantized=quantized)


def _write_quantization(graph, file):
    for tensor in graph.tensors:
        if tensor.quant:
            op_name = tensor.quant['op-name']
            attribs = ', '.join("{} = {}".format(k, _printable_value(v))
                                for k, v in six.iteritems(tensor.quant)
                                if k != 'op-name' and v is not None)
            if attribs:
                print('"{}": {}({});'.format(tensor.name, op_name, attribs), file=file)


def _printable_value(v):
    if type(v) == bool:
        return 'true' if v else 'false'
    elif type(v) == np.ndarray:
        return v.tolist()
    else:
        return v


def _next_version(name, versions):
    version = versions.get(name, 0) + 1
    versions[name] = version
    return '{}_v{}'.format(name, version)


def _generate_custom_fragments(graph, fragments, version):
    versions = {} if version else None
    return '\n'.join(_generate_fragment(op, versions) for op in graph.operations
                     if op.type not in nnef.StandardOperations and op.type not in fragments)


def _generate_fragment(op, versions):
    attribs = {key: _make_attrib_type(value) for key, value in op.attribs.items()}
    inputs = [_make_tensor_type(value) for value in op.inputs]
    outputs = [_make_tensor_type(value) for value in op.outputs]
    dtype = _nnef_dtype(op.attribs.get('dtype'))
    name = _next_version(op.type, versions) if versions is not None else op.type

    return 'fragment ' + _fragment_signature(name, dtype, attribs, inputs, outputs) + ';'


def _fragment_signature(name, dtype, attribs, inputs, outputs):
    str = name
    if dtype is not None:
        str += '<' + dtype + '>'
    str += '( '
    str += _types_str(['_I{}'.format(i + 1) for i in range(len(inputs))], inputs, True)
    if len(inputs) and len(attribs):
        str += ', '
    str += _types_str(attribs.keys(), attribs.values(), False)
    str += ' ) -> ( '
    str += _types_str(['_O{}'.format(i + 1) for i in range(len(outputs))], outputs, True)
    str += ' )'
    return str


def _make_attrib_type(value):
    repeated = False
    if isinstance(value, list):
        if len(value) == 0:
            return None, False
        tp = type(value[0])
        if not all(type(v) == tp for v in value):
            return None, False
        repeated = True
        value = value[0]

    if not isinstance(value, (float, int, bool, str)):
        return None, False

    return _DtypeFromPyType[type(value)], repeated


def _make_tensor_type(value):
    repeated = False
    if isinstance(value, list):
        if len(value) == 0:
            return None, False
        dtype = value[0].dtype
        if not all(v.dtype == dtype for v in value):
            return None, False
        repeated = True
        value = value[0]

    return _nnef_dtype(value.dtype), repeated


def _types_str(names, items, tensor):
    return ', '.join(name + ': ' + ('tensor<{}>'.format(type) if tensor else type) + ('[]' if repeated else '')
                     for name, (type, repeated) in zip(names, items))


class Writer(object):

    def __init__(self, compression=None, extensions=None, fragments=None, fragment_dependencies=None,
                 generate_custom_fragments=False, version_custom_fragments=True, annotate_shapes=False):
        self._compression = compression
        self._extensions = extensions or []
        self._fragments = fragments or {}
        self._fragment_dependencies = fragment_dependencies or {}
        self._generate_custom_fragments = generate_custom_fragments
        self._version_custom_fragments = version_custom_fragments
        self._annotate_shapes = annotate_shapes

    def __call__(self, graph, path):
        folder = None
        try:
            if self._compression is not None:
                folder = tempfile.mkdtemp(prefix="nnef_")
            else:
                folder = path
                if not os.path.exists(folder):
                    os.makedirs(folder)

            used_operators = self._used_operators(graph, self._fragment_dependencies)
            fragments = "".join(text for name, text in six.iteritems(self._fragments) if name in used_operators)
            if self._generate_custom_fragments:
                customs = _generate_custom_fragments(graph, fragments=self._fragments,
                                                     version=self._version_custom_fragments)
                if fragments and customs:
                    fragments += "\n"
                fragments += customs

            if len(fragments):
                if "KHR_enable_fragment_definitions" not in self._extensions:
                    self._extensions.append("KHR_enable_fragment_definitions")
                if "KHR_enable_operator_expressions" not in self._extensions:
                    self._extensions.append("KHR_enable_operator_expressions")

            graph_filename = os.path.join(folder, 'graph.nnef')
            with open(graph_filename, 'w') as file:
                _print(graph, file, extensions=self._extensions, fragments=fragments,
                       version_custom_ops=self._generate_custom_fragments and self._version_custom_fragments,
                       annotate_shapes=self._annotate_shapes)

            for op in graph.operations:
                if op.type == 'variable':
                    filename = op.attribs['label'] + ".dat"
                    if filename.startswith('/'):
                        filename = filename[1:]
                    _write_tensor(np.asarray(op.output.data, order='C'), os.path.join(folder, filename),
                                  quantized=True if op.output.quant else False)

            if any(tensor.quant for tensor in graph.tensors):
                quant_filename = os.path.join(folder, 'graph.quant')
                with open(quant_filename, 'w') as file:
                    _write_quantization(graph, file)
        finally:
            if self._compression is not None and folder:
                tgz_compress(folder, path + '.tgz', compression_level=self._compression)
                shutil.rmtree(folder)

    @staticmethod
    def _used_operators(graph, dependencies):
        used = {op.type for op in graph.operations}
        count = len(used)
        changed = True
        while changed:
            for key, deps in six.iteritems(dependencies):
                if key in used:
                    used.update(deps)

            changed = len(used) > count
            count = len(used)

        return used
