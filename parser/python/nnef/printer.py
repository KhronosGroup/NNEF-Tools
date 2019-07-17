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
    if isinstance(value, _nnef.Identifier):
        return value
    elif isinstance(value, str):
        return "'" + value + "'"
    elif isinstance(value, bool):
        return 'true' if value else 'false'
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, list):
        return '[' + ', '.join(format_argument(item) for item in value) + ']'
    elif isinstance(value, tuple):
        return '(' + ', '.join(format_argument(item) for item in value) + ')'
    else:
        raise TypeError('arguments must be of type int, float, str, nnef.Identifier or list/tuple of such, found: ' + str(type(value)))


def format_result(value):
    if isinstance(value, list):
        return '[' + ', '.join(format_result(item) for item in value) + ']'
    elif isinstance(value, tuple):
        return '(' + ', '.join(format_result(item) for item in value) + ')'
    elif isinstance(value, _nnef.Identifier):
        return value
    else:
        raise TypeError('results must be of type nnef.Identifier or list/tuple of such, found: ' + str(type(value)))


def format_shapes(result, tensors):
    if isinstance(result, list):
        return '[' + ', '.join(format_shapes(item, tensors) for item in result) + ']'
    elif isinstance(result, tuple):
        return '(' + ', '.join(format_shapes(item, tensors) for item in result) + ')'
    elif isinstance(result, _nnef.Identifier):
        return str(tensors[result].shape)
    else:
        raise TypeError('results must be of type nnef.Identifier or list/tuple of such, found: ' + str(type(result)))


def format_invocation(name, attribs, inputs, outputs=None, dtype=None):
    string = str()

    if outputs is not None:
        string += ', '.join([format_result(output) for output in outputs])
        string += ' = '

    string += name

    if dtype is not None:
        string += '<' + dtype + '>'

    string += '('
    string += ', '.join([format_argument(input) for input in inputs])
    if len(inputs) and len(attribs):
        string += ', '
    string += ', '.join(key + ' = ' + format_argument(value) for (key, value) in attribs.items())
    string += ')'

    return string


def format_graph(name, inputs, outputs, operations, tensors, annotate_shapes=False):
    string = 'graph ' + name + '( ' + ', '.join(inputs) + ' ) -> ( ' + ', '.join(outputs) + ' )\n'
    string += '{\n'
    for operation in operations:
        inputs = operation.inputs.values()
        outputs = operation.outputs.values()
        invocation = format_invocation(operation.name, operation.attribs, inputs, outputs, operation.dtype)
        string += '\t' + invocation + ';'
        if annotate_shapes:
            string += '\t# ' + ', '.join(format_shapes(output, tensors) for output in outputs)
        string += '\n'
    string += '}\n'
    return string
