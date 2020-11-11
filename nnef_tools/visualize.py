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

import argparse
from .io.nnef import Reader
from .io.nnef.writer import _DtypeFromNumpy
from graphviz import Digraph
import numpy as np
import os


def _text_with_size(text, size):
    return '<FONT POINT-SIZE="{}">{}</FONT>'.format(size, text)


def _format_tensor_label(tensor):
    return '< {}<BR/> {}>'.format(tensor.name, _text_with_size(_dtype_str(tensor.dtype) + _shape_str(tensor.shape), size=10))


def _dtype_str(dtype):
    return _DtypeFromNumpy[dtype.type if isinstance(dtype, np.dtype) else dtype]


def _shape_str(shape):
    return '[' + ','.join(str(s) if s is not None else '?' for s in shape) + ']' if shape is not None else ''


def _truncate(text, max_length=32):
    return text[:max_length - 3] + "..." if len(text) > max_length else text


def _attribs_str(attribs, separator):
    s = []
    for k, v in sorted(attribs.items(), key=lambda e: e[0]):
        if k == "label":
            v = ".../" + v.split('/')[-1]
        elif k == "dtype":
            v = _dtype_str(v)
        s.append("{}{}{}".format(k, separator, _truncate(str(v))))
    return s


def _format_op_details(op):
    s = [
        "inputs: " + ", ".join(str(tensor.name if tensor.producer is not None else tensor.data) for tensor in op.inputs),
        "outputs: " + ", ".join(str(tensor.name) for tensor in op.outputs),
    ]

    s.extend(_attribs_str(op.attribs, separator=': '))

    return "&#13;&#10;".join(s)


def _format_op_label(op):
    attrs = (_text_with_size(s, size=10) for s in _attribs_str(op.attribs, separator='='))
    return '<{}<BR/>{}>'.format(op.type, '<BR/>'.join(attrs)) if len(op.attribs) else op.type


def _generate_digraph(graph, show_variables, verbose):
    digraph = Digraph()

    for op in graph.operations:
        if (show_variables or op.type != "variable") and op.type != "external":
            digraph.node(str(id(op)), _format_op_label(op) if verbose else op.type, shape="box", tooltip=_format_op_details(op))

    for tensor in graph.tensors:
        if tensor.producer is not None and (show_variables or tensor.producer.type != "variable") and tensor.producer.type != "external":
            for consumer in tensor.consumers:
                digraph.edge(str(id(tensor.producer)), str(id(consumer)),
                             label=_format_tensor_label(tensor) if verbose else "  " + tensor.name,
                             labeltooltip="{}{}".format(_dtype_str(tensor.dtype), _shape_str(tensor.shape)))

    for tensor in graph.inputs:
        digraph.node(str(id(tensor)), _format_tensor_label(tensor) if verbose else tensor.name, shape="ellipse",
                     tooltip="{}{}".format(_dtype_str(tensor.dtype), _shape_str(tensor.shape)))
        for consumer in tensor.consumers:
            digraph.edge(str(id(tensor)), str(id(consumer)), label=None)

    for tensor in graph.outputs:
        digraph.node(str(id(tensor)), _format_tensor_label(tensor) if verbose else tensor.name, shape="ellipse",
                     tooltip="{}{}".format(_dtype_str(tensor.dtype), _shape_str(tensor.shape)))
        digraph.edge(str(id(tensor.producer)), str(id(tensor)), label=None)

    return digraph


def main(args):
    reader = Reader(decomposed=args.decompose, infer_shapes=args.infer_shapes)
    graph = reader(args.model)
    digraph = _generate_digraph(graph, args.show_variables, args.verbose)
    digraph.render(args.model + '.gv', format=args.format, cleanup=True)
    os.rename(args.model + '.gv.' + args.format, args.model + '.' + args.format)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='The model to visualize')
    parser.add_argument('--decompose', type=str, nargs='*', default=None,
                        help='Names of operators to be decomposed by NNEF parser')
    parser.add_argument('--verbose', action='store_true',
                        help='Add more info to the nodes and edges')
    parser.add_argument('--show-variables', action='store_true',
                        help='Show variables explicitly')
    parser.add_argument('--infer-shapes', action='store_true',
                        help='Perform shape inference and show in visualized graph')
    parser.add_argument('--format', type=str, choices=['svg', 'pdf', 'png', 'dot'], default='svg',
                        help='The format of the output')
    exit(main(parser.parse_args()))
