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

from __future__ import division, print_function

import nnef

from . import stdlib_source


class NNEFParserConfig(object):

    def __init__(self, source=None, shapes=None, expand=None):
        if source is None:
            source = ""
        if shapes is None:
            shapes = {}
        if expand is None:
            expand = list()
        if not isinstance(expand, list):
            expand = list(expand)

        self.custom_ops = source
        self.custom_shapes = shapes
        self.expand = expand

    def parse_string(self, graph_str, quant_str=None):
        return nnef.parse_string(graph_str=graph_str,
                                 quant_str=quant_str,
                                 stdlib=self.custom_ops,
                                 lowered=self.expand)

    def load_graph(self, path):
        return nnef.load_graph(path=path,
                               stdlib=self.custom_ops,
                               lowered=self.expand)

    def infer_shapes(self, graph):
        nnef.infer_shapes(graph=graph, custom_shapes=self.custom_shapes)
        return graph


_NonAtomicOperationsSet = {"rms_pool", "softabs", "log2", "linear", "planewise_conv", "planewise_deconv",
                           "separable_conv", "separable_deconv", "l1_normalization", "layer_normalization",
                           "divisive_normalization"}

_NonAtomicCustomOperationsSet = set()

CustomOperations = \
    """
    fragment max_pool_grad(
        orig_input: tensor<scalar>,
        orig_output: tensor<scalar>,
        output_grad: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer, integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( input_grad: tensor<scalar> );

    fragment max_pool_grad_with_index(
        orig_input: tensor<scalar>,
        orig_index: tensor<integer>,
        output_grad: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer, integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( input_grad: tensor<scalar> );

    fragment avg_pool_grad(
        output_grad: tensor<scalar>,
        orig_input_shape: integer[],
        size: integer[],
        border: string = 'constant',
        padding: (integer, integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( input_grad: tensor<scalar> );

    fragment conv_grad_input(
        orig_filter: tensor<scalar>,
        output_grad: tensor<scalar>,
        bias: tensor<scalar> = 0.0,
        orig_input_shape: integer[],
        border: string = 'constant',
        padding: (integer, integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [],
        groups: integer = 1 )
    -> ( input_grad: tensor<scalar> );

    fragment conv_grad_filter(
        orig_input: tensor<scalar>,
        output_grad: tensor<scalar>,
        bias: tensor<scalar> = 0.0,
        orig_filter_shape: integer[],
        border: string = 'constant',
        padding: (integer, integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [],
        groups: integer = 1 )
    -> ( input_grad: tensor<scalar> );
    """


def _max_pool_grad_shape(orig_input, orig_output, output_grad, **kwargs):
    return orig_input


def _max_pool_grad_with_index_shape(orig_input, orig_index, output_grad, **kwargs):
    return orig_input


def _avg_pool_grad_shape(output_grad, orig_input_shape, **kwargs):
    return orig_input_shape


def _conv_grad_filter_shape(orig_input, output_grad, orig_filter_shape, **kwargs):
    return orig_filter_shape


def _conv_grad_input_shape(orig_filter, output_grad, bias, orig_input_shape, **kwargs):
    return orig_input_shape


CustomShapes = {
    "max_pool_grad": _max_pool_grad_shape,
    "max_pool_grad_with_index": _max_pool_grad_with_index_shape,
    "avg_pool_grad": _avg_pool_grad_shape,
    "conv_grad_input": _conv_grad_input_shape,
    "conv_grad_filter": _conv_grad_filter_shape
}

default_config = NNEFParserConfig(source=stdlib_source.SOURCE + '\n\n' + CustomOperations,
                                  shapes=CustomShapes,
                                  expand=list(_NonAtomicOperationsSet | _NonAtomicCustomOperationsSet))


def load_graph(file_name):
    return default_config.infer_shapes(default_config.load_graph(file_name))


def parse_string(string):
    return default_config.infer_shapes(default_config.parse_string(string))
