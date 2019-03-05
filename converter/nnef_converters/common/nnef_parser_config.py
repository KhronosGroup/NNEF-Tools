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

from .types import *


# noinspection PyProtectedMember
class Config(object):
    def __init__(self, key=None, custom_ops=None, custom_shapes=None):
        # type: (Optional[str], Optional[str], Optional[Dict[str, Callable]])->None
        self.key = key if key else "custom_ops_" + str(id(self))
        self.custom_ops = custom_ops
        self.custom_shapes = custom_shapes.copy() if custom_shapes else None

    def __enter__(self):
        if self.custom_ops:
            nnef._register_custom_ops(self.key, self.custom_ops)
        if self.custom_shapes:
            nnef._register_custom_shapes(self.custom_shapes)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.custom_ops:
            nnef._unregister_custom_ops(self.key)
        if self.custom_shapes:
            custom_shapes = self.custom_shapes.copy()
            for k in custom_shapes.keys():
                custom_shapes[k] = None
            nnef._register_custom_shapes(custom_shapes)


_NonAtomicOperationsSet = {"rms_pool", "softabs", "log2", "linear", "planewise_conv", "planewise_deconv",
                           "separable_conv", "separable_deconv", "l1_normalization", "layer_normalization",
                           "divisive_normalization"}

_CustomOperationNames = ["max_pool_grad",
                         "max_pool_grad_with_index",
                         "avg_pool_grad",
                         "conv_grad_input",
                         "conv_grad_filter"]

_NonAtomicCustomOperationsSet = set()

# NNEF must be parsed with this before calling nnef_to_tf.Converter on it
ParserConfigToExpandNonAtomics = Config(
    key='nonatomics', custom_shapes={op: None for op in _NonAtomicOperationsSet | _NonAtomicCustomOperationsSet})

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


def _nnef_get_shape(args, shapes, arg_name):
    return shapes[args[arg_name]] if isinstance(args[arg_name], nnef.Identifier) else []


def _max_pool_grad_shape(proto, args, shapes):
    shapes[args['input_grad']] = _nnef_get_shape(args, shapes, 'orig_input')


def _max_pool_grad_with_index_shape(proto, args, shapes):
    shapes[args['input_grad']] = _nnef_get_shape(args, shapes, 'orig_input')


def _avg_pool_grad_shape(proto, args, shapes):
    shapes[args['input_grad']] = args['orig_input_shape']


def _conv_grad_input_shape(proto, args, shapes):
    shapes[args['input_grad']] = args['orig_input_shape']


def _conv_grad_filter_shape(proto, args, shapes):
    shapes[args['input_grad']] = args['orig_filter_shape']


CustomShapes = {
    "max_pool_grad": _max_pool_grad_shape,
    "max_pool_grad_with_index": _max_pool_grad_with_index_shape,
    "avg_pool_grad": _avg_pool_grad_shape,
    "conv_grad_input": _conv_grad_input_shape,
    "conv_grad_filter": _conv_grad_filter_shape
}

default_config = Config(custom_ops=CustomOperations, custom_shapes=CustomShapes)


def load_model(file_name):
    with ParserConfigToExpandNonAtomics:
        with default_config:
            return nnef.load_model(file_name)


def parse_string(string):
    with ParserConfigToExpandNonAtomics:
        with default_config:
            return nnef.parse_string(string)
