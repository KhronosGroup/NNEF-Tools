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

_NonAtomicOperationsSet = {"rms_pool", "softabs", "log2", "linear", "planewise_conv", "planewise_deconv",
                           "separable_conv", "separable_deconv", "l1_normalization", "layer_normalization",
                           "divisive_normalization"}

_CustomOperationNames = ["max_pool_grad",
                         "max_pool_grad_with_index",
                         "avg_pool_grad",
                         "conv_grad_input",
                         "conv_grad_filter"]

_NonAtomicCustomOperationsSet = {}

AtomicOperations = ([op for op in nnef.StandardOperations if op not in _NonAtomicOperationsSet]
                    + [op for op in _CustomOperationNames if op not in _NonAtomicCustomOperationsSet])

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


def _max_pool_grad_shape(proto, args, shapes):
    shapes[args['input_grad']] = shapes[args['orig_input']]


def _max_pool_grad_with_index_shape(proto, args, shapes):
    shapes[args['input_grad']] = shapes[args['orig_input']]


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

DeferShapes = {
    # TODO fix shapeof in nnef2tf
    # "deconv": ["output_shape"],
    # "reshape": ["shape"]
}


# noinspection PyProtectedMember
class Config(object):
    def __init__(self, key=None, custom_ops=None, custom_shapes=None, deferred_shapes=None):
        # type: (Optional[str], Optional[str], Optional[Dict[str, function]], Optional[Dict[str, List[str]]])->None
        self.key = key if key else "custom_ops_" + str(id(self))
        self.custom_ops = custom_ops
        self.custom_shapes = custom_shapes.copy() if custom_shapes else None
        self.deferred_shapes = deferred_shapes.copy() if deferred_shapes else None

    def __enter__(self):
        if self.custom_ops:
            nnef._register_custom_ops(self.key, self.custom_ops)
        if self.custom_shapes:
            nnef._register_custom_shapes(self.custom_shapes)
        if self.deferred_shapes:
            nnef._register_deferred_shapes(self.deferred_shapes)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.custom_ops:
            nnef._unregister_custom_ops(self.key)
        if self.custom_shapes:
            custom_shapes = self.custom_shapes.copy()
            for k in custom_shapes.keys():
                custom_shapes[k] = None
            nnef._register_custom_shapes(custom_shapes)
        if self.deferred_shapes:
            deferred_shapes = self.deferred_shapes.copy()
            for k in deferred_shapes.keys():
                deferred_shapes[k] = []
            nnef._register_deferred_shapes(deferred_shapes)


default_config = Config(custom_ops=CustomOperations, custom_shapes=CustomShapes, deferred_shapes=DeferShapes)


def parse_file(file_name):
    with default_config:
        return nnef.parse_file(file_name, atomics=AtomicOperations)


def parse_string(string):
    with default_config:
        return nnef.parse_string(string, atomics=AtomicOperations)
