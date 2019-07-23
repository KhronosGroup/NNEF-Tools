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

from __future__ import division, print_function, absolute_import

import typing

from nnef_tools.io.caffe2.caffe2_graph import *
from nnef_tools.conversion.conversion_info import ConversionInfo
from nnef_tools.optimization.data_format_optimizer import *


class Caffe2DataFormatOptimizationDriver(DataFormatOptimizationDriver):
    @property
    def graph_type(self):
        return Caffe2Graph

    @property
    def tensor_type(self):
        return Caffe2Tensor

    @property
    def op_type(self):
        return Caffe2Operation

    @property
    def conv_grad_filter_op_names(self):
        return []

    @property
    def squeeze_op_name(self):
        return "Squeeze"

    @property
    def unsqueeze_op_name(self):
        return "ExpandDims"

    @property
    def transpose_op_name(self):
        return "Transpose"

    @property
    def reshape_op_name(self):
        return "Reshape"

    @property
    def copy_op_name(self):
        return "Copy"

    def get_axes_from_squeeze(self, squeeze):
        return squeeze.attribs["dims"]

    def set_axes_on_squeeze(self, squeeze, axes):
        squeeze.attribs["dims"] = axes

    def get_axes_from_unsqueeze(self, unsqueeze):
        return unsqueeze.attribs["dims"]

    def get_axes_from_transpose(self, transpose):
        return transpose.attribs["axes"]

    def set_axes_on_transpose(self, transpose, axes):
        transpose.attribs["axes"] = axes

    def get_shape_from_reshape(self, reshape):
        # type: (Caffe2Operation)->typing.List[int]

        return reshape.attribs['shape']

    def create_tensor(self, graph, name, shape, dtype):
        return Caffe2Tensor(graph=graph, name=name, shape=shape, dtype=dtype)

    def create_transpose_op(self, graph, input, output, axes):
        return Caffe2Operation(graph=graph, name="Transpose", inputs=input, attribs=dict(dims=axes), outputs=output)

    def create_copy_op(self, graph, input, output):
        return Caffe2Operation(graph=graph, name="Copy", inputs=input, outputs=output)

    def generate_missing_names(self, graph):
        graph.generate_missing_names()

    def get_input_of_transform(self, transform):
        return transform.inputs[0]

    def copy_quantization(self, from_tensor, to_tensor):
        # type:(Caffe2Tensor, Caffe2Tensor)->None
        to_tensor.quantization = from_tensor.quantization


class Optimizer(object):
    def __init__(self,
                 io_transform=None,  # type:typing.Optional[TransformOrTransformDictType]
                 custom_transposable_ops=None,  # type: typing.Optional[typing.List[TransposableOperation]]
                 merge_transforms_into_variables=False,  # type: bool
                 ):
        # type: (...)->None
        self._io_transform = io_transform
        self._custom_transposable_ops = custom_transposable_ops
        self._merge_transforms_into_variables = merge_transforms_into_variables

    def __call__(self, g):
        # type: (Caffe2Graph)->ConversionInfo
        return optimize_impl(g=g,
                             driver=Caffe2DataFormatOptimizationDriver(),
                             remove_unneeded_copies=True,
                             remove_inverse_transposes=False,
                             merge_transforms_into_variables=self._merge_transforms_into_variables,
                             merge_transforms_into_constants=True,
                             transposable_ops=[],
                             io_transform=self._io_transform,
                             verbose=False,
                             rename_tensors=False)
