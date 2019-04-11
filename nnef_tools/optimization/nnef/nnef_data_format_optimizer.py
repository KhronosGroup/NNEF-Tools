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

from nnef_tools.io.nnef.nnef_graph import NNEFGraph, NNEFOperation, NNEFTensor
from nnef_tools.optimization.data_format_optimizer import *
from nnef_tools.conversion.conversion_info import ConversionInfo


class NNEFDataFormatOptimizationDriver(DataFormatOptimizationDriver):
    @property
    def graph_type(self):
        return NNEFGraph

    @property
    def tensor_type(self):
        return NNEFTensor

    @property
    def op_type(self):
        return NNEFOperation

    @property
    def conv_grad_filter_op_names(self):
        return ["conv_grad_filter"]

    @property
    def squeeze_op_name(self):
        return "squeeze"

    @property
    def unsqueeze_op_name(self):
        return "unsqueeze"

    @property
    def transpose_op_name(self):
        return "transpose"

    @property
    def reshape_op_name(self):
        return "reshape"

    @property
    def copy_op_name(self):
        return "copy"

    def get_axes_from_squeeze(self, squeeze):
        return squeeze.attribs["axes"]

    def set_axes_on_squeeze(self, squeeze, axes):
        squeeze.attribs["axes"] = axes

    def get_axes_from_unsqueeze(self, unsqueeze):
        return unsqueeze.attribs["axes"]

    def get_axes_from_transpose(self, transpose):
        return transpose.attribs["axes"]

    def set_axes_on_transpose(self, transpose, axes):
        transpose.attribs["axes"] = axes

    def get_shape_from_reshape(self, reshape):
        return reshape.attribs["shape"]

    def create_tensor(self, graph, name, shape, dtype):
        return NNEFTensor(graph=graph, name=name, shape=shape, dtype=dtype)

    def create_transpose_op(self, graph, input, output, axes):
        return NNEFOperation(graph=graph, name="transpose", inputs=input, attribs=dict(axes=axes), outputs=output)

    def create_copy_op(self, graph, input, output):
        return NNEFOperation(graph=graph, name="copy", inputs=input, outputs=output)

    def generate_missing_names(self, graph):
        graph.generate_missing_names()

    def get_input_of_transform(self, transform):
        return transform.input


def transpose_operation_default(transposer, graph, op, perm):
    # type: (Transposer, NNEFGraph, NNEFOperation, typing.List[int])->None
    if 'axis' in op.attribs:
        op.attribs["axis"] = transposer.apply_permutation_to_axis(op.attribs["axis"], perm)
    if 'axes' in op.attribs:
        op.attribs["axes"] = sorted(transposer.apply_permutation_to_axes(op.attribs["axes"], perm))
    if 'size' in op.attribs:
        op.attribs["size"] = transposer.apply_permutation(op.attribs["size"], perm)
    if op.attribs.get('padding'):
        op.attribs["padding"] = transposer.apply_permutation(op.attribs["padding"], perm)
    if op.attribs.get('stride'):
        op.attribs["stride"] = transposer.apply_permutation(op.attribs["stride"], perm)
    if op.attribs.get('dilation'):
        op.attribs["dilation"] = transposer.apply_permutation(op.attribs["dilation"], perm)


def transpose_operation_slice(transposer, graph, op, perm):
    # type: (Transposer, NNEFGraph, NNEFOperation, typing.List[int])->None
    op.attribs["axes"], op.attribs["begin"], op.attribs["end"] = transposer.zip_inverse(3, sorted(zip(
        transposer.apply_permutation_to_axes(op.attribs["axes"], perm), op.attribs["begin"], op.attribs["end"])))


def _get_default_transposable_ops():
    unary_ops = ['copy', 'neg', 'rcp', 'exp', 'log', 'abs', 'sign', 'not', 'floor', 'ceil', 'round',
                 'sqr', 'sqrt', 'rsqr', 'rsqrt', 'log2']
    binary_ops = ['add', 'sub', 'mul', 'div', 'pow', 'lt', 'gt', 'le', 'ge', 'eq', 'ne', 'and', 'or',
                  'min', 'max']
    activation_ops = ['sigmoid', 'relu', 'leaky_relu', 'prelu', 'elu', 'tanh', 'softplus']
    reduce_ops = ['sum_reduce', 'min_reduce', 'max_reduce', 'mean_reduce', 'any_reduce', 'all_reduce']
    skippable_norm_ops = ['local_mean_normalization', 'local_variance_normalization',
                          'local_contrast_normalization', 'l1_normalization', 'l2_normalization']
    quantization_ops = ['linear_quantize', 'logarithmic_quantize']
    skippable_other_ops = ['select', 'concat', 'stack', 'unstack', 'copy_n', 'add_n', 'moments', 'box',
                           'pad', 'pad_grad', 'clamp', 'split', 'softmax']

    op_names = (unary_ops + binary_ops + activation_ops + reduce_ops + skippable_norm_ops + quantization_ops
                + skippable_other_ops)

    ops = [TransposableOperation(name=name, dg_transpose=transpose_operation_default) for name in op_names]
    ops.append(TransposableOperation(name="slice", dg_transpose=transpose_operation_slice))
    return ops


_DefaultTransposableOps = _get_default_transposable_ops()  # type: typing.List[TransposableOperation]


def optimize(g,  # type: NNEFGraph
             remove_unneeded_copies=False,  # type: bool
             remove_inverse_transposes=False,  # type: bool
             merge_transforms_into_variables=False,  # type: bool
             merge_transforms_into_constants=False,  # type: bool
             custom_transposable_ops=None,  # type: typing.Optional[typing.List[TransposableOperation]]
             io_transform=None,  # type:typing.Optional[TrafoOrTrafoDictType]
             verbose=False,  # type: bool
             rename_tensors=False,  # type: bool
             ):
    transposable_ops = _DefaultTransposableOps + (custom_transposable_ops if custom_transposable_ops else [])
    return optimize_impl(g=g,
                         driver=NNEFDataFormatOptimizationDriver(),
                         remove_unneeded_copies=remove_unneeded_copies,
                         remove_inverse_transposes=remove_inverse_transposes,
                         merge_transforms_into_variables=merge_transforms_into_variables,
                         merge_transforms_into_constants=merge_transforms_into_constants,
                         transposable_ops=transposable_ops,
                         io_transform=io_transform,
                         verbose=verbose,
                         rename_tensors=rename_tensors)


class Optimizer(object):
    def __init__(self,
                 io_transform=None,  # type:typing.Optional[TrafoOrTrafoDictType]
                 custom_transposable_ops=None,  # type: typing.Optional[typing.List[TransposableOperation]]
                 merge_transforms_into_variables=False,  # type: bool
                 ):
        # type: (...)->None
        self._io_transform = io_transform
        self._custom_transposable_ops = custom_transposable_ops
        self._merge_transforms_into_variables = merge_transforms_into_variables

    def __call__(self, g):
        # type: (NNEFGraph)->ConversionInfo
        transposable_ops = (_DefaultTransposableOps +
                            (self._custom_transposable_ops if self._custom_transposable_ops else []))

        return optimize_impl(g=g,
                             driver=NNEFDataFormatOptimizationDriver(),
                             remove_unneeded_copies=True,
                             remove_inverse_transposes=True,
                             merge_transforms_into_variables=self._merge_transforms_into_variables,
                             merge_transforms_into_constants=True,
                             transposable_ops=transposable_ops,
                             io_transform=self._io_transform,
                             verbose=False,
                             rename_tensors=True)
