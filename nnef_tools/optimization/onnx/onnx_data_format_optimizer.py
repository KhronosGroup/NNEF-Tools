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

from nnef_tools.io.onnx.onnx_graph import *
from nnef_tools.optimization.data_format_optimizer import *
from nnef_tools.conversion.conversion_info import ConversionInfo
from nnef_tools.core import utils


class ONNXDataFormatOptimizationDriver(DataFormatOptimizationDriver):
    @property
    def graph_type(self):
        return ONNXGraph

    @property
    def tensor_type(self):
        return ONNXTensor

    @property
    def op_type(self):
        return ONNXOperation

    @property
    def conv_grad_filter_op_names(self):
        return []

    @property
    def squeeze_op_name(self):
        return "Squeeze"

    @property
    def unsqueeze_op_name(self):
        return "Unsqueeze"

    @property
    def transpose_op_name(self):
        return "Transpose"

    @property
    def reshape_op_name(self):
        return "Reshape"

    @property
    def copy_op_name(self):
        return "Identity"

    def get_axes_from_squeeze(self, squeeze):
        return squeeze.attribs["axes"]

    def set_axes_on_squeeze(self, squeeze, axes):
        squeeze.attribs["axes"] = axes

    def get_axes_from_unsqueeze(self, unsqueeze):
        return unsqueeze.attribs["axes"]

    def get_axes_from_transpose(self, transpose):
        return transpose.attribs["perm"]

    def set_axes_on_transpose(self, transpose, axes):
        transpose.attribs["perm"] = axes

    def get_shape_from_reshape(self, reshape):
        # type: (ONNXOperation)->typing.List[int]

        assert reshape.inputs[1].is_constant and reshape.inputs[1].rank == 1
        return reshape.inputs[1].data

    def create_tensor(self, graph, name, shape, dtype):
        return ONNXTensor(graph=graph, name=name, shape=shape, dtype=dtype)

    def create_transpose_op(self, graph, input, output, axes):
        return ONNXOperation(graph=graph, name="Transpose", inputs=input, attribs=dict(perm=axes), outputs=output)

    def create_copy_op(self, graph, input, output):
        return ONNXOperation(graph=graph, name="Identity", inputs=input, outputs=output)

    def generate_missing_names(self, graph):
        graph.generate_missing_names()

    def get_input_of_transform(self, transform):
        return transform.inputs[0]

    def copy_quantization(self, from_tensor, to_tensor):
        pass  # no quantization


def transpose_operation_default(transposer, graph, op, perm):
    # type: (Transposer, ONNXGraph, ONNXOperation, typing.List[int])->None
    def to_flat(padding_pairs):
        return utils.concat_lists(utils.zip_inverse(2, padding_pairs))

    def to_pairs(padding_flat):
        half = len(padding_flat) // 2
        return list(zip(padding_flat[:half], padding_flat[half:]))

    if 'axis' in op.attribs:
        op.attribs["axis"] = transposer.apply_permutation_to_axis(op.attribs["axis"], perm)
    if 'axes' in op.attribs:
        op.attribs["axes"] = sorted(transposer.apply_permutation_to_axes(op.attribs["axes"], perm))
    if op.attribs.get('pads'):
        op.attribs["pads"] = to_flat(transposer.apply_permutation(to_pairs(op.attribs["pads"]), perm))
    if op.attribs.get('strides'):
        op.attribs["strides"] = transposer.apply_permutation(op.attribs["strides"], perm)
    if op.attribs.get('dilations'):
        op.attribs["dilations"] = transposer.apply_permutation(op.attribs["dilations"], perm)


def transpose_operation_slice(transposer, graph, op, perm):
    # type: (Transposer, ONNXGraph, ONNXOperation, typing.List[int])->None
    op.attribs["axes"], op.attribs["starts"], op.attribs["ends"] = transposer.zip_inverse(3, sorted(zip(
        transposer.apply_permutation_to_axes(op.attribs["axes"], perm), op.attribs["starts"], op.attribs["ends"])))


def _get_default_transposable_ops():
    unary_ops = ['Exp', 'Log', 'Abs', 'Sign', 'Reciprocal', 'Neg', 'Identity', 'Not', 'Floor', 'Ceil', 'Sqrt',
                 'Relu', 'Sigmoid', 'Tanh', 'Softmax', 'Softplus', 'Elu', 'LeakyRelu', 'Sin', 'Cos']
    binary_ops = ['Add', 'Sub', 'Mul', 'Div', 'Pow', 'Less', 'Greater', 'Equal', 'And', 'Or', 'Min', 'Max']
    reduce_ops = ['ReduceSum', 'ReduceMean', 'ReduceMax', 'ReduceMin', 'ArgMax', 'ArgMin']
    skippable_other_ops = ['select', 'concat', 'stack', 'unstack', 'copy_n', 'add_n', 'moments', 'box',
                           'pad', 'pad_grad', 'clamp', 'split', 'softmax']

    op_names = (unary_ops + binary_ops + reduce_ops + skippable_other_ops)

    ops = [TransposableOperation(name=name, dg_transpose=transpose_operation_default) for name in op_names]
    ops.append(TransposableOperation(name="Slice", dg_transpose=transpose_operation_slice))
    return ops


_DefaultTransposableOps = _get_default_transposable_ops()  # type: typing.List[TransposableOperation]


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
        # type: (ONNXGraph)->ConversionInfo
        transposable_ops = (_DefaultTransposableOps +
                            (self._custom_transposable_ops if self._custom_transposable_ops else []))

        return optimize_impl(g=g,
                             driver=ONNXDataFormatOptimizationDriver(),
                             remove_unneeded_copies=True,
                             remove_inverse_transposes=True,
                             merge_transforms_into_variables=self._merge_transforms_into_variables,
                             merge_transforms_into_constants=True,
                             transposable_ops=transposable_ops,
                             io_transform=self._io_transform,
                             verbose=False,
                             rename_tensors=True)
