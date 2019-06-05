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

from nnef_tools.conversion.conversion_info import ConversionInfo
from nnef_tools.io.tensorflow.tf_graph import TFGraph, TFOperation, TFTensor
from nnef_tools.optimization.data_format_optimizer import *


class TFDataFormatOptimizationDriver(DataFormatOptimizationDriver):
    @property
    def graph_type(self):
        return TFGraph

    @property
    def tensor_type(self):
        return TFTensor

    @property
    def op_type(self):
        return TFOperation

    @property
    def conv_grad_filter_op_names(self):
        return ["tf.nn.conv2d_backprop_filter",
                "tf.nn.conv3d_backprop_filter_v2",
                "tf.nn.depthwise_conv2d_native_backprop_filter"]

    @property
    def squeeze_op_name(self):
        return "tf.squeeze"

    @property
    def unsqueeze_op_name(self):
        return "tf.expand_dims"

    @property
    def transpose_op_name(self):
        return "tf.transpose"

    @property
    def reshape_op_name(self):
        return "tf.reshape"

    @property
    def copy_op_name(self):
        return "tf.identity"

    def get_axes_from_squeeze(self, squeeze):
        assert isinstance(squeeze.attribs["axis"], list)
        return squeeze.attribs["axis"]

    def set_axes_on_squeeze(self, squeeze, axes):
        squeeze.attribs["axis"] = axes

    def get_axes_from_unsqueeze(self, unsqueeze):
        assert isinstance(unsqueeze.attribs["axis"], int)
        return [unsqueeze.attribs["axis"]]

    def get_axes_from_transpose(self, transpose):
        assert isinstance(transpose.attribs["perm"], list)
        return transpose.attribs["perm"]

    def set_axes_on_transpose(self, transpose, axes):
        transpose.attribs["perm"] = axes

    def get_shape_from_reshape(self, reshape):
        assert isinstance(reshape.attribs["shape"], list)
        return reshape.attribs["shape"]

    def create_tensor(self, graph, name, shape, dtype):
        return TFTensor(graph=graph, name=name, shape=shape, dtype=dtype)

    def create_transpose_op(self, graph, input, output, axes):
        return TFOperation(graph=graph, name="tf.transpose", inputs=input, attribs=dict(perm=axes), outputs=output)

    def create_copy_op(self, graph, input, output):
        return TFOperation(graph=graph, name="tf.identity", inputs=input, outputs=output)

    def generate_missing_names(self, graph):
        graph.generate_missing_names()

    def get_input_of_transform(self, transform):
        return transform.input


def transpose_operation_default(transposer, graph, op, perm):
    # type: (Transposer, TFGraph, TFOperation, typing.List[int])->None
    if "begin" in op.attribs:
        op.attribs["begin"] = transposer.apply_permutation(op.attribs["begin"], perm)
    if "size" in op.attribs:
        op.attribs["size"] = transposer.apply_permutation(op.attribs["size"], perm)
    if "paddings" in op.attribs:
        op.attribs["paddings"] = transposer.apply_permutation(op.attribs["paddings"], perm)
    if "axis" in op.attribs:
        if isinstance(op.attribs["axis"], (list, tuple)):
            op.attribs["axis"] = transposer.apply_permutation_to_axes(op.attribs["axis"], perm)
        else:
            op.attribs["axis"] = transposer.apply_permutation_to_axis(op.attribs["axis"], perm)
    if "axes" in op.attribs:
        op.attribs["axes"] = transposer.apply_permutation_to_axes(op.attribs["axes"], perm)


def _get_default_transposable_ops():
    unary_ops = ['tf.identity', 'tf.negative', 'tf.reciprocal', 'tf.exp', 'tf.log', 'tf.abs', 'tf.sign',
                 'tf.logical_not', 'tf.floor', 'tf.ceil', 'tf.round', 'tf.square', 'tf.sqrt', 'tf.rsqrt']

    binary_ops = ['tf.add', 'tf.subtract', 'tf.multiply', 'tf.divide', 'tf.pow', 'tf.less', 'tf.greater',
                  'tf.less_equal', 'tf.greater_equal', 'tf.equal', 'tf.not_equal', 'tf.logical_and', 'tf.logical_or',
                  'tf.minimum', 'tf.maximum']

    activation_ops = ['tf.nn.sigmoid', 'tf.nn.relu', 'tf.nn.relu6', 'tf.nn.leaky_relu', 'tf.nn.elu', 'tf.nn.tanh',
                      'tf.nn.softplus', 'tf.nn.softsign']

    reduce_ops = ['tf.reduce_sum', 'tf.reduce_mean', 'tf.reduce_max', 'tf.reduce_min', 'tf.reduce_any', 'tf.reduce_all']

    skippable_norm_ops = ['tf.nn.l2_normalize']

    skippable_other_ops = ['tf.where', 'tf.concat', 'tf.stack', 'tf.unstack', 'tf.add_n', 'tf.nn.moments',
                           'tf.pad', '_tf.mirror_pad_grad', 'tf.clip_by_value', 'tf.split', 'tf.nn.softmax', "tf.slice"]

    op_names = (unary_ops + binary_ops + activation_ops + reduce_ops + skippable_norm_ops
                + skippable_other_ops)

    ops = [TransposableOperation(name=name, dg_transpose=transpose_operation_default) for name in op_names]
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
        # type: (TFGraph)->ConversionInfo
        transposable_ops = (_DefaultTransposableOps +
                            (self._custom_transposable_ops if self._custom_transposable_ops else []))

        return optimize_impl(g=g,
                             driver=TFDataFormatOptimizationDriver(),
                             remove_unneeded_copies=True,
                             remove_inverse_transposes=True,
                             merge_transforms_into_variables=self._merge_transforms_into_variables,
                             merge_transforms_into_constants=True,
                             transposable_ops=transposable_ops,
                             io_transform=self._io_transform,
                             verbose=False,
                             rename_tensors=True)
