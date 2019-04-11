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

import numpy as np

from nnef_tools.shape_inference import shape_inference as infer
from nnef_tools.io.tensorflow.tf_graph import *


def _evaluate_constant(tf_tensor):
    # type: (TFTensor)->np.ndarray

    # noinspection PySimplifyBooleanCheck
    if tf_tensor.data == []:
        return np.array([], dtype=np.dtype(tf_tensor.dtype))

    value = np.array(tf_tensor.data, dtype=np.dtype(tf_tensor.dtype))
    last_val = value.flat[-1]
    value2 = np.full(shape=tf_tensor.shape, fill_value=last_val, dtype=np.dtype(tf_tensor.dtype))
    value2.flat[:value.size] = value.flat
    return value2


def evaluate_shape(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.input.shape is not None:
        const_value_by_tensor[op.output] = np.array(op.input.shape)


def evaluate_add(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    a, b = op.inputs
    if a in const_value_by_tensor and b in const_value_by_tensor:
        const_value_by_tensor[op.output] = const_value_by_tensor[a] + const_value_by_tensor[b]


def evaluate_sub(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    a, b = op.inputs
    if a in const_value_by_tensor and b in const_value_by_tensor:
        const_value_by_tensor[op.output] = const_value_by_tensor[a] - const_value_by_tensor[b]


def evaluate_pack(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if all(input in const_value_by_tensor for input in op.inputs):
        const_value_by_tensor[op.output] = np.stack(arrays=[const_value_by_tensor[input] for input in op.inputs],
                                                    axis=op.attribs["axis"])


def evaluate_slice(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None

    input, begin, size = op.inputs
    if input in const_value_by_tensor and begin in const_value_by_tensor and size in const_value_by_tensor:
        input = const_value_by_tensor[input]
        begin = const_value_by_tensor[begin]
        size = const_value_by_tensor[size]
        const_value_by_tensor[op.output] = input[tuple(slice(b, b + s, 1) for b, s in zip(begin, size))]


def evaluate_strided_slice(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None

    input, begin, end, strides = op.inputs
    if (input in const_value_by_tensor and begin in const_value_by_tensor and end in const_value_by_tensor
            and strides in const_value_by_tensor):
        input_arr = const_value_by_tensor[input]
        begin_arr = const_value_by_tensor[begin]
        end_arr = const_value_by_tensor[end]
        strides_arr = const_value_by_tensor[strides]

        ssl_begin, ssl_end, ssl_stride, ssl_shape, reshape_shape = infer.decompose_strided_slice(
            input=input.shape,
            begin=begin_arr.tolist(),
            end=end_arr.tolist(),
            stride=strides_arr.tolist(),
            ellipsis_mask=op.attribs[
                "ellipsis_mask"],
            new_axis_mask=op.attribs[
                "new_axis_mask"],
            shrink_axis_mask=op.attribs[
                "shrink_axis_mask"],
            begin_mask=op.attribs["begin_mask"],
            end_mask=op.attribs["end_mask"])

        const_value_by_tensor[op.output] = input_arr[
            tuple(slice(b, e, s) for b, e, s in zip(ssl_begin, ssl_end, ssl_stride))
        ].reshape(reshape_shape)


_DefaultOpEvaluators = {
    "Shape": evaluate_shape,
    "Slice": evaluate_slice,
    "StridedSlice": evaluate_strided_slice,
    "Pack": evaluate_pack,
    "Add": evaluate_add,
    "Sub": evaluate_sub
}
