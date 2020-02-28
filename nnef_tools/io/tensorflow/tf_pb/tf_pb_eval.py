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


def evaluate_mul(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    a, b = op.inputs
    if a in const_value_by_tensor and b in const_value_by_tensor:
        const_value_by_tensor[op.output] = const_value_by_tensor[a] * const_value_by_tensor[b]


def evaluate_min(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    a, b = op.inputs
    if a in const_value_by_tensor and b in const_value_by_tensor:
        const_value_by_tensor[op.output] = np.minimum(const_value_by_tensor[a], const_value_by_tensor[b])


def evaluate_max(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    a, b = op.inputs
    if a in const_value_by_tensor and b in const_value_by_tensor:
        const_value_by_tensor[op.output] = np.maximum(const_value_by_tensor[a], const_value_by_tensor[b])


def evaluate_real_div(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    a, b = op.inputs
    if a in const_value_by_tensor and b in const_value_by_tensor:
        const_value_by_tensor[op.output] = const_value_by_tensor[a] / const_value_by_tensor[b]


def evaluate_concat_v2(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if all(input in const_value_by_tensor for input in op.inputs):
        inputs = tuple(const_value_by_tensor[input] for input in op.inputs[:-1])
        axis = const_value_by_tensor[op.inputs[-1]]
        const_value_by_tensor[op.output] = np.concatenate(inputs, axis.item())


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


def _is_bit_set(val, idx):
    return ((val >> idx) & 1) != 0


def evaluate_strided_slice(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None

    input, begin, end, strides = op.inputs

    if input in const_value_by_tensor and begin in const_value_by_tensor and end in const_value_by_tensor and \
            strides in const_value_by_tensor:
        begin_mask = op.attribs["begin_mask"]
        end_mask = op.attribs["end_mask"]
        new_axis_mask = op.attribs["new_axis_mask"]
        shrink_axis_mask = op.attribs["shrink_axis_mask"]
        ellipsis_mask = op.attribs["ellipsis_mask"]

        input = const_value_by_tensor[input]
        begin = const_value_by_tensor[begin].tolist()
        end = const_value_by_tensor[end].tolist()
        strides = const_value_by_tensor[strides].tolist()

        index = tuple(b if _is_bit_set(shrink_axis_mask, i) else
                      np.newaxis if _is_bit_set(new_axis_mask, i) else
                      Ellipsis if _is_bit_set(ellipsis_mask, i) else
                      slice(b if not _is_bit_set(begin_mask, i) else None,
                            e if not _is_bit_set(end_mask, i) else None, s)
                      for i, (b, e, s) in enumerate(zip(begin, end, strides)))

        const_value_by_tensor[op.output] = input[index]


def evaluate_cast(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None

    dstT = op.attribs['DstT']

    if op.input in const_value_by_tensor:
        input = const_value_by_tensor[op.input]
        const_value_by_tensor[op.output] = input.astype(_NumpyDtype[dstT])


_NumpyDtype = {
    'DT_INVALID': None,
    'DT_HALF': np.float16,
    'DT_FLOAT': np.float32,
    'DT_DOUBLE': np.float64,
    'DT_INT8': np.int8,
    'DT_INT16': np.int16,
    'DT_INT32': np.int32,
    'DT_INT64': np.int64,
    'DT_UINT8': np.uint8,
    'DT_UINT16': np.uint16,
    'DT_UINT32': np.uint32,
    'DT_UINT64': np.uint64,
    'DT_BOOL': np.bool,
    'DT_STRING': np.str,
    'DT_COMPLEX64': np.complex64,
    'DT_COMPLEX128': np.complex128,
}


_DefaultOpEvaluators = {
    "Shape": evaluate_shape,
    "Slice": evaluate_slice,
    "StridedSlice": evaluate_strided_slice,
    "Pack": evaluate_pack,
    "Add": evaluate_add,
    "Sub": evaluate_sub,
    "Mul": evaluate_mul,
    "Minimum": evaluate_min,
    "Maximum": evaluate_max,
    "RealDiv": evaluate_real_div,
    "ConcatV2": evaluate_concat_v2,
    "Cast": evaluate_cast,
}
