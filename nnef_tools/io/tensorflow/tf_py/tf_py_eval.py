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


def _np_inverse_permutation(p):
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def _list2tuple(x):
    return tuple(x) if isinstance(x, list) else x


def evaluate_constant(tensor, const_value_by_tensor):
    # type: (TFTensor, typing.Dict[TFTensor, np.ndarray])->None
    const_value_by_tensor[tensor] = _evaluate_constant(tensor)


def evaluate_rank(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.input.rank is not None:
        const_value_by_tensor[op.output] = np.array(op.input.rank)


def evaluate_shape(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.input.shape is not None:
        const_value_by_tensor[op.output] = np.array(op.input.shape)


def evaluate_shape_n(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    for i, o in zip(op.inputs, op.outputs):
        if i.shape is not None:
            const_value_by_tensor[o] = np.array(i.shape)


def evaluate_add(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.inputs[0] in const_value_by_tensor and op.inputs[1] in const_value_by_tensor:
        const_value_by_tensor[op.output] = const_value_by_tensor[op.inputs[0]] + const_value_by_tensor[op.inputs[1]]


def evaluate_subtract(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.inputs[0] in const_value_by_tensor and op.inputs[1] in const_value_by_tensor:
        const_value_by_tensor[op.output] = const_value_by_tensor[op.inputs[0]] - const_value_by_tensor[op.inputs[1]]


def evaluate_multiply(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.inputs[0] in const_value_by_tensor and op.inputs[1] in const_value_by_tensor:
        const_value_by_tensor[op.output] = const_value_by_tensor[op.inputs[0]] * const_value_by_tensor[op.inputs[1]]


def evaluate_divide(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.inputs[0] in const_value_by_tensor and op.inputs[1] in const_value_by_tensor:
        const_value_by_tensor[op.output] = const_value_by_tensor[op.inputs[0]] / const_value_by_tensor[op.inputs[1]]


def evaluate_mod(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.inputs[0] in const_value_by_tensor and op.inputs[1] in const_value_by_tensor:
        const_value_by_tensor[op.output] = np.mod(const_value_by_tensor[op.inputs[0]],
                                                  const_value_by_tensor[op.inputs[1]])


def evaluate_floor_div(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.inputs[0] in const_value_by_tensor and op.inputs[1] in const_value_by_tensor:
        const_value_by_tensor[op.output] = np.floor_divide(const_value_by_tensor[op.inputs[0]],
                                                           const_value_by_tensor[op.inputs[1]])


def evaluate_squeeze(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.input in const_value_by_tensor:
        const_value_by_tensor[op.output] = np.squeeze(const_value_by_tensor[op.input], axis=_list2tuple(op.attribs["axis"]))


def evaluate_expand_dims(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.input in const_value_by_tensor:
        const_value_by_tensor[op.output] = np.expand_dims(const_value_by_tensor[op.input], axis=_list2tuple(op.attribs["axis"]))


def evaluate_maximum(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.inputs[0] in const_value_by_tensor and op.inputs[1] in const_value_by_tensor:
        const_value_by_tensor[op.output] = np.maximum(const_value_by_tensor[op.inputs[0]],
                                                      const_value_by_tensor[op.inputs[1]])


def evaluate_minimum(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.inputs[0] in const_value_by_tensor and op.inputs[1] in const_value_by_tensor:
        const_value_by_tensor[op.output] = np.minimum(const_value_by_tensor[op.inputs[0]],
                                                      const_value_by_tensor[op.inputs[1]])


def evaluate_fill(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    const_value_by_tensor[op.output] = np.full(op.attribs["dims"], op.attribs["value"])


def evaluate_dynamic_stitch(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    data = np.array([np.array(d) for d in op.attribs["data"]])
    indices = np.array([np.array(d) for d in op.attribs["indices"]])
    assert len(data[0].shape) == 1
    for i in range(len(data)):
        if len(data[i].shape) == 0:
            data[i] = np.array([data[i].item()])
    for i in range(len(indices)):
        if len(indices[i].shape) == 0:
            indices[i] = np.array([indices[i].item()])
    max_index = -1
    value = np.zeros(shape=[sum(len(d) for d in data)], dtype=data.dtype)
    for data2, indices2 in zip(data, indices):
        for d, i in zip(data2, indices2):
            if i > max_index:
                max_index = i
            value[i] = d
    value = value[:max_index + 1]
    const_value_by_tensor[op.output] = value


def evaluate_stack(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if all(t in const_value_by_tensor for t in op.inputs):
        const_value_by_tensor[op.output] = np.stack([const_value_by_tensor[t] for t in op.inputs],
                                                    axis=op.attribs["axis"])


def evaluate_unstack(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    axis = op.attribs["axis"]
    if op.input in const_value_by_tensor:
        items = np.split(const_value_by_tensor[op.input], axis=axis, indices_or_sections=op.input.shape[axis])
        for (output, item) in zip(op.outputs, items):
            const_value_by_tensor[output] = item


def evaluate_range(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    start = op.attribs["start"]
    limit = op.attribs["limit"]
    delta = op.attribs["delta"]
    dtype = op.attribs["dtype"]

    if limit is None:
        limit = start
        start = 0

    const_value_by_tensor[op.output] = np.arange(start=start, stop=limit, step=delta,
                                                 dtype=None if dtype is None else np.dtype(dtype))


def evaluate_concat(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if all(t in const_value_by_tensor for t in op.inputs):
        cat = np.concatenate(tuple(const_value_by_tensor[t] for t in op.inputs), axis=op.attribs["axis"])
        const_value_by_tensor[op.output] = np.array(cat, dtype=np.dtype(op.output.dtype))


def evaluate_split(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.input in const_value_by_tensor:
        items = np.split(const_value_by_tensor[op.input], axis=op.attribs["axis"],
                         indices_or_sections=op.attribs["num_or_size_splits"])
        for output, item in zip(op.outputs, items):
            const_value_by_tensor[output] = item


def evaluate_slice(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.input in const_value_by_tensor:
        begin = np.array(op.attribs["begin"], dtype=np.int64)
        size = np.array([size_dim if size_dim != -1 else shape_dim
                         for size_dim, shape_dim in zip(op.attribs["size"], op.input.shape)], dtype=np.int64)
        if len(begin.shape) == 0:
            begin = np.array([begin.item()])
        if len(size.shape) == 0:
            size = np.array([size.item()])
        input = const_value_by_tensor[op.input]
        const_value_by_tensor[op.output] = input[tuple(slice(b, b + s, 1) for b, s in zip(begin, size))]


def evaluate_strided_slice(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.input in const_value_by_tensor:
        begin = np.array(op.attribs["begin"], dtype=np.int64)
        end = np.array(op.attribs["end"], dtype=np.int64)
        strides = np.array(op.attribs["strides"], dtype=np.int64)
        input = const_value_by_tensor[op.input]
        const_value_by_tensor[op.output] = input[tuple(slice(b, e, s) for b, e, s in zip(begin, end, strides))]


def evaluate_invert_permutation(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.input in const_value_by_tensor:
        const_value_by_tensor[op.output] = _np_inverse_permutation(const_value_by_tensor[op.input])


def evaluate_reduce_sum(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.input in const_value_by_tensor:
        const_value_by_tensor[op.output] = np.sum(a=const_value_by_tensor[op.input],
                                                  axis=_list2tuple(op.attribs["axis"]),
                                                  keepdims=bool(op.attribs["keepdims"]))


def evaluate_broadcast_gradient_args(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    s0 = np.array(op.attribs["s0"])
    s1 = np.array(op.attribs["s1"])
    s0 = list(s0)
    s1 = list(s1)

    if len(s0) < len(s1):
        s0 = [1] * (len(s1) - len(s0)) + s0
    elif len(s1) < len(s0):
        s1 = [1] * (len(s0) - len(s1)) + s1

    reduce_indices0 = []
    reduce_indices1 = []
    for i, (s, t) in enumerate(zip(s0, s1)):
        if s == 1:
            reduce_indices0.append(i)
        if t == 1:
            reduce_indices1.append(i)

    const_value_by_tensor[op.outputs[0]] = np.array(reduce_indices0)
    const_value_by_tensor[op.outputs[1]] = np.array(reduce_indices1)


def evaluate_reshape(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.input in const_value_by_tensor:
        input_ = const_value_by_tensor[op.input]
        shape = np.array([int(s) for s in op.attribs["shape"]])
        new_shape = None
        if not any(s == -1 for s in shape):
            new_shape = shape
        elif sum(1 if s == -1 else 0 for s in shape) == 1:  # TODO no code duplication
            old_shape_ = op.input.shape
            new_shape_ = list(shape.tolist())
            if old_shape_ is not None and not any(s is None for s in old_shape_):
                prod_old_shape = np.product(old_shape_)
                prod_new_shape = np.product([s for s in new_shape_ if s != -1])
                assert prod_old_shape % prod_new_shape == 0, "New size does not divide old size"
                new_shape_[new_shape_.index(-1)] = int(prod_old_shape // prod_new_shape)
                new_shape = new_shape_
        if new_shape is not None:
            const_value_by_tensor[op.output] = np.reshape(input_, new_shape)


def evaluate_transpose(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.input in const_value_by_tensor:
        input_ = const_value_by_tensor[op.input]
        perm = op.attribs["perm"]  # can be None, but it's OK
        const_value_by_tensor[op.output] = np.transpose(input_, perm)


def evaluate_concat_offset(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    concat_dim = int(op.attribs["concat_dim"])
    shape = op.attribs["shape"]
    results = []
    offsets = [0] * len(shape[0])
    for s in shape:
        results.append(list(offsets))
        offsets[concat_dim] += s[concat_dim]
    for o, r in zip(op.outputs, results):
        const_value_by_tensor[o] = np.array(r)


def evaluate_size(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.input.shape is not None and None not in op.input.shape:
        const_value_by_tensor[op.output] = np.product(op.input.shape)


def evaluate_cast(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.input in const_value_by_tensor:
        const_value_by_tensor[op.output] = const_value_by_tensor[op.input].astype(np.typeDict[op.attribs["dtype"]])


def try_to_evaluate_operation(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None
    if op.name in _DefaultOpEvaluators:
        _DefaultOpEvaluators[op.name](op, const_value_by_tensor)


_DefaultOpEvaluators = {
    "tf.rank": evaluate_rank,
    "tf.shape": evaluate_shape,
    "tf.shape_n": evaluate_shape_n,
    "tf.size": evaluate_size,
    "tf.add": evaluate_add,
    "tf.subtract": evaluate_subtract,
    "tf.multiply": evaluate_multiply,
    "tf.divide": evaluate_divide,
    "tf.floor_div": evaluate_floor_div,
    "tf.mod": evaluate_mod,
    "tf.maximum": evaluate_maximum,
    "tf.minimum": evaluate_minimum,
    "tf.fill": evaluate_fill,
    "tf.dynamic_stitch": evaluate_dynamic_stitch,
    "tf.stack": evaluate_stack,
    "tf.unstack": evaluate_unstack,
    "tf.range": evaluate_range,
    "tf.concat": evaluate_concat,
    "tf.split": evaluate_split,
    "tf.slice": evaluate_slice,
    "tf.strided_slice": evaluate_strided_slice,
    "tf.invert_permutation": evaluate_invert_permutation,
    "tf.reduce_sum": evaluate_reduce_sum,
    "_tf.broadcast_gradient_args": evaluate_broadcast_gradient_args,
    "tf.reshape": evaluate_reshape,
    "tf.transpose": evaluate_transpose,
    "_tf.concat_offset": evaluate_concat_offset,
    "tf.squeeze": evaluate_squeeze,
    "tf.expand_dims": evaluate_expand_dims,
    "tf.cast": evaluate_cast,
}
