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

from nnef_tools.core import utils
from nnef_tools.io.tensorflow.tf_graph import *


def _np_permute(list_, perm_):
    assert len(list_) == len(perm_), "list_ must be same length as perm_"
    return np.array(list_)[np.array(perm_)]


def _tf_broadcasted_shape(shape1, shape2):
    rank_diff = len(shape2) - len(shape1)

    if rank_diff > 0:
        shape1 = [1] * rank_diff + shape1
    else:
        shape2 = [1] * -rank_diff + shape2

    shape = [1] * len(shape1)  # type: typing.List[typing.Optional[int]]

    assert len(shape) == len(shape1) == len(shape2)

    for i, (s, t) in enumerate(zip(shape1, shape2)):
        if s is None or t is None:
            shape[i] = None
            continue

        assert s == t or s == 1 or t == 1, \
            "Broadcast can only happen when the corresponding dimesions are either equal or one of them is 1"
        if s != 1:
            shape[i] = s
        elif t != 1:
            shape[i] = t

    return shape


def _unify_shape(shape):
    if shape is not None:
        return [int(s) for s in shape]
    return None


def evaluate_shape_of_constant(tensor, const_value_by_tensor):
    # type: (TFTensor, typing.Dict[TFTensor, np.ndarray])->None
    if tensor in const_value_by_tensor:
        tensor.shape = _unify_shape(np.shape(const_value_by_tensor[tensor]))


def evaluate_shape_of_transpose(op):
    # type: (TFOperation)->None
    input_shape = op.input.shape
    perm = op.attribs["perm"]
    if perm is None:
        op.output.shape = _unify_shape(reversed(input_shape))
    else:
        op.output.shape = _unify_shape(_np_permute(input_shape, perm))


def evaluate_shape_of_transpose_grad(op):
    # type: (TFOperation)->None
    grad_shape = op.inputs[2].shape  # grad
    perm = op.attribs["orig_perm"]
    if perm is None:
        op.output.shape = _unify_shape(reversed(grad_shape))
    else:
        op.output.shape = _unify_shape(_np_permute(grad_shape, utils.inverse_permutation(perm)))


def evaluate_shape_of_tile(op):
    # type: (TFOperation)->None
    input_shape = op.input.shape
    multiples = op.attribs["multiples"]
    assert len(input_shape) == len(multiples)
    op.output.shape = _unify_shape([int(s * m) for s, m in zip(input_shape, multiples)])


def evaluate_shape_of_reshape(op):
    # type: (TFOperation)->None
    shape = op.attribs["shape"]
    assert shape is not None
    if not any(s == -1 for s in shape):
        op.output.shape = _unify_shape(shape)
    elif sum(1 if s == -1 else 0 for s in shape) == 1:
        old_shape_ = op.input.shape
        new_shape_ = list(shape)
        if old_shape_ is not None and not any(s is None for s in old_shape_):
            prod_old_shape = int(np.product(old_shape_))
            prod_new_shape = int(np.product([s for s in new_shape_ if s != -1]))
            assert prod_old_shape % prod_new_shape == 0, "New size does not divide old size"
            new_shape_[new_shape_.index(-1)] = prod_old_shape // prod_new_shape
            op.output.shape = _unify_shape(new_shape_)


def generic_evaluate_shape_of_unary(op):
    # type: (TFOperation)->None
    op.output.shape = _unify_shape(op.input.shape)


def generic_evaluate_shape_of_binary(op):
    # type: (TFOperation)->None
    op.output.shape = _unify_shape(_tf_broadcasted_shape(op.inputs[0].shape, op.inputs[1].shape))


def evaluate_shape_of_expand_dims(op):
    # type: (TFOperation)->None
    axis = op.attribs["axis"]
    if axis == -1:
        axis = op.input.rank
    op.output.shape = _unify_shape(op.input.shape[:axis] + [1] + op.input.shape[axis:])


def evaluate_shape_of_reduce_sum(op):
    # type: (TFOperation)->None
    input_shape = op.input.shape
    axis = op.attribs["axis"]
    keepdims = op.attribs["keepdims"]
    op.output.shape = _unify_shape([s if i not in axis else 1
                                    for i, s in enumerate(input_shape)
                                    if keepdims or i not in axis])


def evaluate_shape_of_operation(op, const_value_by_tensor):
    # type: (TFOperation, typing.Dict[TFTensor, np.ndarray])->None

    if all(output.shape is not None and all(s is not None for s in output.shape) for output in op.outputs):
        return

    old_shapes = [output.shape for output in op.outputs]

    if all(output in const_value_by_tensor for output in op.outputs):
        for output in op.outputs:
            output.shape = list(np.shape(const_value_by_tensor[output]))
    elif op.name in _DefaultOpShapeEvaluators:
        _DefaultOpShapeEvaluators[op.name](op)

    for old_shape, tensor in zip(old_shapes, op.outputs):
        assert tensor.shape is not None and all(s is not None for s in tensor.shape)
        assert utils.compatible_shapes(old_shape, tensor.shape), \
            "{}: Evaluated shape ({}) not compatible with original shape ({})".format(tensor, tensor.shape, old_shape)


_DefaultOpShapeEvaluators = {
    "tf.transpose": evaluate_shape_of_transpose,
    "_tf.TransposeGrad": evaluate_shape_of_transpose_grad,
    "tf.tile": evaluate_shape_of_tile,
    "tf.reshape": evaluate_shape_of_reshape,
    "tf.nn.softmax": generic_evaluate_shape_of_unary,
    "tf.negative": generic_evaluate_shape_of_unary,
    "tf.multiply": generic_evaluate_shape_of_binary,
    "tf.subtract": generic_evaluate_shape_of_binary,
    "tf.reduce_sum": evaluate_shape_of_reduce_sum
}
