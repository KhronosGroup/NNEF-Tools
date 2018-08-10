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

import tensorflow as tf
# from tensorflow.contrib.layers.python.layers import layers as tf_layers
# from tensorflow.python.layers import utils as tf_utils
from tensorflow.python.ops import gen_array_ops as tf_gen_array_ops
# from tensorflow.python.ops import array_grad as tf_array_grad
from tensorflow.python.ops import gen_math_ops as tf_gen_math_ops
# from tensorflow.python.ops import math_grad as tf_math_grad
from tensorflow.python.ops import gen_nn_ops as tf_gen_nn_ops

from ..common import utils

# from tensorflow.python.ops import nn_grad as tf_nn_grad

_placeholder_counter = 0


def _make_placeholder():
    global _placeholder_counter

    def not_a_tf_function():
        raise NotImplementedError()

    not_a_tf_function.__name__ += str(_placeholder_counter)
    _placeholder_counter += 1
    return not_a_tf_function


if utils.tf_version_greater_equal(1, 3):
    sinh = tf.sinh
    cosh = tf.cosh
else:
    sinh = _make_placeholder()
    cosh = _make_placeholder()


if utils.tf_version_greater_equal(1, 4):
    nn_leaky_relu = tf.nn.leaky_relu
    gen_nn_ops_fused_batch_norm_grad_v2 = tf_gen_nn_ops.fused_batch_norm_grad_v2
else:
    nn_leaky_relu = _make_placeholder()
    gen_nn_ops_fused_batch_norm_grad_v2 = _make_placeholder()

if utils.tf_version_greater_equal(1, 5):
    reduce_arg_keepdims = "keepdims"
    l2_normalize_arg_axis = "axis"
    softmax_arg_axis = "axis"
else:
    reduce_arg_keepdims = "keep_dims"
    l2_normalize_arg_axis = "dim"
    softmax_arg_axis = "dim"

if utils.tf_version_greater_equal(1, 7):
    gen_math_ops_sub = tf_gen_math_ops.sub
    gen_math_ops_mul = tf_gen_math_ops.mul
    gen_math_ops_real_div = tf_gen_math_ops.real_div
    gen_math_ops_neg = tf_gen_math_ops.neg
    gen_math_ops_mat_mul = tf_gen_math_ops.mat_mul
    gen_nn_ops_softmax = tf_gen_nn_ops.softmax
    gen_array_ops_concat_offset = tf_gen_array_ops.concat_offset
    gen_nn_ops_fused_batch_norm_v2 = tf_gen_nn_ops.fused_batch_norm_v2

    # grads
    gen_nn_ops_max_pool_grad = tf_gen_nn_ops.max_pool_grad
    gen_nn_ops_max_pool_grad_name = "tf_gen_nn_ops.max_pool_grad"
    gen_nn_ops_max_pool_grad_with_argmax = tf_gen_nn_ops.max_pool_grad_with_argmax
    gen_nn_ops_max_pool_grad_with_argmax_name = "tf_gen_nn_ops.max_pool_grad_with_argmax"
    gen_nn_ops_avg_pool_grad = tf_gen_nn_ops.avg_pool_grad
    gen_nn_ops_avg_pool_grad_name = "tf_gen_nn_ops.avg_pool_grad"
    gen_math_ops_sqrt_grad = tf_gen_math_ops.sqrt_grad
    gen_nn_ops_elu_grad = tf_gen_nn_ops.elu_grad
    gen_nn_ops_relu_grad = tf_gen_nn_ops.relu_grad
    gen_nn_ops_softplus_grad = tf_gen_nn_ops.softplus_grad
    gen_math_ops_rsqrt_grad = tf_gen_math_ops.rsqrt_grad
    gen_math_ops_sigmoid_grad = tf_gen_math_ops.sigmoid_grad
    gen_math_ops_tanh_grad = tf_gen_math_ops.tanh_grad
    gen_math_ops_reciprocal_grad = tf_gen_math_ops.reciprocal_grad
    gen_nn_ops_lrn_grad = tf_gen_nn_ops.lrn_grad
    gen_array_ops_mirror_pad_grad = tf_gen_array_ops.mirror_pad_grad
else:
    gen_math_ops_sub = tf_gen_math_ops._sub
    gen_math_ops_mul = tf_gen_math_ops._mul
    gen_math_ops_real_div = tf_gen_math_ops._real_div
    gen_math_ops_neg = tf_gen_math_ops._neg
    gen_math_ops_mat_mul = tf_gen_math_ops._mat_mul
    gen_nn_ops_softmax = tf_gen_nn_ops._softmax
    gen_array_ops_concat_offset = tf_gen_array_ops._concat_offset
    gen_nn_ops_fused_batch_norm_v2 = _make_placeholder()

    # grads
    gen_nn_ops_max_pool_grad = tf_gen_nn_ops._max_pool_grad
    gen_nn_ops_max_pool_grad_name = "tf_gen_nn_ops._max_pool_grad"
    gen_nn_ops_max_pool_grad_with_argmax = tf_gen_nn_ops._max_pool_grad_with_argmax
    gen_nn_ops_max_pool_grad_with_argmax_name = "tf_gen_nn_ops._max_pool_grad_with_argmax"
    gen_nn_ops_avg_pool_grad = tf_gen_nn_ops._avg_pool_grad
    gen_nn_ops_avg_pool_grad_name = "tf_gen_nn_ops._avg_pool_grad"
    gen_math_ops_sqrt_grad = tf_gen_math_ops._sqrt_grad
    gen_nn_ops_elu_grad = tf_gen_nn_ops._elu_grad
    gen_nn_ops_relu_grad = tf_gen_nn_ops._relu_grad
    gen_nn_ops_softplus_grad = tf_gen_nn_ops._softplus_grad
    gen_math_ops_rsqrt_grad = tf_gen_math_ops._rsqrt_grad
    gen_math_ops_sigmoid_grad = tf_gen_math_ops._sigmoid_grad
    gen_math_ops_tanh_grad = tf_gen_math_ops._tanh_grad
    gen_math_ops_reciprocal_grad = tf_gen_math_ops._reciprocal_grad
    gen_nn_ops_lrn_grad = tf_gen_nn_ops._lrn_grad
    gen_array_ops_mirror_pad_grad = tf_gen_array_ops._mirror_pad_grad


if utils.tf_version_greater_equal(1, 8):
    gen_array_ops_broadcast_gradient_args = tf_gen_array_ops.broadcast_gradient_args
else:
    gen_array_ops_broadcast_gradient_args = tf_gen_array_ops._broadcast_gradient_args
