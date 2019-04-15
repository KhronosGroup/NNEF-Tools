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

import tensorflow as tf


def _getattr(module, *names):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
        _name = "_" + name
        if hasattr(module, _name):
            return getattr(module, _name)
    return None


class _Functions(object):
    def __init__(self):
        self.sinh = _getattr(tf, "sinh")
        self.cosh = _getattr(tf, "cosh")
        self.leaky_relu = _getattr(tf.nn, "leaky_relu")


class _InternalFunctions(object):
    def __init__(self):
        from tensorflow.python.ops import gen_array_ops as tf_gen_array_ops
        from tensorflow.python.ops import array_grad as tf_array_grad
        from tensorflow.python.ops import gen_math_ops as tf_gen_math_ops
        from tensorflow.python.ops import math_grad as tf_math_grad
        from tensorflow.python.ops import gen_nn_ops as tf_gen_nn_ops
        from tensorflow.python.ops import gen_image_ops as tf_gen_image_ops
        from tensorflow.python.ops import variables as tf_variables

        self.RefVariable = _getattr(tf_variables, "RefVariable")
        self.add = _getattr(tf_gen_math_ops, "add")
        self.div = _getattr(tf_gen_math_ops, "div")
        self.pow = _getattr(tf_gen_math_ops, "pow")
        self.logical_and = _getattr(tf_gen_math_ops, "logical_and")
        self.logical_or = _getattr(tf_gen_math_ops, "logical_or")
        self.reciprocal = _getattr(tf_gen_math_ops, "reciprocal")
        self.logical_not = _getattr(tf_gen_math_ops, "logical_not")
        self.abs = _getattr(tf_gen_math_ops, "abs")
        self.sign = _getattr(tf_gen_math_ops, "sign")
        self.exp = _getattr(tf_gen_math_ops, "exp")
        self.log = _getattr(tf_gen_math_ops, "log")
        self.square = _getattr(tf_gen_math_ops, "square")
        self.floor = _getattr(tf_gen_math_ops, "floor")
        self.ceil = _getattr(tf_gen_math_ops, "ceil")
        self.round = _getattr(tf_gen_math_ops, "round")
        self.greater = _getattr(tf_gen_math_ops, "greater")
        self.greater_equal = _getattr(tf_gen_math_ops, "greater_equal")
        self.less = _getattr(tf_gen_math_ops, "less")
        self.less_equal = _getattr(tf_gen_math_ops, "less_equal")
        self.equal = _getattr(tf_gen_math_ops, "equal")
        self.not_equal = _getattr(tf_gen_math_ops, "not_equal")
        self.sqrt = _getattr(tf_gen_math_ops, "sqrt")
        self.rsqrt = _getattr(tf_gen_math_ops, "rsqrt")
        self.range = _getattr(tf_gen_math_ops, "range")
        self.rank = _getattr(tf_gen_array_ops, "rank")
        self.conv3d_backprop_input_v2 = _getattr(tf_gen_nn_ops, "conv3d_backprop_input_v2")
        self.fused_batch_norm = _getattr(tf_gen_nn_ops, "fused_batch_norm")
        self.transpose = _getattr(tf_gen_array_ops, "transpose")
        self.strided_slice_grad = _getattr(tf_gen_array_ops, "strided_slice_grad")
        self.bias_add_grad = _getattr(tf_gen_nn_ops, "bias_add_grad")
        self.fused_batch_norm_grad = _getattr(tf_gen_nn_ops, "fused_batch_norm_grad")
        self.resize_nearest_neighbor_grad = _getattr(tf_gen_image_ops, "resize_nearest_neighbor_grad")
        self.resize_bilinear_grad = _getattr(tf_gen_image_ops, "resize_bilinear_grad")
        self.resize_bicubic_grad = _getattr(tf_gen_image_ops, "resize_bicubic_grad")
        self.TransposeGrad = _getattr(tf_array_grad, "TransposeGrad")
        self.MinOrMaxGrad = _getattr(tf_math_grad, "MinOrMaxGrad")
        self.fused_batch_norm_grad_v2 = _getattr(tf_gen_nn_ops, "fused_batch_norm_grad_v2")
        self.sub = _getattr(tf_gen_math_ops, "sub")
        self.mul = _getattr(tf_gen_math_ops, "mul")
        self.real_div = _getattr(tf_gen_math_ops, "real_div")
        self.neg = _getattr(tf_gen_math_ops, "neg")
        self.mat_mul = _getattr(tf_gen_math_ops, "mat_mul")
        self.softmax = _getattr(tf_gen_nn_ops, "softmax")
        self.concat_offset = _getattr(tf_gen_array_ops, "concat_offset")
        self.fused_batch_norm_v2 = _getattr(tf_gen_nn_ops, "fused_batch_norm_v2")
        self.max_pool_grad = _getattr(tf_gen_nn_ops, "max_pool_grad")
        self.max_pool_grad_with_argmax = _getattr(tf_gen_nn_ops, "max_pool_grad_with_argmax")
        self.avg_pool_grad = _getattr(tf_gen_nn_ops, "avg_pool_grad")
        self.sqrt_grad = _getattr(tf_gen_math_ops, "sqrt_grad")
        self.elu_grad = _getattr(tf_gen_nn_ops, "elu_grad")
        self.relu_grad = _getattr(tf_gen_nn_ops, "relu_grad")
        self.relu6_grad = _getattr(tf_gen_nn_ops, "relu6_grad")
        self.softplus_grad = _getattr(tf_gen_nn_ops, "softplus_grad")
        self.rsqrt_grad = _getattr(tf_gen_math_ops, "rsqrt_grad")
        self.sigmoid_grad = _getattr(tf_gen_math_ops, "sigmoid_grad")
        self.tanh_grad = _getattr(tf_gen_math_ops, "tanh_grad")
        self.reciprocal_grad = _getattr(tf_gen_math_ops, "reciprocal_grad")
        self.lrn_grad = _getattr(tf_gen_nn_ops, "lrn_grad")
        self.mirror_pad_grad = _getattr(tf_gen_array_ops, "mirror_pad_grad")
        self.broadcast_gradient_args = _getattr(tf_gen_array_ops, "broadcast_gradient_args")


tf_functions = _Functions()
tf_internal = _InternalFunctions()
