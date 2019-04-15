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
from collections import OrderedDict

from nnef_tools.core import utils
from nnef_tools.io.tensorflow.tf_graph import *


class ArgProto(object):
    def __init__(self, arg_names, is_tensor, is_array, is_optional):
        self.arg_names = arg_names
        self.is_tensor = is_tensor
        self.is_array = is_array
        self.is_optional = is_optional

    @property
    def primary_arg_name(self):
        return self.arg_names[0]

    def __repr__(self):
        return "ArgProto({})".format(self.arg_names)


class OpProto(object):
    def __init__(self, op_name, arg_protos):
        self.op_name = op_name
        self.arg_protos = arg_protos  # type: typing.List[ArgProto]

    def list_tensor_arg_protos(self):
        return [a for a in self.arg_protos if a.is_tensor]

    def list_nontensor_arg_protos(self):
        return [a for a in self.arg_protos if not a.is_tensor]

    def __repr__(self):
        return "OpProto({})".format(self.op_name)


def parse_arg_proto(s):
    is_tensor = False
    is_optional = False
    is_array = False

    if s.endswith('?'):
        is_optional = True
        s = s[:-1]
    if s.endswith('[]'):
        is_array = True
        s = s[:-2]
    if ':' in s:
        s, t = s.split(':', 1)
        t = t.strip()
        assert t == "T"
        is_tensor = True

    arg_names = list(t.strip() for t in s.split('/'))

    assert all(utils.is_identifier(n) for n in arg_names), "{}".format(arg_names)
    return ArgProto(arg_names=arg_names, is_tensor=is_tensor, is_array=is_array, is_optional=is_optional)


def parse_op_proto(s):
    assert s and s[-1] == ')'
    s = s[:-1]
    op_name, args = s.split('(', 1)
    arg_protos = []
    for arg in args.split(','):
        arg = arg.strip()
        assert arg
        arg_protos.append(parse_arg_proto(arg))
        assert utils.is_identifier(op_name.replace('.', '_'))
    return OpProto(op_name=op_name, arg_protos=arg_protos)


def args_from_tfop(op, op_proto, allow_missing=False):
    # type: (TFOperation, OpProto, bool)->OrderedDict[str, typing.Any]
    args = OrderedDict()
    i_tensor = 0
    for arg_proto in op_proto.arg_protos:
        if arg_proto.is_tensor and arg_proto.is_array:
            args[arg_proto.primary_arg_name] = []
            while i_tensor < len(op.inputs):
                args[arg_proto.primary_arg_name].append(op.inputs[i_tensor])
                i_tensor += 1
        elif arg_proto.is_tensor and not arg_proto.is_array:
            assert i_tensor < len(op.inputs) or arg_proto.is_optional
            if i_tensor < len(op.inputs):
                args[arg_proto.primary_arg_name] = op.inputs[i_tensor]
                i_tensor += 1
        else:
            if not ((allow_missing or arg_proto.is_optional) and arg_proto.primary_arg_name not in op.attribs):
                args[arg_proto.primary_arg_name] = op.attribs[arg_proto.primary_arg_name]
    return args


CallableOrStr = typing.Union[typing.Callable, str]


class TraceableFunction(object):
    # fun can be str, it will be evaluated before tracing
    def __init__(self,
                 proto,  # type: typing.Union[str, OpProto]
                 fun,  # type: typing.Union[CallableOrStr, typing.List[CallableOrStr]]
                 ):
        # type: (...)->None

        funs = list(fun) if isinstance(fun, (list, tuple)) else [fun]
        funs = [f for f in funs if f is not None]

        self.op_proto = parse_op_proto(proto) if not isinstance(proto, OpProto) else proto
        self.functions = funs

    @staticmethod
    def _eval_function(__fun_name):
        exec("import tensorflow as tf")
        exec("from nnef_tools.io.tensorflow.tf_py.tf_py_compat import tf_internal as _tf")
        exec("from nnef_tools.io.tensorflow.tf_py.tf_py_compat import tf_functions as _tf_functions")
        return eval(__fun_name)

    def eval_functions(self):
        self.functions = [self._eval_function(fun) if utils.is_anystr(fun) else fun
                          for fun in self.functions]

    def __repr__(self):
        return "TraceableFunction({})".format(self.op_proto.op_name)


DefaultTraceableFunctions = [
    TraceableFunction("tf.gradients(xs:T[], ys:T[])", ["tf.gradients"]),
    TraceableFunction("tf.constant(value, dtype, shape, name)", ["tf.constant"]),
    TraceableFunction("tf.placeholder(shape, dtype, name)", ["tf.placeholder"]),
    TraceableFunction("tf.get_variable(shape, dtype, name)", ["tf.get_variable"]),
    TraceableFunction("tf.Variable(initial_value, dtype, name)", ["tf.Variable", "_tf.RefVariable"]),
    TraceableFunction("tf.assign(ref:T, value:T)", ["tf.assign"]),
    TraceableFunction("tf.concat(values:T[], axis)", ["tf.concat"]),
    TraceableFunction("tf.split(value:T, num_or_size_splits, axis)", ["tf.split"]),
    TraceableFunction("tf.reshape(tensor:T, shape)", ["tf.reshape"]),
    TraceableFunction("tf.squeeze(input:T, axis/squeeze_dims[])", ["tf.squeeze"]),
    TraceableFunction("tf.expand_dims(input:T, axis/dim)", ["tf.expand_dims"]),
    TraceableFunction("tf.transpose(a/x:T, perm)", ["tf.transpose", "_tf.transpose"]),
    TraceableFunction("tf.pad(tensor:T, paddings, mode, constant_values)", ["tf.pad"]),
    TraceableFunction("tf.add(x:T, y:T)", ["tf.add", "_tf.add"]),
    TraceableFunction("tf.subtract(x:T, y:T)", ["tf.subtract", "_tf.sub"]),
    TraceableFunction("tf.multiply(x:T, y:T)", ["tf.multiply", "_tf.mul"]),
    TraceableFunction("tf.divide(x:T, y:T)", ["tf.divide", "_tf.div", "_tf.real_div"]),
    TraceableFunction("tf.floor_div(x:T, y:T)", ["tf.floor_div"]),
    TraceableFunction("tf.mod(x:T, y:T)", ["tf.mod"]),
    TraceableFunction("tf.pow(x:T, y:T)", ["tf.pow", "_tf.pow"]),
    TraceableFunction("tf.logical_and(x:T, y:T)", ["tf.logical_and", "_tf.logical_and"]),
    TraceableFunction("tf.logical_or(x:T, y:T)", ["tf.logical_or", "_tf.logical_or"]),
    TraceableFunction("tf.greater(x:T, y:T)", ["tf.greater", "_tf.greater"]),
    TraceableFunction("tf.greater_equal(x:T, y:T)", ["tf.greater_equal", "_tf.greater_equal"]),
    TraceableFunction("tf.less(x:T, y:T)", ["tf.less", "_tf.less"]),
    TraceableFunction("tf.less_equal(x:T, y:T)", ["tf.less_equal", "_tf.less_equal"]),
    TraceableFunction("tf.equal(x:T, y:T)", ["tf.equal", "_tf.equal"]),
    TraceableFunction("tf.not_equal(x:T, y:T)", ["tf.not_equal", "_tf.not_equal"]),
    TraceableFunction("tf.squared_difference(x:T, y:T)", ["tf.squared_difference"]),
    TraceableFunction("tf.minimum(x:T, y:T)", ["tf.minimum"]),
    TraceableFunction("tf.maximum(x:T, y:T)", ["tf.maximum"]),
    TraceableFunction("tf.reciprocal(x:T)", ["tf.reciprocal", "_tf.reciprocal"]),
    TraceableFunction("tf.negative(x:T)", ["tf.negative", "_tf.neg"]),
    TraceableFunction("tf.logical_not(x:T)", ["tf.logical_not", "_tf.logical_not"]),
    TraceableFunction("tf.abs(x:T)", ["tf.abs", "_tf.abs"]),
    TraceableFunction("tf.sign(x:T)", ["tf.sign", "_tf.sign"]),
    TraceableFunction("tf.exp(x:T)", ["tf.exp", "_tf.exp"]),
    TraceableFunction("tf.log(x:T)", ["tf.log", "_tf.log"]),
    TraceableFunction("tf.sqrt(x:T)", ["tf.sqrt", "_tf.sqrt"]),
    TraceableFunction("tf.rsqrt(x:T)", ["tf.rsqrt", "_tf.rsqrt"]),
    TraceableFunction("tf.square(x:T)", ["tf.square", "_tf.square"]),
    TraceableFunction("tf.floor(x:T)", ["tf.floor", "_tf.floor"]),
    TraceableFunction("tf.ceil(x:T)", ["tf.ceil", "_tf.ceil"]),
    TraceableFunction("tf.round(x:T)", ["tf.round", "_tf.round"]),
    TraceableFunction("tf.sin(x:T)", ["tf.sin"]),
    TraceableFunction("tf.cos(x:T)", ["tf.cos"]),
    TraceableFunction("tf.sinh(x:T)", ["_tf_functions.sinh"]),
    TraceableFunction("tf.cosh(x:T)", ["_tf_functions.cosh"]),
    TraceableFunction("tf.nn.sigmoid(x:T)", ["tf.sigmoid", "tf.nn.sigmoid"]),
    TraceableFunction("tf.nn.tanh(x:T)", ["tf.tanh", "tf.nn.tanh"]),
    TraceableFunction("tf.where(condition:T, x:T, y:T)", ["tf.where"]),
    TraceableFunction("tf.reduce_sum(input_tensor:T, axis/reduction_indices[], keepdims/keep_dims)", ["tf.reduce_sum"]),
    TraceableFunction("tf.reduce_mean(input_tensor:T, axis/reduction_indices[], keepdims/keep_dims)",
                      ["tf.reduce_mean"]),
    TraceableFunction("tf.reduce_max(input_tensor:T, axis/reduction_indices[], keepdims/keep_dims)", ["tf.reduce_max"]),
    TraceableFunction("tf.reduce_min(input_tensor:T, axis/reduction_indices[], keepdims/keep_dims)", ["tf.reduce_min"]),
    TraceableFunction("tf.reduce_any(input_tensor:T, axis/reduction_indices[], keepdims/keep_dims)", ["tf.reduce_any"]),
    TraceableFunction("tf.reduce_all(input_tensor:T, axis/reduction_indices[], keepdims/keep_dims)", ["tf.reduce_all"]),
    TraceableFunction("tf.argmax(input:T, axis/dimension)", ["tf.argmax"]),
    TraceableFunction("tf.argmin(input:T, axis/dimension)", ["tf.argmin"]),
    TraceableFunction("tf.matmul(a:T, b:T, transpose_a, transpose_b, adjoint_a?, adjoint_b?)",
                      ["tf.matmul", "_tf.mat_mul"]),
    TraceableFunction("tf.add_n(inputs:T[])", ["tf.add_n"]),
    TraceableFunction("tf.nn.elu(features:T)", ["tf.nn.elu"]),
    TraceableFunction("tf.nn.relu(features:T)", ["tf.nn.relu"]),
    TraceableFunction("tf.nn.relu6(features:T)", ["tf.nn.relu6"]),
    TraceableFunction("tf.nn.softsign(features:T)", ["tf.nn.softsign"]),
    TraceableFunction("tf.nn.softplus(features:T)", ["tf.nn.softplus"]),
    TraceableFunction("tf.nn.leaky_relu(features:T, alpha:T)", ["_tf_functions.leaky_relu"]),
    TraceableFunction("tf.nn.conv1d(value:T, filters:T, stride[], padding, data_format?)", ["tf.nn.conv1d"]),
    TraceableFunction("tf.nn.conv2d(input:T, filter:T, strides, padding, data_format?, dilations?)", ["tf.nn.conv2d"]),
    TraceableFunction("tf.nn.conv3d(input:T, filter:T, strides, padding, data_format?, dilations?)", ["tf.nn.conv3d"]),
    TraceableFunction("tf.nn.convolution(input:T, filter:T, padding, strides, dilation_rate, data_format?)",
                      ["tf.nn.convolution"]),
    TraceableFunction("tf.nn.atrous_conv2d(value:T, filters:T, rate, padding)", ["tf.nn.atrous_conv2d"]),
    TraceableFunction("tf.nn.conv2d_transpose(value:T, filter:T, output_shape, strides, padding, data_format?)",
                      ["tf.nn.conv2d_transpose"]),
    TraceableFunction("tf.nn.conv3d_transpose(value:T, filter:T, output_shape, strides, padding, data_format?)",
                      ["tf.nn.conv3d_transpose"]),
    TraceableFunction("tf.nn.atrous_conv2d_transpose(value:T, filters:T, output_shape, rate, padding)",
                      ["tf.nn.atrous_conv2d_transpose"]),
    TraceableFunction("tf.nn.depthwise_conv2d(input:T, filter:T, strides, padding, rate, data_format?)",
                      ["tf.nn.depthwise_conv2d"]),
    TraceableFunction("tf.nn.depthwise_conv2d_native(input:T, filter:T, strides, padding, data_format?, dilations?)",
                      ["tf.nn.depthwise_conv2d_native"]),
    TraceableFunction("tf.nn.separable_conv2d(input:T, depthwise_filter:T, pointwise_filter:T, strides, padding, "
                      "rate, data_format?)", ["tf.nn.separable_conv2d"]),
    TraceableFunction("tf.nn.conv2d_backprop_input(input_sizes, filter:T, out_backprop:T, strides, padding, "
                      "data_format?, dilations?)", ["tf.nn.conv2d_backprop_input"]),
    TraceableFunction("tf.nn.depthwise_conv2d_native_backprop_input(input_sizes, filter:T, out_backprop:T, strides, "
                      "padding, data_format?, dilations?)", ["tf.nn.depthwise_conv2d_native_backprop_input"]),
    TraceableFunction("tf.nn.conv2d_backprop_filter(input:T, filter_sizes, out_backprop:T, strides, padding, "
                      "data_format?, dilations?)", ["tf.nn.conv2d_backprop_filter"]),
    TraceableFunction("tf.nn.conv3d_backprop_filter_v2(input:T, filter_sizes, out_backprop:T, strides, padding, "
                      "data_format?, dilations?)", ["tf.nn.conv3d_backprop_filter_v2"]),
    TraceableFunction("tf.nn.depthwise_conv2d_native_backprop_filter(input:T, filter_sizes, out_backprop:T, strides, "
                      "padding, data_format?, dilations?)", ["tf.nn.depthwise_conv2d_native_backprop_filter"]),
    TraceableFunction("tf.nn.max_pool(value:T, ksize, strides, padding, data_format?)", ["tf.nn.max_pool"]),
    TraceableFunction("tf.nn.avg_pool(value:T, ksize, strides, padding, data_format?)", ["tf.nn.avg_pool"]),
    TraceableFunction("tf.nn.max_pool_with_argmax(input:T, ksize, strides, padding, data_format?)",
                      ["tf.nn.max_pool_with_argmax"]),
    TraceableFunction("tf.nn.bias_add(value:T, bias:T, data_format?)", ["tf.nn.bias_add"]),
    TraceableFunction("tf.nn.lrn(input:T, depth_radius, bias, alpha, beta)", ["tf.nn.lrn"]),
    TraceableFunction("tf.nn.batch_normalization(x:T, mean:T, variance:T, offset:T, scale:T, variance_epsilon)",
                      ["tf.nn.batch_normalization"]),
    TraceableFunction("tf.nn.fused_batch_norm(x:T, scale:T, offset:T, mean:T?, variance:T?, epsilon, data_format?, "
                      "is_training)",
                      ["tf.nn.fused_batch_norm", "_tf.fused_batch_norm", "_tf.fused_batch_norm_v2"]),
    TraceableFunction("tf.nn.l2_normalize(x:T, axis/dim[], epsilon)", ["tf.nn.l2_normalize"]),
    TraceableFunction("tf.nn.softmax(logits:T, axis/dim?)",
                      ["tf.nn.softmax", "tf.contrib.layers.softmax", "_tf.softmax"]),
    TraceableFunction("tf.nn.moments(x:T, axes[], keep_dims)", ["tf.nn.moments"]),
    TraceableFunction("tf.image.resize_images(images:T, size, method, align_corners)", ["tf.image.resize_images"]),
    TraceableFunction("tf.image.resize_bilinear(images:T, size, align_corners)", ["tf.image.resize_bilinear"]),
    TraceableFunction("tf.image.resize_nearest_neighbor(images:T, size, align_corners)",
                      ["tf.image.resize_nearest_neighbor"]),
    TraceableFunction("tf.image.resize_bicubic(images:T, size, align_corners)", ["tf.image.resize_bicubic"]),
    TraceableFunction("tf.image.resize_area(images:T, size, align_corners)", ["tf.image.resize_area"]),
    TraceableFunction("tf.layers.flatten(inputs:T)", ["tf.layers.flatten", "tf.contrib.layers.flatten"]),
    TraceableFunction("tf.clip_by_value(t:T, clip_value_min:T, clip_value_max:T)", ["tf.clip_by_value"]),
    TraceableFunction("tf.slice(input_:T, begin, size)", ["tf.slice"]),
    TraceableFunction("tf.strided_slice(input_:T, begin, end, strides, begin_mask, end_mask, "
                      "ellipsis_mask, new_axis_mask, shrink_axis_mask, var)", ["tf.strided_slice"]),
    TraceableFunction("tf.stack(values:T[], axis)", ["tf.stack"]),
    TraceableFunction("tf.unstack(value:T, num, axis)", ["tf.unstack"]),
    TraceableFunction("tf.identity(input:T)", ["tf.identity"]),
    TraceableFunction("tf.stop_gradient(input:T)", ["tf.stop_gradient"]),
    TraceableFunction("tf.cast(x:T, dtype)", ["tf.cast"]),
    TraceableFunction("tf.nn.dropout(x:T, keep_prob)", ["tf.nn.dropout"]),
    TraceableFunction("tf.space_to_batch(input:T, paddings, block_size)", ["tf.space_to_batch"]),
    TraceableFunction("tf.space_to_batch_nd(input:T, block_shape, paddings)", ["tf.space_to_batch_nd"]),
    TraceableFunction("tf.batch_to_space(input:T, crops, block_size)", ["tf.batch_to_space"]),
    TraceableFunction("tf.batch_to_space_nd(input:T, block_shape, crops)", ["tf.batch_to_space_nd"]),
    TraceableFunction("tf.zeros(shape, dtype)", ["tf.zeros"]),
    TraceableFunction("tf.ones(shape, dtype)", ["tf.ones"]),
    TraceableFunction("tf.zeros_like(tensor:T, dtype)", ["tf.zeros_like"]),
    TraceableFunction("tf.ones_like(tensor:T, dtype)", ["tf.ones_like"]),
    TraceableFunction("tf.tile(input:T, multiples)", ["tf.tile"]),
    TraceableFunction("tf.dynamic_stitch(indices, data)", ["tf.dynamic_stitch"]),
    TraceableFunction("tf.range(start, limit, delta, dtype)", ["tf.range", "_tf.range"]),
    TraceableFunction("tf.rank(input:T)", ["tf.rank", "_tf.rank"]),
    TraceableFunction("tf.shape(input:T)", ["tf.shape"]),
    TraceableFunction("tf.shape_n(input:T[])", ["tf.shape_n"]),
    TraceableFunction("tf.invert_permutation(x:T)", ["tf.invert_permutation"]),
    TraceableFunction("tf.fill(dims, value)", ["tf.fill"]),
    TraceableFunction("tf.random_uniform(shape, minval, maxval, dtype, seed)", ["tf.random_uniform"]),
    TraceableFunction("_tf.conv3d_backprop_input_v2(input_sizes, filter:T, out_backprop:T, strides, padding, "
                      "data_format?, dilations?)", ["_tf.conv3d_backprop_input_v2"]),
    TraceableFunction("_tf.concat_offset(concat_dim, shape)", ["_tf.concat_offset"]),
    TraceableFunction("_tf.broadcast_gradient_args(s0, s1)", ["_tf.broadcast_gradient_args"]),
    TraceableFunction("_tf.sqrt_grad(y:T, dy:T)", ["_tf.sqrt_grad"]),
    TraceableFunction("_tf.rsqrt_grad(y:T, dy:T)", ["_tf.rsqrt_grad"]),
    TraceableFunction("_tf.sigmoid_grad(y:T, dy:T)", ["_tf.sigmoid_grad"]),
    TraceableFunction("_tf.tanh_grad(y:T, dy:T)", ["_tf.tanh_grad"]),
    TraceableFunction("_tf.reciprocal_grad(y:T, dy:T)", ["_tf.reciprocal_grad"]),
    TraceableFunction("_tf.strided_slice_grad(shape, begin, end, strides, dy:T, begin_mask, end_mask, "
                      "ellipsis_mask, new_axis_mask, shrink_axis_mask)", ["_tf.strided_slice_grad"]),
    TraceableFunction("_tf.bias_add_grad(out_backprop:T, data_format?)", ["_tf.bias_add_grad"]),
    TraceableFunction("_tf.fused_batch_norm_grad(y_backprop:T, x:T, scale:T, reserve_space_1:T?, reserve_space_2:T?, "
                      "epsilon, data_format?, is_training)",
                      ["_tf.fused_batch_norm_grad", "_tf.fused_batch_norm_grad_v2"]),
    TraceableFunction("_tf.max_pool_grad(orig_input:T, orig_output:T, grad:T, ksize, strides, padding, data_format?)",
                      ["_tf.max_pool_grad"]),
    TraceableFunction("_tf.max_pool_grad_with_argmax(input:T, grad:T, argmax:T, ksize, strides, padding)",
                      ["_tf.max_pool_grad_with_argmax"]),
    TraceableFunction("_tf.avg_pool_grad(orig_input_shape, grad:T, ksize, strides, padding, data_format?)",
                      ["_tf.avg_pool_grad"]),
    TraceableFunction("_tf.elu_grad(gradients:T, outputs:T)", ["_tf.elu_grad"]),
    TraceableFunction("_tf.relu_grad(gradients:T, features:T)", ["_tf.relu_grad"]),
    TraceableFunction("_tf.relu6_grad(gradients:T, features:T)", ["_tf.relu6_grad"]),
    TraceableFunction("_tf.softplus_grad(gradients:T, features:T)", ["_tf.softplus_grad"]),
    TraceableFunction("_tf.lrn_grad(input_grads:T, input_image:T, output_image:T, depth_radius, bias, alpha, beta)",
                      ["_tf.lrn_grad"]),
    TraceableFunction("_tf.TransposeGrad(orig_input/_orig_input_0:T, orig_perm/_orig_input_1, "
                      "orig_output/_orig_output_0:T, grad:T)", ["_tf.TransposeGrad"]),
    TraceableFunction("_tf.MinOrMaxGrad(orig_input/_orig_input_0:T, orig_axis/_orig_input_1[], "
                      "orig_output/_orig_output_0:T, grad:T)", ["_tf.MinOrMaxGrad"]),
    TraceableFunction("_tf.resize_nearest_neighbor_grad(grads:T, size, align_corners)",
                      ["_tf.resize_nearest_neighbor_grad"]),
    TraceableFunction("_tf.resize_bilinear_grad(grads:T, original_image:T, align_corners)",
                      ["_tf.resize_bilinear_grad"]),
    TraceableFunction("_tf.resize_bicubic_grad(grads:T, original_image:T, align_corners)",
                      ["_tf.resize_bicubic_grad"]),
    TraceableFunction("_tf.mirror_pad_grad(input:T, paddings, mode)", ["_tf.mirror_pad_grad"]),
]

__all__ = [
    "ArgProto",
    "OpProto",
    "parse_op_proto",
    "parse_arg_proto",
    "TraceableFunction",
    "DefaultTraceableFunctions",
    "args_from_tfop"
]
