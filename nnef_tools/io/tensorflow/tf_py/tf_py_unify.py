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

from nnef_tools.io.tensorflow.tf_py.tf_py_definitions import *


class OpUnifier(object):
    def __init__(self, s):
        # type: (str)->None
        op_names, proto = s.split('(', 1)
        self.op_names = [n.strip() for n in op_names.split('/')]
        self.op_proto = parse_op_proto(self.op_names[0] + '(' + proto)


def _get_arg(op_name, args, arg_proto):
    # type: (str, typing.Dict[str, typing.Any], ArgProto)->typing.Any
    found = True
    value = None
    for arg_name in arg_proto.arg_names:
        if arg_name in args:
            found = True
            if args[arg_name] is not None:
                assert value is None
                if arg_proto.is_array and args[arg_name] is not None:
                    value = utils.listify(args[arg_name])
                else:
                    value = args[arg_name]
    if found:
        return value
    if arg_proto.is_optional:
        return None
    assert False, "Arg '{}' not found for op '{}'".format(arg_proto.primary_arg_name, op_name)


def unify_ops(g):
    # type: (TFGraph)->None
    op_proto_by_name = {trf.op_proto.op_name: trf.op_proto for trf in DefaultTraceableFunctions}

    unifier_by_name = {}
    for unifier in _DefaultOpUnifiers:
        for name in unifier.op_names:
            unifier_by_name[name] = unifier

    for op in list(g.operations):
        if op.name in unifier_by_name:
            unifier = unifier_by_name[op.name]
            args = args_from_tfop(op, op_proto_by_name[op.name])
            is_list = False
            inputs = []
            attribs = {}
            for arg_proto in unifier.op_proto.arg_protos:
                if arg_proto.is_tensor and arg_proto.is_array:
                    inputs += _get_arg(op.name, args, arg_proto)
                    is_list = True
                elif arg_proto.is_tensor and not arg_proto.is_array:
                    inputs.append(_get_arg(op.name, args, arg_proto))
                else:
                    attribs[arg_proto.primary_arg_name] = _get_arg(op.name, args, arg_proto)
            inputs = [i for i in inputs if i is not None]
            inputs = inputs if is_list else tuple(inputs)
            old_outputs = op.outputs if isinstance(op.outputs, tuple) else list(op.outputs)
            op.inputs = []
            op.outputs = []
            g.remove_operation(op)
            TFOperation(graph=g, name=unifier.op_proto.op_name, inputs=inputs, outputs=old_outputs, attribs=attribs)


_DefaultOpUnifiers = [
    OpUnifier("_conv/tf.nn.convolution/tf.nn.conv1d/tf.nn.conv2d/tf.nn.conv3d/tf.nn.atrous_conv2d("
              "input/value/out_backprop:T, filter/filters:T, bias:T?, padding, stride/strides[]?,"
              "dilation/dilations/dilation_rate/rate[]?, data_format?)"),
    OpUnifier("_planewise_conv/tf.nn.depthwise_conv2d/tf.nn.depthwise_conv2d_native("
              "input/value/out_backprop:T, filter/filters:T, bias:T?, padding, stride/strides[]?,"
              "dilation/dilations/dilation_rate/rate[]?, data_format?)"),
    OpUnifier("_separable_conv/tf.nn.separable_conv2d("
              "input/value/out_backprop:T, plane_filter/depthwise_filter:T, point_filter/pointwise_filter:T, bias?,"
              "padding, stride/strides[]?, dilation/dilations/dilation_rate/rate[]?, data_format?)"),
    OpUnifier("_deconv/tf.nn.conv2d_transpose/tf.nn.conv3d_transpose/tf.nn.atrous_conv2d_transpose"
              "/tf.nn.conv2d_backprop_input/_tf.conv3d_backprop_input_v2("
              "input/value/out_backprop:T, filter/filters:T, bias:T?, padding, stride/strides[]?,"
              "dilation/dilations/dilation_rate/rate[]?, data_format?, output_shape/input_sizes?)"),
    OpUnifier("_planewise_deconv/tf.nn.depthwise_conv2d_native_backprop_input("
              "input/value/out_backprop:T, filter/filters:T, bias:T?, padding, stride/strides[]?,"
              "dilation/dilations/dilation_rate/rate[]?, data_format?, output_shape/input_sizes?)"),
    OpUnifier("_conv_grad_filter/tf.nn.conv2d_backprop_filter/tf.nn.conv3d_backprop_filter_v2("
              "orig_input/input:T, output_grad/out_backprop:T, padding, stride/strides[]?,"
              "dilation/dilations/dilation_rate/rate[]?, data_format?, output_shape/filter_sizes?)"),
    OpUnifier("_planewise_conv_grad_filter/tf.nn.depthwise_conv2d_native_backprop_filter("
              "orig_input/input:T, output_grad/out_backprop:T, padding, stride/strides[]?,"
              "dilation/dilations/dilation_rate/rate[]?, data_format?, output_shape/filter_sizes?)"),
    OpUnifier("_max_pool/tf.nn.max_pool(input/value:T, size/ksize, stride/strides, padding, data_format?)"),
    OpUnifier("_avg_pool/tf.nn.avg_pool(input/value:T, size/ksize, stride/strides, padding, data_format?)"),
    OpUnifier("_max_pool_with_index/tf.nn.max_pool_with_argmax(input/value:T, size/ksize, stride/strides, padding,"
              "data_format?)"),
    OpUnifier("_max_pool_grad/_tf.max_pool_grad(orig_input:T, orig_output:T, output_grad/grad:T, size/ksize, padding,"
              "stride/strides?, data_format?)"),
    OpUnifier("_max_pool_grad_with_index/_tf.max_pool_grad_with_argmax(orig_input/input:T, orig_index/argmax:T,"
              "output_grad/grad:T, size/ksize, padding, stride/strides?, data_format?)"),
    OpUnifier("_avg_pool_grad/_tf.avg_pool_grad(output_grad/grad:T, orig_input_shape, size/ksize, padding,"
              " stride/strides?, data_format?)")
]
