# Copyright (c) 2018 The Khronos Group Inc.
# Copyright (c) 2018 Au-Zone Technologies Inc.
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

import os
import sys
import unittest

import nnef
import numpy as np
import six
import tensorflow as tf

from nnef_tools.core import utils
from tests.activation.tf_pb_test_runner import test_tf_pb as test_activations

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


def save_protobuf(filename, output_node_names, sess, recreate):
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    output_graph = filename + "test.pb"

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.isfile(filename + "test.pbtxt") or recreate:
        tf.train.write_graph(input_graph_def, "", filename + "test.pbtxt")

    if not os.path.isfile(output_graph) or recreate:
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def,
                                                                        output_node_names.split(","))
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        tf.train.write_graph(output_graph_def, "", filename + "test.pbtxt")


def get_placeholders():
    return [t for op in tf.get_default_graph().get_operations() for t in op.values() if 'Placeholder' in op.node_def.op]


def run_test(output_name, output_nodes, recreate=True):
    batch_size = 1
    source_shapes = {ph.name: [int(d.value) if d.value is not None else batch_size for d in ph.shape.dims]
                     for ph in get_placeholders()}
    pb_path = os.path.join('out', 'pb', output_name)
    network_name = output_name.rstrip('/').replace('/', '_')
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        save_protobuf(pb_path, output_nodes, sess, recreate)
        sess.close()

    for prefer_nhwc in [True, False]:
        test_activations(filename=os.path.join(pb_path, 'test.pb'),
                         source_shapes=source_shapes,
                         feed_dict={utils.anystr_to_str(k): np.random.random(v)
                                    for k, v in six.iteritems(source_shapes)},
                         prefer_nhwc=prefer_nhwc,
                         network_name=network_name,
                         delete_after_each=False,
                         export_io_only=True)

    return nnef.parse_file(os.path.join('out', network_name + '_nhwc', 'nnef', network_name + '_nnef', 'graph.nnef'))


# Activations Testing
class TestActivations(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestActivations, self).__init__(*args, **kwargs)
        self.name = "testActivations/"

    # ActivationsSigmoid Test
    def sigmoid_network(self, x):
        z1 = tf.sigmoid(x, name="z1")
        return z1

    def test_sigmoid(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + "/"
        x = tf.placeholder(tf.float32, shape=[None, 1, 1, 1], name='x')
        self.sigmoid_network(x)

        run_test(output_name, "z1")

    # Activations_nn_Sigmoid Test
    def nn_sigmoid_network(self, x):
        z1 = tf.nn.sigmoid(x, name="z1")
        return z1

    def test_nn_sigmoid(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 1, 1, 1], name='x')
        self.sigmoid_network(x)

        run_test(output_name, "z1")

    # ActivationsTanh Test
    def tanh_network(self, x):
        z1 = tf.tanh(x, name="z1")
        return z1

    def test_tanh(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 1, 1, 1], name='x')
        self.tanh_network(x)

        run_test(output_name, "z1")

    # Activations_nn_Tanh Test
    def nn_tanh_network(self, x):
        z1 = tf.nn.tanh(x, name="z1")
        return z1

    def test_nn_tanh(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 1, 1, 1], name='x')
        self.nn_tanh_network(x)

        run_test(output_name, "z1")

    # Activations_nn_Elu Test
    def nn_elu_network(self, x):
        z1 = tf.nn.elu(x, name="z1")
        return z1

    def test_nn_elu(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 1, 1, 1], name='x')
        self.nn_elu_network(x)

        run_test(output_name, "z1")

    # Activations_nn_Relu Test
    def nn_relu_network(self, x):
        z1 = tf.nn.relu(x, name="z1")
        return z1

    def test_nn_relu(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 1, 1, 1], name='x')
        self.nn_relu_network(x)

        run_test(output_name, "z1")


class TestBasicMath(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestBasicMath, self).__init__(*args, **kwargs)
        self.name = "testBasicMath/"

    # BasicMathAdd Test
    def add_network(self, x):
        y = tf.constant([1, 2], dtype=tf.float32, name="y")
        z1 = tf.add(x, y, name="z1")
        return z1

    def test_add(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 2, 2, 2], name='x')
        self.add_network(x)

        run_test(output_name, "z1")

        # BasicMathSub Test

    def sub_network(self, x):
        y = tf.constant([-1], dtype=tf.float32, name="y")
        z1 = tf.subtract(x, y, name="z1")
        return z1

    def test_sub(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 2, 2, 2], name='x')
        self.sub_network(x)

        run_test(output_name, "z1")

        # BasicMathMult Test

    def multiply_network(self, x):
        y = tf.constant([-1, 2], dtype=tf.float32, name="y")
        z1 = tf.multiply(x, y, name="z1")
        return z1

    def test_multiply(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 2, 2, 2], name='x')
        self.multiply_network(x)

        run_test(output_name, "z1")

    # BasicMathDiv Test
    def div_network(self, x):
        y = tf.constant([[-1], [-2], [-3], [-4]], dtype=tf.float32, name="y")
        z1 = tf.div(x, y, name="z1")
        return z1

    def test_div(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 1], name='x')
        self.div_network(x)

        run_test(output_name, "z1")

    # BasicMathMatMul Test
    def matmul_network(self, x):
        y = tf.constant([[-1], [-2], [-3], [-4]], dtype=tf.float32, name="y")
        z1 = tf.matmul(x, y, transpose_b=True, name="z1")
        z2 = tf.matmul(x, y, transpose_a=True, name="z2")
        return z1, z2

    def test_matmul(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[4, 1], name='x')
        self.matmul_network(x)

        run_test(output_name, "z1,z2")

    # BasicMathNeg Test
    def neg_network(self, x):
        z1 = tf.negative(x, name="z1")
        return z1

    def test_neg(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 1], name='x')
        self.neg_network(x)

        run_test(output_name, "z1")

    # BasicMathBiasAdd Test
    def bias_add_network(self, x):
        z1 = tf.nn.bias_add(x, tf.constant([1.0]), name="z1")
        return z1

    def test_bias_add(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 1], name='x')
        self.bias_add_network(x)

        run_test(output_name, "z1")


class TestComparisons(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestComparisons, self).__init__(*args, **kwargs)
        self.name = "testComparisons/"

    # Comparisons_Greater Test
    def greater_network(self, x):
        y = tf.constant([[1], [2], [3], [4]], dtype=tf.float32, name="y")
        cond = tf.greater(x, y, name="cond")
        z1 = tf.where(cond, x, y, name="z1")
        return z1

    def test_greater(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[4, 1], name='x')
        self.greater_network(x)

        run_test(output_name, "z1")

    # Comparisons_Greater_Equal Test
    def greater_equal_network(self, x):
        y = tf.constant([[1], [2], [3], [4]], dtype=tf.float32, name="y")
        cond = tf.greater_equal(x, y, name="cond")
        z1 = tf.where(cond, x, y, name="z1")
        return z1

    def test_greater_equal(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[4, 1], name='x')
        self.greater_equal_network(x)

        run_test(output_name, "z1")

    # Comparisons_Less Test
    def less_network(self, x):
        y = tf.constant([[1], [2], [3], [4]], dtype=tf.float32, name="y")
        cond = tf.less(x, y, name="cond")
        z1 = tf.where(cond, x, y, name="z1")
        return z1

    def test_less(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[4, 1], name='x')
        self.less_network(x)

        run_test(output_name, "z1")

    # Comparisons_Less_Equal Test
    def less_equal_network(self, x):
        y = tf.constant([[1], [2], [3], [4]], dtype=tf.float32, name="y")
        cond = tf.less_equal(x, y, name="cond")
        z1 = tf.where(cond, x, y, name="z1")
        return z1

    def test_less_equal(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[4, 1], name='x')
        self.less_equal_network(x)

        run_test(output_name, "z1")

    # Comparisons_Equal Test
    def equal_network(self, x):
        y = tf.constant([[1], [2], [3], [4]], dtype=tf.float32, name="y")
        cond = tf.equal(x, y, name="cond")
        z1 = tf.where(cond, x, y, name="z1")
        return z1

    def test_equal(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[4, 1], name='x')
        self.equal_network(x)

        run_test(output_name, "z1")

    # Comparisons_Not_Equal Test
    def not_equal_network(self, x):
        y = tf.constant([[1], [2], [3], [4]], dtype=tf.float32, name="y")
        cond = tf.not_equal(x, y, name="cond")
        z1 = tf.where(cond, x, y, name="z1")
        return z1

    def test_not_equal(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[4, 1], name='x')
        self.not_equal_network(x)

        run_test(output_name, "z1")


class TestConvolutions(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestConvolutions, self).__init__(*args, **kwargs)
        self.name = "testConvolutions/"

    # Convolutions_nn_conv1d Test
    def nn_conv1d_network(self, x):
        y = tf.constant(np.random.rand(2, 1, 5) * 2 - 1, dtype=tf.float32, name="y")
        z1 = tf.nn.conv1d(x, y, 2, 'SAME', name="z1")
        return z1

    @unittest.skip("Reshape removes knowledge of data format")
    def test_nn_conv1d(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 1], name='x')
        self.nn_conv1d_network(x)

        run_test(output_name, "z1/Squeeze")

    # Convolutions_nn_conv2d Test
    def nn_conv2d_network(self, x):
        y = tf.constant(np.ones([3, 3, 1, 4]) * 2 - 1, dtype=tf.float32, name="y")
        z1 = tf.nn.conv2d(x, y, [1, 2, 2, 1], 'VALID', name="z1")
        z2 = tf.nn.conv2d(x, y, strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 2, 2, 1], name="z2")
        return z1, z2

    def test_nn_conv2d(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 5, 5, 1], name='x')
        self.nn_conv2d_network(x)

        run_test(output_name, "z1,z2")

    # Convolutions_nn_conv3d Test
    def nn_conv3d_network(self, x):
        y = tf.constant(np.ones([3, 3, 3, 1, 5]) * 2 - 1, dtype=tf.float32, name="y")
        z1 = tf.nn.conv3d(x, y, [1, 2, 2, 2, 1], 'VALID', name="z1")
        return z1

    def test_nn_conv3d(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 5, 5, 5, 1], name='x')
        self.nn_conv3d_network(x)

        run_test(output_name, "z1")

    # Convolutions_nn_convolution Test
    def nn_convolution_network(self, x1):
        x2 = tf.constant(np.ones([1, 5, 5, 3]), dtype=tf.float32)
        x3 = tf.constant(np.ones([1, 5, 3]), dtype=tf.float32)
        y1 = tf.constant(np.ones([3, 3, 3, 3, 6]) * 2 - 1, dtype=tf.float32, name="y1")
        y2 = tf.constant(np.ones([3, 3, 3, 6]) * 2 - 1, dtype=tf.float32, name="y2")
        y3 = tf.constant(np.ones([3, 3, 6]) * 2 - 1, dtype=tf.float32, name="y3")
        z1 = tf.nn.convolution(x1, y1, strides=[2, 2, 2], padding='VALID', name="z1")
        z2 = tf.nn.convolution(x2, y2, strides=[2, 2], padding='VALID', name="z2")
        z3 = tf.nn.convolution(x3, y3, strides=[2], padding='VALID', name="z3")
        return z1, z2, z3

    @unittest.skip("Recreates Conv1D, 2D, and 3D, issue with only a single input node, not conversion.")
    def test_nn_convolution(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 5, 5, 5, 3], name='x')
        self.nn_convolution_network(x)

        run_test(output_name, "z1,z2,z3/Squeeze", True)

    # Convolutions_nn_conv2d_transpose Test
    def nn_conv2d_transpose_network(self, x):
        y = tf.constant(np.ones([2, 2, 1, 3]) * 2 - 1, dtype=tf.float32, name="y")
        z1 = tf.nn.conv2d_transpose(x, y, output_shape=[1, 10, 10, 1], strides=[1, 2, 2, 1], padding='SAME', name="z1")
        return z1

    def test_nn_conv2d_transpose(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 5, 5, 3], name='x')
        self.nn_conv2d_transpose_network(x)

        run_test(output_name, "z1")

    # Convolutions_nn_conv3d_transpose Test
    def nn_conv3d_transpose_network(self, x):
        y = tf.constant(np.ones([2, 2, 2, 1, 3]) * 2 - 1, dtype=tf.float32, name="y")
        z1 = tf.nn.conv3d_transpose(x, y, output_shape=[1, 10, 10, 10, 1], strides=[1, 2, 2, 2, 1], padding='SAME',
                                    name="z1")
        return z1

    def test_nn_conv3d_transpose(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 5, 5, 5, 3], name='x')
        self.nn_conv3d_transpose_network(x)

        run_test(output_name, "z1")

    # Convolutions_nn_depthwise_conv2d Test
    def nn_depthwise_conv2d_network(self, x):
        y = tf.constant(np.ones([3, 3, 3, 5]) * 2 - 1, dtype=tf.float32, name="y")
        z1 = tf.nn.depthwise_conv2d(x, y, [1, 2, 2, 1], 'SAME', name="z1")
        return z1

    def test_nn_depthwise_conv2d(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 5, 5, 3], name='x')
        self.nn_depthwise_conv2d_network(x)

        run_test(output_name, "z1")

    # Convolutions_nn_separable_conv2d Test
    def nn_separable_conv2d_network(self, x):
        y = tf.constant(np.ones([3, 3, 1, 1]) * 2 - 1, dtype=tf.float32, name="y")
        c1 = tf.constant(np.ones([1, 1, 1, 1]), dtype=tf.float32, name="c1")
        z1 = tf.nn.separable_conv2d(x, y, c1, [1, 2, 2, 1], 'SAME', name="z1")
        return z1

    def test_nn_separable_conv2d(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 5, 5, 1], name='x')
        self.nn_separable_conv2d_network(x)

        run_test(output_name, "z1")


class TestLogicals(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestLogicals, self).__init__(*args, **kwargs)
        self.name = "testLogicals/"

    # Logicals_logical_and Test
    def logical_and_network(self, x):
        y = tf.constant([[1], [2], [3], [4]], dtype=tf.float32, name="y")
        log1 = tf.greater(x, y, name='log1')
        log2 = tf.less(x, y, name='log2')
        log_op = tf.logical_and(log1, log2, name="log_op")
        z1 = tf.where(log_op, x, y, name='z1')
        return z1

    def test_logical_and(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[4, 1], name='x')
        self.logical_and_network(x)

        run_test(output_name, "z1")

    # Logicals_logical_or Test
    def logical_or_network(self, x):
        y = tf.constant([[1], [2], [3], [4]], dtype=tf.float32, name="y")
        log1 = tf.greater(x, y, name='log1')
        log2 = tf.less(x, y, name='log2')
        log_op = tf.logical_or(log1, log2, name="log_op")
        z1 = tf.where(log_op, x, y, name='z1')
        return z1

    def test_logical_or(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[4, 1], name='x')
        self.logical_or_network(x)

        run_test(output_name, "z1")

    # Logicals_logical_not Test
    def logical_not_network(self, x):
        y = tf.constant([[1], [2], [3], [4]], dtype=tf.float32, name="y")
        log1 = tf.greater(x, y, name='log1')
        log_op = tf.logical_not(log1, name="log_op")
        z1 = tf.where(log_op, x, y, name='z1')
        return z1

    def test_logical_not(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[4, 1], name='x')
        self.logical_not_network(x)

        run_test(output_name, "z1")


class TestMathFunc(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestMathFunc, self).__init__(*args, **kwargs)
        self.name = "testMathFunc/"

    # TestMathFunc_pow Test
    def pow_network(self, x):
        y = tf.constant([[-1], [-2], [-3], [-4]], dtype=tf.float32, name="y")
        z1 = tf.pow(x, y, name="z1")
        return z1

    def test_pow(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.pow_network(x)

        run_test(output_name, "z1")

    # TestMathFunc_abs Test
    def abs_network(self, x):
        z1 = tf.abs(x, name="z1")
        return z1

    def test_abs(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.abs_network(x)

        run_test(output_name, "z1")

    # TestMathFunc_sign Test
    def sign_network(self, x):
        z1 = tf.sign(x, name="z1")
        return z1

    def test_sign(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.sign_network(x)

        run_test(output_name, "z1")

    # TestMathFunc_exp Test
    def exp_network(self, x):
        z1 = tf.exp(x, name="z1")
        return z1

    def test_exp(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.exp_network(x)

        run_test(output_name, "z1")

    # TestMathFunc_log Test
    def log_network(self, x):
        z1 = tf.log(x, name="z1")
        return z1

    def test_log(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.log_network(x)

        run_test(output_name, "z1")

    # TestMathFunc_sqrt Test
    def sqrt_network(self, x):
        z1 = tf.sqrt(x, name="z1")
        return z1

    def test_sqrt(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.sqrt_network(x)

        run_test(output_name, "z1")

    # TestMathFunc_square Test
    def square_network(self, x):
        z1 = tf.square(x, name="z1")
        return z1

    def test_square(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.square_network(x)

        run_test(output_name, "z1")

    # TestMathFunc_floor Test
    def floor_network(self, x):
        z1 = tf.floor(x, name="z1")
        return z1

    def test_floor(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.floor_network(x)

        run_test(output_name, "z1")

    # TestMathFunc_ceil Test
    def ceil_network(self, x):
        z1 = tf.ceil(x, name="z1")
        return z1

    def test_ceil(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.ceil_network(x)

        run_test(output_name, "z1")

    # TestMathFunc_round Test
    def round_network(self, x):
        z1 = tf.round(x, name="z1")
        return z1

    def test_round(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.round_network(x)

        run_test(output_name, "z1")

    # TestMathFunc_minimum Test
    def minimum_network(self, x):
        y = tf.constant([[-1], [-2], [-3], [-4]], dtype=tf.float32, name="y")
        z1 = tf.minimum(x, y, name="z1")
        return z1

    def test_minimum(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.minimum_network(x)

        run_test(output_name, "z1")

    # TestMathFunc_maximum Test
    def maximum_network(self, x):
        y = tf.constant([[-1], [-2], [-3], [-4]], dtype=tf.float32, name="y")
        z1 = tf.maximum(x, y, name="z1")
        return z1

    def test_maximum(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.maximum_network(x)

        run_test(output_name, "z1")

    # TestMathFunc_reduce_sum Test
    def reduce_sum_network(self, x):
        z1 = tf.reduce_sum(x, name="z1")
        z2 = tf.reduce_sum(x, axis=[1, 1, 3], name="z2")
        z3 = tf.reduce_sum(x, axis=[0, 1], keepdims=True, name="z3")
        z4 = tf.reduce_sum(x, axis=[3], name="z4")
        return z1, z2, z3, z4

    def test_reduce_sum(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 2], name='x')
        self.reduce_sum_network(x)

        run_test(output_name, "z1,z2,z3,z4")

    # TestMathFunc_reduce_mean Test
    def reduce_mean_network(self, x):
        z1 = tf.reduce_mean(x, name="z1")
        z2 = tf.reduce_mean(x, axis=[1, 1, 3], name="z2")
        z3 = tf.reduce_mean(x, axis=[0, 1], keepdims=True, name="z3")
        z4 = tf.reduce_mean(x, axis=[3], name="z4")
        return z1, z2, z3, z4

    def test_reduce_mean(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.reduce_mean_network(x)

        run_test(output_name, "z1,z2,z3,z4")

    # TestMathFunc_reduce_max Test
    def reduce_max_network(self, x):
        z1 = tf.reduce_max(x, name="z1")
        z2 = tf.reduce_max(x, axis=[1, 1, 3], name="z2")
        z3 = tf.reduce_max(x, axis=[0, 1], keepdims=True, name="z3")
        z4 = tf.reduce_max(x, axis=[3], name="z4")
        return z1, z2, z3, z4

    def test_reduce_max(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.reduce_max_network(x)

        run_test(output_name, "z1,z2,z3,z4")


class TestNormalization(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestNormalization, self).__init__(*args, **kwargs)
        self.name = "testNormalization/"

    # Normalization_nn_lrn Test
    def nn_lrn_network(self, x):
        z1 = tf.nn.lrn(x, name="z1")
        z2 = tf.nn.lrn(x, depth_radius=3, bias=2, alpha=2, beta=0.3, name="z2")
        return z1, z2

    def test_nn_lrn(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.nn_lrn_network(x)

        run_test(output_name, "z1,z2")

    # Normalization_nn_local_response_normalization Test
    def nn_local_response_normalization_network(self, x):
        z1 = tf.nn.local_response_normalization(x, name="z1")
        z2 = tf.nn.local_response_normalization(x, depth_radius=3, bias=2, alpha=2, beta=0.3, name="z2")
        return z1, z2

    def test_nn_local_response_normalization(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.nn_local_response_normalization_network(x)

        run_test(output_name, "z1,z2")

    # Normalization_nn_batch_normalization Test
    def nn_batch_normalization_network(self, x):
        y = tf.constant(np.ones([4]) * 10, dtype=tf.float32, name="y")
        z1 = tf.nn.batch_normalization(x, y, y, y, y, variance_epsilon=0.001, name="z1")
        return z1

    def test_nn_batch_normalization(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.nn_batch_normalization_network(x)

        run_test(output_name, "z1/add_1")

    # Normalization_nn_fused_batch_norm Test
    def nn_fused_batch_norm_network(self, x):
        y = tf.constant(np.ones([4]) * 10, dtype=tf.float32, name="y")
        z1 = tf.nn.fused_batch_norm(x, y, y, y, y, is_training=False, name="z1")
        return z1

    def test_nn_fused_batch_norm(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.nn_fused_batch_norm_network(x)

        run_test(output_name, "z1")

    # Normalization_nn_fused_batch_norm Test
    def nn_fused_batch_norm_network2(self, x):
        a = tf.constant(np.ones([4]) * 10, dtype=tf.float32, name="a")
        b = tf.constant(np.ones([4]) * 10, dtype=tf.float32, name="b")
        c = tf.constant(np.ones([4]) * 10, dtype=tf.float32, name="c")
        d = tf.constant(np.ones([4]) * 10, dtype=tf.float32, name="d")
        z1 = tf.nn.fused_batch_norm(x, a, b, c, d, is_training=False, name="z1")
        return z1

    def test_nn_fused_batch_norm2(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.nn_fused_batch_norm_network2(x)

        run_test(output_name, "z1")

    # Normalization_nn_l2_normalize Test
    def nn_l2_normalize_network(self, x):
        z1 = tf.nn.l2_normalize(x, name="z1")
        return z1

    def test_nn_l2_normalize(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.nn_l2_normalize_network(x)

        run_test(output_name, "z1")


class TestOther(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestOther, self).__init__(*args, **kwargs)
        self.name = "testOther/"

    # Other_image_resize_images Test
    def image_resize_images_network(self, x):
        z1 = tf.image.resize_images(x, [2, 2], method=tf.image.ResizeMethod.AREA)
        z2 = tf.image.resize_images(x, [8, 8])
        z3 = tf.image.resize_images(x, [2, 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        z4 = tf.image.resize_images(x, [8, 8], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        z1 = tf.nn.relu(z1, name='z1')
        z2 = tf.nn.relu(z2, name='z2')
        z3 = tf.nn.relu(z3, name='z3')
        z4 = tf.nn.relu(z4, name='z4')
        return z1, z2, z3, z4

    def test_image_resize_images(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[1, 4, 4, 1], name='x')
        self.image_resize_images_network(x)

        run_test(output_name, "z1,z2,z3,z4")

    def squeeze_network(self, x):
        z1 = tf.squeeze(x, name="z1")
        return z1

    def test_squeeze(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[3, 1, 5, 1, 2, 1], name='x')
        self.squeeze_network(x)

        run_test(output_name, "z1")


class TestPooling(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestPooling, self).__init__(*args, **kwargs)
        self.name = "testPooling/"

    # Pooling_nn_max_pool Test
    def nn_max_pool_network(self, x):
        z1 = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 3, 3, 1], 'VALID', name="z1")
        z2 = tf.nn.max_pool(x, [1, 1, 1, 1], [1, 1, 1, 1], 'SAME', name="z2")
        return z1, z2

    def test_nn_max_pool(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.nn_max_pool_network(x)

        run_test(output_name, "z1,z2")

    # Pooling_nn_max_pool_with_argmax Test
    def nn_max_pool_with_argmax_network(self, x):
        z1 = tf.nn.max_pool_with_argmax(x, [1, 2, 2, 1], [1, 3, 3, 1], 'SAME', name="z1")
        z2 = tf.nn.max_pool_with_argmax(x, [1, 1, 1, 1], [1, 1, 1, 1], 'VALID', name="z2")
        return z1, z2

    def test_nn_max_pool_with_argmax(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.nn_max_pool_with_argmax_network(x)

        run_test(output_name, "z1,z2")

    # Pooling_nn_avg_pool Test
    def nn_avg_pool_network(self, x):
        z1 = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 3, 3, 1], 'SAME', name="z1")
        z2 = tf.nn.avg_pool(x, [1, 1, 1, 1], [1, 1, 1, 1], 'VALID', name="z2")
        return z1, z2

    def test_nn_avg_pool(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.nn_avg_pool_network(x)

        run_test(output_name, "z1,z2")


class TestSofts(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSofts, self).__init__(*args, **kwargs)
        self.name = "testSofts/"

    # Softs_nn_softsign Test
    def nn_softsign_network(self, x):
        z1 = tf.nn.softsign(x, name="z1")
        return z1

    @unittest.skip("softsign doesn't exist within NNEF currently")
    def test_nn_softsign(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.nn_softsign_network(x)

        run_test(output_name, "z1")

    # Softs_nn_softplus Test
    def nn_softplus_network(self, x):
        z1 = tf.nn.softplus(x, name="z1")
        return z1

    def test_nn_softplus(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 4], name='x')
        self.nn_softplus_network(x)

        run_test(output_name, "z1")

    # Softs_nn_softmax Test
    def nn_softmax_network(self, x):
        z1 = tf.nn.softmax(x, name="z1")
        return z1

    def test_nn_softmax(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[1, 4], name='x')
        self.nn_softmax_network(x)

        run_test(output_name, "z1")


class TestVarOps(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestVarOps, self).__init__(*args, **kwargs)
        self.name = "testVarOps/"

    # VarOps_concat Test
    def concat_network(self, x):
        y = tf.constant([[-1], [-2], [-3], [-4]], dtype=tf.float32, name="y")
        z1 = tf.concat([x, y], 0, name="z1")
        return z1

    def test_concat(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[4, 1], name='x')
        self.concat_network(x)

        run_test(output_name, "z1")

    # VarOps_split Test
    def split_network(self, x):
        a, b = tf.split(x, 2, name="splitting")
        z1 = tf.add(a, b, 'z1')
        return z1

    def test_split(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[4, 1], name='x')
        self.split_network(x)

        run_test(output_name, "z1")

    # VarOps_reshape Test
    def reshape_network(self, x):
        z1 = tf.reshape(x, [8, 4], name="z1")
        return z1

    def test_reshape(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[1, 4, 4, 2], name='x')
        self.reshape_network(x)

        run_test(output_name, "z1")

    # VarOps_transpose Test
    def transpose_network(self, x):
        z1 = tf.transpose(x, [1, 3, 2, 0], name="z1")
        return z1

    def test_transpose(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[1, 4, 4, 2], name='x')
        self.transpose_network(x)

        run_test(output_name, "z1")

    # Variable_where Test
    def where_network(self, x):
        y = tf.constant([[1], [1], [1], [1]], dtype=tf.float32, name='y')
        cond = tf.greater_equal(x, y, name='cond')
        z1 = tf.where(cond, x, y, name="z1")
        return z1

    def test_where(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[4, 1], name='x')
        self.where_network(x)

        run_test(output_name, "z1")


class TestVariables(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestVariables, self).__init__(*args, **kwargs)
        self.name = "testVariables/"

    # Variable_where Test
    def assign_network(self, x):
        t1 = tf.Variable(initial_value=[0.0, 0.0, 0.0, 0.0], name='t1')
        c1 = tf.constant([10.0, 20.0, 30.0, 40.0], dtype=tf.float32, name="c1")
        t1.assign(c1)
        z1 = tf.add(x, t1, name='z1')
        return z1

    @unittest.skip("Frozen graph removes all assign nodes.")
    def test_assign(self):
        tf.reset_default_graph()
        output_name = self.name + sys._getframe().f_code.co_name[5:] + '/'
        x = tf.placeholder(tf.float32, shape=[4], name='x')
        self.assign_network(x)

        run_test(output_name, "z1")


if __name__ == '__main__':
    unittest.main()
