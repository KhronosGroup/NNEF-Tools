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

import os
import shutil
import unittest

import tensorflow as tf

from tests.conversion.tflite_test_runner import TFLiteTestRunner

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


class TFLiteLayerTestCases(TFLiteTestRunner):

    @classmethod
    def setUpClass(cls):
        if os.path.exists("out"):
            shutil.rmtree("out")

    @staticmethod
    def to_tflite(fun):
        tf.reset_default_graph()
        input, output = fun()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [input], [output])
            tflite_model = converter.convert()
            filename = "out/orig/{}.tflite".format(fun.__name__)
            dir = os.path.dirname(filename)
            if not os.path.exists(dir):
                os.makedirs(dir)
            with open(filename, "wb") as f:
                f.write(tflite_model)
            return filename

    @staticmethod
    def conv_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

        return input_, tf.nn.conv2d(input_, filter_, [1, 1, 1, 1], 'VALID')

    def test_conv(self):
        self._test_model(self.to_tflite(self.conv_network))

    @staticmethod
    def conv_relu_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")
        conv = tf.nn.conv2d(input_, filter_, [1, 1, 1, 1], 'VALID')
        relu = tf.nn.relu(conv)
        return input_, relu

    def test_conv_relu(self):
        self._test_model(self.to_tflite(self.conv_relu_network))

    @staticmethod
    def depthwise_conv_relu_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 2], name="filter")
        conv = tf.nn.depthwise_conv2d(input_, filter_, [1, 1, 1, 1], 'VALID')
        relu = tf.nn.relu(conv)
        return input_, relu

    def test_depthwise_conv_relu(self):
        self._test_model(self.to_tflite(self.depthwise_conv_relu_network))

    @staticmethod
    def max_pool_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        return input_, tf.nn.relu(tf.nn.max_pool(input_, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID'))

    def test_max_pool(self):
        self._test_model(self.to_tflite(self.max_pool_network))

    @staticmethod
    def avg_pool_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        return input_, tf.nn.relu(tf.nn.avg_pool(input_, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID'))

    def test_avg_pool(self):
        self._test_model(self.to_tflite(self.avg_pool_network))

    @staticmethod
    def concat_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        return input_, tf.concat([input_, input_ + input_], axis=2)

    def test_concat(self):
        self._test_model(self.to_tflite(self.concat_network))

    @staticmethod
    def softmax_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        return input_, tf.nn.softmax(input_)

    def test_softmax(self):
        self._test_model(self.to_tflite(self.softmax_network))

    @staticmethod
    def network1():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")
        conv = tf.nn.conv2d(input_, filter_, [1, 1, 1, 1], 'VALID')
        bias = tf.constant(dtype=tf.float32, shape=[16], value=4.5, name="bias")
        bias_add = tf.nn.bias_add(conv, bias)
        relu = tf.nn.relu(bias_add)
        max = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        filter2_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 16, 16], name="filter2")
        conv2 = tf.nn.conv2d(max, filter2_, [1, 1, 1, 1], 'VALID')
        bias_add2 = tf.nn.bias_add(conv2, bias)
        relu2 = tf.nn.relu(bias_add2)

        return input_, relu2

    def test_network1(self):
        self._test_model(self.to_tflite(self.network1))

    @staticmethod
    def abs_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        return input_, tf.abs(input_)

    @unittest.skip('Not yet supported in tflite')
    def test_abs(self):
        self._test_model(self.to_tflite(self.abs_network))

    @staticmethod
    def add_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        return input_, tf.nn.relu(tf.add(input_, input_))

    def test_add(self):
        self._test_model(self.to_tflite(self.add_network))

    @staticmethod
    def subtract_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        return input_, tf.nn.relu(tf.subtract(input_, input_))

    def test_subtract(self):
        self._test_model(self.to_tflite(self.subtract_network))

    @staticmethod
    def multiply_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        return input_, tf.nn.relu(tf.multiply(input_, input_))

    def test_multiply(self):
        self._test_model(self.to_tflite(self.multiply_network))

    @staticmethod
    def add_n_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        return input_, tf.add_n([input_, input_, input_])

    @unittest.skip('Not yet supported in tflite')
    def test_add_n(self):
        self._test_model(self.to_tflite(self.add_n_network))

    @staticmethod
    def argmax_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        output_ = tf.argmax(input_, axis=3)
        return input_, output_

    def test_argmax(self):
        self._test_model(self.to_tflite(self.argmax_network))

    @staticmethod
    def argmin_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
        return input_, tf.argmin(input_, axis=3)

    def test_argmin(self):
        self._test_model(self.to_tflite(self.argmin_network))

    @staticmethod
    def deconv_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 16], name="input")
        filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

        return input_, tf.nn.relu(tf.nn.conv2d_transpose(value=input_,
                                                         filter=filter_,
                                                         output_shape=[1, 64, 64, 3],
                                                         strides=[1, 1, 1, 1],
                                                         padding='SAME'))

    def test_deconv(self):
        self._test_model(self.to_tflite(self.deconv_network))

    @staticmethod
    def elu_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.nn.elu(input_)

    @unittest.skip('Not yet supported in tflite')
    def test_elu(self):
        self._test_model(self.to_tflite(self.elu_network))

    @staticmethod
    def equal_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.equal(input_, input_)

    def test_equal(self):
        self._test_model(self.to_tflite(self.equal_network))

    @staticmethod
    def exp_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.exp(input_)

    def test_exp(self):
        self._test_model(self.to_tflite(self.exp_network), max_val=1.0)

    @staticmethod
    def fill_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.fill([1, 2, 2, 3], 0.6) + input_

    def test_fill(self):
        self._test_model(self.to_tflite(self.fill_network))

    @staticmethod
    def floor_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.floor(input_)

    def test_floor(self):
        self._test_model(self.to_tflite(self.floor_network))

    @staticmethod
    def ceil_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.ceil(input_)

    @unittest.skip('Not yet supported in tflite')
    def test_ceil(self):
        self._test_model(self.to_tflite(self.ceil_network))

    @staticmethod
    def greater_equal_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.greater_equal(input_, input_)

    def test_greater_equal(self):
        self._test_model(self.to_tflite(self.greater_equal_network))

    @staticmethod
    def greater_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.greater(input_, input_)

    def test_greater(self):
        self._test_model(self.to_tflite(self.greater_network))

    @staticmethod
    def l2_normalize_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.nn.relu(tf.nn.l2_normalize(input_, axis=-1))

    def test_l2_normalize(self):
        self._test_model(self.to_tflite(self.l2_normalize_network))

    @staticmethod
    def matmul_network():
        input_ = tf.placeholder(tf.float32, shape=[2, 5], name="input")
        filter_ = tf.get_variable(dtype=tf.float32, shape=[5, 6], name="filter")

        return input_, tf.matmul(input_, filter_) + tf.nn.relu(tf.matmul(input_, filter_))

    def test_matmul(self):
        self._test_model(self.to_tflite(self.matmul_network))

    @staticmethod
    def leaky_relu_network():
        input_ = tf.placeholder(tf.float32, shape=[2, 5], name="input")

        return input_, tf.nn.leaky_relu(input_, alpha=0.3)

    def test_leaky_relu(self):
        self._test_model(self.to_tflite(self.leaky_relu_network))

    @staticmethod
    def less_equal_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.less_equal(input_, input_)

    def test_less_equal(self):
        self._test_model(self.to_tflite(self.less_equal_network))

    @staticmethod
    def less_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.less(input_, input_)

    def test_less(self):
        self._test_model(self.to_tflite(self.less_network))

    @staticmethod
    def lrn_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.nn.lrn(input_)

    def test_lrn(self):
        self._test_model(self.to_tflite(self.lrn_network))

    @staticmethod
    def logical_or_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.logical_or(input_ < 128, input_ > 200)

    def test_logical_or(self):
        self._test_model(self.to_tflite(self.logical_or_network))

    @staticmethod
    def logical_and_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.logical_and(input_ < 128, input_ > 200)

    def test_logical_and(self):
        self._test_model(self.to_tflite(self.logical_and_network))

    @staticmethod
    def logical_not_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.logical_not(input_ < 128)

    def test_logical_not(self):
        self._test_model(self.to_tflite(self.logical_not_network))

    @staticmethod
    def log_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.log(input_)

    def test_log(self):
        self._test_model(self.to_tflite(self.log_network), max_val=1.0)

    @staticmethod
    def negative_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.negative(input_)

    def test_negative(self):
        self._test_model(self.to_tflite(self.negative_network), max_val=1.0)

    @staticmethod
    def maximum_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.maximum(input_, input_)

    def test_maximum(self):
        self._test_model(self.to_tflite(self.maximum_network))

    @staticmethod
    def minimum_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.minimum(input_, input_)

    def test_minimum(self):
        self._test_model(self.to_tflite(self.minimum_network))

    @staticmethod
    def divide_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.nn.relu(tf.divide(input_, input_))

    def test_divide(self):
        self._test_model(self.to_tflite(self.divide_network))

    @staticmethod
    def expand_dims_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.expand_dims(input_, axis=0)

    def test_expand_dims(self):
        self._test_model(self.to_tflite(self.expand_dims_network))

    @staticmethod
    def not_equal_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.not_equal(input_, input_)

    def test_not_equal(self):
        self._test_model(self.to_tflite(self.not_equal_network))

    @staticmethod
    def stack_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2], name="input")
        return input_, tf.stack([input_, input_], axis=1)

    def test_stack(self):
        self._test_model(self.to_tflite(self.stack_network))

    @staticmethod
    def unstack_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2], name="input")
        return input_, tf.unstack(input_, axis=1)[0]

    def test_unstack(self):
        self._test_model(self.to_tflite(self.unstack_network))

    @staticmethod
    def pow_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.pow(input_, input_)

    def test_pow(self):
        self._test_model(self.to_tflite(self.pow_network), max_val=1.0)

    @staticmethod
    def mean_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.reduce_mean(input_, axis=1)

    def test_mean(self):
        self._test_model(self.to_tflite(self.mean_network))

    @staticmethod
    def sum_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.reduce_sum(input_, axis=1)

    def test_sum(self):
        self._test_model(self.to_tflite(self.sum_network))

    @staticmethod
    def max_reduce_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.reduce_max(input_, axis=1)

    def test_max_reduce(self):
        self._test_model(self.to_tflite(self.max_reduce_network))

    @staticmethod
    def min_reduce_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.reduce_min(input_, axis=1)

    def test_min_reduce(self):
        self._test_model(self.to_tflite(self.min_reduce_network))

    @staticmethod
    def relu6_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.nn.relu6(input_)

    def test_relu6(self):
        self._test_model(self.to_tflite(self.relu6_network))

    @staticmethod
    def resize_nearest_neighbor_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.image.resize_nearest_neighbor(input_, (4, 4), align_corners=False)

    @unittest.skip('Not yet supported in tflite')
    def test_resize_nearest_neighbor(self):
        self._test_model(self.to_tflite(self.resize_nearest_neighbor_network))

    @staticmethod
    def where_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.where(input_ < 128, input_, -input_)

    def test_where(self):
        self._test_model(self.to_tflite(self.where_network))

    @staticmethod
    def slice_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.slice(input_, [0, 1, 1, 0], [1, 1, 1, 2])

    def test_slice(self):
        self._test_model(self.to_tflite(self.slice_network))

    @staticmethod
    def split_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 10, 3], name="input")
        return input_, tf.split(input_, 5, 2)[2]

    def test_split(self):
        self._test_model(self.to_tflite(self.split_network))

    @staticmethod
    def split_v_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 10, 3], name="input")
        return input_, tf.split(input_, [5, 3, 1, 1], 2)[2]

    @unittest.skip('Not yet supported in tflite')
    def test_split_v(self):
        self._test_model(self.to_tflite(self.split_v_network))

    @staticmethod
    def sqrt_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.sqrt(input_)

    def test_sqrt(self):
        self._test_model(self.to_tflite(self.sqrt_network))

    @staticmethod
    def square_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.square(input_)

    def test_square(self):
        self._test_model(self.to_tflite(self.square_network))

    @staticmethod
    def transpose_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.transpose(input_, perm=[0, 3, 1, 2])

    def test_transpose(self):
        self._test_model(self.to_tflite(self.transpose_network))

    @staticmethod
    def zeros_like_network():
        input_ = tf.placeholder(tf.float32, shape=[1, 2, 2, 3], name="input")
        return input_, tf.zeros_like(input_) + input_

    @unittest.skip('Not yet supported in tflite')
    def test_zeros_like(self):
        self._test_model(self.to_tflite(self.zeros_like_network))


if __name__ == '__main__':
    unittest.main()
