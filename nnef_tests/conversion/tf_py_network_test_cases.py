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
import unittest

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.alexnet import alexnet_v2
from tensorflow.contrib.slim.python.slim.nets.inception_v1 import inception_v1
from tensorflow.contrib.slim.python.slim.nets.inception_v2 import inception_v2
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3
from tensorflow.contrib.slim.python.slim.nets.overfeat import overfeat
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_50
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_50
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16

from nnef_tests.conversion.tf_py_test_runner import TFPyTestRunner

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')

TEST_MODULE = "nnef_tests.conversion.tf_py_network_test_cases"


def network_alexnet_v2():
    input_shape = [1, 224, 224, 3]
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=input_shape)
    net, _end_points = alexnet_v2(input_, num_classes=1000, is_training=False)
    return net


def network_inception_v1():
    input_shape = [1, 224, 224, 3]
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=input_shape)
    net, _end_points = inception_v1(input_, num_classes=1000, is_training=False)
    return net


def network_inception_v2():
    input_shape = [1, 224, 224, 3]
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=input_shape)
    net, _end_points = inception_v2(input_, num_classes=1000, is_training=False)
    return net


def network_inception_v3():
    input_shape = [1, 224, 224, 3]
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=input_shape)
    net, _end_points = inception_v3(input_, num_classes=1000, is_training=False)
    return net


def network_overfeat():
    input_shape = [1, 231, 231, 3]
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=input_shape)
    net, _end_points = overfeat(input_, num_classes=1000, is_training=False)
    return net


def network_resnet_v1_50():
    input_shape = [1, 224, 224, 3]
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=input_shape)
    net, _end_points = resnet_v1_50(input_, num_classes=1000, is_training=False)
    return net


def network_resnet_v2_50():
    input_shape = [1, 224, 224, 3]
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=input_shape)
    net, _end_points = resnet_v2_50(input_, num_classes=1000, is_training=False)
    return net


def network_vgg_16():
    input_shape = [1, 224, 224, 3]
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=input_shape)
    net, _end_points = vgg_16(input_, num_classes=1000, is_training=False)
    return net


class TFPyNetworkTestCases(TFPyTestRunner):

    def setUp(self):
        super(TFPyNetworkTestCases, self).setUp()
        self.delete_dats_and_checkpoints = True

    def test_inception_v1(self):
        self._test(network_inception_v1, test_module=TEST_MODULE)

    def test_inception_v2(self):
        self._test(network_inception_v2, test_module=TEST_MODULE)

    def test_inception_v3(self):
        self._test(network_inception_v3, test_module=TEST_MODULE)

    def test_overfeat(self):
        self._test(network_overfeat, test_module=TEST_MODULE)

    def test_resnet_v1_50(self):
        self._test(network_resnet_v1_50, test_module=TEST_MODULE)

    def test_resnet_v2_50(self):
        self._test(network_resnet_v2_50, test_module=TEST_MODULE)

    def test_vgg_16(self):
        self._test(network_vgg_16, test_module=TEST_MODULE)

    def test_alexnet_v2(self):
        self._test(network_alexnet_v2, test_module=TEST_MODULE)


if __name__ == "__main__":
    unittest.main()
