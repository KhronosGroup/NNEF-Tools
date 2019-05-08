#!/usr/bin/env python

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

try:
    import tensorflow as tf
except ImportError:
    print("Must have TensorFlow to run this!")
    exit(1)

from tensorflow.contrib.slim.python.slim.nets.alexnet import alexnet_v2
from tensorflow.contrib.slim.python.slim.nets.inception_v1 import inception_v1
from tensorflow.contrib.slim.python.slim.nets.inception_v2 import inception_v2
from nnef_tools.core import utils
from nnef_tools.convert import convert_using_command
from nnef_tools.export_activation import export_activation_using_command
from tests.conversion.tf_py_test_runner import save_random_checkpoint, get_feed_dict

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


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


# From: https://www.tensorflow.org/lite/guide/hosted_models
class TFPyNetworkTestCases(unittest.TestCase):
    def _test_network(self, fun):
        with utils.EnvVars(TF_CPP_MIN_LOG_LEVEL=3):
            fun_name = fun.__name__

            network_outputs = fun()
            checkpoint_path = os.path.join("out", "checkpoint", fun_name, fun_name + ".ckpt")
            feed_dict = get_feed_dict()
            checkpoint_path = save_random_checkpoint(network_outputs, checkpoint_path, feed_dict)

            tf.reset_default_graph()
            tf.set_random_seed(0)

            command = """
            ./nnef_tools/convert.py 
                --input-format tensorflow-py \\
                --input-model tests.activation_export.tf_py_network_test_cases.{fun_name} {cp_path}\\
                --output-format nnef \\
                --output-model out/nnef/{fun_name}.nnef.tgz \\
                --compress \\
                --conversion-info""".format(fun_name=fun_name, cp_path=checkpoint_path)
            print(command)
            convert_using_command(command)

            command = """
            ./nnef_tools/export_activation.py
                --input-format tensorflow-py \\
                --input-model tests.activation_export.tf_py_network_test_cases.{fun_name} {cp_path} \\
                --conversion-info out/nnef/{fun_name}.nnef.tgz.conversion.json
            """.format(fun_name=fun_name, cp_path=checkpoint_path)
            print(command)
            export_activation_using_command(command)

    def test_inception_v1(self):
        self._test_network(network_inception_v1)

    def test_inception_v2(self):
        self._test_network(network_inception_v2)

    def test_alexnet_v2(self):
        self._test_network(network_alexnet_v2)


if __name__ == '__main__':
    unittest.main()
