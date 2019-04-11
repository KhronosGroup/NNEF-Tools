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
import shutil
import unittest

import numpy as np
import tensorflow as tf

from nnef_tools import convert
from tests.activation.file_downloader import download_and_untar_once

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


# This test only tests the outputs of the networks

def load_graph_from_pb(frozen_graph_filename):
    import tensorflow as tf
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def get_placeholders():
    return [tensor
            for op in tf.get_default_graph().get_operations()
            if 'Placeholder' in op.node_def.op
            for tensor in op.values()]


def get_tensors_with_no_consumers():
    return [tensor
            for op in tf.get_default_graph().get_operations()
            if not any(tensor.consumers() for tensor in op.values())
            for tensor in op.values()]


class TFPbNetworkTestCases2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        np.random.seed(0)
        tf.set_random_seed(0)

        if os.path.exists('out'):
            shutil.rmtree('out')

    @staticmethod
    def _test_gen():
        networks_224 = ['frozen_inception_v1', 'frozen_inception_v2', 'frozen_resnet_v1_101', 'frozen_resnet_v1_152',
                        'frozen_resnet_v1_50', 'frozen_resnet_v2_101', 'frozen_resnet_v2_152', 'frozen_resnet_v2_50',
                        'frozen_vgg_16', 'frozen_vgg_19', 'mobilenet_v2_1.0_224_frozen', 'mobilenet_v2_1.4_224_frozen']
        networks_299 = ['frozen_inception_resnet_v2', 'frozen_inception_v3', 'frozen_inception_v4']
        for network in networks_224 + networks_299:
            print("""
def test_{}(self):
    self._test_network('{}', {})
            """.format(network.replace('.', '_').replace('-', '_'), network, 224 if network in networks_224 else 299))

    def _test_network(self, path, size):
        network = os.path.basename(path.rsplit('.', 1)[0])
        input_shape = "(float32, [2, {size}, {size}, 3])".format(size=size)
        command = """
        ./nnef_tools/convert.py --input-framework=tensorflow-pb \\
                                --input-model={} \\
                                --input-shape="{}" \\
                                --output-framework=nnef \\
                                --output-directory=out/nnef/{} \\
                                --compress""".format(path, input_shape, network)
        print(command)
        convert.convert_using_command(command)

        command = """
        ./nnef_tools/convert.py --input-framework=nnef \\
                                --input-model=out/nnef/{}/model.nnef.tgz \\
                                --output-framework=tensorflow-pb \\
                                --output-directory=out/tensorflow-pb/{}""".format(network, network)
        print(command)
        convert.convert_using_command(command)

        tf.reset_default_graph()
        load_graph_from_pb(path)

        [input] = get_placeholders()
        outputs = get_tensors_with_no_consumers()
        feed = np.random.random([2, size, size, 3])

        with tf.Session() as sess:
            activations = sess.run(outputs, feed_dict={input: feed})

        tf.reset_default_graph()
        load_graph_from_pb('out/tensorflow-pb/{}/model.pb'.format(network))

        [input] = get_placeholders()
        outputs = get_tensors_with_no_consumers()

        with tf.Session() as sess:
            activations2 = sess.run(outputs, feed_dict={input: feed})

        for a1, a2 in zip(activations, activations2):
            self.assertTrue(np.allclose(a1, a2))

    def test_densenet(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/densenet_2018_04_27.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/densenet_2018_04_27.pb")
        self._test_network(path, 224)

    def test_squeezenet(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/squeezenet_2018_04_27.pb")
        self._test_network(path, 224)

    def test_nasnet_mobile(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/nasnet_mobile_2018_04_27.pb")
        self._test_network(path, 224)

    def test_nasnet_large(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_large_2018_04_27.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/nasnet_large_2018_04_27.pb")
        self._test_network(path, 331)

    def test_inception_v3(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/inception_v3_2018_04_27.pb")
        self._test_network(path, 299)

    def test_inception_v4(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/inception_v4_2018_04_27.pb")
        self._test_network(path, 299)

    def test_inception_resnet_v2(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/inception_resnet_v2_2018_04_27.pb")
        self._test_network(path, 299)

    def test_mobilenet_v1_0_25_128(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/mobilenet_v1_0.25_128.pb")
        self._test_network(path, 128)

    def test_mobilenet_v1_1_0_128(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_128.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/mobilenet_v1_1.0_128.pb")
        self._test_network(path, 128)

    def test_mobilenet_v2_1_0_224(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/mobilenet_v2_1.0_224.pb")
        self._test_network(path, 224)


if __name__ == '__main__':
    unittest.main()
