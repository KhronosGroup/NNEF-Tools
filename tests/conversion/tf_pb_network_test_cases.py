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

from tests.conversion.file_downloader import download_and_untar_once
from tests.conversion.tf_pb_test_runner import TFPbTestRunner

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


# From: https://www.tensorflow.org/lite/guide/hosted_models
class TFPbNetworkTestCases(TFPbTestRunner):
    def test_densenet(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/densenet_2018_04_27.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/densenet_2018_04_27.pb")
        self._test_network(path)

    def test_squeezenet(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/squeezenet_2018_04_27.pb")
        self._test_network(path)

    def test_nasnet_mobile(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/nasnet_mobile_2018_04_27.pb")
        self._test_network(path)

    def test_nasnet_large(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_large_2018_04_27.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/nasnet_large_2018_04_27.pb")
        self._test_network(path)

    def test_inception_v3(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/inception_v3_2018_04_27.pb")
        self._test_network(path)

    def test_inception_v4(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/inception_v4_2018_04_27.pb")
        self._test_network(path)

    def test_inception_resnet_v2(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/inception_resnet_v2_2018_04_27.pb")
        self._test_network(path)

    def test_mobilenet_v1_0_25_128(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/mobilenet_v1_0.25_128.pb")
        self._test_network(path)

    def test_mobilenet_v1_1_0_128(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_128.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/mobilenet_v1_1.0_128.pb")
        self._test_network(path)

    def test_mobilenet_v2_1_0_224(self):
        path = download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz",
            member="*.pb",
            path="_models/tensorflow-pb/mobilenet_v2_1.0_224.pb")
        self._test_network(path, source_shape=[1, 224, 224, 3])


if __name__ == '__main__':
    unittest.main()
