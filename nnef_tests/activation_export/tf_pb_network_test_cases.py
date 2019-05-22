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

from nnef_tools.convert import convert_using_command
from nnef_tools.export_activation import export_activation_using_command
from nnef_tests.file_downloader import download_and_untar_once

try:
    import tensorflow as tf
except ImportError:
    print("Must have TensorFlow to run this!")
    exit(1)

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


# From: https://www.tensorflow.org/lite/guide/hosted_models
class TFPbNetworkTestCases(unittest.TestCase):
    def _test_network(self, path):
        network = os.path.basename(path.rsplit('.', 1)[0])

        command = """
        ./nnef_tools/convert.py --input-format=tensorflow-pb \\
                                --input-model={path} \\
                                --output-format=nnef \\
                                --output-model=out/nnef/{network}.nnef.tgz \\
                                --compress \\
                                --conversion-info""".format(path=path, network=network)
        print(command)
        convert_using_command(command)

        command = """
        ./nnef_tools/export_activation.py --input-format tensorflow-pb \\
                                          --input-model {path} \\
                                          --conversion-info out/nnef/{network}.nnef.tgz.conversion.json
        """.format(path=path, network=network)
        print(command)
        export_activation_using_command(command)

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


if __name__ == '__main__':
    unittest.main()
