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
from nnef_tools.core import utils
from nnef_tools.export_activation import export_activation_using_command
from nnef_tests.file_downloader import download_once

try:
    with utils.EnvVars(GLOG_minloglevel=3):
        import caffe
except ImportError:
    print("Must have Caffe to run this!")
    exit(1)

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


class CaffeNetworkTestCases(unittest.TestCase):
    def _test_network(self, prototxt_path, model_path):
        network = os.path.basename(model_path.rsplit('.', 1)[0])

        command = """
        ./nnef_tools/convert.py --input-format caffe \\
                                --input-model {prototxt_path} {caffemodel_path} \\
                                --output-format nnef \\
                                --output-model out/nnef/{network}.nnef.tgz \\
                                --compress \\
                                --conversion-info""".format(prototxt_path=prototxt_path,
                                                            caffemodel_path=model_path,
                                                            network=network)
        print(command)
        convert_using_command(command)

        command = """
        ./nnef_tools/export_activation.py --input-format caffe \\
                                          --input-model {prototxt_path} {caffemodel_path} \\
                                          --conversion-info out/nnef/{network}.nnef.tgz.conversion.json
        """.format(prototxt_path=prototxt_path, caffemodel_path=model_path, network=network)
        print(command)
        export_activation_using_command(command)

    def test_mobilenet_v1(self):
        self._test_network(
            model_path=download_once(
                url="https://github.com/shicai/MobileNet-Caffe/blob/master/mobilenet.caffemodel?raw=true",
                path="_models/caffe/mobilenet_v1/"),
            prototxt_path=download_once(
                url="https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_deploy.prototxt",
                path="_models/caffe/mobilenet_v1/"))

    def test_squeezenet_v1_0(self):
        self._test_network(
            model_path=download_once(
                url="https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel?raw=true",
                path="_models/caffe/squeezenet_v1_0/"),
            prototxt_path=download_once(
                url="https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.0/deploy.prototxt",
                path="_models/caffe/squeezenet_v1_0/"))


if __name__ == '__main__':
    unittest.main()
