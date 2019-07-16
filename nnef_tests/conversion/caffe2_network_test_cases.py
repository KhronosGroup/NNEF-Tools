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

from nnef_tests.conversion.caffe2_test_runner import Caffe2TestRunner
from nnef_tests.file_downloader import download_once

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


def download_caffe2_model(name):
    predict_net_url = "https://s3.amazonaws.com/download.caffe2.ai/models/{}/predict_net.pb".format(name)
    init_net_url = "https://s3.amazonaws.com/download.caffe2.ai/models/{}/init_net.pb".format(name)
    value_info_url = "https://raw.githubusercontent.com/caffe2/models/master/{}/value_info.json".format(name)
    out_dir = "_models/caffe2/{}/".format(name)
    return (download_once(predict_net_url, out_dir),
            download_once(init_net_url, out_dir),
            download_once(value_info_url, out_dir))


# From https://github.com/caffe2/models
class Caffe2NetworkTestCases(Caffe2TestRunner):

    def test_bvlc_alexnet(self):
        self._test_model(*download_caffe2_model('bvlc_alexnet'))

    def test_bvlc_googlenet(self):
        self._test_model(*download_caffe2_model('bvlc_googlenet'))

    def test_bvlc_reference_caffenet(self):
        self._test_model(*download_caffe2_model('bvlc_reference_caffenet'))

    def test_bvlc_reference_rcnn_ilsvrc13(self):
        self._test_model(*download_caffe2_model('bvlc_reference_rcnn_ilsvrc13'))

    def test_densenet121(self):
        self._test_model(*download_caffe2_model('densenet121'))

    def test_inception_v1(self):
        self._test_model(*download_caffe2_model('inception_v1'))

    def test_inception_v2(self):
        self._test_model(*download_caffe2_model('inception_v2'))

    def test_resnet50(self):
        self._test_model(*download_caffe2_model('resnet50'))

    def test_squeezenet(self):
        self._test_model(*download_caffe2_model('squeezenet'))

    def test_vgg19(self):
        self._test_model(*download_caffe2_model('vgg19'))

    def test_zfnet512(self):
        self._test_model(*download_caffe2_model('zfnet512'))

    def test_mobilenet_v2(self):
        self._test_model(
            download_once(
                'https://s3.amazonaws.com/download.caffe2.ai/models/mobilenet_v2/predict_net.pb',
                '_models/caffe2/mobilenet_v2/'),
            download_once(
                'https://s3.amazonaws.com/download.caffe2.ai/models/mobilenet_v2/init_net.pb',
                '_models/caffe2/mobilenet_v2/'),
            download_once(  # not at the usual place
                'https://media.githubusercontent.com/media/caffe2/models/master/mobilenet_v2/value_info.json',
                '_models/caffe2/mobilenet_v2/'))


if __name__ == '__main__':
    unittest.main()
