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

from nnef_tests.file_downloader import download_once, download_and_untar_once
from nnef_tests.conversion.onnx_test_runner import ONNXTestRunner

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


class ONNXNetworkTestCases(ONNXTestRunner):
    def test_alexnet(self):
        self._test_model(
            download_and_untar_once(url="https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_alexnet.tar.gz",
                                    member="*.onnx",
                                    path="_models/onnx/alexnet.onnx"))

    def test_inception(self):
        self._test_model(
            download_and_untar_once(url="https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v1.tar.gz",
                                    member="*.onnx",
                                    path="_models/onnx/inception_v1.onnx"))

    def test_resnet101v2(self):
        self._test_model(
            download_once(url="https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.onnx",
                          path="_models/onnx/"), run=False)

    def test_mnist(self):
        self._test_model(
            download_and_untar_once(url="https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz",
                                    member="*.onnx",
                                    path="_models/onnx/mnist.onnx"), compare=False)

    def test_vgg16(self):
        self._test_model(download_once(url="https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.onnx",
                                       path="_models/onnx/"))

    def test_zfnet512(self):
        self._test_model(
            download_and_untar_once(url="https://s3.amazonaws.com/download.onnx/models/opset_8/zfnet512.tar.gz",
                                    member="*.onnx",
                                    path="_models/onnx/zfnet512.onnx"))

    def test_squeezenet1_1(self):
        self._test_model(
            download_once(url="https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.onnx",
                          path="./_models/onnx/"))

    def test_caffenet(self):
        self._test_model(
            download_and_untar_once(
                url="https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_reference_caffenet.tar.gz",
                member="*.onnx",
                path="_models/onnx/bvlc_reference_caffenet.onnx"))

    def test_tiny_yolo_v2_opset7(self):
        self._test_model(
            download_and_untar_once(
                url="https://onnxzoo.blob.core.windows.net/models/opset_7/tiny_yolov2/tiny_yolov2.tar.gz",
                member="*.onnx",
                path="_models/onnx/tiny_yolov2_opset7.onnx"),
            compare=False)

    def test_emotion_ferplus_opset1(self):
        self._test_model(
            download_and_untar_once(
                url="https://onnxzoo.blob.core.windows.net/models/opset_2/emotion_ferplus/emotion_ferplus.tar.gz",
                member="*.onnx",
                path="_models/onnx/emotion_ferplus_opset1.onnx"),
            compare=False)

    def test_tiny_yolo_v2_opset8(self):
        self._test_model(
            download_and_untar_once(
                url="https://onnxzoo.blob.core.windows.net/models/opset_8/tiny_yolov2/tiny_yolov2.tar.gz",
                member="*.onnx",
                path="_models/onnx/tiny_yolov2_opset8.onnx"),
            compare=False)

    def test_shufflenet(self):
        self._test_model(
            download_and_untar_once(
                url="https://s3.amazonaws.com/download.onnx/models/opset_8/shufflenet.tar.gz",
                member="*.onnx",
                path="_models/onnx/shufflenet.onnx"))

    def test_googlenet(self):
        self._test_model(
            download_and_untar_once(
                url="https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_googlenet.tar.gz",
                member="*.onnx",
                path="_models/onnx/bvlc_googlenet.onnx"))

    def test_tiny_yolo_v2_opset1(self):
        self._test_model(
            download_and_untar_once(
                url="https://onnxzoo.blob.core.windows.net/models/opset_1/tiny_yolov2/tiny_yolov2.tar.gz",
                member="*.onnx",
                path="_models/onnx/tiny_yolov2_opset1.onnx"),
            compare=False)

    def test_resnet18v1(self):
        # ONNX has some problem with running the BatchNorm version 7 and it cannot convert it to newer version
        self._test_model(download_once(url="https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.onnx",
                                       path="_models/onnx/"), run=False)

    def test_arcface(self):
        self._test_model(download_once(url="https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100/resnet100.onnx",
                                       path="_models/onnx/arcface_resnet100.onnx"), compare=False)

    def test_mobilenetv2_1_0(self):
        self._test_model(
            download_once(url="https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx",
                          path="_models/onnx/"))

    def test_resnet18v2(self):
        self._test_model(download_once(url="https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.onnx",
                                       path="_models/onnx/"))


if __name__ == '__main__':
    unittest.main()
