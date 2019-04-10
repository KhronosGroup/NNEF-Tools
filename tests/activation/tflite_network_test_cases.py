from __future__ import division, print_function, absolute_import

import os
import unittest

from tests.activation.file_downloader import download_and_untar_once
from tests.activation.tflite_test_runner import TFLiteTestRunner

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


class TFLiteNetworkTestCases(TFLiteTestRunner):
    def test_mnasnet_0_5_224(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/mnasnet_0.5_224_09_07_2018.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/mnasnet_0.5_224_09_07_2018.tflite"))

    def test_mobilenet_v2_1_0_224_quant(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/mobilenet_v2_1.0_224_quant.tflite"))

    def test_nasnet_mobile(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/nasnet_mobile_2018_04_27.tflite"))

    def test_inception_v4_299_quant(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/inception_v4_299_quant_20181026.tflite"))

    def test_inception_v3(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/inception_v3_2018_04_27.tflite"))

    def test_mobilenet_v1_0_25_128(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/mobilenet_v1_0.25_128.tflite"))

    def test_mobilenet_v2_1_0_224(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/mobilenet_v2_1.0_224.tflite"))

    def test_inception_resnet_v2(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/inception_resnet_v2_2018_04_27.tflite"))

    def test_inception_v1_224_quant(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/inception_v1_224_quant_20181026.tflite"))

    def test_mobilenet_v1_0_25_128_quant(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/mobilenet_v1_0.25_128_quant.tflite"))

    def test_resnet_v2_101_299(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/resnet_v2_101.tflite"))

    def test_inception_v4(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/inception_v4_2018_04_27.tflite"))

    def test_nasnet_large(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_large_2018_04_27.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/nasnet_large_2018_04_27.tflite"))

    def test_squeezenet(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/squeezenet_2018_04_27.tflite"))

    def test_densenet(self):
        self._test_model(download_and_untar_once(
            url="http://download.tensorflow.org/models/tflite/model_zoo/upload_20180427/densenet_2018_04_27.tgz",
            member="*.tflite",
            path="_models/tensorflow-lite/densenet_2018_04_27.tflite"))


if __name__ == '__main__':
    unittest.main()
