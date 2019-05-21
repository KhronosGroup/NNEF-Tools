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

from tests.conversion.caffe_test_runner import CaffeTestRunner
from tests.file_downloader import download_once, download_and_untar_once

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


# From https://github.com/BVLC/caffe/wiki/Model-Zoo
class CaffeNetworkTestCases(CaffeTestRunner):

    # Official models
    def test_bvlc_alexnet(self):
        self._test_model(
            model_path=download_once(
                url="http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel",
                path="_models/caffe/bvlc_alexnet/"),
            prototxt_path=download_once(
                url="https://raw.githubusercontent.com/BVLC/caffe/1.0/models/bvlc_alexnet/deploy.prototxt",
                path="_models/caffe/bvlc_alexnet/"))

    def test_bvlc_googlenet(self):
        self._test_model(
            model_path=download_once(
                url="http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel",
                path="_models/caffe/bvlc_googlenet/"),
            prototxt_path=download_once(
                url="https://raw.githubusercontent.com/BVLC/caffe/1.0/models/bvlc_googlenet/deploy.prototxt",
                path="_models/caffe/bvlc_googlenet/"))

    def test_bvlc_reference_caffenet(self):
        self._test_model(
            model_path=download_once(
                url="http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel",
                path="_models/caffe/bvlc_reference_caffenet/"),
            prototxt_path=download_once(
                url="https://raw.githubusercontent.com/BVLC/caffe/1.0/models/bvlc_reference_caffenet/deploy.prototxt",
                path="_models/caffe/bvlc_reference_caffenet/"))

    def test_bvlc_reference_rcnn_ilsvrc13(self):
        self._test_model(
            model_path=download_once(
                url="http://dl.caffe.berkeleyvision.org/bvlc_reference_rcnn_ilsvrc13.caffemodel",
                path="_models/caffe/bvlc_reference_rcnn_ilsvrc13/"),
            prototxt_path=download_once(
                url="https://raw.githubusercontent.com/BVLC/caffe/1.0/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt",
                path="_models/caffe/bvlc_reference_rcnn_ilsvrc13/"))

    def test_finetune_flickr_style(self):
        self._test_model(
            model_path=download_once(
                url="http://dl.caffe.berkeleyvision.org/finetune_flickr_style.caffemodel",
                path="_models/caffe/finetune_flickr_style/"),
            prototxt_path=download_once(
                url="https://raw.githubusercontent.com/BVLC/caffe/1.0/models/finetune_flickr_style/deploy.prototxt",
                path="_models/caffe/finetune_flickr_style/"))

    # Non-official models

    def test_resnet_50(self):
        self._test_model(
            model_path=download_once(
                url="https://iuxblw.bn.files.1drv.com/y4mF6T7yV0cre7yjwvz4rwfcUk8adzqXG4OmKYGhHuzBXL7MGquwpfvxP883jlrV2xnt2H1w9gv3UFSqVtdWc-JrzglovVYLwh8dlorrKMoADRPiMe0aDyLXGxD6_ru9rA_2vcsnIgy7EV-JIKAI9Pd5nkOKUROu8hJWq6B1-TUJMxIlbrBcso7QNmARdgam8-E_VUqRYY5G0CtaptNdeorQA/ResNet-50-model.caffemodel",
                path="_models/caffe/resnet_50/"),
            prototxt_path=download_once(
                url="https://iuxblw.bn.files.1drv.com/y4mxb8uknveQa_FiQdSOcLSwSboxCuDM2U2lwq2UudIQzXJMUVx1SCabgoQxLBDpDhpxIZBagj4yVVdHvs235X5XQdRqJt1LVYwyS58By14iQGaauVhjLSJOA2_x1dy44Xe2BLkwlcxbL2CdPOAWo-uMTJhFOoVnfXXbyfh7Sgb5U4wr7fY0ewmXKbJTqF6B6fh3OErOXD0jhHXxndAfjPjtw/ResNet-50-deploy.prototxt",
                path="_models/caffe/resnet_50/"))

    def test_mobilenet_v1(self):
        self._test_model(
            model_path=download_once(
                url="https://github.com/shicai/MobileNet-Caffe/blob/master/mobilenet.caffemodel?raw=true",
                path="_models/caffe/mobilenet_v1/"),
            prototxt_path=download_once(
                url="https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_deploy.prototxt",
                path="_models/caffe/mobilenet_v1/"))

    def test_mobilenet_v2(self):
        self._test_model(
            model_path=download_once(
                url="https://github.com/shicai/MobileNet-Caffe/blob/master/mobilenet_v2.caffemodel?raw=true",
                path="_models/caffe/mobilenet_v2/"),
            prototxt_path=download_once(
                url="https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2_deploy.prototxt",
                path="_models/caffe/mobilenet_v2/"))

    def test_squeezenet_v1_0(self):
        self._test_model(
            model_path=download_once(
                url="https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel?raw=true",
                path="_models/caffe/squeezenet_v1_0/"),
            prototxt_path=download_once(
                url="https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.0/deploy.prototxt",
                path="_models/caffe/squeezenet_v1_0/"))

    def test_squeezenet_v1_1(self):
        self._test_model(
            model_path=download_once(
                url="https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel?raw=true",
                path="_models/caffe/squeezenet_v1_1/"),
            prototxt_path=download_once(
                url="https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/deploy.prototxt",
                path="_models/caffe/squeezenet_v1_1/"))

    def test_nin_imagenet(self):
        self._test_model(
            model_path=download_once(
                url="https://www.dropbox.com/s/0cidxafrb2wuwxw/nin_imagenet.caffemodel?dl=1",
                path="_models/caffe/nin_imagenet/"),
            prototxt_path=download_once(
                url="https://gist.githubusercontent.com/tzutalin/0e3fd793a5b13dd7f647/raw/207d710d2e089423eda4b0b76ca4b139b7a461f7/deploy.prototxt",
                path="_models/caffe/nin_imagenet/"))

    def test_googlenet_cars(self):
        self._test_model(
            model_path=download_once(
                url="http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/googlenet_finetune_web_car_iter_10000.caffemodel",
                path="_models/caffe/googlenet_cars/"),
            prototxt_path=download_once(
                url="https://gist.githubusercontent.com/bogger/b90eb88e31cd745525ae/raw/b5dd8c1a58318fdceeeac00322c90e4a865c3229/deploy.prototxt",
                path="_models/caffe/googlenet_cars/"))

    def test_vgg_cnn_s(self):
        self._test_model(
            model_path=download_once(
                url="http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/bvlc/VGG_CNN_S.caffemodel",
                path="_models/caffe/vgg_cnn_s/"),
            prototxt_path=download_once(
                url="https://gist.githubusercontent.com/ksimonyan/fd8800eeb36e276cd6f9/raw/e8dbbd31fc037fdf5430d89c102619e31ca7e8ef/VGG_CNN_S_deploy.prototxt",
                path="_models/caffe/vgg_cnn_s/"))

    def test_vgg_16(self):
        self._test_model(
            model_path=download_once(
                url="http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel",
                path="_models/caffe/vgg_16/"),
            prototxt_path=download_once(
                url="https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt",
                path="_models/caffe/vgg_16/"))

    def test_places205_alexnet(self):
        model_path, prototxt_path = download_and_untar_once(
            url="http://places.csail.mit.edu/model/placesCNN.tar.gz",
            member=['*.caffemodel',
                    '*deploy.prototxt'],
            path=["_models/caffe/places_205_alexnet/places_205_alexnet.caffemodel",
                  "_models/caffe/places_205_alexnet/deploy.prototxt"])
        self._test_model(model_path=model_path, prototxt_path=prototxt_path)

    # Just io test (We don't have the caffemodel for these old networks)

    def test_legacy_io_alexnet(self):
        self._test_io(download_once(
            url="https://raw.githubusercontent.com/intelcaffe/caffe-old/master/examples/imagenet/alexnet_deploy.prototxt",
            path="_models/caffe/old_alexnet/"))

    def test_legacy_io_caffenet(self):
        self._test_io(download_once(
            url="https://raw.githubusercontent.com/intelcaffe/caffe-old/master/examples/imagenet/imagenet_deploy.prototxt",
            path="_models/caffe/old_caffenet/"))


if __name__ == '__main__':
    unittest.main()
