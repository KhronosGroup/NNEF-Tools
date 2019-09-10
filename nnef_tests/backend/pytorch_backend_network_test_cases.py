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

from nnef_tools.run import run_using_command
from nnef_tests.file_downloader import download_once, download_and_untar_once

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


class NetworkTestCases(unittest.TestCase):
    # The specified input ranges are just educated guesses

    def test_alexnet(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/bvlc_alexnet.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/bvlc_alexnet.onnx.nnef.tgz \\
                    --input="Random(uniform, 0, 255)" \\
                    --stats \\
                    --stats-path ../../../out/alexnet/
        """)

    def test_vgg16(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/vgg16.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/vgg16.onnx.nnef.tgz \\
                    --input-shape "[2, 3, 224, 224]" \\
                    --input="Random(uniform, 0, 1)" \\
                    --stats \\
                    --stats-path ../../../out/vgg16/
        """)

    def test_vgg19(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/vgg19.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/vgg19.onnx.nnef.tgz \\
                    --input="Random(uniform, 0, 1)" \\
                    --stats \\
                    --stats-path ../../../out/vgg19/
        """)

    def test_googlenet_inception_v1(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/inception_v1.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/inception_v1.onnx.nnef.tgz \\
                    --input="Random(uniform, 0, 255)" \\
                    --stats \\
                    --stats-path ../../../out/googlenet_inception_v1/ \\
                    --permissive
        """)

    def test_googlenet_inception_v2(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/inception_v2.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/inception_v2.onnx.nnef.tgz \\
                    --input="Random(uniform, 0, 255)" \\
                    --stats \\
                    --stats-path ../../../out/googlenet_inception_v2/
        """)

    def test_googlenet_inception_v3(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/inception_v3.tfpb.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/inception_v3.tfpb.nnef.tgz \\
                    --input="Random(uniform, -1, 1)" \\
                    --stats \\
                    --stats-path ../../../out/googlenet_inception_v3/
        """)

    def test_googlenet_inception_v4(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/inception_v4.tfpb.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/inception_v4.tfpb.nnef.tgz \\
                    --input="Random(uniform, -1, 1)" \\
                    --stats \
                    --stats-path ../../../out/googlenet_inception_v4/
        """)

    def test_bvlc_googlenet(self):
        # This will append the stats file to the archive
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/bvlc_googlenet.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/bvlc_googlenet.onnx.nnef.tgz \\
                    --input="Random(uniform, 0, 255)" \\
                    --stats
        """)

    def test_resnet_v1_18(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v1_18.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/resnet_v1_18.onnx.nnef.tgz \\
                    --input="Random(uniform, 0, 1)" \\
                    --stats \\
                    --stats-path ../../../out/resnet_v1_18/
        """)

    def test_resnet_v1_34(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v1_34.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/resnet_v1_34.onnx.nnef.tgz \\
                    --input="Random(uniform, 0, 1)" \\
                    --stats \\
                    --stats-path ../../../out/resnet_v1_34/
        """)

    def test_resnet_v1_50(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v1_50.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/resnet_v1_50.onnx.nnef.tgz \\
                    --input="Random(uniform, 0, 1)" \\
                    --stats \\
                    --stats-path ../../../out/resnet_v1_50/
        """)

    def test_resnet_v1_101(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v1_101.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/resnet_v1_101.onnx.nnef.tgz \\
                    --input="Random(uniform, 0, 1)" \\
                    --stats \\
                    --stats-path ../../../out/resnet_v1_101/
        """)

    def test_resnet_v2_18(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v2_18.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/resnet_v2_18.onnx.nnef.tgz \\
                    --input="Random(uniform, 0, 1)" \\
                    --stats \\
                    --stats-path ../../../out/resnet_v2_18/
        """)

    def test_resnet_v2_34(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v2_34.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/resnet_v2_34.onnx.nnef.tgz \\
                    --input="Random(uniform, 0, 1)" \\
                    --stats \\
                    --stats-path ../../../out/resnet_v2_34/
        """)

    def test_resnet_v2_50(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v2_50.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/resnet_v2_50.onnx.nnef.tgz \\
                    --input="Random(uniform, 0, 1)" \\
                    --stats \\
                    --stats-path ../../../out/resnet_v2_50/
        """)

    def test_resnet_v2_101(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v2_101.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/resnet_v2_101.onnx.nnef.tgz \\
                    --input="Random(uniform, 0, 1)" \\
                    --stats \\
                    --stats-path ../../../out/resnet_v2_101/
        """)

    def test_inception_resnet_v2(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/inception_resnet_v2.tfpb.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/inception_resnet_v2.tfpb.nnef.tgz \\
                    --input="Random(uniform, 0, 1)" \\
                    --stats \\
                    --stats-path ../../../out/inception_resnet_v2/
        """)

    def test_mobilenet_v1_tf(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/mobilenet_v1_1.0.tfpb.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/mobilenet_v1_1.0.tfpb.nnef.tgz \\
                    --input="Random(uniform, -1, 1)" \\
                    --stats \\
                    --stats-path ../../../out/mobilenet_v1_tf/
        """)

    def test_mobilenet_v2_tf(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/mobilenet_v2_1.0.tfpb.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/mobilenet_v2_1.0.tfpb.nnef.tgz \\
                    --input="Random(uniform, -1, 1)" \\
                    --stats \\
                    --stats-path ../../../out/mobilenet_v2_tf/
        """)

    def test_mobilenet_v2_onnx(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/mobilenet_v2_1.0.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/mobilenet_v2_1.0.onnx.nnef.tgz \\
                    --input="Random(uniform, -1, 1)" \\
                    --stats \\
                    --stats-path ../../../out/mobilenet_v2_onnx/
        """)

    def test_squeezenet_tf(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/squeezenet.tfpb.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/squeezenet.tfpb.nnef.tgz \\
                    --input="Random(uniform, 0, 255)" \\
                    --stats \\
                    --stats-path ../../../out/squeezenet_tf/
        """)

    def test_squeezenet_1_0_onnx(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/squeezenet_v1.0.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/squeezenet_v1.0.onnx.nnef.tgz \\
                    --input="Random(uniform, -127.5, 127.5)" \\
                    --stats \\
                    --stats-path ../../../out/squeezenet_1_0_onnx/
        """)

    def test_squeezenet_1_1_onnx(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/squeezenet_v1.1.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/squeezenet_v1.1.onnx.nnef.tgz \\
                    --input="Random(uniform, -127.5, 127.5)" \\
                    --stats \\
                    --stats-path ../../../out/squeezenet_1_1_onnx/
        """)

    def test_squeezenet_1_0_caffe(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/squeezenet_v1.0.caffemodel.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/squeezenet_v1.0.caffemodel.nnef.tgz \\
                    --input="Random(uniform, -127.5, 127.5)" \\
                    --stats \\
                    --stats-path ../../../out/squeezenet_1_0_caffe/
        """)

    def test_squeezenet_1_1_caffe(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/squeezenet_v1.1.caffemodel.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/squeezenet_v1.1.caffemodel.nnef.tgz \\
                    --input="Random(uniform, -127.5, 127.5)" \\
                    --stats \\
                    --stats-path ../../../out/squeezenet_1_1_caffe/
        """)

    def test_shufflenet(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/shufflenet.onnx.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/shufflenet.onnx.nnef.tgz \\
                    --input="Random(uniform, -1, 1)" \\
                    --stats \\
                    --stats-path ../../../out/shufflenet/
        """)

    def test_nasnet_mobile(self):
        download_once("https://sfo2.digitaloceanspaces.com/nnef-public/nasnet_mobile.tfpb.nnef.tgz",
                      path="_models/nnef/")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/nasnet_mobile.tfpb.nnef.tgz \\
                    --input="Random(uniform, -1, 1)" \\
                    --stats \\
                    --stats-path ../../../out/nasnet_mobile/
        """)

    def test_squeezenet_generated(self):
        download_and_untar_once("https://sfo2.digitaloceanspaces.com/nnef-public/squeezenet.tfpb.nnef.tgz",
                                member='graph.nnef',
                                path="_models/nnef/generated_squeezenet.nnef")
        run_using_command("""
./nnef_tools/run.py --input-model _models/nnef/generated_squeezenet.nnef \\
                    --input="Random(uniform, 0, 255)" \\
                    --generate-weights
        """)
