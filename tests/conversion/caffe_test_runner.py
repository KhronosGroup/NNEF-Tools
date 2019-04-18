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

import numpy as np
import typing

from nnef_tools.convert import convert_using_command
from nnef_tools.core import utils
from nnef_tools.io.caffe.caffe_io import Reader, Writer


class CaffeTestRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        utils.rmtree('out', exist_ok=True)

    def _test_model(self, prototxt_path, model_path, nonnegative_input=False):
        network_name = utils.split_path(prototxt_path)[-2]
        print('Testing {}: {}, {}'.format(network_name, prototxt_path, model_path))
        reader = Reader()
        g = reader(prototxt_path, model_path)

        writer = Writer()
        our_prototxt_path = os.path.join('out', 'caffe-ours', network_name, network_name + '.prototxt')
        our_model_path = os.path.join('out', 'caffe-ours', network_name, network_name + '.caffemodel')
        writer(g, our_prototxt_path)
        del g

        nnef_path = os.path.join('out', 'nnef', network_name + '.nnef')
        command = """
        ./nnef_tools/convert.py --input-format caffe \\
                                --output-format nnef \\
                                --input-model {prototxt} {caffemodel} \\
                                --output-model {nnef}
        """.format(prototxt=prototxt_path, caffemodel=model_path, nnef=nnef_path)
        print(command)
        convert_using_command(command)

        converted_prototxt_path = os.path.join('out', 'caffe-converted', network_name, network_name + '.prototxt')
        converted_model_path = os.path.join('out', 'caffe-converted', network_name, network_name + '.caffemodel')
        command = """
        ./nnef_tools/convert.py --input-format nnef \\
                                --output-format caffe \\
                                --input-model {nnef} \\
                                --output-model {prototxt}
        """.format(nnef=nnef_path, prototxt=converted_prototxt_path)
        print(command)
        convert_using_command(command)

        if int(os.environ.get('NNEF_ACTIVATION_TESTING', '1')):
            with utils.EnvVars(GLOG_minloglevel=3):
                import caffe
                import numpy as np

                def random_inputs(net):
                    # type: (caffe.Net) -> typing.Dict[str, np.ndarray]
                    np.random.seed(0)
                    sub = 0.0 if nonnegative_input else 0.5
                    return {input: np.random.random(list(net.blobs[input].shape)) - sub for input in net.inputs}

                print("Running original net...")
                net = caffe.Net(prototxt_path, model_path, caffe.TEST)
                out_orig = net.forward(**random_inputs(net))

                print("Running generated net...")
                net = caffe.Net(our_prototxt_path, our_model_path, caffe.TEST)
                out_ours = net.forward(**random_inputs(net))
                self.assertEqual(len(out_orig), len(out_ours))
                for orig_name, other_name, orig_arr, other_arr in zip(out_orig.keys(),
                                                                      out_ours.keys(),
                                                                      out_orig.values(),
                                                                      out_ours.values()):
                    self.assertTrue(np.allclose(orig_arr, other_arr))

                print("Running converted net...")
                net = caffe.Net(converted_prototxt_path, converted_model_path, caffe.TEST)
                out_converted = net.forward(**random_inputs(net))

                self.assertEqual(len(out_orig), len(out_converted))
                for orig_name, other_name, orig_arr, other_arr in zip(out_orig.keys(),
                                                                      out_converted.keys(),
                                                                      out_orig.values(),
                                                                      out_converted.values()):
                    self.assertTrue(np.allclose(orig_arr, other_arr, rtol=1e-5, atol=1e-5))
        print('Done')

    def _test_io(self, prototxt_path):
        network_name = utils.split_path(prototxt_path)[-2]
        print('Testing {}: {}'.format(network_name, prototxt_path))
        reader = Reader()
        g = reader(graph_filename=prototxt_path)

        writer = Writer()
        our_prototxt_path = os.path.join('out', 'caffe-ours', network_name, network_name + '.prototxt')
        writer(g, our_prototxt_path)
        del g

        if int(os.environ.get('NNEF_ACTIVATION_TESTING', '1')):
            with utils.EnvVars(GLOG_minloglevel=3):
                import caffe
                print("Loading generated net...")
                caffe.Net(our_prototxt_path, caffe.TEST)

        print('Done')
