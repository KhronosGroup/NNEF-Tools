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

from nnef_tools import convert


class TFLiteTestRunner(unittest.TestCase):

    @staticmethod
    def run_model(model_path, input_data=None, max_val=255.0):
        import tensorflow as tf
        import numpy as np

        interpreter = tf.contrib.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        input_type = input_details[0]['dtype']
        # change the following line to feed into your own data.
        if input_data is None:
            input_data = np.array(max_val * np.random.random_sample(input_shape), dtype=input_type)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index']), input_data

    def _test_model(self, filename, run=True, compare=True, max_val=255.0):

        network_name = filename.rsplit('/', 1)[1].rsplit('.', 1)[0].replace('.', '_').replace('-', '_')
        print(filename)
        command = """
        ./nnef_tools/convert.py --input-format=tensorflow-lite \\
                                --output-format=nnef \\
                                --input-model={input} \\
                                --output-model=out/nnef/{network}.nnef \\
                                --permissive \\
                                --io-transform SMART_NHWC_TO_NCHW \\
                                --conversion-info
        """.format(input=filename, network=network_name)
        print(command)
        convert.convert_using_command(command)

        command = """
        ./nnef_tools/convert.py --input-format=nnef \\
                                --output-format=tensorflow-lite \\
                                --input-model=out/nnef/{network}.nnef \\
                                --output-model=out/tflite/{network}.tflite \\
                                --permissive \\
                                --io-transform SMART_NCHW_TO_NHWC \\
                                --conversion-info
        """.format(network=network_name)
        print(command)
        convert.convert_using_command(command)

        activation_testing = int(os.environ.get('NNEF_ACTIVATION_TESTING', '1'))
        print("Activation testing is", "ON" if activation_testing else "OFF")
        if activation_testing:
            import numpy as np

            output, output2, input = None, None, None
            if run:
                output, input = self.run_model(model_path=filename, max_val=max_val)
                output2, _ = self.run_model(model_path="out/tflite/{}.tflite".format(network_name),
                                            input_data=input,
                                            max_val=max_val)
            if compare:
                print('Compare:')
                print(output.shape, np.min(output), np.mean(output), np.max(output))
                print(output2.shape, np.min(output2), np.mean(output2), np.max(output2))
                self.assertTrue(np.all(np.isfinite(output)))
                self.assertTrue(np.all(np.isfinite(output2)))
                self.assertTrue(np.allclose(output, output2, atol=1e-5))
