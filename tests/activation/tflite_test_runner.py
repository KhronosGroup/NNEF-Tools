from __future__ import division, print_function, absolute_import

import unittest

import numpy as np
import tensorflow as tf

from nnef_tools import convert
from nnef_tools.io.nnef import nnef_io
from nnef_tools.io.nnef.parser_config import NNEFParserConfig


class TFLiteTestRunner(unittest.TestCase):

    @staticmethod
    def run_model(model_path, input_data=None, max_val=255.0):
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
            input_data = np.array(max_val*np.random.random_sample(input_shape), dtype=input_type)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index']), input_data

    def _test_model(self, filename, run=True, compare=True, max_val=255.0):
        output, output2, input = None, None, None
        if run:
            output, input = self.run_model(model_path=filename, max_val=max_val)

        network_name = filename.rsplit('/', 1)[1].rsplit('.', 1)[0].replace('.', '_').replace('-', '_')
        print(filename)
        command = """
        ./nnef_tools/convert.py --input-framework=tensorflow-lite \\
                                --output-framework=nnef \\
                                --input-model={input} \\
                                --output-directory=out/nnef/{network} \\
                                --permissive
        """.format(input=filename, network=network_name)
        print(command)
        convert.convert_using_command(command)

        command = """
        ./nnef_tools/convert.py --input-framework=nnef \\
                                --output-framework=tensorflow-lite \\
                                --input-model=out/nnef/{network}/model \\
                                --output-directory=out/tflite/{network} \\
                                --permissive
        """.format(network=network_name)
        print(command)
        convert.convert_using_command(command)

        if run:
            output2, _ = self.run_model(model_path="out/tflite/{}/model.tflite".format(network_name),
                                        input_data=input,
                                        max_val=max_val)

        if compare:
            print('Compare:')
            print(output.shape, np.min(output), np.mean(output), np.max(output))
            print(output2.shape, np.min(output2), np.mean(output2), np.max(output2))
            self.assertTrue(np.all(np.isfinite(output)))
            self.assertTrue(np.all(np.isfinite(output2)))
            self.assertTrue(np.allclose(output, output2, atol=1e-5))
