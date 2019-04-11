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
from collections import defaultdict

import caffe2.python.onnx.backend as backend
import caffe2.python.onnx.frontend
import numpy as np
import onnx
from caffe2.python import model_helper

from nnef_tools import convert
from nnef_tools.io.onnx import onnx_io


def check_onnx_model(filename):
    model = onnx.load(filename)
    onnx.checker.check_model(model)


def run_onnx_model(filename, input):
    model = onnx.load(filename)
    try:
        rep = backend.prepare(model, device="CUDA:0")
        outputs = rep.run(input)
    except Exception:
        print("Couldn't run in CUDA, running on CPU:")
        rep = backend.prepare(model, device="CPU")
        outputs = rep.run(input)

    return outputs


class ONNXTestRunner(unittest.TestCase):
    def setUp(self):
        self.op_num = defaultdict(lambda: 0)

    def _test_from_caffe2(self, net_fun, custom_converters=None):
        predict_net, init_net, value_info = net_fun()
        name = net_fun.__name__.rsplit('_', 1)[0]
        filename = os.path.join('out', 'source_onnx', name + '.onnx')
        onnx_model = caffe2.python.onnx.frontend.caffe2_net_to_onnx_model(
            predict_net,
            init_net,
            value_info,
        )
        onnx.checker.check_model(onnx_model)

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, 'wb') as f:
            onnx.save_model(onnx_model, f)

        self._test_model(filename, custom_converters=custom_converters)

    def _test_from_onnx_graph(self, g, name, run=True, custom_converters=None):
        filename = os.path.join('out', 'source_onnx', name + '.onnx')

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        onnx_io.write_onnx_to_protobuf(g, filename)

        self._test_model(filename, run=run, custom_converters=custom_converters)

    def get_unary_network_function(self, op_name, kwargs=None, dtype=None, shape=None):
        if kwargs is None:
            kwargs = {}
        if dtype is None:
            dtype = onnx.TensorProto.FLOAT
        if shape is None:
            shape = [1, 1, 5, 5]

        def f():
            value_info = {'x': (dtype, shape)}
            model = model_helper.ModelHelper(name='test_network')
            model.net.AddExternalInput('x')
            getattr(model.net, op_name)('x', 'y', **kwargs)
            model.net.AddExternalOutput(model.net.GetBlobRef('y'))

            return model.net.Proto(), model.param_init_net.Proto(), value_info

        f.__name__ = '{}{}_network'.format(op_name, self.op_num[op_name])
        self.op_num[op_name] += 1
        return f

    def get_binary_network_function(self, op_name, dtype=onnx.TensorProto.FLOAT):
        def f():
            print(op_name, dtype)
            value_info = {'x': (dtype, [1, 1, 5, 5]),
                          'y': (dtype, [1, 1, 5, 1])}
            model = model_helper.ModelHelper(name='test_network')
            model.net.AddExternalInput('x')
            model.net.AddExternalInput('y')
            getattr(model.net, op_name)(['x', 'y'], 'z')
            model.net.AddExternalOutput(model.net.GetBlobRef('z'))

            return model.net.Proto(), model.param_init_net.Proto(), value_info

        f.__name__ = '{}{}_network'.format(op_name, self.op_num[op_name])
        self.op_num[op_name] += 1
        return f

    def get_ternary_network_function(self, op_name, dtype=onnx.TensorProto.FLOAT):
        def f():
            print(op_name, dtype)
            value_info = {'x': (dtype, [1, 1, 5, 5]),
                          'y': (dtype, [1, 1, 5, 5]),
                          'z': (dtype, [1, 1, 5, 5])}
            model = model_helper.ModelHelper(name='test_network')
            model.net.AddExternalInput('x')
            model.net.AddExternalInput('y')
            model.net.AddExternalInput('z')
            getattr(model.net, op_name)(['x', 'y', 'z'], 'w')
            model.net.AddExternalOutput(model.net.GetBlobRef('w'))

            return model.net.Proto(), model.param_init_net.Proto(), value_info

        f.__name__ = '{}{}_network'.format(op_name, self.op_num[op_name])
        self.op_num[op_name] += 1
        return f

    def _test_model(self, filename, run=True, compare=True, source_shape="None", custom_converters=None):
        if custom_converters is None:
            custom_converters = []

        convs = ["onnx_to_nnef_" + conv for conv in custom_converters]
        network_name = filename.rsplit('/', 1)[1].rsplit('.', 1)[0].replace('.', '_').replace('-', '_')
        print(filename)
        command = """
        ./nnef_tools/convert.py --input-framework=onnx \\
                                --output-framework=nnef \\
                                --input-model={} \\
                                --output-directory=out/nnef/{} \\
                                --input-shape="{}" \\
                                --custom-converters="{}" \\
                                --permissive
        """.format(filename, network_name, source_shape, ','.join(convs))
        print(command)
        convert.convert_using_command(command)

        convs = ["nnef_to_onnx_" + conv for conv in custom_converters]
        command = """
        ./nnef_tools/convert.py --input-framework=nnef \\
                                --output-framework=onnx \\
                                --input-model=out/nnef/{}/model \\
                                --output-directory=out/onnx/{} \\
                                --custom-converters="{}" \\
                                --permissive
        """.format(network_name, network_name, ','.join(convs))
        print(command)
        convert.convert_using_command(command)

        filename2 = os.path.join('out', 'onnx', network_name, 'model.onnx')
        check_onnx_model(filename2)

        g = onnx_io.read_onnx_from_protobuf(filename2)
        input_shapes = [input.shape for input in g.inputs]
        input_dtypes = [input.dtype for input in g.inputs]

        del g

        if run:
            inputs = []
            for input_shape, input_dtype in zip(input_shapes, input_dtypes):
                if input_dtype == 'FLOAT':
                    inputs.append(np.random.random(input_shape).astype(np.float32) * 0.8 + 0.1)
                elif input_dtype == 'BOOL':
                    inputs.append(np.random.random(input_shape) > 0.5)
                elif input_dtype == 'INT64':
                    inputs.append((np.random.random(input_shape) * 1000).astype(np.int32))
                else:
                    assert False

            outputs = None
            if compare:
                print('Running original ONNX:')
                outputs = run_onnx_model(filename, inputs)

            print('Running our ONNX:')
            outputs2 = run_onnx_model(filename2, inputs)

            if compare:
                print('Comparing:')
                for output, output2 in zip(outputs, outputs2):
                    # print('Max dist:', np.max(np.abs(output-output2)))
                    self.assertTrue(np.all(np.isfinite(output)))
                    self.assertTrue(np.all(np.isfinite(output2)))
                    self.assertTrue(np.allclose(output, output2, atol=1e-5))
