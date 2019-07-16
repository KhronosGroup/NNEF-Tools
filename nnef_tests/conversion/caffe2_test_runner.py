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

import numpy as np
import six

from nnef_tools.convert import convert_using_command
from nnef_tools.io.caffe2.caffe2_io import Reader, Writer
from nnef_tools.io.caffe2.caffe2_pb import dtype_id_to_name
from nnef_tools.core import utils, json_utils

DTYPE_ID_FLOAT = 1


class Input(object):
    def __init__(self, name, shape, dtype=DTYPE_ID_FLOAT):
        if shape is None:
            shape = [1, 2, 3, 4]
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return "Input('{}', {}, {})".format(self.name, self.shape, self.dtype)


def run_caffe2_model(predict_net_path, init_net_path, feed_dict):
    from caffe2.python import workspace
    with open(init_net_path, "rb") as f:
        init_net = f.read()
    with open(predict_net_path, "rb") as f:
        predict_net = f.read()

    predictor = workspace.Predictor(init_net, predict_net)
    return [np.array(arr) for arr in predictor.run(feed_dict)]


class Caffe2TestRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        utils.rmtree('out', exist_ok=True)

    def setUp(self):
        self.op_num = defaultdict(lambda: 0)

    def _get_network_name(self, op_name):
        self.op_num[op_name] += 1
        return op_name + str(self.op_num[op_name])

    def _test_model(self, predict_net_path, init_net_path, value_info_path,
                    feed_dict_override=None, test_shapes=True, test_outputs=True,
                    can_run=True, can_compare=True, can_convert=True):
        network_name = utils.split_path(predict_net_path)[-2]
        print('Testing {}: {}, {}, {}'.format(network_name, predict_net_path, init_net_path, value_info_path))
        reader = Reader()
        g = reader(predict_net_path, init_net_path, value_info_path)
        input_name_shape_dtypes = [(tensor.name, tensor.shape, tensor.dtype) for tensor in g.inputs]
        output_shapes = [t.shape for t in g.outputs]

        our_dir = os.path.join('out', 'caffe2', network_name)
        if can_convert:
            print("Converting...")
            nnef_path = os.path.join('out', 'nnef', network_name + '.nnef')
            command = """
            ./nnef_tools/convert.py --input-format caffe2 \\
                                    --output-format nnef \\
                                    --input-model {predict_net} {init_net} {value_info} \\
                                    --output-model {nnef}
            """.format(predict_net=predict_net_path, init_net=init_net_path, value_info=value_info_path, nnef=nnef_path)
            print(command)
            convert_using_command(command)

            command = """
            ./nnef_tools/convert.py --input-format nnef \\
                                    --output-format caffe2 \\
                                    --input-model {nnef} \\
                                    --output-model {out_dir}
            """.format(nnef=nnef_path, out_dir=our_dir)
            print(command)
            convert_using_command(command)
        else:
            print("Writing out model without conversion...")
            writer = Writer()
            writer(g, our_dir)
            del g

        activation_testing = int(os.environ.get('NNEF_ACTIVATION_TESTING', '1'))
        if not activation_testing:
            print("Activation testing is OFF")
        if can_run and activation_testing:

            from caffe2.python import workspace

            feed_dict = self._get_random_feed_dict(input_name_shape_dtypes)

            if feed_dict_override:
                feed_dict.update(feed_dict_override)

            print('Running original Caffe2 model...')
            outputs = run_caffe2_model(predict_net_path, init_net_path, feed_dict)

            if can_convert:
                print('Running converted Caffe2 model...')
            else:
                print('Running generated Caffe2 model...')
            feed_dict2 = {k.replace('/', '_'): v for k, v in six.iteritems(feed_dict)}
            outputs2 = run_caffe2_model(os.path.join(our_dir, 'predict_net.pb'),
                                        os.path.join(our_dir, 'init_net.pb'),
                                        feed_dict2)

            if can_compare:
                print("Comparing...")
                self.assertEqual({k.replace('/', '_'): v for k, v in six.iteritems(json_utils.load(value_info_path))},
                                 json_utils.load(os.path.join(our_dir, 'value_info.json')))

                for output, output2, output_shape in zip(outputs, outputs2, output_shapes):
                    if test_shapes:
                        self.assertEqual(tuple(output_shape), output.shape)
                        self.assertEqual(tuple(output_shape), output2.shape)
                    if test_outputs:
                        self.assertTrue(np.all(np.isfinite(output)))
                        self.assertTrue(np.all(np.isfinite(output2)))
                        self.assertTrue(np.allclose(output, output2, atol=1e-5))
        print('Passed.\n\n')

    def _debug_model_outputs(self, predict_net_path, init_net_path, value_info_path, feed_dict_override=None):
        value_info = json_utils.load(value_info_path)

        feed_dict = self._get_random_feed_dict((name, shape, dtype)
                                               for name, [dtype, shape] in six.iteritems(value_info))

        if feed_dict_override:
            feed_dict.update(feed_dict_override)

        outputs = run_caffe2_model(predict_net_path, init_net_path, feed_dict)

        print("Outputs:")
        for output in outputs:
            print(output.shape, output)

    def _test_model_fun(self, model_name, model_fun, inputs=None, input_shape=None, input_dtype=None,
                        test_outputs=True, feed_dict_override=None, can_run=True, can_convert=True):
        from caffe2.python.model_helper import ModelHelper

        if inputs is not None:
            assert utils.is_unique(inputs, key=lambda input: input.name)
            assert input_shape is None
            assert input_dtype is None
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
        if input_dtype is None:
            input_dtype = DTYPE_ID_FLOAT

        numbered_model_name = self._get_network_name(model_name)
        model = ModelHelper(name=numbered_model_name)
        outputs = model_fun(model)
        if outputs is None:
            outputs = []
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        model.net.AddExternalOutputs(*outputs)
        if inputs is None:
            inputs = [Input(str(input), input_shape, input_dtype) for input in model.net.external_inputs]
        paths = self._save_model(dir=os.path.join('out', 'caffe2_orig', numbered_model_name),
                                 predict_net=model.net.Proto(),
                                 init_net=model.param_init_net.Proto(),
                                 value_info={
                                     input.name: [input.dtype, input.shape if input.shape else [1]]
                                     for input in inputs
                                 })
        debug_model_outputs = False
        if debug_model_outputs:
            if can_run:
                self._debug_model_outputs(*paths, feed_dict_override=feed_dict_override)
        else:
            self._test_model(*paths, test_outputs=test_outputs,
                             feed_dict_override=feed_dict_override, can_run=can_run, can_convert=can_convert)

    def _test_layer(self, _op_name, _inputs, _output_count=1,
                    _feed_dict_override=None, _can_run=True, _can_convert=True, **kwargs):
        def model_fun(model):
            def input_to_0d(input_name):
                input_0d, input_old_shape = model.net.Reshape(
                    input_name, [input_name + '_0d', input_name + '_old_shape'], shape=[])
                return input_0d

            input_names = [input.name if input.shape else input_to_0d(input.name) for input in _inputs]
            output_names = ['output' + str(i) for i in range(_output_count)]

            return getattr(model.net, _op_name)(input_names, output_names, **kwargs)

        self._test_model_fun(_op_name, model_fun, _inputs,
                             feed_dict_override=_feed_dict_override, can_run=_can_run, can_convert=_can_convert)

    @staticmethod
    def _get_random_feed_dict(input_name_shape_dtypes):
        feed_dict = {}
        for input_name, input_shape, input_dtype in input_name_shape_dtypes:
            if utils.is_anyint(input_dtype):
                input_dtype = dtype_id_to_name(input_dtype)
            if input_dtype == 'FLOAT':
                feed_dict[input_name] = np.random.random(input_shape).astype(np.float32) - 0.5
            elif input_dtype == 'INT8':
                feed_dict[input_name] = np.random.randint(-128, 128, input_shape, np.int8)
            elif input_dtype == 'UINT8':
                feed_dict[input_name] = np.random.randint(0, 256, input_shape, np.uint8)
            elif input_dtype == 'INT32':
                feed_dict[input_name] = np.random.randint(0, 1000, input_shape, np.int32)
            elif input_dtype == 'BOOL':
                feed_dict[input_name] = np.random.random(input_shape) > 0.5
            else:
                assert False
        return feed_dict

    @staticmethod
    def _save_model(dir, predict_net, init_net, value_info):
        utils.makedirs(dir, exist_ok=True)
        predict_net_path = os.path.join(dir, 'predict_net.pb')
        init_net_path = os.path.join(dir, 'init_net.pb')
        value_info_path = os.path.join(dir, 'value_info.json')
        with open(predict_net_path, 'wb') as f:
            f.write(predict_net.SerializeToString())
        with open(init_net_path, 'wb') as f:
            f.write(init_net.SerializeToString())
        json_utils.dump(value_info, value_info_path, indent=False)
        return predict_net_path, init_net_path, value_info_path
