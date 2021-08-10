# Copyright (c) 2020 The Khronos Group Inc.
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

import numpy as np
import nnef_tools.io.nnef as nnef_io
import nnef_tools.io.tf.lite as lite_io
import nnef_tools.conversion.tflite_to_nnef as tflite_to_nnef
import nnef_tools.conversion.nnef_to_tflite as nnef_to_tflite
import nnef_tools.optimization.nnef_optimizer as nnef_opt
import nnef_tools.optimization.tflite_optimizer as tflite_opt
import unittest
import tempfile
import os
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf


UNITTEST_FOLDER = os.environ.get('UNITTEST_FOLDER')


class TestEnv(unittest.TestCase):

    _network_folder = os.path.join(UNITTEST_FOLDER, 'tflite/nets/') if UNITTEST_FOLDER else None
    _output_folder = os.path.join(UNITTEST_FOLDER, 'tflite/ops/') if UNITTEST_FOLDER else None
    _mirror_unsupported = False
    _io_transpose = True
    _optimize = True

    def setUp(self) -> None:
        self._tflite_reader = lite_io.Reader()
        self._tflite_writer = lite_io.Writer()
        self._tflite_to_nnef_converter = tflite_to_nnef.Converter(io_transpose=self._io_transpose,
                                                                  mirror_unsupported=self._mirror_unsupported)
        self._nnef_to_tflite_converter = nnef_to_tflite.Converter(io_transpose=self._io_transpose,
                                                                  mirror_unsupported=self._mirror_unsupported)
        self._nnef_reader = nnef_io.Reader(custom_shapes=self._nnef_to_tflite_converter.defined_shapes(),
                                           decomposed=self._nnef_to_tflite_converter.decomposed_operations())
        self._nnef_writer = nnef_io.Writer(fragments=self._tflite_to_nnef_converter.defined_operations())
        self._nnef_optimizer = nnef_opt.Optimizer()
        self._tflite_optimizer = tflite_opt.Optimizer()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def _convert_to_nnef(self, filename):
        tflite_graph = self._tflite_reader(filename)
        if self._optimize:
            tflite_graph = self._tflite_optimizer(tflite_graph)
        nnef_graph = self._tflite_to_nnef_converter(tflite_graph)
        if self._optimize:
            nnef_graph = self._nnef_optimizer(nnef_graph)
        self._nnef_writer(nnef_graph, filename + '.nnef')

    def _convert_from_nnef(self, filename):
        nnef_graph = self._nnef_reader(filename)
        tflite_graph = self._nnef_to_tflite_converter(nnef_graph)
        self._tflite_writer(tflite_graph, filename + '.tflite')

    def _save_default_graph(self, inputs, outputs, filename):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            converter = tf.lite.TFLiteConverter.from_session(sess, inputs, outputs)
            tflite_model = converter.convert()
            with open(filename, "wb") as file:
                file.write(tflite_model)

    @staticmethod
    def _exec_model(model_path):
        np.random.seed(0)

        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        for input in interpreter.get_input_details():
            shape = input['shape']
            dtype = input['dtype']
            data = TestEnv._random_data(dtype, shape)
            interpreter.set_tensor(input['index'], data)

        interpreter.invoke()

        return [TestEnv._dequantize(interpreter.get_tensor(output['index']), *output['quantization'])
                for output in interpreter.get_output_details()]

    @staticmethod
    def _dequantize(data, scale, zero_point):
        return scale * (data - zero_point) if scale else data

    @staticmethod
    def _random_data(dtype, shape):
        if dtype == np.bool:
            return np.random.random(shape) > 0.5
        elif np.issubdtype(dtype, np.integer):
            return np.maximum(np.floor(np.random.random(shape) * 256).astype(dtype), 255)
        else:
            return np.random.random(shape).astype(dtype)

    def _test_conversion(self, name, inputs, outputs, epsilon=1e-5):
        filename = tempfile.mktemp() if self._output_folder is None else self._output_folder + name + '.tflite'
        self._save_default_graph(inputs, outputs, filename)
        self._test_conversion_from_file(filename, epsilon=epsilon)

    def _test_conversion_from_file(self, filename, epsilon=1e-5):
        self._convert_to_nnef(filename)
        self._convert_from_nnef(filename + '.nnef')

        original_outputs = self._exec_model(filename)
        converted_outputs = self._exec_model(filename + '.nnef.tflite')

        self.assertEqual(len(original_outputs), len(converted_outputs))
        for original, converted in zip(original_outputs, converted_outputs):
            if original.dtype == np.bool:
                self.assertTrue(np.all(original == converted))
            else:
                diff = np.max(np.abs(original - converted))
                self.assertLess(diff, epsilon)


class TestCases(TestEnv):

    def test_conv1d(self):
        input = tf.placeholder(shape=(4, 32, 3), dtype=tf.float32)
        filter = tf.constant(np.random.random(size=(5, 3, 16)), dtype=tf.float32)
        output = tf.nn.conv1d(input, filter, stride=1, padding='SAME')

        self._test_conversion('conv1d', [input], [output])

    def test_conv2d(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        filter = tf.constant(np.random.random(size=(5, 5, 3, 16)), dtype=tf.float32)
        output = tf.nn.conv2d(input, filter, strides=1, padding='SAME')

        self._test_conversion('conv2d', [input], [output])

    def test_conv2d_transpose(self):
        input = tf.placeholder(shape=(4, 32, 32, 16), dtype=tf.float32)
        filter = tf.constant(np.random.random(size=(5, 5, 3, 16)), dtype=tf.float32)
        output = tf.nn.conv2d_transpose(input, filter, strides=1, padding='SAME', output_shape=(4, 32, 32, 3))

        self._test_conversion('conv2d_transpose', [input], [output])

    def test_depthwise_conv2d(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        filter = tf.constant(np.random.random(size=(5, 5, 3, 2)), dtype=tf.float32)
        output = tf.nn.depthwise_conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

        self._test_conversion('depthwise_conv2d', [input], [output])

    def test_max_pool2d(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.max_pool2d(input, ksize=3, strides=1, padding='SAME')

        self._test_conversion('max_pool2d', [input], [output])

    def test_avg_pool2d(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.avg_pool2d(input, ksize=3, strides=1, padding='SAME')

        self._test_conversion('avg_pool2d', [input], [output])

    def test_reshape(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.reshape(input, shape=(4, 32 * 32 * 3))

        self._test_conversion('reshape', [input], [output])

    def test_flatten(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.reshape(input, shape=(4, -1))

        self._test_conversion('flatten', [input], [output])

    def test_squeeze(self):
        input = tf.placeholder(shape=(4, 32, 32, 1), dtype=tf.float32)
        squeezed = tf.squeeze(input, axis=[3])
        output = tf.expand_dims(squeezed, axis=[3])

        self._test_conversion('squeeze', [input], [output])

    def test_transpose(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        trans = tf.transpose(input, perm=(0, 3, 1, 2))
        output = tf.transpose(trans, perm=(0, 2, 3, 1))

        self._test_conversion('transpose', [input], [output])

    def test_concat(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.concat([input1, input2], axis=3)

        self._test_conversion('concat', [input1, input2], [output])

    def test_split_sizes(self):
        input = tf.placeholder(shape=(4, 32, 32, 6), dtype=tf.float32)
        [output1, output2] = tf.split(input, axis=3, num_or_size_splits=[3, 3])

        self._test_conversion('split-sizes', [input], [output1, output2])

    def test_split_num(self):
        input = tf.placeholder(shape=(4, 32, 32, 6), dtype=tf.float32)
        [output1, output2] = tf.split(input, axis=3, num_or_size_splits=2)

        self._test_conversion('split-num', [input], [output1, output2])

    def test_pad(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.pad(input, paddings=[[0, 0], [1, 2], [1, 2], [0, 0]])

        self._test_conversion('pad', [input], [output])

    def test_pad_reflect(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.pad(input, paddings=[[0, 0], [1, 2], [1, 2], [0, 0]], mode='REFLECT')

        self._test_conversion('pad_reflect', [input], [output])

    def test_pad_symmetric(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.pad(input, paddings=[[0, 0], [1, 2], [1, 2], [0, 0]], mode='SYMMETRIC')

        self._test_conversion('pad_symmetric', [input], [output])

    def test_tile(self):
        input = tf.placeholder(shape=(4, 1, 1, 3), dtype=tf.float32)
        output = tf.tile(input, multiples=(1, 32, 32, 1))

        self._test_conversion('tile', [input], [output])

    def test_slice(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.slice(input, begin=[0, 1, 1, 0], size=[4, 30, 30, 3])

        self._test_conversion('slice', [input], [output])

    def test_strided_slice(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = input[:, 1:-1, 1:-1, :]

        self._test_conversion('strided_slice', [input], [output])

    def test_strided_slice_flip(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = input[:, -2:0:-1, -2:0:-1, :]

        self._test_conversion('strided_slice_flip', [input], [output])

    def test_gather(self):
        input = tf.placeholder(shape=(4, 32, 32, 16), dtype=tf.float32)
        indices = tf.constant(np.random.random_integers(size=(24,), low=0, high=15), dtype=tf.int32)
        output = tf.gather(input, indices, axis=3)

        self._test_conversion('gather', [input], [output])

    def test_identity(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.identity(input)

        self._test_conversion('identity', [input], [output])

    def test_relu(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.relu(input)

        self._test_conversion('relu', [input], [output])

    def test_relu6(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.relu6(input)

        self._test_conversion('relu6', [input], [output])

    def test_elu(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.elu(input)

        self._test_conversion('elu', [input], [output])

    def test_sigmoid(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.sigmoid(input)

        self._test_conversion('sigmoid', [input], [output])

    def test_tanh(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.tanh(input)

        self._test_conversion('tanh', [input], [output])

    def test_sin(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.sin(input)

        self._test_conversion('sin', [input], [output])

    def test_cos(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.cos(input)

        self._test_conversion('cos', [input], [output])

    def test_log(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.log(input)

        self._test_conversion('log', [input], [output])

    def test_exp(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.exp(input)

        self._test_conversion('exp', [input], [output])

    def test_neg(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.negative(input)

        self._test_conversion('neg', [input], [output])

    def test_floor(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.floor(input)

        self._test_conversion('floor', [input], [output])

    def test_ceil(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.ceil(input)

        self._test_conversion('ceil', [input], [output])

    def test_round(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.round(input)

        self._test_conversion('round', [input], [output])

    def test_sqr(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.square(input)

        self._test_conversion('sqr', [input], [output])

    def test_sqrt(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.sqrt(input)

        self._test_conversion('sqrt', [input], [output])

    def test_rsqrt(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.rsqrt(input)

        self._test_conversion('rsqrt', [input], [output])

    def test_not(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        output = tf.math.logical_not(input)

        self._test_conversion('not', [input], [output])

    def test_abs(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.abs(input)

        self._test_conversion('abs', [input], [output])

    def test_leaky_relu(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.leaky_relu(input, alpha=0.1)

        self._test_conversion('leaky_relu', [input], [output])

    def test_add(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.add(input1, input2)

        self._test_conversion('add', [input1, input2], [output])

    def test_sub(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.subtract(input1, input2)

        self._test_conversion('sub', [input1, input2], [output])

    def test_mul(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.multiply(input1, input2)

        self._test_conversion('mul', [input1, input2], [output])

    def test_div(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.divide(input1, input2)

        self._test_conversion('div', [input1, input2], [output])

    def test_pow(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.pow(input1, input2)

        self._test_conversion('pow', [input1, input2], [output])

    def test_min(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.minimum(input1, input2)

        self._test_conversion('min', [input1, input2], [output])

    def test_max(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.maximum(input1, input2)

        self._test_conversion('max', [input1, input2], [output])

    def test_and(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        output = tf.logical_and(input1, input2)

        self._test_conversion('and', [input1, input2], [output])

    def test_or(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        output = tf.logical_or(input1, input2)

        self._test_conversion('or', [input1, input2], [output])

    def test_lt(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.less(input1, input2)

        self._test_conversion('lt', [input1, input2], [output])

    def test_le(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.less_equal(input1, input2)

        self._test_conversion('le', [input1, input2], [output])

    def test_gt(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.greater(input1, input2)

        self._test_conversion('gt', [input1, input2], [output])

    def test_ge(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.greater_equal(input1, input2)

        self._test_conversion('ge', [input1, input2], [output])

    def test_eq(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.equal(input1, input2)

        self._test_conversion('eq', [input1, input2], [output])

    def test_ne(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.not_equal(input1, input2)

        self._test_conversion('ne', [input1, input2], [output])

    def test_min_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.reduce_min(input, axis=3, keepdims=True)

        self._test_conversion('min_reduce', [input], [output])

    def test_max_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.reduce_max(input, axis=3, keepdims=True)

        self._test_conversion('max_reduce', [input], [output])

    def test_mean_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.reduce_mean(input, axis=3, keepdims=True)

        self._test_conversion('mean_reduce', [input], [output])

    def test_sum_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.reduce_sum(input, axis=3, keepdims=True)

        self._test_conversion('sum_reduce', [input], [output])

    def test_any_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        output = tf.reduce_any(input, axis=3, keepdims=True)

        self._test_conversion('any_reduce', [input], [output])

    def test_argmin_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.argmin(input, axis=-1)

        self._test_conversion('axgmin_reduce', [input], [output])

    def test_argmax_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.argmax(input, axis=-1)

        self._test_conversion('axgmax_reduce', [input], [output])

    def test_stack(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 1), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 1), dtype=tf.float32)
        input1 = tf.squeeze(input1, axis=3)
        input2 = tf.squeeze(input2, axis=3)
        output = tf.stack([input1, input2], axis=3)

        self._test_conversion('stack', [input1, input2], [output])

    def test_unstack(self):
        input = tf.placeholder(shape=(4, 32, 32, 2), dtype=tf.float32)
        [output1, output2] = tf.unstack(input, axis=3)
        output1 = tf.expand_dims(output1, axis=3)
        output2 = tf.expand_dims(output2, axis=3)

        self._test_conversion('unstack', [input], [output1, output2])

    def test_conv_bias_relu_pool(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        filter = tf.constant(np.random.random(size=(5, 5, 3, 16)), dtype=tf.float32)
        bias = tf.constant(np.random.random(size=16,), dtype=tf.float32)
        mean = tf.constant(np.random.random(size=16, ), dtype=tf.float32)
        variance = tf.constant(np.random.random(size=16, ), dtype=tf.float32)
        scale = tf.constant(np.random.random(size=16, ), dtype=tf.float32)
        offset = tf.constant(np.random.random(size=16, ), dtype=tf.float32)
        filtered = tf.nn.conv2d(input, filter, strides=1, padding='SAME')
        biased = tf.nn.bias_add(filtered, bias)
        normed, _mean, _variance = tf.nn.fused_batch_norm(biased, scale, offset, mean, variance, is_training=False)
        relu = tf.nn.relu(normed)
        pooled = tf.nn.max_pool2d(relu, ksize=2, strides=2, padding='SAME')

        self._test_conversion('conv_bias_relu_pool', [input], [pooled])

    def test_softmax(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.softmax(input)

        self._test_conversion('softmax', [input], [output])

    def test_matmul(self):
        input1 = tf.placeholder(shape=(10, 100), dtype=tf.float32)
        input2 = tf.placeholder(shape=(100, 20), dtype=tf.float32)
        output = tf.matmul(input1, input2)

        self._test_conversion('matmul', [input1, input2], [output])

    def test_matmul_trans(self):
        input1 = tf.placeholder(shape=(10, 100), dtype=tf.float32)
        input2 = tf.placeholder(shape=(20, 100), dtype=tf.float32)
        output = tf.matmul(input1, input2, transpose_b=True)

        self._test_conversion('matmul-trans', [input1, input2], [output])

    def test_l2_normalize(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.l2_normalize(input, axis=-1)

        self._test_conversion('l2_normalize', [input], [output])

    def test_lrn(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.local_response_normalization(input, depth_radius=3)

        self._test_conversion('lrn', [input], [output])

    def test_upsample_nearest(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.image.resize_nearest_neighbor(input, size=(64, 64))

        self._test_conversion('upsample-nearest', [input], [output])

    def test_downsample_nearest(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.image.resize_nearest_neighbor(input, size=(16, 16))

        self._test_conversion('downsample-nearest', [input], [output])

    def test_upsample_linear(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.image.resize_bilinear(input, size=(64, 64))

        self._test_conversion('upsample-linear', [input], [output])

    def test_select(self):
        cond = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        left = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        right = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.where(cond, left, right)

        self._test_conversion('select', [cond, left, right], [output])

    def test_batch_norm(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        mean = tf.constant(np.random.random((3,)), dtype=tf.float32)
        variance = tf.constant(np.random.random((3,)), dtype=tf.float32)
        scale = tf.constant(np.random.random((3,)), dtype=tf.float32)
        offset = tf.constant(np.random.random((3,)), dtype=tf.float32)
        output = tf.nn.batch_normalization(input, scale=scale, offset=offset, mean=mean, variance=variance,
                                           variance_epsilon=1e-5)

        self._test_conversion('batch_norm', [input], [output])

    def test_add_n(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.add_n([input, input, input])

        self._test_conversion('add_n', [input], [output])

    def test_cast(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.int32)
        output = tf.cast(input, tf.float32)

        self._test_conversion('cast', [input], [output])


@unittest.skipIf(TestEnv._network_folder is None or not os.path.isdir(TestEnv._network_folder),
                 "no network test folder provided")
class NetworkTestCases(TestEnv):

    def test_inception_v1(self):
        self._test_conversion_from_file(self._network_folder + 'inception_v1.tflite')

    def test_inception_v2(self):
        self._test_conversion_from_file(self._network_folder + 'inception_v2.tflite')

    def test_inception_v3(self):
        self._test_conversion_from_file(self._network_folder + 'inception_v3.tflite')

    def test_inception_v4(self):
        self._test_conversion_from_file(self._network_folder + 'inception_v4.tflite')

    def test_mobilenet_v1(self):
        self._test_conversion_from_file(self._network_folder + 'mobilenet_v1.tflite')

    def test_mobilenet_v2(self):
        self._test_conversion_from_file(self._network_folder + 'mobilenet_v2.tflite')
