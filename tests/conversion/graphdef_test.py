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
from nnef_tools.io.tf.graphdef.protobuf import GraphDef
import nnef_tools.io.nnef as nnef_io
import nnef_tools.io.tf.graphdef as graphdef
import nnef_tools.conversion.tf_to_nnef as tf_to_nnef
import nnef_tools.conversion.nnef_to_tf as nnef_to_tf
import nnef_tools.optimization.nnef_optimizer as nnef_opt
import nnef_tools.optimization.tf_optimizer as tf_opt
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

    _network_folder = os.path.join(UNITTEST_FOLDER, 'tf/nets/') if UNITTEST_FOLDER else None
    _output_folder = os.path.join(UNITTEST_FOLDER, 'tf/ops/') if UNITTEST_FOLDER else None
    _io_transpose = True
    _optimize = True

    def setUp(self) -> None:
        self._tf_reader = graphdef.Reader(fold_constants=True)
        self._tf_writer = graphdef.Writer()
        self._tf_optimizer = tf_opt.Optimizer()
        self._nnef_optimizer = nnef_opt.Optimizer()
        self._tf_to_nnef_converter = tf_to_nnef.Converter(io_transpose=self._io_transpose)
        self._nnef_to_tf_converter = nnef_to_tf.Converter(io_transpose=self._io_transpose)
        self._nnef_reader = nnef_io.Reader(custom_shapes=self._nnef_to_tf_converter.defined_shapes(),
                                           decomposed=self._nnef_to_tf_converter.decomposed_operations())
        self._nnef_writer = nnef_io.Writer(fragments=self._tf_to_nnef_converter.defined_operations())

    def tearDown(self) -> None:
        tf.reset_default_graph()

    @staticmethod
    def _save_graph_def(graph_def, filename):
        with open(filename, 'wb') as file:
            file.write(graph_def.SerializeToString())

    @staticmethod
    def _load_graph_def(filename):
        graph_def = GraphDef()
        with open(filename, 'rb') as file:
            graph_def.ParseFromString(file.read())
        return graph_def

    @staticmethod
    def _numpy_dtype(tensor):
        return np.bool_ if tensor.dtype.is_bool else tensor.dtype.as_numpy_dtype()

    @staticmethod
    def _exec_graph_def(graph_def, only_first_output=False):
        np.random.seed(0)

        tf.reset_default_graph()
        tf.import_graph_def(graph_def, name='')
        ops = tf.get_default_graph().get_operations()

        consumed = {tensor for op in ops for tensor in op.inputs}
        inputs = [op.outputs[0] for op in ops if op.type == 'Placeholder']
        if only_first_output:
            outputs = [op.outputs[0] for op in ops if len(op.inputs) and op.outputs[0] not in consumed]
        else:
            outputs = [tensor for op in ops if len(op.inputs) for tensor in op.outputs if tensor not in consumed]
        feed_dict = {tensor: TestEnv._random_data(TestEnv._numpy_dtype(tensor), tensor.shape.as_list())
                     for tensor in inputs}

        with tf.Session() as sess:
            return sess.run(outputs, feed_dict=feed_dict)

    @staticmethod
    def _random_data(dtype, shape):
        if dtype == bool:
            return np.random.random(shape) > 0.5
        else:
            return np.random.random(shape).astype(dtype)

    def _convert_to_nnef(self, filename, input_shapes=None):
        tf_graph = self._tf_reader(filename, input_shapes=input_shapes)
        tf_graph = self._tf_optimizer(tf_graph)
        nnef_graph = self._tf_to_nnef_converter(tf_graph)
        if self._optimize:
            nnef_graph = self._nnef_optimizer(nnef_graph)
        self._nnef_writer(nnef_graph, filename + '.nnef')

    def _convert_from_nnef(self, filename):
        nnef_graph = self._nnef_reader(filename)
        tf_graph = self._nnef_to_tf_converter(nnef_graph)
        self._tf_writer(tf_graph, filename + '.pb')

    def _test_conversion(self, name, only_first_output=True, epsilon=1e-5):
        filename = tempfile.mktemp() if self._output_folder is None else TestEnv._output_folder + name + '.pb'

        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        self._save_graph_def(graph_def, filename)

        self._test_conversion_from_file(filename, only_first_output=only_first_output, epsilon=epsilon)

    def _test_conversion_from_file(self, filename, only_first_output=True, input_shapes=None, epsilon=1e-5):
        self._convert_to_nnef(filename, input_shapes)
        self._convert_from_nnef(filename + '.nnef')

        original_graph_def = self._load_graph_def(filename)
        converted_graph_def = self._load_graph_def(filename + '.nnef.pb')

        if input_shapes is not None:
            original_graph_def = graphdef.set_input_shapes(original_graph_def, input_shapes)

        original_outputs = self._exec_graph_def(original_graph_def, only_first_output)
        converted_outputs = self._exec_graph_def(converted_graph_def, only_first_output)

        self.assertEqual(len(original_outputs), len(converted_outputs))
        for original, converted in zip(original_outputs, converted_outputs):
            if original.dtype == bool:
                self.assertTrue(np.all(original == converted))
            else:
                diff = np.max(np.abs(original - converted))
                self.assertLess(diff, epsilon)


class TestCases(TestEnv):

    def test_conv1d(self):
        input = tf.placeholder(shape=(4, 32, 3), dtype=tf.float32)
        filter = tf.constant(np.random.random(size=(5, 3, 16)), dtype=tf.float32)
        output = tf.nn.conv1d(input, filter, stride=1, padding='SAME')

        self._test_conversion('conv1d')

    def test_conv2d(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        filter = tf.constant(np.random.random(size=(5, 5, 3, 16)), dtype=tf.float32)
        output = tf.nn.conv2d(input, filter, strides=1, padding='SAME')

        self._test_conversion('conv2d')

    def test_conv3d(self):
        input = tf.placeholder(shape=(4, 32, 32, 32, 3), dtype=tf.float32)
        filter = tf.constant(np.random.random(size=(5, 5, 5, 3, 16)), dtype=tf.float32)
        output = tf.nn.conv3d(input, filter, strides=[1, 1, 1, 1, 1], padding='SAME')

        self._test_conversion('conv3d')

    def test_conv2d_explicit_padding(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        filter = tf.constant(np.random.random(size=(5, 5, 3, 16)), dtype=tf.float32)
        output = tf.nn.conv2d(input, filter, strides=1, padding=[(0, 0), (2, 2), (2, 2), (0, 0)])

        self._test_conversion('conv2d-explicit-padding')

    def test_conv2d_transpose(self):
        input = tf.placeholder(shape=(4, 32, 32, 16), dtype=tf.float32)
        filter = tf.constant(np.random.random(size=(5, 5, 3, 16)), dtype=tf.float32)
        output = tf.nn.conv2d_transpose(input, filter, strides=1, padding='SAME', output_shape=(4, 32, 32, 3))

        self._test_conversion('conv2d_transpose')

    def test_depthwise_conv2d(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        filter = tf.constant(np.random.random(size=(5, 5, 3, 2)), dtype=tf.float32)
        output = tf.nn.depthwise_conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

        self._test_conversion('depthwise_conv2d')

    def test_depthwise_conv2d_transpose(self):
        input = tf.placeholder(shape=(4, 32, 32, 6), dtype=tf.float32)
        filter = tf.constant(np.random.random(size=(5, 5, 3, 2)), dtype=tf.float32)
        output = tf.nn.depthwise_conv2d_backprop_input([4, 32, 32, 3], filter, input, strides=[1, 1, 1, 1], padding='SAME')

        self._test_conversion('depthwise_conv2d_transpose')

    def test_conv2d_dilated(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        filter = tf.constant(np.random.random(size=(5, 5, 3, 16)), dtype=tf.float32)
        output = tf.nn.conv2d(input, filter, strides=1, dilations=2, padding='SAME')

        self._test_conversion('conv2d_dilated')

    def test_max_pool2d(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.max_pool2d(input, ksize=3, strides=1, padding='SAME')

        self._test_conversion('max_pool2d')

    def test_max_pool2d_with_index(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        maximum, index = tf.nn.max_pool_with_argmax(input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        self._test_conversion('max_pool2d_with_index')

    def test_avg_pool2d(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.avg_pool2d(input, ksize=3, strides=1, padding='SAME')

        self._test_conversion('avg_pool2d')

    def test_min_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.reduce_min(input, axis=3, keepdims=True)

        self._test_conversion('min_reduce')

    def test_max_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.reduce_max(input, axis=3, keepdims=True)

        self._test_conversion('max_reduce')

    def test_mean_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.reduce_mean(input, axis=3, keepdims=True)

        self._test_conversion('mean_reduce')

    def test_sum_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.reduce_sum(input, axis=3, keepdims=True)

        self._test_conversion('sum_reduce')

    def test_any_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        output = tf.reduce_any(input, axis=3, keepdims=True)

        self._test_conversion('any_reduce')

    def test_all_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        output = tf.reduce_all(input, axis=3, keepdims=True)

        self._test_conversion('all_reduce')

    def test_argmin_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.argmin(input, axis=-1)
        output = tf.expand_dims(output, axis=-1)

        self._test_conversion('axgmin_reduce')

    def test_argmax_reduce(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.argmax(input, axis=-1)
        output = tf.expand_dims(output, axis=-1)

        self._test_conversion('axgmax_reduce')

    def test_concat(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        otuput = tf.concat([input1, input2], axis=3)

        self._test_conversion('concat')

    def test_split_sizes(self):
        input = tf.placeholder(shape=(4, 32, 32, 6), dtype=tf.float32)
        [output1, output2] = tf.split(input, axis=3, num_or_size_splits=[3, 3])

        self._test_conversion('split-sizes')

    def test_split_num(self):
        input = tf.placeholder(shape=(4, 32, 32, 6), dtype=tf.float32)
        [output1, output2] = tf.split(input, axis=3, num_or_size_splits=2)

        self._test_conversion('split-num')

    def test_reshape(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.reshape(input, shape=(4, 32 * 32 * 3))

        self._test_conversion('reshape')

    def test_flatten(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.reshape(input, shape=(4, -1))

        self._test_conversion('flatten')

    def test_squeeze(self):
        input = tf.placeholder(shape=(4, 32, 32, 1), dtype=tf.float32)
        squeezed = tf.squeeze(input, axis=[3])
        output = tf.expand_dims(squeezed, axis=[3])

        self._test_conversion('squeeze')

    def test_squeeze_all(self):
        input = tf.placeholder(shape=(4, 32, 32, 1), dtype=tf.float32)
        squeezed = tf.squeeze(input)
        output = tf.expand_dims(squeezed, axis=[3])

        self._test_conversion('squeeze_all')

    def test_transpose(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        trans = tf.transpose(input, perm=(0, 3, 1, 2))
        output = tf.transpose(trans, perm=(0, 2, 3, 1))

        self._test_conversion('transpose')

    def test_stack(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 1), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 1), dtype=tf.float32)
        input1 = tf.squeeze(input1, axis=3)
        input2 = tf.squeeze(input2, axis=3)
        output = tf.stack([input1, input2], axis=3)

        self._test_conversion('stack')

    def test_unstack(self):
        input = tf.placeholder(shape=(4, 32, 32, 2), dtype=tf.float32)
        [output1, output2] = tf.unstack(input, axis=3)
        output1 = tf.expand_dims(output1, axis=3)
        output2 = tf.expand_dims(output2, axis=3)

        self._test_conversion('unstack')

    def test_add(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.add(input1, input2)

        self._test_conversion('add')

    def test_add_broadcast(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(3,), dtype=tf.float32)
        output = tf.add(input1, input2)

        self._test_conversion('add-broadcast')

    def test_sub(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.subtract(input1, input2)

        self._test_conversion('sub')

    def test_sub_broadcast(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(3,), dtype=tf.float32)
        output = tf.subtract(input1, input2)

        self._test_conversion('sub-broadcast')

    def test_mul(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.multiply(input1, input2)

        self._test_conversion('mul')

    def test_mul_broadcast(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(3,), dtype=tf.float32)
        output = tf.multiply(input1, input2)

        self._test_conversion('mul-broadcast')

    def test_div(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.divide(input1, input2)

        self._test_conversion('div')

    def test_div_boradcast(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(3,), dtype=tf.float32)
        output = tf.divide(input1, input2)

        self._test_conversion('div-broadcast')

    def test_pow(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.pow(input1, input2)

        self._test_conversion('pow')

    def test_min(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.minimum(input1, input2)

        self._test_conversion('min')

    def test_max(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.maximum(input1, input2)

        self._test_conversion('max')

    def test_and(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        output = tf.logical_and(input1, input2)

        self._test_conversion('and')

    def test_or(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        output = tf.logical_or(input1, input2)

        self._test_conversion('or')

    def test_lt(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.less(input1, input2)

        self._test_conversion('lt')

    def test_gt(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.greater(input1, input2)

        self._test_conversion('gt')

    def test_le(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.less_equal(input1, input2)

        self._test_conversion('le')

    def test_ge(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.greater_equal(input1, input2)

        self._test_conversion('ge')

    def test_eq(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.equal(input1, input2)

        self._test_conversion('eq')

    def test_ne(self):
        input1 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        input2 = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.not_equal(input1, input2)

        self._test_conversion('ne')

    def test_identity(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.identity(input)

        self._test_conversion('identity')

    def test_relu(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.relu(input)

        self._test_conversion('relu')

    def test_elu(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.elu(input)

        self._test_conversion('elu')

    def test_selu(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.selu(input)

        self._test_conversion('selu')

    def test_relu6(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.relu6(input)

        self._test_conversion('relu6')

    def test_leaky_relu(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.leaky_relu(input, alpha=0.1)

        self._test_conversion('leaky_relu')

    def test_sigmoid(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.sigmoid(input)

        self._test_conversion('sigmoid')

    def test_softplus(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.softplus(input)

        self._test_conversion('softplus')

    def test_exp(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.exp(input)

        self._test_conversion('exp')

    def test_log(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.log(input)

        self._test_conversion('log')

    def test_sin(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.sin(input)

        self._test_conversion('sin')

    def test_cos(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.cos(input)

        self._test_conversion('cos')

    def test_tan(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.tan(input)

        self._test_conversion('tan')

    def test_sinh(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.sinh(input)

        self._test_conversion('sinh')

    def test_cosh(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.cosh(input)

        self._test_conversion('cosh')

    def test_tanh(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.tanh(input)

        self._test_conversion('tanh')

    def test_sign(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.sign(input)

        self._test_conversion('sign')

    def test_abs(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.abs(input)

        self._test_conversion('abs')

    def test_neg(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.negative(input)

        self._test_conversion('neg')

    def test_rcp(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.reciprocal(input)

        self._test_conversion('rcp')

    def test_floor(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.floor(input)

        self._test_conversion('floor')

    def test_ceil(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.ceil(input)

        self._test_conversion('ceil')

    def test_round(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.round(input)

        self._test_conversion('round')

    def test_sqr(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.square(input)

        self._test_conversion('sqr')

    def test_sqrt(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.sqrt(input)

        self._test_conversion('sqrt')

    def test_rsqrt(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.rsqrt(input)

        self._test_conversion('rsqrt')

    def test_not(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        output = tf.math.logical_not(input)

        self._test_conversion('not')

    def test_cast(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.int32)
        output = tf.cast(input, dtype=tf.float32)

        self._test_conversion('cast')

    def test_cast_ints(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.int32)
        cast = tf.cast(input, dtype=tf.int8)
        output = tf.cast(cast, dtype=tf.int32)

        self._test_conversion('cast_ints')

    def test_cast_float_bool(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.cast(input, dtype=tf.bool)

        self._test_conversion('cast_float_bool')

    def test_select(self):
        cond = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.bool)
        left = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        right = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.where(cond, left, right)

        self._test_conversion('select')

    def test_clamp(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.clip_by_value(input, 0.2, 0.8)

        self._test_conversion('clamp')

    def test_batch_norm(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        mean = tf.constant(np.random.random((3,)), dtype=tf.float32)
        variance = tf.constant(np.random.random((3,)), dtype=tf.float32)
        scale = tf.constant(np.random.random((3,)), dtype=tf.float32)
        offset = tf.constant(np.random.random((3,)), dtype=tf.float32)
        outputs = tf.nn.batch_normalization(input, scale=scale, offset=offset, mean=mean, variance=variance,
                                            variance_epsilon=1e-5)

        self._test_conversion('batch_norm')

    def test_fused_batch_norm(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        mean = tf.constant(np.random.random((3,)), dtype=tf.float32)
        variance = tf.constant(np.random.random((3,)), dtype=tf.float32)
        scale = tf.constant(np.random.random((3,)), dtype=tf.float32)
        offset = tf.constant(np.random.random((3,)), dtype=tf.float32)
        outputs = tf.nn.fused_batch_norm(input, scale=scale, offset=offset, mean=mean, variance=variance,
                                         is_training=False)

        self._test_conversion('fused_batch_norm')

    def test_bias_add(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        bias = tf.constant(np.random.random((3,)), dtype=tf.float32)
        output = tf.nn.bias_add(input, bias)

        self._test_conversion('bias_add')

    def test_softmax(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.nn.softmax(input)

        self._test_conversion('softmax')

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

        self._test_conversion('conv_bias_relu_pool', epsilon=1e-4)

    def test_conv_mul_add(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        filter = tf.constant(np.random.random(size=(5, 5, 3, 16)), dtype=tf.float32)
        bias = tf.constant(np.random.random(size=16, ), dtype=tf.float32)
        scale = tf.constant(np.random.random(size=16, ), dtype=tf.float32)
        offset = tf.constant(np.random.random(size=16, ), dtype=tf.float32)
        filtered = tf.nn.conv2d(input, filter, strides=1, padding='SAME')
        biased = tf.nn.bias_add(filtered, bias)
        scaled = biased * scale
        output = scaled + offset

        self._test_conversion('conv_mul_add', epsilon=1e-4)

    def test_mul_conv(self):
        input = tf.placeholder(shape=(4, 32, 32, 8), dtype=tf.float32)
        filter1 = tf.constant(np.random.random(size=(5, 5, 8, 16)), dtype=tf.float32)
        filter2 = tf.constant(np.random.random(size=(5, 5, 8, 16)), dtype=tf.float32)
        bias1 = tf.constant(np.random.random(size=16, ), dtype=tf.float32)
        bias2 = tf.constant(np.random.random(size=16, ), dtype=tf.float32)
        scale = tf.constant(np.random.random(size=8, ), dtype=tf.float32)
        scaled = input * scale
        filtered1 = tf.nn.conv2d(scaled, filter1, strides=1, padding='SAME')
        filtered2 = tf.nn.conv2d(scaled, filter2, strides=1, padding='SAME')
        biased1 = tf.nn.bias_add(filtered1, bias1)
        biased2 = tf.nn.bias_add(filtered2, bias2)

        self._test_conversion('mul_conv', epsilon=1e-4)

    def test_matmul(self):
        input1 = tf.placeholder(shape=(10, 100), dtype=tf.float32)
        input2 = tf.placeholder(shape=(100, 20), dtype=tf.float32)
        output = tf.matmul(input1, input2)

        self._test_conversion('matmul')

    def test_matmul_trans(self):
        input1 = tf.placeholder(shape=(10, 100), dtype=tf.float32)
        input2 = tf.placeholder(shape=(20, 100), dtype=tf.float32)
        output = tf.matmul(input1, input2, transpose_b=True)

        self._test_conversion('matmul-trans')

    def test_pad(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.pad(input, paddings=[[0, 0], [1, 2], [1, 2], [0, 0]])

        self._test_conversion('pad')

    def test_pad_mirror(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.pad(input, paddings=[[0, 0], [1, 2], [1, 2], [0, 0]], mode='REFLECT')

        self._test_conversion('pad_reflect')

    def test_pad_symmetric(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.pad(input, paddings=[[0, 0], [1, 2], [1, 2], [0, 0]], mode='SYMMETRIC')

        self._test_conversion('pad_symmetric')

    def test_slice(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.slice(input, begin=[0, 1, 1, 0], size=[4, 30, 30, 3])

        self._test_conversion('slice')

    def test_strided_slice(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = input[:, 1:-1, 1:-1, :]

        self._test_conversion('strided_slice')

    def test_strided_slice_shrink_axis(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = input[:, 1:-1, 1:-1, 1]
        output = tf.expand_dims(output, axis=3)

        self._test_conversion('strided_slice-shrink_axis')

    def test_strided_slice_new_axis(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = input[:, 1:-1, 1:-1, tf.newaxis, :]
        output = tf.squeeze(output, axis=3)

        self._test_conversion('strided_slice-new_axis')

    def test_strided_slice_flip(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = input[:, -2:0:-1, -2:0:-1, :]

        self._test_conversion('strided_slice-flip')

    def test_tile(self):
        input = tf.placeholder(shape=(4, 1, 1, 3), dtype=tf.float32)
        output = tf.tile(input, multiples=(1, 32, 32, 1))

        self._test_conversion('tile')

    def test_gather(self):
        input = tf.placeholder(shape=(4, 32, 32, 16), dtype=tf.float32)
        indices = tf.constant(np.random.random_integers(size=(24,), low=0, high=15), dtype=tf.int32)
        output = tf.gather(input, indices, axis=3)

        self._test_conversion('gather')

    def test_upsample_nearest(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.image.resize_nearest_neighbor(input, size=(64, 64))

        self._test_conversion('upsample-nearest')

    def test_downsample_nearest(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.image.resize_nearest_neighbor(input, size=(16, 16))

        self._test_conversion('downsample-nearest')

    def test_downsample_area(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.image.resize_area(input, size=(16, 16))

        self._test_conversion('downsample-area')

    def test_upsample_linear(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.image.resize_bilinear(input, size=(64, 64))

        self._test_conversion('upsample-linear')

    def test_lrn(self):
        input = tf.placeholder(shape=(4, 32, 32, 8), dtype=tf.float32)
        output = tf.nn.local_response_normalization(input, depth_radius=2)

        self._test_conversion('lrn')

    def test_l2_normalize(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.math.l2_normalize(input, axis=-1)

        self._test_conversion('l2_normalize')

    def test_add_n(self):
        input = tf.placeholder(shape=(4, 32, 32, 3), dtype=tf.float32)
        output = tf.add_n([input, input, input])

        self._test_conversion('add_n')


@unittest.skipIf(TestEnv._network_folder is None or not os.path.isdir(TestEnv._network_folder),
                 "no network test folder provided")
class NetworkTestCases(TestEnv):

    def test_mobilenet_v1(self):
        self._test_conversion_from_file(self._network_folder + 'mobilenet_v1.pb',
                                        input_shapes={'input': (1, 224, 224, 3)})

    def test_mobilenet_v2(self):
        self._test_conversion_from_file(self._network_folder + 'mobilenet_v2.pb',
                                        input_shapes={'input': (1, 224, 224, 3)})

    def test_inception_v3(self):
        self._test_conversion_from_file(self._network_folder + 'inception_v3.pb',
                                        input_shapes={'input': (1, 299, 299, 3)})

    def test_inception_v4(self):
        self._test_conversion_from_file(self._network_folder + 'inception_v4.pb',
                                        input_shapes={'input': (1, 299, 299, 3)})

    def test_inception_resnet_v2(self):
        self._test_conversion_from_file(self._network_folder + 'inception_resnet_v2.pb',
                                        input_shapes={'input': (1, 299, 299, 3)})

    def test_squeezenet(self):
        self._test_conversion_from_file(self._network_folder + 'squeezenet.pb',
                                        input_shapes={'Placeholder': (1, 224, 224, 3)})

    def test_nasnet(self):
        self._test_conversion_from_file(self._network_folder + 'nasnet_mobile.pb',
                                        input_shapes={'input': (1, 224, 224, 3)})


if __name__ == '__main__':
    unittest.main()
