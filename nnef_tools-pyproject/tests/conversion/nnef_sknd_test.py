# Copyright (c) 2017-2025 The Khronos Group Inc.
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

from nnef_tools.io import nnef as nnef_io
from nnef_tools.io import skriptnd as skriptnd_io
from nnef_tools.conversion import nnef_to_sknd
from nnef_tools.optimization import skriptnd_optimizer as sknd_optimizer
from nnef_tools.optimization import nnef_optimizer
import sknd_test
import numpy as np
import nnef
import os


UNITTEST_FOLDER = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../unittest/'))


class TestEnv(sknd_test.TestEnv):

    _network_folder = os.path.join(UNITTEST_FOLDER, 'nnef2/nnef/nets/') if UNITTEST_FOLDER else None
    _output_folder = os.path.join(UNITTEST_FOLDER, 'nnef2/nnef/ops/') if UNITTEST_FOLDER else None

    _NumpyDtype = {
        'scalar': np.float32,
        'integer': np.int32,
        'logical': np.bool_,
    }

    _DecomposedForConversion = [
        'separable_conv',
        'separable_deconv',
        'local_mean_normalization',
        'local_variance_normalization',
        'local_contrast_normalization',
    ]

    _DecomposedForExecution = [
        'rms_pool',
        'clamp',
        'prelu',
        'leaky_relu',
        'separable_conv',
        'separable_deconv',
        'local_response_normalization',
        'local_mean_normalization',
        'local_variance_normalization',
        'local_contrast_normalization',
        'l1_normalization',
        'l2_normalization',
        'batch_normalization',
        'nearest_downsample',
        'nearest_upsample',
        'area_downsample',
    ]

    def setUp(self) -> None:
        self._nnef_reader = nnef_io.Reader(decomposed=self._DecomposedForConversion)
        self._nnef_writer = nnef_io.Writer()
        self._nnef_to_sknd_converter = nnef_to_sknd.Converter()
        self._sknd_reader = skriptnd_io.Reader(atomic=lambda name: not name.startswith('main.'))
        self._sknd_writer = skriptnd_io.Writer(operators=nnef_to_sknd.Converter.defined_operations(),
                                               inline_subgraphs=False)
        self._sknd_optimizer = sknd_optimizer.Optimizer(optimize_batch_norm=False)
        self._nnef_optimizer = nnef_optimizer.Optimizer()
        self._optimize = True
        self._execute = True

    def tearDown(self) -> None:
        pass

    def _convert_to_sknd(self, filename, input_shape=None):
        nnef_model = self._nnef_reader(filename)
        sknd_model = self._nnef_to_sknd_converter(nnef_model)
        if input_shape is not None:
            self._set_max_input_shapes(sknd_model, input_shape)
        output_filename = filename + '.nnef2'
        self._sknd_writer(sknd_model, output_filename)
        if self._optimize:
            sknd_model = self._sknd_reader(output_filename)
            self._sknd_optimizer(sknd_model)
            self._sknd_writer(sknd_model, output_filename)

    @staticmethod
    def _exec_orig_model(filename, input_shape=None, input_range=None):
        np.random.seed(0)

        graph = nnef.load_graph(filename, load_variables=False)
        nnef.infer_shapes(graph)

        if not isinstance(input_shape, list):
            input_shape = [input_shape] * len(graph.inputs)
        if not isinstance(input_range, list):
            input_range = [input_range] * len(graph.inputs)

        input_tensors = [graph.tensors[iden] for iden in graph.inputs]

        inputs = [TestEnv._random_data(TestEnv._NumpyDtype[input.dtype],
                                                   input_shape[idx] or input.shape, input_range[idx])
                  for idx, input in enumerate(input_tensors)]

        with nnef.Session(filename, lowered=TestEnv._DecomposedForExecution) as session:
            return session(*inputs)

    def _test_conversion(self, name, code, epsilon=1e-5, input_range=None, execute=True, compile=True):
        path = self._output_folder + name + '.nnef'
        if not os.path.exists(path):
            os.makedirs(path)

        code = 'version 1.0;\n' + code

        filename = path + '/graph.nnef'
        with open(filename, 'w') as file:
            print(code, file=file)

        graph = nnef.parse_string(code)
        nnef.infer_shapes(graph)

        for op in graph.operations:
            if op.name == 'variable':
                tensor = graph.tensors[op.outputs['output']]
                filename = path + '/' + op.attribs['label'] + '.dat'
                data = self._random_data(self._NumpyDtype[tensor.dtype], tensor.shape)
                with open(filename, 'wb') as file:
                    nnef.write_tensor(file, data)

        self._test_conversion_from_file(path, epsilon=epsilon, input_range=input_range, execute=execute, compile=compile)

    def _test_unary(self, name, dtype='scalar', epsilon=1e-5, input_range=None):
        code = """
        graph G(input) -> (output)
        {{
            input = external<{dtype}>(shape = [1, 16, 32, 32]);
            output = {name}(input);
        }}
        """.format(name=name, dtype=dtype)

        self._test_conversion(name, code, epsilon=epsilon, input_range=input_range)

    def _test_binary(self, name, dtype='scalar', epsilon=1e-5, input_range=None):
        code = """
        graph G(input1, input2) -> (output)
        {{
            input1 = external<{dtype}>(shape = [1, 16, 32, 32]);
            input2 = external<{dtype}>(shape = [1, 16, 32, 32]);
            output = {name}(input1, input2);
        }}
        """.format(name=name, dtype=dtype)

        self._test_conversion(name, code, epsilon=epsilon, input_range=input_range)

    def _test_reduce(self, name, dtype='scalar', epsilon=1e-5, input_range=None):
        code = """
        graph G(input) -> (output)
        {{
            input = external<{dtype}>(shape = [1, 16, 32, 32]);
            output = {name}(input, axes = [2, 3]);
        }}
        """.format(name=name, dtype=dtype)

        self._test_conversion(name, code, epsilon=epsilon, input_range=input_range)


class TestCases(TestEnv):

    def test_conv1d(self):
        code = """
        graph G(input) -> (output)
        {
            filter = variable<scalar>(shape = [16, 3, 5], label = 'filter');
            bias = variable<scalar>(shape = [1, 16], label = 'bias');
            input = external<scalar>(shape = [1, 3, 32]);
            output = conv(input, filter, bias, stride = [1], dilation = [1], padding = []);
        }
        """

        self._test_conversion('conv1d', code)

    def test_conv2d(self):
        code = """
        graph G(input) -> (output)
        {
            filter = variable<scalar>(shape = [16, 3, 5, 5], label = 'filter');
            bias = variable<scalar>(shape = [1, 16], label = 'bias');
            input = external<scalar>(shape = [1, 3, 32, 32]);
            output = conv(input, filter, bias, padding = []);
        }
        """

        self._test_conversion('conv2d', code)

    def test_conv2d_padded(self):
        code = """
        graph G(input) -> (output)
        {
            filter = variable<scalar>(shape = [16, 3, 5, 5], label = 'filter');
            bias = variable<scalar>(shape = [1, 16], label = 'bias');
            input = external<scalar>(shape = [1, 3, 32, 32]);
            output = conv(input, filter, bias, padding = [(2, 2), (2, 2)]);
        }
        """

        self._test_conversion('conv2d_padded', code)

    def test_conv2d_valid(self):
        code = """
        graph G(input) -> (output)
        {
            filter = variable<scalar>(shape = [16, 3, 5, 5], label = 'filter');
            bias = variable<scalar>(shape = [1, 16], label = 'bias');
            input = external<scalar>(shape = [1, 3, 32, 32]);
            output = conv(input, filter, bias, padding = [(0, 0), (0, 0)]);
        }
        """

        self._test_conversion('conv2d_valid', code)

    def test_conv2d_strided(self):
        code = """
        graph G(input) -> (output)
        {
            filter = variable<scalar>(shape = [16, 3, 5, 5], label = 'filter');
            bias = variable<scalar>(shape = [1, 16], label = 'bias');
            input = external<scalar>(shape = [1, 3, 32, 32]);
            output = conv(input, filter, bias, stride = [2, 2]);
        }
        """

        self._test_conversion('conv2d_strided', code)

    def test_conv2d_nobias(self):
        code = """
        graph G(input) -> (output)
        {
            filter = variable<scalar>(shape = [16, 3, 5, 5], label = 'filter');
            input = external<scalar>(shape = [1, 3, 32, 32]);
            output = conv(input, filter);
        }
        """

        self._test_conversion('conv2d_nobias', code)

    def test_deconv2d(self):
        code = """
        graph G(input) -> (output)
        {
            filter = variable<scalar>(shape = [16, 3, 5, 5], label = 'filter');
            bias = variable<scalar>(shape = [1, 3], label = 'bias');
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = deconv(input, filter, bias, padding = []);
        }
        """

        self._test_conversion('deconv2d', code)

    def test_separable_conv2d(self):
        code = """
        graph G(input) -> (output)
        {
            plane_filter = variable<scalar>(shape = [3, 1, 5, 5], label = 'plane_filter');
            point_filter = variable<scalar>(shape = [16, 3, 1, 1], label = 'point_filter');
            bias = variable<scalar>(shape = [1, 16], label = 'bias');
            input = external<scalar>(shape = [1, 3, 32, 32]);
            output = separable_conv(input, plane_filter, point_filter, bias);
        }
        """

        self._test_conversion('separable_conv2d', code)

    def test_separable_deconv2d(self):
        code = """
        graph G(input) -> (output)
        {
            plane_filter = variable<scalar>(shape = [3, 1, 5, 5], label = 'plane_filter');
            point_filter = variable<scalar>(shape = [16, 3, 1, 1], label = 'point_filter');
            bias = variable<scalar>(shape = [1, 3], label = 'bias');
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = separable_deconv(input, plane_filter, point_filter, bias);
        }
        """

        self._test_conversion('separable_deconv2d', code)

    def test_box(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = box(input, size = [1, 1, 3, 3]);
        }
        """

        self._test_conversion('box', code)

    def test_box_normalize(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = box(input, size = [1, 1, 3, 3], normalize = true);
        }
        """

        self._test_conversion('box_normalize', code)

    def test_box_strided(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = box(input, size = [1, 1, 2, 2], stride = [1, 1, 2, 2]);
        }
        """

        self._test_conversion('box_strided', code)

    def test_avg_pool(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = avg_pool(input, size = [1, 1, 3, 3]);
        }
        """

        self._test_conversion('avg_pool', code)

    def test_avg_pool_strided(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = avg_pool(input, size = [1, 1, 3, 3], stride = [1, 1, 2, 2]);
        }
        """

        self._test_conversion('avg_pool_strided', code)

    def test_avg_pool_padded(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = avg_pool(input, size = [1, 1, 3, 3], padding = [(1, 1), (1, 1), (1, 1), (1, 1)]);
        }
        """

        self._test_conversion('avg_pool_padded', code)

    def test_avg_pool_ignore_border(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = avg_pool(input, size = [1, 1, 3, 3], border = 'ignore');
        }
        """

        self._test_conversion('avg_pool_ignore_border', code)

    def test_max_pool(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = max_pool(input, size = [1, 1, 3, 3]);
        }
        """

        self._test_conversion('max_pool', code)

    def test_max_pool_strided(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = max_pool(input, size = [1, 1, 3, 3], stride = [1, 1, 2, 2]);
        }
        """

        self._test_conversion('max_pool_strided', code)

    def test_rms_pool(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = rms_pool(input, size = [1, 1, 3, 3]);
        }
        """

        self._test_conversion('rms_pool', code)

    def test_rms_pool_ignore_border(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = rms_pool(input, size = [1, 1, 3, 3], border = 'ignore');
        }
        """

        self._test_conversion('rms_pool_ignore_border', code)

    def test_copy(self):
        self._test_unary('copy')

    def test_neg(self):
        self._test_unary('neg')

    def test_rcp(self):
        self._test_unary('rcp')

    def test_sqr(self):
        self._test_unary('sqr')

    def test_sqrt(self):
        self._test_unary('sqrt')

    def test_rsqr(self):
        self._test_unary('rsqr')

    def test_rsqrt(self):
        self._test_unary('rsqrt')

    def test_exp(self):
        self._test_unary('exp')

    def test_log(self):
        self._test_unary('log')

    def test_log2(self):
        self._test_unary('log2')

    def test_abs(self):
        self._test_unary('abs')

    def test_sign(self):
        self._test_unary('sign')

    def test_sin(self):
        self._test_unary('sin')

    def test_cos(self):
        self._test_unary('cos')

    def test_tan(self):
        self._test_unary('tan')

    def test_asin(self):
        self._test_unary('asin')

    def test_acos(self):
        self._test_unary('acos')

    def test_atan(self):
        self._test_unary('atan')

    def test_sinh(self):
        self._test_unary('sinh')

    def test_cosh(self):
        self._test_unary('cosh')

    def test_tanh(self):
        self._test_unary('tanh')

    def test_asinh(self):
        self._test_unary('asinh')

    def test_acosh(self):
        self._test_unary('acosh', input_range=(1.0, 10.0))

    def test_atanh(self):
        self._test_unary('atanh')

    def test_floor(self):
        self._test_unary('floor')

    def test_ceil(self):
        self._test_unary('ceil')

    def test_round(self):
        self._test_unary('round')

    def test_not(self):
        self._test_unary('not', dtype='logical')

    def test_sigmoid(self):
        self._test_unary('sigmoid')

    def test_relu(self):
        self._test_unary('relu')

    def test_gelu(self):
        self._test_unary('gelu', epsilon=1e-2)

    def test_silu(self):
        self._test_unary('silu')

    def test_softplus(self):
        self._test_unary('softplus')

    def test_add(self):
        self._test_binary('add')

    def test_sub(self):
        self._test_binary('sub')

    def test_mul(self):
        self._test_binary('mul')

    def test_div(self):
        self._test_binary('div')

    def test_pow(self):
        self._test_binary('pow')

    def test_min(self):
        self._test_binary('min')

    def test_max(self):
        self._test_binary('max')

    def test_lt(self):
        self._test_binary('lt')

    def test_gt(self):
        self._test_binary('gt')

    def test_le(self):
        self._test_binary('le')

    def test_ge(self):
        self._test_binary('ge')

    def test_eq(self):
        self._test_binary('eq')

    def test_ne(self):
        self._test_binary('ne')

    def test_and(self):
        self._test_binary('and', dtype='logical')

    def test_or(self):
        self._test_binary('or', dtype='logical')

    def test_sum_reduce(self):
        self._test_reduce('sum_reduce')

    def test_mean_reduce(self):
        self._test_reduce('mean_reduce')

    def test_min_reduce(self):
        self._test_reduce('min_reduce')

    def test_max_reduce(self):
        self._test_reduce('max_reduce')

    def test_any_reduce(self):
        self._test_reduce('any_reduce', dtype='logical')

    def test_all_reduce(self):
        self._test_reduce('all_reduce', dtype='logical')

    def test_sum_reduce_normalize(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = sum_reduce(input, axes = [2, 3], normalize = true);
        }
        """

        self._test_conversion('sum_reduce_normalize', code)

    def test_argmin_reduce(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = argmin_reduce(input, axes = [1]);
        }
        """

        self._test_conversion('argmin_reduce', code)

    def test_argmax_reduce(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = argmax_reduce(input, axes = [1]);
        }
        """

        self._test_conversion('argmax_reduce', code)

    def test_select(self):
        code = """
        graph G(cond, input1, input2) -> (output)
        {
            cond = external<logical>(shape = [1, 16, 32, 32]);
            input1 = external<scalar>(shape = [1, 16, 32, 32]);
            input2 = external<scalar>(shape = [1, 16, 32, 32]);
            output = select(cond, input1, input2);
        }
        """

        self._test_conversion('select', code)

    def test_clamp(self):
        code = """
        graph G(input, min, max) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            min = external<scalar>(shape = [1, 16, 32, 32]);
            max = external<scalar>(shape = [1, 16, 32, 32]);
            output = clamp(input, min, max);
        }
        """

        self._test_conversion('clamp', code, input_range=[(0.0, 1.0), (0.0, 0.5), (0.5, 1.0)])

    def test_add_aligned(self):
        code = """
        graph G(input1, input2) -> (output)
        {
            input1 = external<scalar>(shape = [1, 16, 32, 32]);
            input2 = external<scalar>(shape = [1, 16]);
            output = add(input1, input2);
        }
        """

        self._test_conversion('add_aligned', code)

    def test_prelu(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            alpha = variable<scalar>(shape = [1, 16], label = 'alpha');
            output = prelu(input, alpha);
        }
        """

        self._test_conversion('prelu', code)

    def test_leaky_relu(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = leaky_relu(input, alpha = 0.1);
        }
        """

        self._test_conversion('leaky_relu', code)

    def test_elu(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = elu(input, alpha = 0.5);
        }
        """

        self._test_conversion('elu', code)

    def test_selu(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = selu(input, lambda = 1.0);
        }
        """

        self._test_conversion('selu', code)

    def test_softmax(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = softmax(input, axes = [1]);
        }
        """

        self._test_conversion('softmax', code)

    def test_matmul(self):
        code = """
        graph G(A, B) -> (C)
        {
            A = external<scalar>(shape = [100, 200]);
            B = external<scalar>(shape = [200, 300]);
            C = matmul(A, B);
        }
        """

        self._test_conversion('matmul', code)

    def test_matmul_trA(self):
        code = """
        graph G(A, B) -> (C)
        {
            A = external<scalar>(shape = [200, 100]);
            B = external<scalar>(shape = [200, 300]);
            C = matmul(A, B, transposeA = true);
        }
        """

        self._test_conversion('matmul_trA', code)

    def test_matmul_trB(self):
        code = """
        graph G(A, B) -> (C)
        {
            A = external<scalar>(shape = [100, 200]);
            B = external<scalar>(shape = [300, 200]);
            C = matmul(A, B, transposeB = true);
        }
        """

        self._test_conversion('matmul_trB', code)

    def test_linear(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [100, 200]);
            filter = variable<scalar>(shape = [300, 200], label = 'filter');
            bias = variable<scalar>(shape = [1, 300], label = 'bias');
            output = linear(input, filter, bias);
        }
        """

        self._test_conversion('linear', code)

    def test_linear_nobias(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [100, 200]);
            filter = variable<scalar>(shape = [300, 200], label = 'filter');
            output = linear(input, filter);
        }
        """

        self._test_conversion('linear_nobias', code)

    def test_local_response_norm_channels(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = local_response_normalization(input, size = [1, 5, 1, 1]);
        }
        """

        self._test_conversion('local_response_norm_channels', code)

    def test_local_response_norm_spatial(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = local_response_normalization(input, size = [1, 1, 5, 5]);
        }
        """

        self._test_conversion('local_response_norm_spatial', code)

    def test_batch_norm(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            mean = variable<scalar>(shape = [1, 16], label = 'mean');
            variance = variable<scalar>(shape = [1, 16], label = 'variance');
            bias = variable<scalar>(shape = [1, 16], label = 'bias');
            scale = variable<scalar>(shape = [1, 16], label = 'scale');
            output = batch_normalization(input, mean, variance, bias, scale, epsilon = 1e-3);
        }
        """

        self._test_conversion('batch_norm', code)

    def test_l1_norm(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = l1_normalization(input, axes = [1, 2, 3]);
        }
        """

        self._test_conversion('l1_norm', code)

    def test_l2_norm(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = l2_normalization(input, axes = [1, 2, 3]);
        }
        """

        self._test_conversion('l2_norm', code)

    def test_local_mean_norm(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = local_mean_normalization(input, size = [1, 1, 5, 5]);
        }
        """

        self._test_conversion('local_mean_norm', code)

    def test_local_variance_norm(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = local_variance_normalization(input, size = [1, 1, 5, 5], epsilon = 0.01);
        }
        """

        self._test_conversion('local_variance_norm', code)

    def test_local_contrast_norm(self):
        code = """
        graph G(input) -> (output)
        {
            input = external<scalar>(shape = [1, 16, 32, 32]);
            output = local_contrast_normalization(input, size = [1, 1, 5, 5], epsilon = 0.01);
        }
        """

        self._test_conversion('local_contrast_norm', code)

    def test_nearest_downsample(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = nearest_downsample(input, factor = [2, 2]);
           }
           """

        self._test_conversion('nearest_downsample', code)

    def test_nearest_upsample(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = nearest_upsample(input, factor = [2, 2]);
           }
           """

        self._test_conversion('nearest_upsample', code)

    def test_area_downsample(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = area_downsample(input, factor = [2, 2]);
           }
           """

        self._test_conversion('area_downsample', code)

    def test_linear_upsample_symmetric(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = multilinear_upsample(input, factor = [2, 2], method = 'symmetric', border = 'constant');
           }
           """

        self._test_conversion('linear_upsample_symmetric', code)

    def test_linear_upsample_symmetric_replicate_border(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = multilinear_upsample(input, factor = [2, 2], method = 'symmetric', border = 'replicate');
           }
           """

        self._test_conversion('linear_upsample_symmetric_replicate_border', code)

    def test_linear_upsample_asymmetric(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = multilinear_upsample(input, factor = [2, 2], method = 'asymmetric', border = 'constant');
           }
           """

        self._test_conversion('linear_upsample_asymmetric', code)

    def test_linear_upsample_asymmetric_replicate_border(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = multilinear_upsample(input, factor = [2, 2], method = 'asymmetric', border = 'replicate');
           }
           """

        self._test_conversion('linear_upsample_asymmetric_replicate_border', code)

    def test_linear_upsample_aligned(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = multilinear_upsample(input, factor = [2, 2], method = 'aligned', border = 'replicate');
           }
           """

        self._test_conversion('linear_upsample_aligned', code, execute=False)

    def test_reshape(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 10, 10]);
               output = reshape(input, axis_start = 1, shape = [1600]);
           }
           """

        self._test_conversion('reshape', code)

    def test_transpose(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = transpose(input, axes = [0, 2, 3, 1]);
           }
           """

        self._test_conversion('transpose', code)

    def test_squeeze(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [4, 1, 32, 1, 32]);
               output = squeeze(input, axes = [1, 3]);
           }
           """

        self._test_conversion('squeeze', code)

    def test_unsqueeze(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [4, 32, 32]);
               output = unsqueeze(input, axes = [1, 3]);
           }
           """

        self._test_conversion('unsqueeze', code)

    def test_concat(self):
        code = """
           graph G(input1, input2) -> (output)
           {
               input1 = external<scalar>(shape = [1, 8, 32, 32]);
               input2 = external<scalar>(shape = [1, 8, 32, 32]);
               output = concat([input1, input2], axis = 1);
           }
           """

        self._test_conversion('concat', code)

    def test_split(self):
        code = """
           graph G(input) -> (output1, output2)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               [output1, output2] = split(input, axis = 1, ratios = [1, 1]);
           }
           """

        self._test_conversion('split', code)

    def test_stack(self):
        code = """
           graph G(input1, input2, input3) -> (output)
           {
               input1 = external<scalar>(shape = [4, 32, 32]);
               input2 = external<scalar>(shape = [4, 32, 32]);
               input3 = external<scalar>(shape = [4, 32, 32]);
               output = concat([input1, input2, input3], axis = 1);
           }
           """

        self._test_conversion('stack', code)

    def test_unstack(self):
        code = """
           graph G(input) -> (output1, output2, output3)
           {
               input = external<scalar>(shape = [4, 3, 32, 32]);
               [output1, output2, output3] = unstack(input, axis = 1);
           }
           """

        self._test_conversion('unstack', code)

    def test_pad(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = pad(input, padding = [(0, 0), (0, 0), (1, 1), (1, 1)]);
           }
           """

        self._test_conversion('pad', code)

    def test_pad_constant(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = pad(input, padding = [(0, 0), (0, 0), (1, 1), (1, 1)], value = 42.0);
           }
           """

        self._test_conversion('pad_constant', code)

    def test_pad_replicate(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = pad(input, padding = [(0, 0), (0, 0), (1, 1), (1, 1)], border = "replicate");
           }
           """

        self._test_conversion('pad_replicate', code)

    def test_pad_reflect(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = pad(input, padding = [(0, 0), (0, 0), (1, 1), (1, 1)], border = "reflect");
           }
           """

        self._test_conversion('pad_reflect', code, execute=False)

    def test_pad_symmetric(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = pad(input, padding = [(0, 0), (0, 0), (1, 1), (1, 1)], border = "reflect-even");
           }
           """

        self._test_conversion('pad_symmetric', code, execute=False)

    def test_slice(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = slice(input, axes = [2, 3], begin = [1, 1], end = [-1, -1]);
           }
           """

        self._test_conversion('slice', code)

    def test_tile(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = tile(input, repeats = [1, 2, 4, 4]);
           }
           """

        self._test_conversion('tile', code)

    def test_gather(self):
        code = """
           graph G(input, indices) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               indices = external<integer>(shape = [8]);
               output = gather(input, indices, axis = 1);
           }
           """

        self._test_conversion('gather', code, input_range=[(1, 100), (0, 15)])

    def test_cast(self):
        code = """
           graph G(input) -> (output)
           {
               input = external<scalar>(shape = [1, 16, 32, 32]);
               output = cast<integer>(input);
           }
           """

        self._test_conversion('cast', code)
