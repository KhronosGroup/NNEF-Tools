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

import unittest
import sknd_test
import skriptnd as sknd
from nnef_tools.io import skriptnd as skriptnd_io
from nnef_tools.optimization import sknd_transposer
import numpy as np
import os


UNITTEST_FOLDER = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../unittest/'))


class TestEnv(sknd_test.TestEnv):

    _network_folder = os.path.join(UNITTEST_FOLDER, 'nnef2/tf/nets/') if UNITTEST_FOLDER else None
    _output_folder = os.path.join(UNITTEST_FOLDER, 'nnef2/trans/ops/') if UNITTEST_FOLDER else None

    _NumpyDtype = {
        sknd.Dtype.Real: np.float32,
        sknd.Dtype.Int: np.int64,
        sknd.Dtype.Bool: np.bool_,
    }

    def setUp(self) -> None:
        self._skriptnd_reader = skriptnd_io.Reader(atomic=True)
        self._skriptnd_writer = skriptnd_io.Writer(inline_subgraphs=False)
        self._skriptnd_transposer = sknd_transposer.NXCtoNCX()
        self._execute = True
        self._transpose_outputs = True

    def tearDown(self) -> None:
        pass

    @staticmethod
    def nxc2ncx(shape):
        return [shape[0], shape[-1], *shape[1:-1]] if len(shape) > 2 else shape

    def _convert_to_sknd(self, filename, input_shape=None):
        sknd_model = self._skriptnd_reader(filename)
        self._skriptnd_transposer(sknd_model)
        if input_shape is not None:
            self._set_max_input_shapes(sknd_model, input_shape)
        output_filename = filename + '.nnef2'
        self._skriptnd_writer(sknd_model, output_filename)

    def _exec_orig_model(self, path, input_shape=None, input_range=None):
        np.random.seed(0)

        model = sknd.read_model(path)
        if not model:
            return None

        compiled_model = sknd.compile_model(model, keep_generated_code=False)

        if not isinstance(input_shape, list):
            input_shape = [input_shape] * len(model.graphs[0].inputs)
        if not isinstance(input_range, list):
            input_range = [input_range] * len(model.graphs[0].inputs)

        inputs = [TestEnv._random_data(sknd.DtypeToNumpy[input.dtype],
                                       TestEnv.nxc2ncx(input_shape[idx] or input.shape),
                                       input_range[idx])
                  for idx, input in enumerate(model.graphs[0].inputs)]

        inputs = [np.ascontiguousarray(np.transpose(input, axes=[0, *range(2, len(input.shape)), 1]))
                  if len(input.shape) > 2 else input for input in inputs]

        outputs = compiled_model(*inputs)

        if self._transpose_outputs:
            outputs = [np.transpose(output, axes=[0, -1, *range(1, len(output.shape) - 1)])
                       if len(output.shape) > 2 else output for output in outputs]

        return outputs

    def _test_conversion(self, name, code, epsilon=1e-5, input_range=None, execute=True, compile=True,
                         transpose_outputs=True):
        path = self._output_folder + name + '.nnef2'
        if not os.path.exists(path):
            os.makedirs(path)

        filename = path + '/main.sknd'
        with open(filename, 'w') as file:
            print(code, file=file)

        model = sknd.parse_string(code)
        if model is None:
            raise ValueError('invalid model')

        for graph in model.graphs:
            for tensor in graph.variables:
                filename = path + '/' + tensor.name + '.dat'
                data = self._random_data(self._NumpyDtype[tensor.dtype], tensor.shape)
                with open(filename, 'wb') as file:
                    sknd.write_tensor(file, data)

        self._transpose_outputs = transpose_outputs
        self._test_conversion_from_file(path, epsilon=epsilon, input_range=input_range, execute=execute, compile=compile)


class TestCases(TestEnv):

    def test_iden(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,224,224,3];
            }
            @output {
                output: real[4,224,224,3];
            }
            @compose {
                output = math.iden(input);
            }
        }
        """

        self._test_conversion('iden', code)

    def test_relu(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,224,224,3];
            }
            @output {
                output: real[4,224,224,3];
            }
            @compose {
                output = nn.relu(input);
            }
        }
        """

        self._test_conversion('relu', code)

    def test_add(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input1: real[4,224,224,3];
                input2: real[4,224,224,3];
            }
            @output {
                output: real[4,224,224,3];
            }
            @compose {
                output = math.add(input1, input2);
            }
        }
        """

        self._test_conversion('add', code)

    def test_add_channelwise(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input1: real[4,224,224,3];
                input2: real[3];
            }
            @output {
                output: real[4,224,224,3];
            }
            @compose {
                output = math.add(input1, input2);
            }
        }
        """

        self._test_conversion('add_channelwise', code)

    def test_select(self):
        code = """
        import nn;
        graph G
        {
            @input {
                cond: bool[4,224,224,3];
                input1: real[4,224,224,3];
                input2: real[4,224,224,3];
            }
            @output {
                output: real[4,224,224,3];
            }
            @compose {
                output = math.select(cond, input1, input2);
            }
        }
        """

        self._test_conversion('select', code)

    def test_select_channelwise(self):
        code = """
        import nn;
        graph G
        {
            @input {
                cond: bool[];
                input1: real[4,224,224,3];
                input2: real[3];
            }
            @output {
                output: real[4,224,224,3];
            }
            @compose {
                output = math.select(cond, input1, input2);
            }
        }
        """

        self._test_conversion('select_channelwise', code)

    def test_conv1d(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,224,3];
            }
            @output {
                output: real[4,224,6];
            }
            @variable {
                filter: real[5,3,6];
                bias: real[6];
            }
            @compose {
                output = nn.conv{data_format="NXC", filter_format="XCN"}(input, filter, bias);
            }
        }
        """

        self._test_conversion('conv1d', code)

    def test_conv2d(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,224,224,3];
            }
            @output {
                output: real[4,224,224,6];
            }
            @variable {
                filter: real[5,5,3,6];
                bias: real[6];
            }
            @compose {
                output = nn.conv{data_format="NXC", filter_format="XCN"}(input, filter, bias);
            }
        }
        """

        self._test_conversion('conv2d', code)

    def test_deconv2d(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,224,224,6];
            }
            @output {
                output: real[4,224,224,3];
            }
            @variable {
                filter: real[5,5,3,6];
                bias: real[3];
            }
            @compose {
                output = nn.deconv{data_format="NXC", filter_format="XCN"}(input, filter, bias);
            }
        }
        """

        self._test_conversion('deconv2d', code)

    def test_pool2d(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,224,224,3];
            }
            @output {
                output: real[4,112,112,3];
            }
            @compose {
                output = nn.avg_pool{size=3, stride=2, axes=[1, 2]}(input);
            }
        }
        """

        self._test_conversion('pool2d', code)

    def test_reduce2d(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,224,224,3];
            }
            @output {
                output: real[4,1,1,3];
            }
            @compose {
                output = math.max_reduce{axes=[1, 2]}(input);
            }
        }
        """

        self._test_conversion('reduce2d', code)

    def test_moments(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,224,224,3];
            }
            @output {
                mean: real[4,1,1,3];
                variance: real[4,1,1,3];
            }
            @compose {
                mean, variance = math.moments{axes=[1, 2]}(input);
            }
        }
        """

        self._test_conversion('moments', code)

    def test_concat(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input1: real[4,224,224,3];
                input2: real[4,224,224,3];
            }
            @output {
                output: real[4,224,224,6];
            }
            @compose {
                output = layout.concat{axis=-1}([input1, input2]);
            }
        }
        """

        self._test_conversion('concat', code)

    def test_split(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,224,224,6];
            }
            @output {
                output1: real[4,224,224,3];
                output2: real[4,224,224,3];
            }
            @compose {
                [output1, output2] = layout.split{axis=-1, count=2}(input);
            }
        }
        """

        self._test_conversion('split', code)

    def test_reshape(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,10,10,3];
            }
            @output {
                output: real[4,300];
            }
            @compose {
                output = layout.reshape{axis=1, shape=[300]}(input);
            }
        }
        """

        self._test_conversion('reshape', code)

    def test_transpose(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,10,10,3];
            }
            @output {
                output: real[4,3,10,10];
            }
            @compose {
                output = layout.transpose{perm=[0,3,1,2]}(input);
            }
        }
        """

        self._test_conversion('transpose', code, transpose_outputs=False)

    def test_flatten(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,10,10,3];
            }
            @output {
                output: real[4,300];
            }
            @compose {
                output = layout.flatten{axis=1}(input);
            }
        }
        """

        self._test_conversion('flatten', code)

    def test_squeeze(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,1,1,64];
            }
            @output {
                output: real[4,64];
            }
            @compose {
                output = layout.squeeze{axes=[1,2]}(input);
            }
        }
        """

        self._test_conversion('squeeze', code)

    def test_stack(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input1: real[4,224,224,1];
                input2: real[4,224,224,1];
                input3: real[4,224,224,1];
            }
            @output {
                output: real[4,224,224,3];
            }
            @compose {
                output = layout.stack{axis=-1, squeeze=true}([input1, input2, input3]);
            }
        }
        """

        self._test_conversion('stack', code)

    def test_unstack(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,224,224,3];
            }
            @output {
                output1: real[4,224,224,1];
                output2: real[4,224,224,1];
                output3: real[4,224,224,1];
            }
            @compose {
                [output1, output2, output3] = layout.unstack{axis=-1, squeeze=false}(input);
            }
        }
        """

        self._test_conversion('unstack', code)

    def test_unstack_squeezed(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,224,224,3];
            }
            @output {
                output1: real[4,224,224];
                output2: real[4,224,224];
                output3: real[4,224,224];
            }
            @compose {
                [output1, output2, output3] = layout.unstack{axis=-1, squeeze=true}(input);
            }
        }
        """

        self._test_conversion('unstack-squeezed', code, transpose_outputs=False)

    def test_tile(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,64,64,3];
            }
            @output {
                output: real[4,256,256,3];
            }
            @compose {
                output = layout.tile{axes=[1,2], repeats=[4,4]}(input);
            }
        }
        """

        self._test_conversion('tile', code)

    def test_broadcast(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,1,1,3];
            }
            @output {
                output: real[4,256,256,3];
            }
            @compose {
                output = layout.broadcast{axes=[1,2], shape=[256,256]}(input);
            }
        }
        """

        self._test_conversion('broadcast', code)

    def test_slice(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,256,256,3];
            }
            @output {
                output: real[4,252,252,3];
            }
            @compose {
                output = layout.slice{axes=[1,2], begin=[2,2], end=[-2,-2]}(input);
            }
        }
        """

        self._test_conversion('slice', code)

    def test_pad(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,256,256,3];
            }
            @output {
                output: real[4,260,260,3];
            }
            @compose {
                output = layout.pad{axes=[1,2], padding=[2,2,2,2]}(input);
            }
        }
        """

        self._test_conversion('pad', code)

    def test_space_to_batch(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,256,256,3];
            }
            @output {
                output: real[16,128,128,3];
            }
            @compose {
                output = layout.space_to_batch{block_size=[2,2], data_format="NXC"}(input);
            }
        }
        """

        self._test_conversion('space_to_batch', code)

    def test_shuffle(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,256,256,3];
            }
            @output {
                output: real[4,256,256,3];
            }
            @compose {
                output = layout.shuffle{axis=-1, groups=3}(input);
            }
        }
        """

        self._test_conversion('shuffle', code)

    def test_gather(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,256,256,64];
                index: int[10];
            }
            @output {
                output: real[4,256,256,10];
            }
            @compose {
                output = layout.gather{axis=-1}(input, index);
            }
        }
        """

        self._test_conversion('gather', code, input_range=[(0, 1), (0, 10)])

    def test_scatter(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,256,256,64];
                index: int[4,256,256,10];
                update: real[4,256,256,10];
            }
            @output {
                output: real[4,256,256,64];
            }
            @compose {
                output = layout.scatter{axis=-1}(input, index, update);
            }
        }
        """

        self._test_conversion('scatter', code, input_range=[(0, 1), (0, 10), (0, 1)])

    def test_sum(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input1: real[4,256,256,3];
                input2: real[4,256,256,3];
                input3: real[4,256,256,3];
            }
            @output {
                output: real[4,256,256,3];
            }
            @compose {
                output = math.sum_n([input1, input2, input3]);
            }
        }
        """

        self._test_conversion('sum', code)

    def test_batch_norm(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,256,256,16];
            }
            @variable {
                mean: real[16];
                variance: real[16];
                offset: real[16];
                scale: real[16];
            }
            @output {
                output: real[4,256,256,16];
            }
            @compose {
                output = nn.batch_norm{channel_axis=-1}(input, mean, variance, offset, scale);
            }
        }
        """

        self._test_conversion('batch_norm', code)

    def test_mean_variance_norm(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,256,256,16];
            }
            @output {
                output: real[4,256,256,16];
            }
            @compose {
                output = nn.mean_variance_norm{axes=[1,2]}(input);
            }
        }
        """

        self._test_conversion('mean_variance_norm', code)

    def test_local_response_norm(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,256,256,16];
            }
            @output {
                output: real[4,256,256,16];
            }
            @compose {
                output = nn.local_response_norm{axes=[1,2], size=5}(input);
            }
        }
        """

        self._test_conversion('local_response_norm', code)

    def test_l2_norm(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,256,256,16];
            }
            @output {
                output: real[4,256,256,16];
            }
            @compose {
                output = nn.l2_norm{axes=[1,2]}(input);
            }
        }
        """

        self._test_conversion('l2_norm', code)

    def test_softmax(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,256,256,16];
            }
            @output {
                output: real[4,256,256,16];
            }
            @compose {
                output = nn.softmax{axes=[-1]}(input);
            }
        }
        """

        self._test_conversion('softmax', code)

    def test_linear_upsample(self):
        code = """
        import image;
        graph G
        {
            @input {
                input: real[4,128,128,3];
            }
            @output {
                output: real[4,256,256,3];
            }
            @compose {
                output = image.linear_upsample{axes=[1,2], factor=[2,2]}(input);
            }
        }
        """

        self._test_conversion('linear_upsample', code)

    def test_resize(self):
        code = """
        import image;
        graph G
        {
            @input {
                input: real[4,128,128,3];
            }
            @output {
                output: real[4,256,256,3];
            }
            @compose {
                output = image.resize{axes=[1,2], size=[256,256]}(input);
            }
        }
        """

        self._test_conversion('resize', code)

    def test_quantize(self):
        code = """
        import quant;
        graph G
        {
            @input {
                input: real[4,256,256,3];
            }
            @output {
                output: real[4,256,256,3];
            }
            @compose {
                output = quant.zero_point_linear_quantize{channel_axis=-1, zero_point=0, scale=1.0, bits=8}(input);
            }
        }
        """

        self._test_conversion('quantize', code)

    def test_prelu(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,224,224,3];
                alpha: real[3];
            }
            @output {
                output: real[4,224,224,3];
            }
            @compose {
                output = nn.prelu{axis=-1}(input, alpha);
            }
        }
        """

        self._test_conversion('prelu', code)

    def test_argmin(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,224,224,3];
            }
            @output {
                output: int[4,224,224,1];
            }
            @compose {
                output = math.argmin{axis=-1}(input);
            }
        }
        """

        self._test_conversion('argmin', code)

    def test_argmin_nd(self):
        code = """
        import nn;
        graph G
        {
            @input {
                input: real[4,224,224,3];
            }
            @output {
                output1: int[4,1,1,3];
                output2: int[4,1,1,3];
            }
            @compose {
                [output1, output2] = math.argmin_nd{axes=[1,2]}(input);
            }
        }
        """

        self._test_conversion('argmin_nd', code)

    def test_axpb(self):
        code = """
        import nn;
        graph G
        {
            @input {
                a: real[4,224,224,3];
                x: real[];
                b: real[3];
            }
            @output {
                output: real[4,224,224,3];
            }
            @compose {
                output = math.axpb(a, x, b);
            }
        }
        """

        self._test_conversion('axpb', code)

    def test_axpby(self):
        code = """
        import nn;
        graph G
        {
            @input {
                a: real[];
                x: real[4,224,224,3];
                b: real[3];
                y: real[4,224,224,3];
            }
            @output {
                output: real[4,224,224,3];
            }
            @compose {
                output = math.axpby(a, x, b, y);
            }
        }
        """

        self._test_conversion('axpby', code)

    def test_clamp(self):
        code = """
        import nn;
        graph G
        {
            @input {
                x: real[4,224,224,3];
                min: real[3];
                max: real[3];
            }
            @output {
                output: real[4,224,224,3];
            }
            @compose {
                output = math.clamp(x, min, max);
            }
        }
        """

        self._test_conversion('clamp', code)


@unittest.skipIf(TestEnv._network_folder is None or not os.path.isdir(TestEnv._network_folder),
                 "no network test folder provided")
class NetworkTestCases(TestEnv):

    def test_mobilenet_v1(self):
        self._test_conversion_from_file(self._network_folder + 'mobilenet_v1.pb.nnef2')

    def test_inception_v3(self):
        self._test_conversion_from_file(self._network_folder + 'inception_v3.pb.nnef2')

    def test_inception_v4(self):
        self._test_conversion_from_file(self._network_folder + 'inception_v4.pb.nnef2')

    def test_inception_resnet_v2(self):
        self._test_conversion_from_file(self._network_folder + 'inception_resnet_v2.pb.nnef2')

    def test_squeezenet(self):
        self._test_conversion_from_file(self._network_folder + 'squeezenet.pb.nnef2')

    def test_nasnet(self):
        self._test_conversion_from_file(self._network_folder + 'nasnet.pb.nnef2')


if __name__ == '__main__':
    unittest.main()
