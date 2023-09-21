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

import nnef_tools.io.nnef as nnef_io
import nnef_tools.io.onnx as onnx_io
import nnef_tools.conversion.onnx_to_nnef as onnx_to_nnef
import nnef_tools.conversion.nnef_to_onnx as nnef_to_onnx
import nnef_tools.optimization.nnef_optimizer as nnef_opt
import nnef_tools.optimization.onnx_optimizer as onnx_opt
import numpy as np
import unittest
import tempfile
import onnx
import sys
import os
from onnx import helper, TensorProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE


UNITTEST_FOLDER = os.environ.get('UNITTEST_FOLDER')


class TestEnv(unittest.TestCase):

    _type_to_numpy = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int8)": np.int8,
        "tensor(int16)": np.int16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
        "tensor(uint16)": np.uint16,
        "tensor(uint32)": np.uint32,
        "tensor(uint64)": np.uint64,
        "tensor(bool)": np.bool_,
    }

    _network_folder = os.path.join(UNITTEST_FOLDER, 'onnx/nets/') if UNITTEST_FOLDER else None
    _output_folder = os.path.join(UNITTEST_FOLDER, 'onnx/ops/') if UNITTEST_FOLDER else None
    _infer_shapes = False
    _optimize = True

    def setUp(self) -> None:
        self._onnx_reader = onnx_io.Reader(simplify=False)
        self._onnx_writer = onnx_io.Writer()
        self._nnef_optimizer = nnef_opt.Optimizer()
        self._onnx_optimizer = onnx_opt.Optimizer()
        self._onnx_to_nnef_converter = onnx_to_nnef.Converter(infer_shapes=self._infer_shapes)
        self._nnef_to_onnx_converter = nnef_to_onnx.Converter()
        self._nnef_reader = nnef_io.Reader(custom_shapes=self._nnef_to_onnx_converter.defined_shapes(),
                                           decomposed=self._nnef_to_onnx_converter.decomposed_operations())
        self._nnef_writer = nnef_io.Writer(fragments=self._onnx_to_nnef_converter.defined_operations(),
                                           fragment_dependencies=self._onnx_to_nnef_converter.defined_operation_dependencies())

    def tearDown(self) -> None:
        pass

    def _convert_to_nnef(self, filename):
        onnx_graph = self._onnx_reader(filename)
        if self._optimize:
            onnx_graph = self._onnx_optimizer(onnx_graph)
        nnef_graph = self._onnx_to_nnef_converter(onnx_graph)
        if self._optimize:
            nnef_graph = self._nnef_optimizer(nnef_graph)
        self._nnef_writer(nnef_graph, filename + '.nnef')

    def _convert_from_nnef(self, filename):
        nnef_graph = self._nnef_reader(filename)
        onnx_graph = self._nnef_to_onnx_converter(nnef_graph)
        self._onnx_writer(onnx_graph, filename + '.onnx')

    @staticmethod
    def _random_data(dtype, shape):
        if dtype == bool:
            return np.random.random(shape) > 0.5
        else:
            return np.random.random(shape).astype(dtype)

    @staticmethod
    def _exec_model(filename):
        import onnxruntime
        np.random.seed(0)

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = onnxruntime.InferenceSession(filename, sess_options=options,
                                               providers=['CPUExecutionProvider'])

        inputs = {input.name: TestEnv._random_data(TestEnv._type_to_numpy[input.type], input.shape)
                  for input in session.get_inputs()}
        outputs = session.run([output.name for output in session.get_outputs()], inputs)

        return outputs

    @staticmethod
    def _create_tensor(value_info, data):
        name, shape, dtype = onnx_io.reader._get_value_info(value_info)
        if data is None:
            data = TestEnv._random_data(dtype, shape)
        elif not isinstance(data, np.ndarray):
            data = np.array(data)
        return helper.make_tensor(name, NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], shape, vals=data.flat)

    @staticmethod
    def _create_model(name, nodes, inputs, outputs, constants, values, opset_version, ir_version):
        tensors = [TestEnv._create_tensor(item, values.get(item.name)) for item in constants]
        graph_def = helper.make_graph(nodes, name, inputs, outputs, value_info=constants, initializer=tensors)
        model_def = helper.make_model(graph_def, producer_name='nnef-to-onnx-test')
        model_def.opset_import[0].version = opset_version
        model_def.ir_version = ir_version
        onnx.checker.check_model(model_def, full_check=True)
        return model_def

    @staticmethod
    def _save_model(model_def, filename):
        with open(filename, 'wb') as file:
            file.write(model_def.SerializeToString())

    def _test_conversion(self, name, nodes, inputs, outputs, constants=None, values=None, opset_version=11, ir_version=6, epsilon=1e-5):
        filename = tempfile.mktemp() if self._output_folder is None else TestEnv._output_folder + name + '.onnx'
        model_def = self._create_model('G', nodes, inputs, outputs, constants or [], values or {}, opset_version, ir_version)
        self._save_model(model_def, filename)
        self._test_conversion_from_file(filename, epsilon=epsilon)

    def _test_conversion_from_file(self, filename, epsilon=1e-5):
        self._convert_to_nnef(filename)
        self._convert_from_nnef(filename + '.nnef')

        original_outputs = self._exec_model(filename)
        converted_outputs = self._exec_model(filename + '.nnef.onnx')

        self.assertEqual(len(original_outputs), len(converted_outputs))
        for original, converted in zip(original_outputs, converted_outputs):
            if original.dtype == bool:
                self.assertTrue(np.all(original == converted))
            else:
                diff = np.max(np.abs(original - converted))
                self.assertLess(diff, epsilon)

    def _test_unary(self, op_type, dtype=TensorProto.FLOAT):
        input = helper.make_tensor_value_info('input', dtype, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', dtype, [1, 3, 32, 32])
        node = helper.make_node(
            op_type=op_type,
            inputs=['input'],
            outputs=['output'],
        )

        self._test_conversion(op_type.lower(), [node], [input], [output])

    def _test_binary(self, op_type, input_dtype=TensorProto.FLOAT, output_dtype=TensorProto.FLOAT):
        input1 = helper.make_tensor_value_info('input1', input_dtype, [1, 3, 32, 32])
        input2 = helper.make_tensor_value_info('input2', input_dtype, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', output_dtype, [1, 3, 32, 32])
        node = helper.make_node(
            op_type=op_type,
            inputs=['input1', 'input2'],
            outputs=['output'],
        )

        self._test_conversion(op_type.lower(), [node], [input1, input2], [output])

    def _test_reduce(self, op_type, keepdims, dtype=TensorProto.FLOAT, p=None):
        input = helper.make_tensor_value_info('input', dtype, [1, 16, 32, 32])
        output = helper.make_tensor_value_info('output', dtype, [1, 1, 32, 32] if keepdims else [1, 32, 32])
        node = helper.make_node(
            op_type=op_type,
            inputs=['input'],
            outputs=['output'],
            axes=[1],
            keepdims=keepdims,
        )

        self._test_conversion(op_type.lower(), [node], [input], [output])


class TestCases(TestEnv):

    def test_conv1d(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32])
        filter = helper.make_tensor_value_info('filter', TensorProto.FLOAT, [16, 3, 5])
        bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [16])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32])
        node = helper.make_node(
            op_type='Conv',
            inputs=['input', 'filter', 'bias'],
            outputs=['output'],
            auto_pad='SAME_UPPER',
        )

        self._test_conversion('conv1d', [node], [input], [output], constants=[filter, bias])

    def test_conv2d(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        filter = helper.make_tensor_value_info('filter', TensorProto.FLOAT, [16, 3, 5, 5])
        bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [16])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
        node = helper.make_node(
            op_type='Conv',
            inputs=['input', 'filter', 'bias'],
            outputs=['output'],
            auto_pad='SAME_UPPER',
        )

        self._test_conversion('conv2d', [node], [input], [output], constants=[filter, bias])

    def test_conv2d_nobias(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        filter = helper.make_tensor_value_info('filter', TensorProto.FLOAT, [16, 3, 5, 5])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
        node = helper.make_node(
            op_type='Conv',
            inputs=['input', 'filter'],
            outputs=['output'],
            auto_pad='SAME_UPPER',
        )

        self._test_conversion('conv2d-nobias', [node], [input], [output], constants=[filter])

    def test_conv2d_valid(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        filter = helper.make_tensor_value_info('filter', TensorProto.FLOAT, [16, 3, 5, 5])
        bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [16])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 28, 28])
        node = helper.make_node(
            op_type='Conv',
            inputs=['input', 'filter', 'bias'],
            outputs=['output'],
            auto_pad='VALID',
        )

        self._test_conversion('conv2d-valid', [node], [input], [output], constants=[filter, bias])

    def test_conv2d_pads(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        filter = helper.make_tensor_value_info('filter', TensorProto.FLOAT, [16, 3, 5, 5])
        bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [16])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 30, 30])
        node = helper.make_node(
            op_type='Conv',
            inputs=['input', 'filter', 'bias'],
            outputs=['output'],
            pads=[1, 1, 1, 1],
        )

        self._test_conversion('conv2d-pads', [node], [input], [output], constants=[filter, bias])

    def test_conv2d_same_lower(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        filter = helper.make_tensor_value_info('filter', TensorProto.FLOAT, [16, 3, 5, 5])
        bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [16])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
        node = helper.make_node(
            op_type='Conv',
            inputs=['input', 'filter', 'bias'],
            outputs=['output'],
            auto_pad="SAME_LOWER",
        )

        self._test_conversion('conv2d-same-lower', [node], [input], [output], constants=[filter, bias])

    def test_conv2d_transpose(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        filter = helper.make_tensor_value_info('filter', TensorProto.FLOAT, [16, 3, 5, 5])
        bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [3])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='ConvTranspose',
            inputs=['input', 'filter', 'bias'],
            outputs=['output'],
            auto_pad='SAME_UPPER',
        )

        self._test_conversion('conv2d_transpose', [node], [input], [output], constants=[filter, bias])


    def test_conv2d_transpose_output_shape(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        filter = helper.make_tensor_value_info('filter', TensorProto.FLOAT, [16, 3, 5, 5])
        bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [3])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='ConvTranspose',
            inputs=['input', 'filter', 'bias'],
            outputs=['output'],
            auto_pad='SAME_UPPER',
            output_shape=[1, 3, 32, 32],
        )

        self._test_conversion('conv2d_transpose-output_shape', [node], [input], [output], constants=[filter, bias])

    def test_conv2d_transpose_output_padding_strided(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        filter = helper.make_tensor_value_info('filter', TensorProto.FLOAT, [16, 3, 3, 3])
        bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [3])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 64, 64])
        node = helper.make_node(
            op_type='ConvTranspose',
            inputs=['input', 'filter', 'bias'],
            outputs=['output'],
            pads=(1, 1, 1, 1),
            output_padding=(1, 1),
            strides=(2, 2),
        )

        self._test_conversion('conv2d_transpose-output_padding-strided', [node], [input], [output], constants=[filter, bias])

    def test_conv3d(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32, 32])
        filter = helper.make_tensor_value_info('filter', TensorProto.FLOAT, [16, 3, 5, 5, 5])
        bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [16])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32, 32])
        node = helper.make_node(
            op_type='Conv',
            inputs=['input', 'filter', 'bias'],
            outputs=['output'],
            auto_pad='SAME_UPPER',
        )

        self._test_conversion('conv3d', [node], [input], [output], constants=[filter, bias])

    def test_max_pool1d(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32])
        node = helper.make_node(
            op_type='MaxPool',
            inputs=['input'],
            outputs=['output'],
            kernel_shape=[3],
            auto_pad='SAME_UPPER',
        )

        self._test_conversion('max_pool1d', [node], [input], [output])

    def test_max_pool2d(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='MaxPool',
            inputs=['input'],
            outputs=['output'],
            kernel_shape=[3, 3],
            auto_pad='SAME_UPPER',
        )

        self._test_conversion('max_pool2d', [node], [input], [output])

    def test_max_pool2d_valid(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 30, 30])
        node = helper.make_node(
            op_type='MaxPool',
            inputs=['input'],
            outputs=['output'],
            kernel_shape=[3, 3],
            auto_pad='VALID',
        )

        self._test_conversion('max_pool2d-valid', [node], [input], [output])

    def test_max_pool2d_pads(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='MaxPool',
            inputs=['input'],
            outputs=['output'],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )

        self._test_conversion('max_pool2d-pads', [node], [input], [output])

    def test_max_pool3d(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32, 32])
        node = helper.make_node(
            op_type='MaxPool',
            inputs=['input'],
            outputs=['output'],
            kernel_shape=[3, 3, 3],
            auto_pad='SAME_UPPER',
        )

        self._test_conversion('max_pool3d', [node], [input], [output])

    def test_avg_pool2d(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='AveragePool',
            inputs=['input'],
            outputs=['output'],
            kernel_shape=[3, 3],
            auto_pad='SAME_UPPER',
        )

        self._test_conversion('avg_pool2d', [node], [input], [output])

    def test_global_avg_pool2d(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 1, 1])
        node = helper.make_node(
            op_type='GlobalAveragePool',
            inputs=['input'],
            outputs=['output'],
        )

        self._test_conversion('global_avg_pool2d', [node], [input], [output])

    def test_global_max_pool2d(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 1, 1])
        node = helper.make_node(
            op_type='GlobalMaxPool',
            inputs=['input'],
            outputs=['output'],
        )

        self._test_conversion('global_max_pool2d', [node], [input], [output])

    def test_lp_pool2d(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='LpPool',
            inputs=['input'],
            outputs=['output'],
            kernel_shape=[3, 3],
            auto_pad='SAME_UPPER',
            p=2,
        )

        self._test_conversion('lp_pool2d', [node], [input], [output])

    def test_global_lp_pool2d(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 1, 1])
        node = helper.make_node(
            op_type='GlobalLpPool',
            inputs=['input'],
            outputs=['output'],
            p=2,
        )

        self._test_conversion('global_lp_pool2d', [node], [input], [output])

    def test_batch_norm(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        mean = helper.make_tensor_value_info('mean', TensorProto.FLOAT, [3])
        variance = helper.make_tensor_value_info('variance', TensorProto.FLOAT, [3])
        offset = helper.make_tensor_value_info('offset', TensorProto.FLOAT, [3])
        scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [3])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='BatchNormalization',
            inputs=['input', 'scale', 'offset', 'mean', 'variance'],
            outputs=['output'],
        )

        self._test_conversion('batch_norm', [node], [input], [output], [mean, variance, offset, scale])

    def test_transpose(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 32, 32, 3])
        node = helper.make_node(
            op_type='Transpose',
            inputs=['input'],
            outputs=['output'],
            perm=[0, 2, 3, 1],
        )

        self._test_conversion('transpose', [node], [input], [output])

    def test_reshape(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3 * 32 * 32])
        shape = helper.make_tensor_value_info('shape', TensorProto.INT64, [2])
        node = helper.make_node(
            op_type='Reshape',
            inputs=['input', 'shape'],
            outputs=['output'],
        )

        self._test_conversion('reshape', [node], [input, shape], [output], constants=[shape],
                              values={'shape': [1, 3 * 32 * 32]})

    def test_flatten(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3 * 32 * 32])
        node = helper.make_node(
            op_type='Flatten',
            inputs=['input'],
            outputs=['output'],
            axis=1,
        )

        self._test_conversion('flatten', [node], [input], [output])

    def test_squeeze(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 32, 32])
        node = helper.make_node(
            op_type='Squeeze',
            inputs=['input'],
            outputs=['output'],
            axes=[1],
        )

        self._test_conversion('squeeze', [node], [input], [output])

    def test_unsqueeze(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 32, 32])
        node = helper.make_node(
            op_type='Unsqueeze',
            inputs=['input'],
            outputs=['output'],
            axes=[1],
        )

        self._test_conversion('unsqueeze', [node], [input], [output])

    def test_matmul(self):
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [10, 20])
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [20, 30])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [10, 30])
        node = helper.make_node(
            op_type='MatMul',
            inputs=['input1', 'input2'],
            outputs=['output'],
        )

        self._test_conversion('matmul', [node], [input1, input2], [output])

    def test_gemm(self):
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [10, 20])
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [20, 30])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [10, 30])
        node = helper.make_node(
            op_type='Gemm',
            inputs=['input1', 'input2'],
            outputs=['output'],
        )

        self._test_conversion('gemm', [node], [input1, input2], [output])

    def test_linear(self):
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [10, 20])
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [30, 20])
        input3 = helper.make_tensor_value_info('input3', TensorProto.FLOAT, [30])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [10, 30])
        node = helper.make_node(
            op_type='Gemm',
            inputs=['input1', 'input2', 'input3'],
            outputs=['output'],
            transB=1,
        )

        self._test_conversion('linear', [node], [input1], [output], constants=[input2, input3])

    def test_lrn(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
        node = helper.make_node(
            op_type='LRN',
            inputs=['input'],
            outputs=['output'],
            size=5,
        )

        self._test_conversion('lrn', [node], [input], [output])

    def test_concat(self):
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [1, 3, 32, 32])
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 6, 32, 32])
        node = helper.make_node(
            op_type='Concat',
            inputs=['input1', 'input2'],
            outputs=['output'],
            axis=1,
        )

        self._test_conversion('concat', [node], [input1, input2], [output])

    def test_split(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 6, 32, 32])
        output1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [1, 3, 32, 32])
        output2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='Split',
            inputs=['input'],
            outputs=['output1', 'output2'],
            axis=1,
            split=[3, 3],
        )

        self._test_conversion('split', [node], [input], [output1, output2])

    def test_split_dynamic(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 6, 32, 32])
        split = helper.make_tensor_value_info('split', TensorProto.INT64, [2])
        output1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [1, 3, 32, 32])
        output2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='Split',
            inputs=['input', 'split'],
            outputs=['output1', 'output2'],
            axis=1,
        )

        self._test_conversion('split', [node], [input, split], [output1, output2],
                              constants=[split], values={'split': [3, 3]}, opset_version=13)

    def test_sum(self):
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [1, 3, 32, 32])
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='Sum',
            inputs=['input1', 'input2'],
            outputs=['output'],
        )

        self._test_conversion('sum', [node], [input1, input2], [output])

    def test_softmax(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
        node = helper.make_node(
            op_type='Softmax',
            inputs=['input'],
            outputs=['output'],
            axis=1,
        )

        self._test_conversion('softmax', [node], [input], [output])

    def test_leaky_relu(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
        node = helper.make_node(
            op_type='LeakyRelu',
            inputs=['input'],
            outputs=['output'],
        )

        self._test_conversion('leaky_relu', [node], [input], [output])

    def test_prelu(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        alpha = helper.make_tensor_value_info('alpha', TensorProto.FLOAT, [16, 1, 1])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
        node = helper.make_node(
            op_type='PRelu',
            inputs=['input', 'alpha'],
            outputs=['output'],
        )

        self._test_conversion('prelu', [node], [input, alpha], [output])

    def test_where(self):
        cond = helper.make_tensor_value_info('cond', TensorProto.BOOL, [1, 1, 32, 32])
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [1, 3, 32, 32])
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [1, 3, 1, 1])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='Where',
            inputs=['cond', 'input1', 'input2'],
            outputs=['output'],
        )

        self._test_conversion('where', [node], [cond, input1, input2], [output])

    def test_clip(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        min = helper.make_tensor_value_info('min', TensorProto.FLOAT, [])
        max = helper.make_tensor_value_info('max', TensorProto.FLOAT, [])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='Clip',
            inputs=['input', 'min', 'max'],
            outputs=['output'],
        )

        self._test_conversion('clip', [node], [input, min, max], [output])

    def test_argmin(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.INT64, [1, 1, 32, 32])
        node = helper.make_node(
            op_type='ArgMin',
            inputs=['input'],
            outputs=['output'],
            axis=1,
            keepdims=True,
        )

        self._test_conversion('argmin_reduce', [node], [input], [output])

    def test_argmax(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.INT64, [1, 32, 32])
        node = helper.make_node(
            op_type='ArgMax',
            inputs=['input'],
            outputs=['output'],
            axis=1,
            keepdims=False,
        )

        self._test_conversion('argmax_reduce', [node], [input], [output])

    def test_pad(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 34, 34])
        pads = helper.make_tensor_value_info('pads', TensorProto.INT64, [8])
        node = helper.make_node(
            op_type='Pad',
            inputs=['input', 'pads'],
            outputs=['output'],
        )

        self._test_conversion('pad', [node], [input], [output], constants=[pads],
                              values={'pads': [0, 0, 1, 1, 0, 0, 1, 1]})

    def test_tile(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 64, 64])
        repeats = helper.make_tensor_value_info('repeats', TensorProto.INT64, [4])
        node = helper.make_node(
            op_type='Tile',
            inputs=['input', 'repeats'],
            outputs=['output'],
        )

        self._test_conversion('tile', [node], [input], [output], constants=[repeats],
                              values={'repeats': [1, 1, 2, 2]})

    def test_expand(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 3, 1, 1])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 3, 32, 32])
        repeats = helper.make_tensor_value_info('shape', TensorProto.INT64, [4])
        node = helper.make_node(
            op_type='Expand',
            inputs=['input', 'shape'],
            outputs=['output'],
        )

        self._test_conversion('expand', [node], [input], [output], constants=[repeats],
                              values={'shape': [4, 3, 32, 32]})

    def test_slice(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 30, 30])
        starts = helper.make_tensor_value_info('starts', TensorProto.INT64, [2])
        ends = helper.make_tensor_value_info('ends', TensorProto.INT64, [2])
        axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [2])
        node = helper.make_node(
            op_type='Slice',
            inputs=['input', 'starts', 'ends', 'axes'],
            outputs=['output'],
        )

        self._test_conversion('slice', [node], [input], [output], constants=[starts, ends, axes],
                              values={'starts': [1, 1], 'ends': [-1, -1], 'axes': [2, 3]})

    def test_strided_slice(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        starts = helper.make_tensor_value_info('starts', TensorProto.INT64, [2])
        ends = helper.make_tensor_value_info('ends', TensorProto.INT64, [2])
        axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [2])
        steps = helper.make_tensor_value_info('steps', TensorProto.INT64, [2])
        node = helper.make_node(
            op_type='Slice',
            inputs=['input', 'starts', 'ends', 'axes', 'steps'],
            outputs=['output'],
        )

        self._test_conversion('strided_slice', [node], [input], [output], constants=[starts, ends, axes, steps],
                              values={'starts': [-1, -1], 'ends': [-sys.maxsize, -sys.maxsize], 'axes': [2, 3], 'steps': [-1, -1]})

    def test_flip(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 30, 30])
        starts = helper.make_tensor_value_info('starts', TensorProto.INT64, [2])
        ends = helper.make_tensor_value_info('ends', TensorProto.INT64, [2])
        axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [2])
        steps = helper.make_tensor_value_info('steps', TensorProto.INT64, [2])
        node = helper.make_node(
            op_type='Slice',
            inputs=['input', 'starts', 'ends', 'axes', 'steps'],
            outputs=['output'],
        )

        self._test_conversion('flip', [node], [input], [output], constants=[starts, ends, axes, steps],
                              values={'starts': [-2, -2], 'ends': [0, 0], 'axes': [2, 3], 'steps': [-1, -1]})

    def test_l1_normalization(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
        node = helper.make_node(
            op_type='LpNormalization',
            inputs=['input'],
            outputs=['output'],
            axis=1,
            p=1,
        )

        self._test_conversion('l1_normalization', [node], [input], [output])

    def test_l2_normalization(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
        node = helper.make_node(
            op_type='LpNormalization',
            inputs=['input'],
            outputs=['output'],
            axis=1,
            p=2,
        )

        self._test_conversion('l2_normalization', [node], [input], [output])

    def test_mean_variance_normalization(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
        node = helper.make_node(
            op_type='MeanVarianceNormalization',
            inputs=['input'],
            outputs=['output'],
            axes=[0, 2, 3],
        )

        self._test_conversion('mean_variance_normalization', [node], [input], [output])

    def test_instance_normalization(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        scale = helper.make_tensor_value_info('scale', TensorProto.FLOAT, [16])
        offset = helper.make_tensor_value_info('offset', TensorProto.FLOAT, [16])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
        node = helper.make_node(
            op_type='InstanceNormalization',
            inputs=['input', 'scale', 'offset'],
            outputs=['output'],
        )

        self._test_conversion('instance_normalization', [node], [input], [output], constants=[scale, offset])

    def test_lp_reduce(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 32, 32])
        node = helper.make_node(
            op_type='ReduceL1',
            inputs=['input'],
            outputs=['output'],
            axes=[1],
            keepdims=True,
        )

        self._test_conversion('lp_reduce', [node], [input], [output])

    def test_nearest_upsample(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 64, 64])
        node = helper.make_node(
            op_type='Upsample',
            inputs=['input'],
            outputs=['output'],
            scales=[1.0, 1.0, 2.0, 2.0],
            mode='nearest',
        )

        self._test_conversion('nearest_upsample', [node], [input], [output], opset_version=8)

    def test_linear_upsample(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 64, 64])
        node = helper.make_node(
            op_type='Upsample',
            inputs=['input'],
            outputs=['output'],
            scales=[1.0, 1.0, 2.0, 2.0],
            mode='linear',
        )

        self._test_conversion('linear_upsample', [node], [input], [output], opset_version=8)

    def test_resize_nearest_upsample(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4])
        roi = helper.make_tensor_value_info('roi', TensorProto.FLOAT, [0])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 64, 64])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', 'roi', 'scales'],
            outputs=['output'],
            mode='nearest',
        )

        self._test_conversion('resize_nearest_upsample', [node], [input], [output], constants=[scales, roi],
                              values={'scales': [1.0, 1.0, 2.0, 2.0], 'roi': []})

    def test_resize_linear_upsample(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4])
        roi = helper.make_tensor_value_info('roi', TensorProto.FLOAT, [0])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 64, 64])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', 'roi', 'scales'],
            outputs=['output'],
            mode='linear',
        )

        self._test_conversion('resize_liner_upsample', [node], [input], [output], constants=[scales, roi],
                              values={'scales': [1.0, 1.0, 2.0, 2.0], 'roi': []})

    def test_resize_nearest_downsample(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4])
        roi = helper.make_tensor_value_info('roi', TensorProto.FLOAT, [0])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 16, 16])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', 'roi', 'scales'],
            outputs=['output'],
            mode='nearest',
        )

        self._test_conversion('resize_nearest_downsample', [node], [input], [output], constants=[scales, roi],
                              values={'scales': [1.0, 1.0, 0.5, 0.5], 'roi': []})

    def test_cast(self):
        input = helper.make_tensor_value_info('input', TensorProto.INT32, [1, 4, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4, 32, 32])
        node = helper.make_node(
            op_type='Cast',
            inputs=['input'],
            outputs=['output'],
            to=TensorProto.FLOAT,
        )

        self._test_conversion('cast', [node], [input], [output])

    def test_gather(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        indices = helper.make_tensor_value_info('indices', TensorProto.INT32, [24])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 24, 32, 32])
        node = helper.make_node(
            op_type='Gather',
            inputs=['input', 'indices'],
            outputs=['output'],
            axis=1,
        )

        self._test_conversion('gather', [node], [input, indices], [output])

    def test_lstm(self):
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [5, 4, 32])
        W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 256, 32])
        R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 256, 64])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 512])
        h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [1, 4, 64])
        c0 = helper.make_tensor_value_info('c0', TensorProto.FLOAT, [1, 4, 64])
        hn = helper.make_tensor_value_info('hn', TensorProto.FLOAT, [1, 4, 64])
        cn = helper.make_tensor_value_info('cn', TensorProto.FLOAT, [1, 4, 64])
        node = helper.make_node(
            op_type='LSTM',
            inputs=['X', 'W', 'R', 'B', '', 'h0', 'c0'],
            outputs=['', 'hn', 'cn'],
            hidden_size=64,
            direction="forward",
        )

        self._test_conversion('lstm', [node], [X, h0, c0], [hn, cn], constants=[W, R, B])

    def test_min_recude(self):
        self._test_reduce('ReduceMin', keepdims=False)

    def test_max_recude(self):
        self._test_reduce('ReduceMax', keepdims=False)

    def test_mean_recude(self):
        self._test_reduce('ReduceMean', keepdims=False)

    def test_sum_recude(self):
        self._test_reduce('ReduceSum', keepdims=False)

    def test_max_recude_keepdims(self):
        self._test_reduce('ReduceMax', keepdims=True)

    def test_relu(self):
        self._test_unary('Relu')

    def test_sigmoid(self):
        self._test_unary('Sigmoid')

    def test_tanh(self):
        self._test_unary('Tanh')

    def test_softplus(self):
        self._test_unary('Softplus')

    def test_selu(self):
        self._test_unary('Selu')

    def test_not(self):
        self._test_unary('Not', dtype=TensorProto.BOOL)

    def test_elu(self):
        self._test_unary('Elu')

    def test_erf(self):
        self._test_unary('Erf')

    def test_abs(self):
        self._test_unary('Abs')

    def test_sign(self):
        self._test_unary('Sign')

    def test_sin(self):
        self._test_unary('Sin')

    def test_cos(self):
        self._test_unary('Cos')

    def test_tan(self):
        self._test_unary('Tan')

    def test_asin(self):
        self._test_unary('Asin')

    def test_acos(self):
        self._test_unary('Acos')

    def test_atan(self):
        self._test_unary('Atan')

    def test_sinh(self):
        self._test_unary('Sinh')

    def test_cosh(self):
        self._test_unary('Cosh')

    def test_tanh(self):
        self._test_unary('Tanh')

    def test_exp(self):
        self._test_unary('Exp')

    def test_log(self):
        self._test_unary('Log')

    def test_neg(self):
        self._test_unary('Neg')

    def test_sqrt(self):
        self._test_unary('Sqrt')

    def test_ceil(self):
        self._test_unary('Ceil')

    def test_floor(self):
        self._test_unary('Floor')

    def test_round(self):
        self._test_unary('Round')

    def test_add(self):
        self._test_binary('Add')

    def test_sub(self):
        self._test_binary('Sub')

    def test_mul(self):
        self._test_binary('Mul')

    def test_div(self):
        self._test_binary('Div')

    def test_pow(self):
        self._test_binary('Pow')

    def test_min(self):
        self._test_binary('Min')

    def test_max(self):
        self._test_binary('Max')

    def test_and(self):
        self._test_binary('And', input_dtype=TensorProto.BOOL, output_dtype=TensorProto.BOOL)

    def test_or(self):
        self._test_binary('Or', input_dtype=TensorProto.BOOL, output_dtype=TensorProto.BOOL)

    def test_equal(self):
        self._test_binary('Equal', output_dtype=TensorProto.BOOL)

    def test_less(self):
        self._test_binary('Less', output_dtype=TensorProto.BOOL)

    def test_greater(self):
        self._test_binary('Greater', output_dtype=TensorProto.BOOL)


@unittest.skipIf(TestEnv._network_folder is None or not os.path.isdir(TestEnv._network_folder),
                 "no network test folder provided")
class NetworkTestCases(TestEnv):

    def test_alexnet(self):
        self._test_conversion_from_file(self._network_folder + 'alexnet.onnx')

    def test_googlenet(self):
        self._test_conversion_from_file(self._network_folder + 'googlenet.onnx')

    def test_inception_v1(self):
        self._test_conversion_from_file(self._network_folder + 'inception_v1.onnx')

    def test_inception_v2(self):
        self._test_conversion_from_file(self._network_folder + 'inception_v2.onnx')

    def test_mobilenet_v2(self):
        self._test_conversion_from_file(self._network_folder + 'mobilenet_v2.onnx', epsilon=1e-4)

    def test_resnet18_v1(self):
        self._test_conversion_from_file(self._network_folder + 'resnet18_v1.onnx')

    def test_resnet18_v2(self):
        self._test_conversion_from_file(self._network_folder + 'resnet18_v2.onnx')

    def test_squeezenet_v1(self):
        self._test_conversion_from_file(self._network_folder + 'squeezenet_v1.onnx')

    def test_squeezenet_v2(self):
        self._test_conversion_from_file(self._network_folder + 'squeezenet_v2.onnx')

    def test_shufflenet(self):
        self._test_conversion_from_file(self._network_folder + 'shufflenet.onnx')


if __name__ == '__main__':
    unittest.main()
