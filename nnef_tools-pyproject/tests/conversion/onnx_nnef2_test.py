from nnef_tools.io import onnx as onnx_io
from nnef_tools.io import skriptnd as skriptnd_io
from nnef_tools.conversion import onnx_to_nnef2
from nnef_tools.conversion.onnx_to_nnef2 import ShapeExpr
from nnef_tools.optimization import skriptnd_optimizer
from nnef_tools.optimization import onnx_optimizer
from skriptnd import DtypeToNumpy
import numpy as np
import unittest
import tempfile
import onnx
import sys
import os
from onnx import helper, TensorProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE


UNITTEST_FOLDER = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../unittest/'))


class TestEnv(unittest.TestCase):

    DefaultOpsetVersion = 13
    DefaultIrVersion = 6

    _onnx_type_to_numpy = {
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

    _skriptnd_dtype_to_numpy = DtypeToNumpy

    _network_folder = os.path.join(UNITTEST_FOLDER, 'nnef2/onnx/nets/') if UNITTEST_FOLDER else None
    _output_folder = os.path.join(UNITTEST_FOLDER, 'nnef2/onnx/ops/') if UNITTEST_FOLDER else None
    _optimize = True
    _execute = True

    def setUp(self) -> None:
        self._onnx_reader = onnx_io.Reader(simplify=False, enforce_output_shapes=True)
        self._onnx_writer = onnx_io.Writer()
        self._onnx_to_skriptnd_converter = onnx_to_nnef2.Converter()
        self._skriptnd_reader = skriptnd_io.Reader(atomics=True)
        self._skriptnd_writer = skriptnd_io.Writer(operators=onnx_to_nnef2.Converter.defined_operations(),
                                                   imports=onnx_to_nnef2.Converter.defined_imports(),
                                                   inline_subgraphs=False)
        self._skriptnd_optimizer = skriptnd_optimizer.Optimizer()
        self._onnx_optimizer = onnx_optimizer.Optimizer()

    def tearDown(self) -> None:
        pass

    def _convert_to_skriptnd(self, filename, input_shape=None):
        onnx_model = self._onnx_reader(filename)
        if self._optimize:
            self._onnx_optimizer(onnx_model)
        nnef_model = self._onnx_to_skriptnd_converter(onnx_model)
        if input_shape is not None:
            self._set_max_input_shapes(nnef_model, input_shape)
        output_filename = filename + '.nnef2'
        self._skriptnd_writer(nnef_model, output_filename)

    def _set_max_input_shapes(self, model, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape] * len(model.main.inputs)

        for idx, input in enumerate(model.main.inputs):
            shape = input_shape[idx]
            assert all(s is None or s == shape[i] for i, s in enumerate(input.shape))
            input.shape = tuple(s if s is not None else ShapeExpr(ShapeExpr.Op.Tilde, [shape[i]])
                                for i, s in enumerate(input.shape))

    @staticmethod
    def _random_data(dtype, shape, range=None):
        if dtype == np.bool_:
            return np.array(np.random.random(shape) > 0.5)
        elif dtype == np.float32 or dtype == np.float64:
            data = np.random.random(shape).astype(dtype)
            if range:
                lo, hi = range
                data *= hi - lo
                data += lo
            return data
        else:
            lo, hi = range if range else (0, 100)
            return np.random.randint(low=lo, high=hi, size=shape, dtype=dtype)

    @staticmethod
    def _exec_onnx_model(filename, input_shape=None, input_range=None):
        import onnxruntime
        np.random.seed(0)

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = onnxruntime.InferenceSession(filename, sess_options=options,
                                               providers=['CPUExecutionProvider'])

        if not isinstance(input_shape, list):
            input_shape = [input_shape] * len(session.get_inputs())
        if not isinstance(input_range, list):
            input_range = [input_range] * len(session.get_inputs())

        inputs = {input.name: TestEnv._random_data(TestEnv._onnx_type_to_numpy[input.type],
                                                   input_shape[idx] or input.shape, input_range[idx])
                  for idx, input in enumerate(session.get_inputs())}

        return session.run([output.name for output in session.get_outputs()], inputs)

    @staticmethod
    def _exec_skriptnd_model(path, input_shape=None, input_range=None):
        import skriptnd as nd
        np.random.seed(0)

        model = nd.read_model(path)
        if not model:
            return None

        compiled_model = nd.compile_model(model, keep_generated_code=False)

        if not isinstance(input_shape, list):
            input_shape = [input_shape] * len(model.graphs[0].inputs)
        if not isinstance(input_range, list):
            input_range = [input_range] * len(model.graphs[0].inputs)

        inputs = [TestEnv._random_data(TestEnv._skriptnd_dtype_to_numpy[input.dtype],
                                       input_shape[idx] or input.shape, input_range[idx])
                  for idx, input in enumerate(model.graphs[0].inputs)]

        return compiled_model(*inputs)

    def _compile_skriptnd_model(self, path):
        import skriptnd as nd

        model = nd.read_model(path)
        if not model:
            return None

        return nd.compile_model(model)

    @staticmethod
    def _create_tensor(value_info, data, range=None):
        name, shape, dtype = onnx_io.reader._get_value_info(value_info)
        if data is None:
            data = TestEnv._random_data(dtype, shape, range)
        elif not isinstance(data, np.ndarray):
            data = np.array(data)
        return helper.make_tensor(name, NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], shape, vals=data.flat)

    @staticmethod
    def _create_onnx_model(name, nodes, inputs, outputs, constants, values, ranges, opset_version, ir_version):
        tensors = [TestEnv._create_tensor(item, values.get(item.name), ranges.get(item.name) if isinstance(ranges, dict) else ranges)
                   for item in constants]
        graph_def = helper.make_graph(nodes, name, inputs, outputs, value_info=constants, initializer=tensors)
        model_def = helper.make_model(graph_def, producer_name='nnef-to-onnx-test')
        model_def.opset_import[0].version = opset_version
        model_def.ir_version = ir_version
        onnx.checker.check_model(model_def, full_check=True)
        return model_def

    @staticmethod
    def _save_onnx_model(model_def, filename):
        with open(filename, 'wb') as file:
            file.write(model_def.SerializeToString())

    def _test_conversion(self, name, nodes, inputs, outputs, constants=None, values=None, ranges=None,
                         opset_version=DefaultOpsetVersion, ir_version=DefaultIrVersion,
                         epsilon=1e-5, input_range=None):
        filename = tempfile.mktemp() if self._output_folder is None else TestEnv._output_folder + name + '.onnx'
        model_def = self._create_onnx_model(name, nodes, inputs, outputs, constants or [], values or {}, ranges or {},
                                            opset_version, ir_version)
        self._save_onnx_model(model_def, filename)
        self._test_conversion_from_file(filename, epsilon=epsilon, input_range=input_range)

    def _test_conversion_from_file(self, filename, epsilon=1e-5, input_shape=None, input_range=None):
        self._convert_to_skriptnd(filename, input_shape=input_shape)

        if not self._execute:
            assert self._compile_skriptnd_model(filename + '.nnef2') is not None
            return

        original_outputs = self._exec_onnx_model(filename, input_shape=input_shape, input_range=input_range)
        converted_outputs = self._exec_skriptnd_model(filename + '.nnef2', input_shape=input_shape, input_range=input_range)

        self.assertTrue(original_outputs is not None)
        self.assertTrue(converted_outputs is not None)

        self.assertEqual(len(original_outputs), len(converted_outputs))
        for idx, (original, converted) in enumerate(zip(original_outputs, converted_outputs)):
            self.assertEqual(original.shape, converted.shape)
            if all(s != 0 for s in original.shape):
                if original.dtype != np.float32 and original.dtype != np.float64:
                    self.assertTrue(np.all(original == converted))
                else:
                    max = np.max(np.abs(original))
                    diff = np.max(np.abs(original - converted))
                    print("Max absolute difference for output #{}: {}".format(idx, diff))
                    if max != 0:
                        print("Max relative difference for output #{}: {}".format(idx, diff / max))
                    self.assertLess(diff / max if max != 0 else diff, epsilon)

    def _test_unary(self, op_type, dtype=TensorProto.FLOAT, input_range=None, opset_version=DefaultOpsetVersion):
        input = helper.make_tensor_value_info('input', dtype, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', dtype, [1, 3, 32, 32])
        node = helper.make_node(
            op_type=op_type,
            inputs=['input'],
            outputs=['output'],
        )

        self._test_conversion(op_type.lower(), [node], [input], [output],
                              input_range=input_range, opset_version=opset_version)

    def _test_binary(self, op_type, input_dtype=TensorProto.FLOAT, output_dtype=TensorProto.FLOAT,
                     input_range=None, opset_version=DefaultOpsetVersion):
        input1 = helper.make_tensor_value_info('input1', input_dtype, [1, 3, 32, 32])
        input2 = helper.make_tensor_value_info('input2', input_dtype, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', output_dtype, [1, 3, 32, 32])
        node = helper.make_node(
            op_type=op_type,
            inputs=['input1', 'input2'],
            outputs=['output'],
        )

        self._test_conversion(op_type.lower(), [node], [input1, input2], [output],
                              input_range=input_range, opset_version=opset_version)

    def _test_reduce(self, op_type, keepdims, dtype=TensorProto.FLOAT, opset_version=DefaultOpsetVersion):
        input = helper.make_tensor_value_info('input', dtype, [1, 16, 32, 32])
        output = helper.make_tensor_value_info('output', dtype, [1, 1, 32, 32] if keepdims else [1, 32, 32])
        node = helper.make_node(
            op_type=op_type,
            inputs=['input'],
            outputs=['output'],
            axes=[1],
            keepdims=keepdims,
        )

        self._test_conversion(op_type.lower() + ('_keepdims' if keepdims else ''), [node], [input], [output],
                              opset_version=opset_version)

    def _test_reduce_dynamic(self, op_type, keepdims, dtype=TensorProto.FLOAT, opset_version=DefaultOpsetVersion):
        input = helper.make_tensor_value_info('input', dtype, [1, 16, 32, 32])
        axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [1])
        output = helper.make_tensor_value_info('output', dtype, [1, 1, 32, 32] if keepdims else [1, 32, 32])
        node = helper.make_node(
            op_type=op_type,
            inputs=['input', 'axes'],
            outputs=['output'],
            keepdims=keepdims,
        )

        self._test_conversion(op_type.lower() + ('_keepdims' if keepdims else ''), [node], [input], [output],
                              constants=[axes], values={'axes': [1]}, opset_version=opset_version)


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

    def test_conv2d_wide(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        filter = helper.make_tensor_value_info('filter', TensorProto.FLOAT, [16, 3, 15, 15])
        bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [16])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
        node = helper.make_node(
            op_type='Conv',
            inputs=['input', 'filter', 'bias'],
            outputs=['output'],
            auto_pad='SAME_UPPER',
        )

        self._test_conversion('conv2d_wide', [node], [input], [output], constants=[filter, bias])

    def test_conv2d_deep(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1024, 32, 32])
        filter = helper.make_tensor_value_info('filter', TensorProto.FLOAT, [256, 1024, 3, 3])
        bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [256])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 256, 32, 32])
        node = helper.make_node(
            op_type='Conv',
            inputs=['input', 'filter', 'bias'],
            outputs=['output'],
            auto_pad='SAME_UPPER',
        )

        self._test_conversion('conv2d_deep', [node], [input], [output], constants=[filter, bias])

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

    def test_conv2d_strided(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        filter = helper.make_tensor_value_info('filter', TensorProto.FLOAT, [16, 3, 3, 3])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 16, 16])
        node = helper.make_node(
            op_type='Conv',
            inputs=['input', 'filter'],
            outputs=['output'],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
        )

        self._test_conversion('conv2d-strided', [node], [input], [output], constants=[filter])

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

        self._test_conversion('conv2d_transpose', [node], [input], [output], constants=[filter, bias], epsilon=1e-3)

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

        self._test_conversion('conv2d_transpose-output_shape', [node], [input], [output], constants=[filter, bias], epsilon=1e-3)

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

        self._test_conversion('conv3d', [node], [input], [output], constants=[filter, bias], epsilon=1e-3)

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

    def test_avg_pool2d_strided(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 16, 16])
        node = helper.make_node(
            op_type='AveragePool',
            inputs=['input'],
            outputs=['output'],
            kernel_shape=[3, 3],
            strides=[2, 2],
            auto_pad='SAME_UPPER',
        )

        self._test_conversion('avg_pool2d_strided', [node], [input], [output])

    def test_avg_pool2d_include_pad(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='AveragePool',
            inputs=['input'],
            outputs=['output'],
            kernel_shape=[5, 5],
            auto_pad='SAME_UPPER',
            count_include_pad=1,
        )

        self._test_conversion('avg_pool2d_include_pad', [node], [input], [output])

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

    def test_dynamic_reshape(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        index0 = helper.make_tensor_value_info('index0', TensorProto.INT32, [1])
        index1 = helper.make_tensor_value_info('index1', TensorProto.INT32, [1])
        index2 = helper.make_tensor_value_info('index2', TensorProto.INT32, [1])
        index3 = helper.make_tensor_value_info('index3', TensorProto.INT32, [1])
        indentity = helper.make_node(
            op_type='Identity',
            inputs=['input'],
            outputs=['preproc'],
        )
        shape = helper.make_node(
            op_type='Shape',
            inputs=['preproc'],
            outputs=['shape'],
        )
        gather0 = helper.make_node(
            op_type='Gather',
            inputs=['shape', 'index0'],
            outputs=['shape0'],
            axis=0,
        )
        gather1 = helper.make_node(
            op_type='Gather',
            inputs=['shape', 'index1'],
            outputs=['shape1'],
            axis=0,
        )
        gather2 = helper.make_node(
            op_type='Gather',
            inputs=['shape', 'index2'],
            outputs=['shape2'],
            axis=0,
        )
        gather3 = helper.make_node(
            op_type='Gather',
            inputs=['shape', 'index3'],
            outputs=['shape3'],
            axis=0,
        )
        concat = helper.make_node(
            op_type='Concat',
            inputs=['shape0', 'shape1', 'shape2', 'shape3'],
            outputs=['concat'],
            axis=0,
        )
        reshape = helper.make_node(
            op_type='Reshape',
            inputs=['input', 'concat'],
            outputs=['output'],
        )

        self._test_conversion('dynamic_reshape', [indentity, shape, gather0, gather1, gather2, gather3, concat, reshape],
                              [input], [output], constants=[index0, index1, index2, index3],
                              values={'index0': [0], 'index1': [1], 'index2': [2], 'index3': [3]})

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
        axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [1])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 32, 32])
        node = helper.make_node(
            op_type='Squeeze',
            inputs=['input', 'axes'],
            outputs=['output'],
        )

        self._test_conversion('squeeze', [node], [input], [output], constants=[axes], values={'axes': [1]})

    def test_unsqueeze(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 32, 32])
        axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [1])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 32, 32])
        node = helper.make_node(
            op_type='Unsqueeze',
            inputs=['input', 'axes'],
            outputs=['output'],
        )

        self._test_conversion('unsqueeze', [node], [input], [output], constants=[axes], values={'axes': [1]})

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

        self._test_conversion('lrn', [node], [input], [output], epsilon=1e-3)

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
        split = helper.make_tensor_value_info('split', TensorProto.INT64, [2])
        output1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [1, 3, 32, 32])
        output2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='Split',
            inputs=['input', 'split'],
            outputs=['output1', 'output2'],
            axis=1,
        )

        self._test_conversion('split', [node], [input], [output1, output2], constants=[split],
                              values={'split': [3, 3]})

    def test_split_equal(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 6, 32, 32])
        output1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [1, 3, 32, 32])
        output2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='Split',
            inputs=['input'],
            outputs=['output1', 'output2'],
            axis=1,
        )

        self._test_conversion('split_equal', [node], [input], [output1, output2])

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

    def test_thresholded_relu(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
        node = helper.make_node(
            op_type='ThresholdedRelu',
            inputs=['input'],
            outputs=['output'],
            alpha=0.1,
        )

        self._test_conversion('thresholded_relu', [node], [input], [output])

    def test_prelu(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 32, 32])
        alpha = helper.make_tensor_value_info('alpha', TensorProto.FLOAT, [16, 1, 1])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])
        node = helper.make_node(
            op_type='PRelu',
            inputs=['input', 'alpha'],
            outputs=['output'],
        )

        self._test_conversion('prelu', [node], [input], [output], constants=[alpha])

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

        self._test_conversion('clip', [node], [input], [output], constants=[min, max], values={'min': 0.25, 'max': 0.75})

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

    def test_pad_zero(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 34, 34])
        pads = helper.make_tensor_value_info('pads', TensorProto.INT64, [8])
        node = helper.make_node(
            op_type='Pad',
            inputs=['input', 'pads'],
            outputs=['output'],
        )

        self._test_conversion('pad-zero', [node], [input], [output], constants=[pads],
                              values={'pads': [0, 0, 1, 1, 0, 0, 1, 1]})

    def test_pad_constant(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 34, 34])
        axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [2])
        pads = helper.make_tensor_value_info('pads', TensorProto.INT64, [4])
        value = helper.make_tensor_value_info('value', TensorProto.FLOAT, [])
        node = helper.make_node(
            op_type='Pad',
            inputs=['input', 'pads', 'value', 'axes'],
            outputs=['output'],
        )

        self._test_conversion('pad-constant', [node], [input], [output], constants=[pads, value, axes],
                              values={'pads': [1, 1, 1, 1], 'value': 42.0, 'axes': [2, 3]}, opset_version=19)

    def test_pad_reflect(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 34, 34])
        pads = helper.make_tensor_value_info('pads', TensorProto.INT64, [8])
        node = helper.make_node(
            op_type='Pad',
            inputs=['input', 'pads'],
            outputs=['output'],
            mode="reflect",
        )

        self._test_conversion('pad-reflect', [node], [input], [output], constants=[pads],
                              values={'pads': [0, 0, 1, 1, 0, 0, 1, 1]})

    def test_pad_replicate(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 34, 34])
        pads = helper.make_tensor_value_info('pads', TensorProto.INT64, [8])
        node = helper.make_node(
            op_type='Pad',
            inputs=['input', 'pads'],
            outputs=['output'],
            mode="edge",
        )

        self._test_conversion('pad-replicate', [node], [input], [output], constants=[pads],
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

    def test_slice_without_end(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 31, 31])
        starts = helper.make_tensor_value_info('starts', TensorProto.INT64, [2])
        ends = helper.make_tensor_value_info('ends', TensorProto.INT64, [2])
        axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [2])
        node = helper.make_node(
            op_type='Slice',
            inputs=['input', 'starts', 'ends', 'axes'],
            outputs=['output'],
        )

        self._test_conversion('slice-wo-end', [node], [input], [output], constants=[starts, ends, axes],
                              values={'starts': [1, 1], 'ends': [65535, 65535], 'axes': [2, 3]})

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

    def test_linear_upsample2x(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 64, 64])
        node = helper.make_node(
            op_type='Upsample',
            inputs=['input'],
            outputs=['output'],
            scales=[1.0, 1.0, 2.0, 2.0],
            mode='linear',
        )

        self._test_conversion('linear_upsample2x', [node], [input], [output], opset_version=8)

    def test_linear_upsample3x(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 96, 96])
        node = helper.make_node(
            op_type='Upsample',
            inputs=['input'],
            outputs=['output'],
            scales=[1.0, 1.0, 3.0, 3.0],
            mode='linear',
        )

        self._test_conversion('linear_upsample3x', [node], [input], [output], opset_version=8)

    def test_linear_upsample4x(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 128, 128])
        node = helper.make_node(
            op_type='Upsample',
            inputs=['input'],
            outputs=['output'],
            scales=[1.0, 1.0, 4.0, 4.0],
            mode='linear',
        )

        self._test_conversion('linear_upsample4x', [node], [input], [output], opset_version=8)

    def test_linear_upsample5x(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 160, 160])
        node = helper.make_node(
            op_type='Upsample',
            inputs=['input'],
            outputs=['output'],
            scales=[1.0, 1.0, 5.0, 5.0],
            mode='linear',
        )

        self._test_conversion('linear_upsample5x', [node], [input], [output], opset_version=8)

    def test_resize_nearest_upsample(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 64, 64])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', '', 'scales'],
            outputs=['output'],
            mode='nearest',
        )

        self._test_conversion('resize_nearest_upsample', [node], [input], [output], constants=[scales],
                              values={'scales': [1.0, 1.0, 2.0, 2.0]})

    def test_resize_linear_upsample2x(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 64, 64])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', '', 'scales'],
            outputs=['output'],
            mode='linear',
        )

        self._test_conversion('resize_liner_upsample2x', [node], [input], [output], constants=[scales],
                              values={'scales': [1.0, 1.0, 2.0, 2.0]})

    def test_resize_linear_upsample3x(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 96, 96])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', '', 'scales'],
            outputs=['output'],
            mode='linear',
        )

        self._test_conversion('resize_liner_upsample3x', [node], [input], [output], constants=[scales],
                              values={'scales': [1.0, 1.0, 3.0, 3.0]})

    def test_resize_linear_upsample4x(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 128, 128])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', '', 'scales'],
            outputs=['output'],
            mode='linear',
        )

        self._test_conversion('resize_liner_upsample4x', [node], [input], [output], constants=[scales],
                              values={'scales': [1.0, 1.0, 4.0, 4.0]})

    def test_resize_linear_upsample5x(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 160, 160])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', '', 'scales'],
            outputs=['output'],
            mode='linear',
        )

        self._test_conversion('resize_liner_upsample5x', [node], [input], [output], constants=[scales],
                              values={'scales': [1.0, 1.0, 5.0, 5.0]})

    def test_resize_nearest_downsample(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 16, 16])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', '', 'scales'],
            outputs=['output'],
            mode='nearest',
        )

        self._test_conversion('resize_nearest_downsample', [node], [input], [output], constants=[scales],
                              values={'scales': [1.0, 1.0, 0.5, 0.5]})

    def test_resize_nearest_symmetric(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        sizes = helper.make_tensor_value_info('sizes', TensorProto.INT64, [4])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 25, 25])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', '', '', 'sizes'],
            outputs=['output'],
            mode='nearest',
            coordinate_transformation_mode='half_pixel',
        )

        self._test_conversion('resize_nearest_symmetric', [node], [input], [output], constants=[sizes],
                              values={'sizes': [1, 3, 25, 25]})

    def test_resize_nearest_asymmetric(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        sizes = helper.make_tensor_value_info('sizes', TensorProto.INT64, [4])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 25, 25])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', '', '', 'sizes'],
            outputs=['output'],
            mode='nearest',
            coordinate_transformation_mode='asymmetric',
        )

        self._test_conversion('resize_nearest_asymmetric', [node], [input], [output], constants=[sizes],
                              values={'sizes': [1, 3, 25, 25]})

    def test_resize_nearest_aligned(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        sizes = helper.make_tensor_value_info('sizes', TensorProto.INT64, [4])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 25, 25])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', '', '', 'sizes'],
            outputs=['output'],
            mode='nearest',
            coordinate_transformation_mode='align_corners',
        )

        self._test_conversion('resize_nearest_aligned', [node], [input], [output], constants=[sizes],
                              values={'sizes': [1, 3, 25, 25]})

    def test_resize_nearest_ceil(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        sizes = helper.make_tensor_value_info('sizes', TensorProto.INT64, [4])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 25, 25])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', '', '', 'sizes'],
            outputs=['output'],
            mode='nearest',
            coordinate_transformation_mode='asymmetric',
            nearest_mode='ceil',
        )

        self._test_conversion('resize_nearest_ceil', [node], [input], [output], constants=[sizes],
                              values={'sizes': [1, 3, 25, 25]})

    def test_resize_linear_aligned(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        sizes = helper.make_tensor_value_info('sizes', TensorProto.INT64, [4])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 25, 25])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', '', '', 'sizes'],
            outputs=['output'],
            mode='linear',
            coordinate_transformation_mode='align_corners',
        )

        self._test_conversion('resize_linear_aligned', [node], [input], [output], constants=[sizes],
                              values={'sizes': [1, 3, 25, 25]})

    def test_resize_linear_symmetric(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        sizes = helper.make_tensor_value_info('sizes', TensorProto.INT64, [4])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 25, 25])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', '', '', 'sizes'],
            outputs=['output'],
            mode='linear',
            coordinate_transformation_mode='half_pixel',
        )

        self._test_conversion('resize_linear_symmetric', [node], [input], [output], constants=[sizes],
                              values={'sizes': [1, 3, 25, 25]})

    def test_resize_linear_asymmetric(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        sizes = helper.make_tensor_value_info('sizes', TensorProto.INT64, [4])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 25, 25])
        node = helper.make_node(
            op_type='Resize',
            inputs=['input', '', '', 'sizes'],
            outputs=['output'],
            mode='linear',
            coordinate_transformation_mode='asymmetric',
        )

        self._test_conversion('resize_linear_asymmetric', [node], [input], [output], constants=[sizes],
                              values={'sizes': [1, 3, 25, 25]})

    def test_cast(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.INT32, [1, 4, 32, 32])
        node = helper.make_node(
            op_type='Cast',
            inputs=['input'],
            outputs=['output'],
            to=TensorProto.INT32,
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

        self._test_conversion('gather', [node], [input, indices], [output], input_range=(0, 16))

    def test_gather_nd(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 16, 16, 8])
        indices = helper.make_tensor_value_info('indices', TensorProto.INT64, [4, 32, 2])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 32, 8])
        node = helper.make_node(
            op_type='GatherND',
            inputs=['input', 'indices'],
            outputs=['output'],
            batch_dims=1,
        )

        self._test_conversion('gather_nd', [node], [input, indices], [output], input_range=(0, 16))

    def test_scatter(self):
        data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [8, 16])
        indices = helper.make_tensor_value_info('indices', TensorProto.INT32, [8, 12])
        updates = helper.make_tensor_value_info('updates', TensorProto.FLOAT, [8, 12])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [8, 16])
        node = helper.make_node(
            op_type='Scatter',
            inputs=['data', 'indices', 'updates'],
            outputs=['output'],
            axis=1,
        )

        self._test_conversion('scatter', [node], [data, indices, updates], [output],
                              input_range=(0, 16), opset_version=9)

    def test_scatter_nd(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 8, 16])
        indices = helper.make_tensor_value_info('indices', TensorProto.INT64, [2, 1])
        updates = helper.make_tensor_value_info('updates', TensorProto.FLOAT, [2, 8, 16])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 8, 16])
        node = helper.make_node(
            op_type='ScatterND',
            inputs=['input', 'indices', 'updates'],
            outputs=['output'],
        )

        self._test_conversion('scatter_nd', [node], [input, indices, updates], [output],
                              input_range=(0, 4))

    def test_if(self):
        cond = helper.make_tensor_value_info('cond', TensorProto.BOOL, [])
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [1, 16, 32, 32])
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [1, 16, 32, 32])
        output1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [1, 16, 32, 32])
        output2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, [1, 16, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 32, 32])

        then_node = helper.make_node(
            op_type='Add',
            inputs=['input1', 'input2'],
            outputs=['output1'],
        )
        then_graph = helper.make_graph([then_node], "then-graph", [], [output1])

        else_node = helper.make_node(
            op_type='Sub',
            inputs=['input1', 'input2'],
            outputs=['output2'],
        )
        else_graph = helper.make_graph([else_node], "else-graph", [], [output2])

        node = helper.make_node(
            op_type='If',
            inputs=['cond'],
            outputs=['output'],
            then_branch=then_graph,
            else_branch=else_graph,
        )

        self._test_conversion('if_then', [node], [cond, input1, input2], [output])

    def test_while_loop(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [])
        prod_in = helper.make_tensor_value_info('prod_in', TensorProto.FLOAT, [])
        prod_out = helper.make_tensor_value_info('prod_out', TensorProto.FLOAT, [])
        cond_in = helper.make_tensor_value_info('cond_in', TensorProto.BOOL, [])
        cond_out = helper.make_tensor_value_info('cond_out', TensorProto.BOOL, [])
        cond = helper.make_tensor_value_info('cond', TensorProto.BOOL, [])
        two = helper.make_tensor_value_info('two', TensorProto.FLOAT, [])
        hundred = helper.make_tensor_value_info('hundred', TensorProto.FLOAT, [])
        iter_count = helper.make_tensor_value_info('iter', TensorProto.INT64, [])

        mul_node = helper.make_node(
            op_type='Mul',
            inputs=['prod_in', 'two'],
            outputs=['prod_out'],
        )
        cmp_node = helper.make_node(
            op_type='Less',
            inputs=['prod_out', 'hundred'],
            outputs=['cond_out'],
        )
        body_graph = helper.make_graph([mul_node, cmp_node], "body-graph", [iter_count, cond_in, prod_in], [cond_out, prod_out],
                                       value_info=[two, hundred],
                                       initializer=[TestEnv._create_tensor(two, 2.0), TestEnv._create_tensor(hundred, 100.0)])

        node = helper.make_node(
            op_type='Loop',
            inputs=['', 'cond', 'input'],
            outputs=['output'],
            body=body_graph,
        )

        self._test_conversion('while_loop', [node], [input], [output], constants=[cond], values={'cond': True})

    def test_for_loop(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [])
        prod_in = helper.make_tensor_value_info('prod_in', TensorProto.FLOAT, [])
        prod_out = helper.make_tensor_value_info('prod_out', TensorProto.FLOAT, [])
        cond = helper.make_tensor_value_info('cond', TensorProto.BOOL, [])
        cond_in = helper.make_tensor_value_info('cond_in', TensorProto.BOOL, [])
        cond_out = helper.make_tensor_value_info('cond_out', TensorProto.BOOL, [])
        two = helper.make_tensor_value_info('two', TensorProto.FLOAT, [])
        count = helper.make_tensor_value_info('count', TensorProto.INT64, [])
        index = helper.make_tensor_value_info('index', TensorProto.INT64, [])

        mul_node = helper.make_node(
            op_type='Mul',
            inputs=['prod_in', 'two'],
            outputs=['prod_out'],
        )
        iden_node = onnx.helper.make_node(
            'Identity',
            inputs=['cond_in'],
            outputs=['cond_out']
        )
        body_graph = helper.make_graph([mul_node, iden_node], "body-graph", [index, cond_in, prod_in], [cond_out, prod_out],
                                       value_info=[two], initializer=[TestEnv._create_tensor(two, 2.0)])

        node = helper.make_node(
            op_type='Loop',
            inputs=['count', 'cond', 'input'],
            outputs=['output'],
            body=body_graph,
        )

        self._test_conversion('for_loop', [node], [input], [output], constants=[count, cond], values={'count': 10, 'cond': True})

    def test_lstm_static(self):
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [5, 4, 32])
        W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 256, 32])
        R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 256, 64])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 512])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, 1, 4, 64])
        h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [1, 4, 64])
        c0 = helper.make_tensor_value_info('c0', TensorProto.FLOAT, [1, 4, 64])
        hn = helper.make_tensor_value_info('hn', TensorProto.FLOAT, [1, 4, 64])
        cn = helper.make_tensor_value_info('cn', TensorProto.FLOAT, [1, 4, 64])
        node = helper.make_node(
            op_type='LSTM',
            inputs=['X', 'W', 'R', 'B', '', 'h0', 'c0'],
            outputs=['Y', 'hn', 'cn'],
            hidden_size=64,
            direction="forward",
        )

        self._test_conversion('lstm_static', [node], [X, h0, c0], [Y, hn, cn],
                              constants=[W, R, B])

    def test_lstm_dynamic(self):
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [5, 4, 32])
        W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 256, 32])
        R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 256, 64])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 512])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, 1, 4, 64])
        h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, [1, 4, 64])
        c0 = helper.make_tensor_value_info('c0', TensorProto.FLOAT, [1, 4, 64])
        hn = helper.make_tensor_value_info('hn', TensorProto.FLOAT, [1, 4, 64])
        cn = helper.make_tensor_value_info('cn', TensorProto.FLOAT, [1, 4, 64])
        lens = helper.make_tensor_value_info('lens', TensorProto.INT32, [4])
        node = helper.make_node(
            op_type='LSTM',
            inputs=['X', 'W', 'R', 'B', 'lens', 'h0', 'c0'],
            outputs=['Y', 'hn', 'cn'],
            hidden_size=64,
            direction="forward",
        )

        self._test_conversion('lstm_dynamic', [node], [X, h0, c0], [Y, hn, cn],
                              constants=[W, R, B, lens], values={'lens': [2, 5, 4, 3]})

    def test_nonzero(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [10, 20, 30])
        indices = helper.make_tensor_value_info('indices', TensorProto.INT64, [3, None])
        node = helper.make_node(
            op_type='NonZero',
            inputs=['input'],
            outputs=['indices'],
        )

        self._test_conversion('nonzero', [node], [input], [indices])

    def test_top_k(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [10, 20, 30])
        k = helper.make_tensor_value_info('k', TensorProto.INT64, [1])
        values = helper.make_tensor_value_info('values', TensorProto.FLOAT, [10, 5, 30])
        indices = helper.make_tensor_value_info('indices', TensorProto.INT64, [10, 5, 30])
        node = helper.make_node(
            op_type='TopK',
            inputs=['input', 'k'],
            outputs=['values', 'indices'],
            axis=1,
        )

        self._test_conversion('top_k', [node], [input, k], [values, indices], constants=[k], values={'k': [5]})

    def test_nms(self):
        boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [16, 1024, 4])
        scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [16, 10, 1024])
        max_boxes = helper.make_tensor_value_info('max_boxes', TensorProto.INT64, [])
        iou_threshold = helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT, [])
        score_threshold = helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT, [])
        indices = helper.make_tensor_value_info('indices', TensorProto.INT64, [None, 3])
        node = helper.make_node(
            op_type='NonMaxSuppression',
            inputs=['boxes', 'scores', 'max_boxes', 'iou_threshold', 'score_threshold'],
            outputs=['indices'],
        )

        self._test_conversion('nms', [node], [boxes, scores], [indices],
                              constants=[max_boxes, iou_threshold, score_threshold],
                              values={'max_boxes': 5, 'iou_threshold': 0.25, 'score_threshold': 0.1})

    def test_nms_nomax(self):
        boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [16, 1024, 4])
        scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [16, 10, 1024])
        iou_threshold = helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT, [])
        score_threshold = helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT, [])
        indices = helper.make_tensor_value_info('indices', TensorProto.INT64, [None, 3])
        node = helper.make_node(
            op_type='NonMaxSuppression',
            inputs=['boxes', 'scores', '', 'iou_threshold', 'score_threshold'],
            outputs=['indices'],
        )

        self._test_conversion('nms_nomax', [node], [boxes, scores], [indices],
                              constants=[iou_threshold, score_threshold],
                              values={'iou_threshold': 0.25, 'score_threshold': 0.1})

    def test_nms_centered(self):
        boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [16, 1024, 4])
        scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [16, 10, 1024])
        max_boxes = helper.make_tensor_value_info('max_boxes', TensorProto.INT64, [])
        iou_threshold = helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT, [])
        score_threshold = helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT, [])
        indices = helper.make_tensor_value_info('indices', TensorProto.INT64, [None, 3])
        node = helper.make_node(
            op_type='NonMaxSuppression',
            inputs=['boxes', 'scores', 'max_boxes', 'iou_threshold', 'score_threshold'],
            outputs=['indices'],
            center_point_box=1,
        )

        self._test_conversion('nms_centered', [node], [boxes, scores], [indices],
                              constants=[max_boxes, iou_threshold, score_threshold],
                              values={'max_boxes': 5, 'iou_threshold': 0.25, 'score_threshold': 0.1})

    def test_min_reduce(self):
        self._test_reduce('ReduceMin', keepdims=False)

    def test_max_reduce(self):
        self._test_reduce('ReduceMax', keepdims=False)

    def test_mean_reduce(self):
        self._test_reduce('ReduceMean', keepdims=False)

    def test_sum_reduce(self):
        self._test_reduce_dynamic('ReduceSum', keepdims=False)

    def test_prod_reduce(self):
        self._test_reduce('ReduceProd', keepdims=False)

    def test_l1_reduce(self):
        self._test_reduce('ReduceL1', keepdims=False)

    def test_l2_reduce(self):
        self._test_reduce('ReduceL2', keepdims=False)

    def test_max_reduce_keepdims(self):
        self._test_reduce('ReduceMax', keepdims=True)

    def test_identity(self):
        self._test_unary('Identity')

    def test_relu(self):
        self._test_unary('Relu')

    def test_sigmoid(self):
        self._test_unary('Sigmoid')

    def test_softplus(self):
        self._test_unary('Softplus')

    def test_selu(self):
        self._test_unary('Selu')

    def test_not(self):
        self._test_unary('Not', dtype=TensorProto.BOOL)

    def test_elu(self):
        self._test_unary('Elu')

    def test_gelu(self):
        self._test_unary('Gelu', opset_version=20)

    def test_gelu_approx(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='Gelu',
            inputs=['input'],
            outputs=['output'],
            approximate='tanh',
        )

        self._test_conversion('gelu_approx', [node], [input], [output],
                              opset_version=20)

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

    def test_asinh(self):
        self._test_unary('Asinh')

    def test_acosh(self):
        self._test_unary('Acosh', input_range=(1.0, 10.0))

    def test_atanh(self):
        self._test_unary('Atanh')

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
        self._test_binary('Div', input_range=(1, 100))

    def test_mod(self):
        self._test_binary('Mod', input_dtype=TensorProto.INT64, output_dtype=TensorProto.INT64, input_range=(1, 100))

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

    def test_xor(self):
        self._test_binary('Xor', input_dtype=TensorProto.BOOL, output_dtype=TensorProto.BOOL)

    def test_equal(self):
        self._test_binary('Equal', output_dtype=TensorProto.BOOL)

    def test_less(self):
        self._test_binary('Less', output_dtype=TensorProto.BOOL)

    def test_greater(self):
        self._test_binary('Greater', output_dtype=TensorProto.BOOL)

    def test_mod_real(self):
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [1, 3, 32, 32])
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [1, 3, 32, 32])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node(
            op_type='Mod',
            inputs=['input1', 'input2'],
            outputs=['output'],
            fmod=1,
        )

        self._test_conversion('mod_real', [node], [input1, input2], [output], input_range=(1, 100))

    def test_nonzero_gather(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [100])
        zero = helper.make_tensor_value_info('zero', TensorProto.FLOAT, [])
        greater = helper.make_tensor_value_info('greater', TensorProto.BOOL, [100])
        indices = helper.make_tensor_value_info('indices', TensorProto.INT64, [1, None])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, None])
        greater_node = helper.make_node(
            op_type='Greater',
            inputs=['input', 'zero'],
            outputs=['greater'],
        )
        nz_node = helper.make_node(
            op_type='NonZero',
            inputs=['greater'],
            outputs=['indices'],
        )
        gather_node = helper.make_node(
            op_type='Gather',
            inputs=['input', 'indices'],
            outputs=['output'],
            axis=0,
        )

        self._test_conversion('nonzero_gather', [greater_node, nz_node, gather_node], [input], [output],
                              input_range=(-1,1), constants=[zero], values={'zero': 0.0})

    def test_nonzero_gather_concat_split(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [100])
        zero = helper.make_tensor_value_info('zero', TensorProto.FLOAT, [])
        greater = helper.make_tensor_value_info('greater', TensorProto.BOOL, [100])
        less = helper.make_tensor_value_info('less', TensorProto.BOOL, [100])
        indices1 = helper.make_tensor_value_info('indices1', TensorProto.INT64, [1, None])
        indices2 = helper.make_tensor_value_info('indices2', TensorProto.INT64, [1, None])
        gather1 = helper.make_tensor_value_info('gather1', TensorProto.FLOAT, [1, None])
        gather2 = helper.make_tensor_value_info('gather2', TensorProto.FLOAT, [1, None])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, None])
        output1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [1, None])
        output2 = helper.make_tensor_value_info('output2', TensorProto.FLOAT, [1, None])
        greater_node = helper.make_node(
            op_type='Greater',
            inputs=['input', 'zero'],
            outputs=['greater'],
        )
        less_node = helper.make_node(
            op_type='LessOrEqual',
            inputs=['input', 'zero'],
            outputs=['less'],
        )
        nz_node1 = helper.make_node(
            op_type='NonZero',
            inputs=['greater'],
            outputs=['indices1'],
        )
        nz_node2 = helper.make_node(
            op_type='NonZero',
            inputs=['less'],
            outputs=['indices2'],
        )
        gather_node1 = helper.make_node(
            op_type='Gather',
            inputs=['input', 'indices1'],
            outputs=['gather1'],
            axis=0,
        )
        gather_node2 = helper.make_node(
            op_type='Gather',
            inputs=['input', 'indices2'],
            outputs=['gather2'],
            axis=0,
        )
        concat_node = helper.make_node(
            op_type='Concat',
            inputs=['gather1', 'gather2'],
            outputs=['output'],
            axis=1,
        )
        split_node = helper.make_node(
            op_type='Split',
            inputs=['output'],
            outputs=['output1', 'output2'],
            axis=1,
            num_outputs=2,
        )

        self._test_conversion('nonzero_gather_concat_split',
                              [greater_node, less_node, nz_node1, nz_node2, gather_node1, gather_node2, concat_node, split_node],
                              [input], [output, output1, output2],
                              input_range=(-1,1), constants=[zero], values={'zero': 0.0}, opset_version=18)

    def test_constant_of_shape(self):
        shape = helper.make_tensor_value_info('shape', TensorProto.INT64, [2])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [10,20])
        value = helper.make_tensor('value', TensorProto.FLOAT, [1], [42])
        node = helper.make_node(
            op_type='ConstantOfShape',
            inputs=['shape'],
            outputs=['output'],
            value=value,
        )

        self._test_conversion('constant', [node], [], [output],
                              constants=[shape], values={'shape': [10, 20]})

    def test_range(self):
        start = helper.make_tensor_value_info('start', TensorProto.FLOAT, [])
        limit = helper.make_tensor_value_info('limit', TensorProto.FLOAT, [])
        delta = helper.make_tensor_value_info('delta', TensorProto.FLOAT, [])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [5])
        node = helper.make_node(
            op_type='Range',
            inputs=['start', 'limit', 'delta'],
            outputs=['output'],
        )

        self._test_conversion('range', [node], [], [output],
                              constants=[start, limit, delta],
                              values={'start': 1.0, 'limit': 10.0, 'delta': 2.0})

    def test_shape(self):
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 8, 8])
        output = helper.make_tensor_value_info('output', TensorProto.INT64, [4])
        node = helper.make_node(
            op_type='Shape',
            inputs=['input'],
            outputs=['output'],
        )

        self._test_conversion('shape', [node], [input], [output])


@unittest.skipIf(TestEnv._network_folder is None or not os.path.isdir(TestEnv._network_folder),
                 "no network test folder provided")
class NetworkTestCases(TestEnv):

    def test_alexnet(self):
        self._test_conversion_from_file(self._network_folder + 'alexnet.onnx')

    def test_caffenet(self):
        self._test_conversion_from_file(self._network_folder + 'caffenet.onnx')

    def test_vgg16(self):
        self._test_conversion_from_file(self._network_folder + 'vgg16.onnx')

    def test_googlenet(self):
        self._test_conversion_from_file(self._network_folder + 'googlenet.onnx')

    def test_inception_v1(self):
        self._test_conversion_from_file(self._network_folder + 'inception_v1.onnx')

    def test_inception_v2(self):
        self._test_conversion_from_file(self._network_folder + 'inception_v2.onnx')

    def test_mobilenet_v1(self):
        self._test_conversion_from_file(self._network_folder + 'mobilenet_v1.onnx',
                                        input_shape=(1, 3, 224, 224), epsilon=1e-4)

    def test_mobilenet_v2(self):
        self._test_conversion_from_file(self._network_folder + 'mobilenet_v2.onnx', epsilon=1e-4)

    def test_resnet50_v1(self):
        self._test_conversion_from_file(self._network_folder + 'resnet50_v1.onnx',
                                        input_shape=(1, 3, 224, 224))

    def test_resnet50_v2(self):
        self._test_conversion_from_file(self._network_folder + 'resnet50_v2.onnx')

    def test_squeezenet_v1(self):
        self._test_conversion_from_file(self._network_folder + 'squeezenet_v1.onnx')

    def test_shufflenet_v1(self):
        self._test_conversion_from_file(self._network_folder + 'shufflenet_v1.onnx')

    def test_shufflenet_v2(self):
        self._test_conversion_from_file(self._network_folder + 'shufflenet_v2.onnx')

    def test_densenet(self):
        self._test_conversion_from_file(self._network_folder + 'densenet.onnx')

    def test_retinanet(self):
        self._test_conversion_from_file(self._network_folder + 'retinanet.onnx')

    def test_resnext50(self):
        self._test_conversion_from_file(self._network_folder + 'resnext50.onnx')

    def test_ssd(self):
        self._test_conversion_from_file(self._network_folder + 'ssd.onnx')

    def test_ssd_resnet34(self):
        self._test_conversion_from_file(self._network_folder + 'ssd_resnet34.fixed.onnx',
                                        input_shape=(1, 3, 1200, 1200))

    def test_ssd_resnet34_noloop(self):
        self._test_conversion_from_file(self._network_folder + 'ssd_resnet34_noloop.onnx')

    def test_yolo_v3(self):
        self._test_conversion_from_file(self._network_folder + 'yolo_v3.onnx',
                                        input_shape=[(1, 3, 224, 224), (1, 2)])

    def test_gpt_encoder(self):
        self._test_conversion_from_file(self._network_folder + 'gpt-encoder.onnx',
                                        input_shape=(1, 3, 128, 256))

    def test_gpt_decoder(self):
        self._test_conversion_from_file(self._network_folder + 'gpt-decoder.onnx',
                                        input_shape=(1, 8, 16), input_range=(0, 1024),
                                        epsilon=1e-4)

    def test_bert(self):
        self._test_conversion_from_file(self._network_folder + 'bert.onnx',
                                        input_shape=(1, 384), input_range=[(0, 30522), (0, 1), (0, 2)],
                                        epsilon=1e-4)

    def test_efficientnet(self):
        self._test_conversion_from_file(self._network_folder + 'efficientnet-lite.onnx')

    def test_unet3d(self):
        self._test_conversion_from_file(self._network_folder + 'unet3d.onnx', epsilon=1e-3)

    def test_gpt2(self):
        self._test_conversion_from_file(self._network_folder + 'gpt2-10.onnx',
                                        input_shape=(1, 1, 8))


if __name__ == '__main__':
    unittest.main()
