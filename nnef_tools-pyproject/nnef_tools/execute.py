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

from .utils import stdio
from .interpreter import Statistics
from collections import namedtuple
import importlib
import argparse
import numpy as np
import json
import nnef
import six
import sys
import os


_onnx_dtype_to_numpy = {
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

_nnef_dtype_to_numpy = {
    'scalar': np.float32,
    'integer': np.int32,
    'logical': np.bool_,
}

_numpy_dtype_remap = {
    np.short: np.int64,
    np.longlong: np.int64,
    np.ushort: np.uint64,
    np.uint: np.uint64,
    np.ulonglong: np.uint64,
    np.double: np.float64,
    np.longdouble: np.float64,
}


def _is_lambda(v):
    LAMBDA = lambda: 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


def uniform(min=0, max=1):
    return lambda shape: np.random.uniform(min, max, shape)


def normal(mean=0, std=1):
    return lambda shape: np.random.normal(mean, std, shape)


def needs_transpose(io_transpose, name):
    return io_transpose is not None and (len(io_transpose) == 0 or name in io_transpose)


def transpose_channels_last_to_first(x):
    rank = len(x.shape)
    return np.transpose(x, axes=[0, rank - 1] + list(range(1, rank - 1)))


def transpose_channels_first_to_last(x):
    rank = len(x.shape)
    return np.transpose(x, axes=[0] + list(range(2, rank)) + [1])


def read_input(file, name, shape, dtype, transpose):
    data = nnef.read_tensor(file)

    any_batch = shape[0] == 0
    offset = int(any_batch)
    if tuple(data.shape[offset:]) != tuple(shape[offset:]):
        raise ValueError("Mismatch between declared and read shape for input '{}'; {} vs {}"
                         .format(name, data.shape, shape))
    if data.dtype != dtype:
        raise ValueError("Mismatch between declared and read dtype for input '{}'; {} vs {}"
                         .format(name, data.dtype, dtype))

    return transpose_channels_first_to_last(data) if transpose else data


def compute_statistics(array):
    if array.size == 0:
        return Statistics(num=0, min=0.0, max=0.0, sum=0.0, ssum=0.0)

    return Statistics(
        num=array.size,
        min=float(np.min(array)),
        max=float(np.max(array)),
        sum=float(np.sum(array)),
        ssum=float(np.sum(array * array)),
    )


class RandomInputSource:

    def __init__(self, distribution):
        self._distribution = distribution

    def __call__(self, name, shape, dtype):
        return self._distribution(shape).astype(dtype)


class StreamInputSource:

    def __init__(self, stream, io_transpose):
        self._stream = stream
        self._io_transpose = io_transpose

    def __call__(self, name, shape, dtype):
        return read_input(self._stream, name, shape, dtype,
                          transpose=needs_transpose(self._io_transpose, name))


class FileInputSource:

    def __init__(self, folder, io_transpose):
        self._folder = folder
        self._io_transpose = io_transpose

    def __call__(self, name, shape, dtype):
        with open(os.path.join(self._folder, name + '.dat')) as file:
            return read_input(file, name, shape, dtype,
                              transpose=needs_transpose(self._io_transpose, name))


TensorInfo = namedtuple('TensorInfo', ['name', 'shape', 'dtype'])


class Executor:

    def input_info(self):
        raise NotImplementedError()

    def output_info(self):
        raise NotImplementedError()

    def tensor_info(self):
        raise NotImplementedError()

    def __call__(self, inputs, output_names=None, collect_statistics=False):
        raise NotImplementedError()


class TFExecutor(Executor):

    def __init__(self, model_path):
        try:
            import tensorflow.compat.v1 as tf
        except ImportError:
            import tensorflow as tf
        from .io.tf.graphdef.protobuf import GraphDef

        self.Session = tf.Session

        graph_def = GraphDef()
        with open(model_path, 'rb') as file:
            graph_def.ParseFromString(file.read())

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        ops = self.graph.get_operations()
        consumed = {tensor for op in ops for tensor in op.inputs}

        self.inputs = [op.outputs[0] for op in ops if op.type == 'Placeholder']
        self.outputs = [tensor for op in ops if len(op.inputs) for tensor in op.outputs
                        if tensor not in consumed and tensor.name.endswith(':0')]

    def input_info(self):
        return [TensorInfo(tensor.name, tuple(tensor.shape.as_list()), tensor.dtype.as_numpy_dtype)
                for tensor in self.inputs]

    def output_info(self):
        return [TensorInfo(tensor.name, tuple(tensor.shape.as_list()), tensor.dtype.as_numpy_dtype)
                for tensor in self.outputs]

    def tensor_info(self):
        tensors = [tensor for op in self.graph.get_operations() for tensor in op.outputs]
        return [TensorInfo(tensor.name, tuple(tensor.shape.as_list()), tensor.dtype.as_numpy_dtype)
                for tensor in tensors]

    def __call__(self, inputs, output_names=None, collect_statistics=False):
        ops = self.graph.get_operations()

        if output_names is not None:
            tensor_names = {tensor.name for op in ops for tensor in op.outputs}
            invalid = {name for name in output_names if name not in tensor_names}
            if len(invalid):
                raise ValueError('Invalid tensor name(s): {}'.format(invalid))

            outputs = {tensor.name: tensor for op in ops for tensor in op.outputs if tensor.name in output_names}
        else:
            outputs = {tensor.name: tensor for tensor in self.outputs}

        if collect_statistics:
            tensors = {tensor.name: tensor for op in ops for tensor in op.outputs if tensor.name.endswith(':0')}
            with self.Session(graph=self.graph) as sess:
                values = sess.run(tensors, feed_dict=inputs)

                outputs = {name: values[name] for name in outputs}

                stats = {}
                for name, array in six.iteritems(values):
                    stats[name] = compute_statistics(array)

                return outputs, stats
        else:
            with self.Session(graph=self.graph) as sess:
                outputs = sess.run(outputs, feed_dict=inputs)

            return outputs, None


class TFLiteExecutor(Executor):

    def __init__(self, model_path):
        try:
            import tensorflow.compat.v1 as tf
        except ImportError:
            import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def input_info(self):
        return [TensorInfo(tensor['name'], tensor['shape'], tensor['dtype'])
                for tensor in self.interpreter.get_input_details()]

    def output_info(self):
        return [TensorInfo(tensor['name'], tensor['shape'], tensor['dtype'])
                for tensor in self.interpreter.get_output_details()]

    def tensor_info(self):
        return [TensorInfo(tensor['name'], tensor['shape'], tensor['dtype'])
                for tensor in self.interpreter.get_tensor_details()]

    def __call__(self, inputs, output_names=None, collect_statistics=False):
        for tensor in self.interpreter.get_input_details():
            self.interpreter.set_tensor(tensor['index'], inputs[tensor['name']])

        self.interpreter.invoke()

        if output_names is not None:
            tensor_names = {tensor['name'] for tensor in self.interpreter.get_tensor_details()}
            invalid = {name for name in output_names if name not in tensor_names}
            if len(invalid):
                raise ValueError('Invalid tensor name(s): {}'.format(invalid))

            outputs = {tensor['name']: self.interpreter.get_tensor(tensor['index'])
                       for tensor in self.interpreter.get_tensor_details()
                       if tensor['name'] in output_names}
        else:
            outputs = {tensor['name']: self.interpreter.get_tensor(tensor['index'])
                       for tensor in self.interpreter.get_output_details()}

        stats = {tensor['name']: compute_statistics(self.interpreter.get_tensor(tensor['index']))
                 for tensor in self.interpreter.get_tensor_details()} if collect_statistics else None

        return outputs, stats


class ONNXExecutor(Executor):

    def __init__(self, model_path, require_intermediates=False):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

        self.session = onnxruntime.InferenceSession(model_path, sess_options=options,
                                                    providers=['CPUExecutionProvider'])
        self.inputs = [TensorInfo(tensor.name, tensor.shape, _onnx_dtype_to_numpy[tensor.type])
                       for tensor in self.session.get_inputs()]
        self.outputs = [TensorInfo(tensor.name, tensor.shape, _onnx_dtype_to_numpy[tensor.type])
                        for tensor in self.session.get_outputs()]

        if require_intermediates:
            import onnx
            from onnx.shape_inference import infer_shapes

            model = onnx.load_model(model_path)
            model = infer_shapes(model)

            for info in model.graph.value_info:
                output_info = model.graph.output.add()
                output_info.ParseFromString(info.SerializeToString())

            self.session = onnxruntime.InferenceSession(model.SerializeToString(), sess_options=options,
                                                        providers=['CPUExecutionProvider'])

    def input_info(self):
        return self.inputs

    def output_info(self):
        return self.outputs

    def tensor_info(self):
        return None

    def __call__(self, inputs, output_names=None, collect_statistics=False):
        if output_names is not None:
            inputs_as_outputs = {name: inputs[name] for name in output_names if name in inputs}
            output_names = [name for name in output_names if name not in inputs]
        else:
            output_names = [output.name for output in self.outputs]
            inputs_as_outputs = {}

        if collect_statistics:
            original_outputs = [output.name for output in self.outputs]
            fetch_names = [tensor.name for tensor in self.session.get_outputs()
                           if tensor.name not in original_outputs] + original_outputs
            values = self.session.run(fetch_names, inputs)
            outputs = {name: value for name, value in zip(fetch_names, values) if name in set(output_names)}

            stats = {}
            for name, value in zip(fetch_names, values):
                stats[name] = compute_statistics(value)
        else:
            values = self.session.run(output_names, inputs)
            outputs = {name: value for name, value in zip(output_names, values)}
            stats = None

        outputs.update(inputs_as_outputs)

        return outputs, stats


class NNEFExecutor(Executor):

    def __init__(self, model_path, custom_operators, decomposed):
        from .interpreter.pytorch import Interpreter
        self.interpreter = Interpreter(model_path, custom_operators=custom_operators, decomposed=decomposed)

    def input_info(self):
        return [TensorInfo(tensor.name, tensor.shape, _nnef_dtype_to_numpy[tensor.dtype])
                for tensor in self.interpreter.input_details()]

    def output_info(self):
        return [TensorInfo(tensor.name, tensor.shape, _nnef_dtype_to_numpy[tensor.dtype])
                for tensor in self.interpreter.output_details()]

    def tensor_info(self):
        return [TensorInfo(tensor.name, tensor.shape, _nnef_dtype_to_numpy[tensor.dtype])
                for tensor in self.interpreter.tensor_details()]

    def __call__(self, inputs, output_names=None, collect_statistics=False):
        inputs = [inputs[tensor.name] for tensor in self.interpreter.input_details()]
        if collect_statistics:
            return self.interpreter(inputs, output_names, collect_statistics)
        else:
            return self.interpreter(inputs, output_names, collect_statistics), None


def get_executor(format, model_path, require_intermediates, custom_operators, decomposed):
    if format == 'tf':
        return TFExecutor(model_path)
    elif format == 'tflite':
        return TFLiteExecutor(model_path)
    elif format == 'onnx':
        return ONNXExecutor(model_path, require_intermediates)
    elif format == 'nnef':
        return NNEFExecutor(model_path, custom_operators, decomposed)
    else:
        return None


def write_nnef_tensor(filename, value):
    with open(filename, 'wb') as file:
        dtype = _numpy_dtype_remap.get(value.dtype.type)
        if dtype is not None:
            value = value.astype(dtype)

        nnef.write_tensor(file, value)


def write_statistics(filename, statistics):
    statistics = {name: {'min': stats.min, 'max': stats.max, 'mean': stats.mean(), 'std': stats.std()}
                  for name, stats in six.iteritems(statistics)}

    with open(filename, 'w') as file:
        json.dump(statistics, file, indent=4)


def get_custom_operators(module_names):
    CUSTOM_OPERATORS = "CUSTOM_OPERATORS"

    operators = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, CUSTOM_OPERATORS):
            operators.update(getattr(module, CUSTOM_OPERATORS))

    return operators


def batched_info(tensor_info, batch_size):
    for info in tensor_info:
        if info.shape[0] != batch_size and info.shape[0] != 1 and not isinstance(info.shape[0], str):
            raise ValueError('invalid input shape {} for batch size {}'.format(info.shape, batch_size))

    return [TensorInfo(name=info.name, shape=(batch_size, *info.shape[1:]), dtype=info.dtype)
            for info in tensor_info]


def accumulate_statistics(global_stats, local_stats):
    if global_stats is None:
        return local_stats
    for name, stats in six.iteritems(local_stats):
        global_stats[name] += stats
    return global_stats


def main(args):
    if args.input_path is not None:
        source = FileInputSource(args.input_path, args.io_transpose)
    elif args.random is not None:
        if args.batch_size == 0:
            print('batch-size must not be 0 when inputs are random generated', file=sys.stderr)
            return -1

        try:
            distribution = eval(args.random)
            if not _is_lambda(distribution):
                distribution = distribution()
            source = RandomInputSource(distribution)
        except Exception as e:
            print("Could not evaluate distribution: " + str(e), file=sys.stderr)
            return -1
    else:
        if not stdio.is_stdin_piped():
            print('Input must be piped', file=sys.stderr)
            return -1

        stdio.set_stdin_to_binary()
        source = StreamInputSource(sys.stdin, args.io_transpose)

    output_names = eval(args.output_names) if args.output_names is not None and args.output_names != "*" else args.output_names
    custom_operators = get_custom_operators(args.custom_operators) if args.custom_operators is not None else None

    if args.random is not None and args.seed is not None:
        np.random.seed(args.seed)

    collect_statistics = args.statistics is not None

    try:
        executor = get_executor(args.format, args.model, collect_statistics, custom_operators, args.decompose)

        if isinstance(output_names, dict):
            fetch_names = output_names.keys()
        elif output_names == "*":
            tensors = executor.tensor_info()
            fetch_names = [info.name for info in tensors] if tensors is not None else None
        else:
            fetch_names = output_names

        input_info = executor.input_info()
        if args.batch_size is not None:
            input_info = batched_info(input_info, args.batch_size)

        output_info = executor.output_info()

        inputs = {info.name: source(info.name, info.shape, info.dtype) for info in input_info}

        batch_size = args.batch_size
        if batch_size == 0:
            batch_size = next(iter(six.itervalues(inputs))).shape[0]
            if not all(input.shape[0] == batch_size for input in six.itervalues(inputs)):
                print('All inputs must have the same batch-size', file=sys.stderr)
                return -1

        if batch_size is not None and batch_size != 1:
            slices = {name: [] for name in fetch_names} if fetch_names is not None else \
                     {info.name: [] for info in output_info}
            stats = None

            for k in range(batch_size):
                slice_inputs = {name: np.expand_dims(data[k], axis=0) for name, data in six.iteritems(inputs)}
                slice_outputs, slice_stats = executor(slice_inputs, fetch_names, collect_statistics)

                for name, data in six.iteritems(slice_outputs):
                    slices[name].append(data)

                if collect_statistics:
                    stats = accumulate_statistics(stats, slice_stats)

            outputs = {name: np.concatenate(items, axis=0) for name, items in six.iteritems(slices)}
        else:
            outputs, stats = executor(inputs, fetch_names, collect_statistics)

    except ValueError as e:
        print(e, file=sys.stderr)
        return -1

    for name, value in six.iteritems(outputs):
        if needs_transpose(args.io_transpose, name):
            outputs[name] = transpose_channels_last_to_first(value)

    if isinstance(output_names, dict):
        outputs = {output_names[name]: value for name, value in six.iteritems(outputs)}

    if args.tensor_mapping is not None:
        with open(args.tensor_mapping) as file:
            tensor_mapping = json.load(file)

        if stats is not None:
            stats = {tensor_mapping.get(key, key): value for key, value in six.iteritems(stats)}

    if stats is not None:
        write_statistics(args.statistics, stats)
        print('Written {}'.format(args.statistics))

    if args.output_path is not None:
        for name, value in six.iteritems(outputs):
            filename = os.path.join(args.output_path, name + ".dat")
            write_nnef_tensor(filename, value)
            print('Written {}'.format(filename))
    else:
        if not stdio.is_stdout_piped():
            if collect_statistics:
                return 0

            print('Output must be piped', file=sys.stderr)
            return -1

        stdio.set_stdout_to_binary()

        for name, value in six.iteritems(outputs):
            nnef.write_tensor(sys.stdout, value)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='The model to execute')
    parser.add_argument('--format', type=str, required=True, choices=['tf', 'tflite', 'onnx', 'nnef'],
                        help='The format of the model')
    parser.add_argument('--random', type=str, default=None,
                        help='Random distribution for input generation')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for input generation')
    parser.add_argument('--input-path', type=str, default=None,
                        help='Folder to read inputs from')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Folder to save outputs into')
    parser.add_argument('--output-names', type=str, default=None,
                        help='The set (dict) of tensor names (to file names) considered as outputs to be saved. '
                             'Use * to save all tensors')
    parser.add_argument('--io-transpose', type=str, nargs='*', default=None,
                        help='The inputs/outputs to transpose from channels last to channels first dimension order')
    parser.add_argument('--decompose', type=str, nargs='*', default=None,
                        help='Names of operators to be decomposed by NNEF parser')
    parser.add_argument('--statistics', type=str, nargs='?', default=None, const='stats.json',
                        help='Calculate activations statistics and save to output path in json format')
    parser.add_argument('--custom-operators', type=str, nargs='+', default=None,
                        help='Module(s) containing custom operator code')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Specify batch-size for single-batch models')
    parser.add_argument('--tensor-mapping', type=str, default=None,
                        help='Use mapping of tensor names for statistics')
    exit(main(parser.parse_args()))
