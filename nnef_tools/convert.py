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

from .conversion import *
from .model import utils
import importlib
import argparse
import six


def get_reader(input_format, decomposed, fold_constants, custom_shapes):
    if input_format == 'tf':
        from .io.tf.graphdef import Reader
        return Reader(fold_constants=fold_constants)
    elif input_format == 'tflite':
        from .io.tf.lite import Reader
        return Reader()
    elif input_format == 'nnef':
        from .io.nnef import Reader
        return Reader(decomposed=decomposed, custom_shapes=custom_shapes)
    elif input_format == 'onnx':
        from .io.onnx import Reader
        return Reader(simplify=fold_constants)
    elif input_format == 'caffe2':
        from .io.caffe2 import Reader
        return Reader()
    elif input_format == 'caffe':
        from .io.caffe2 import Reader
        return Reader(legacy=True)
    else:
        return None


def get_writer(output_format, fragments, generate_fragments, annotate_shapes, compression):
    if output_format == 'tf':
        from .io.tf.graphdef import Writer
        return Writer()
    elif output_format == 'tflite':
        from .io.tf.lite import Writer
        return Writer()
    elif output_format == 'nnef':
        from .io.nnef import Writer
        return Writer(fragments=fragments, generate_custom_fragments=generate_fragments,
                      annotate_shapes=annotate_shapes, compression=compression)
    elif output_format == 'onnx':
        from .io.onnx import Writer
        return Writer()
    elif output_format == 'caffe2':
        from .io.caffe2 import Writer
        return Writer()
    else:
        return None


def get_converter(input_format, output_format, io_transpose, custom_transforms, custom_functions,
                  mirror_unsupported, keep_io_names):
    if input_format == 'tf' and output_format == 'nnef':
        from .conversion.tf_to_nnef import Converter
        return Converter(io_transpose=io_transpose,
                         custom_transforms=custom_transforms,
                         custom_functions=custom_functions,
                         mirror_unsupported=mirror_unsupported,
                         keep_io_names=keep_io_names)
    elif input_format == 'nnef' and output_format == 'tf':
        from .conversion.nnef_to_tf import Converter
        return Converter(io_transpose=io_transpose,
                         custom_transforms=custom_transforms,
                         custom_functions=custom_functions,
                         mirror_unsupported=mirror_unsupported)
    elif input_format == 'tflite' and output_format == 'nnef':
        from .conversion.tflite_to_nnef import Converter
        return Converter(io_transpose=io_transpose,
                         custom_transforms=custom_transforms,
                         custom_functions=custom_functions,
                         mirror_unsupported=mirror_unsupported,
                         keep_io_names=keep_io_names)
    elif input_format == 'nnef' and output_format == 'tflite':
        from .conversion.nnef_to_tflite import Converter
        return Converter(io_transpose=io_transpose,
                         custom_transforms=custom_transforms,
                         custom_functions=custom_functions,
                         mirror_unsupported=mirror_unsupported)
    elif (input_format == 'onnx' or input_format == 'caffe2' or input_format == 'caffe') and output_format == 'nnef':
        from .conversion.onnx_to_nnef import Converter
        return Converter(custom_transforms=custom_transforms,
                         custom_functions=custom_functions,
                         mirror_unsupported=mirror_unsupported)
    elif input_format == 'nnef' and (output_format == 'onnx' or output_format == 'caffe2'):
        from .conversion.nnef_to_onnx import Converter
        return Converter(custom_transforms=custom_transforms,
                         custom_functions=custom_functions,
                         mirror_unsupported=mirror_unsupported)
    else:
        return None


def get_optimizer(format, keep_io_names):
    if format == 'nnef':
        from .optimization.nnef_optimizer import Optimizer
        return Optimizer(keep_io_names=keep_io_names)
    elif format == 'tf':
        from .optimization.tf_optimizer import Optimizer
        return Optimizer(keep_io_names=keep_io_names)
    elif format == 'tflite':
        from .optimization.tflite_optimizer import Optimizer
        return Optimizer(keep_io_names=keep_io_names)
    elif format == 'onnx':
        from .optimization.onnx_optimizer import Optimizer
        return Optimizer(keep_io_names=keep_io_names)
    else:
        return None


def get_custom_converters(module_names):
    CUSTOM_TRANSFORMS = "CUSTOM_TRANSFORMS"

    transforms = {}
    functions = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, CUSTOM_TRANSFORMS):
            transforms.update(getattr(module, CUSTOM_TRANSFORMS))

        functions.update(Converter.find_public_functions(module))

    return transforms, functions


def get_custom_shapes(module_names):
    CUSTOM_SHAPES = "CUSTOM_SHAPES"

    shapes = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, CUSTOM_SHAPES):
            shapes.update(getattr(module, CUSTOM_SHAPES))

    return shapes


def get_custom_fragments(module_names):
    CUSTOM_FRAGMENTS = "CUSTOM_FRAGMENTS"

    fragments = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, CUSTOM_FRAGMENTS):
            fragments.update(getattr(module, CUSTOM_FRAGMENTS))

    return fragments


def needs_conversion(input_format, output_format):
    if input_format == 'caffe2' and output_format == 'onnx':
        return False
    elif input_format == 'onnx' and output_format == 'caffe2':
        return False
    elif input_format == 'caffe' and (output_format == 'onnx' or output_format == 'caffe2'):
        return False
    else:
        return input_format != output_format


def main(args):
    io_transpose = False if args.io_transpose is None else True if len(args.io_transpose) == 0 else args.io_transpose

    custom_transforms, custom_functions = get_custom_converters(args.custom_converters) \
        if args.custom_converters is not None else (None, None)

    converter = None
    if needs_conversion(args.input_format, args.output_format):
        converter = get_converter(args.input_format, args.output_format, io_transpose,
                                  custom_transforms, custom_functions,
                                  args.mirror_unsupported, args.keep_io_names)
        if converter is None:
            print("Unsupported conversion: {} to {}".format(args.input_format, args.output_format))
            return -1

    decomposed = converter.decomposed_operations() if converter else []
    fragments = converter.defined_operations() if converter else {}
    custom_shapes = converter.custom_shapes() if converter else {}

    if args.decompose is not None:
        decomposed += args.decompose

    if args.custom_shapes is not None:
        custom_shapes.update(get_custom_shapes(args.custom_shapes))

    if args.custom_fragments is not None:
        fragments.update(get_custom_fragments(args.custom_fragments))

    reader = get_reader(args.input_format, decomposed=decomposed, fold_constants=args.fold_constants,
                        custom_shapes=custom_shapes)
    if reader is None:
        print("Unsupported input-format: {}".format(args.input_format))
        return -1

    writer = get_writer(args.output_format, fragments=fragments, generate_fragments=args.generate_custom_fragments,
                        annotate_shapes=args.annotate_shapes, compression=args.compress)
    if writer is None:
        print("Unsupported output-format: {}".format(args.output_format))
        return -1

    default_output_model = args.input_model + '.' + (args.output_format if args.output_format != 'tf' else 'pb')

    reader_kwargs = {}
    if args.input_shapes is not None:
        input_shapes = eval(args.input_shapes)
        if not isinstance(input_shapes, dict) or not all(isinstance(name, str) and isinstance(shape, tuple)
                                                        for name, shape in six.iteritems(input_shapes)):
            print("'Input-shape' must be a dict of strings to tuples")
            return -1

        reader_kwargs['input_shapes'] = input_shapes

    try:
        graph = reader(args.input_model, **reader_kwargs)

        if args.input_names is not None or args.output_names is not None:
            not_found_names = []

            if args.input_names is not None:
                input_names = set(args.input_names)
                inputs = [tensor for tensor in graph.tensors if tensor.name in input_names]

                if len(inputs) != len(input_names):
                    found_names = [tensor.name for tensor in inputs]
                    not_found_names.append([input_name for input_name in input_names if input_name not in found_names])
                else:
                    graph.inputs = inputs

            if args.output_names is not None:
                output_names = set(args.output_names)
                outputs = [tensor for tensor in graph.tensors if tensor.name in output_names]

                if len(outputs) != len(output_names):
                    found_names = [tensor.name for tensor in outputs]
                    not_found_names.append([output_name for output_name in output_names if output_name not in found_names])
                else:
                    graph.outputs = outputs

            if len(not_found_names) > 0:
                print("Could not find tensor(s) in graph: {}".format(not_found_names))
                return -1

            utils.remove_unreachable(graph)

        optimizer = get_optimizer(args.input_format, args.keep_io_names)
        if optimizer:
            graph = optimizer(graph, only_required=True)

        if converter:
            graph.sort()
            graph = converter(graph)

        if args.optimize:
            optimizer = get_optimizer(args.output_format, args.keep_io_names)
            if optimizer:
                graph = optimizer(graph)

        print("Writing '{}'".format(args.output_model or default_output_model))
        writer(graph, args.output_model or default_output_model)

        return 0
    except ConversionError as e:
        print(e)
        return -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-model', type=str, required=True,
                        help='The input model')
    parser.add_argument('--output-model', type=str, default=None,
                        help='The output model')
    parser.add_argument('--input-format', type=str, required=True, choices=['tf', 'tflite', 'onnx', 'nnef', 'caffe2', 'caffe'],
                        help='The format of the input model')
    parser.add_argument('--output-format', type=str, required=True, choices=['tf', 'tflite', 'onnx', 'nnef', 'caffe2'],
                        help='The format of the output model')
    parser.add_argument('--input-shapes', type=str, default=None,
                        help='The (dict of) shape(s) to use for input(s).')
    parser.add_argument('--io-transpose', type=str, nargs='*', default=None,
                        help='The inputs/outputs to transpose')
    parser.add_argument('--fold-constants', action='store_true',
                        help='Enable folding of constant ops')
    parser.add_argument('--optimize', action='store_true',
                        help='Turn on optimization of resulting NNEF model')
    parser.add_argument('--custom-converters', type=str, nargs='+',
                        help='Module(s) containing custom converter code')
    parser.add_argument('--custom-shapes', type=str, nargs='+',
                        help='Module(s) containing custom shape inference code (when converting to NNEF)')
    parser.add_argument('--custom-fragments', type=str, nargs='+',
                        help='Module(s) containing custom fragment code (when converting to NNEF)')
    parser.add_argument('--mirror-unsupported', action='store_true',
                        help='Enable mirror-conversion of unsupported operations')
    parser.add_argument('--generate-custom-fragments', action='store_true',
                        help='Enable automatic generation of fragments for custom operations')
    parser.add_argument('--keep-io-names', action='store_true',
                        help='Keep the names of model inputs/outputs if possible')
    parser.add_argument('--decompose', type=str, nargs='*', default=None,
                        help='Names of operators to be decomposed by NNEF parser')
    parser.add_argument('--input-names', type=str, nargs='+',
                        help='Names of input tensor where the graph is cut before conversion')
    parser.add_argument('--output-names', type=str, nargs='+',
                        help='Names of output tensor where the graph is cut before conversion')
    parser.add_argument('--annotate-shapes', action='store_true',
                        help='Add tensor shapes as comments to NNEF output model')
    parser.add_argument('--compress', type=int, nargs='?', default=None, const=1,
                        help='Compress output NNEF folder at the given compression level')
    exit(main(parser.parse_args()))
