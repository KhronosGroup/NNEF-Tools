#!/usr/bin/env python

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

import sys

# python2 ensure that load from current directory is enabled
if len(sys.path) == 0 or sys.path[0] != '':
    sys.path.insert(0, '')

import shlex
import argparse
import importlib
import os
import typing
import six

from nnef_tools.conversion import conversion_info
from nnef_tools.core import utils
from nnef_tools.io.nnef.parser_config import NNEFParserConfig
from nnef_tools.optimization.data_format_optimizer import IOTransform


def check_tf_py_input_model(s):
    parts = s.split(':')
    if len(parts) not in [1, 2] or not parts[0] or '.' not in parts[0]:
        print('Error: Can not parse the --input-model parameter: {}.'.format(s), file=sys.stderr)
        exit(1)


def parse_io_transform(s):
    if not s:
        return IOTransform.IDENTITY

    # import possible IOTransforms into local scope
    locals().update({k: v for k, v in six.iteritems(IOTransform.__dict__) if not k.startswith('_')})

    try:
        return eval(s)
    except Exception as e:
        print('Error: Can not evaluate the --io-transformation parameter: {}.'.format(s), file=sys.stderr)
        exit(1)


def tf_pb_parse_input_shapes(s):
    if not s:
        return None

    # allow parsing dtypes without quotes
    for dtype in ['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64',
                  'uint8', 'uint16', 'uint32', 'uint64', 'bool']:
        locals()[dtype] = dtype

    try:
        return eval(s)
    except Exception as e:
        print('Error: Can not evaluate the --input-shape parameter: {}.'.format(s), file=sys.stderr)
        exit(1)


def onnx_parse_input_shapes(s):
    if not s:
        return None

    # allow parsing dtypes without quotes
    for dtype in ['FLOAT16', 'FLOAT', 'DOUBLE',
                  'INT8', 'INT16', 'INT32', 'INT64',
                  'UINT8', 'UINT16', 'UINT32', 'UINT64',
                  'BOOL']:
        locals()[dtype] = dtype

    try:
        return eval(s)
    except Exception as e:
        print('Error: Can not evaluate the --input-shape parameter: {}.'.format(s), file=sys.stderr)
        exit(1)


def try_eval_tf(__import, __fun_name):
    exec("import tensorflow as tf")
    exec("from nnef_tools.io.tensorflow.tf_py.tf_py_compat import tf_internal as _tf")
    try:
        if __import:
            exec(__import)
        return eval(__fun_name)
    except (ImportError, NameError):
        # print("Custom function not found: {}".format(__import))
        return None


def try_import(__import):
    if __import:
        try:
            exec(__import)
            return True
        except ImportError:
            # print("Custom function not found: {}".format(import_))
            return False
    return False


def get_tf_py_custom_traceable_functions(module_names_comma_sep):
    from nnef_tools.io.tensorflow.tf_py.tf_py_definitions import TraceableFunction
    module_names = [n.strip() for n in module_names_comma_sep.split(',')] if module_names_comma_sep else []
    custom_traceable_functions = []
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, "TENSORFLOW_PY_OP_DEFINITIONS"):
            opdefs = module.TENSORFLOW_PY_OP_DEFINITIONS
            for opdef in opdefs:
                funs = []
                for import_, fun_name in zip(opdef.imports, opdef.op_names):
                    fun = try_eval_tf(import_, fun_name)
                    if fun is not None:
                        funs.append(fun)

                custom_traceable_functions.append(TraceableFunction(opdef.op_proto, funs))
    return custom_traceable_functions


def get_tf_py_imports_and_op_protos(module_names_comma_sep):
    module_names = [n.strip() for n in module_names_comma_sep.split(',')] if module_names_comma_sep else []
    imports = []
    op_protos = []
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, "TENSORFLOW_PY_OP_DEFINITIONS"):
            opdefs = module.TENSORFLOW_PY_OP_DEFINITIONS
            for opdef in opdefs:
                imports_for_this_op = []
                for import_, fun_name in zip(opdef.imports, opdef.op_names):
                    if try_import(import_):
                        imports_for_this_op.append(import_.strip())
                op_protos.append(opdef.op_proto)
                if imports_for_this_op:
                    imports.append("\n".join(imports_for_this_op))
    return "\n".join(imports), op_protos


def get_reader(input_format,
               output_format,
               input_shape=None,
               permissive=False,
               with_weights=True,
               custom_converters=None):
    if input_format == 'nnef':
        from nnef_tools.io.nnef.nnef_io import Reader

        configs = [NNEFParserConfig.STANDARD_CONFIG]

        if output_format in ['tensorflow-pb', 'tensorflow-py', 'tensorflow-lite']:
            from nnef_tools.conversion.tensorflow import nnef_to_tf
            configs.append(nnef_to_tf.ParserConfig)
        elif output_format in ['onnx']:
            from nnef_tools.conversion.onnx import nnef_to_onnx
            configs.append(nnef_to_onnx.ParserConfig)
        else:
            assert False

        configs += NNEFParserConfig.load_configs(custom_converters, load_standard=False)

        return Reader(parser_configs=configs)
    elif input_format == 'tensorflow-pb':
        # TODO custom converter
        from nnef_tools.io.tensorflow.tf_pb_io import Reader
        return Reader(convert_to_tf_py=True, input_shape=input_shape)
    elif input_format == 'tensorflow-py':
        from nnef_tools.io.tensorflow.tf_py_io import Reader
        if custom_converters:
            custom_traceable_functions = get_tf_py_custom_traceable_functions(custom_converters)
        else:
            custom_traceable_functions = None
        return Reader(expand_gradients=True, custom_traceable_functions=custom_traceable_functions)
    elif input_format == 'tensorflow-lite':
        # TODO custom converter
        from nnef_tools.io.tensorflow.tflite_io import Reader
        return Reader(convert_to_tf_py=True)
    elif input_format == 'onnx':
        # TODO custom converter
        from nnef_tools.io.onnx.onnx_io import Reader
        return Reader(propagate_shapes=True, input_shape=input_shape)
    else:
        assert False


def get_custom_converters(input_format, output_format, custom_converters):
    module_names = [n.strip() for n in custom_converters.split(',')] if custom_converters else []
    custom_converter_by_op_name = {}

    for module_name in module_names:
        module = importlib.import_module(module_name)
        attrname = (input_format + "_to_" + output_format + "_converters").replace('-', '_').upper()
        if hasattr(module, attrname):
            custom_converter_by_op_name.update(getattr(module, attrname))

    return custom_converter_by_op_name


def get_converter(input_format, output_format, prefer_nchw=False, permissive=False, custom_converters=None):
    if input_format in ['tensorflow-py', 'tensorflow-pb', 'tensorflow-lite'] and output_format == 'nnef':
        from nnef_tools.conversion.tensorflow.tf_to_nnef import Converter

        if input_format == 'tensorflow-py' and custom_converters:
            custom_converter_by_op_name = get_custom_converters(input_format, output_format, custom_converters)
        else:
            custom_converter_by_op_name = None

        return Converter(enable_imprecise_image_resize=permissive,
                         custom_converter_by_op_name=custom_converter_by_op_name)
    elif input_format == 'nnef' and output_format in ['tensorflow-py', 'tensorflow-pb', 'tensorflow-lite']:
        from nnef_tools.conversion.tensorflow.nnef_to_tf import Converter

        if output_format == 'tensorflow-py' and custom_converters:
            custom_converter_by_op_name = get_custom_converters(input_format, output_format, custom_converters)
        else:
            custom_converter_by_op_name = None

        return Converter(prefer_nhwc=not prefer_nchw,
                         enable_imprecise_image_resize=permissive,
                         enable_imprecise_padding_border=permissive,
                         custom_converter_by_op_name=custom_converter_by_op_name)
    elif input_format == 'onnx' and output_format == 'nnef':
        from nnef_tools.conversion.onnx.onnx_to_nnef import Converter

        custom_converter_by_op_name = (get_custom_converters(input_format, output_format, custom_converters)
                                       if custom_converters else None)

        return Converter(custom_converter_by_op_name=custom_converter_by_op_name)
    elif input_format == 'nnef' and output_format == 'onnx':
        from nnef_tools.conversion.onnx.nnef_to_onnx import Converter

        custom_converter_by_op_name = (get_custom_converters(input_format, output_format, custom_converters)
                                       if custom_converters else None)

        return Converter(enable_imprecise_image_resize=permissive,
                         custom_converter_by_op_name=custom_converter_by_op_name)
    else:
        assert False


def get_data_format_optimizer(input_format, output_format, io_transformation):
    if input_format in ['tensorflow-py', 'tensorflow-pb', 'tensorflow-lite', 'onnx'] and output_format == 'nnef':
        from nnef_tools.optimization.nnef.nnef_data_format_optimizer import Optimizer
        return Optimizer(io_transform=io_transformation, merge_transforms_into_variables=True)
    elif input_format == 'nnef' and output_format in ['tensorflow-py', 'tensorflow-pb', 'tensorflow-lite']:
        from nnef_tools.optimization.tensorflow.tf_data_format_optimizer import Optimizer
        return Optimizer(io_transform=io_transformation, merge_transforms_into_variables=True)
    elif input_format == 'nnef' and output_format == 'onnx':
        from nnef_tools.optimization.onnx.onnx_data_format_optimizer import Optimizer
        return Optimizer(io_transform=io_transformation, merge_transforms_into_variables=True)
    else:
        assert False


def get_writer(input_format, output_format, with_weights=True, custom_converters=None):
    if output_format == 'nnef':
        from nnef_tools.io.nnef.nnef_io import Writer
        return Writer(write_weights=with_weights)
    elif output_format == 'tensorflow-py':
        from nnef_tools.io.tensorflow.tf_py_io import Writer

        if custom_converters:
            custom_imports, custom_op_protos = get_tf_py_imports_and_op_protos(custom_converters)
        else:
            custom_imports, custom_op_protos = None, None

        return Writer(write_weights=with_weights, custom_op_protos=custom_op_protos, custom_imports=custom_imports)
    elif output_format == 'tensorflow-pb':
        from nnef_tools.io.tensorflow.tf_pb_io import Writer
        return Writer(convert_from_tf_py=True)
    elif output_format == 'tensorflow-lite':
        from nnef_tools.io.tensorflow.tflite_io import Writer
        return Writer(convert_from_tf_py=True)
    elif output_format == 'onnx':
        from nnef_tools.io.onnx.onnx_io import Writer
        return Writer()
    else:
        assert False


def get_extension(output_format, compress):
    if output_format == 'nnef':
        if compress:
            return ".nnef.tgz"
        else:
            return ".nnef"
    elif output_format == 'tensorflow-py':
        return ".py"
    elif output_format == 'tensorflow-pb':
        return ".pb"
    elif output_format == 'onnx':
        return ".onnx"
    elif output_format == 'tensorflow-lite':
        return ".tflite"
    else:
        assert False


def get_output_path(output_format, output_directory, output_prefix, compress):
    return os.path.join(output_directory, output_prefix + get_extension(output_format, compress))


def fix_extension(path, ext):
    if path.endswith(ext):
        return path
    return path + ext


def fix_output_path(output_format, output_path, compress):
    return fix_extension(output_path, get_extension(output_format, compress))


def ensure_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def convert_using_premade_objects(in_filename, out_filename, out_info_filename, reader, converter,
                                  data_format_optimizer, writer):
    source_graph = reader(in_filename)
    target_graph, conv_info = converter(source_graph)
    opt_info = data_format_optimizer(target_graph) if data_format_optimizer is not None else None
    ensure_dirs(os.path.dirname(out_filename))
    write_info = writer(target_graph, out_filename)

    if out_info_filename:
        combined_info = conversion_info.compose(conv_info, opt_info, write_info)
        ensure_dirs(os.path.dirname(out_info_filename))
        conversion_info.dump(combined_info, out_info_filename)


def convert(input_format,
            output_format,
            input_model,
            output_path,
            input_shape=None,
            prefer_nchw=False,
            io_transformation=None,
            compress=False,
            permissive=False,
            with_weights=True,
            custom_converters=None,
            conversion_info=False  # type: typing.Union[bool, str, None]
            ):
    if input_format in ['tensorflow-pb', 'tensorflow-lite', 'onnx', 'nnef']:
        output_prefix = os.path.basename(os.path.abspath(input_model))
    elif input_format == 'tensorflow-py':
        assert '.' in input_model
        output_prefix = input_model.split(':')[0].split('.')[1]
    else:
        assert False

    if output_path:
        if output_path.endswith('/') or output_path.endswith('\\'):
            output_path = get_output_path(output_format=output_format,
                                          output_directory=os.path.abspath(output_path),
                                          output_prefix=output_prefix,
                                          compress=compress)
        else:
            output_path = fix_output_path(output_format=output_format,
                                          output_path=output_path,
                                          compress=compress)
    else:
        if input_format in ['tensorflow-pb', 'tensorflow-lite', 'onnx', 'nnef']:
            output_dir = os.path.dirname(os.path.abspath(input_model))
            if not output_dir:
                output_dir = '.'
        else:
            output_dir = '.'

        output_path = get_output_path(output_format=output_format,
                                      output_directory=output_dir,
                                      output_prefix=output_prefix,
                                      compress=compress)

    if conversion_info is True:
        conversion_info = output_path + ".conversion.json"
    elif conversion_info:
        if conversion_info.endswith('/') or conversion_info.endswith('\\'):
            conversion_info += os.path.basename(output_path) + '.conversion.json'
        else:
            conversion_info = fix_extension(conversion_info, '.json')

    convert_using_premade_objects(in_filename=input_model,
                                  out_filename=output_path,
                                  out_info_filename=conversion_info,
                                  reader=get_reader(input_format=input_format,
                                                    output_format=output_format,
                                                    input_shape=input_shape,
                                                    permissive=permissive,
                                                    with_weights=with_weights,
                                                    custom_converters=custom_converters),
                                  converter=get_converter(input_format=input_format,
                                                          output_format=output_format,
                                                          prefer_nchw=prefer_nchw,
                                                          permissive=permissive,
                                                          custom_converters=custom_converters),
                                  data_format_optimizer=get_data_format_optimizer(input_format=input_format,
                                                                                  output_format=output_format,
                                                                                  io_transformation=io_transformation),
                                  writer=get_writer(input_format=input_format,
                                                    output_format=output_format,
                                                    with_weights=with_weights,
                                                    custom_converters=custom_converters))


def tf_py_has_checkpoint(input_model):
    parts = input_model.split(':')
    return len(parts) >= 2 and parts[1]


def convert_using_args(args):
    convert(input_format=args.input_format,
            output_format=args.output_format,
            input_model=args.input_model,
            output_path=args.output_model,
            input_shape=args.input_shape,
            prefer_nchw=args.prefer_nchw,
            io_transformation=args.io_transformation,
            compress=args.compress,
            permissive=args.permissive,
            with_weights=not args.no_weights,
            custom_converters=args.custom_converters,
            conversion_info=args.conversion_info)


def get_args(argv):
    parser = argparse.ArgumentParser(description="NNEFTools/convert: Neural network conversion tool",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="""Tips:
- If you refer to a Python package or module that is not in the current directory,
please add its location to PYTHONPATH.
- Quote parameters if they contain spaces or special characters.""")

    parser.add_argument("--input-format",
                        choices=['nnef', 'tensorflow-pb', 'tensorflow-py', 'tensorflow-lite', 'onnx'],
                        required=True,
                        help="""Input format""")
    parser.add_argument("--output-format",
                        choices=['nnef', 'tensorflow-pb', 'tensorflow-py', 'tensorflow-lite', 'onnx'],
                        required=True,
                        help="""Output format""")
    parser.add_argument("--input-model",
                        required=True,
                        help="""nnef: path of NNEF file, directory or nnef.tgz file.
    In case of a single NNEF file, no weights are loaded.
onnx: path of ONNX file
tensorflow-pb: path of pb file
tensorflow-py: package.module.function or package.module.function:checkpoint_path.ckpt
tensorflow-lite: path of tflite file""")

    parser.add_argument("--output-model",
                        help="""Path of output file.
If it ends with '/' or '\\', it is considered a directory and the filename is auto-generated, similarly to the default.
Default: tensorflow-py: {function name}.{output extension}, others: {input path}.{output extension}
Example for default name generation: module.function -> function.nnef
                                     directory/network.onnx -> directory/network.onnx.nnef""")

    parser.add_argument("--io-transformation",
                        default="IDENTITY",
                        help="""The transformation to apply to input and output tensors.
It can be a single transformation (applied to all IO tensors) or a tensor_name->transformation dict.
Available transformations:
- Transpose(axes: List[int])
- IDENTITY
- SMART_NHWC_TO_NCHW
- SMART_NCHW_TO_NHWC
The 'SMART' transformations work like a Transpose for rank >= 3 and IDENTITY otherwise.
Default: IDENTITY
""")

    parser.add_argument('--input-shape',
                        default="",
                        help="""onnx: The shape of input tensors must be specified if they are not set in the protobuf file.
    The value must be a (dtype, shape) tuple or a tensor_name->(dtype, shape) dict.
    DType must be one of:
        FLOAT16, FLOAT, DOUBLE,
        INT8, INT16, INT32, INT64,
        UINT8, UINT16, UINT32, UINT64,
        BOOL
tensorflow-pb: The shape of input tensors must be specified if they are not set in the protobuf file.
    The value must be a (dtype, shape) tuple or a tensor_name->(dtype, shape) dict.
    DType must be one of:
        float16, float32, float64,
        int8, int16, int32, int64,
        uint8, uint16, uint32, uint64
        bool
Default: (empty).
""")

    parser.add_argument("--prefer-nchw",
                        action="store_true",
                        help="""tensorflow-py/pb: Generate the NCHW version of operations where both NHWC and NCHW is available""")

    parser.add_argument("--compress",
                        action="store_true",
                        help="""nnef: Create a compressed model.nnef.tgz file""")

    parser.add_argument("--permissive",
                        action="store_true",
                        help="""Allow some imprecise conversions""")

    parser.add_argument('--custom-converters',
                        help="""Module(s) of custom converters: e.g. "package.module", "package.module1,package.module2""")

    parser.add_argument("--conversion-info",
                        nargs='?',
                        default=False,
                        const=True,
                        help="""Set this to write a conversion info file. 
Without an argument: Write to {output path}.conversion.json. 
With an argument: Write to the path defined by the argument.""")

    args = parser.parse_args(args=argv[1:])
    args.io_transformation = parse_io_transform(args.io_transformation)
    if args.input_format == 'tensorflow-py':
        check_tf_py_input_model(args.input_model)
    if args.input_format == 'tensorflow-pb':
        args.input_shape = tf_pb_parse_input_shapes(args.input_shape)
    elif args.input_format == 'onnx':
        args.input_shape = onnx_parse_input_shapes(args.input_shape)
    else:
        args.input_shape = None

    args.no_weights = False

    if sum(framework == 'nnef' for framework in [args.input_format, args.output_format]) != 1:
        print("Error: Either input or output framework must be nnef.", file=sys.stderr)
        exit(1)

    if args.input_format == 'tensorflow-py' and not tf_py_has_checkpoint(args.input_model):
        args.no_weights = True

    if args.compress and args.output_format != 'nnef':
        print("Error: --compress can now be only used with NNEF as output framework.")
        exit(1)

    if args.input_format == "nnef" and not os.path.isdir(args.input_model) and not args.input_model.endswith('.tgz'):
        args.no_weights = True

    return args


def convert_using_argv(argv):
    args = get_args(argv)

    if 'tensorflow-py' in [args.input_format, args.output_format]:
        if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        if not tf.test.is_gpu_available():
            print('Warning: Tensorflow is using CPU. Some operations are only supported in the GPU version.',
                  file=sys.stderr)

    try:
        convert_using_args(args)
    except utils.NNEFToolsException as e:
        print("Error: " + str(e), file=sys.stderr)
        exit(1)


# Call this if you don't want to reload the whole program for each run
def convert_using_command(command):
    def command_to_argv(command):
        return shlex.split(command.replace('\\', ' ').replace('\n', ' ').replace('\r', ' '))

    return convert_using_argv(command_to_argv(command))


if __name__ == '__main__':
    convert_using_argv(sys.argv)
