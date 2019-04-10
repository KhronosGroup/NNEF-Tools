#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import sys

# python2 ensure that load from current directory is enabled
if len(sys.path) == 0 or sys.path[0] != '':
    sys.path.insert(0, '')

import shlex
import argparse
import importlib
import os

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


def get_tf_py_custom_traceable_functions(module_names_comma_sep):
    exec("import tensorflow as tf")
    exec("from nnef_tools.io.tensorflow.tf_py.tf_py_definitions import tf_internal as _tf")
    from nnef_tools.io.tensorflow.tf_py.tf_py_definitions import TraceableFunction
    module_names = [n.strip() for n in module_names_comma_sep.split(',')] if module_names_comma_sep else []
    custom_traceable_functions = []
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, "TENSORFLOW_PY_OP_DEFINITIONS"):
            opdefs = module.TENSORFLOW_PY_OP_DEFINITIONS
            for opdef in opdefs:
                if opdef.imports:
                    exec(opdef.imports)
                fun_names = opdef.op_names
                funs = []
                for fun_name in fun_names:
                    funs.append(eval(fun_name))
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
                op_protos.append(opdef.op_proto)
                imports.append(opdef.imports)
    return "\n".join(imports), op_protos


def get_reader(input_framework,
               output_framework,
               input_shape=None,
               permissive=False,
               with_weights=True,
               custom_converters=None):
    if input_framework == 'nnef':
        from nnef_tools.io.nnef.nnef_io import Reader

        configs = [NNEFParserConfig.load_config('std')]

        if output_framework in ['tensorflow-pb', 'tensorflow-py', 'tensorflow-lite']:
            from nnef_tools.conversion.tensorflow import nnef_to_tf
            configs.append(nnef_to_tf.ParserConfig)
        elif output_framework in ['onnx']:
            from nnef_tools.conversion.onnx import nnef_to_onnx
            configs.append(nnef_to_onnx.ParserConfig)
        else:
            assert False

        module_names = [n.strip() for n in custom_converters.split(',')] if custom_converters else []
        for module_name in module_names:
            module = importlib.import_module(module_name)

            custom_fragments = ""
            if hasattr(module, "NNEF_OP_DEFINITIONS"):
                custom_fragments = module.NNEF_OP_DEFINITIONS
            custom_expands = []
            if hasattr(module, "NNEF_LOWERED_OPS"):
                custom_expands = module.NNEF_LOWERED_OPS
            custom_shapes = {}
            if hasattr(module, "NNEF_SHAPE_PROPAGATORS"):
                custom_shapes = module.NNEF_SHAPE_PROPAGATORS

            if custom_shapes or custom_fragments or custom_expands:
                configs.append(NNEFParserConfig(source=custom_fragments, shapes=custom_shapes, expand=custom_expands))

        return Reader(parser_configs=configs)
    elif input_framework == 'tensorflow-pb':
        # TODO custom converter
        from nnef_tools.io.tensorflow.tf_pb_io import Reader
        return Reader(convert_to_tf_py=True, input_shape=input_shape)
    elif input_framework == 'tensorflow-py':
        from nnef_tools.io.tensorflow.tf_py_io import Reader
        if custom_converters:
            custom_traceable_functions = get_tf_py_custom_traceable_functions(custom_converters)
        else:
            custom_traceable_functions = None
        return Reader(expand_gradients=True, custom_traceable_functions=custom_traceable_functions)
    elif input_framework == 'tensorflow-lite':
        # TODO custom converter
        from nnef_tools.io.tensorflow.tflite_io import Reader
        return Reader(convert_to_tf_py=True)
    elif input_framework == 'onnx':
        # TODO custom converter
        from nnef_tools.io.onnx.onnx_io import Reader
        return Reader(propagate_shapes=True, input_shape=input_shape)
    else:
        assert False


def get_custom_converters(input_framework, output_framework, custom_converters):
    module_names = [n.strip() for n in custom_converters.split(',')] if custom_converters else []
    custom_converter_by_op_name = {}

    for module_name in module_names:
        module = importlib.import_module(module_name)
        attrname = (input_framework + "_to_" + output_framework + "_converters").replace('-', '_').upper()
        if hasattr(module, attrname):
            custom_converter_by_op_name.update(getattr(module, attrname))

    return custom_converter_by_op_name


def get_converter(input_framework, output_framework, prefer_nchw=False, permissive=False, custom_converters=None):
    if input_framework in ['tensorflow-py', 'tensorflow-pb', 'tensorflow-lite'] and output_framework == 'nnef':
        from nnef_tools.conversion.tensorflow.tf_to_nnef import Converter

        if input_framework == 'tensorflow-py' and custom_converters:
            custom_converter_by_op_name = get_custom_converters(input_framework, output_framework, custom_converters)
        else:
            custom_converter_by_op_name = None

        return Converter(enable_imprecise_image_resize=permissive,
                         custom_converter_by_op_name=custom_converter_by_op_name)
    elif input_framework == 'nnef' and output_framework in ['tensorflow-py', 'tensorflow-pb', 'tensorflow-lite']:
        from nnef_tools.conversion.tensorflow.nnef_to_tf import Converter

        if output_framework == 'tensorflow-py' and custom_converters:
            custom_converter_by_op_name = get_custom_converters(input_framework, output_framework, custom_converters)
        else:
            custom_converter_by_op_name = None

        return Converter(prefer_nhwc=not prefer_nchw,
                         enable_imprecise_image_resize=permissive,
                         enable_imprecise_padding_border=permissive,
                         custom_converter_by_op_name=custom_converter_by_op_name)
    elif input_framework == 'onnx' and output_framework == 'nnef':
        from nnef_tools.conversion.onnx.onnx_to_nnef import Converter

        custom_converter_by_op_name = (get_custom_converters(input_framework, output_framework, custom_converters)
                                       if custom_converters else None)

        return Converter(custom_converter_by_op_name=custom_converter_by_op_name)
    elif input_framework == 'nnef' and output_framework == 'onnx':
        from nnef_tools.conversion.onnx.nnef_to_onnx import Converter

        custom_converter_by_op_name = (get_custom_converters(input_framework, output_framework, custom_converters)
                                       if custom_converters else None)

        return Converter(enable_imprecise_image_resize=permissive,
                         custom_converter_by_op_name=custom_converter_by_op_name)
    else:
        assert False


def get_data_format_optimizer(input_framework, output_framework, io_transformation):
    if input_framework in ['tensorflow-py', 'tensorflow-pb', 'tensorflow-lite', 'onnx'] and output_framework == 'nnef':
        from nnef_tools.optimization.nnef.nnef_data_format_optimizer import Optimizer
        return Optimizer(io_transform=io_transformation, merge_transforms_into_variables=True)
    elif input_framework == 'nnef' and output_framework in ['tensorflow-py', 'tensorflow-pb', 'tensorflow-lite']:
        from nnef_tools.optimization.tensorflow.tf_data_format_optimizer import Optimizer
        return Optimizer(io_transform=io_transformation, merge_transforms_into_variables=True)
    elif input_framework == 'nnef' and output_framework == 'onnx':
        from nnef_tools.optimization.onnx.onnx_data_format_optimizer import Optimizer
        return Optimizer(io_transform=io_transformation, merge_transforms_into_variables=True)
    else:
        assert False


def get_writer(input_framework, output_framework, with_weights=True, custom_converters=None):
    if output_framework == 'nnef':
        from nnef_tools.io.nnef.nnef_io import Writer
        return Writer(write_weights=with_weights)
    elif output_framework == 'tensorflow-py':
        from nnef_tools.io.tensorflow.tf_py_io import Writer

        if custom_converters:
            custom_imports, custom_op_protos = get_tf_py_imports_and_op_protos(custom_converters)
        else:
            custom_imports, custom_op_protos = None, None

        return Writer(write_weights=with_weights, custom_op_protos=custom_op_protos, custom_imports=custom_imports)
    elif output_framework == 'tensorflow-pb':
        from nnef_tools.io.tensorflow.tf_pb_io import Writer
        return Writer(convert_from_tf_py=True)
    elif output_framework == 'tensorflow-lite':
        from nnef_tools.io.tensorflow.tflite_io import Writer
        return Writer(convert_from_tf_py=True)
    elif output_framework == 'onnx':
        from nnef_tools.io.onnx.onnx_io import Writer
        return Writer()
    else:
        assert False


def get_output_file_name(output_framework, output_directory, compress):
    if output_framework == 'nnef':
        if compress:
            return os.path.join(output_directory, "model.nnef.tgz")
        else:
            return os.path.join(output_directory, "model")
    elif output_framework == 'tensorflow-py':
        return os.path.join(output_directory, "model")
    elif output_framework == 'tensorflow-pb':
        return os.path.join(output_directory, "model.pb")
    elif output_framework == 'onnx':
        return os.path.join(output_directory, "model.onnx")
    elif output_framework == 'tensorflow-lite':
        return os.path.join(output_directory, "model.tflite")
    else:
        assert False


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
    combined_info = conversion_info.compose(conv_info, opt_info, write_info)
    ensure_dirs(os.path.dirname(out_info_filename))
    conversion_info.dump(combined_info, out_info_filename)


def convert(input_framework,
            output_framework,
            input_model,
            output_directory,
            input_shape=None,
            prefer_nchw=False,
            io_transformation=None,
            compress=False,
            permissive=False,
            with_weights=True,
            custom_converters=None):
    convert_using_premade_objects(in_filename=input_model,
                                  out_filename=get_output_file_name(output_framework=output_framework,
                                                                    output_directory=output_directory,
                                                                    compress=compress),
                                  out_info_filename=os.path.join(output_directory, "conversion.json"),
                                  reader=get_reader(input_framework=input_framework,
                                                    output_framework=output_framework,
                                                    input_shape=input_shape,
                                                    permissive=permissive,
                                                    with_weights=with_weights,
                                                    custom_converters=custom_converters),
                                  converter=get_converter(input_framework=input_framework,
                                                          output_framework=output_framework,
                                                          prefer_nchw=prefer_nchw,
                                                          permissive=permissive,
                                                          custom_converters=custom_converters),
                                  data_format_optimizer=get_data_format_optimizer(input_framework=input_framework,
                                                                                  output_framework=output_framework,
                                                                                  io_transformation=io_transformation),
                                  writer=get_writer(input_framework=input_framework,
                                                    output_framework=output_framework,
                                                    with_weights=with_weights,
                                                    custom_converters=custom_converters))


def tf_py_has_checkpoint(input_model):
    parts = input_model.split(':')
    return len(parts) >= 2 and parts[1]


def convert_using_args(args):
    convert(input_framework=args.input_framework,
            output_framework=args.output_framework,
            input_model=args.input_model,
            output_directory=args.output_directory,
            input_shape=args.input_shape,
            prefer_nchw=args.prefer_nchw,
            io_transformation=args.io_transformation,
            compress=args.compress,
            permissive=args.permissive,
            with_weights=not args.no_weights,
            custom_converters=args.custom_converters)


def get_args(argv):
    parser = argparse.ArgumentParser(description="NNEFTools/convert: Neural network conversion tool",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="""Tips:
- If you refer to a Python package or module that is not in the current directory, 
please add its location to PYTHONPATH.
- Quote parameters if they contain spaces or special characters.""")

    parser.add_argument("--input-framework",
                        choices=['nnef', 'tensorflow-pb', 'tensorflow-py', 'tensorflow-lite', 'onnx'],
                        required=True,
                        help="""Input framework""")
    parser.add_argument("--output-framework",
                        choices=['nnef', 'tensorflow-pb', 'tensorflow-py', 'tensorflow-lite', 'onnx'],
                        required=True,
                        help="""Output framework""")
    parser.add_argument("--input-model",
                        required=True,
                        help="""nnef: path of NNEF file, directory or nnef.tgz file.
    In case of a single NNEF file, no weights are loaded.
onnx: path of ONNX file 
tensorflow-pb: path of pb file
tensorflow-py: package.module.function or package.module.function:checkpoint_path.ckpt
tensorflow-lite: path of tflite file""")

    parser.add_argument("--output-directory",
                        default="convert.out",
                        help="""Path of output directory. 
Default: convert.out""")

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

    args = parser.parse_args(args=argv[1:])
    args.io_transformation = parse_io_transform(args.io_transformation)
    if args.input_framework == 'tensorflow-py':
        check_tf_py_input_model(args.input_model)
    if args.input_framework == 'tensorflow-pb':
        args.input_shape = tf_pb_parse_input_shapes(args.input_shape)
    elif args.input_framework == 'onnx':
        args.input_shape = onnx_parse_input_shapes(args.input_shape)
    else:
        args.input_shape = None

    args.no_weights = False

    if sum(framework == 'nnef' for framework in [args.input_framework, args.output_framework]) != 1:
        print("Error: Either input or output framework must be nnef.", file=sys.stderr)
        exit(1)

    if args.input_framework == 'tensorflow-py' and not tf_py_has_checkpoint(args.input_model):
        args.no_weights = True

    if args.compress and args.output_framework != 'nnef':
        print("Error: --compress can now be only used with NNEF as output framework.")
        exit(1)

    if args.input_framework == "nnef" and not os.path.isdir(args.input_model) and not args.input_model.endswith('.tgz'):
        args.no_weights = True

    return args


def convert_using_argv(argv):
    args = get_args(argv)

    if 'tensorflow-py' in [args.input_framework, args.output_framework]:
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
