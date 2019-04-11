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

import argparse
import importlib
import os

import numpy as np
import six

from nnef_tools.activation_export.input_source import RandomInput, ImageInput, NNEFTensorInput, create_input
from nnef_tools.conversion import conversion_info
from nnef_tools.core import utils


def check_tf_py_input_model(s):
    parts = s.split(':')
    if len(parts) not in [1, 2] or not parts[0] or '.' not in parts[0]:
        print('Error: Can not parse the --input-model parameter: {}.'.format(s), file=sys.stderr)
        exit(1)


def parse_input_shapes(s):
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


def parse_input(s):
    if not s:
        return RandomInput(0.0, 1.0, 0, 255, 0.5)

    locals()['Random'] = RandomInput
    locals()['Image'] = ImageInput
    locals()['Tensor'] = NNEFTensorInput
    # allow parsing without quotes
    locals()['RGB'] = 'RGB'
    locals()['BGR'] = 'BGR'
    locals()['NCHW'] = 'NCHW'
    locals()['NHWC'] = 'NHWC'

    try:
        return eval(s)
    except Exception as e:
        print('Error: Can not evaluate the --input parameter: {}.'.format(s), file=sys.stderr)
        exit(1)


def ensure_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def tf_py_has_checkpoint(input_model):
    parts = input_model.split(':')
    return len(parts) >= 2 and parts[1]


def get_args():
    parser = argparse.ArgumentParser(description="NNEFTools/export_activation: Neural network activation export tool",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="""Tips:
- If you refer to a Python package or module that is not in the current directory,
please add its location to PYTHONPATH.
- Quote parameters if they contain spaces or special characters.
""")

    parser.add_argument("--input-framework",
                        choices=['tensorflow-pb', 'tensorflow-py'],
                        required=True,
                        help="""Input framework""")

    parser.add_argument("--input-model",
                        required=True,
                        help="""tensorflow-pb: path of pb file
tensorflow-py: package.module:function or package.module:function:checkpoint_path.ckpt""")

    parser.add_argument("--conversion-info",
                        required=True,
                        help="""Path of the conversion.json file that was created by the convert tool when converting the model to NNEF.""")

    parser.add_argument("--output-directory",
                        default="export_activation.out",
                        help="""Path of output directory. Default: export_activation.out""")

    parser.add_argument('--input',
                        default="Random(0.0, 1.0, 0, 255, 0.5)",
                        help="""An input_source or a tensor_name->input_source dict in Python syntax.
 Tensor names should be the names in the input model (not the converted names if different).
 The following input sources are supported:
 - Random(min, max) for int and float
 - Random(true_prob) for bool
 - Random(float_min, float_max, int_min, int_max, true_prob) for all types
    - Keyword arguments can not be used with Random.
 - Image(filename, color_format='RGB', data_format='NCHW', sub=127.5, div=127.5) for int and float
   - Arguments:
     - filename: string or list of strings, path(s) of jpg/png images
     - color_format: RGB or BGR
     - data_format: NCHW or NHWC
     - sub: float
     - div: float
   - The applied image preprocessing is the following in pseudocode (where input_size and input_dtype comes from the network or --input-shape):
       image = ((uint8_image.astype(float)-sub)/div).resize(input_size).astype(input_dtype)
 - Tensor(filename) for all types
   - filename must be the path of an NNEF tensor file (.dat)
 Default: Random(0.0, 1.0, 0, 255, 0.5).""")

    parser.add_argument('--input-shape',
                        default="",
                        help="""tensorflow-pb: The dtype and shape of input tensors must be specified if they are not set in the protobuf file.
The value must be a (dtype, shape) tuple or a tensor_name->(dtype, shape) dict.
Dtype must be one of:
    float16, float32, float64,
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64
    bool
Default: (empty).""")

    parser.add_argument('--tensors-at-once',
                        default=25,
                        type=int,
                        help="""Number of tensors to evaluate at once.
On a computer with low (gpu) memory, a lower number is appropriate.
All tensors will evaluated but in groups of size '--tensors-at-once'.
Default: 25.""")

    args = parser.parse_args()
    args.input = parse_input(args.input)
    if args.input_framework == 'tensorflow-py':
        check_tf_py_input_model(args.input_model)
    if args.input_framework == 'tensorflow-pb':
        args.input_shape = parse_input_shapes(args.input_shape)
    else:
        args.input_shape = None

    args.no_weights = False

    if args.input_framework == 'tensorflow-py' and not tf_py_has_checkpoint(args.input_model) and not args.no_weights:
        args.no_weights = True

    return args


def tf_set_default_graph_from_pb(frozen_graph_filename):
    import tensorflow as tf
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def get_function_by_path(path):
    """

    :param path: "package.module.function" or "package.module.function:checkpoint_path.ckpt"
    :return: function
    """
    parts = path.split(':')

    assert len(parts) in [1, 2]
    if len(parts) == 1 or not parts[1]:
        package_and_module, function_name = parts[0].rsplit('.', 1)
    else:
        package_and_module, function_name = parts[0].rsplit('.', 1)

    sys.path.insert(0, '.')
    try:
        module = importlib.import_module(package_and_module)
    except ImportError:
        raise RuntimeError("Error: Can not import module {}".format(package_and_module))
    finally:
        sys.path = sys.path[1:]

    try:
        function_ = getattr(module, function_name)
        function_.__name__ = function_name
        return function_
    except AttributeError:
        raise RuntimeError(
            "Error: Function {} not found in module {}".format(function_name, package_and_module))


def tf_reset():
    import tensorflow as tf
    tf.reset_default_graph()
    tf.set_random_seed(0)


def tf_get_placeholders():
    import tensorflow as tf
    tensors = []
    for op in tf.get_default_graph().get_operations():
        if "Placeholder" in op.node_def.op:
            tensors.append(op.outputs[0])
    return sorted(tensors, key=lambda t: t.name)


def is_compatible(orig_shape, shape):
    if orig_shape is None:
        return True
    if len(orig_shape) != len(shape):
        return False
    for o, s in zip(orig_shape, shape):
        if o != s and o != -1:
            return False
    return True


def tf_shape_normalize(shape):
    if shape.dims is None:
        return None
    else:
        return [int(dim.value) if dim.value is not None else -1 for dim in shape.dims]


def tf_get_input_shapes(input_shape=None):
    if isinstance(input_shape, dict):
        input_shape = {(k + ':0' if ':' not in k else k): v for k, v in six.iteritems(input_shape)}
    placeholders = tf_get_placeholders()
    new_input_shapes = {}
    for tensor in placeholders:
        if isinstance(input_shape, dict):
            if tensor.name in input_shape:
                dtype, shape = input_shape[tensor.name]
                if not is_compatible(orig_shape=tf_shape_normalize(tensor.shape), shape=shape):
                    raise utils.NNEFToolsException(
                        "The specified shape is incompatible with the original shape for {}. {} vs. {}".format(
                            tensor.name, shape, tf_shape_normalize(tensor.shape)))
                if tensor.dtype is not None and tensor.dtype.name != dtype:
                    raise utils.NNEFToolsException(
                        "The specified dtype is incompatible with the original dtype for {}. {} vs. {}".format(
                            tensor.name, dtype, tensor.dtype.name))
            else:
                dtype, shape = tensor.dtype.name, tf_shape_normalize(tensor.shape)
        elif input_shape is not None:
            dtype, shape = input_shape
            if not is_compatible(orig_shape=tf_shape_normalize(tensor.shape), shape=shape):
                raise utils.NNEFToolsException(
                    "The specified shape is incompatible with the original shape for {}. {} vs. {}".format(
                        tensor.name, shape, tf_shape_normalize(tensor.shape)))
            if tensor.dtype is not None and tensor.dtype.name != dtype:
                raise utils.NNEFToolsException(
                    "The specified dtype is incompatible with the original dtype for {}. {} vs. {}".format(
                        tensor.name, dtype, tensor.dtype.name))
        else:
            dtype, shape = tensor.dtype.name, tf_shape_normalize(tensor.shape)
        new_input_shapes[tensor.name] = (dtype, shape)

    for k, v in six.iteritems(new_input_shapes):
        if v[0] is None or v[1] is None or None in v[1]:
            raise utils.NNEFToolsException("Source tensor '{}' has incomplete dtype or shape: {} {}\n"
                                           "Please specify it in --input-shape or through the corresponding API."
                                           .format(k, v[0], v[1]))
    return new_input_shapes


def create_feed_dict(input_sources, input_shapes):
    if not isinstance(input_sources, dict):
        input_sources = {k: input_sources for k in six.iterkeys(input_shapes)}

    np.random.seed(0)
    feed_dict = {}
    for name, (dtype, shape) in six.iteritems(input_shapes):
        assert name in input_sources
        feed_dict[name] = create_input(input_source=input_sources[name], np_dtype=dtype, shape=shape)

    return feed_dict


def tf_export_activations(input_shapes,
                          input_sources,
                          conversion_info_file_name,
                          output_directory,
                          checkpoint_path=None,
                          init_variables=False,
                          tensors_per_iter=25):
    from nnef_tools.activation_export.tensorflow.tf_activation_exporter import export
    input_shapes = tf_get_input_shapes(input_shape=input_shapes)
    feed_dict = create_feed_dict(input_sources=input_sources, input_shapes=input_shapes)
    info = conversion_info.load(conversion_info_file_name)
    export(output_path=output_directory,
           feed_dict=feed_dict,
           conversion_info=info,
           checkpoint_path=checkpoint_path,
           tensors_per_iter=tensors_per_iter,
           init_variables=init_variables)


def export_activations(input_framework, input_model, input_shape, input_source, conversion_info, output_directory,
                       tensors_per_iter):
    if input_framework == 'tensorflow-py':
        tf_reset()
        network_function = get_function_by_path(input_model)
        network_function()
        parts = input_model.split(':')
        ensure_dirs(output_directory)
        checkpoint_path = parts[1] if len(parts) >= 2 and parts[1] else None
        tf_export_activations(input_shapes=input_shape,
                              input_sources=input_source,
                              conversion_info_file_name=conversion_info,
                              output_directory=output_directory,
                              checkpoint_path=checkpoint_path,
                              init_variables=not checkpoint_path,
                              tensors_per_iter=tensors_per_iter)
    elif input_framework == 'tensorflow-pb':
        tf_reset()
        tf_set_default_graph_from_pb(input_model)
        ensure_dirs(output_directory)
        tf_export_activations(input_shapes=input_shape,
                              input_sources=input_source,
                              conversion_info_file_name=conversion_info,
                              output_directory=output_directory,
                              tensors_per_iter=tensors_per_iter,
                              init_variables=False)
    else:
        assert False


def export_activations_using_command_line_args(args):
    export_activations(input_framework=args.input_framework,
                       input_model=args.input_model,
                       input_shape=args.input_shape,
                       input_source=args.input,
                       conversion_info=args.conversion_info,
                       output_directory=args.output_directory,
                       tensors_per_iter=args.tensors_at_once)


def main():
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args = get_args()

    try:
        export_activations_using_command_line_args(args)
    except utils.NNEFToolsException as e:
        print("Error: " + str(e), file=sys.stderr)
        exit(1)


if __name__ == '__main__':
    main()
