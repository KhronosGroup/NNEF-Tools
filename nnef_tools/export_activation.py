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

# Python2: Ensure that load from current directory is enabled, but load from the directory of the script is disabled
if len(sys.path) == 0:
    sys.path.append('')
if sys.path[0] != '':
    sys.path[0] = ''

import argparse
import importlib
import os

import numpy as np
import six

from nnef_tools.io.input_source import RandomInput, ImageInput, NNEFTensorInput, create_feed_dict
from nnef_tools.conversion import conversion_info
from nnef_tools.core import utils


def parse_input_shapes(s):
    if not s:
        return None

    try:
        return eval(s)
    except Exception:
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


def get_args(argv):
    parser = argparse.ArgumentParser(description="NNEFTools/export_activation: Neural network activation export tool",
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="""Tips:
- If you refer to a Python package or module that is not in the current directory,
please add its location to PYTHONPATH.
- Quote parameters if they contain spaces or special characters.
""")

    parser.add_argument("--input-format",
                        choices=['tensorflow-pb', 'tensorflow-py', 'caffe'],
                        required=True,
                        help="""Input format""")

    parser.add_argument("--input-model",
                        nargs='+',
                        required=True,
                        help="""tensorflow-pb: filename.pb
    tensorflow-py: package.module.function [filename.ckpt]
    """)

    parser.add_argument("--conversion-info",
                        required=True,
                        help="""Path of the conversion.json file that was created by the convert tool when converting the model to NNEF.""")

    parser.add_argument("--output-path",
                        help="""Path of output directory. Default: {model_name}.activations where --conversion-info={model_name}.conversion.json """)

    parser.add_argument('--input',
                        default="Random(0.0, 1.0, 0, 255, 0.5)",
                        help="""An input_source or a tensor_name->input_source dict in Python syntax.
 Tensor names should be the names in the input model (not the converted names if different).
 The following input sources are supported:
 - Random(min, max) for int and float
 - Random(true_prob) for bool
 - Random(float_min, float_max, int_min, int_max, true_prob) for all types
    - Keyword arguments can not be used with Random.
 - Image(filename, color_format='RGB', data_format='NCHW', range=None, norm=None) for int and float
   - Arguments:
     - filename: string or list of strings, path(s) of jpg/png images, can have * in them
     - color_format: RGB or BGR
     - data_format: NCHW or NHWC
     - range: [start, end] closed range
     - norm: [mean, std] or [[mean0, mean1, mean2], [std0, std1, std2]]
   - The image is processed as follows:
     - The image is loaded to float32[width, height, channels], values ranging from 0 to 255.
     - The image is reordered to RGB or BGR, as requested.
     - If range is not None, the image is transformed to the specified range. 
       (The transform does not depend on the content of the image.)
     - If norm is not None: image = (image - mean) / std
     - The image is transformed to NCHW or NHWC as requested. 
     - The image is casted to the target data type.
 - Tensor(filename) for all types
   - filename must be the path of an NNEF tensor file (.dat)
 Default: Random(0.0, 1.0, 0, 255, 0.5).""")

    parser.add_argument('--input-shape',
                        default="",
                        help="""tensorflow-pb:
The shape of input tensors might be incomplete in the input model.
For example they could be [?, 224, 224, 3] (unknown batch size) or ? (unknown rank and shape).

Set all missing dimensions to 10 (if the rank is known):
--input-shape=10

Set all input shapes to [10, 224, 224, 3]:
--input-shape="[10, 224, 224, 3]"

Set different input shapes for each input:
--input-shape="{'input_name_1': [10, 224, 224, 3], 'input_name_2': [10, 299, 299, 3]}"
To get the input names, and (possibly incomplete) input shapes, you can run the converter without --input-shape, 
they will be listed if any of them is incomplete.

Default: Unknown dimensions are set to 1. If the rank is unknown this option can not be omitted.
""")

    parser.add_argument('--tensors-at-once',
                        type=int,
                        help="""Number of tensors to evaluate at once.
On a computer with low (gpu) memory, a lower number is appropriate.
All tensors will evaluated but in groups of size '--tensors-at-once'.
Default: 25.

Not supported for Caffe.
""")

    args = parser.parse_args(args=argv[1:])
    args.input = parse_input(args.input)

    allowed_input_length = {
        'tensorflow-pb': [1],
        'tensorflow-py': [1, 2],
        'caffe': [2],
    }
    if not len(args.input_model) in allowed_input_length[args.input_format]:
        print("Error: {} values specified to --input-model, allowed: {}"
              .format(len(args.input_model), ', '.join(str(i) for i in allowed_input_length[args.input_format])),
              file=sys.stderr)
        exit(1)

    if args.input_format == 'tensorflow-pb':
        args.input_shape = parse_input_shapes(args.input_shape)
    else:
        args.input_shape = None

    args.no_weights = False

    if args.input_format == 'tensorflow-py' and len(args.input_model) < 2:
        args.no_weights = True

    if args.input_format == 'caffe':
        if args.tensors_at_once is not None:
            print("Error: --tensors-at-once is not supported for Caffe.")
            exit(1)
    else:
        if args.tensors_at_once is None:
            args.tensors_at_once = 25

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


def tf_shape_normalize(shape):
    if shape.dims is None:
        return None
    else:
        return [int(dim.value) if dim.value is not None else -1 for dim in shape.dims]


def tf_get_input_shapes(input_shape=None):
    def get_shape_for(name):
        if isinstance(input_shape, dict) and name in input_shape:
            return input_shape[name]
        elif isinstance(input_shape, list):
            return list(input_shape)
        elif utils.is_anyint(input_shape):
            return utils.anyint_to_int(input_shape)
        return None

    if isinstance(input_shape, dict):
        input_shape = {(k + ':0' if ':' not in k else k): v for k, v in six.iteritems(input_shape)}

    placeholders = tf_get_placeholders()
    new_input_shapes = {}

    if input_shape is None:
        if any(tf_shape_normalize(tensor.shape) is None or -1 in tf_shape_normalize(tensor.shape)
               for tensor in placeholders):
            for tensor in placeholders:
                print("Info: Input shape: {}: {}".format(tensor.name, tf_shape_normalize(tensor.shape)))

    for tensor in placeholders:
        tensor_shape = tf_shape_normalize(tensor.shape)
        shape_for_this = get_shape_for(tensor.name) if tensor.name else None
        if isinstance(shape_for_this, list):
            if not utils.compatible_shapes(tensor_shape, shape_for_this):
                raise utils.NNEFToolsException(
                    "The specified shape is incompatible with the original shape for {}. {} vs. {}".format(
                        tensor.name, shape_for_this, tensor_shape))
            tensor_shape = shape_for_this
        elif shape_for_this is None or isinstance(shape_for_this, int):
            if tensor_shape is None:
                raise utils.NNEFToolsException(
                    "The full shape must be specified for {}, because it is unknown.".format(tensor.name))
            elif -1 in tensor_shape:
                if shape_for_this is None:
                    shape_for_this = 1
                    print("Warning: Incomplete input shape is auto-fixed: {}. {} -> {}. "
                          "Use --input-shape if other shape is desired.".format(
                        tensor.name, tensor_shape, [shape_for_this if dim == -1 else dim for dim in tensor_shape]))
                tensor_shape = [shape_for_this if dim == -1 else dim for dim in tensor_shape]
        else:
            assert False

        if tensor.dtype is None:
            raise utils.NNEFToolsException("An input tensor has incomplete dtype, "
                                           "we have thought that this is impossible, "
                                           "please file a bug report to NNEF Tools.")

        new_input_shapes[tensor.name] = (tensor.dtype.name, tensor_shape)

    return new_input_shapes


def tf_export_activations(input_shapes,
                          input_sources,
                          conversion_info_file_name,
                          output_path,
                          checkpoint_path=None,
                          init_variables=False,
                          tensors_per_iter=25):
    from nnef_tools.activation_export.tensorflow.tf_activation_exporter import export
    input_shapes = tf_get_input_shapes(input_shape=input_shapes)
    np.random.seed(0)
    feed_dict = create_feed_dict(input_sources=input_sources, input_shapes=input_shapes)
    info = conversion_info.load(conversion_info_file_name)
    export(output_path=output_path,
           feed_dict=feed_dict,
           conversion_info=info,
           checkpoint_path=checkpoint_path,
           tensors_per_iter=tensors_per_iter,
           init_variables=init_variables)


def export_activation(input_format, input_model, input_shape, input_source, conversion_info, output_path,
                      tensors_per_iter):
    if output_path is None:
        output_path = conversion_info
        if output_path.endswith('.conversion.json'):
            output_path = output_path[:-len('.conversion.json')]
        output_path += ".activations"

    if input_format == 'tensorflow-py':
        tf_reset()
        network_function = get_function_by_path(input_model[0])
        network_function()
        ensure_dirs(output_path)
        checkpoint_path = input_model[1] if len(input_model) >= 2 else None
        tf_export_activations(input_shapes=input_shape,
                              input_sources=input_source,
                              conversion_info_file_name=conversion_info,
                              output_path=output_path,
                              checkpoint_path=checkpoint_path,
                              init_variables=not checkpoint_path,
                              tensors_per_iter=tensors_per_iter)
    elif input_format == 'tensorflow-pb':
        tf_reset()
        tf_set_default_graph_from_pb(input_model[0])
        ensure_dirs(output_path)
        tf_export_activations(input_shapes=input_shape,
                              input_sources=input_source,
                              conversion_info_file_name=conversion_info,
                              output_path=output_path,
                              tensors_per_iter=tensors_per_iter,
                              init_variables=False)
    elif input_format == 'caffe':
        from nnef_tools.activation_export.caffe.caffe_activation_exporter import export as caffe_export_activations
        ensure_dirs(output_path)
        caffe_export_activations(prototxt_path=input_model[0],
                                 caffemodel_path=input_model[1],
                                 input_source=input_source,
                                 conversion_info_path=conversion_info,
                                 output_path=output_path)
    else:
        assert False


def export_activation_using_argv(argv):
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args = get_args(argv=argv)

    try:
        export_activation(input_format=args.input_format,
                          input_model=args.input_model,
                          input_shape=args.input_shape,
                          input_source=args.input,
                          conversion_info=args.conversion_info,
                          output_path=args.output_path,
                          tensors_per_iter=args.tensors_at_once)
    except utils.NNEFToolsException as e:
        print("Error: " + str(e), file=sys.stderr)
        exit(1)


# Call this if you don't want to reload the whole program for each run
def export_activation_using_command(command):
    return export_activation_using_argv(utils.command_to_argv(command))


if __name__ == '__main__':
    export_activation_using_argv(sys.argv)
