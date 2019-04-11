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

# TODO Not final api (used for testing but probably it will be removed)

from __future__ import division, print_function, absolute_import

import os
import shutil
import typing
from enum import Enum

from nnef_tools.activation_export.tensorflow import tf_activation_exporter
from nnef_tools.conversion import conversion_info
from nnef_tools.conversion.tensorflow import nnef_to_tf, tf_pb_to_tf_py, tf_to_nnef
from nnef_tools.conversion.tensorflow.nnef_to_tf import Converter as NNEFToTFConverter
from nnef_tools.conversion.tensorflow.tf_to_nnef import Converter as TFToNNEFConverter
from nnef_tools.core import utils
from nnef_tools.io.nnef import nnef_io
from nnef_tools.io.nnef.nnef_graph import NNEFGraph, NNEFTensor, NNEFOperation
from nnef_tools.io.nnef.parser_config import NNEFParserConfig
from nnef_tools.io.tensorflow import tf_pb_io
from nnef_tools.io.tensorflow.tf_graph import TFGraph, TFTensor, TFOperation
from nnef_tools.optimization.nnef import nnef_data_format_optimizer
from nnef_tools.optimization.nnef.nnef_data_format_optimizer import IOTransform
from nnef_tools.optimization.tensorflow import tf_data_format_optimizer


class OptimizationLevel(Enum):
    NONE = 0
    INTERNAL = 1
    FULL = 2


_TFToNNEFConverterFunType = \
    typing.Callable[[TFToNNEFConverter, TFOperation, NNEFGraph], None]
_NNEFToTFConverterFunType = \
    typing.Callable[[NNEFToTFConverter, NNEFOperation, TFGraph], None]


def _load_graph(frozen_graph_filename):
    import tensorflow as tf
    tf.reset_default_graph()
    tf.set_random_seed(0)
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def _call_optimizer(optimizer,
                    graph,
                    custom_transposable_ops,
                    io_transform,
                    verbose,
                    rename_tensors,
                    optimization_level,
                    remove_unneeded_copies,
                    remove_inverse_transposes,
                    merge_transforms_into_variables,
                    merge_transforms_into_constants):
    if optimization_level is None:
        optimization_level = OptimizationLevel.NONE

    if optimization_level == OptimizationLevel.NONE:
        remove_unneeded_copies = utils.first_set(remove_unneeded_copies, False)
        remove_inverse_transposes = utils.first_set(remove_inverse_transposes, False)
        merge_transforms_into_variables = utils.first_set(merge_transforms_into_variables, False)
        merge_transforms_into_constants = utils.first_set(merge_transforms_into_constants, False)
    elif optimization_level == OptimizationLevel.INTERNAL:
        remove_unneeded_copies = utils.first_set(remove_unneeded_copies, True)
        remove_inverse_transposes = utils.first_set(remove_inverse_transposes, True)
        merge_transforms_into_variables = utils.first_set(merge_transforms_into_variables, False)
        merge_transforms_into_constants = utils.first_set(merge_transforms_into_constants, False)
    elif optimization_level == OptimizationLevel.FULL:
        remove_unneeded_copies = utils.first_set(remove_unneeded_copies, True)
        remove_inverse_transposes = utils.first_set(remove_inverse_transposes, True)
        merge_transforms_into_variables = utils.first_set(merge_transforms_into_variables, True)
        merge_transforms_into_constants = utils.first_set(merge_transforms_into_constants, True)

    return optimizer.optimize(g=graph,
                              remove_unneeded_copies=remove_unneeded_copies,
                              remove_inverse_transposes=remove_inverse_transposes,
                              merge_transforms_into_constants=merge_transforms_into_constants,
                              merge_transforms_into_variables=merge_transforms_into_variables,
                              custom_transposable_ops=custom_transposable_ops,
                              io_transform=io_transform,
                              verbose=verbose,
                              rename_tensors=rename_tensors)


def convert_tf_pb_to_nnef(
        # Main parameters
        file_name,
        output_directory,
        network_name,
        source_shapes=None,
        source_dtypes=None,
        compress=False,

        # Extra parameters
        verbose=False,
        allow_extensions=False,
        allow_gradients=False,
        optimization_level=None,
        io_transform=None,
        activation_export_feed_dict=None,
        activation_export_io_only=False,
        overwrite=False,

        # Module level parameters
        converter_allow_imprecise_image_resize=False,
        optimizer_remove_unneeded_copies=None,
        optimizer_remove_inverse_transposes=None,
        optimizer_merge_transforms_into_variables=None,
        optimizer_merge_transforms_into_constants=None):
    # type: (...)->None

    assert utils.is_identifier(network_name), \
        "Network name must be None or a valid python identifier"

    if os.path.exists(output_directory):
        if overwrite:
            shutil.rmtree(output_directory)
        else:
            assert False, "{} exists, delete it or use overwrite=True".format(output_directory)

    g = tf_pb_io.read_tf_graph_from_protobuf(file_name)

    tf_pb_to_tf_py.evaluate_and_convert(tf_graph=g, source_shapes=source_shapes, source_dtypes=source_dtypes)

    converter = tf_to_nnef.Converter(
        enable_imprecise_image_resize=converter_allow_imprecise_image_resize)

    h, conv_info = converter(g)
    h.name = network_name

    conversion_info.dump(conv_info, os.path.join(output_directory, "step1.json"))

    nnef_io.write(h, os.path.join(output_directory,
                                  network_name + ("_not_optimized.nnef.tgz" if compress else "_not_optimized_nnef")))

    opt_info = _call_optimizer(optimizer=nnef_data_format_optimizer,
                               graph=h,
                               custom_transposable_ops=None,
                               io_transform=io_transform,
                               verbose=verbose,
                               rename_tensors=True,
                               optimization_level=optimization_level,
                               remove_unneeded_copies=optimizer_remove_unneeded_copies,
                               remove_inverse_transposes=optimizer_remove_inverse_transposes,
                               merge_transforms_into_variables=optimizer_merge_transforms_into_variables,
                               merge_transforms_into_constants=optimizer_merge_transforms_into_constants)

    conversion_info.dump(opt_info, os.path.join(output_directory, "step2.json"))
    conv_info = conversion_info.compose(conv_info, opt_info)
    conversion_info.dump(conv_info, os.path.join(output_directory, "conversion.json"))

    nnef_io.write(h, os.path.join(output_directory, network_name + (".nnef.tgz" if compress else "_nnef")))

    if activation_export_feed_dict:
        _load_graph(file_name)
        tf_activation_exporter.export(output_path=os.path.join(output_directory, "activations"),
                                      feed_dict=activation_export_feed_dict,
                                      conversion_info=conv_info,
                                      verbose=verbose,
                                      input_output_only=activation_export_io_only)

    if verbose:
        print("Done.")


def convert_nnef_to_tf_pb(
        # Main parameters
        nnef_tgz_or_dir_path,
        output_directory,
        custom_converter_by_op_name=None,  # type: typing.Dict[str, _NNEFToTFConverterFunType]

        # Extra parameters
        with_weights=True,
        verbose=False,
        prefer_nhwc=True,
        overwrite=False,
        optimization_level=None,
        io_transform=None,

        # Module level parameters
        converter_allow_imprecise_image_resize=False,
        converter_allow_imprecise_padding_border=False,
        optimizer_remove_unneeded_copies=None,
        optimizer_remove_inverse_transposes=None,
        optimizer_merge_transforms_into_variables=None,
        optimizer_merge_transforms_into_constants=None,
        extra_parser_configs=None):
    if os.path.exists(output_directory):
        if overwrite:
            shutil.rmtree(output_directory)
        else:
            assert False, "{} exists, delete it or use overwrite=True".format(output_directory)

    parser_configs = [nnef_to_tf.ParserConfig, NNEFParserConfig.STANDARD_CONFIG]
    if extra_parser_configs:
        parser_configs += extra_parser_configs
    g = nnef_io.read(nnef_tgz_or_dir_path, parser_configs=parser_configs)

    converter = nnef_to_tf.Converter(
        prefer_nhwc=prefer_nhwc,
        enable_imprecise_image_resize=converter_allow_imprecise_image_resize,
        enable_imprecise_padding_border=converter_allow_imprecise_padding_border,
        custom_converter_by_op_name=custom_converter_by_op_name)
    h, conv_info = converter(g)
    conversion_info.dump(conv_info, os.path.join(output_directory, "step1.json"))

    opt_info = _call_optimizer(optimizer=tf_data_format_optimizer,
                               graph=h,
                               custom_transposable_ops=None,
                               io_transform=io_transform,
                               verbose=verbose,
                               rename_tensors=True,
                               optimization_level=optimization_level,
                               remove_unneeded_copies=optimizer_remove_unneeded_copies,
                               remove_inverse_transposes=optimizer_remove_inverse_transposes,
                               merge_transforms_into_variables=optimizer_merge_transforms_into_variables,
                               merge_transforms_into_constants=optimizer_merge_transforms_into_constants)
    conversion_info.dump(opt_info, os.path.join(output_directory, "step2.json"))

    rename_info = tf_pb_io.write_tf_graph_to_protobuf(graph=h,
                                                      filename=os.path.join(output_directory, 'graph.pb'),
                                                      convert_from_tf_py=True)

    conversion_info.dump(rename_info, os.path.join(output_directory, "step3.json"))

    conv_info = conversion_info.compose(conv_info, opt_info, rename_info)
    conversion_info.dump(conv_info, os.path.join(output_directory, "conversion.json"))

    if verbose:
        print("Done.")


__all__ = [
    'nnef_io',
    'tf_pb_io',
    'NNEFGraph',
    'NNEFTensor',
    'NNEFOperation',
    'TFGraph',
    'TFTensor',
    'TFOperation',
    'TFToNNEFConverter',
    'convert_tf_pb_to_nnef',
    'convert_nnef_to_tf_pb',
    'OptimizationLevel',
    'IOTransform',
    'NNEFParserConfig',
    'NNEFToTFConverter',
]
