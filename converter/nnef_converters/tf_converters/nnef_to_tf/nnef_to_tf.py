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

from __future__ import division, print_function

import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops as tf_gen_nn_ops

from . import converters
from . import transformations
from .dog_to_tf import tfdog_to_source
from ...common import dog
from ...common import nnef_parser_config
from ...common import nnef_shape_optimizer
from ...common import nnef_to_dog
from ...common import utils
from ...common.types import *


def debug_print_nnefgraph(nnefgraph):
    property_by_name, nnefops = nnefgraph
    shape_by_nnefdn_name = property_by_name["shapes"]
    for op_name, arg_by_name, result_by_name in nnefops:
        print(result_by_name, "=", op_name, arg_by_name)
    for dn_name, shape in shape_by_nnefdn_name.items():
        print("{}.shape={}".format(dn_name, shape))
    print()


def _convert(nnefgraph, verbose=False):
    # debug_print_nnefgraph(nnefgraph)
    nnefdog = nnef_to_dog.nnefgraph_to_nnefdog(nnefgraph)

    converter = converters.Converter(nnefdog)

    transformations.transform_extract_padding(nnefdog)
    transformations.transform_extract_padding_for_grads(nnefdog)
    transformations.transform_extract_bias_add(nnefdog)
    transformations.transform_extract_bias_add_for_grads(nnefdog)

    transformations.transform_transpose_to_target_lang(nnefdog)

    nnefdog.ops = nnef_shape_optimizer.transform_shape_optimizations(
        nnefops=nnefdog.ops,
        is_source_lang=True,
        dg_is_output_nnefdn=lambda nnefdn: nnefdn.name in nnefdog.output_dn_names)

    for nnefop in nnefdog.ops:
        if nnefop.name in converters.DefaultConverters:
            converters.DefaultConverters[nnefop.name](nnefop, converter)
        else:
            utils.print_error("No converter for {}".format(nnefop.name))

    nnef_shape_optimizer.copy_transformations(converter.nnefdn_by_tfdn)
    if verbose:
        nnef_shape_optimizer.add_comments(converter.nnefdn_by_tfdn.keys())

    tfdog = dog.Graph(
        nnefdog.name,
        list(converter.nnefop_by_tfop.keys()),
        {tfdn.name: tfdn for tfdn in converter.nnefdn_by_tfdn.keys()},
        nnefdog.input_dn_names,
        nnefdog.output_dn_names
    )

    trafo_dict = nnef_shape_optimizer.get_trafo_dict(tfdog.dn_by_name.values(),
                                                     [tfdog.dn_by_name[name] for name in tfdog.output_dn_names],
                                                     input_op_name="tf.placeholder",
                                                     variable_op_name="tf.get_variable",
                                                     constant_op_name="tf.constant",
                                                     label_arg_name="name")

    return tfdog, trafo_dict


def tfsource_to_function(src):
    # remove imports
    lines = []
    after_imports = False
    for line in src.split("\n"):
        if after_imports:
            lines.append(line)
        if line == "":
            after_imports = True
    src = "\n".join(lines)

    out_locals = {}
    exec(src, {"tf": tf, "tf_gen_nn_ops": tf_gen_nn_ops}, out_locals)
    return list(out_locals.values())[0]


def read_variables_to_tf(nnef_dir_name, trafo_dict):
    data = {}
    for label, trafos in trafo_dict["variable"].items():
        nnefarray = utils.read_nnef_tensor("{}/{}.dat".format(nnef_dir_name, label))
        data[label] = utils.np_apply_transforms(nnefarray, trafos)
    return data


def read_placeholders_to_tf(nnef_dir_name, nnef_activations_dir_name, trafo_dict):
    data = {}
    for name, trafos in trafo_dict["input"].items():
        nnefarray = utils.read_nnef_tensor("{}/{}.dat".format(nnef_activations_dir_name, name))
        data[name] = utils.np_apply_transforms(nnefarray, trafos)
    return data


def create_checkpoint_with_values(net_fun, file_name, variable_value_by_name):
    def get_tensors(op_name_part):
        tensors = []
        for op in tf.get_default_graph().get_operations():
            if op_name_part in op.node_def.op:
                tensors.append(op.outputs[0])
        return sorted(tensors, key=lambda t: t.name)

    def get_variables():
        return get_tensors("Variable")

    tf.reset_default_graph()
    net_fun()
    variables = get_variables()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        for variable in variables:
            value = variable_value_by_name[variable.name[:-2]]
            variable_shape = variable.shape.as_list()
            value_shape = list(value.shape)

            # TODO maybe not needed now
            if ((variable_shape == [1] and value_shape == [])
                    or (variable_shape == [] and value_shape == [1])
                    or (len(variable_shape) == 1 and value_shape == [1, variable_shape[0]])):
                value = np.reshape(value, variable_shape)

            sess.run(tf.assign(variable, value))
        saver.save(sess, file_name)


def _has_dat_file(dir_name):
    for root, dir_names, file_names in os.walk(dir_name):
        if any(file_name.endswith('.dat') for file_name in file_names):
            return True
    return False


def convert(nnef_path, output_path=".", verbose=False, _print_trafos=True, _extended_mode=False):
    # type: (str, str, bool, bool, bool)->Optional[Any]

    tmp_dir_name = None

    try:
        if verbose:
            print("Converting...")

        if nnef_path.endswith(".nnef.tgz"):
            tmp_dir_name = tempfile.mkdtemp(prefix="nnef_to_tf_")
            utils.tgz_extract(nnef_path, tmp_dir_name)
            nnef_dir_path = tmp_dir_name
            nnef_file_path = os.path.join(nnef_dir_path, "graph.nnef")
            export_variables = _has_dat_file(nnef_dir_path)
            if verbose and not export_variables:
                print("No weights were present in {}".format(nnef_path))
        elif nnef_path.endswith(".nnef"):
            nnef_dir_path = utils.without_file_name(nnef_path)
            nnef_file_path = nnef_path
            export_variables = False
            if verbose:
                print("No weights are exported when an NNEF file is given as input. "
                      "If you need that, try an archive or a directory.")
        else:
            nnef_dir_path = utils.without_slash(nnef_path)
            nnef_file_path = os.path.join(nnef_dir_path, "graph.nnef")
            export_variables = _has_dat_file(nnef_dir_path)
            if verbose and not export_variables:
                print("No weights were present in {}".format(nnef_path))

        tfdog, trafos = _convert(nnef_parser_config.parse_file(nnef_file_path), verbose=verbose)
        src = tfdog_to_source(tfdog, generate_name_map=_extended_mode)

        net_fun = tfsource_to_function(src)
        graph_name = net_fun.__name__

        output_path = utils.without_slash(output_path)
        output_py_path = os.path.join(output_path, graph_name + '.py')
        utils.ensure_dir(output_path)

        with open(output_py_path, "w") as f:
            print(src, end="", file=f)

        if _print_trafos:
            nnef_shape_optimizer.print_trafos(trafos, keys=["input", "output"])

        if verbose:
            print("Wrote {}".format(output_py_path))

        if export_variables:
            var_dict = read_variables_to_tf(nnef_dir_path, trafos)
            if var_dict:
                output_checkpoint_dir = os.path.join(output_path, graph_name + "_ckpt")
                output_checkpoint_path = os.path.join(output_checkpoint_dir, graph_name + ".ckpt")
                utils.ensure_dir(output_checkpoint_dir)

                create_checkpoint_with_values(net_fun, output_checkpoint_path, var_dict)
                if verbose:
                    print("Wrote {}".format(output_checkpoint_dir + "/*"))
        if _extended_mode:
            placeholder_value_by_name = read_placeholders_to_tf(nnef_dir_path,
                                                                os.path.join(nnef_dir_path, "activations"), trafos)
            feed_dict = {(k + ':0'): v for k, v in placeholder_value_by_name.items()}
            return (
                net_fun,
                feed_dict
            )

        utils.raise_if_had_error(listing=src)
    finally:
        if tmp_dir_name:
            shutil.rmtree(tmp_dir_name)
