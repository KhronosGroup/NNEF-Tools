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

from .dog_to_caffe import caffedog_to_prototxt
from .. import common
from ..common import CaffeGraph
from ..common import EXTRA_VARIABLE_LABELS, VARIABLE_LABEL_SKIP
from ..nnef_to_caffe import converters, transformations
from ...common import nnef_parser_config
from ...common import nnef_to_dog
from ...common import utils
from ...common.nnef_dog_types import NnefGraph
from ...common.types import *


def nnefdog_to_caffedog(nnefdog):
    # type: (NnefGraph)->CaffeGraph

    transformations.constant_to_variable(nnefdog)

    converter = converters.Converter(nnefdog)
    op_converters = converters.DefaultConverters
    for nnefop in nnefdog.ops:
        if nnefop.name in op_converters:
            op_converters[nnefop.name](nnefop, converter)
        else:
            utils.print_error("No converter for {}".format(nnefop.name))

    def get_name_by_source_name(source_name):
        if source_name in converter.caffedn_by_source_name:
            return converter.caffedn_by_source_name[source_name].name
        return "<<<ERROR>>>"

    caffedog = CaffeGraph(graph_name=nnefdog.name,
                          ops=list(converter.nnefop_by_caffeop.keys()),
                          dn_by_name={caffedn.name: caffedn for caffedn in converter.nnefdn_by_caffedn.keys()},
                          input_dn_names=[get_name_by_source_name(name) for name in nnefdog.input_dn_names],
                          output_dn_names=[get_name_by_source_name(name) for name in nnefdog.output_dn_names])

    transformations.unite_powers(caffedog)
    transformations.merge_up_bias(caffedog)

    def without_nones(x):
        # type: (List[Optional[str]])->List[str]
        return [y for y in x if y is not None]

    caffedog.extra[EXTRA_VARIABLE_LABELS] = {common.get_layer_name(op): without_nones(op.extra[EXTRA_VARIABLE_LABELS])
                                             for op in caffedog.ops if op.extra.get(EXTRA_VARIABLE_LABELS)}
    return caffedog


def convert_variables(nnef_dir_name, variable_labels, caffe_prototxt_file_name, caffe_model_file_name):
    import numpy as np
    import caffe

    net = caffe.Net(caffe_prototxt_file_name, caffe.TEST)
    for name, variables in net.params.items():
        labels = variable_labels.get(name, [])
        if len(labels) < len(variables):
            utils.print_error("{}: Missing weights: {}".format(name, list(range(len(labels), len(variables)))))
        if len(variables) < len(labels):
            utils.print_error("{}: Extra weights: {}".format(name, list(range(len(variables), len(labels)))))
        for variable, label in zip(variables, labels):
            if label == VARIABLE_LABEL_SKIP:
                continue
            if isinstance(label, list):
                if len(label) == 1:
                    variable.data[...] = label[0]
                else:
                    variable.data[...] = np.array(label).reshape(variable.data.shape)
                continue
            arr = utils.read_nnef_tensor("{}/{}.dat".format(nnef_dir_name, label))  # type: np.ndarray
            if ((len(arr.shape) == 2 and arr.shape == (1,) + variable.data.shape)
                    or (len(arr.shape) == 1 and (1,) + arr.shape == variable.data.shape)):
                arr = arr.reshape(variable.data.shape)
            if arr.shape != variable.data.shape:
                utils.print_error("{}: shape mismatch: nnef{} != caffe{}".format(label, arr.shape, variable.data.shape))
                continue
            variable.data[...] = arr
    utils.raise_if_had_error()
    net.save(caffe_model_file_name)


def convert_internal(nnef_dir_name, caffe_dir_name, with_variables=True, standard_naming=False, verbose=False):
    # type: (str, str, bool, bool, bool)->None
    nnef_dir_name = utils.without_slash(nnef_dir_name)
    nnef_file_name = nnef_dir_name + "/graph.nnef"

    caffe_dir_name = utils.without_slash(caffe_dir_name)
    utils.ensure_dir(caffe_dir_name)

    nnefgraph = nnef_parser_config.load_model(nnef_file_name)
    nnefdog = nnef_to_dog.nnefgraph_to_nnefdog(nnefgraph, with_weights=False)
    caffedog = nnefdog_to_caffedog(nnefdog)
    caffesrc = caffedog_to_prototxt(caffedog)
    utils.raise_if_had_error(listing=caffesrc)

    if standard_naming:
        caffe_prototxt_file_name = os.path.join(caffe_dir_name, caffedog.name + ".prototxt")
        caffe_model_file_name = os.path.join(caffe_dir_name, caffedog.name + ".caffemodel")
    else:
        caffe_prototxt_file_name = os.path.join(caffe_dir_name, "graph.prototxt")
        caffe_model_file_name = os.path.join(caffe_dir_name, "graph.caffemodel")

    with open(caffe_prototxt_file_name, "w") as f:
        print(caffesrc, file=f)

    if with_variables:
        convert_variables(nnef_dir_name,
                          caffedog.extra[EXTRA_VARIABLE_LABELS],
                          caffe_prototxt_file_name,
                          caffe_model_file_name)

    if verbose:
        print("Wrote {}".format(caffe_prototxt_file_name))
    if verbose and with_variables:
        print("Wrote {}".format(caffe_model_file_name))


def _has_dat_file(dir_name):
    for root, dir_names, file_names in os.walk(dir_name):
        if any(file_name.endswith('.dat') for file_name in file_names):
            return True
    return False


def convert(nnef_path, output_path=".", verbose=False):
    # type: (str, str, bool)->None

    tmp_dir_name = None

    try:
        if verbose:
            print("Converting...")

        if os.path.isfile(nnef_path):
            assert nnef_path.endswith('.tgz') or nnef_path.endswith('.nnef'), \
                "Please specify a .nnef or a .tgz file or a directory"

        if os.path.isdir(nnef_path):
            nnef_dir_path = utils.without_slash(nnef_path)
            export_variables = True
        elif nnef_path.endswith(".tgz"):
            tmp_dir_name = tempfile.mkdtemp(prefix="nnef_to_tf_")
            utils.tgz_extract(nnef_path, tmp_dir_name)
            nnef_dir_path = tmp_dir_name
            export_variables = True
        elif nnef_path.endswith(".nnef"):
            nnef_dir_path = utils.without_file_name(nnef_path)
            export_variables = False
            if verbose:
                print("No weights are exported when an NNEF file is given as input. "
                      "If you need that, try an archive or a directory.")
        else:
            assert False

        convert_internal(nnef_dir_name=nnef_dir_path,
                         caffe_dir_name=output_path,
                         with_variables=export_variables,
                         standard_naming=True,
                         verbose=verbose)
    finally:
        if tmp_dir_name:
            shutil.rmtree(tmp_dir_name)
