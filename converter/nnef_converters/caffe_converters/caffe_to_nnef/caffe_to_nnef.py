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

from . import caffe_to_dog
from . import converters
from . import transformations
from ..common import EXTRA_WEIGHTS, EXTRA_ACTIVATIONS, CaffeGraph
from ...common import dog_to_nnef
from ...common import utils
from ...common.types import *


def _write_activations(activations_dir, caffedog):
    # type: (str, CaffeGraph)->None

    for dn in caffedog.dn_by_name.values():
        if dn.extra[EXTRA_ACTIVATIONS] is not None:
            path = "{}/{}.dat".format(activations_dir, dn.name)
            utils.write_nnef_tensor(path, dn.extra[EXTRA_ACTIVATIONS])


def prototxt_to_nnef(prototxt_file_name,
                     nnef_file_name,
                     caffemodel_file_name=None,
                     with_activations=False,
                     with_variables=True):
    # type: (str, str, Optional[str], bool, bool)->None

    nnefdir = utils.without_file_name(nnef_file_name)
    utils.ensure_dir(nnefdir)

    caffedog = caffe_to_dog.prototxt_to_caffedog(prototxt_file_name,
                                                 caffemodel_file_name=caffemodel_file_name,
                                                 with_activations=with_activations,
                                                 with_variables=with_variables)

    transformations.fix_names(caffedog)
    transformations.resolve_inplace(caffedog)
    if with_activations:
        _write_activations(nnefdir + "/activations", caffedog)
    transformations.merge_batch_norm_and_scale(caffedog)
    transformations.batch_norm_to_scale(caffedog)
    transformations.remove_passthroughs(caffedog)

    nnefdog = converters.Converter(caffedog).convert()
    nnefdog.remove_unreachables()
    # noinspection PyTypeChecker
    transformations.reorder(nnefdog)

    nnefsrc = dog_to_nnef.nnefdog_to_source(nnefdog)
    utils.raise_if_had_error(listing=nnefsrc)

    with open(nnef_file_name, "w") as f:
        print(nnefsrc, file=f)

    if with_variables:  # TODO don't even calculate with variables when not needed
        for op in nnefdog.ops:
            if op.name == "variable":
                path = "{}/{}.dat".format(nnefdir, op.args["label"])
                utils.write_nnef_tensor(path, op.extra[EXTRA_WEIGHTS])


def export_activations(prototxt_file_name, target_dir, caffemodel_file_name=None):
    # type: (str, str, Optional[str])->None

    caffedog = caffe_to_dog.prototxt_to_caffedog(prototxt_file_name,
                                                 caffemodel_file_name=caffemodel_file_name,
                                                 with_activations=True)

    transformations.fix_names(caffedog)
    transformations.resolve_inplace(caffedog)
    _write_activations(target_dir, caffedog)


def convert(prototxt_path, caffemodel_path=None, output_path=".", verbose=False):
    # type: (str, str, Optional[str], bool)->None

    network_name = os.path.basename(prototxt_path).split(".prototxt")[0]  # type: str

    tmp_dir_name = None
    try:
        tmp_dir_name = tempfile.mkdtemp(prefix="tf_to_nnef_")

        if verbose:
            print("Converting...")

        prototxt_to_nnef(prototxt_file_name=prototxt_path,
                         nnef_file_name=os.path.join(tmp_dir_name, "graph.nnef"),
                         caffemodel_file_name=caffemodel_path,
                         with_variables=caffemodel_path is not None)

        if verbose:
            print("Compressing...")

        utils.ensure_dir(output_path)
        tgz_file_name = os.path.join(output_path, network_name + ".nnef.tgz")
        utils.tgz_compress(tmp_dir_name, tgz_file_name)

        if verbose:
            print("Wrote {}".format(tgz_file_name))

    finally:
        if tmp_dir_name:
            shutil.rmtree(tmp_dir_name)

