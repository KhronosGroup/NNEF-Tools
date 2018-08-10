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
import subprocess
from tempfile import NamedTemporaryFile

import numpy as np

os.environ['GLOG_minloglevel'] = '3'  # suppress Caffe verbose/warning logs

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf.text_format as protobuf_text_format

from . import converters
from ...common import CaffeGraph, CaffeOp, CaffeDN
from ...common import EXTRA_ACTIVATIONS
from ....common import dog
from ....common.types import *
from ....common import utils


def _get_random_inputs(net):
    # type: (caffe.Net) -> dict[str, np.ndarray]

    def get_seed(name):
        import hashlib
        return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)

    d = {}
    for input_ in net.inputs:
        np.random.seed(get_seed(input_))
        d[input_] = np.random.random(list(net.blobs[input_].shape))

    return d


def prototxt_to_caffedog(prototxt_file_name, caffemodel_file_name=None, with_activations=False, with_variables=True):
    # type: (str, Optional[str], bool)->CaffeGraph

    path_without_ext = prototxt_file_name.split(".prototxt")[0]

    name = path_without_ext.split('/')[-1].split('\\')[-1]
    if not name:
        name = "Graph"

    if caffemodel_file_name is not None:
        assert os.path.isfile(prototxt_file_name)
        assert os.path.isfile(caffemodel_file_name)

        n_instance = caffe.Net(prototxt_file_name, caffemodel_file_name, caffe.TEST)
    else:
        n_instance = caffe.Net(prototxt_file_name, caffe.TEST)
        if with_variables:
            n_instance.save(path_without_ext + ".caffemodel")

    if with_activations:
        n_instance.forward(**_get_random_inputs(n_instance))

    upgrade_bin = "/usr/bin/upgrade_net_proto_text"
    if not os.path.exists(upgrade_bin):
        upgrade_bin = os.path.join(os.environ['CAFFE_BIN_FOLDER'], "upgrade_net_proto_text.bin")

    with NamedTemporaryFile(mode='r', suffix='.prototxt') as tmp_file:
        subprocess.check_call([upgrade_bin, prototxt_file_name, tmp_file.name])
        text = tmp_file.read()

    netparam = caffe_pb2.NetParameter()
    protobuf_text_format.Merge(text, netparam)
    layers = netparam.ListFields()[-1][1]

    caffedn_by_name = {}  # type: Dict[str, CaffeDN]

    def get_caffedn_by_name(name):
        # type: (str)->CaffeDN
        if name in caffedn_by_name:
            return caffedn_by_name[name]
        else:
            dn = CaffeDN(name)
            caffedn_by_name[name] = dn
            dn.shape = list(n_instance.blobs[name].data.shape)
            return dn

    converter = converters.Converter(n_instance)
    caffeops = []
    for layer in layers:
        op = CaffeOp(layer.type)
        for i, arg_node_name in enumerate(layer.bottom):
            op.add_arg(dog.gen_arg_name(i), get_caffedn_by_name(arg_node_name))
        for i, result_node_name in enumerate(layer.top):
            op.add_result(dog.gen_result_name(i), get_caffedn_by_name(result_node_name), overwrite_producer=False)

        converter_by_layer_type = converters.DefaultConverters
        if layer.type in converter_by_layer_type:
            converter_by_layer_type[layer.type](layer, op, converter)
        else:
            utils.print_error("caffe_to_dog: No converter for: {}".format(layer.type))

        op.add_arg("name", layer.name)
        caffeops.append(op)

    if with_activations:
        for caffedn in caffedn_by_name.values():
            caffedn.extra[EXTRA_ACTIVATIONS] = n_instance.blobs[caffedn.name].data.copy()

    return CaffeGraph(graph_name=name,
                      ops=caffeops,
                      dn_by_name=caffedn_by_name,
                      input_dn_names=converter.input_names,
                      output_dn_names=[dn.name for dn in caffedn_by_name.values() if not dn.consumers])
