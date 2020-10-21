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

from caffe2.python.onnx.backend import Caffe2Backend
from ..onnx.writer import build_model, build_dtype
from ..onnx.reader import _get_value_info
from ...utils.types import as_str
import json
import os


def save_caffe2_model(folder, init_net, predict_net, value_info):
    with open(os.path.join(folder, 'init_net.pb'), 'wb') as file:
        file.write(init_net.SerializeToString())

    with open(os.path.join(folder, 'predict_net.pb'), 'wb') as file:
        file.write(predict_net.SerializeToString())

    with open(os.path.join(folder, 'value_info.json'), 'w') as file:
        json.dump(value_info, file)


def get_value_info(onnx_model):
    initializer_names = {as_str(info.name) for info in onnx_model.graph.initializer}

    value_info = {}
    for info in onnx_model.graph.input:
        name, shape, dtype = _get_value_info(info)
        if name not in initializer_names:
            value_info[name] = (build_dtype(dtype), shape)

    return value_info


class Writer:

    def __init__(self):
        pass

    def __call__(self, graph, folder):
        onnx_model = build_model(graph, ir_version=6, opset_version=9)
        if not onnx_model.graph.name:
            onnx_model.graph.name = 'Graph'

        init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)
        value_info = get_value_info(onnx_model)

        if not os.path.exists(folder):
            os.mkdir(folder)

        save_caffe2_model(folder, init_net, predict_net, value_info)
