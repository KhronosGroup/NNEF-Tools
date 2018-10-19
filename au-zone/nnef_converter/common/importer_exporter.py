# Copyright (c) 2018 The Khronos Group Inc.
# Copyright (c) 2018 Au-Zone Technologies Inc.
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

import os
from .nnef_node import Node


class ImporterExporter(object):
    def __init__(self):
        pass

    def run(self):
        pass

    def openProtobuf(self, model_filename, graph_def):
        with open(model_filename, "rb") as f:
            graph_def.ParseFromString(f.read())
        return graph_def

    def chdir_to_modeldir(self, output_model_path, is_nnef_model=False):
        assert isinstance(output_model_path, str), "Output model path is required to be of type str."

        network_dir, model_filename = os.path.split(output_model_path)

        # For NNEF model, we're doing extra check on the model's folder.
        if is_nnef_model:
            network_dir, folder_name = os.path.split(network_dir)
            if folder_name in Node.type_nnef_primitive:
                print('renaming %s to %s_ to avoid conflicts' %
                      (folder_name, folder_name))
                folder_name = folder_name + "_"
        else:
            _, folder_name = os.path.split(network_dir)

        if not os.path.exists(network_dir):
            os.makedirs(network_dir)
        os.chdir(network_dir)

        return model_filename, folder_name
