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

import numpy as np
import networkx as nx
import os


class Converter(object):
    """
    Converter object holds the NX graph, importer and exporter.
    """
    def __init__(self, importer, exporter):
        # self.nxgraph = nx.Graph(imported_from='tensorflow', input-model='frozen_001.pb', date='now')
        # self.nxgraph = nx.Graph()
        self.importer = importer
        self.exporter = exporter

    def run(self):
        # print("converter runs")
        cwd = os.getcwd()

        self.nnef_graph = self.importer.run()  # model_input

        os.chdir(cwd)
        # TODO: put this optional numpy as well?
        # self.save_to_graphml(dest_path)
        self.exporter.run(self.nnef_graph)
        os.chdir(cwd)

    @property
    def save_to_graphml(self, filename):
#        with open(filename, 'wb') as of:
#            nx.save()
#        print ("IR network structure is saved as [{}].".format(filename))
        return filename

#    @staticmethod
#    def channel_first_conv_kernel_to_IR(tensor):
#        dim = tensor.ndim
#        tensor = np.transpose(tensor, list(range(2, dim)) + [1, 0])
#        return tensor

