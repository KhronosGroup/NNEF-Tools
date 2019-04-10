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

from __future__ import division

import nnef

import networkx as nx
from .nnef_version import  NNEFVersion
from .nnef_node import  Node
import os

class NNEFGraph(object):
    def __init__(self,
                 model_name=None,
                 input_list=None,
                 output_list=None,
                 pre_compile_callback=None,
                 post_compile_callback=None,
                 node_pool = None):

        if model_name is None and \
           input_list is None and \
           output_list is None and \
           pre_compile_callback is None and \
           post_compile_callback is None and \
           node_pool is None:
            return

        assert isinstance(input_list, list) or isinstance(input_list, Node), "Input list must be a single Node or list of Nodes"
        assert isinstance(output_list, list) or isinstance(output_list, Node), "Output list must be a single Node or list of Nodes"

        self.network_dir = os.getcwd()

        input_list_ = []
        output_list_ = []
        if isinstance(input_list, Node):
            input_list_.append(input_list)
        else:
            input_list_ = input_list
        if isinstance(output_list, Node):
            output_list_.append(output_list)
        else:
            output_list_ = output_list

        if(model_name == 'graph' or model_name in Node.type_nnef_primitive):
            model_name += '_'
        self.model_name = model_name
        bottom_top_nodes = []
        self.nx_graph = nx.OrderedDiGraph()

        if node_pool is None:
            '''
                Add all nodes without connecting edges.
                (by walking output list)
            '''
            def add_node_to_nx_graph(node):
                bottom_top_nodes.append(node)

                for key, param in node.parameters.items():
                    if isinstance(param, Node) and param.name not in self.nx_graph:
                        add_node_to_nx_graph(param)

            for output_node in output_list_:
                add_node_to_nx_graph(output_node)

            for node in reversed(bottom_top_nodes):
                type_node = 'node'
                if node in input_list_:
                    type_node = 'input'
                if node in output_list_:
                    type_node = 'output'

                self.nx_graph.add_node(node.name, node=node, type_node=type_node)
        else:
            for _,node in node_pool.items():
                type_node = 'node'
                if node in input_list_:
                    type_node = 'input'
                if node in output_list_:
                    type_node = 'output'

                self.nx_graph.add_node(node.name, node=node, type_node=type_node)

        '''
            pre_compile_callback:   Callback to provide converter's specific interaction with the nxgraph if required
                                    BEFORE connecting the edges
        '''
        if pre_compile_callback is not None:
            pre_compile_callback(self.nx_graph)
        '''
            "Compile" the nodes: creates edge connections, calls node specific callbacks (if defined)
        '''
        for node in self.nx_graph:
            self.nx_graph.node[node]['node'].compile(self.nx_graph)

        '''
            post_compile_callback:  Callback to provide converter's specific interaction with the nxgraph if required
                                    AFTER connecting the edges
        '''
        if post_compile_callback is not None:
            post_compile_callback(self.nx_graph)

        self.trim_nx_graph(input_list_, output_list_)

        #nx.write_yaml(self.nx_graph, 'graph.yaml')

    '''
        To be used only in an environment where tf is the framework (vs protobuf definitions)
    '''
    def run(self, output_tensors, input_data, needs_enable_eager_execution=False):
        assert isinstance(self.nx_graph, nx.OrderedDiGraph), "Invalid nx graph"

        cwd = os.getcwd()
        os.chdir(self.network_dir)

        import tensorflow as tf
        if needs_enable_eager_execution:
            tf.enable_eager_execution()

        assert tf.executing_eagerly(), "TF not executing eagerly"

        output_results = {}
        for node, data in self.nx_graph.nodes(data=True):
            node = data['node']
            if node.op == 'external':
                assert node.name in input_data, "Input data missing for external node"
                node.run(external_tensor=input_data[node.name])
            elif node.name in output_tensors:
                output_results[node.name] = node.run(nx_graph=self.nx_graph)
            elif node.op is "variable":
                output_results[node.name] = node.run(nx_graph=self.nx_graph)
            else:
                node.run(nx_graph=self.nx_graph)
        os.chdir(cwd)
        return output_results

    def trim_nx_graph(self, input_list, output_list):
        ancestors = set()
        for node in output_list:
            ancestors = ancestors.union(nx.ancestors(self.nx_graph, node.name))
        remove_nodes = list(self.nx_graph.nodes())
        for ancestor in ancestors:
            remove_nodes.remove(ancestor)
        for node in output_list:
            remove_nodes.remove(node.name)
        for node in input_list:
            assert node.name not in remove_nodes, "Input node %s is not required for the outputs" % (node.name)
        for node in remove_nodes:
            self.nx_graph.remove_node(node)

    def set_nx_graph(self, nx_graph):
        assert isinstance(nx_graph, nx.OrderedDiGraph), "nx_graph isn't a nx.OrderedDiGraph"
        self.nx_graph = nx_graph.copy()

    def get_nx_graph(self):
        return self.nx_graph

    def load_from_file(self, load_path):
        assert False, "Unimplemented"

    def save_to_file(self, save_file):
        assert isinstance(save_file, str), "Output model path is required to be of type str."
        network_dir, model_filename = os.path.split(save_file)
        assert model_filename == "graph.nnef", "NNEF Format requires to write to a file named 'graph.nnef'"

        cwd = os.getcwd()

        self.network_dir = network_dir
        if not os.path.exists(network_dir):
            os.makedirs(network_dir)
        os.chdir(network_dir)

        with open(model_filename, 'w') as f:
            f.write("\nversion %s;\n\n" % (NNEFVersion().version))

            fragments = set()
            inputs = ''
            outputs = ''
            for node, data in self.nx_graph.nodes(data=True):
                if data['node'].op in ['gru']:
                    fragments.add(data['node'].op)
                assert 'type_node' in data, "Node in graph is missing type information"
                if data['type_node'] == 'input':
                    inputs += node + ', '
                elif data['type_node'] == 'output':
                    outputs += node + ', '
            inputs = inputs[:-2]
            outputs = outputs[:-2]

            if fragments:
                f.write("extension KHR_enable_fragment_definitions, KHR_enable_operator_expressions;\n\n")

            for frag in fragments:
                if frag == 'gru':
                    gru = \
                    "fragment gru( \
                    \n\tinput: tensor<scalar>, \
                    \n\tchannels: integer, \
                    \n\tscope: string ) \
                    \n-> ( output: tensor<scalar> ) \
                    \n{ \
                    \n\tbatch = shape_of(input)[0]; \
                    \n\n\th = variable(shape = [batch,channels], label = scope + '/h'); \
                    \n\n\tm = concat([input, h], axis = 1); \
                    \n\n\tz = sigmoid(linear_layer(m, channels = channels, scope = scope + '/z')); \
                    \n\tr = sigmoid(linear_layer(m, channels = channels, scope = scope + '/r')); \
                    \n\ts = tanh(linear_layer(concat([input, r * h], axis = 1), channels = channels, scope = scope + '/s')); \
                    \n\n\toutput = update(h, z * s + (1.0 - z) * h); \
                    \n}\n\n"
                    f.write(gru)

            f.write("graph %s( %s ) -> ( %s )\n" % (self.model_name.replace('/', '_'), inputs, outputs))
            f.write("{\n")

            for node, data in self.nx_graph.nodes(data=True):
                if 'node' in data:
                    if data['node'].op == 'output_val':
                        continue
                    #print((data['node'].nnef_node_definition()))
                    f.write("\t")
                    f.write(data['node'].nnef_node_definition())
                    f.write(";\n")

                    assert data['node'] is not None, "Node doesn't have NNEF node!"

                    nnef_node = data['node']
                    if data['node'].op == 'variable':
                        #print("=> Node %s is saving tensor to disk(%s)" % (
                        #    nnef_node.name, nnef_node.parameters['label']))
                        loc = nnef_node.parameters['label'].rfind('/')
                        if loc != -1:
                            if not os.path.isdir(nnef_node.parameters['label'][:loc]):
                                os.makedirs(nnef_node.parameters['label'][:loc])
                        dat_file = open(nnef_node.parameters['label'] + '.dat', 'wb')
                        nnef.write_tensor(dat_file, nnef_node.tensor)
                        dat_file.close()
                else:
                    print("===> %s doesn't have node!?" % (node))
            f.write("}")

        os.chdir(cwd)
