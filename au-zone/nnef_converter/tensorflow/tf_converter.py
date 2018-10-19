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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import textwrap
import collections

import networkx as nx
import numpy as np

from ..common.importer_exporter import ImporterExporter
from ..common.nnef_data import  * #NNEFTensor, TensorDataFile
from ..common.nnef_converter import *
from ..common.nnef_graph import *
from ..common import nnef_node as node

from .core.framework import graph_pb2
from .core.framework import attr_value_pb2
from .core.framework.node_def_pb2 import NodeDef

class TensorflowLogger(object):
    single_line_sep = "---------------------------------------------------------------------------------------------------------------------------------"
    double_line_sep = "===================================================================================================="

    def __init__(self):
        super(TensorflowLogger, self).__init__()
        self.logger = logging.getLogger('nnef_convert')

    def log_tf_node_info(self, tfnode, inputs, attrs):
        title = "Importing Tensorflow Node:         "
        preferredWidth = 250
        wrapper = textwrap.TextWrapper(initial_indent=title,
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(self.single_line_sep)
        self.logger.info(wrapper.fill("Name \t%s"%(tfnode.name)))
        wrapper = textwrap.TextWrapper(initial_indent=' ' * len(title),
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(wrapper.fill("Op   \t%s"%(tfnode.op)))

        unused_input_found = False
        used_input_found = False
        unused_attribute_found = False
        used_attribute_found = False

        if inputs is not None and tfnode.input is not None:
            for cnt, input_item in enumerate(tfnode.input):
                if cnt in inputs.values():
                    if not used_input_found:
                        used_input_found = True
                        self.logger.info(wrapper.fill(""))
                        self.logger.info(wrapper.fill("Used Inputs:"))
                    self.logger.info(wrapper.fill("\t%s"%(input_item)))

            for cnt, input_item in enumerate(tfnode.input):
                if cnt not in inputs.values():
                    if not unused_input_found:
                        unused_input_found = True
                        self.logger.info(wrapper.fill(""))
                        self.logger.info(wrapper.fill("Unused Inputs:"))
                    self.logger.info(wrapper.fill("\t%s"%(input_item)))

        if attrs is not None and tfnode.attr is not None:
            for key, value in tfnode.attr.items():
                if key in attrs:
                    if not used_attribute_found:
                        used_attribute_found = True
                        self.logger.info(wrapper.fill(""))
                        self.logger.info(wrapper.fill("Used Attributes:"))
                    if self.log_level == 'debug':
                        self.logger.debug(wrapper.fill("\t'%s': %s" % (key, value)))
                    else:
                        self.logger.info(wrapper.fill("\t'%s'" % key))

            for key, value in tfnode.attr.items():
                if key not in attrs:
                    if not unused_attribute_found:
                        unused_attribute_found = True
                        self.logger.info(wrapper.fill(""))
                        self.logger.info(wrapper.fill("Unused Attributes:"))
                    if self.log_level == 'debug':
                        self.logger.debug(wrapper.fill("\t'%s': %s" % (key, value)))
                    else:
                        self.logger.info(wrapper.fill("\t'%s'" % key))

    def print_msg_nodeop_nodename(self, title, op, name, level="info"):
        preferred_width = 250
        wrapper = textwrap.TextWrapper(initial_indent=title,
                                       width=preferred_width,
                                       subsequent_indent=' ' * len(title))

        if level == "debug":
            log_fct = self.logger.debug
        elif level == "warning":
            log_fct = self.logger.warning
        elif level == "error":
            log_fct = self.logger.error
        elif level == "critical":
            log_fct = self.logger.critical
        else:
            log_fct = self.logger.info

        log_fct(self.single_line_sep)
        log_fct(wrapper.fill("Name \t%s"%(name)))
        wrapper = textwrap.TextWrapper(initial_indent=' ' * len(title),
                                       width=preferred_width,
                                       subsequent_indent=' ' * len(title))
        log_fct(wrapper.fill("Op   \t%s"%(op)))

    def log_removing_node(self, nnef_node):
        title = "Removing Node From Pool:           "
        preferredWidth = 250
        wrapper = textwrap.TextWrapper(initial_indent=title,
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(self.single_line_sep)
        self.logger.info(wrapper.fill("Name \t%s"%(nnef_node.name)))
        wrapper = textwrap.TextWrapper(initial_indent=' ' * len(title),
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(wrapper.fill("Op   \t%s"%(nnef_node.op)))

    def log_skipping_nodes(self, tfnode):
        self.print_msg_nodeop_nodename("Skipping Op:                       ", tfnode.op, tfnode.name)
        title = "Skipping Tensorflow Node:          "
        preferredWidth = 250
        wrapper = textwrap.TextWrapper(initial_indent=title,
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(self.single_line_sep)
        self.logger.info(wrapper.fill("Name \t%s"%(tfnode.name)))
        wrapper = textwrap.TextWrapper(initial_indent=' ' * len(title),
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(wrapper.fill("Op   \t%s"%(tfnode.op)))

    def log_unsupported_nodes(self, tfnode):
        title = "Unsupported Tensorflow Node:       "
        preferredWidth = 250
        wrapper = textwrap.TextWrapper(initial_indent=title,
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(self.single_line_sep)
        self.logger.info(wrapper.fill("Name \t%s"%(tfnode.name)))
        wrapper = textwrap.TextWrapper(initial_indent=' ' * len(title),
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(wrapper.fill("Op   \t%s"%(tfnode.op)))

    def log_total_conversions(self):
        title = "Finished Converting Model:         "
        preferredWidth = 250
        wrapper = textwrap.TextWrapper(initial_indent=title,
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(self.double_line_sep)
        self.logger.info(wrapper.fill("Total Tensorflow Nodes \t%s"%(str(self.total))))
        wrapper = textwrap.TextWrapper(initial_indent=' ' * len(title),
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(wrapper.fill("Successfully Converted \t%s"%(str(self.successful))))
        self.logger.info(wrapper.fill("Nodes in Graph         \t%s"%(str(len(self.node_pool.keys())-self.removed_nodes))))

def convert_format(convert_list, in_format='nhwc', out_format='nchw'):
    in_format = in_format.lower()
    out_format = out_format.lower()

    in_n_loc = in_format.find('n')
    out_n_loc = out_format.find('n')
    in_c_loc = in_format.find('c')
    out_c_loc = out_format.find('c')

    n = convert_list[in_n_loc]
    c = convert_list[in_c_loc]

    sizes = []
    for i in range(len(convert_list)):
        if(i != in_n_loc and i != in_c_loc):
            sizes.append(convert_list[i])

    out_list = [None]*len(convert_list)
    out_list[out_n_loc] = n
    out_list[out_c_loc] = c

    j = 0
    for i in range(len(out_list)):
        if(i != out_n_loc and i != out_c_loc):
            out_list[i] = sizes[j]
            j += 1

    return out_list

class TensorflowImporter(TensorflowLogger, ImporterExporter):
    def __init__(self, input_model, input_nodes, output_nodes, log_level='info'):
        super(TensorflowImporter, self).__init__()

        self.node_pool = collections.OrderedDict()
        self.input_model = input_model
        self.log_level = log_level
        self.input_nodes = {}
        self.output_nodes = {}
        self.name_convs = {}
        self.start_format = None
        self.start_length = 0
        self.successful = 0
        self.total = 0
        self.removed_nodes = 0
        self.graph = super(TensorflowImporter, self).openProtobuf(self.input_model, graph_pb2.GraphDef())

        i = 1
        if input_nodes is not None:
            for in_node_str in input_nodes.split(','):
                if len(input_nodes.split(',')) == 1:
                    self.input_nodes[in_node_str] = "input"
                else:
                    self.input_nodes[in_node_str] = "input" + str(i)
                i += 1
        else:
            input_nodes = []
            for tfnode in self.graph.node:
                if hasattr(tfnode, 'op'):
                    if tfnode.op == 'Placeholder':
                        input_nodes.append(tfnode.name)
            if len(input_nodes) == 1:
                self.input_nodes[input_nodes[0]] = "input"
            else:
                for i in range(len(input_nodes)):
                    self.input_nodes[input_nodes[i]] = "input" + str(i+1)

        i = 1
        if output_nodes is not None:
            for out_node_str in output_nodes.split(','):
                if len(output_nodes.split(',')) == 1:
                    self.output_nodes[out_node_str] = "output"
                else:
                    self.output_nodes[out_node_str] = "output" + str(i)
                i += 1
        else:
            #Unable to use single loop for case of out of order nodes (MobileNetV2 from Model Zoo)
            output_nodes = []
            for tfnode in self.graph.node:
                if hasattr(tfnode, 'op'):
                    output_nodes.append(tfnode.name)
            for tfnode in self.graph.node:
                if hasattr(tfnode, 'op'):
                    for input_val in tfnode.input:
                        if input_val in output_nodes:
                            output_nodes.remove(input_val)

            if len(output_nodes) == 1:
                self.output_nodes[output_nodes[0]] = "output"
            else:
                for i in range(len(output_nodes)):
                    self.output_nodes[output_nodes[i]] = "output" + str(i+1)

    def run(self):
        self.nxgraph = nx.OrderedDiGraph()
        self.create_nodes()
        self.log_total_conversions()

        input_nodes = self.get_input_nodes()
        output_nodes = self.get_output_nodes()

        return NNEFGraph(os.path.basename(self.input_model).split('.')[0],
                         input_nodes,
                         output_nodes,
                         pre_compile_callback=self.pre_compile_callback,
                         post_compile_callback=self.post_compile_callback,
                         node_pool=self.node_pool)

    def get_input_nodes(self):
        input_node_list = []
        if 'input' in self.node_pool.keys():
            input_node_list.append(self.get_node_from_pool_by_name('input', get_orig=True))
        else:
            i = 1
            while 'input' + str(i) in self.node_pool.keys():
                input_node_list.append(self.get_node_from_pool_by_name('input'+str(i), get_orig=True))
                i += 1

        return input_node_list

    def get_output_nodes(self):
        output_node_list = []
        if 'output' in self.node_pool.keys():
            output_node_list.append(self.get_node_from_pool_by_name('output', get_orig=True))
        else:
            i = 1
            while 'output' + str(i) in self.node_pool.keys():
                output_node_list.append(self.get_node_from_pool_by_name('output' + str(i), get_orig=True))
                i += 1

        return output_node_list

    def create_nodes(self):
        for tfnode in self.graph.node:
            self.total += 1
            if self.start_format == None and tfnode.attr['data_format'].s != b'':
                self.start_format = tfnode.attr['data_format'].s.decode('ascii')

            if hasattr(tfnode, 'op'):
                node_op = tfnode.op
                if hasattr(self, "import_" + node_op):
                    func = getattr(self, "import_" + node_op)
                    nnef_node, tf_inputs, tf_attrs = func(tfnode)
                    self.successful += 1
                    if nnef_node is not None:
                        self.add_node_to_pool(nnef_node, tfnode, tf_inputs, tf_attrs)
                else:
                    self.import_UNKNOWN(tfnode)
            else:
                self.logger.error("Node doesn't have op attr.: %s"%(tfnode.name))

    def add_node_to_pool(self, nnef_node, tfnode, tf_inputs, tf_attrs):
        if nnef_node.name not in self.node_pool.keys():
            self.log_tf_node_info(tfnode, tf_inputs, tf_attrs)
            self.node_pool[nnef_node.name] = nnef_node

    def remove_node_from_pool(self, nnef_node):
        self.log_removing_node(nnef_node)
        self.node_pool.pop(nnef_node.name, None)

    def get_node_from_pool(self, tfnode, idx):
        node_name = self.gen_node_name(self.get_tfnode_input(tfnode, idx))

        #Handles cases where nodes are out of order within Protocol Buffer
        try:
            nnef_node = self.get_node_from_pool_by_name(node_name)
        except:
            for tfnode in self.graph.node:
                if self.gen_node_name(tfnode.name) == node_name:
                    if hasattr(tfnode, 'op'):
                        node_op = tfnode.op
                        if hasattr(self, "import_" + node_op):
                            func = getattr(self, "import_" + node_op)
                            nnef_node, tf_inputs, tf_attrs = func(tfnode)
                            if nnef_node is not None:
                                self.add_node_to_pool(nnef_node, tfnode, tf_inputs, tf_attrs)
                    break
            nnef_node = self.get_node_from_pool_by_name(node_name)

        if nnef_node.op == 'idn':
            nnef_node = self.get_node_from_pool_by_name(nnef_node.name).parameters['x']
        return nnef_node

    def get_node_from_pool_by_name(self, node_name, get_orig=False):
        if node_name in self.name_convs and not get_orig:
            node_name = self.name_convs[node_name]

        assert node_name in self.node_pool.keys(), "Node pool doesn't contain required node: %s" % node_name
        return self.node_pool[node_name]

    def shape_nx_graph(self, nx_graph):
        remove_nodes = []
        if self.start_format is None:
            for nnef_node_name in nx_graph:
                if nx_graph.node[nnef_node_name]['node'].op == 'pad':
                    remove_nodes.append(nnef_node_name)
            for nnef_node_name in remove_nodes:
                nx_graph.remove_node(nnef_node_name)
            return
        else:
            nnef_format = 'NC...'

        if len(self.start_format) != self.start_length:
            if(self.start_format == 'NHWC' and self.start_length == 3):
                self.start_format = 'NHC'
            elif(self.start_format == 'NCHW' and self.start_length == 3):
                self.start_format = 'NCH'
            else:
                raise ValueError("Issue with compatibility of start_format : " + self.start_format +
                                 " and start_length : " + str(self.start_length))

        current_format = self.start_format

        indexes = list(range(len(self.start_format)))
        mapping = {}
        for i in range(len(self.start_format)):
            if self.start_format[i] in nnef_format:
                index = nnef_format.find(self.start_format[i])
                indexes.pop(indexes.index(index))
                mapping[i] = index

        for i in range(len(self.start_format)):
            if i not in mapping:
                mapping[i] = indexes[0]
                indexes.pop(0)

        for nnef_node_name in nx_graph:
            if nx_graph.node[nnef_node_name]['node'].op == 'pad':
                remove_nodes.append(nnef_node_name)
                continue
            nnef_node = nx_graph.node[nnef_node_name]['node']
            if nnef_node.op not in ['variable', 'constant', 'reshape']:
                if '_data_format' in nnef_node.parameters and nnef_node.parameters['_data_format'] != '':
                    self.current_format = nnef_node.parameters['_data_format']
                if 'shape' in nnef_node.parameters and current_format is not None:
                    nnef_node.parameters['shape'] = convert_format(nnef_node.parameters['shape'], current_format, nnef_format)
                if nnef_node.op == 'transpose' and current_format is not None:
                    new_format = ''
                    for i in nnef_node.parameters['axes']:
                        new_format += current_format[i]
                    new_perms = list(range(len(nnef_node.parameters['axes'])))
                    nnef_node.parameters['axes'] = convert_format(new_perms, nnef_format, new_format)
                    current_format = new_format
                if 'axes' in nnef_node.parameters and current_format is not None and nnef_node.op not in ['softmax', 'transpose']:
                    new_axes = []
                    for i in nnef_node.parameters['axes']:
                        new_axes.append(mapping[i])
                    nnef_node.parameters['axes'] = new_axes
                if 'axis' in nnef_node.parameters and current_format is not None:
                    new_axis = mapping[nnef_node.parameters['axis']]
                    nnef_node.parameters['axis'] = new_axis
                if nnef_node.output_shape is not None and \
                len(nnef_node.output_shape) == len(self.start_format) and \
                current_format is not None:
                    nnef_node.output_shape = convert_format(nnef_node.output_shape, current_format, nnef_format)
            elif nnef_node.op == 'reshape':
                if not nnef_node.parameters['_maintain_format']:
                    current_format = None

        nx_graph.remove_nodes_from(remove_nodes)

    # Helper function to convert node names to lower case and remove illegal characters ('/', ...)
    def gen_node_name(self, node_name):
        try:
            if isinstance(node_name, unicode):
                node_name = node_name.encode('ascii')
        except NameError:
            node_name = node_name
        assert isinstance(node_name, str), "self.gen_node_name: node_name is not a str"

        if node_name in self.input_nodes:
            node_name = self.input_nodes[node_name]
            return node_name
        if node_name in self.output_nodes:
            node_name = self.output_nodes[node_name]
            return node_name

        name = node_name.lower()
        if name[-5:] == '/read':
            name = name[:-5]
        name = name.replace('/', '_')
        name = name.replace(':', '_')
        return name

    def get_tfnode_input(self, tfnode, idx):
        assert idx < len(tfnode.input), "Bad index for accessing Tensorflow's op input %s"%idx
        return tfnode.input[idx]

    '''
        Called by the NNEF graph when all nodes are there, with no edge yet.
    '''
    def pre_compile_callback(self, nx_graph):
        # Cleaning up "idn" nodes
        remove_nodes = []
        for nnef_node_name in nx_graph:
            if nx_graph.node[nnef_node_name]['node'].op is 'idn':
                remove_nodes.append(nx_graph.node[nnef_node_name]['node'].name)
        nx_graph.remove_nodes_from(remove_nodes)

        return
    '''
        Called by the NNEF graph after edges are connected.
    '''
    def post_compile_callback(self, nx_graph):
        self.shape_nx_graph(nx_graph)

    @staticmethod
    def nnef_padding(padding, rank):
        return [] if padding.upper() == b'SAME' else [(0, 0)] * rank

    @staticmethod
    def tensor_shape_to_list(shapes):
        return [dim.size for dim in shapes.dim]

    def new_get_attr(self, tfnode, attribute, *args):
        if attribute == 'ksize' and tfnode.attr['ksize'].list.i is not None:
            ksize = tfnode.attr[attribute].list.i
            ksize = [int(v) for v in ksize]
            ksize = convert_format(ksize, args[1], 'NC...')
            return ksize

        elif attribute == 'padding' and tfnode.attr[attribute].s is not None:
            value = tfnode.attr[attribute].s
            rank = args[0] if args[0] is not None else 4
            padding = self.nnef_padding(value, rank)
            return padding

        elif attribute == 'strides' and tfnode.attr['strides'].list.i is not None:
            strides = tfnode.attr[attribute].list.i
            strides = [int(v) for v in strides]
            strides = convert_format(strides, args[1], 'NC...')
            return strides

        elif attribute == 'dilations' and tfnode.attr['dilations'].list.i is not None:
            dilations = tfnode.attr['dilations'].list.i
            dilations = [int(v) for v in dilations]
            if dilations:
                dilations = convert_format(dilations, args[1], 'NC...')
                return dilations

        elif attribute == 'alpha' and tfnode.attr['alpha'].f is not None:
            value = tfnode.attr['alpha'].f
            return value

        elif attribute == 'beta' and tfnode.attr['beta'].f is not None:
            value = tfnode.attr['beta'].f
            return value

        elif attribute == 'bias' and tfnode.attr['bias'].f is not None:
            value = tfnode.attr['bias'].f
            return value

        elif attribute == 'transpose_a':
            value = self._get_attr(tfnode, attribute)
            return value

        elif attribute == 'transpose_b':
            value = self._get_attr(tfnode, attribute)
            return value

        elif attribute == 'epsilon':
            value = tfnode.attr['epsilon'].f
            return value

        else:
            self._get_attr(tfnode, attribute)

    def _get_attr(self, tfnode, name, default_value=None):
        if name in tfnode.attr:
            attr = tfnode.attr[name]
            field = attr.WhichOneof('value')
            val = getattr(attr, field) if field else default_value
            if isinstance(val, attr_value_pb2.AttrValue.ListValue):
                return list(val.ListFields()[0][1])
            else:
                return val.decode('utf-8') if isinstance(val, bytes) else val
        else:
            return default_value

    def get_numpy_from_tf_tensor(self, tf_tensor):
        if tf_tensor.dtype == 3:
            nnef_dtype = np.int32
        elif tf_tensor.dtype == 1:
            nnef_dtype = np.float32
        tf_shape = self.tensor_shape_to_list(tf_tensor.tensor_shape)

        if not tf_shape:
            assert False, "tf_shape is None!"
            tf_shape = [1]

        if (len(tf_shape) == 1 and tf_shape[0] != 1):
            tf_shape = [1, tf_shape[0]]

        return tf_tensor.tensor_content, nnef_dtype, tf_shape

    def define_elementwise_binary_output_shape(self, nnef_node_x, nnef_node_y):
        y_size = x_size = 1
        for i in nnef_node_x.output_shape:
            x_size *= i
        for i in nnef_node_y.output_shape:
            y_size *= i
        if x_size >= y_size:
            output_shape = nnef_node_x.output_shape[:]
        else:
            output_shape = nnef_node_y.output_shape[:]
        return output_shape

    @staticmethod
    def _get_scopes(layer_name):
        return layer_name.split('/')

    def import_UNKNOWN(self, tfnode):
        self.log_unsupported_nodes(tfnode)
        return

    def import_NoOp(self, tfnode):
        return None, None, None, None

    def import_Const(self, tfnode):
        if tfnode.attr['value'].tensor.tensor_content == b'':
            shape = self.tensor_shape_to_list(tfnode.attr['value'].tensor.tensor_shape)
            if len(shape) == 1:
                if shape[0] == 0:
                    nnef_node = node.Constant(value=[],
                                              shape=shape,
                                              _uid=self.gen_node_name(tfnode.name),
                                              _np_dtype=None,
                                              _output_shape=shape)

                    return nnef_node, {}, {}

                else:
                    shape = [1] + shape

            value = None
            if tfnode.attr['value'].tensor.dtype == 3:
                value = [float(tfnode.attr['value'].tensor.int_val[0])]
                np_dtype = np.int32
            elif tfnode.attr['value'].tensor.dtype == 1:
                value = [float(tfnode.attr['value'].tensor.float_val[0])]
                np_dtype = np.float32
            elif tfnode.attr['value'].tensor.dtype == 10:
                raise ValueError("Type logical is not currently supported within NNEF as a constant or variable")
            else:
                raise ValueError("Type " + str(tfnode.attr['value'].tensor.dtype) + " is not currently supported")

            if shape == []:
                shape = [1, 1]

            nnef_node = node.Constant(value=value,
                                      shape=shape,
                                      _uid=self.gen_node_name(tfnode.name),
                                      _np_dtype=np_dtype,
                                      _output_shape=shape)

            inputs = {}
            attrs = {'value': value, 'shape':shape}
        else:
            np_tensor, np_dtype, shape = self.get_numpy_from_tf_tensor(tfnode.attr['value'].tensor)
            
            try:
                if isinstance(tfnode.name, unicode):
                    label = tfnode.name.encode('ascii')
                else: 
                    label = tfnode.name
            except NameError:
                label = tfnode.name

            nnef_node = node.Variable(label=label,
                                      shape=shape,
                                      _np_dtype=np_dtype,
                                      _np_tensor=np_tensor,
                                      _output_shape=shape,
                                      _uid=self.gen_node_name(tfnode.name))

            inputs = {}
            attrs = {'label': tfnode.name, 'shape':shape}
        return nnef_node, inputs, attrs

    def import_Abs(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Abs(x=nnef_node_x,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Add(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.Add(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_AvgPool(self, tfnode):
        tf_inputs = {'input':0}
        data_format = tfnode.attr['data_format'].s.decode('ascii')
        padding = self.new_get_attr(tfnode, 'padding', 4, data_format)
        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])

        if nnef_node_input.op == 'pad':
            pad_array = nnef_node_input.parameters['padding']
            nnef_node_input = nnef_node_input.parameters['input']

            padding = []
            if data_format == 'NHWC':
                padding.append((pad_array[0][0], pad_array[0][1]))
                padding.append((pad_array[-1][0], pad_array[-1][1]))
                for i in range(1, len(pad_array)-1):
                    padding.append((pad_array[i][0], pad_array[i][1]))
            else:
                padding
                for i in range(len(pad_array)):
                    padding.append((pad_array[i][0], pad_array[i][1]))

        sizes = self.new_get_attr(tfnode, 'ksize', None, data_format)
        strides = self.new_get_attr(tfnode, 'strides', None, data_format)
        dilations = [1, 1, 1, 1]

        #Modify tensor data for filter
        in_shape = nnef_node_input.output_shape[:]
        in_shape = convert_format(in_shape, data_format, 'NCHW')
        if nnef_node_input.op == 'reshape':
            nnef_node_input.parameters['shape'] = in_shape

        #Calculate output shape
        output_shape = len(in_shape) * [0]
        for i in range(len(in_shape)):
            if padding == []:
                output_shape[i] = math.ceil(in_shape[i] / strides[i])
            else:
                fd = (sizes[i] - 1) * dilations[i] + 1
                padding_add = padding[i][0] + padding[i][1]
                output_shape[i] = math.floor((in_shape[i] + padding_add - fd) / strides[i]) + 1

        output_shape = convert_format(output_shape, 'NCHW', data_format)
        output_shape = [int(v) for v in output_shape]
        
        nnef_node = node.AvgPool(input=nnef_node_input,
                                 size=sizes,
                                 padding=padding,
                                 stride=strides,
                                 dilation=dilations,
                                 _uid=self.gen_node_name(tfnode.name),
                                 _output_shape=output_shape,
                                 _data_format=data_format)

        attrs = {'padding':padding, 'ksize':sizes, 'strides':strides}
        return nnef_node, tf_inputs, attrs

    def import_BiasAdd(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.Add(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Ceil(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Ceil(x=nnef_node_x,
                              _uid=self.gen_node_name(tfnode.name),
                              _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_ConcatV2(self, tfnode):
        tf_inputs = {}
        attrs = {}
        nnef_nodes = []
        for i in range(len(tfnode.input)-1):
            nnef_nodes.append(self.get_node_from_pool(tfnode, i))
            tf_inputs['value_' + str(i)] = i

        nnef_node_axis = self.get_node_from_pool(tfnode, len(tfnode.input)-1)
        tf_inputs['value_' + str(len(tfnode.input)-1)] = len(tfnode.input)-1
        axis = int(nnef_node_axis.parameters['value'][0])

        self.remove_node_from_pool(nnef_node_axis)

        output_shape = nnef_nodes[0].output_shape[:]
        for nnef_node in nnef_nodes[1:]:
            output_shape[axis] += nnef_node.output_shape[axis]

        nnef_node = node.Concat(values=nnef_nodes,
                                axis=axis,
                                _uid=self.gen_node_name(tfnode.name),
                                _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Conv2D(self, tfnode):
        tf_inputs = {'input': 0, 'filter':1}
        data_format = tfnode.attr['data_format'].s.decode('ascii')
        padding = self.new_get_attr(tfnode, 'padding', 2, data_format)

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_filter = self.get_node_from_pool(tfnode, tf_inputs['filter'])

        if nnef_node_input.op == 'pad':
            pad_array = nnef_node_input.parameters['padding']
            nnef_node_input = nnef_node_input.parameters['input']
            padding = []
            if data_format == 'NHWC':
                for i in range(1, len(pad_array)-1):
                    padding.append((pad_array[i][0], pad_array[i][1]))
            else:
                for i in range(2, len(pad_array)):
                    padding.append((pad_array[i][0], pad_array[i][1]))

        strides = self.new_get_attr(tfnode, 'strides', None, data_format)
        strides = strides[2:]

        dilations = self.new_get_attr(tfnode, 'dilations', None, data_format)
        dilations = dilations[2:]

        #Modify tensor data for filter
        in_shape = nnef_node_input.output_shape[:]
        in_shape = convert_format(in_shape, data_format, 'NCHW')
        if nnef_node_input.op == 'reshape':
            nnef_node_input.parameters['shape'] = in_shape

        if nnef_node_filter.op == 'variable':
            filter_tdf = nnef_node_filter.get_tensordatafile()
            nnef_tensor = np.transpose(filter_tdf.get_data().get_array()[0], [3, 2, 0, 1])
            filter_tdf.get_data().set_array(nnef_tensor)
            new_shape = list(np.shape(filter_tdf.get_data().get_array()[0]))
            filter_tdf.header.set_tensor_dimensions(new_shape)
            nnef_node_filter.parameters['shape'] = new_shape
            nnef_node_filter.output_shape = new_shape
        elif nnef_node_filter.op == 'reshape':
            current_shape = nnef_node_filter.parameters['shape'][:]
            new_shape = convert_format(current_shape, 'HWNC', 'CNHW')
            nnef_node_filter.parameters['shape'] = new_shape
            nnef_node_filter.output_shape = new_shape
        else:
            new_shape = [1]*len(in_shape)

        #Calculate output shape
        output_shape = len(in_shape) * [0]
        output_shape[0] = in_shape[0]
        output_shape[1] = new_shape[0]

        for i in range(2, len(in_shape)):
            if padding == []:
                output_shape[i] = math.ceil(in_shape[i] / strides[i-2])
            else:
                fd = (new_shape[i] - 1) * dilations[i-2] + 1
                padding_add = padding[i-2][0] + padding[i-2][1]
                output_shape[i] = math.floor((in_shape[i] + padding_add - fd) / strides[i-2]) + 1

        output_shape = convert_format(output_shape, 'NCHW', data_format)
        output_shape = [int(v) for v in output_shape]

        nnef_node = node.Conv(input=nnef_node_input,
                              filter=nnef_node_filter,
                              padding=padding,
                              stride=strides,
                              dilation=dilations,
                              _uid=self.gen_node_name(tfnode.name),
                              _output_shape=output_shape,
                              _data_format=data_format)

        attrs = {'padding': padding, 'strides': strides, 'dilations': dilations}
        return nnef_node, tf_inputs, attrs

    def import_Conv3D(self, tfnode):
        tf_inputs = {'input': 0, 'filter':1}

        data_format = tfnode.attr['data_format'].s.decode('ascii')
        padding = self.new_get_attr(tfnode, 'padding', 3, data_format)

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_filter = self.get_node_from_pool(tfnode, tf_inputs['filter'])

        if nnef_node_input.op == 'pad':
            pad_array = nnef_node_input.parameters['padding']
            nnef_node_input = nnef_node_input.parameters['input']

            padding = []
            if data_format == 'NDHWC':
                for i in range(1, len(pad_array)-1):
                    padding.append((pad_array[i][0], pad_array[i][1]))
            else:
                for i in range(2, len(pad_array)):
                    padding.append((pad_array[i][0], pad_array[i][1]))

        strides = self.new_get_attr(tfnode, 'strides', None, data_format)
        strides = strides[2:]

        dilations = self.new_get_attr(tfnode, 'dilations', None, data_format)
        dilations = dilations[2:]

        #Modify tensor data for filter
        in_shape = nnef_node_input.output_shape[:]
        in_shape = convert_format(in_shape, data_format, 'NCDHW')
        if nnef_node_input.op == 'reshape':
            nnef_node_input.parameters['shape'] = in_shape

        if nnef_node_filter.op == 'variable':
            filter_tdf = nnef_node_filter.get_tensordatafile()
            nnef_tensor = np.transpose(filter_tdf.get_data().get_array()[0], [4, 3, 0, 1, 2])
            filter_tdf.get_data().set_array(nnef_tensor)
            new_shape = list(np.shape(filter_tdf.get_data().get_array()[0]))
            filter_tdf.header.set_tensor_dimensions(new_shape)
            nnef_node_filter.parameters['shape'] = new_shape
        elif nnef_node_filter.op == 'reshape':
            current_shape = nnef_node_filter.parameters['shape'][:]
            new_shape = convert_format(current_shape, 'DHWNC', 'CNDHW')
            nnef_node_filter.parameters['shape'] = new_shape
        else:
            new_shape = [1]*len(in_shape)

        #Calculate output shape
        output_shape = len(in_shape) * [0]
        output_shape[0] = in_shape[0]
        output_shape[1] = new_shape[0]

        for i in range(2, len(in_shape)):
            if padding == []:
                output_shape[i] = math.ceil(in_shape[i] / strides[i-2])
            else:
                fd = (new_shape[i] - 1) * dilations[i-2] + 1
                padding_add = padding[i-2][0] + padding[i-2][1]
                output_shape[i] = math.floor((in_shape[i] + padding_add - fd) / strides[i-2]) + 1

        output_shape = convert_format(output_shape, 'NCDHW', data_format)
        output_shape = [int(v) for v in output_shape]

        nnef_node = node.Conv(input=nnef_node_input,
                              filter=nnef_node_filter,
                              padding=padding,
                              stride=strides,
                              dilation=dilations,
                              _uid=self.gen_node_name(tfnode.name),
                              _output_shape=output_shape,
                              _data_format=data_format)

        attrs = {'padding': padding, 'strides': strides, 'dilations': dilations}
        return nnef_node, tf_inputs, attrs

    def import_Conv2DBackpropInput(self, tfnode):
        tf_inputs = {'input': 2, 'filter':1}
        output_node = self.get_node_from_pool(tfnode, 0)
        self.remove_node_from_pool(output_node)
        data_format = tfnode.attr['data_format'].s.decode('ascii')
        padding = self.new_get_attr(tfnode, 'padding', 2, data_format)

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_filter = self.get_node_from_pool(tfnode, tf_inputs['filter'])

        if nnef_node_input.op == 'pad':
            pad_array = nnef_node_input.parameters['padding']
            nnef_node_input = nnef_node_input.parameters['input']

            padding = []
            if data_format == 'NHWC':
                for i in range(1, len(pad_array)-1):
                    padding.append((pad_array[i][0], pad_array[i][1]))
            else:
                for i in range(2, len(pad_array)):
                    padding.append((pad_array[i][0], pad_array[i][1]))

        strides = self.new_get_attr(tfnode, 'strides', None, data_format)
        strides = strides[2:]
        dilations = self.new_get_attr(tfnode, 'dilations', None, data_format)
        dilations = dilations[2:]

        #Modify tensor data for filter
        in_shape = nnef_node_input.output_shape[:]
        in_shape = convert_format(in_shape, data_format, 'NCHW')
        if nnef_node_input.op == 'reshape':
            nnef_node_input.parameters['shape'] = in_shape
        if nnef_node_filter.op == 'variable':
            filter_tdf = nnef_node_filter.get_tensordatafile()
            nnef_tensor = np.transpose(filter_tdf.get_data().get_array()[0], [3, 2, 0, 1])
            filter_tdf.get_data().set_array(nnef_tensor)
            new_shape = list(np.shape(filter_tdf.get_data().get_array()[0]))
            filter_tdf.header.set_tensor_dimensions(new_shape)
            nnef_node_filter.parameters['shape'] = new_shape
        elif nnef_node_filter.op == 'reshape':
            current_shape = nnef_node_filter.parameters['shape'][:]
            new_shape = convert_format(current_shape, 'HWNC', 'CNHW')
            nnef_node_filter.parameters['shape'] = new_shape
        else:
            new_shape = [1]*len(in_shape)
        #Calculate output shape
        output_shape = len(in_shape) * [0]
        output_shape[0] = in_shape[0]
        output_shape[1] = new_shape[1]

        for i in range(2, len(in_shape)):
            fd = (new_shape[i] - 1) * dilations[i-2] + 1
            if padding == []:
                padding_add = new_shape[i] - 2
            else:
                padding_add = padding[i-2][0] + padding[i-2][1]
            output_shape[i] = (in_shape[i] - 1)*strides[i-2] + fd - padding_add

        output_shape = convert_format(output_shape, 'NCHW', data_format)
        output_shape = [int(v) for v in output_shape]
        
        nnef_node = node.Deconv(input=nnef_node_input,
                                filter=nnef_node_filter,
                                padding=padding,
                                stride=strides,
                                dilation=dilations,
                                _uid=self.gen_node_name(tfnode.name),
                                _output_shape=output_shape,
                                _data_format=data_format)

        attrs = {'padding': padding, 'strides': strides, 'dilations': dilations}
        return nnef_node, tf_inputs, attrs

    def import_Conv3DBackpropInputV2(self, tfnode):
        tf_inputs = {'input': 2, 'filter':1}
        output_node = self.get_node_from_pool(tfnode, 0)
        self.remove_node_from_pool(output_node)

        data_format = tfnode.attr['data_format'].s.decode('ascii')
        padding = self.new_get_attr(tfnode, 'padding', 3, data_format)

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_filter = self.get_node_from_pool(tfnode, tf_inputs['filter'])

        if nnef_node_input.op == 'pad':
            pad_array = nnef_node_input.parameters['padding']
            nnef_node_input = nnef_node_input.parameters['input']

            padding = []
            if data_format == 'NHWC':
                for i in range(1, len(pad_array)-1):
                    padding.append((pad_array[i][0], pad_array[i][1]))
            else:
                for i in range(2, len(pad_array)):
                    padding.append((pad_array[i][0], pad_array[i][1]))

        strides = self.new_get_attr(tfnode, 'strides', None, data_format)
        strides = strides[2:]
        dilations = self.new_get_attr(tfnode, 'dilations', None, data_format)
        dilations = dilations[2:]

        #Modify tensor data for filter
        in_shape = nnef_node_input.output_shape[:]
        in_shape = convert_format(in_shape, data_format, 'NCDHW')
        if nnef_node_input.op == 'reshape':
            nnef_node_input.parameters['shape'] = in_shape

        if nnef_node_filter.op == 'variable':
            filter_tdf = nnef_node_filter.get_tensordatafile()
            nnef_tensor = np.transpose(filter_tdf.get_data().get_array()[0], [4, 3, 0, 1, 2])
            filter_tdf.get_data().set_array(nnef_tensor)
            new_shape = list(np.shape(filter_tdf.get_data().get_array()[0]))
            filter_tdf.header.set_tensor_dimensions(new_shape)
            nnef_node_filter.parameters['shape'] = new_shape
        elif nnef_node_filter.op == 'reshape':
            current_shape = nnef_node_filter.parameters['shape'][:]
            new_shape = convert_format(current_shape, 'DHWNC', 'CNDHW')
            nnef_node_filter.parameters['shape'] = new_shape
        else:
            new_shape = [1]*len(in_shape)

        #Calculate output shape
        output_shape = len(in_shape) * [0]
        output_shape[0] = in_shape[0]
        output_shape[1] = new_shape[1]

        for i in range(2, len(in_shape)):
            fd = (new_shape[i] - 1) * dilations[i-2] + 1
            if padding == []:
                padding_add = new_shape[i] - 2
            else:
                padding_add = padding[i-2][0] + padding[i-2][1]
            output_shape[i] = (in_shape[i] - 1)*strides[i-2] + fd - padding_add

        output_shape = convert_format(output_shape, 'NCDHW', data_format)
        output_shape = [int(v) for v in output_shape]

        nnef_node = node.Deconv(input=nnef_node_input,
                                filter=nnef_node_filter,
                                padding=padding,
                                stride=strides,
                                dilation=dilations,
                                _uid=self.gen_node_name(tfnode.name),
                                _output_shape=output_shape,
                                _data_format=data_format)

        attrs = {'padding': padding, 'strides': strides, 'dilations': dilations}
        return nnef_node, tf_inputs, attrs

    def import_CudnnRNN(self, tfnode):
        assert tfnode.attr['rnn_mode'].s.decode('ascii') == 'gru', "CudnnRNN import only supports GRU"
        tf_inputs = {'input': 0 }
        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        shape = [nnef_node_input.output_shape[0], nnef_node_input.output_shape[2]]

        nnef_node_reshape = node.Reshape(input=nnef_node_input,
                                         shape=shape,
                                         _uid=self.gen_node_name(tfnode.name) + '_reshape',
                                         _output_shape=shape,
                                         _maintain_format=False)

        self.node_pool[nnef_node_reshape.name] = nnef_node_reshape

        channels = 512
        scope = tfnode.name
        h = node.Variable(shape=[shape[0], channels],
                          label=scope + '/h',
                          _uid=self.gen_node_name(tfnode.name) + '_h',
                          _output_shape=[shape[0], channels],
                          _np_tensor=np.random.randn(*[shape[0], channels]).astype(np.float32),
                          _np_dtype=np.dtype(np.float32))
        h.tensor_data_file.write_to_disk(h.parameters['label'] + '.dat')

        z_filter = node.Variable(shape=[channels, shape[1]+channels],
                          label=scope + '/z/filter',
                          _uid=self.gen_node_name(tfnode.name) + '_z_filter',
                          _output_shape=[channels, shape[1]+channels],
                          _np_tensor=np.random.randn(*[channels, shape[1]+channels]).astype(np.float32),
                          _np_dtype=np.dtype(np.float32))
        z_filter.tensor_data_file.write_to_disk(h.parameters['label'] + '.dat')

        r_filter = node.Variable(shape=[channels, shape[1]+channels],
                          label=scope + '/r/filter',
                          _uid=self.gen_node_name(tfnode.name) + '_r_filter',
                          _output_shape=[channels, shape[1]+channels],
                          _np_tensor=np.random.randn(*[channels, shape[1]+channels]).astype(np.float32),
                          _np_dtype=np.dtype(np.float32))
        r_filter.tensor_data_file.write_to_disk(h.parameters['label'] + '.dat')

        s_filter = node.Variable(shape=[channels, shape[1]+channels],
                          label=scope + '/s/filter',
                          _uid=self.gen_node_name(tfnode.name) + '_s_filter',
                          _output_shape=[channels, shape[1]+channels],
                          _np_tensor=np.random.randn(*[channels, shape[1]+channels]).astype(np.float32),
                          _np_dtype=np.dtype(np.float32))
        s_filter.tensor_data_file.write_to_disk(h.parameters['label'] + '.dat')

        z_bias = node.Variable(shape=[1, channels],
                          label=scope + '/z/bias',
                          _uid=self.gen_node_name(tfnode.name) + '_z_bias',
                          _output_shape=[1, channels],
                          _np_tensor=np.random.randn(*[1, channels]).astype(np.float32),
                          _np_dtype=np.dtype(np.float32))
        z_bias.tensor_data_file.write_to_disk(h.parameters['label'] + '.dat')

        r_bias = node.Variable(shape=[1, channels],
                          label=scope + '/r/bias',
                          _uid=self.gen_node_name(tfnode.name) + '_r_bias',
                          _output_shape=[1, channels],
                          _np_tensor=np.random.randn(*[1, channels]).astype(np.float32),
                          _np_dtype=np.dtype(np.float32))
        r_bias.tensor_data_file.write_to_disk(h.parameters['label'] + '.dat')

        s_bias = node.Variable(shape=[1, channels],
                          label=scope + '/s/bias',
                          _uid=self.gen_node_name(tfnode.name) + '_s_bias',
                          _output_shape=[1, channels],
                          _np_tensor=np.random.randn(*[1, channels]).astype(np.float32),
                          _np_dtype=np.dtype(np.float32))
        s_bias.tensor_data_file.write_to_disk(h.parameters['label'] + '.dat')

        nnef_node_gru = node.Gru(input=nnef_node_reshape,
                                 channels=channels,
                                 scope=scope,
                                 _uid=self.gen_node_name(tfnode.name) + '_gru',
                                 _output_shape=nnef_node_reshape.output_shape[:])

        self.node_pool[nnef_node_gru.name] = nnef_node_gru

        shape_2 = [1, 1, channels]
        nnef_node = node.Reshape(input=nnef_node_gru,
                                 shape=shape_2,
                                 _uid=self.gen_node_name(tfnode.name),
                                 _output_shape=shape_2,
                                 _maintain_format=False)

        return nnef_node, tf_inputs, {}

    def import_DepthwiseConv2dNative(self, tfnode):
        tf_inputs = {'input': 0, 'filter':1}
        rank = 4
        data_format = tfnode.attr['data_format'].s.decode('ascii')

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_filter = self.get_node_from_pool(tfnode, tf_inputs['filter'])
        groups = nnef_node_input.output_shape[data_format.index('C')]

        padding = self.new_get_attr(tfnode, 'padding', rank-2, data_format)
        strides = self.new_get_attr(tfnode, 'strides', None, data_format)
        dilations = self.new_get_attr(tfnode, 'dilations', None, data_format)

        strides = strides[2:]
        dilations = dilations[2:]

        if nnef_node_input.op == 'pad':
            pad_array = nnef_node_input.parameters['padding']
            nnef_node_input = nnef_node_input.parameters['input']

            padding = []
            if data_format == 'NHWC':
                for i in range(1, len(pad_array)-1):
                    padding.append((pad_array[i][0], pad_array[i][1]))
            else:
                for i in range(2, len(pad_array)):
                    padding.append((pad_array[i][0], pad_array[i][1]))

        #Modify tensor data for filter
        in_shape = nnef_node_input.output_shape[:]
        in_shape = convert_format(in_shape, data_format, 'NCHW')
        if nnef_node_filter.op == 'variable':
            filter_tdf = nnef_node_filter.get_tensordatafile()
            nnef_tensor = filter_tdf.get_data().get_array()[0]
            shape = list(np.shape(nnef_tensor))
            nnef_tensor = np.reshape(nnef_tensor, [shape[0], shape[1], shape[2]*shape[3], 1])
            nnef_tensor = np.transpose(nnef_tensor, [2, 3, 0, 1])
            filter_tdf.get_data().set_array(nnef_tensor)
            new_shape = list(np.shape(filter_tdf.get_data().get_array()[0]))
            filter_tdf.header.set_tensor_dimensions(new_shape)
            nnef_node_filter.parameters['shape'] = new_shape
            nnef_node_filter.output_shape = new_shape
        else:
            new_shape = [1]*len(in_shape)

        #Calculate output shape
        output_shape = len(in_shape) * [0]
        output_shape[0] = in_shape[0]
        output_shape[1] = new_shape[1] * new_shape[0]
        for i in range(2, len(in_shape)):
            if padding == []:
                output_shape[i] = math.ceil(in_shape[i] / strides[i - 2])
            else:
                fd = (new_shape[i+1] - 1) * dilations[i - 2] + 1
                output_shape[i] = math.floor((in_shape[i] - fd) / strides[i - 2]) + 1

        output_shape = convert_format(output_shape, 'NCHW', data_format)
        output_shape = [int(v) for v in output_shape]

        nnef_node = node.Conv(input=nnef_node_input,
                              filter=nnef_node_filter,
                              padding=padding,
                              stride=strides,
                              dilation=dilations,
                              groups=groups,
                              _uid=self.gen_node_name(tfnode.name),
                              _output_shape=output_shape,
                              _data_format=data_format)

        attrs = {'padding': padding, 'strides': strides, 'dilations': dilations}
        return nnef_node, tf_inputs, attrs

    def import_Elu(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Elu(x=nnef_node_x,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape)
    
        return nnef_node, tf_inputs, attrs

    def import_Equal(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.EQ(x=nnef_node_x,
                            y=nnef_node_y,
                            _uid=self.gen_node_name(tfnode.name),
                            _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Exp(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Exp(x=nnef_node_x,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_ExpandDims(self, tfnode):
        tf_inputs = {'input': 0}
        attrs = {}

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        output_shape = nnef_node_input.output_shape[:]

        axis = int(self.get_node_from_pool(tfnode, 1).parameters['value'][0])
        output_shape.insert(axis, 1)

        nnef_node = node.Reshape(input=nnef_node_input,
                                 shape=output_shape,
                                 _uid=self.gen_node_name(tfnode.name),
                                 _output_shape=output_shape,
                                 _maintain_format=True)

        return nnef_node, tf_inputs, attrs

    def import_Floor(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Floor(x=nnef_node_x,
                               _uid=self.gen_node_name(tfnode.name),
                               _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_FusedBatchNorm(self, tfnode):
        tf_inputs = {'input': 0, 'scale': 1, 'offset': 2, 'mean': 3, 'variance': 4}
        data_format = tfnode.attr['data_format'].s.decode('ascii')

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_scale = self.get_node_from_pool(tfnode, tf_inputs['scale'])
        nnef_node_offset = self.get_node_from_pool(tfnode, tf_inputs['offset'])
        nnef_node_mean = self.get_node_from_pool(tfnode, tf_inputs['mean'])
        nnef_node_variance = self.get_node_from_pool(tfnode, tf_inputs['variance'])

        epsilon = self.new_get_attr(tfnode, 'epsilon', None)
        output_shape = nnef_node_input.output_shape[:]

        nnef_node = node.BatchNormalization(input=nnef_node_input,
                                            mean=nnef_node_mean,
                                            variance=nnef_node_variance,
                                            offset=nnef_node_offset,
                                            scale=nnef_node_scale,
                                            epsilon=epsilon,
                                            _uid=self.gen_node_name(tfnode.name),
                                            _output_shape=output_shape,
                                            _data_format=data_format)

        attrs = {'epsilon': epsilon, 'data_format': data_format}
        return nnef_node, tf_inputs, attrs

    def import_Greater(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.GT(x=nnef_node_x,
                            y=nnef_node_y,
                            _uid=self.gen_node_name(tfnode.name),
                            _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_GreaterEqual(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.GE(x=nnef_node_x,
                            y=nnef_node_y,
                            _uid=self.gen_node_name(tfnode.name),
                            _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Identity(self, tfnode):
        if self.gen_node_name(tfnode.name) == self.gen_node_name(tfnode.input[0]):
            return None, None, None
        else:
            self.removed_nodes += 1
            tf_inputs = {'x': 0}
            attrs = {}

            nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
            output_shape = nnef_node_x.output_shape[:]

            nnef_node = node.Idn(x=nnef_node_x,
                                 _uid=self.gen_node_name(tfnode.name),
                                 _output_shape=output_shape)

            return nnef_node, tf_inputs, attrs

    def import_Less(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.LT(x=nnef_node_x,
                            y=nnef_node_y,
                            _uid=self.gen_node_name(tfnode.name),
                            _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_LessEqual(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.LE(x=nnef_node_x,
                            y=nnef_node_y,
                            _uid=self.gen_node_name(tfnode.name),
                            _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Log(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Log(x=nnef_node_x,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_LogicalAnd(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.And(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_LogicalNot(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Not(x=nnef_node_x,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_LogicalOr(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.Or(x=nnef_node_x,
                            y=nnef_node_y,
                            _uid=self.gen_node_name(tfnode.name),
                            _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_LRN(self, tfnode):
        tf_inputs = {'input': 0}
        attrs = {}

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        output_shape = nnef_node_input.output_shape[:]
        alpha = tfnode.attr['alpha'].f
        beta = tfnode.attr['beta'].f
        bias = tfnode.attr['bias'].f
        size = [1, tfnode.attr['depth_radius'].i, 1, 1]

        nnef_node = node.LocalResponseNormalization(input=nnef_node_input,
                                                    alpha=alpha,
                                                    beta=beta,
                                                    bias=bias,
                                                    size=size,
                                                    _uid=self.gen_node_name(tfnode.name),
                                                    _output_shape=output_shape)

        attrs = {'alpha' :alpha, 'beta': beta, 'bias': bias, 'depth_radius': size}
        return nnef_node, tf_inputs, attrs

    def import_MatMul(self, tfnode):
        tf_inputs = {'A': 0, 'B': 1}

        nnef_node_A = self.get_node_from_pool(tfnode, tf_inputs['A'])
        nnef_node_B = self.get_node_from_pool(tfnode, tf_inputs['B'])

        output_shape = []
        for i in nnef_node_A.output_shape[0:-1]:
            output_shape.append(i)
        for i in nnef_node_B.output_shape[1:]:
            output_shape.append(i)

        trA = self.new_get_attr(tfnode, 'transpose_a', None)
        trB = self.new_get_attr(tfnode, 'transpose_b', None)

        nnef_node = node.Matmul(A=nnef_node_A,
                                B=nnef_node_B,
                                transposeA=trA,
                                transposeB=trB,
                                _uid=self.gen_node_name(tfnode.name),
                                _output_shape=output_shape)

        attrs = {'transpose_a': trA, 'transpose_b': trB}
        return nnef_node, tf_inputs, attrs

    def import_Max(self, tfnode):
        tf_inputs = {'input': 0, 'axis':1}
        attrs = {'keep_dims': None}

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_axis = self.get_node_from_pool(tfnode, tf_inputs['axis'])

        input_shape = nnef_node_input.output_shape[:]
        if nnef_node_axis.op == 'variable':
            shape = nnef_node_axis.get_tensordatafile().get_data().get_array()[0][0]
        else:
            shape = [int(nnef_node_axis.parameters['value'][0])]

        axes = []
        for i in shape:
            if i not in axes:
                axes.append(i)
        axes.sort()
        self.remove_node_from_pool(nnef_node_axis)

        output_shape = input_shape[:]
        if(tfnode.attr['keep_dims'].b or tfnode.attr['keepdims'].b):
            for i in axes:
                output_shape[i] = 1

            nnef_node_max = node.MaxReduce(input=nnef_node_input,
                                           axes=axes,
                                           _output_shape=output_shape,
                                           _uid=self.gen_node_name(tfnode.name) + '_max',)

            self.node_pool[nnef_node_max.name] = nnef_node_max

            nnef_node = node.Reshape(input=nnef_node_max,
                                     shape=output_shape,
                                     _output_shape=output_shape,
                                     _uid=self.gen_node_name(tfnode.name),
                                     _maintain_format=True)

        else:
            axes.sort(reverse=True)
            for i in axes:
                output_shape.pop(i)
            if output_shape == []:
                output_shape = [1]
            axes.sort()

            nnef_node = node.MaxReduce(input=nnef_node_input,
                                       axes=axes,
                                       _output_shape=output_shape,
                                       _uid=self.gen_node_name(tfnode.name))

        return nnef_node, tf_inputs, attrs

    def import_Maximum(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)
        nnef_node = node.Max(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_MaxPool(self, tfnode):
        rank = 4
        tf_inputs = {'input':0}
        data_format = tfnode.attr['data_format'].s.decode('ascii')
        padding = self.new_get_attr(tfnode, 'padding', rank, data_format)

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])

        if nnef_node_input.op == 'pad':
            pad_array = nnef_node_input.parameters['padding']
            nnef_node_input = nnef_node_input.parameters['input']

            padding = []
            if data_format == 'NHWC':
                padding.append((pad_array[0][0], pad_array[0][1]))
                padding.append((pad_array[-1][0], pad_array[-1][1]))
                for i in range(1, len(pad_array)-1):
                    padding.append((pad_array[i][0], pad_array[i][1]))
            else:
                padding
                for i in range(len(pad_array)):
                    padding.append((pad_array[i][0], pad_array[i][1]))

        sizes = self.new_get_attr(tfnode, 'ksize', None, data_format)
        strides = self.new_get_attr(tfnode, 'strides', None, data_format)
        dilations = [1, 1, 1, 1]

        #Modify tensor data for filter
        in_shape = nnef_node_input.output_shape
        in_shape = convert_format(in_shape, data_format, 'NCHW')
        if nnef_node_input.op == 'reshape':
            nnef_node_input.parameters['shape'] = in_shape

        #Calculate output shape
        output_shape = len(in_shape) * [0]

        for i in range(len(in_shape)):
            if padding == []:
                output_shape[i] = math.ceil(in_shape[i] / strides[i])
            else:
                fd = (sizes[i] - 1) * dilations[i] + 1
                padding_add = padding[i][0] + padding[i][1]
                output_shape[i] = math.floor((in_shape[i] + padding_add - fd) / strides[i]) + 1

        output_shape = convert_format(output_shape, 'NCHW', data_format)
        output_shape = [int(v) for v in output_shape]

        nnef_node = node.MaxPool(input=nnef_node_input,
                                 size=sizes,
                                 padding=padding,
                                 stride=strides,
                                 dilation=dilations,
                                 _uid=self.gen_node_name(tfnode.name),
                                 _output_shape=output_shape,
                                 _data_format=data_format)

        attrs = {'padding': padding, 'strides': strides, 'dilations': dilations}
        return nnef_node, tf_inputs, attrs

    def import_MaxPoolWithArgmax(self, tfnode):
        tf_inputs = {'input':0}
        attrs = {'padding': 4, 'ksize': None, 'strides': None}
        data_format = 'NHWC'
        padding = self.new_get_attr(tfnode, 'padding', 4, data_format)
        main_nnef_node_name = self.gen_node_name(tfnode.name) + ', ' + self.gen_node_name(tfnode.name + ':1')

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])

        if nnef_node_input.op == 'pad':
            pad_array = nnef_node_input.parameters['padding']
            nnef_node_input = nnef_node_input.parameters['input']

            padding = []
            if data_format == 'NHWC':
                padding.append((pad_array[0][0], pad_array[0][1]))
                padding.append((pad_array[-1][0], pad_array[-1][1]))
                for i in range(1, len(pad_array)-1):
                    padding.append((pad_array[i][0], pad_array[i][1]))
            else:
                padding
                for i in range(len(pad_array)):
                    padding.append((pad_array[i][0], pad_array[i][1]))


        sizes = self.new_get_attr(tfnode, 'ksize', None, data_format)
        strides = self.new_get_attr(tfnode, 'strides', None, data_format)
        dilations = [1, 1, 1, 1]

        #Modify tensor data for filter
        in_shape = nnef_node_input.output_shape
        in_shape = convert_format(in_shape, data_format, 'NCHW')
        if nnef_node_input.op == 'reshape':
            nnef_node_input.parameters['shape'] = in_shape

        #Calculate output shape
        output_shape = len(in_shape) * [0]

        for i in range(len(in_shape)):
            if padding == []:
                output_shape[i] = math.ceil(in_shape[i] / strides[i])
            else:
                fd = (sizes[i] - 1) * dilations[i] + 1
                padding_add = padding[i][0] + padding[i][1]
                output_shape[i] = math.floor((in_shape[i] + padding_add - fd) / strides[i]) + 1

        output_shape = convert_format(output_shape, 'NCHW', data_format)
        output_shape = [int(v) for v in output_shape]

        nnef_node_main = node.MaxPoolWithIndex(input=nnef_node_input,
                                               padding=padding,
                                               size=sizes,
                                               stride=strides,
                                               dilation=dilations,
                                               _uid=main_nnef_node_name,
                                               _output_shape=output_shape,
                                               _data_format=data_format)

        nnef_node_pool = node.OutputVal(base_node=nnef_node_main,
                                        base_index=0,
                                        _uid=self.gen_node_name(tfnode.name),
                                        _output_shape=output_shape,
                                        _data_format=data_format)

        self.node_pool[nnef_node_pool.name] = nnef_node_pool

        nnef_node_index = node.OutputVal(base_node=nnef_node_main,
                                         base_index=1,
                                         _uid=self.gen_node_name(tfnode.name + ':1'),
                                         _output_shape=output_shape)

        self.node_pool[nnef_node_index.name] = nnef_node_index

        return nnef_node_main, tf_inputs, attrs

    def import_Mean(self, tfnode):
        tf_inputs = {'input': 0, 'axis':1}
        attrs = {'keep_dims': None}

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_axis = self.get_node_from_pool(tfnode, tf_inputs['axis'])

        input_shape = nnef_node_input.output_shape[:]
        if nnef_node_axis.op == 'variable':
            shape = nnef_node_axis.get_tensordatafile().get_data().get_array()[0][0]
        else:
            shape = [int(nnef_node_axis.parameters['value'][0])]

        axes = []
        for i in shape:
            if i not in axes:
                axes.append(i)
        axes.sort()
        self.remove_node_from_pool(nnef_node_axis)

        output_shape = input_shape[:]
        if(tfnode.attr['keep_dims'].b or tfnode.attr['keepdims'].b):
            for i in axes:
                output_shape[i] = 1

            nnef_node_mean = node.MeanReduce(input=nnef_node_input,
                                             axes=axes,
                                             _output_shape=output_shape,
                                             _uid=self.gen_node_name(tfnode.name) + '_mean',)

            self.node_pool[nnef_node_mean.name] = nnef_node_mean

            nnef_node = node.Reshape(input=nnef_node_mean,
                                     shape=output_shape,
                                     _output_shape=output_shape,
                                     _uid=self.gen_node_name(tfnode.name),
                                     _maintain_format=True)

        else:
            axes.sort(reverse=True)
            for i in axes:
                output_shape.pop(i)
            if output_shape == []:
                output_shape = [1]
            axes.sort()

            nnef_node = node.MeanReduce(input=nnef_node_input,
                                        axes=axes,
                                        _output_shape=output_shape,
                                        _uid=self.gen_node_name(tfnode.name))

        return nnef_node, tf_inputs, attrs

    def import_Minimum(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.Min(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape,
                             _data_format=tfnode.attr['data_format'])

        return nnef_node, tf_inputs, attrs

    def import_Mul(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.Mul(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape,
                             _data_format=tfnode.attr['data_format'])

        return nnef_node, tf_inputs, attrs

    def import_Neg(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Neg(x=nnef_node_x,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_NotEqual(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.NE(x=nnef_node_x,
                            y=nnef_node_y,
                            _uid=self.gen_node_name(tfnode.name),
                            _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Pack(self, tfnode):
        tf_inputs = {}
        attrs = {'axis':None}

        nnef_nodes = []
        for i in range(len(tfnode.input)):
            nnef_node_val = self.get_node_from_pool(tfnode, i)
            nnef_nodes.append(nnef_node_val)
            tf_inputs['value_' + str(i)] = i

        axis = tfnode.attr['axis'].i

        output_shape = nnef_nodes[0].output_shape[:]
        output_shape.insert(axis, len(nnef_nodes))

        nnef_node = node.Stack(values=nnef_nodes,
                               axis=axis,
                               _uid=self.gen_node_name(tfnode.name),
                               _output_shape=output_shape)


        return nnef_node, tf_inputs, attrs

    def import_Pad(self, tfnode):
        tf_inputs = {'input': 0, 'pads' : 1}
        attrs = {}

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_pad = self.get_node_from_pool(tfnode, tf_inputs['pads'])

        if nnef_node_pad.op == 'variable' and nnef_node_pad.parameters['shape'][0] > 1:
            padding = nnef_node_pad.get_tensordatafile().get_data().get_array()[0]
            nnef_node = node.Pad(input=nnef_node_input,
                                 padding=padding,
                                 _uid=self.gen_node_name(tfnode.name),
                                 _output_shape=nnef_node_input.output_shape[:])
        else:
            raise ValueError("Currently unsupported pad arguments")

        return nnef_node, tf_inputs, attrs

    def import_Placeholder(self, tfnode):
        size = []
        shape = self._get_attr(tfnode, 'shape')
        if shape is None:
            size = [1, 224, 224, 3]
        else:
            for dimen in shape.dim:
                size.append(dimen.size)

        if size[0] < 0:
            shape = [1]
        else:
            shape = [size[0]]
        for i in range(1, len(size)):
            shape.append(size[i])

        if self.start_length == 0:
            self.start_length = len(shape)

        nnef_node = node.External(shape=shape,
                                  _uid=self.gen_node_name(tfnode.name),
                                  _output_shape=shape)

        inputs = {}
        attrs = {'shape': None}
        return nnef_node, inputs, attrs

    def import_PlaceholderWithDefault(self, tfnode):
        self.name_convs[self.gen_node_name(tfnode.name)] = self.gen_node_name(tfnode.input[0])

        return None, None, None

    def import_Pow(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Pow(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_RealDiv(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.Div(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Relu(self, tfnode):
        tf_inputs = {'x': 0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Relu(x=nnef_node_x,
                              _uid=self.gen_node_name(tfnode.name),
                              _output_shape=output_shape,)

        return nnef_node, tf_inputs, attrs

    def import_Relu6(self, tfnode):
        tf_inputs = {'x': 0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node_relu = node.Relu(x=nnef_node_x,
                                   _uid=self.gen_node_name(tfnode.name)+'_relu',
                                   _output_shape=output_shape)

        self.node_pool[nnef_node_relu.name] = nnef_node_relu

        nnef_node = node.Min(x=nnef_node_relu,
                             y=6.0,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Reshape(self, tfnode):
        tf_inputs = {'input': 0, 'shape': 1}
        attrs = {}

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_shape = self.get_node_from_pool(tfnode, tf_inputs['shape'])

        self.name_convs[nnef_node_input.name] = self.gen_node_name(tfnode.name)

        shape = None
        if nnef_node_shape.op == 'variable':
            shape = list(nnef_node_shape.get_tensordatafile().get_data().get_array()[0][0])
            self.remove_node_from_pool(nnef_node_shape)
        elif nnef_node_shape.op == 'shape_of':
            shape = nnef_node_shape.output_shape[:]
        else:
            shape = np.reshape(np.asarray(nnef_node_shape.get_value(), dtype=np.int32), [-1])
            self.remove_node_from_pool(nnef_node_shape)

        if shape == [-1, 10, 768] and tfnode.name == 'Reshape_4':
            shape = [1, 1, 768]

        if shape == [-1, 10, 768] and tfnode.name == 'Reshape_4':
            shape = [1, 1, 768]

        in_shape = nnef_node_input.output_shape[:]
        output_shape = []
        for i in shape:
            output_shape.append(i)
        if -1 in output_shape:
            in_size = 1
            for i in in_shape:
                in_size *= i
            neg_index = -1
            for i in range(len(output_shape)):
                if output_shape[i] == -1:
                    neg_index = i
                else:
                    in_size = in_size/output_shape[i]
            output_shape[neg_index] = int(in_size)

        output_shape = [int(v) for v in output_shape]

        nnef_node = node.Reshape(input=nnef_node_input,
                                 shape=output_shape,
                                 _uid=self.gen_node_name(tfnode.name),
                                 _output_shape=output_shape,
                                 _maintain_format=False)

        return nnef_node, tf_inputs, attrs

    def import_ResizeArea(self, tfnode):
        if self.start_format == None:
            self.start_format = 'NHWC'

        tf_inputs = {'input':0, 'factor': 1}
        attrs = {}

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_factor = self.get_node_from_pool(tfnode, tf_inputs['factor'])
        input_shape = nnef_node_input.output_shape[:]

        if nnef_node_factor.op == 'variable':
            output_size = nnef_node_factor.get_tensordatafile().get_data().get_array()[0][0]
            self.remove_node_from_pool(nnef_node_factor)
        else:
            print(nnef_node_factor.op)
            assert False, "Not currently handled"

        factor = []
        output_shape = [input_shape[0]]
        for i in range(len(input_shape[1:-1])):
            assert input_shape[i+1]%output_size[i] == 0, "Unable to convert, ResizeArea uses non-integer factors"
            factor.append(int(input_shape[i+1]/output_size[i]))
            output_shape.append(int(output_size[i]))
        output_shape.append(input_shape[-1])

        nnef_node = node.AreaDownsample(input=nnef_node_input,
                                        factor=factor,
                                        _uid=self.gen_node_name(tfnode.name),
                                        _output_shape=output_shape,
                                        _data_format='NHWC')

        return nnef_node, tf_inputs, attrs

    def import_ResizeBilinear(self, tfnode):
        if self.start_format == None:
            self.start_format = 'NHWC'

        tf_inputs = {'input':0, 'factor': 1}
        attrs = {}

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_factor = self.get_node_from_pool(tfnode, tf_inputs['factor'])
        input_shape = nnef_node_input.output_shape[:]

        if nnef_node_factor.op == 'variable':
            output_size = nnef_node_factor.get_tensordatafile().get_data().get_array()[0][0]
            self.remove_node_from_pool(nnef_node_factor)
        else:
            print(nnef_node_factor.op)
            assert False, "Not currently handled"

        factor = []
        output_shape = [input_shape[0]]
        for i in range(len(input_shape[1:-1])):
            assert output_size[i]%input_shape[i+1] == 0, "Unable to convert, ResizeBilinear uses non-integer factors"
            factor.append(int(output_size[i]/input_shape[i+1]))
            output_shape.append(int(output_size[i]))
        output_shape.append(input_shape[-1])

        nnef_node = node.MultilinearUpsample(input=nnef_node_input,
                                             factor=factor,
                                             _uid=self.gen_node_name(tfnode.name),
                                             _output_shape=output_shape,
                                             _data_format='NHWC')

        return nnef_node, tf_inputs, attrs

    def import_ResizeNearestNeighbor(self, tfnode):
        if self.start_format == None:
            self.start_format = 'NHWC'

        tf_inputs = {'input':0, 'factor': 1}
        attrs = {}

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_factor = self.get_node_from_pool(tfnode, tf_inputs['factor'])
        input_shape = nnef_node_input.output_shape[:]

        if nnef_node_factor.op == 'variable':
            output_size = nnef_node_factor.get_tensordatafile().get_data().get_array()[0][0]
            self.remove_node_from_pool(nnef_node_factor)
        else:
            print(nnef_node_factor.op)
            assert False, "Not currently handled"

        factor = []
        output_shape = [input_shape[0]]

        if input_shape[1] < output_size[0]:
            for i in range(len(input_shape[1:-1])):
                assert output_size[i]%input_shape[i+1] == 0, "Unable to convert, ResizeNearestNeighbor uses non-integer factors"
                factor.append(int(output_size[i]/input_shape[i+1]))
                output_shape.append(int(output_size[i]))
            output_shape.append(input_shape[-1])

            nnef_node = node.NearestUpsample(input=nnef_node_input,
                                             factor=factor,
                                             _uid=self.gen_node_name(tfnode.name),
                                             _output_shape=output_shape,
                                             _data_format='NHWC')
        else:
            for i in range(len(input_shape[1:-1])):
                assert input_shape[i+1]%output_size[i] == 0, "Unable to convert, ResizeNearestNeighbor uses non-integer factors"
                factor.append(int(input_shape[i+1]/output_size[i]))
                output_shape.append(int(output_size[i]))
            output_shape.append(input_shape[-1])

            nnef_node = node.NearestDownsample(input=nnef_node_input,
                                               factor=factor,
                                               _uid=self.gen_node_name(tfnode.name),
                                               _output_shape=output_shape,
                                               _data_format='NHWC')

        return nnef_node, tf_inputs, attrs

    def import_Round(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Round(x=nnef_node_x,
                               _uid=self.gen_node_name(tfnode.name),
                               _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Rsqrt(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Rsqrt(x=nnef_node_x,
                               _uid=self.gen_node_name(tfnode.name),
                               _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Select(self, tfnode):
        tf_inputs = {'condition':0, 'true_value':1, 'false_value':2}
        attrs = {}

        nnef_node_condition = self.get_node_from_pool(tfnode, tf_inputs['condition'])
        nnef_node_true_value = self.get_node_from_pool(tfnode, tf_inputs['true_value'])
        nnef_node_false_value = self.get_node_from_pool(tfnode, tf_inputs['false_value'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_true_value, nnef_node_false_value)

        nnef_node = node.Select(condition=nnef_node_condition,
                                true_value=nnef_node_true_value,
                                false_value=nnef_node_false_value,
                                _uid=self.gen_node_name(tfnode.name),
                                _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Shape(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.ShapeOf(x=nnef_node_x,
                                 _uid=self.gen_node_name(tfnode.name),
                                 _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Sigmoid(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Sigmoid(x=nnef_node_x,
                                 _uid=self.gen_node_name(tfnode.name),
                                 _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Sign(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Sign(x=nnef_node_x,
                              _uid=self.gen_node_name(tfnode.name),
                              _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Slice(self, tfnode):
        tf_inputs = {'input': 0, 'begin':1, 'end':2}
        attrs = {}
        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_begin = self.get_node_from_pool(tfnode, tf_inputs['begin'])
        nnef_node_end = self.get_node_from_pool(tfnode, tf_inputs['end'])

        if nnef_node_begin.op == 'variable':
            begin = nnef_node_begin.get_tensordatafile().get_data().get_array()[0][0]
        elif nnef_node_begin.op == 'constant':
            begin = np.reshape(np.asarray(nnef_node_begin.parameters['value'], dtype=np.int32), nnef_node_begin.parameters['shape'])
        else:
            begin = nnef_node_begin.get_value()
        if nnef_node_end.op == 'variable':
            end = nnef_node_end.get_tensordatafile().get_data().get_array()[0][0]
        elif nnef_node_end == 'constant':
            end = np.reshape(np.asarray(nnef_node_end.parameters['value'], dtype=np.int32), nnef_node_end.parameters['shape'])
        else:
            end = nnef_node_end.get_value()

        axes = list(range(len(begin)))
        output_shape = len(axes)*[0]
        for i in range(len(axes)):
            output_shape[i] = int(end[i]-begin[i])

        nnef_node = node.Slice(input=nnef_node_input,
                               axes=axes,
                               begin=list(begin),
                               end=list(end),
                               _uid=self.gen_node_name(tfnode.name),
                               _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs


    def import_Softmax(self, tfnode):
        tf_inputs = {'x': 0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Softmax(x=nnef_node_x,
                                 _uid=self.gen_node_name(tfnode.name),
                                 _output_shape=output_shape,)

        return nnef_node, tf_inputs, attrs

    def import_Softplus(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Softplus(x=nnef_node_x,
                                  _uid=self.gen_node_name(tfnode.name),
                                  _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Softsign(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Softsign(x=nnef_node_x,
                                  _uid=self.gen_node_name(tfnode.name),
                                  _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Split(self, tfnode):
        tf_inputs = {'value':1, 'axis': 0}
        attrs = {'num_split': None}

        nnef_node_value = self.get_node_from_pool(tfnode, tf_inputs['value'])
        nnef_node_axis = self.get_node_from_pool(tfnode, tf_inputs['axis'])
        self.remove_node_from_pool(nnef_node_axis)

        split_axis = int(nnef_node_axis.parameters['value'][0])

        num_split = tfnode.attr['num_split'].i
        names = []
        if num_split >= 1:
            new_name = '['
            for i in range(num_split):
                if i == 0:
                    new_name = new_name + self.gen_node_name(tfnode.name) + ', '
                    names.append(self.gen_node_name(tfnode.name))
                else:
                    new_name = new_name + self.gen_node_name(tfnode.name + ':' + str(i)) + ', '
                    names.append(self.gen_node_name(tfnode.name + ':' + str(i)))

            new_name = new_name[:-2] + ']'

        input_shape = nnef_node_value.output_shape[:]
        ratio = math.floor(input_shape[split_axis]/num_split)
        modu = input_shape[split_axis]%num_split

        ratios = []
        for i in range(len(names)):
            rat_val = ratio
            if modu != 0:
                rat_val += 1
                modu -= 1
            ratios.append(int(rat_val))

        nnef_node_split = node.Split(value=nnef_node_value,
                                     axis=split_axis,
                                     ratios=ratios,
                                     _uid=new_name,
                                     _output_shape=input_shape)

        for i in range(len(names)):
            out_shape = input_shape[:]
            out_shape[split_axis] = ratios[i]
            nnef_node = node.OutputVal(base_node=nnef_node_split,
                                       base_index=i,
                                       _uid=names[i],
                                       _output_shape=out_shape)
            self.node_pool[nnef_node.name] = nnef_node

        return nnef_node_split, tf_inputs, attrs

    def import_Sqrt(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Sqrt(x=nnef_node_x,
                              _uid=self.gen_node_name(tfnode.name),
                              _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Square(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Sqr(x=nnef_node_x,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Squeeze(self, tfnode):
        tf_inputs = {'input': 0}
        attrs = {}

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])

        input_shape = nnef_node_input.output_shape
        output_shape = []
        if tfnode.attr['squeeze_dims'].list.i:
            for i in range(len(input_shape)):
                if i not in tfnode.attr['squeeze_dims'].list.i:
                    output_shape.append(input_shape[i])
        else:
            for i in input_shape:
                if i != 1:
                    output_shape.append(i)

        output_shape = [int(v) for v in output_shape]

        nnef_node = node.Reshape(input=nnef_node_input,
                                 shape=output_shape,
                                 _uid=self.gen_node_name(tfnode.name),
                                 _output_shape=output_shape,
                                 _maintain_format=False)

        return nnef_node, tf_inputs, attrs

    def import_StridedSlice(self, tfnode):
        tf_inputs = {'input': 0, 'begin':1, 'end':2, 'strides':3}
        attrs = {}
        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_begin = self.get_node_from_pool(tfnode, tf_inputs['begin'])
        nnef_node_end = self.get_node_from_pool(tfnode, tf_inputs['end'])
        nnef_node_strides = self.get_node_from_pool(tfnode, tf_inputs['strides'])

        if nnef_node_begin.op == 'variable':
            begin = nnef_node_begin.get_tensordatafile().get_data().get_array()[0][0]
        elif nnef_node_begin.op == 'constant':
            begin = np.reshape(np.asarray(nnef_node_begin.parameters['value'], dtype=np.int32), nnef_node_begin.parameters['shape'])
        else:
            begin = nnef_node_begin.get_value()
        if nnef_node_end.op == 'variable':
            end = nnef_node_end.get_tensordatafile().get_data().get_array()[0][0]
        elif nnef_node_end.op == 'constant':
            end = np.reshape(np.asarray(nnef_node_end.parameters['value'], dtype=np.int32), nnef_node_end.parameters['shape'])
        else:
            end = nnef_node_end.get_value()
        if nnef_node_strides.op == 'variable':
            strides = nnef_node_strides.get_tensordatafile().get_data().get_array()[0][0]
        elif nnef_node_strides.op == 'constant':
            strides = np.reshape(np.asarray(nnef_node_strides.parameters['value'], dtype=np.int32), nnef_node_strides.parameters['shape'])
        else:
            strides = nnef_node_strides.get_value()

        for stride in strides:
            assert stride == 1, "Slice operation uses a stride that is not one, currently unsupported."
        axes = list(range(len(begin)))
        output_shape = len(axes)*[0]
        for i in range(len(axes)):
            if begin[i] == -1 and end[i] == 0:
                output_shape[i] = 0
            elif end[i] == 0:
                output_shape[i] = int(nnef_node_input.output_shape[i] - begin[i])
            else:
                output_shape[i] = int(end[i]-begin[i])

        if 0 in output_shape:
            nnef_node_slice = node.Slice(input=nnef_node_input,
                                   axes=axes,
                                   begin=list(begin),
                                   end=list(end),
                                   _uid=self.gen_node_name(tfnode.name) + '_slice',
                                   _output_shape=output_shape)

            self.node_pool[nnef_node_slice.name] = nnef_node_slice

            squeeze_shape = [value for value in output_shape if value != 0]

            nnef_node = node.Reshape(input=nnef_node_slice,
                                     shape=squeeze_shape,
                                     _uid=self.gen_node_name(tfnode.name),
                                     _output_shape=squeeze_shape,
                                     _maintain_format=False)
        else:
            nnef_node = node.Slice(input=nnef_node_input,
                                   axes=axes,
                                   begin=list(begin),
                                   end=list(end),
                                   _uid=self.gen_node_name(tfnode.name),
                                   _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Sub(self, tfnode):
        tf_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        nnef_node_y = self.get_node_from_pool(tfnode, tf_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.Sub(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(tfnode.name),
                             _output_shape=output_shape,
                             _data_format=tfnode.attr['data_format'])

        return nnef_node, tf_inputs, attrs

    def import_Sum(self, tfnode):
        tf_inputs = {'input':0, 'axis':1}
        attrs = {'keep_dims': None}

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_axis = self.get_node_from_pool(tfnode, tf_inputs['axis'])

        input_shape = nnef_node_input.output_shape[:]
        if nnef_node_axis.op == 'variable':
            shape = nnef_node_axis.get_tensordatafile().get_data().get_array()[0][0]
        else:
            shape = [int(nnef_node_axis.parameters['value'][0])]

        axes = []
        for i in shape:
            if i not in axes:
                axes.append(i)
        axes.sort()
        self.remove_node_from_pool(nnef_node_axis)

        output_shape = input_shape[:]
        if(tfnode.attr['keep_dims'].b or tfnode.attr['keepdims'].b):
            for i in axes:
                output_shape[i] = 1

            nnef_node_sum = node.SumReduce(input=nnef_node_input,
                                           axes=axes,
                                           _output_shape=output_shape,
                                           _uid=self.gen_node_name(tfnode.name) + '_sum',)

            self.node_pool[nnef_node_sum.name] = nnef_node_sum

            nnef_node = node.Reshape(input=nnef_node_sum,
                                     shape=output_shape,
                                     _output_shape=output_shape,
                                     _uid=self.gen_node_name(tfnode.name),
                                     _maintain_format=True)

        else:
            axes.sort(reverse=True)
            for i in axes:
                output_shape.pop(i)
            if output_shape == []:
                output_shape = [1]
            axes.sort()

            nnef_node = node.SumReduce(input=nnef_node_input,
                                       axes=axes,
                                       _output_shape=output_shape,
                                       _uid=self.gen_node_name(tfnode.name))

        return nnef_node, tf_inputs, attrs

    def import_Tanh(self, tfnode):
        tf_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(tfnode, tf_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Tanh(x=nnef_node_x,
                              _uid=self.gen_node_name(tfnode.name),
                              _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

    def import_Transpose(self, tfnode):
        tf_inputs = {'input':0, 'axes': 1}
        attrs = {}

        nnef_node_input = self.get_node_from_pool(tfnode, tf_inputs['input'])
        nnef_node_axes = self.get_node_from_pool(tfnode, tf_inputs['axes'])

        axes = list(nnef_node_axes.get_tensordatafile().get_data().get_array()[0][0])
        self.remove_node_from_pool(nnef_node_axes)

        output_shape = []
        for i in range(len(nnef_node_input.output_shape)):
            output_shape.append(nnef_node_input.output_shape[axes[i]])

        nnef_node = node.Transpose(input=nnef_node_input,
                                   axes=axes,
                                   _uid=self.gen_node_name(tfnode.name),
                                   _output_shape=output_shape)

        return nnef_node, tf_inputs, attrs

class TensorflowExporter(TensorflowLogger, ImporterExporter):
    def __init__(self, output_model):
        super(TensorflowExporter, self).__init__()
        self.output_model = output_model
        self.mapping = {0:0, 1:3, 2:1, 3:2}

    def run(self, nnef_graph):
        self.nxgraph = nnef_graph.get_nx_graph()
        self.generate_tf_graph()

    def format_name(self, name):
        index = name.find('_')
        if index != -1:
            newName = name[:index] + '/' + name[index+1:]
        else:
            newName = name
        newName = name.replace('_', '/')
        return newName

    def add_input(self, tfnode, nnef_node, param_name, order=[]):
        index = 0
        if isinstance(nnef_node.parameters[param_name], list):
            nnef_node_param = nnef_node.parameters[param_name][index]
        else:
            nnef_node_param = nnef_node.parameters[param_name]

        while nnef_node_param is not None:
            if not isinstance(nnef_node_param, Node):
                nodeConst = NodeDef(name=self.format_name(nnef_node.name) + '/' + param_name, op='Const')
                nodeConst.attr['dtype'].type = 1
                nodeConst.attr['value'].tensor.dtype = 1
                nodeConst.attr['value'].tensor.float_val.extend([nnef_node_param])

                self.tf_graph.node.extend([nodeConst])
                tfnode.input.extend([nodeConst.name])

            elif nnef_node_param.op == 'variable':
                tfnode.input.extend([self.format_name(nnef_node_param.name)])
                for n in self.tf_graph.node:
                    if (n.name) == tfnode.input[-1]:
                        if n.attr['value'].tensor.tensor_shape.dim:
                            return
                        shapes = nnef_node_param.parameters['shape']
                        if not order:
                            for i in range(0, len(shapes)):
                                n.attr['value'].tensor.tensor_shape.dim.add().size = shapes[i]
                            np_array_read = np.asarray(nnef_node_param.get_tensordatafile().get_data().get_array()[0], dtype=np.float32)
                            n.attr['value'].tensor.tensor_content = np_array_read.tobytes()
                        else:
                            new_shape = []
                            for i in range(len(order)):
                                new_shape.append(shapes[order[i]])
                                n.attr['value'].tensor.tensor_shape.dim.add().size = new_shape[i]
                            np_array_read = np.asarray(nnef_node_param.get_tensordatafile().get_data().get_array()[0], dtype=np.float32)
                            np_array_read = np.reshape(np_array_read, shapes)
                            if len(new_shape) < len(shapes):
                                np_array_read = np.reshape(np_array_read, new_shape)
                            else:
                                np_array_read = np.transpose(np_array_read, order)
                            n.attr['value'].tensor.tensor_content = np.reshape(np_array_read, new_shape).tobytes()
                        break
            elif nnef_node_param.op == 'reshape':
                tfnode.input.extend([self.format_name(nnef_node_param.name)])
                if order != []:
                    for n in self.tf_graph.node:
                        if n.name == tfnode.input[-1]:
                            for n_shape in self.tf_graph.node:
                                if n_shape.name == n.input[1]:
                                    reshape_list = list(np.frombuffer(n_shape.attr['value'].tensor.tensor_content, dtype=np.int32))
                                    new_reshape = []
                                    for i in order:
                                        new_reshape.append(reshape_list[i])
                                    n_shape.attr['value'].tensor.tensor_content = np.asarray(new_reshape, dtype=np.int32).tobytes()
            elif nnef_node_param.op == 'output_val':
                base_name = nnef_node_param.parameters['base_node'].name[:nnef_node_param.parameters['base_node'].name.find(',')]
                if base_name[0] == '[':
                    base_name = base_name[1:]
                if nnef_node_param.parameters['base_index'] == 0:
                    tfnode.input.extend([self.format_name(base_name)])
                else:
                    name = self.format_name(base_name) + ':' + str(nnef_node_param.parameters['base_index'])
                    tfnode.input.extend([name])
            else:
                tfnode.input.extend([self.format_name(nnef_node_param.name)])

            if isinstance(nnef_node.parameters[param_name], list):
                index += 1
                if index >= len(nnef_node.parameters[param_name]):
                    nnef_node_param = None
                else:
                    nnef_node_param = nnef_node.parameters[param_name][index]
            else:
                nnef_node_param = None

    def generate_tf_graph(self, ):
        self.tf_graph = graph_pb2.GraphDef()

        for nnef_node, data in self.nxgraph.nodes(data=True):
            if 'node' in data:
                nnef_node = data['node']
                if nnef_node.name:
                    if hasattr(self, "export_" + nnef_node.op):
                        func = getattr(self, "export_" + nnef_node.op)
                        func(nnef_node)
                    else:
                        self.export_UNKNOWN(nnef_node)
            else:
                print('WARNING: nnef_node missing from op: ', nnef_node)

        network_dir, model_filename = os.path.split(self.output_model)
        if not os.path.exists(network_dir):
            os.makedirs(network_dir)
        with open(self.output_model, "wb") as f:
            f.write(self.tf_graph.SerializeToString())

    def export_UNKNOWN(self, nnef_node):
        print(nnef_node.op + " is currently not supported!\n")
        input()

    def export_abs(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Abs')
        self.add_input(tfnode, nnef_node, 'x')

        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_add(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Add')
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_and(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='LogicalAnd')
        #Going to be issues with type
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')

        self.tf_graph.node.extend([tfnode])

    def export_area_downsample(self, nnef_node):
        tfnode_shape = NodeDef(name=self.format_name(nnef_node.name) + '/size', op='Const')
        tfnode_shape.attr['dtype'].type = 3
        tfnode_shape.attr['value'].tensor.dtype = 3
        output_size = np.asarray(nnef_node.output_shape[2:], dtype=np.int32)
        tfnode_shape.attr['value'].tensor.tensor_shape.dim.add().size = len(output_size)
        tfnode_shape.attr['value'].tensor.tensor_content = output_size.tobytes()
        self.tf_graph.node.extend([tfnode_shape])

        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='ResizeArea')
        self.add_input(tfnode, nnef_node, 'input')
        node.input.extend([tfnode_shape.name])
        tfnode.attr['T'].type = 1
        tfnode.attr['align_corners'].b = False

        self.tf_graph.node.extend([tfnode])

    def export_avg_pool(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='AvgPool')
        self.add_input(tfnode, nnef_node, 'input')

        tfnode.attr['T'].type = 1
        tfnode.attr['data_format'].s = b'NHWC'
        sizes = nnef_node.parameters['size']
        nhwc_sizes = [sizes[0]]
        for i in range(2, len(sizes)):
            nhwc_sizes.append(sizes[i])
        nhwc_sizes.append(sizes[1])
        tfnode.attr['ksize'].list.i.extend(nhwc_sizes)

        dilations = nnef_node.parameters['dilation']
        if dilations != [] and dilations != [1]*4:
            raise ValueError("TensorFlow does not support dilated pooling")

        strides = nnef_node.parameters['stride']
        if strides == []:
            nhwc_strides = [1]*4
        else:
            nhwc_strides = [strides[0]]
            for i in range(2, len(strides)):
                nhwc_strides.append(strides[i])
            nhwc_strides.append(strides[1])
        tfnode.attr['strides'].list.i.extend(nhwc_strides)

        if nnef_node.parameters['padding'] == []:
            tfnode.attr['padding'].s = 'SAME'.encode('utf-8')
        elif(nnef_node.parameters['padding'] == [(0, 0), (0, 0), (0, 0), (0, 0)]):
            tfnode.attr['padding'].s = 'VALID'.encode('utf-8')
        else:
            pads = nnef_node.parameters['padding']
            padding = [[pads[0][0], pads[0][1]]]
            for i in range(2, len(pads)):
                padding = padding + [[pads[i][0], pads[i][1]]]
            padding = padding + [[pads[1][0], pads[1][1]]]
            padding = np.asarray(padding, dtype=np.int32)
            tfnode_pad_const = NodeDef(name=self.format_name(nnef_node.name) + "/Pad/paddings", op='Const')
            tfnode_pad_const.attr['dtype'].type = 3
            tfnode_pad_const.attr['value'].tensor.dtype = 3
            for size in np.shape(padding):
                tfnode_pad_const.attr['value'].tensor.tensor_shape.dim.add().size = size
            tfnode_pad_const.attr['value'].tensor.tensor_content = padding.tobytes()
            self.tf_graph.node.extend([tfnode_pad_const])

            tfnode_pad = NodeDef(name=self.format_name(nnef_node.name) + "/Pad", op='Pad')
            tfnode_pad.input.extend([tfnode.input[0]])
            tfnode_pad.input.extend([tfnode_pad_const.name])
            tfnode_pad.attr['T'].type = 1
            tfnode_pad.attr['Tpaddings'].type = 3
            self.tf_graph.node.extend([tfnode_pad])

            tfnode.input[0] = tfnode_pad.name
            tfnode.attr['padding'].s = 'VALID'.encode('utf-8')

        self.tf_graph.node.extend([tfnode])

    def export_batch_normalization(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='FusedBatchNorm')
        self.add_input(tfnode, nnef_node, 'input', [0, 2, 3, 1])
        self.add_input(tfnode, nnef_node, 'scale', [1])
        self.add_input(tfnode, nnef_node, 'offset', [1])
        self.add_input(tfnode, nnef_node, 'mean', [1])
        self.add_input(tfnode, nnef_node, 'variance', [1])

        tfnode.attr['T'].type = 1
        tfnode.attr['data_format'].s = b'NHWC'
        tfnode.attr['epsilon'].f = nnef_node.parameters['epsilon']
        tfnode.attr['is_training'].b = False

        self.tf_graph.node.extend([tfnode])

    def export_ceil(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Ceil')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_concat(self, nnef_node):
        tfnode_axis = NodeDef(name=self.format_name(nnef_node.name) + "/axis", op="Const")
        tfnode_axis.attr['dtype'].type = 3
        tfnode_axis.attr['value'].tensor.dtype = 3
        tfnode_axis.attr['value'].tensor.tensor_shape.dim.extend([])
        if len(nnef_node.output_shape) == 4:
            tfnode_axis.attr['value'].tensor.int_val.extend([self.mapping[nnef_node.parameters['axis']]])
        else:
            tfnode_axis.attr['value'].tensor.int_val.extend([nnef_node.parameters['axis']])
        self.tf_graph.node.extend([tfnode_axis])

        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='ConcatV2')
        self.add_input(tfnode, nnef_node, 'values')
        tfnode.input.append((self.format_name(nnef_node.name) + '/axis').encode('utf-8'))
        tfnode.attr['N'].i = len(nnef_node.parameters['values'])
        tfnode.attr['T'].type = 1
        tfnode.attr['Tidx'].type = 3

        self.tf_graph.node.extend([tfnode])

    def export_constant(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Const')
        tfnode.attr['dtype'].type = 1
        tfnode.attr['value'].tensor.dtype = 1
        tfnode.attr['value'].tensor.float_val.extend([nnef_node.parameters['value'][0]])
        if len(nnef_node.parameters['shape']) == 2 and nnef_node.parameters['shape'][0] == 1:
            nnef_node.parameters['shape'].pop(0)
        for i in range(len(nnef_node.parameters['shape'])):
            tfnode.attr['value'].tensor.tensor_shape.dim.add().size = nnef_node.parameters['shape'][i]
        self.tf_graph.node.extend([tfnode])

    def export_conv(self, nnef_node):
        conv_len = len(nnef_node.parameters['input'].output_shape)
        if(conv_len == 4) and nnef_node.parameters['groups'] == nnef_node.parameters['input'].output_shape[1]:
            return self.export_planewise_conv(nnef_node)

        assert nnef_node.parameters['groups'] == 1, "TensorFlow does not support grouped convolutions currently."
        if conv_len == 4:
            tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Conv2D')
            self.add_input(tfnode, nnef_node, 'input', [0, 2, 3, 1])
            self.add_input(tfnode, nnef_node, 'filter', [2, 3, 1, 0])
            tfnode.attr['data_format'].s = 'NHWC'.encode('utf-8')
            tfnode.attr['use_cudnn_on_gpu'].b = True
        elif conv_len == 5:
            tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Conv3D')
            self.add_input(tfnode, nnef_node, 'input', [0, 2, 3, 4, 1])
            self.add_input(tfnode, nnef_node, 'filter', [2, 3, 4, 1, 0])
            tfnode.attr['data_format'].s = 'NDHWC'.encode('utf-8')
        else:
            raise ValueError("Cannot handle input_size " + str(conv_len) + " yet.")

        tfnode.attr['T'].type = 1

        if 'dilation' in nnef_node.parameters:
            dilations = nnef_node.parameters['dilation']
            if dilations:
                nhwc_dilations = [1]
                for i in range(0, len(dilations)):
                    nhwc_dilations.append(dilations[i])
                nhwc_dilations.append(1)
            else:
                nhwc_dilations = [1]*conv_len
        else:
            nhwc_dilations = [1]*conv_len

        tfnode.attr['dilations'].list.i.extend(nhwc_dilations)

        strides = nnef_node.parameters['stride']
        if strides:
            nhwc_strides = [1]
            for i in range(0, len(strides)):
                nhwc_strides.append(strides[i])
            nhwc_strides.append(1)
        else:
            nhwc_strides = [1]*conv_len

        tfnode.attr['strides'].list.i.extend(nhwc_strides)

        if nnef_node.parameters['padding'] == []:
            tfnode.attr['padding'].s = 'SAME'.encode('utf-8')
        elif nnef_node.parameters['padding'] == [(0, 0)]*(conv_len-2):
            tfnode.attr['padding'].s = 'VALID'.encode('utf-8')
        else:
            pads = nnef_node.parameters['padding']
            padding = [[0, 0]]
            for pad in pads:
                padding = padding + [[pad[0], pad[1]]]
            padding = padding + [[0, 0]]
            padding = np.asarray(padding, dtype=np.int32)

            tfnode_pad_const = NodeDef(name=self.format_name(nnef_node.name) + "/Pad/paddings", op='Const')
            tfnode_pad_const.attr['dtype'].type = 3
            tfnode_pad_const.attr['value'].tensor.dtype = 3
            for size in np.shape(padding):
                tfnode_pad_const.attr['value'].tensor.tensor_shape.dim.add().size = size
            tfnode_pad_const.attr['value'].tensor.tensor_content = padding.tobytes()
            self.tf_graph.node.extend([tfnode_pad_const])

            tfnode_pad = NodeDef(name=self.format_name(nnef_node.name) + "/Pad", op='Pad')
            tfnode_pad.input.extend([tfnode.input[0]])
            tfnode_pad.input.extend([tfnode_pad_const.name])
            tfnode_pad.attr['T'].type = 1
            tfnode_pad.attr['Tpaddings'].type = 3
            self.tf_graph.node.extend([tfnode_pad])

            tfnode.input[0] = tfnode_pad.name
            tfnode.attr['padding'].s = 'VALID'.encode('utf-8')

        if('bias' in nnef_node.parameters and nnef_node.parameters['bias'] != 0):
            tfnode.name = tfnode.name + '/conv'
            self.tf_graph.node.extend([tfnode])
            tfnode_add = NodeDef(name=self.format_name(nnef_node.name), op='Add')
            tfnode_add.input.extend([tfnode.name])
            self.add_input(tfnode_add, nnef_node, 'bias')
            tfnode_add.attr['T'].type = 1
            self.tf_graph.node.extend([tfnode_add])
        else:
            self.tf_graph.node.extend([tfnode])

    def export_deconv(self, nnef_node):
        assert nnef_node.parameters['groups'] == 1, "TensorFlow does not support grouped convolutions currently."

        conv_len = len(nnef_node.output_shape)
        tfnode_const = NodeDef(name=self.format_name(nnef_node.name) + '/output_shape', op='Const')
        tfnode_const.attr['dtype'].type = 3
        tfnode_const.attr['value'].tensor.dtype = 3
        tfnode_const.attr['value'].tensor.tensor_shape.dim.add().size = conv_len
        new_out_shape = [nnef_node.output_shape[0]]
        for i in range(2, conv_len):
            new_out_shape.append(nnef_node.output_shape[i])
        new_out_shape.append(nnef_node.output_shape[1])
        tfnode_const.attr['value'].tensor.tensor_content = np.asarray(new_out_shape, dtype=np.int32).tobytes()

        self.tf_graph.node.extend([tfnode_const])

        if conv_len == 4:
            tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Conv2DBackpropInput')
            tfnode.input.extend([tfnode_const.name])
            self.add_input(tfnode, nnef_node, 'filter', [2, 3, 1, 0])
            self.add_input(tfnode, nnef_node, 'input', [0, 2, 3, 1])
            tfnode.attr['data_format'].s = 'NHWC'.encode('utf-8')
        elif conv_len == 5:
            tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Conv3DBackpropInputV2')
            tfnode.input.extend([tfnode_const.name])
            self.add_input(tfnode, nnef_node, 'filter', [2, 3, 4, 1, 0])
            self.add_input(tfnode, nnef_node, 'input', [0, 2, 3, 4, 1])
            tfnode.attr['data_format'].s = 'NDHWC'.encode('utf-8')
        else:
            raise ValueError("Cannot handle input_size " + str(conv_len) + " yet.")

        tfnode.attr['T'].type = 1

        if 'dilation' in nnef_node.parameters:
            dilations = nnef_node.parameters['dilation']
            if dilations:
                nhwc_dilations = [1]
                for i in range(0, len(dilations)):
                    nhwc_dilations.append(dilations[i])
                nhwc_dilations.append(1)
            else:
                nhwc_dilations = [1]*conv_len
        else:
            nhwc_dilations = [1]*conv_len

        tfnode.attr['dilations'].list.i.extend(nhwc_dilations)

        strides = nnef_node.parameters['stride']
        if strides:
            nhwc_strides = [1]
            for i in range(0, len(strides)):
                nhwc_strides.append(strides[i])
            nhwc_strides.append(1)
        else:
            nhwc_strides = [1]*conv_len

        tfnode.attr['strides'].list.i.extend(nhwc_strides)

        if conv_len == 4:
            tfnode.attr['use_cudnn_on_gpu'].b = True

        if nnef_node.parameters['padding'] == []:
            tfnode.attr['padding'].s = 'SAME'.encode('utf-8')
        elif nnef_node.parameters['padding'] == [(0, 0)]*(conv_len-2):
            tfnode.attr['padding'].s = 'VALID'.encode('utf-8')
        else:
            pads = nnef_node.parameters['padding']
            padding = [[0, 0]]
            for pad in pads:
                padding = padding + [[pad[0], pad[1]]]
            padding = padding + [[0, 0]]
            padding = np.asarray(padding, dtype=np.int32)

            tfnode_pad_const = NodeDef(name=self.format_name(nnef_node.name) + "/Pad/paddings", op='Const')
            tfnode_pad_const.attr['dtype'].type = 3
            tfnode_pad_const.attr['value'].tensor.dtype = 3
            for size in np.shape(padding):
                tfnode_pad_const.attr['value'].tensor.tensor_shape.dim.add().size = size
            tfnode_pad_const.attr['value'].tensor.tensor_content = padding.tobytes()
            self.tf_graph.node.extend([tfnode_pad_const])

            tfnode_pad = NodeDef(name=self.format_name(nnef_node.name) + "/Pad", op='Pad')
            tfnode_pad.input.extend([tfnode.input[0]])
            tfnode_pad.input.extend([tfnode_pad_const.name])
            tfnode_pad.attr['T'].type = 1
            tfnode_pad.attr['Tpaddings'].type = 3
            self.tf_graph.node.extend([tfnode_pad])

            tfnode.input[0] = tfnode_pad.name
            tfnode.attr['padding'].s = 'VALID'.encode('utf-8')

        if('bias' in nnef_node.parameters and nnef_node.parameters['bias'] != 0):
            tfnode.name = tfnode.name + '/conv'
            self.tf_graph.node.extend([tfnode])
            tfnode_add = NodeDef(name=self.format_name(nnef_node.name), op='Add')
            tfnode_add.input.extend([tfnode.name])
            self.add_input(tfnode_add, nnef_node, 'bias')
            tfnode_add.attr['T'].type = 1
            self.tf_graph.node.extend([tfnode_add])

        else:
            self.tf_graph.node.extend([tfnode])

    def export_div(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='RealDiv')
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_elu(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Elu')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_eq(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Equal')
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_exp(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Exp')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_external(self, nnef_node):
        tfnode = NodeDef(name=nnef_node.name, op='Placeholder')

        tfnode.attr['dtype'].type = 1
        tfnode.attr['shape'].shape.dim.add().size = nnef_node.parameters['shape'][0]
        for i in range(2, len(nnef_node.parameters['shape'])):
            tfnode.attr['shape'].shape.dim.add().size = nnef_node.parameters['shape'][i]
        tfnode.attr['shape'].shape.dim.add().size = nnef_node.parameters['shape'][1]
        self.tf_graph.node.extend([tfnode])

    def export_floor(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Floor')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_ge(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='GreaterEqual')
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_gt(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Greater')
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_le(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='LessEqual')
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_linear(self, nnef_node):
        input_shape = nnef_node.parameters['input'].output_shape
        if len(input_shape) > 2:
            i = 1
            squeeze_dims = []
            for j in range(2, len(input_shape)):
                squeeze_dims.append(i)
                i += 1

            tfnode_squeeze = NodeDef(name=self.format_name(nnef_node.name) + '/squeeze', op='Squeeze')
            self.add_input(tfnode_squeeze, nnef_node, 'input')
            tfnode_squeeze.attr['T'].type = 1
            tfnode_squeeze.attr['squeeze_dims'].list.i.extend(squeeze_dims)
            self.tf_graph.node.extend([tfnode_squeeze])

        tfnode = NodeDef(name=self.format_name(nnef_node.name) + '/linear', op='MatMul')
        if len(input_shape) > 2:
            tfnode.input.extend([tfnode_squeeze.name])
        else:
            self.add_input(tfnode, nnef_node, 'input')
        self.add_input(tfnode, nnef_node, 'filter')

        tfnode.attr['T'].type = 1
        tfnode.attr['transpose_a'].b = False
        tfnode.attr['transpose_b'].b = True

        self.tf_graph.node.extend([tfnode])

        tfnode_add = NodeDef(name=self.format_name(nnef_node.name), op='Add')
        tfnode_add.input.extend([tfnode.name])
        self.add_input(tfnode_add, nnef_node, 'bias')
        tfnode_add.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode_add])

    def export_local_response_normalization(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='LRN')
        self.add_input(tfnode, nnef_node, 'input')
        tfnode.attr['T'].type = 1
        tfnode.attr['alpha'].f = nnef_node.parameters['alpha']
        tfnode.attr['beta'].f = nnef_node.parameters['beta']
        tfnode.attr['bias'].f = nnef_node.parameters['bias']
        tfnode.attr['depth_radius'].i = nnef_node.parameters['size'][1]

        self.tf_graph.node.extend([tfnode])

    def export_log(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Log')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_lt(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Less')
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_l2_normalization(self, nnef_node):
        assert(nnef_node.parameters['bias'] == 0.0), "TensorFlow cannot handle non-zero bias for op: l2_normalize"

        tfnode_square = NodeDef(name=self.format_name(nnef_node.name) + '/Square', op='Square')
        self.add_input(tfnode_square, nnef_node, 'input')
        tfnode_square.attr['T'].type = 1
        self.tf_graph.node.extend([tfnode_square])

        tfnode_const = NodeDef(name=self.format_name(nnef_node.name) + '/Const', op='Const')
        tfnode_const.attr['dtype'].type = 3
        tfnode_const.attr['value'].tensor.tensor_shape.dim.add().size = len(nnef_node.parameters['axes'])
        tfnode_const.attr['value'].tensor.tensor_content = np.asarray(nnef_node.parameters['axes'], dtype=np.int32).tobytes()
        self.tf_graph.node.extend([tfnode_const])

        tfnode_sum = NodeDef(name=self.format_name(nnef_node.name) + '/Sum', op='Sum')
        tfnode_sum.input.extend([tfnode_square.name])
        tfnode_sum.input.extend([tfnode_const.name])
        tfnode_sum.attr['T'].type = 1
        tfnode_sum.attr['Tidx'].type = 3
        tfnode_sum.attr['keep_dims'].b = True
        self.tf_graph.node.extend([tfnode_sum])

        tfnode_max_y = NodeDef(name=self.format_name(nnef_node.name) + '/Maximum/y', op='Const')
        tfnode_max_y.attr['dtype'].type = 1
        tfnode_max_y.attr['value'].tensor.dtype = 1
        tfnode_max_y.attr['value'].tensor.float_val.extend([nnef_node.parameters['epsilon']])
        self.tf_graph.node.extend([tfnode_max_y])

        tfnode_max = NodeDef(name=self.format_name(nnef_node.name) + '/Maximum', op='Maximum')
        tfnode_max.input.extend([tfnode_sum.name])
        tfnode_max.input.extend([tfnode_max_y.name])
        tfnode_max.attr['T'].type = 1
        self.tf_graph.node.extend([tfnode_max])

        tfnode_rsqrt = NodeDef(name=self.format_name(nnef_node.name) + '/Rsqrt', op='Rsqrt')
        tfnode_rsqrt.input.extend([tfnode_max.name])
        tfnode_rsqrt.attr['T'].type = 1
        self.tf_graph.node.extend([tfnode_rsqrt])

        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Mul')
        self.add_input(tfnode, nnef_node, 'input')
        tfnode.input.extend([tfnode_rsqrt.name])
        tfnode.attr['T'].type = 1
        self.tf_graph.node.extend([tfnode])

    def export_matmul(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='MatMul')
        self.add_input(tfnode, nnef_node, 'A')
        self.add_input(tfnode, nnef_node, 'B')

        tfnode.attr['T'].type = 1
        tfnode.attr['transpose_a'].b = nnef_node.parameters['transposeA']
        tfnode.attr['transpose_b'].b = nnef_node.parameters['transposeB']

        self.tf_graph.node.extend([tfnode])

    def export_max(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Maximum')
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_max_pool(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='MaxPool')
        self.add_input(tfnode, nnef_node, 'input')

        tfnode.attr['T'].type = 1
        tfnode.attr['data_format'].s = 'NHWC'.encode('utf-8')
        sizes = nnef_node.parameters['size']
        nhwc_sizes = [sizes[0]]
        for i in range(2, len(sizes)):
            nhwc_sizes.append(sizes[i])
        nhwc_sizes.append(sizes[1])
        tfnode.attr['ksize'].list.i.extend(nhwc_sizes)

        dilations = nnef_node.parameters['dilation']
        if dilations != [] and dilations != [1]*4:
            raise ValueError("TensorFlow does not support dilated pooling")

        strides = nnef_node.parameters['stride']
        nhwc_strides = [strides[0]]
        for i in range(2, len(strides)):
            nhwc_strides.append(strides[i])
        nhwc_strides.append(strides[1])
        tfnode.attr['strides'].list.i.extend(nhwc_strides)

        if nnef_node.parameters['padding'] == []:
            tfnode.attr['padding'].s = 'SAME'.encode('utf-8')
        elif(nnef_node.parameters['padding'] == [(0, 0), (0, 0), (0, 0), (0, 0)]):
            tfnode.attr['padding'].s = 'VALID'.encode('utf-8')
        else:
            pads = nnef_node.parameters['padding']
            padding = [[pads[0][0], pads[0][1]]]
            for i in range(2, len(pads)):
                padding = padding + [[pads[i][0], pads[i][1]]]
            padding = padding + [[pads[1][0], pads[1][1]]]
            padding = np.asarray(padding, dtype=np.int32)

            tfnode_pad_const = NodeDef(name=self.format_name(nnef_node.name) + "/Pad/paddings", op='Const')
            tfnode_pad_const.attr['dtype'].type = 3
            tfnode_pad_const.attr['value'].tensor.dtype = 3
            for size in np.shape(padding):
                tfnode_pad_const.attr['value'].tensor.tensor_shape.dim.add().size = size
            tfnode_pad_const.attr['value'].tensor.tensor_content = padding.tobytes()
            self.tf_graph.node.extend([tfnode_pad_const])

            tfnode_pad = NodeDef(name=self.format_name(nnef_node.name) + "/Pad", op='Pad')
            tfnode_pad.input.extend([tfnode.input[0]])
            tfnode_pad.input.extend([tfnode_pad_const.name])
            tfnode_pad.attr['T'].type = 1
            tfnode_pad.attr['Tpaddings'].type = 3
            self.tf_graph.node.extend([tfnode_pad])

            tfnode.input[0] = tfnode_pad.name
            tfnode.attr['padding'].s = 'VALID'.encode('utf-8')

        self.tf_graph.node.extend([tfnode])

    def export_max_pool_with_index(self, nnef_node):
        name = nnef_node.name[:nnef_node.name.find(',')]
        tfnode = NodeDef(name=self.format_name(name), op='MaxPoolWithArgmax')
        self.add_input(tfnode, nnef_node, 'input')

        tfnode.attr['T'].type = 1
        tfnode.attr['Targmax'].type = 3
        sizes = nnef_node.parameters['size']
        nhwc_sizes = [sizes[0]]
        for i in range(2, len(sizes)):
            nhwc_sizes.append(sizes[i])
        nhwc_sizes.append(sizes[1])
        tfnode.attr['ksize'].list.i.extend(nhwc_sizes)

        dilations = nnef_node.parameters['dilation']
        if dilations != [] and dilations != [1]*4:
            raise ValueError("TensorFlow does not support dilated pooling")

        strides = nnef_node.parameters['stride']
        nhwc_strides = [strides[0]]
        for i in range(2, len(strides)):
            nhwc_strides.append(strides[i])
        nhwc_strides.append(strides[1])
        tfnode.attr['strides'].list.i.extend(nhwc_strides)

        if nnef_node.parameters['padding'] == []:
            tfnode.attr['padding'].s = 'SAME'.encode('utf-8')
        elif(nnef_node.parameters['padding'] == [(0, 0), (0, 0), (0, 0), (0, 0)]):
            tfnode.attr['padding'].s = 'VALID'.encode('utf-8')
        else:
            pads = nnef_node.parameters['padding']
            padding = [[pads[0][0], pads[0][1]]]
            for i in range(2, len(pads)):
                padding = padding + [[pads[i][0], pads[i][1]]]
            padding = padding + [[pads[1][0], pads[1][1]]]
            padding = np.asarray(padding, dtype=np.int32)

            tfnode_pad_const = NodeDef(name=self.format_name(nnef_node.name) + "/Pad/paddings", op='Const')
            tfnode_pad_const.attr['dtype'].type = 3
            tfnode_pad_const.attr['value'].tensor.dtype = 3
            for size in np.shape(padding):
                tfnode_pad_const.attr['value'].tensor.tensor_shape.dim.add().size = size
            tfnode_pad_const.attr['value'].tensor.tensor_content = padding.tobytes()
            self.tf_graph.node.extend([tfnode_pad_const])

            tfnode_pad = NodeDef(name=self.format_name(nnef_node.name) + "/Pad", op='Pad')
            tfnode_pad.input.extend([tfnode.input[0]])
            tfnode_pad.input.extend([tfnode_pad_const.name])
            tfnode_pad.attr['T'].type = 1
            tfnode_pad.attr['Tpaddings'].type = 3
            self.tf_graph.node.extend([tfnode_pad])

            tfnode.input[0] = tfnode_pad.name
            tfnode.attr['padding'].s = 'VALID'.encode('utf-8')

        self.tf_graph.node.extend([tfnode])

    def export_max_reduce(self, nnef_node):
        tfnode_axes = NodeDef(name=self.format_name(nnef_node.name) + '/axes', op='Const')
        tfnode_axes.attr['dtype'].type = 3
        array_length = len(nnef_node.parameters['axes'])
        new_axes = []
        for axis in nnef_node.parameters['axes']:
            new_axes.append(self.mapping[axis])

        tfnode_axes.attr['value'].tensor.dtype = 3
        tfnode_axes.attr['value'].tensor.tensor_shape.dim.add().size = array_length
        if array_length == 1:
            tfnode_axes.attr['value'].tensor.int_val.extend([new_axes[0]])
        else:
            tfnode_axes.attr['value'].tensor.tensor_content = np.asarray(new_axes, dtype=np.int32).tobytes()

        self.tf_graph.node.extend([tfnode_axes])

        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Max')
        self.add_input(tfnode, nnef_node, 'input')
        tfnode.input.extend([tfnode_axes.name])
        tfnode.attr['T'].type = 1
        tfnode.attr['Tidx'].type = 3
        tfnode.attr['keep_dims'].b = False

        self.tf_graph.node.extend([tfnode])

    def export_mean_reduce(self, nnef_node):
        tfnode_axes = NodeDef(name=self.format_name(nnef_node.name) + '/axes', op='Const')
        tfnode_axes.attr['dtype'].type = 3
        array_length = len(nnef_node.parameters['axes'])
        new_axes = []
        for axis in nnef_node.parameters['axes']:
            new_axes.append(self.mapping[axis])

        tfnode_axes.attr['value'].tensor.dtype = 3
        tfnode_axes.attr['value'].tensor.tensor_shape.dim.add().size = array_length
        if array_length == 1:
            tfnode_axes.attr['value'].tensor.int_val.extend([new_axes[0]])
        else:
            tfnode_axes.attr['value'].tensor.tensor_content = np.asarray(new_axes, dtype=np.int32).tobytes()

        self.tf_graph.node.extend([tfnode_axes])

        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Mean')
        self.add_input(tfnode, nnef_node, 'input')
        tfnode.input.extend([tfnode_axes.name])
        tfnode.attr['T'].type = 1
        tfnode.attr['Tidx'].type = 3
        tfnode.attr['keep_dims'].b = False

        self.tf_graph.node.extend([tfnode])

    def export_min(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Minimum')
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_mul(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Mul')
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_multilinear_upsample(self, nnef_node):
        tfnode_shape = NodeDef(name=self.format_name(nnef_node.name) + '/size', op='Const')
        tfnode_shape.attr['dtype'].type = 3
        tfnode_shape.attr['value'].tensor.dtype = 3
        output_size = np.asarray(nnef_node.output_shape[2:], dtype=np.int32)
        tfnode_shape.attr['value'].tensor.tensor_shape.dim.add().size = len(output_size)
        tfnode_shape.attr['value'].tensor.tensor_content = output_size.tobytes()
        self.tf_graph.node.extend([tfnode_shape])

        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='ResizeBilinear')
        self.add_input(tfnode, nnef_node, 'input')
        tfnode.input.extend([tfnode_shape.name])
        tfnode.attr['T'].type = 1
        tfnode.attr['align_corners'].b = False

        self.tf_graph.node.extend([tfnode])

    def export_ne(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='NotEqual')
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_nearest_downsample(self, nnef_node):
        tfnode_shape = NodeDef(name=self.format_name(nnef_node.name) + '/size', op='Const')
        tfnode_shape.attr['dtype'].type = 3
        tfnode_shape.attr['value'].tensor.dtype = 3
        output_size = np.asarray(nnef_node.output_shape[2:], dtype=np.int32)
        tfnode_shape.attr['value'].tensor.tensor_shape.dim.add().size = len(output_size)
        tfnode_shape.attr['value'].tensor.tensor_content = output_size.tobytes()

        self.tf_graph.node.extend([tfnode_shape])

        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='ResizeNearestNeighbor')
        self.add_input(tfnode, nnef_node, 'input')
        tfnode.input.extend([tfnode_shape.name])
        tfnode.attr['T'].type = 1
        tfnode.attr['align_corners'].b = False

        self.tf_graph.node.extend([tfnode])

    def export_nearest_upsample(self, nnef_node):
        tfnode_shape = NodeDef(name=self.format_name(nnef_node.name) + '/size', op='Const')
        tfnode_shape.attr['dtype'].type = 3
        tfnode_shape.attr['value'].tensor.dtype = 3
        output_size = np.asarray(nnef_node.output_shape[2:], dtype=np.int32)
        tfnode_shape.attr['value'].tensor.tensor_shape.dim.add().size = len(output_size)
        tfnode_shape.attr['value'].tensor.tensor_content = output_size.tobytes()

        self.tf_graph.node.extend([tfnode_shape])

        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='ResizeNearestNeighbor')
        self.add_input(tfnode, nnef_node, 'input')
        tfnode.input.extend([tfnode_shape.name])
        tfnode.attr['T'].type = 1
        tfnode.attr['align_corners'].b = False

        self.tf_graph.node.extend([tfnode])

    def export_neg(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Neg')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_not(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='LogicalNot')
        self.add_input(tfnode, nnef_node, 'x')

        self.tf_graph.node.extend([tfnode])

    def export_or(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='LogicalOr')
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')

        self.tf_graph.node.extend([tfnode])

    def export_output_val(self, nnef_node):
        return

    def export_planewise_conv(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='DepthwiseConv2dNative')
        self.add_input(tfnode, nnef_node, 'input', [0, 2, 3, 1])

        #Converts to Tensorflow format of [height, width, channels in, channel multiplier]
        if nnef_node.parameters['input'].output_shape[1] != nnef_node.parameters['filter'].parameters['shape'][0]:
            filter_node = nnef_node.parameters['filter']
            np_array_read = filter_node.get_tensordatafile().get_data().get_array()[0]
            np_array_read = np.reshape(np_array_read, filter_node.parameters['shape'])
            new_shape = [nnef_node.parameters['input'].output_shape[1]]
            new_shape.append(int(filter_node.output_shape[0]/new_shape[0]))
            new_shape += filter_node.parameters['shape'][2:]
            filter_node.parameters['shape'] = new_shape
            np_array_read = np.reshape(np_array_read, new_shape)
            filter_node.get_tensordatafile().get_data().set_array(np_array_read, override=True)
        self.add_input(tfnode, nnef_node, 'filter', [2, 3, 0, 1])

        tfnode.attr['T'].type = 1
        tfnode.attr['data_format'].s = 'NHWC'.encode('utf-8')

        if 'dilation' in nnef_node.parameters:
            dilations = nnef_node.parameters['dilation']
            if dilations:
                nhwc_dilations = [1]
                for i in range(0, len(dilations)):
                    nhwc_dilations.append(dilations[i])
                nhwc_dilations.append(1)
            else:
                nhwc_dilations = [1]*4
        else:
            nhwc_dilations = [1]*4

        tfnode.attr['dilations'].list.i.extend(nhwc_dilations)

        strides = nnef_node.parameters['stride']
        if strides:
            nhwc_strides = [1]
            for i in range(0, len(strides)):
                nhwc_strides.append(strides[i])
            nhwc_strides.append(1)
        else:
            nhwc_strides = [1]*4

        tfnode.attr['strides'].list.i.extend(nhwc_strides)

        if nnef_node.parameters['padding'] == []:
            tfnode.attr['padding'].s = 'SAME'.encode('utf-8')
        elif nnef_node.parameters['padding'] == [(0, 0)]*(2):
            tfnode.attr['padding'].s = 'VALID'.encode('utf-8')
        else:
            pads = nnef_node.parameters['padding']
            padding = [[0, 0]]
            for pad in pads:
                padding = padding + [[pad[0], pad[1]]]
            padding = padding + [[0, 0]]
            padding = np.asarray(padding, dtype=np.int32)

            tfnode_pad_const = NodeDef(name=self.format_name(nnef_node.name) + "/Pad/paddings", op='Const')
            tfnode_pad_const.attr['dtype'].type = 3
            tfnode_pad_const.attr['value'].tensor.dtype = 3
            for size in np.shape(padding):
                tfnode_pad_const.attr['value'].tensor.tensor_shape.dim.add().size = size
            tfnode_pad_const.attr['value'].tensor.tensor_content = padding.tobytes()
            self.tf_graph.node.extend([tfnode_pad_const])

            tfnode_pad = NodeDef(name=self.format_name(nnef_node.name) + "/Pad", op='Pad')
            tfnode_pad.input.extend([tfnode.input[0]])
            tfnode_pad.input.extend([tfnode_pad_const.name])
            tfnode_pad.attr['T'].type = 1
            tfnode_pad.attr['Tpaddings'].type = 3
            self.tf_graph.node.extend([tfnode_pad])

            tfnode.input[0] = tfnode_pad.name
            tfnode.attr['padding'].s = 'VALID'.encode('utf-8')

        if('bias' in nnef_node.parameters and nnef_node.parameters['bias'] != 0):
            tfnode.name = tfnode.name + '/conv'
            self.tf_graph.node.extend([tfnode])
            tfnode_add = NodeDef(name=self.format_name(nnef_node.name), op='Add')
            tfnode_add.input.extend([tfnode.name])
            self.add_input(tfnode_add, nnef_node, 'bias')
            tfnode_add.attr['T'].type = 1
            self.tf_graph.node.extend([tfnode_add])
        else:
            self.tf_graph.node.extend([tfnode])

    def export_pow(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Pow')
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_relu(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Relu')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_reshape(self, nnef_node):
        tfnode_shape = NodeDef(name=self.format_name(nnef_node.name) + '/shape', op='Const')
        tfnode_shape.attr['dtype'].type = 3
        tfnode_shape.attr['value'].tensor.dtype = 3
        tfnode_shape.attr['value'].tensor.tensor_shape.dim.add().size = len(nnef_node.parameters['shape'])
        if len(nnef_node.parameters['shape']) == 1:
            tfnode_shape.attr['value'].tensor.int_val = nnef_node.parameters['shape'][0]
        else:
            shape_array = np.asarray(nnef_node.parameters['shape'])
            tfnode_shape.attr['value'].tensor.tensor_content = np.asarray(shape_array, dtype=np.int32).tobytes()

        self.tf_graph.node.extend([tfnode_shape])

        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Reshape')
        self.add_input(tfnode, nnef_node, 'input')
        tfnode.input.extend([self.format_name(nnef_node.name) + '/shape'])
        tfnode.attr['T'].type = 1
        tfnode.attr['Tshape'].type = 3

        self.tf_graph.node.extend([tfnode])

    def export_round(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Round')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_rsqrt(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Rsqrt')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_select(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Select')
        self.add_input(tfnode, nnef_node, 'condition')
        self.add_input(tfnode, nnef_node, 'true_value')
        self.add_input(tfnode, nnef_node, 'false_value')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_separable_conv(self, nnef_node):
        tfnode_depthwise = NodeDef(name=self.format_name(nnef_node.name) + '/depthwise', op='DepthwiseConv2dNative')
        self.add_input(tfnode_depthwise, nnef_node, 'input', [0, 2, 3, 1])

        #Converts to Tensorflow format of [height, width, channels in, channel multiplier]
        if nnef_node.parameters['input'].output_shape[1] != nnef_node.parameters['plane_filter'].parameters['shape'][0]:
            filter_node = nnef_node.parameters['plane_filter']
            np_array_read = filter_node.get_tensordatafile().get_data().get_array()[0]
            np_array_read = np.reshape(np_array_read, filter_node.parameters['shape'])
            new_shape = [nnef_node.parameters['input'].output_shape[1]]
            new_shape.append(int(filter_node.output_shape[0]/new_shape[0]))
            new_shape += filter_node.parameters['shape'][2:]
            filter_node.parameters['shape'] = new_shape
            np_array_read = np.reshape(np_array_read, new_shape)
            filter_node.get_tensordatafile().get_data().set_array(np_array_read, override=True)
        self.add_input(tfnode_depthwise, nnef_node, 'plane_filter', [2, 3, 0, 1])

        tfnode_depthwise.attr['T'].type = 1
        tfnode_depthwise.attr['data_format'].s = 'NHWC'.encode('utf-8')

        if 'dilation' in nnef_node.parameters:
            dilations = nnef_node.parameters['dilation']
            if dilations:
                nhwc_dilations = [1]
                for i in range(0, len(dilations)):
                    nhwc_dilations.append(dilations[i])
                nhwc_dilations.append(1)
            else:
                nhwc_dilations = [1]*4
        else:
            nhwc_dilations = [1]*4

        tfnode_depthwise.attr['dilations'].list.i.extend(nhwc_dilations)

        strides = nnef_node.parameters['stride']
        if strides:
            nhwc_strides = [1]
            for i in range(0, len(strides)):
                nhwc_strides.append(strides[i])
            nhwc_strides.append(1)
        else:
            nhwc_strides = [1]*4

        tfnode_depthwise.attr['strides'].list.i.extend(nhwc_strides)

        if nnef_node.parameters['padding'] == []:
            tfnode_depthwise.attr['padding'].s = 'SAME'.encode('utf-8')
        elif nnef_node.parameters['padding'] == [(0, 0)]*(2):
            tfnode_depthwise.attr['padding'].s = 'VALID'.encode('utf-8')
        else:
            pads = nnef_node.parameters['padding']
            padding = [[0, 0]]
            for pad in pads:
                padding = padding + [[pad[0], pad[1]]]
            padding = padding + [[0, 0]]
            padding = np.asarray(padding, dtype=np.int32)

            tfnode_pad_const = NodeDef(name=self.format_name(nnef_node.name) + "/depthwise/Pad/paddings", op='Const')
            tfnode_pad_const.attr['dtype'].type = 3
            tfnode_pad_const.attr['value'].tensor.dtype = 3
            for size in np.shape(padding):
                tfnode_pad_const.attr['value'].tensor.tensor_shape.dim.add().size = size
            tfnode_pad_const.attr['value'].tensor.tensor_content = padding.tobytes()
            self.tf_graph.node.extend([tfnode_pad_const])

            tfnode_pad = NodeDef(name=self.format_name(nnef_node.name) + "/depthwise/Pad", op='Pad')
            tfnode_pad.input.extend([tfnode_depthwise.input[0]])
            tfnode_pad.input.extend([tfnode_pad_const.name])
            tfnode_pad.attr['T'].type = 1
            tfnode_pad.attr['Tpaddings'].type = 3
            self.tf_graph.node.extend([tfnode_pad])

            tfnode_depthwise.input[0] = tfnode_pad.name
            tfnode_depthwise.attr['padding'].s = 'VALID'.encode('utf-8')

        self.tf_graph.node.extend([tfnode_depthwise])

        conv_len = len(nnef_node.parameters['input'].output_shape)
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Conv2D')
        tfnode.input.extend([tfnode_depthwise.name])
        self.add_input(tfnode, nnef_node, 'point_filter', [2, 3, 1, 0])
        tfnode.attr['data_format'].s = 'NHWC'.encode('utf-8')
        tfnode.attr['use_cudnn_on_gpu'].b = True

        tfnode.attr['T'].type = 1

        nhwc_dilations = [1]*conv_len
        tfnode.attr['dilations'].list.i.extend(nhwc_dilations)

        nhwc_strides = [1]*conv_len
        tfnode.attr['strides'].list.i.extend(nhwc_strides)

        tfnode.attr['padding'].s = 'SAME'.encode('utf-8')

        if('bias' in nnef_node.parameters and nnef_node.parameters['bias'] != 0):
            tfnode.name = tfnode.name + '/conv'
            self.tf_graph.node.extend([tfnode])
            tfnode_add = NodeDef(name=self.format_name(nnef_node.name), op='Add')
            tfnode_add.input.extend([tfnode.name])
            self.add_input(tfnode_add, nnef_node, 'bias')
            tfnode_add.attr['T'].type = 1
            self.tf_graph.node.extend([tfnode_add])
        else:
            self.tf_graph.node.extend([tfnode])

    def export_shape_of(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Shape')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1
        tfnode.attr['out_type'].type = 3

        self.tf_graph.node.extend([tfnode])

    def export_sigmoid(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Sigmoid')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_sign(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Sign')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

    def export_slice(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Slice')
        self.add_input(tfnode, nnef_node, 'input')
        tfnode.attr['T'].type = 1
        tfnode.attr['Index'].type = 3

        tfnode_begin = NodeDef(name=self.format_name(nnef_node.name) + '/begin', op='Const')
        tfnode_begin.attr['dtype'].type = 3
        tfnode_begin.attr['value'].tensor.dtype = 3
        tfnode_begin.attr['value'].tensor.tensor_shape.dim.add().size = len(nnef_node.parameters['begin'])
        begin_array = nnef_node.parameters['begin'][:]
        for i in range(len(begin_array)):
            if begin_array[i] == -1:
                begin_array[i] = 0
        if len(begin_array) == 1:
            tfnode_begin.attr['value'].tensor.int_val = begin_array[0]
        else:
            size_array = np.asarray(begin_array)
            tfnode_begin.attr['value'].tensor.tensor_content = np.asarray(size_array, dtype=np.int32).tobytes()
        self.tf_graph.node.extend([tfnode_begin])

        tfnode_size = NodeDef(name=self.format_name(nnef_node.name) + '/size', op='Const')
        tfnode_size.attr['dtype'].type = 3
        tfnode_size.attr['value'].tensor.dtype = 3
        tfnode_size.attr['value'].tensor.tensor_shape.dim.add().size = len(nnef_node.parameters['end'])
        end_array = nnef_node.parameters['end'][:]
        for i in range(len(end_array)):
            if end_array[i] == 0:
                end_array[i] = nnef_node.output_shape[i] - begin_array[i]
            else:
                end_array[i] = end_array[i] - begin_array[i]
        if len(end_array) == 1:
            tfnode_size.attr['value'].tensor.int_val = end_array[0]
        else:
            size_array = np.asarray(end_array)
            tfnode_size.attr['value'].tensor.tensor_content = np.asarray(size_array, dtype=np.int32).tobytes()
        self.tf_graph.node.extend([tfnode_size])

        tfnode.input.extend([tfnode_begin.name])
        tfnode.input.extend([tfnode_size.name])
        self.tf_graph.node.extend([tfnode])


    def export_softmax(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Softmax')

        if len(nnef_node.output_shape) != 2:
            tfnode_shape = NodeDef(name=self.format_name(nnef_node.name) + '/shape', op='Const')
            tfnode_shape.attr['dtype'].type = 3
            tfnode_shape.attr['value'].tensor.dtype = 3
            tfnode_shape.attr['value'].tensor.tensor_shape.dim.add().size = 2
            size = 1
            for i in range(1,len(nnef_node.output_shape)):
                size *= nnef_node.output_shape[i]
            new_shape = [nnef_node.output_shape[0], size]
            tfnode_shape.attr['value'].tensor.tensor_content = np.asarray(new_shape, dtype=np.int32).tobytes()

            self.tf_graph.node.extend([tfnode_shape])

            tfnode_reshape = NodeDef(name=self.format_name(nnef_node.name) + '/reshape', op='Reshape')
            self.add_input(tfnode_reshape, nnef_node, 'x')
            tfnode_reshape.input.extend([self.format_name(nnef_node.name) + '/shape'])
            tfnode_reshape.attr['T'].type = 1
            tfnode_reshape.attr['Tshape'].type = 3

            self.tf_graph.node.extend([tfnode_reshape])

            tfnode.input.extend([self.format_name(nnef_node.name) + '/reshape'])
        else:
            tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Softmax')
            self.add_input(tfnode, nnef_node, 'x')

        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_softplus(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Softplus')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_softsign(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Softsign')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_split(self, nnef_node):
        tfnode_name = nnef_node.name[1:nnef_node.name.find(',')]
        tfnode_const = NodeDef(name=self.format_name(tfnode_name) + '/split_dim', op='Const')
        tfnode_const.attr['dtype'].type = 3
        tfnode_const.attr['value'].tensor.dtype = 3
        tfnode_const.attr['value'].tensor.int_val.extend([nnef_node.parameters['axis']])

        self.tf_graph.node.extend([tfnode_const])

        tfnode = NodeDef(name=self.format_name(tfnode_name), op='Split')
        tfnode.input.extend([tfnode_const.name])
        self.add_input(tfnode, nnef_node, 'value')
        tfnode.attr['T'].type = 1
        tfnode.attr['num_split'].i = len(nnef_node.parameters['ratios'])

        self.tf_graph.node.extend([tfnode])

    def export_sqr(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Square')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_sqrt(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Sqrt')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_sub(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Sub')
        self.add_input(tfnode, nnef_node, 'x')
        self.add_input(tfnode, nnef_node, 'y')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_sum_reduce(self, nnef_node):
        #Generate const node for axis
        tfnode_axes = NodeDef(name=self.format_name(nnef_node.name) + '/axes', op='Const')
        tfnode_axes.attr['dtype'].type = 3
        array_length = len(nnef_node.parameters['axes'])
        new_axes = []
        for axis in nnef_node.parameters['axes']:
            new_axes.append(self.mapping[axis])

        tfnode_axes.attr['value'].tensor.dtype = 3
        tfnode_axes.attr['value'].tensor.tensor_shape.dim.add().size = array_length
        if array_length == 1:
            tfnode_axes.attr['value'].tensor.int_val.extend([new_axes[0]])
        else:
            tfnode_axes.attr['value'].tensor.tensor_content = np.asarray(new_axes, dtype=np.int32).tobytes()

        self.tf_graph.node.extend([tfnode_axes])

        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Sum')
        self.add_input(tfnode, nnef_node, 'input')
        tfnode.input.extend([tfnode_axes.name])
        tfnode.attr['T'].type = 1
        tfnode.attr['Tidx'].type = 3
        tfnode.attr['keep_dims'].b = False

        self.tf_graph.node.extend([tfnode])

    def export_tanh(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Tanh')
        self.add_input(tfnode, nnef_node, 'x')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_transpose(self, nnef_node):
        tfnode_perm = NodeDef(name=self.format_name(nnef_node.name) + '/perm', op='Const')
        tfnode_perm.attr['dtype'].type = 3
        tfnode_perm.attr['value'].tensor.dtype = 3
        tfnode_perm.attr['value'].tensor.tensor_shape.dim.add().size = len(nnef_node.parameters['axes'])
        tfnode_perm.attr['value'].tensor.tensor_content = np.asarray(nnef_node.parameters['axes'], dtype=np.int32).tobytes()

        self.tf_graph.node.extend([tfnode_perm])

        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Transpose')
        self.add_input(tfnode, nnef_node, 'input')
        tfnode.input.extend([tfnode_perm.name])
        tfnode.attr['T'].type = 1
        tfnode.attr['Tperm'].type = 3

        self.tf_graph.node.extend([tfnode])

    def export_update(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Identity')
        self.add_input(tfnode, nnef_node, 'value')
        tfnode.attr['T'].type = 1

        self.tf_graph.node.extend([tfnode])

    def export_variable(self, nnef_node):
        tfnode = NodeDef(name=self.format_name(nnef_node.name), op='Const')
        tfnode.attr['dtype'].type = 1
        tfnode.attr['value'].tensor.dtype = 1

        self.tf_graph.node.extend([tfnode])
