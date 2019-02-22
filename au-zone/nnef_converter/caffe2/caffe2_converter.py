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

import math
import logging
import textwrap
import collections
import json

import networkx as nx
import numpy as np

from ..common.importer_exporter import ImporterExporter
from ..common.nnef_converter import *
from ..common.nnef_graph import *
from ..common import nnef_node as node

from ..caffe2.protobufs.caffe2.proto import caffe2_pb2


class Caffe2Logger(object):
    single_line_sep = "---------------------------------------------------------------------------------------------------------------------------------"
    double_line_sep = "===================================================================================================="

    def __init__(self):
        super(Caffe2Logger, self).__init__()
        self.logger = logging.getLogger('nnef_convert')

    def log_tf_node_info(self, caffe2node, inputs, attrs):
        title = "Importing Caffe2 Node:             "
        preferredWidth = 250
        wrapper = textwrap.TextWrapper(initial_indent=title,
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(self.single_line_sep)
        self.logger.info(wrapper.fill("Name \t%s"%(caffe2node.output[0])))
        wrapper = textwrap.TextWrapper(initial_indent=' ' * len(title),
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(wrapper.fill("Op   \t%s"%(caffe2node.type)))

        unused_input_found = False
        used_input_found = False
        unused_attribute_found = False
        used_attribute_found = False

        if inputs is not None and caffe2node.input is not None:
            for cnt, input_item in enumerate(caffe2node.input):
                if cnt in inputs.values():
                    if not used_input_found:
                        used_input_found = True
                        self.logger.info(wrapper.fill(""))
                        self.logger.info(wrapper.fill("Used Inputs:"))
                    self.logger.info(wrapper.fill("\t%s"%(input_item)))

            for cnt, input_item in enumerate(caffe2node.input):
                if cnt not in inputs.values():
                    if not unused_input_found:
                        unused_input_found = True
                        self.logger.info(wrapper.fill(""))
                        self.logger.info(wrapper.fill("Unused Inputs:"))
                    self.logger.info(wrapper.fill("\t%s"%(input_item)))

        if attrs is not None and caffe2node.arg is not None:
            for arg in caffe2node.arg:
                if arg.name in attrs:
                    if not used_attribute_found:
                        used_attribute_found = True
                        self.logger.info(wrapper.fill(""))
                        self.logger.info(wrapper.fill("Used Attributes:"))
                    if self.log_level == 'debug':
                        self.logger.debug(wrapper.fill("\t'%s': %s" % (arg.name)))
                    else:
                        self.logger.info(wrapper.fill("\t'%s'" % arg.name))

            for arg in caffe2node.arg:
                if arg.name not in attrs:
                    if not unused_attribute_found:
                        unused_attribute_found = True
                        self.logger.info(wrapper.fill(""))
                        self.logger.info(wrapper.fill("Unused Attributes:"))
                    if self.log_level == 'debug':
                        self.logger.debug(wrapper.fill("\t'%s': %s" % (arg.name)))
                    else:
                        self.logger.info(wrapper.fill("\t'%s'" % arg.name))

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

    def log_skipping_nodes(self, caffe2node):
        self.print_msg_nodeop_nodename("Skipping Op:                       ",
                                       caffe2node.type, caffe2node.output[0])
        title = "Skipping Caffe2 Node:              "
        preferredWidth = 250
        wrapper = textwrap.TextWrapper(initial_indent=title,
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(self.single_line_sep)
        self.logger.info(wrapper.fill("Name \t%s"%(caffe2node.output[0])))
        wrapper = textwrap.TextWrapper(initial_indent=' ' * len(title),
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(wrapper.fill("Op   \t%s"%(caffe2node.type)))

    def log_unsupported_nodes(self, caffe2node):
        title = "Unsupported Caffe2 Node:           "
        preferredWidth = 250
        wrapper = textwrap.TextWrapper(initial_indent=title,
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(self.single_line_sep)
        self.logger.info(wrapper.fill("Name \t%s"%(caffe2node.output[0])))
        wrapper = textwrap.TextWrapper(initial_indent=' ' * len(title),
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(wrapper.fill("Op   \t%s"%(caffe2node.type)))

    def log_total_conversions(self):
        title = "Finished Converting Model:         "
        preferredWidth = 250
        wrapper = textwrap.TextWrapper(initial_indent=title,
                                       width=preferredWidth,
                                       subsequent_indent=' ' * len(title))
        self.logger.info(self.double_line_sep)
        self.logger.info(wrapper.fill("Total Caffe2 Nodes     \t%s"%(str(self.total))))
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

class Caffe2Importer(Caffe2Logger, ImporterExporter):
    skip_prefix = [
        "^",
        "train_op",
        "save",
        "gradients",
        "init",
        "global_step",
        "distort_image",
        "Adagrad",
        "Adam",
    ]

    skip_scope = [
        "random_uniform",
        "Initializer",
        "optimizer",
        "weight_loss",
        "parallel_read",
        "case"
    ]

    skip_op = set([
        "L2Loss",
        "VariableV2",
        "Assign",
        "RandomUniform",
        "FIFOQueueV2",
        "Assert",
        "Unpack",
        "NextIteration",
        "TensorArrayV3",
        "Range",
        "TensorArrayScatterV3",
        "TensorArrayReadV3",
        "TensorArrayWriteV3",
        "Dequantize",
        "RandomShuffleQueueV2",
        "QueueDequeueManyV2",
        "CreateCommonWorld",
        "CloneCommonWorld",
        "DestroyCommonWorld",
        "Broadcast"
    ])

    def __init__(self, input_model, data_model, value_info, output_nodes, log_level='info'):
        super(Caffe2Importer, self).__init__()

        self.node_pool = collections.OrderedDict()
        self.input_model = input_model
        self.data_model = data_model
        self.log_level = log_level
        self.nxgraph = nx.OrderedDiGraph()
        self.data_graph = super(Caffe2Importer, self).openProtobuf(self.data_model, caffe2_pb2.NetDef())
        self.net_graph = super(Caffe2Importer, self).openProtobuf(self.input_model, caffe2_pb2.NetDef())
        self.input_nodes = {}
        self.output_nodes = {}
        self.name_convs = {}
        self.start_format = None
        self.start_length = 0
        self.successful = 0
        self.total = 0
        self.removed_nodes = 0

        value_info_file = open(value_info)
        parsed_value_info = json.load(value_info_file)

        i = 1
        if len(parsed_value_info) == 1:
            self.input_nodes[list(parsed_value_info.keys())[0]] = "input"
            shape = list(parsed_value_info.values())[0][1]

            if shape[0] < 1:
                shape[0] = 1

            if self.start_length == 0:
                self.start_length = len(shape)

            nnef_node = node.External(shape=shape,
                                      _uid='input',
                                      _output_shape=shape)

            self.node_pool['input'] = nnef_node

        elif len(parsed_value_info) > 1:
            i = 1
            for key, val in parsed_value_info.items():
                self.input_nodes[key] = "input" + str(i)
                shape = val[1]

                if shape[0] < 1:
                    shape[0] = 1

                if self.start_length == 0:
                    self.start_length = len(shape)

                nnef_node = node.External(shape=shape,
                                          _uid='input' + str(i),
                                          _output_shape=shape)

                self.node_pool['input' + str(i)] = nnef_node
                i += 1
        else:
            raise ValueError("There are no inputs within the value-info file.")

        i = 1
        for out_node in self.net_graph.external_output:
            if len(self.net_graph.external_output) == 1:
                self.output_nodes[out_node] = "output"
            else:
                self.output_nodes[out_node] = "output" + str(i)
            i += 1

    def run(self):
        self.create_data_nodes()
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
            input_node_list.append(self.get_node_from_pool_by_name('input'))
        else:
            i = 1
            while 'input' + str(i) in self.node_pool.keys():
                input_node_list.append(self.get_node_from_pool_by_name('input'+str(i)))
                i += 1

        return input_node_list

    def get_output_nodes(self):
        output_node_list = []
        if 'output' in self.node_pool.keys():
            output_node_list.append(self.get_node_from_pool_by_name('output'))
        else:
            i = 1
            while 'output' + str(i) in self.node_pool.keys():
                output_node_list.append(self.get_node_from_pool_by_name('output' + str(i)))
                i += 1

        return output_node_list

    def create_data_nodes(self):
        for caffe2node in self.data_graph.op:
            if caffe2node.output[0] in self.input_nodes:
                continue

            if caffe2node.type == 'GivenTensorFill':
                for arg in caffe2node.arg:
                    if arg.name == 'shape':
                        shape = list(arg.ints)
                        if len(shape) == 1:
                            shape = [1] + shape
                    if arg.name == 'values':
                        tensor_content = np.asarray(arg.floats, dtype=np.float32)
                tensor_content = np.reshape(tensor_content, shape)
                try:
                    if isinstance(caffe2node.output[0], unicode):
                        label = caffe2node.output[0].encode('ascii')
                    else:
                        label = caffe2node.output[0]
                except NameError:
                    label = caffe2node.output[0]

                nnef_node = node.Variable(label=label,
                                          shape=shape,
                                          _np_dtype=tensor_content.dtype,
                                          _np_tensor=tensor_content,
                                          _output_shape=shape,
                                          _uid=self.gen_node_name(caffe2node.output[0]))

                attrs = {'shape':shape, 'values':None}

            elif caffe2node.type == 'GivenTensorIntFill':
                for arg in caffe2node.arg:
                    if arg.name != 'values':
                        shape = list(arg.ints)
                        if len(shape) == 1:
                            shape = [1] + shape
                    if arg.name == 'values':
                        values = list(arg.ints)

                nnef_node = node.Constant(value=values,
                                          shape=shape,
                                          _uid=self.gen_node_name(caffe2node.output[0]),
                                          _output_shape=shape,
                                          _np_dtype=np.int32)

            else:
                print(caffe2node)
                input()

            attrs = {'shape':shape, 'values': None}
            self.add_node_to_pool(nnef_node, caffe2node, {}, attrs)
        return

    def create_nodes(self):
        for caffe2node in self.net_graph.op:
            self.total += 1
            if self._skip_node(caffe2node):
                self.log_skipping_nodes(caffe2node)
                continue
            if hasattr(caffe2node, 'type'):
                node_op = caffe2node.type
                if hasattr(self, "import_" + node_op):
                    func = getattr(self, "import_" + node_op)
                    nnef_node, caffe2_inputs, caffe2_attrs = func(caffe2node)
                    self.successful += 1
                    if nnef_node is not None:
                        self.add_node_to_pool(nnef_node, caffe2node, caffe2_inputs, caffe2_attrs)
                else:
                    self.import_UNKNOWN(caffe2node)
            else:
                self.logger.error("Node doesn't have op attr.: %s"%(caffe2node.output[0]))

    def add_node_to_pool(self, nnef_node, caffe2node, caffe2_inputs, caffe2_attrs):
        if nnef_node.name not in self.node_pool.keys():
            self.log_tf_node_info(caffe2node, caffe2_inputs, caffe2_attrs)
            self.node_pool[nnef_node.name] = nnef_node

    def remove_node_from_pool(self, nnef_node):
        self.log_removing_node(nnef_node)
        self.node_pool.pop(nnef_node.name, None)

    def get_node_from_pool(self, caffe2node, idx):
        node_name = self.gen_node_name(self.get_caffe2_input(caffe2node, idx), get_orig=True)
        #Handles cases where nodes are out of order within Protocol Buffer
        try:
            nnef_node = self.get_node_from_pool_by_name(node_name)
        except:
            for caffe2node in self.net_graph.op:
                if self.gen_node_name(caffe2node.output[0], get_orig=True) == node_name:
                    if hasattr(caffe2node, 'type'):
                        node_op = caffe2node.op
                        if hasattr(self, "import_" + node_op):
                            func = getattr(self, "import_" + node_op)
                            nnef_node, caffe2_inputs, caffe2_attrs = func(caffe2node)
                            if nnef_node is not None:
                                self.add_node_to_pool(nnef_node, caffe2node, caffe2_inputs, caffe2_attrs)
                    break
            nnef_node = self.get_node_from_pool_by_name(node_name)

        if nnef_node.op == 'idn':
            nnef_node = self.get_node_from_pool_by_name(nnef_node.name).parameters['x']
        return nnef_node

    def get_node_from_pool_by_name(self, node_name):
        if node_name in self.name_convs:
            node_name = self.name_convs[node_name]
        assert node_name in self.node_pool.keys(), "Node pool doesn't contain required node: %s" % node_name
        return self.node_pool[node_name]

    @classmethod
    def _skip_node(cls, caffe2node):

        for prefix in cls.skip_prefix:
            if caffe2node.output[0].lower().startswith(prefix.lower()):
                return True

        scopes = Caffe2Importer._get_scopes(caffe2node.output[0])

        for s in scopes:
            if s.lower() in cls.skip_scope:
                return True

        if caffe2node.type in cls.skip_op:
            return True

        return False

    # Helper function to convert node names to lower case and remove illegal characters ('/', ...)
    def gen_node_name(self, node_name, get_orig=False):
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

        if name in self.node_pool and not get_orig:
            base_name = name
            i = 1
            while name in self.node_pool:
                name = base_name + "_" + str(i)
                i += 1
            self.name_convs[base_name] = name

        return name

    def get_caffe2_input(self, caffe2node, idx):
        assert idx < len(caffe2node.input), "Bad index for accessing Caffe2's op input %s"%idx
        return caffe2node.input[idx]

    '''
        Called by the NNEF graph when all nodes are there, with no edge yet.
    '''
    def pre_compile_callback(self, nx_graph):
        # Cleaning up "idn" nodes
        remove_nodes = []
        for nnef_node_name in nx_graph:
            if nx_graph.node[nnef_node_name]['node'].op == 'idn':
                remove_nodes.append(nx_graph.node[nnef_node_name]['node'].name)
        nx_graph.remove_nodes_from(remove_nodes)

        return
    '''
        Called by the NNEF graph after edges are connected.
    '''
    def post_compile_callback(self, nx_graph):
        pass

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

    def import_UNKNOWN(self, caffe2node):
        self.log_unsupported_nodes(caffe2node)
        return

    def import_NoOp(self, caffe2node):
        return None, None, None

    def import_Add(self, caffe2node):
        caffe2_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        nnef_node_y = self.get_node_from_pool(caffe2node, caffe2_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.Add(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(caffe2node.output[0]),
                             _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Allreduce(self, caffe2node):
        caffe2_inputs = {'input': 1}
        attrs = {}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])
        axes = list(range(len(nnef_node_input.output_shape)))

        nnef_node = node.SumReduce(input=nnef_node_input,
                                   axes=axes,
                                   _uid=self.gen_node_name(caffe2node.output[0]),
                                   _output_shape=[1])

        return nnef_node, caffe2_inputs, attrs

    def import_And(self, caffe2node):
        caffe2_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        nnef_node_y = self.get_node_from_pool(caffe2node, caffe2_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.And(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(caffe2node.output[0]),
                             _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Append(self, caffe2node):
        caffe2_inputs = {'input_1': 0, 'input_2': 1}
        attrs = {}

        nnef_node_input1 = self.get_node_from_pool(caffe2node, caffe2_inputs['input_1'])
        nnef_node_input2 = self.get_node_from_pool(caffe2node, caffe2_inputs['input_2'])

        values = [nnef_node_input1, nnef_node_input2]
        axis = len(nnef_node_input1.output_shape) - 1

        output_shape = nnef_node_input1.output_shape[:]
        output_shape[-1] = output_shape[-1] + nnef_node_input2.output_shape[-1]

        nnef_node = node.Concat(values=values,
                                axis=axis,
                                _uid=self.gen_node_name(caffe2node.output[0]),
                                _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_AveragePool(self, caffe2node):
        caffe2_inputs = {'input':0}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])
        in_shape = nnef_node_input.output_shape

        indv_pad = False
        legacy_pad = False
        global_pooling = False
        strides = [1] * len(in_shape)
        padding = [(0, 0)] * len(in_shape)
        dilations = [1] * len(in_shape)
        for arg in caffe2node.arg:
            if arg.name == 'kernels':
                sizes = list(arg.ints)
                sizes = [int(v) for v in sizes]
                if len(sizes) == (len(in_shape) - 2):
                    sizes = [1, 1] + sizes
                elif len(sizes) < (len(in_shape) - 2):
                    sizes = [1, 1] + (sizes * (len(in_shape) - 2))
            if arg.name == 'strides':
                strides = list(arg.ints)
                strides = [int(v) for v in strides]
                if len(strides) == (len(in_shape) - 2):
                    strides = [1, 1] + strides
                elif len(strides) < (len(in_shape) - 2):
                    strides = [1, 1] + (strides * (len(in_shape) - 2))
            if arg.name == 'pads':
                pads = list(arg.ints)
                if len(pads) == len(in_shape):
                    padding = []
                else:
                    padding = [(0, 0), (0, 0)]
                for i in pads:
                    padding.append((int(i), int(i)))
                if len(padding) < len(in_shape):
                    padding = padding[:2] + ((len(in_shape) - 2) * [padding[2]])
            if arg.name == 'dilations':
                dilations = list(arg.ints)
                if len(dilations) == (len(in_shape) - 2):
                    dilations = [1, 1] + dilations[2:]
                elif len(dilations) < (len(in_shape) - 2):
                    dilations = [1, 1] + (dilations * (len(in_shape) - 2))
            if arg.name == 'kernel':
                if arg.i == 0:
                    sizes = [1, 1] + in_shape[2:]
                else:
                    sizes = [1, 1] + (len(in_shape)-2)*[int(arg.i)]
            if arg.name == 'stride':
                strides = [1, 1] + (len(in_shape)-2)*[int(arg.i)]
            if arg.name == 'pad':
                pad = arg.i
                padding = [(0, 0), (0, 0)] + (len(in_shape)-2)*[(int(arg.i), int(arg.i))]
            if arg.name == 'pad_l':
                indv_pad = True
                pad_l = int(arg.i)
            if arg.name == 'pad_r':
                pad_r = int(arg.i)
            if arg.name == 'pad_t':
                pad_t = int(arg.i)
            if arg.name == 'pad_b':
                pad_b = int(arg.i)
            if arg.name == 'legacy_pad':
                legacy_pad = True
                pad_val = int(arg.i)
            if arg.name == 'global_pooling':
                global_pooling = True
        if indv_pad:
            padding = [(0, 0), (0, 0), (pad_l, pad_r), (pad_t, pad_b)]
        if global_pooling:
            padding = [(0, 0)]*len(in_shape)
            strides = [1]*len(in_shape)
        dilations = [1] * len(in_shape)
        sizes = [int(v) for v in sizes]

        attrs = {'kernel': sizes, 'stride': strides, 'strides': strides,
                 'pad': padding, 'pads': padding, 'dilations': dilations,
                 'pad_l':None, 'pad_r':None, 'pad_t':None, 'pad_b':None}

        output_shape = len(in_shape) * [0]
        for i in range(len(in_shape)):
            if padding == []:
                output_shape[i] = math.ceil(in_shape[i] / strides[i])
            else:
                fd = (sizes[i] - 1) * dilations[i] + 1
                padding_add = padding[i][0] + padding[i][1]
                output_shape[i] = math.floor((in_shape[i] + padding_add - fd) / strides[i]) + 1

        if legacy_pad and pad_val == 3:
            output_shape = len(in_shape) * [0]
            for i in range(len(in_shape)):
                output_shape[i] = math.ceil((in_shape[i] + padding[i][0] * 2 - sizes[i]) / strides[i] + 1)
                if padding[i][0] > 0 and (output_shape[i] - 1) * strides[i] >= in_shape[i] + padding[i][0]:
                    output_shape[i] -= 1
            std_output_shape = len(in_shape) * [0]
            for i in range(len(in_shape)):
                std_output_shape[i] = int((in_shape[i] + padding[i][0] * 2 - sizes[i]) / strides[i] + 1)
                pad_tail = padding[i][0] + strides[i] * (output_shape[i] - std_output_shape[i])
                padding[i] = (padding[i][0], int(pad_tail))

        nnef_node = node.AvgPool(input=nnef_node_input,
                                 size=sizes,
                                 padding=padding,
                                 stride=strides,
                                 dilation=dilations,
                                 _uid=self.gen_node_name(caffe2node.output[0]),
                                 _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Concat(self, caffe2node):
        caffe2_inputs = {}
        attrs = {'order':None}
        values = []
        for i in range(len(caffe2node.input)):
            caffe2_inputs['value_' + str(i)] = i
            values.append(self.get_node_from_pool(caffe2node, i))
        for arg in caffe2node.arg:
            if arg.name == 'axis':
                axis = arg.i
            if arg.name == 'order':
                axis = arg.s.decode('ascii').find('C')

        output_shape = values[0].output_shape[:]
        for i in range(1, len(values)):
            output_shape[axis] += values[i].output_shape[axis]

        nnef_node = node.Concat(values=values,
                                axis=axis,
                                _uid=self.gen_node_name(caffe2node.output[0]),
                                _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Conditional(self, caffe2node):
        caffe2_inputs = {'condition': 0, 'true_value': 1, 'false_value': 2}
        attrs = {}

        nnef_node_condition = self.get_node_from_pool(caffe2node, caffe2_inputs['condition'])
        nnef_node_true = self.get_node_from_pool(caffe2node, caffe2_inputs['true_value'])
        nnef_node_false = self.get_node_from_pool(caffe2node, caffe2_inputs['false_value'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_true, nnef_node_false)

        nnef_node = node.Select(condition=nnef_node_condition,
                                true_value=nnef_node_true,
                                false_value=nnef_node_false,
                                _uid=self.gen_node_name(caffe2node.output[0]),
                                _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Conv(self, caffe2node):
        if len(caffe2node.input) == 3:
            caffe2_inputs = {'input':0, 'filter':1, 'bias':2}
            bias = self.get_node_from_pool(caffe2node, caffe2_inputs['bias'])
        else:
            caffe2_inputs = {'input':0, 'filter':1}
            bias = 0.0

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])
        nnef_node_filter = self.get_node_from_pool(caffe2node, caffe2_inputs['filter'])
        in_shape = nnef_node_input.output_shape
        filter_shape = nnef_node_filter.output_shape

        indv_pad = False
        legacy_pad = False
        global_pooling = False
        groups = 1
        strides = [1] * (len(in_shape)-2)
        padding = [(0, 0)] * (len(in_shape)-2)
        dilations = [1] * (len(in_shape)-2)

        for arg in caffe2node.arg:
            if arg.name == 'strides':
                strides = list(arg.ints)
                strides = [int(v) for v in strides]
                if len(strides) > (len(in_shape) - 2):
                    strides = strides[2:]
                elif len(strides) < (len(in_shape) - 2):
                    strides = strides * (len(in_shape) - 2)
            if arg.name == 'pads':
                pads = list(arg.ints)
                padding = []
                for i in pads:
                    padding.append((int(i), int(i)))
                if len(padding) > (len(in_shape) - 2):
                    padding = padding[2:]
                elif len(padding) < (len(in_shape) - 2):
                    padding = padding * (len(in_shape) - 2)
            if arg.name == 'dilations':
                dilations = list(arg.ints)
                if len(dilations) > (len(in_shape) - 2):
                    dilations = dilations[2:]
                elif len(dilations) < (len(in_shape) - 2):
                    dilations = dilations * (len(in_shape) - 2)
            if arg.name == 'stride':
                strides = (len(in_shape)-2)*[int(arg.i)]
            if arg.name == 'group':
                groups = int(arg.i)
            if arg.name == 'pad':
                pad = int(arg.i)
                padding = (len(in_shape)-2)*[(int(arg.i), int(arg.i))]
            if arg.name == 'pad_l':
                indv_pad = True
                pad_l = int(arg.i)
            if arg.name == 'pad_r':
                pad_r = int(arg.i)
            if arg.name == 'pad_t':
                pad_t = int(arg.i)
            if arg.name == 'pad_b':
                pad_b = int(arg.i)
            if arg.name == 'legacy_pad':
                legacy_pad = True
                pad_val = int(arg.i)
            if arg.name == 'global_pooling':
                global_pooling = True
        if indv_pad:
            padding = [(pad_l, pad_r), (pad_t, pad_b)]
        if global_pooling:
            padding = [(0, 0)]*len(in_shape)
            strides = [1]*len(in_shape)

        output_shape = len(in_shape) * [0]
        output_shape[0] = in_shape[0]
        output_shape[1] = filter_shape[0]

        for i in range(2, len(in_shape)):
            if padding == []:
                output_shape[i] = math.ceil(in_shape[i] / strides[i-2])
            else:
                fd = (filter_shape[i] - 1) * dilations[i-2] + 1
                padding_add = padding[i-2][0] + padding[i-2][1]
                output_shape[i] = math.floor((in_shape[i] + padding_add - fd) / strides[i-2]) + 1

        if legacy_pad and pad_val == 3:
            std_output_shape = output_shape[:]
            for i in range(2,len(in_shape)):
                output_shape[i] = math.ceil((in_shape[i] + padding[i-2][0] * 2 - filter_shape[i]) / strides[i-2] + 1)
                if padding[i-2][0] > 0 and (output_shape[i] - 1) * strides[i-2] >= in_shape[i] + padding[i-2][0]:
                    output_shape[i] -= 1
            for i in range(2,len(in_shape)):
                std_output_shape[i] = int((in_shape[i] + padding[i-2][0] * 2 - filter_shape[i]) / strides[i-2] + 1)
                pad_tail = padding[i-2][0] + strides[i-2] * (output_shape[i] - std_output_shape[i])
                padding[i-2] = (padding[i-2][0], int(pad_tail))

        nnef_node = node.Conv(input=nnef_node_input,
                              filter=nnef_node_filter,
                              bias=bias,
                              padding=padding,
                              stride=strides,
                              dilation=dilations,
                              groups=groups,
                              _uid=self.gen_node_name(caffe2node.output[0]),
                              _output_shape=output_shape)

        attrs = {'stride': strides, 'strides': strides, 'groups': groups,
                 'pad': padding, 'pads': padding, 'dilations': dilations,
                 'pad_l':None, 'pad_r':None, 'pad_t':None, 'pad_b':None}

        return nnef_node, caffe2_inputs, attrs

    def import_ConvTranspose(self, caffe2node):
        if len(caffe2node.input) == 3:
            caffe2_inputs = {'input':0, 'filter':1, 'bias':2}
            bias = self.get_node_from_pool(caffe2node, caffe2_inputs['bias'])
        else:
            caffe2_inputs = {'input':0, 'filter':1}
            bias = 0.0

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])
        nnef_node_filter = self.get_node_from_pool(caffe2node, caffe2_inputs['filter'])

        if nnef_node_filter.op == 'variable':
            if not nnef_node_filter.modified:
                nnef_tensor = np.transpose(nnef_node_filter.tensor, [1, 0, 2, 3])
                nnef_node_filter.tensor = nnef_tensor
                new_shape = list(np.shape(nnef_tensor))
                nnef_node_filter.parameters['shape'] = new_shape
                nnef_node_filter.output_shape = new_shape
                nnef_node_filter.modified = True
            else:
                new_shape = nnef_node_filter.parameters['shape']
        elif nnef_node_filter.op == 'reshape':
            current_shape = nnef_node_filter.parameters['shape'][:]
            new_shape = convert_format(current_shape, 'NCHW', 'CNHW')
            nnef_node_filter.parameters['shape'] = new_shape
        else:
            new_shape = [1]*len(nnef_node_input.output_shape)

        in_shape = nnef_node_input.output_shape
        filter_shape = nnef_node_filter.output_shape

        indv_pad = False
        legacy_pad = False
        global_pooling = False
        groups = 1
        strides = [1] * (len(in_shape)-2)
        padding = [(0, 0)] * (len(in_shape)-2)
        dilations = [1] * (len(in_shape)-2)

        for arg in caffe2node.arg:
            if arg.name == 'strides':
                strides = list(arg.ints)
                strides = [int(v) for v in strides]
                if len(strides) > (len(in_shape) - 2):
                    strides = strides[2:]
                elif len(strides) < (len(in_shape) - 2):
                    strides = strides * (len(in_shape) - 2)
            if arg.name == 'pads':
                pads = list(arg.ints)
                padding = []
                for i in pads:
                    padding.append((int(i), int(i)))
                if len(padding) > (len(in_shape) - 2):
                    padding = padding[2:]
                elif len(padding) < (len(in_shape) - 2):
                    padding = padding * (len(in_shape) - 2)
            if arg.name == 'dilations':
                dilations = list(arg.ints)
                if len(dilations) > (len(in_shape) - 2):
                    dilations = dilations[2:]
                elif len(dilations) < (len(in_shape) - 2):
                    dilations = dilations * (len(in_shape) - 2)
            if arg.name == 'stride':
                strides = (len(in_shape)-2)*[int(arg.i)]
            if arg.name == 'group':
                groups = int(arg.i)
            if arg.name == 'pad':
                pad = int(arg.i)
                padding = (len(in_shape)-2)*[(int(arg.i), int(arg.i))]
            if arg.name == 'pad_l':
                indv_pad = True
                pad_l = int(arg.i)
            if arg.name == 'pad_r':
                pad_r = int(arg.i)
            if arg.name == 'pad_t':
                pad_t = int(arg.i)
            if arg.name == 'pad_b':
                pad_b = int(arg.i)
            if arg.name == 'legacy_pad':
                legacy_pad = True
                pad_val = int(arg.i)
            if arg.name == 'global_pooling':
                global_pooling = True
        if indv_pad:
            padding = [(pad_l, pad_r), (pad_t, pad_b)]
        if global_pooling:
            padding = [(0, 0)]*len(in_shape)
            strides = [1]*len(in_shape)

        output_shape = len(in_shape) * [0]
        output_shape[0] = in_shape[0]
        output_shape[1] = filter_shape[0]

        for i in range(2, len(in_shape)):
            if padding == []:
                output_shape[i] = math.ceil(in_shape[i] / strides[i-2])
            else:
                fd = (filter_shape[i] - 1) * dilations[i-2] + 1
                padding_add = padding[i-2][0] + padding[i-2][1]
                output_shape[i] = math.floor((in_shape[i] + padding_add - fd) / strides[i-2]) + 1

        if legacy_pad and pad_val == 3:
            std_output_shape = output_shape[:]
            for i in range(2,len(in_shape)):
                output_shape[i] = math.ceil((in_shape[i] + padding[i-2][0] * 2 - filter_shape[i]) / strides[i-2] + 1)
                if padding[i-2][0] > 0 and (output_shape[i] - 1) * strides[i-2] >= in_shape[i] + padding[i-2][0]:
                    output_shape[i] -= 1
            for i in range(2,len(in_shape)):
                std_output_shape[i] = int((in_shape[i] + padding[i-2][0] * 2 - filter_shape[i]) / strides[i-2] + 1)
                pad_tail = padding[i-2][0] + strides[i-2] * (output_shape[i] - std_output_shape[i])
                padding[i-2] = (padding[i-2][0], int(pad_tail))

        nnef_node = node.Deconv(input=nnef_node_input,
                                filter=nnef_node_filter,
                                bias=bias,
                                padding=padding,
                                stride=strides,
                                dilation=dilations,
                                groups=groups,
                                _uid=self.gen_node_name(caffe2node.output[0]),
                                _output_shape=output_shape)

        attrs = {'stride': strides, 'strides': strides, 'groups': groups,
                 'pad': padding, 'pads': padding, 'dilations': dilations,
                 'pad_l':None, 'pad_r':None, 'pad_t':None, 'pad_b':None}

        return nnef_node, caffe2_inputs, attrs

    def import_DepthConcat(self, caffe2node):
        caffe2_inputs = {}
        attrs = {'order':None}
        values = []
        for i in range(len(caffe2node.input)):
            caffe2_inputs['value_' + str(i)] = i
            values.append(self.get_node_from_pool(caffe2node, i))
        for arg in caffe2node.arg:
            if arg.name == 'axis':
                axis = arg.i
            if arg.name == 'order':
                axis = arg.s.decode('ascii').find('C')

        output_shape = values[0].output_shape[:]
        for i in range(1, len(values)):
            output_shape[axis] += values[i].output_shape[axis]

        nnef_node = node.Concat(values=values,
                                axis=axis,
                                _uid=self.gen_node_name(caffe2node.output[0]),
                                _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_DepthSplit(self, caffe2node):
        caffe2_inputs = {'value': 0}
        attrs = {'axis': None}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['value'])

        split_name = '['
        output_names = []
        for output in caffe2node.output:
            out_name = self.gen_node_name(output)
            split_name += out_name + ', '
            output_names.append(out_name)
        split_name = split_name[:-2] + ']'

        ratios = []
        for arg in caffe2node.arg:
            if arg.name == 'axis':
                axis = arg.i
            if arg.name == 'split':
                ratios = list(arg.ints)
                attrs['split'] = None

        if not ratios:
            if len(caffe2node.input) == 1:
                ratios = [int(nnef_node_input.output_shape[axis]/len(output_names))]*len(output_names)
            else:
                nnef_node_shape = self.get_node_from_pool(caffe2node, 1)
                ratios = nnef_node_shape.parameters['value']

        input_shape = nnef_node_input.output_shape[:]

        nnef_node_split = node.Split(value=nnef_node_input,
                                     axis=axis,
                                     ratios=ratios,
                                     _uid=split_name,
                                     _output_shape=input_shape)

        for i in range(len(output_names)):
            out_shape = input_shape[:]
            out_shape[axis] = ratios[i]
            nnef_node = node.OutputVal(base_node=nnef_node_split,
                                       base_index=i,
                                       _uid=output_names[i],
                                       _output_shape=out_shape)
            self.node_pool[nnef_node.name] = nnef_node

        return nnef_node_split, caffe2_inputs, attrs

    def import_DotProduct(self, caffe2node):
        caffe2_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        nnef_node_y = self.get_node_from_pool(caffe2node, caffe2_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.Mul(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(caffe2node.output[0]),
                             _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Elu(self, caffe2node):
        caffe2_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Elu(x=nnef_node_x,
                             _uid=self.gen_node_name(caffe2node.output[0]),
                             _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Exp(self, caffe2node):
        caffe2_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])

        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Exp(x=nnef_node_x,
                             _uid=self.gen_node_name(caffe2node.output[0]),
                             _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_FC(self, caffe2node):
        if len(caffe2node.input) == 3:
            caffe2_inputs = {'input':0, 'filter':1, 'bias':2}
            bias = self.get_node_from_pool(caffe2node, caffe2_inputs['bias'])
        else:
            caffe2_inputs = {'input':0, 'filter':1}
            bias = 0.0
        attrs = {}

        node_name = self.gen_node_name(caffe2node.output[0])
        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])
        nnef_node_filter = self.get_node_from_pool(caffe2node, caffe2_inputs['filter'])

        if nnef_node_input.output_shape[-1] != nnef_node_filter.output_shape[-1]:
            total_values = 1
            for val in nnef_node_input.output_shape:
                total_values *= val
            shape = [int(total_values/nnef_node_filter.output_shape[-1]), nnef_node_filter.output_shape[-1]]

            node_reshape = node.Reshape(input=nnef_node_input,
                                        shape=shape,
                                        _uid=node_name + '_reshape',
                                        _output_shape=shape)

            self.node_pool[node_reshape.name] = node_reshape

            output_shape = [shape[0], nnef_node_filter.output_shape[0]]
            nnef_node = node.Linear(input=node_reshape,
                                    filter=nnef_node_filter,
                                    bias=bias,
                                    _uid=node_name,
                                    _output_shape=output_shape)

        else:
            output_shape = [nnef_node_input.output_shape[0], nnef_node_filter.output_shape[0]]
            nnef_node = node.Linear(input=nnef_node_input,
                                    filter=nnef_node_filter,
                                    bias=bias,
                                    _uid=node_name,
                                    _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Flatten(self, caffe2node):
        caffe2_inputs = {'input':0}
        attrs = {'axis':None}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])

        axis = 1
        for arg in caffe2node.arg:
            if arg.name == 'axis':
                axis = arg.i

        size_0 = 1
        size_1 = 1
        for i in range(len(nnef_node_input.output_shape)):
            if i < axis:
                size_0 *= nnef_node_input.output_shape[i]
            else:
                size_1 *= nnef_node_input.output_shape[i]

        shape = [size_0, size_1]

        nnef_node = node.Reshape(input=nnef_node_input,
                                 shape=shape,
                                 _uid=self.gen_node_name(caffe2node.output[0]),
                                 _output_shape=shape)

        return nnef_node, caffe2_inputs, attrs

    def import_GE(self, caffe2node):
        caffe2_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        nnef_node_y = self.get_node_from_pool(caffe2node, caffe2_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.GE(x=nnef_node_x,
                            y=nnef_node_y,
                            _uid=self.gen_node_name(caffe2node.output[0]),
                            _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_GT(self, caffe2node):
        caffe2_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        nnef_node_y = self.get_node_from_pool(caffe2node, caffe2_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.GT(x=nnef_node_x,
                            y=nnef_node_y,
                            _uid=self.gen_node_name(caffe2node.output[0]),
                            _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_LE(self, caffe2node):
        caffe2_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        nnef_node_y = self.get_node_from_pool(caffe2node, caffe2_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.LE(x=nnef_node_x,
                            y=nnef_node_y,
                            _uid=self.gen_node_name(caffe2node.output[0]),
                            _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_LeakyRelu(self, caffe2node):
        caffe2_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        alpha = 0.01
        for arg in caffe2node.arg:
            if arg.name == 'alpha':
                alpha = arg.f

        nnef_node = node.LeakyRelu(x=nnef_node_x,
                                   alpha=alpha,
                                   _uid=self.gen_node_name(caffe2node.output[0]),
                                   _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_LRN(self, caffe2node):
        caffe2_inputs = {'input': 0}
        attrs = {}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])
        output_shape = nnef_node_input.output_shape[:]
        #Find real defaults
        size = [1] * len(nnef_node_input.output_shape)
        alpha = 1.0
        beta = 0.5
        bias = 1.0
        for arg in caffe2node.arg:
            if arg.name == 'size':
                size = [1, arg.i, 1, 1]
                attrs[arg.name] = size
            if arg.name == 'alpha':
                alpha = arg.f
                attrs[arg.name] = alpha
            if arg.name == 'beta':
                beta = arg.f
                attrs[arg.name] = beta
            if arg.name == 'bias':
                bias = arg.f
                attrs[arg.name] = bias

        nnef_node = node.LocalResponseNormalization(input=nnef_node_input,
                                                    alpha=alpha,
                                                    beta=beta,
                                                    bias=bias,
                                                    size=size,
                                                    _uid=self.gen_node_name(caffe2node.output[0]),
                                                    _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_LT(self, caffe2node):
        caffe2_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        nnef_node_y = self.get_node_from_pool(caffe2node, caffe2_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.LT(x=nnef_node_x,
                            y=nnef_node_y,
                            _uid=self.gen_node_name(caffe2node.output[0]),
                            _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_MatMul(self, caffe2node):
        caffe2_inputs = {'A':0, 'B':1}
        attrs = {'trans_a':None, 'trans_b':None}

        nnef_node_A = self.get_node_from_pool(caffe2node, caffe2_inputs['A'])
        nnef_node_B = self.get_node_from_pool(caffe2node, caffe2_inputs['B'])

        trA = False
        trB = False

        for arg in caffe2node.arg:
            if arg.name == 'trans_a' and arg.i == 1:
                trA = True
            if arg.name == 'trans_b' and arg.i == 1:
                trB = True

        output_shape = []
        if trA:
            output_shape.append(nnef_node_A.output_shape[-1])
        else:
            output_shape.append(nnef_node_A.output_shape[0])

        if trB:
            output_shape.append(nnef_node_B.output_shape[0])
        else:
            output_shape.append(nnef_node_B.output_shape[-1])

        nnef_node = node.Matmul(A=nnef_node_A,
                                B=nnef_node_B,
                                transposeA=trA,
                                transposeB=trB,
                                _uid=self.gen_node_name(caffe2node.output[0]),
                                _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Max(self, caffe2node):
        if len(caffe2node.input) != 2:
            raise AssertionError("NNEF accepts exactly 2 tensors for op: max")

        caffe2_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        nnef_node_y = self.get_node_from_pool(caffe2node, caffe2_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.Max(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(caffe2node.output[0]),
                             _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_MaxPool(self, caffe2node):
        caffe2_inputs = {'input':0}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])
        in_shape = nnef_node_input.output_shape

        indv_pad = False
        legacy_pad = False
        global_pooling = False
        strides = [1] * len(in_shape)
        padding = [(0, 0)] * len(in_shape)
        dilations = [1] * len(in_shape)
        for arg in caffe2node.arg:
            if arg.name == 'kernels':
                sizes = list(arg.ints)
                if len(sizes) == (len(in_shape) - 2):
                    sizes = [1, 1] + sizes
                elif len(sizes) < (len(in_shape) - 2):
                    sizes = [1, 1] + (sizes * (len(in_shape) - 2))
            if arg.name == 'strides':
                strides = list(arg.ints)
                strides = [int(v) for v in strides]
                if len(strides) == (len(in_shape) - 2):
                    strides = [1, 1] + strides
                elif len(strides) < (len(in_shape) - 2):
                    strides = [1, 1] + (strides * (len(in_shape) - 2))
            if arg.name == 'pads':
                pads = list(arg.ints)
                if len(pads) == len(in_shape):
                    padding = []
                else:
                    padding = [(0, 0), (0, 0)]
                for i in pads:
                    padding.append((int(i), int(i)))
                if len(padding) < len(in_shape):
                    padding = padding[:2] + ((len(in_shape) - 2) * [padding[2]])
            if arg.name == 'dilations':
                dilations = list(arg.ints)
                if len(dilations) == (len(in_shape) - 2):
                    dilations = [1, 1] + dilations[2:]
                elif len(dilations) < (len(in_shape) - 2):
                    dilations = [1, 1] + (dilations * (len(in_shape) - 2))
            if arg.name == 'kernel':
                if arg.i == 0:
                    sizes = [1, 1] + in_shape[2:]
                else:
                    sizes = [1, 1] + (len(in_shape)-2)*[int(arg.i)]
            if arg.name == 'stride':
                strides = [1, 1] + (len(in_shape)-2)*[int(arg.i)]
            if arg.name == 'pad':
                pad = arg.i
                padding = [(0, 0), (0, 0)] + (len(in_shape)-2)*[(int(arg.i), int(arg.i))]
            if arg.name == 'pad_l':
                indv_pad = True
                pad_l = int(arg.i)
            if arg.name == 'pad_r':
                pad_r = int(arg.i)
            if arg.name == 'pad_t':
                pad_t = int(arg.i)
            if arg.name == 'pad_b':
                pad_b = int(arg.i)
            if arg.name == 'legacy_pad':
                legacy_pad = True
                pad_val = int(arg.i)
            if arg.name == 'global_pooling':
                global_pooling = True
        if indv_pad:
            padding = [(0, 0), (0, 0), (pad_l, pad_r), (pad_t, pad_b)]
        if global_pooling:
            padding = [(0, 0)]*len(in_shape)
            strides = [1]*len(in_shape)
        dilations = [1] * len(in_shape)
        sizes = [int(v) for v in sizes]

        attrs = {'kernel': sizes, 'stride': strides, 'strides': strides,
                 'pad': padding, 'pads': padding, 'dilations': dilations,
                 'pad_l':None, 'pad_r':None, 'pad_t':None, 'pad_b':None}

        output_shape = len(in_shape) * [0]
        for i in range(len(in_shape)):
            if padding == []:
                output_shape[i] = math.ceil(in_shape[i] / strides[i])
            else:
                fd = (sizes[i] - 1) * dilations[i] + 1
                padding_add = padding[i][0] + padding[i][1]
                output_shape[i] = math.floor((in_shape[i] + padding_add - fd) / strides[i]) + 1

        if legacy_pad and pad_val == 3:
            output_shape = len(in_shape) * [0]
            for i in range(len(in_shape)):
                output_shape[i] = math.ceil((in_shape[i] + padding[i][0] * 2 - sizes[i]) / strides[i] + 1)
                if padding[i][0] > 0 and (output_shape[i] - 1) * strides[i] >= in_shape[i] + padding[i][0]:
                    output_shape[i] -= 1
            std_output_shape = len(in_shape) * [0]
            for i in range(len(in_shape)):
                std_output_shape[i] = int((in_shape[i] + padding[i][0] * 2 - sizes[i]) / strides[i] + 1)
                pad_tail = padding[i][0] + strides[i] * (output_shape[i] - std_output_shape[i])
                padding[i] = (padding[i][0], int(pad_tail))

        nnef_node = node.MaxPool(input=nnef_node_input,
                                 size=sizes,
                                 padding=padding,
                                 stride=strides,
                                 dilation=dilations,
                                 _uid=self.gen_node_name(caffe2node.output[0]),
                                 _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_MaxPoolWithIndex(self, caffe2node):
        caffe2_inputs = {'input':0}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])
        in_shape = nnef_node_input.output_shape

        indv_pad = False
        legacy_pad = False
        global_pooling = False
        strides = [1] * len(in_shape)
        padding = [(0, 0)] * len(in_shape)
        dilations = [1] * len(in_shape)
        for arg in caffe2node.arg:
            if arg.name == 'kernels':
                sizes = list(arg.ints)
                if len(sizes) == (len(in_shape) - 2):
                    sizes = [1, 1] + sizes
                elif len(sizes) < (len(in_shape) - 2):
                    sizes = [1, 1] + (sizes * (len(in_shape) - 2))
            if arg.name == 'strides':
                strides = list(arg.ints)
                strides = [int(v) for v in strides]
                if len(strides) == (len(in_shape) - 2):
                    strides = [1, 1] + strides
                elif len(strides) < (len(in_shape) - 2):
                    strides = [1, 1] + (strides * (len(in_shape) - 2))
            if arg.name == 'pads':
                pads = list(arg.ints)
                if len(pads) == len(in_shape):
                    padding = []
                else:
                    padding = [(0, 0), (0, 0)]
                for i in pads:
                    padding.append((int(i), int(i)))
                if len(padding) < len(in_shape):
                    padding = padding[:2] + ((len(in_shape) - 2) * [padding[2]])
            if arg.name == 'dilations':
                dilations = list(arg.ints)
                if len(dilations) == (len(in_shape) - 2):
                    dilations = [1, 1] + dilations[2:]
                elif len(dilations) < (len(in_shape) - 2):
                    dilations = [1, 1] + (dilations * (len(in_shape) - 2))
            if arg.name == 'kernel':
                if arg.i == 0:
                    sizes = [1, 1] + in_shape[2:]
                else:
                    sizes = [1, 1] + (len(in_shape)-2)*[int(arg.i)]
            if arg.name == 'stride':
                strides = [1, 1] + (len(in_shape)-2)*[int(arg.i)]
            if arg.name == 'pad':
                pad = arg.i
                padding = [(0, 0), (0, 0)] + (len(in_shape)-2)*[(int(arg.i), int(arg.i))]
            if arg.name == 'pad_l':
                indv_pad = True
                pad_l = int(arg.i)
            if arg.name == 'pad_r':
                pad_r = int(arg.i)
            if arg.name == 'pad_t':
                pad_t = int(arg.i)
            if arg.name == 'pad_b':
                pad_b = int(arg.i)
            if arg.name == 'legacy_pad':
                legacy_pad = True
                pad_val = int(arg.i)
            if arg.name == 'global_pooling':
                global_pooling = True
        if indv_pad:
            padding = [(0, 0), (0, 0), (pad_l, pad_r), (pad_t, pad_b)]
        if global_pooling:
            padding = [(0, 0)]*len(in_shape)
            strides = [1]*len(in_shape)
        dilations = [1] * len(in_shape)
        sizes = [int(v) for v in sizes]

        attrs = {'kernel': sizes, 'stride': strides, 'strides': strides,
                 'pad': padding, 'pads': padding, 'dilations': dilations,
                 'pad_l':None, 'pad_r':None, 'pad_t':None, 'pad_b':None}

        output_shape = len(in_shape) * [0]
        for i in range(len(in_shape)):
            if padding == []:
                output_shape[i] = math.ceil(in_shape[i] / strides[i])
            else:
                fd = (sizes[i] - 1) * dilations[i] + 1
                padding_add = padding[i][0] + padding[i][1]
                output_shape[i] = math.floor((in_shape[i] + padding_add - fd) / strides[i]) + 1

        if legacy_pad and pad_val == 3:
            output_shape = len(in_shape) * [0]
            for i in range(len(in_shape)):
                output_shape[i] = math.ceil((in_shape[i] + padding[i][0] * 2 - sizes[i]) / strides[i] + 1)
                if padding[i][0] > 0 and (output_shape[i] - 1) * strides[i] >= in_shape[i] + padding[i][0]:
                    output_shape[i] -= 1
            std_output_shape = len(in_shape) * [0]
            for i in range(len(in_shape)):
                std_output_shape[i] = int((in_shape[i] + padding[i][0] * 2 - sizes[i]) / strides[i] + 1)
                pad_tail = padding[i][0] + strides[i] * (output_shape[i] - std_output_shape[i])
                padding[i] = (padding[i][0], int(pad_tail))

        nnef_node = node.MaxPool(input=nnef_node_input,
                                 size=sizes,
                                 padding=padding,
                                 stride=strides,
                                 dilation=dilations,
                                 _uid=self.gen_node_name(caffe2node.output[0]) + ', ' + self.gen_node_name(caffe2node.output[1]),
                                 _output_shape=output_shape)

        nnef_node_pool = node.OutputVal(base_node=nnef_node,
                                        base_index=0,
                                        _uid=self.gen_node_name(caffe2node.output[0]),
                                        _output_shape=output_shape)
        self.node_pool[nnef_node_pool.name] = nnef_node_pool

        nnef_node_index = node.OutputVal(base_node=nnef_node,
                                         base_index=1,
                                         _uid=self.gen_node_name(caffe2node.output[1]),
                                         _output_shape=[1])
        self.node_pool[nnef_node_index.name] = nnef_node_index

        return nnef_node, caffe2_inputs, attrs

    def import_Mul(self, caffe2node):
        caffe2_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        nnef_node_y = self.get_node_from_pool(caffe2node, caffe2_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.Mul(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(caffe2node.output[0]),
                             _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Negative(self, caffe2node):
        caffe2_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Neg(x=nnef_node_x,
                             _uid=self.gen_node_name(caffe2node.output[0]),
                             _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Not(self, caffe2node):
        caffe2_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Not(x=nnef_node_x,
                             _uid=self.gen_node_name(caffe2node.output[0]),
                             _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Normalize(self, caffe2node):
        caffe2_inputs = {'input': 0}
        attrs = {'axis':None}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])
        output_shape = nnef_node_input.output_shape[:]

        for arg in caffe2node.arg:
            if arg.name == 'axis':
                axes = [arg.i]

        nnef_node = node.L2Normalization(input=nnef_node_input,
                                         axes=axes,
                                         _uid=self.gen_node_name(caffe2node.output[0]),
                                         _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Or(self, caffe2node):
        caffe2_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        nnef_node_y = self.get_node_from_pool(caffe2node, caffe2_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.Or(x=nnef_node_x,
                            y=nnef_node_y,
                            _uid=self.gen_node_name(caffe2node.output[0]),
                            _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Reduce(self, caffe2node):
        caffe2_inputs = {'input': 1}
        attrs = {'root': None}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])
        axes = list(range(len(nnef_node_input.output_shape)))

        nnef_node = node.SumReduce(input=nnef_node_input,
                                   axes=axes,
                                   _uid=self.gen_node_name(caffe2node.output[0]),
                                   _output_shape=[1])

        return nnef_node, caffe2_inputs, attrs

    def import_ReduceFrontMean(self, caffe2node):
        caffe2_inputs = {'input': 0}
        attrs = {}

        if len(caffe2node.input) == 2:
            raise AssertionError("NNEF cannot handle subsampling within op: mean_reduce")

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])

        num_dims = -1
        for arg in caffe2node.arg:
            if arg.name == 'num_reduce_dims':
                num_dims = arg.i
        if num_dims == -1:
            axes = list(range(len(nnef_node_input.output_shape)))
            output_shape = [1]
        else:
            axes = list(range(num_dims))
            output_shape = nnef_node_input.output_shape[:]
            output_shape = output_shape[num_dims:]
            if output_shape == []:
                output_shape = [1]

        nnef_node = node.MeanReduce(input=nnef_node_input,
                                    axes=axes,
                                    _uid=self.gen_node_name(caffe2node.output[0]),
                                    _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_ReduceMean(self, caffe2node):
        caffe2_inputs = {'input': 0}
        attrs = {'axes': None, 'keepdims': None}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])
        keep_dims = True

        for arg in caffe2node.arg:
            if arg.name == 'axes':
                axes = list(arg.ints)
            if arg.name == 'keepdims':
                if arg.i == 0:
                    keep_dims = False

        output_shape = []
        for i in range(len(nnef_node_input.output_shape)):
            if i not in axes:
                output_shape.append(nnef_node_input.output_shape[i])

        if keep_dims:
            nnef_node = node.MeanReduce(input=nnef_node_input,
                                        axes=axes,
                                        _uid=self.gen_node_name(caffe2node.output[0]) + "_reduce",
                                        _output_shape=output_shape)

            self.node_pool[nnef_node.name] = nnef_node

            output_shape = nnef_node_input.output_shape[:]
            for i in axes:
                output_shape[i] = 1

            nnef_node_reshape = node.Reshape(input=nnef_node,
                                             shape=output_shape,
                                             _uid=self.gen_node_name(caffe2node.output[0]),
                                             _output_shape=output_shape)

            return nnef_node_reshape, caffe2_inputs, attrs

        else:
            nnef_node = node.MeanReduce(input=nnef_node_input,
                                        axes=axes,
                                        _uid=self.gen_node_name(caffe2node.output[0]),
                                        _output_shape=output_shape)

            return nnef_node, caffe2_inputs, attrs

    def import_ReduceSum(self, caffe2node):
        caffe2_inputs = {'input': 0}
        attrs = {'axes': None, 'keepdims': None}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])
        keep_dims = True

        for arg in caffe2node.arg:
            if arg.name == 'axes':
                axes = list(arg.ints)
            if arg.name == 'keepdims':
                if arg.i == 0:
                    keep_dims = False

        output_shape = []
        for i in range(len(nnef_node_input.output_shape)):
            if i not in axes:
                output_shape.append(nnef_node_input.output_shape[i])

        if keep_dims:
            nnef_node = node.SumReduce(input=nnef_node_input,
                                       axes=axes,
                                       _uid=self.gen_node_name(caffe2node.output[0]) + "_reduce",
                                       _output_shape=output_shape)

            self.node_pool[nnef_node.name] = nnef_node

            output_shape = nnef_node_input.output_shape[:]
            for i in axes:
                output_shape[i] = 1

            nnef_node_reshape = node.Reshape(input=nnef_node,
                                             shape=output_shape,
                                             _uid=self.gen_node_name(caffe2node.output[0]),
                                             _output_shape=output_shape)

            return nnef_node_reshape, caffe2_inputs, attrs

        else:
            nnef_node = node.SumReduce(input=nnef_node_input,
                                       axes=axes,
                                       _uid=self.gen_node_name(caffe2node.output[0]),
                                       _output_shape=output_shape)

            return nnef_node, caffe2_inputs, attrs

    def import_Relu(self, caffe2node):
        caffe2_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Relu(x=nnef_node_x,
                              _uid=self.gen_node_name(caffe2node.output[0]),
                              _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Reshape(self, caffe2node):
        caffe2_inputs = {'input': 0}
        attrs = {}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])

        if len(caffe2node.input) == 1:
            shape = []
            for arg in caffe2node.arg:
                if arg.name == 'shape':
                    shape = list(arg.ints)
        else:
            caffe2_inputs['shape'] = 1
            shape_node = self.get_node_from_pool(caffe2node, 1)
            shape = shape_node.parameters['value']

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

        nnef_node = node.Reshape(input=nnef_node_input,
                                 shape=output_shape,
                                 _uid=self.gen_node_name(caffe2node.output[0]),
                                 _output_shape=output_shape,
                                 _maintain_format=False)

        return nnef_node, caffe2_inputs, attrs

    def import_Sigmoid(self, caffe2node):
        caffe2_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Sigmoid(x=nnef_node_x,
                                 _uid=self.gen_node_name(caffe2node.output[0]),
                                 _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Softmax(self, caffe2node):
        caffe2_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Softmax(x=nnef_node_x,
                                 _uid=self.gen_node_name(caffe2node.output[0]),
                                 _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Softplus(self, caffe2node):
        caffe2_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Softplus(x=nnef_node_x,
                                  _uid=self.gen_node_name(caffe2node.output[0]),
                                  _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_SpatialBN(self, caffe2node):
        caffe2_inputs = {'input':0, 'scale':1, 'offset':2, 'mean':3, 'variance':4}
        attrs = {}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])
        nnef_node_scale = self.get_node_from_pool(caffe2node, caffe2_inputs['scale'])
        nnef_node_offset = self.get_node_from_pool(caffe2node, caffe2_inputs['offset'])
        nnef_node_mean = self.get_node_from_pool(caffe2node, caffe2_inputs['mean'])
        nnef_node_variance = self.get_node_from_pool(caffe2node, caffe2_inputs['variance'])

        for arg in caffe2node.arg:
            if arg.name == 'epsilon':
                epsilon = arg.f

        output_shape = nnef_node_input.output_shape[:]

        nnef_node = node.BatchNormalization(input=nnef_node_input,
                                            mean=nnef_node_mean,
                                            variance=nnef_node_variance,
                                            offset=nnef_node_offset,
                                            scale=nnef_node_scale,
                                            epsilon=epsilon,
                                            _uid=self.gen_node_name(caffe2node.output[0]),
                                            _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Split(self, caffe2node):
        caffe2_inputs = {'value': 0}
        attrs = {'axis': None}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['value'])

        split_name = '['
        output_names = []
        for output in caffe2node.output:
            out_name = self.gen_node_name(output)
            split_name += out_name + ', '
            output_names.append(out_name)
        split_name = split_name[:-2] + ']'

        ratios = []
        for arg in caffe2node.arg:
            if arg.name == 'axis':
                axis = arg.i
            if arg.name == 'split':
                ratios = list(arg.ints)
                attrs['split'] = None

        if not ratios:
            if len(caffe2node.input) == 1:
                ratios = [int(nnef_node_input.output_shape[axis]/len(output_names))]*len(output_names)
            else:
                nnef_node_shape = self.get_node_from_pool(caffe2node, 1)
                caffe2_inputs['ratios'] = 1
                ratios = nnef_node_shape.parameters['value']

        input_shape = nnef_node_input.output_shape[:]

        nnef_node_split = node.Split(value=nnef_node_input,
                                     axis=axis,
                                     ratios=ratios,
                                     _uid=split_name,
                                     _output_shape=input_shape)

        for i in range(len(output_names)):
            out_shape = input_shape[:]
            out_shape[axis] = ratios[i]
            nnef_node = node.OutputVal(base_node=nnef_node_split,
                                       base_index=i,
                                       _uid=output_names[i],
                                       _output_shape=out_shape)
            self.node_pool[nnef_node.name] = nnef_node

        return nnef_node_split, caffe2_inputs, attrs

    def import_Sub(self, caffe2node):
        caffe2_inputs = {'x':0, 'y':1}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        nnef_node_y = self.get_node_from_pool(caffe2node, caffe2_inputs['y'])

        output_shape = self.define_elementwise_binary_output_shape(nnef_node_x, nnef_node_y)

        nnef_node = node.Sub(x=nnef_node_x,
                             y=nnef_node_y,
                             _uid=self.gen_node_name(caffe2node.output[0]),
                             _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Sum(self, caffe2node):
        caffe2_inputs = {}
        attrs = {}

        node_list = []
        for i in range(len(caffe2node.input)):
            node_list.append(self.get_node_from_pool(caffe2node, i))
            caffe2_inputs['input_' + str(i)] = i

        output_shape = node_list[0].output_shape[:]

        nnef_node = node.AddN(x=node_list,
                              _uid=self.gen_node_name(caffe2node.output[0]),
                              _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Tanh(self, caffe2node):
        caffe2_inputs = {'x':0}
        attrs = {}

        nnef_node_x = self.get_node_from_pool(caffe2node, caffe2_inputs['x'])
        output_shape = nnef_node_x.output_shape[:]

        nnef_node = node.Tanh(x=nnef_node_x,
                              _uid=self.gen_node_name(caffe2node.output[0]),
                              _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs

    def import_Transpose(self, caffe2node):
        caffe2_inputs = {'input': 0}
        attrs = {'axes': None}

        nnef_node_input = self.get_node_from_pool(caffe2node, caffe2_inputs['input'])
        axes = list(range(len(nnef_node_input.output_shape)-1, -1, -1))

        for arg in caffe2node.arg:
            if arg.name == 'axes':
                axes = list(arg.ints)

        output_shape = []
        for i in range(len(axes)):
            output_shape.append(nnef_node_input.output_shape[axes[i]])

        nnef_node = node.Transpose(input=nnef_node_input,
                                   axes=axes,
                                   _uid=self.gen_node_name(caffe2node.output[0]),
                                   _output_shape=output_shape)

        return nnef_node, caffe2_inputs, attrs


class Caffe2Exporter(Caffe2Logger, ImporterExporter):
    def __init__(self, output_model):
        super(Caffe2Exporter, self).__init__()
        self.output_model = output_model
        self.value_info = {}

    def run(self, nnef_graph):
        print("Caffe2Exporter runs!")
        self.nxgraph = nnef_graph.get_nx_graph()
        self.generate_caffe2_graph()

    def formatName(self, name):
        index = name.find('_')
        if index != -1:
            newName = name[:index] + '/' + name[index+1:]
        else:
            newName = name
        newName = name.replace('_', '/')
        return newName

    def addInput(self, caffe2node, nnef_node, param_name):
        param_input = nnef_node.parameters[param_name]
        if isinstance(param_input, Node):
            caffe2node.input.extend([param_input.name])
        elif isinstance(param_input, list):
            for element in param_input:
                if isinstance(element, Node):
                    caffe2node.input.extend([element.name])
                else:
                    print("Type: " + str(type(element)) + " is not currently supported for inputs")
        else:
            print("Type: " + str(type(param_input)) + " is not currently supported for inputs")
        return

    def generate_caffe2_graph(self, ):
        print("NNEF graph has %s nodes"%(self.nxgraph.number_of_nodes()))

        self.predict_graph = caffe2_pb2.NetDef()
        self.data_graph = caffe2_pb2.NetDef()

        if 'input' in self.nxgraph:
            self.predict_graph.external_input.extend(['input'])
        else:
            i = 1
            while 'input' + str(i) in self.nxgraph:
                self.predict_graph.external_input.extend(['input' + str(i)])
                i += 1

        if 'output' in self.nxgraph:
            self.predict_graph.external_output.extend(['output'])
        else:
            i = 1
            while 'output' + str(i) in self.nxgraph:
                self.predict_graph.external_output.extend(['output' + str(i)])
                i += 1

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
                print("===> %s doesn't have nnef_node!?" % (nnef_node))

        print("=============")
        print("Caffe2 Graphs Completed!")
        print("Caffe2 graphs have %s nodes"%(len(self.predict_graph.op) + len(self.data_graph.op)))
        model_filename, _ = self.chdir_to_modeldir(self.output_model)

        self.predict_graph.name = 'caffe2_output'

        with open("predict_" + model_filename + 'txt', "w") as f:
            f.write(str(self.predict_graph))

        with open("predict_" + model_filename, "wb") as f:
            f.write(self.predict_graph.SerializeToString())

        if len(self.data_graph.op) != 0:
            with open("init_" + model_filename, "wb") as f:
                f.write(self.data_graph.SerializeToString())

        value_info = json.dumps(self.value_info)
        value_info_file = open("value_info.json", "w")
        value_info_file.write(value_info)
        value_info_file.close()

    def export_UNKNOWN(self, nnef_node):
        print(nnef_node.op + " is currently not supported!\n")
        input()

    def export_add(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Add"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')
        self.addInput(caffe2_node, nnef_node, 'y')

        self.predict_graph.op.extend([caffe2_node])

    def export_add_n(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Sum"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')

        self.predict_graph.op.extend([caffe2_node])

    def export_and(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "And"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')
        self.addInput(caffe2_node, nnef_node, 'y')

        self.predict_graph.op.extend([caffe2_node])

    def export_avg_pool(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "AveragePool"
        caffe2_node.output.extend([nnef_node.name])

        self.addInput(caffe2_node, nnef_node, 'input')

        arg_kernels = caffe2_pb2.Argument()
        arg_kernels.name = "kernels"
        arg_kernels.ints.extend(nnef_node.parameters['size'][2:])
        caffe2_node.arg.extend([arg_kernels])

        arg_dilations = caffe2_pb2.Argument()
        arg_dilations.name = "dilations"
        arg_dilations.ints.extend(nnef_node.parameters['dilation'][2:])
        caffe2_node.arg.extend([arg_dilations])

        arg_strides = caffe2_pb2.Argument()
        arg_strides.name = "strides"
        arg_strides.ints.extend(nnef_node.parameters['stride'][2:])
        caffe2_node.arg.extend([arg_strides])

        if not nnef_node.parameters['padding']:
            input_shape = nnef_node.parameters['input'].output_shape
            output_shape = nnef_node.output_shape
            for i in range(len(input_shape)):
                fd = (nnef_node.parameters['size'][i] - 1) * nnef_node.parameters['dilation'][i] + 1
                t = (output_shape[i] - 1) * nnef_node.parameters['stride'][i] + fd - input_shape[i]
                nnef_node.parameters['padding'].append((math.floor(t/2), math.ceil(t/2)))

        arg_pads = caffe2_pb2.Argument()
        arg_pads.name = "pads"
        all_same = True
        for pad in nnef_node.parameters['padding']:
            if pad[0] != pad[1]:
                all_same = False
                break
        if all_same:
            for pad in nnef_node.parameters['padding'][2:]:
                arg_pads.ints.extend([pad[0]])
                arg_pads.ints.extend([pad[1]])
            caffe2_node.arg.extend([arg_pads])
        else:
            if len(nnef_node.output_shape) == 4:
                arg_pads.name = "pad_l"
                arg_pads.i = nnef_node.parameters['padding'][2][0]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_r"
                arg_pads.i = nnef_node.parameters['padding'][2][1]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_t"
                arg_pads.i = nnef_node.parameters['padding'][3][0]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_b"
                arg_pads.i = nnef_node.parameters['padding'][3][1]
                caffe2_node.arg.extend([arg_pads])
            else:
                raise ValueError("Caffe2 cannot handle non-equal padding for tensors that are not 4-dimensional")

        self.predict_graph.op.extend([caffe2_node])

    def export_batch_normalization(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "SpatialBN"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'input')
        self.addInput(caffe2_node, nnef_node, 'scale')
        self.addInput(caffe2_node, nnef_node, 'offset')
        self.addInput(caffe2_node, nnef_node, 'mean')
        self.addInput(caffe2_node, nnef_node, 'variance')

        arg_eps = caffe2_pb2.Argument()
        arg_eps.name = "epsilon"
        arg_eps.f = nnef_node.parameters['epsilon']
        caffe2_node.arg.extend([arg_eps])

        self.predict_graph.op.extend([caffe2_node])

    def export_concat(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Concat"
        caffe2_node.output.extend([nnef_node.name])
        caffe2_node.output.extend([nnef_node.name + "_dims"])
        self.addInput(caffe2_node, nnef_node, 'values')

        arg_axis = caffe2_pb2.Argument()
        arg_axis.name = "axis"
        arg_axis.i = nnef_node.parameters['axis']
        caffe2_node.arg.extend([arg_axis])

        self.predict_graph.op.extend([caffe2_node])

    def export_constant(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "GivenTensorFill"
        caffe2_node.output.extend([nnef_node.name])

        arg_shape = caffe2_pb2.Argument()
        arg_shape.name = "shape"
        if len(nnef_node.output_shape) == 2 and nnef_node.output_shape[0] == 1:
            arg_shape.ints.extend([nnef_node.output_shape[1]])
        else:
            for shape_val in nnef_node.output_shape:
                arg_shape.ints.extend([shape_val])
        caffe2_node.arg.extend([arg_shape])

        arg_values = caffe2_pb2.Argument()
        arg_values.name = "values"
        if len(nnef_node.parameters["value"]) == 1:
            total_size = 1
            for i in nnef_node.output_shape:
                total_size *= i
            for i in range(total_size):
                arg_values.floats.extend([nnef_node.parameters["value"][0]])
        else:
            for val in nnef_node.parameters["value"]:
                arg_values.floats.extend([val])
        caffe2_node.arg.extend([arg_values])

        self.data_graph.op.extend([caffe2_node])
        self.predict_graph.external_input.extend([caffe2_node.output[0]])

    def export_conv(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Conv"
        caffe2_node.output.extend([nnef_node.name])

        self.addInput(caffe2_node, nnef_node, 'input')
        self.addInput(caffe2_node, nnef_node, 'filter')
        if isinstance(nnef_node.parameters['bias'], Node):
            self.addInput(caffe2_node, nnef_node, 'bias')

        arg_kernels = caffe2_pb2.Argument()
        arg_kernels.name = "kernels"
        arg_kernels.ints.extend(nnef_node.parameters['filter'].output_shape[2:])
        caffe2_node.arg.extend([arg_kernels])

        arg_dilations = caffe2_pb2.Argument()
        arg_dilations.name = "dilations"
        arg_dilations.ints.extend(nnef_node.parameters['dilation'])
        caffe2_node.arg.extend([arg_dilations])

        arg_strides = caffe2_pb2.Argument()
        arg_strides.name = "strides"
        arg_strides.ints.extend(nnef_node.parameters['stride'])
        caffe2_node.arg.extend([arg_strides])

        if not nnef_node.parameters['padding']:
            input_shape = nnef_node.parameters['input'].output_shape
            output_shape = nnef_node.output_shape
            size = nnef_node.parameters['filter'].parameters['shape']
            for i in range(2, len(input_shape)):
                fd = (size[i] - 1) * nnef_node.parameters['dilation'][i-2] + 1
                t = (output_shape[i] - 1) * nnef_node.parameters['stride'][i-2] + fd - input_shape[i]
                nnef_node.parameters['padding'].append((math.floor(t/2), math.ceil(t/2)))

        arg_pads = caffe2_pb2.Argument()
        arg_pads.name = "pads"
        all_same = True
        for pad in nnef_node.parameters['padding']:
            if pad[0] != pad[1]:
                all_same = False
                break
        if all_same:
            for pad in nnef_node.parameters['padding']:
                arg_pads.ints.extend([pad[0]])
                arg_pads.ints.extend([pad[1]])
            caffe2_node.arg.extend([arg_pads])
        else:
            if len(nnef_node.output_shape) == 4:
                arg_pads.name = "pad_l"
                arg_pads.i = nnef_node.parameters['padding'][0][0]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_r"
                arg_pads.i = nnef_node.parameters['padding'][0][1]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_t"
                arg_pads.i = nnef_node.parameters['padding'][1][0]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_b"
                arg_pads.i = nnef_node.parameters['padding'][1][1]
                caffe2_node.arg.extend([arg_pads])
            else:
                raise ValueError("Caffe2 cannot handle non-equal padding for tensors that are not 4-dimensional")

        arg_group = caffe2_pb2.Argument()
        arg_group.name = "group"
        arg_group.i = nnef_node.parameters['groups']
        caffe2_node.arg.extend([arg_group])

        self.predict_graph.op.extend([caffe2_node])

    def export_deconv(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "ConvTranspose"
        caffe2_node.output.extend([nnef_node.name])

        self.addInput(caffe2_node, nnef_node, 'input')
        self.addInput(caffe2_node, nnef_node, 'filter')
        if isinstance(nnef_node.parameters['bias'], Node):
            self.addInput(caffe2_node, nnef_node, 'bias')

        arg_kernels = caffe2_pb2.Argument()
        arg_kernels.name = "kernels"
        arg_kernels.ints.extend(nnef_node.parameters['filter'].output_shape[2:])
        caffe2_node.arg.extend([arg_kernels])

        arg_dilations = caffe2_pb2.Argument()
        arg_dilations.name = "dilations"
        arg_dilations.ints.extend(nnef_node.parameters['dilation'])
        caffe2_node.arg.extend([arg_dilations])

        arg_strides = caffe2_pb2.Argument()
        arg_strides.name = "strides"
        arg_strides.ints.extend(nnef_node.parameters['stride'])
        caffe2_node.arg.extend([arg_strides])

        if not nnef_node.parameters['padding']:
            input_shape = nnef_node.parameters['input'].output_shape
            output_shape = nnef_node.output_shape
            size = nnef_node.parameters['filter'].output_shape
            for i in range(2, len(input_shape)):
                fd = (size[i] - 1) * nnef_node.parameters['dilation'][i-2] + 1
                t = (output_shape[i] - 1) * nnef_node.parameters['stride'][i-2] + fd - input_shape[i]
                nnef_node.parameters['padding'].append((math.floor(t/2), math.ceil(t/2)))

        arg_pads = caffe2_pb2.Argument()
        arg_pads.name = "pads"
        all_same = True
        for pad in nnef_node.parameters['padding']:
            if pad[0] != pad[1]:
                all_same = False
                break
        if all_same:
            for pad in nnef_node.parameters['padding']:
                arg_pads.ints.extend([pad[0]])
            caffe2_node.arg.extend([arg_pads])
        else:
            if len(nnef_node.output_shape) == 4:
                arg_pads.name = "pad_l"
                arg_pads.i = nnef_node.parameters['padding'][0][0]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_r"
                arg_pads.i = nnef_node.parameters['padding'][0][1]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_t"
                arg_pads.i = nnef_node.parameters['padding'][1][0]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_b"
                arg_pads.i = nnef_node.parameters['padding'][1][1]
                caffe2_node.arg.extend([arg_pads])
            else:
                raise ValueError("Caffe2 cannot handle non-equal padding for tensors that are not 4-dimensional")

        arg_group = caffe2_pb2.Argument()
        arg_group.name = "group"
        arg_group.i = nnef_node.parameters['groups']
        caffe2_node.arg.extend([arg_group])

        self.predict_graph.op.extend([caffe2_node])

    def export_elu(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Elu"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')

        self.predict_graph.op.extend([caffe2_node])

    def export_exp(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Exp"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')

        self.predict_graph.op.extend([caffe2_node])

    def export_external(self, nnef_node):
        self.value_info[nnef_node.name] = [1, nnef_node.parameters['shape']]

    def export_ge(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "GE"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')
        self.addInput(caffe2_node, nnef_node, 'y')

        self.predict_graph.op.extend([caffe2_node])

    def export_gt(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "GT"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')
        self.addInput(caffe2_node, nnef_node, 'y')

        self.predict_graph.op.extend([caffe2_node])

    def export_le(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "LE"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')
        self.addInput(caffe2_node, nnef_node, 'y')

        self.predict_graph.op.extend([caffe2_node])

    def export_leaky_relu(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "LeakyRelu"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')

        self.predict_graph.op.extend([caffe2_node])

    def export_linear(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "FC"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'input')
        self.addInput(caffe2_node, nnef_node, 'filter')
        self.addInput(caffe2_node, nnef_node, 'bias')

        self.predict_graph.op.extend([caffe2_node])

    def export_local_response_normalization(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "LRN"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'input')

        arg_alpha = caffe2_pb2.Argument()
        arg_alpha.name = "alpha"
        arg_alpha.f = nnef_node.parameters['alpha']
        caffe2_node.arg.extend([arg_alpha])

        arg_beta = caffe2_pb2.Argument()
        arg_beta.name = "beta"
        arg_beta.f = nnef_node.parameters['beta']
        caffe2_node.arg.extend([arg_beta])

        arg_bias = caffe2_pb2.Argument()
        arg_bias.name = "bias"
        arg_bias.f = nnef_node.parameters['bias']
        caffe2_node.arg.extend([arg_bias])

        arg_size = caffe2_pb2.Argument()
        arg_size.name = "size"
        arg_size.i = nnef_node.parameters['size'][1]
        caffe2_node.arg.extend([arg_size])

        self.predict_graph.op.extend([caffe2_node])

    def export_lt(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "LT"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')
        self.addInput(caffe2_node, nnef_node, 'y')

        self.predict_graph.op.extend([caffe2_node])

    def export_l2_normalization(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Normalize"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'input')

        if len(nnef_node.parameters['axes']) != 1:
            raise AssertionError("Caffe2 accepts a single axis for op: Normalize")
        else:
            arg_axis = caffe2_pb2.Argument()
            arg_axis.name = "axis"
            arg_axis.i = nnef_node.parameters['axes'][0]
            caffe2_node.arg.extend([arg_axis])

        if nnef_node.parameters['bias'] != 0:
            raise ValueError("Caffe2 cannot support non-zero biases for op: Normalize")

        if nnef_node.parameters['epsilon'] != 0:
            raise ValueError("Caffe2 cannot support non-zero epsilon for op: Normalize")

        self.predict_graph.op.extend([caffe2_node])

    def export_matmul(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "MatMul"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'A')
        self.addInput(caffe2_node, nnef_node, 'B')

        if nnef_node.parameters['transposeA']:
            arg_trA = caffe2_pb2.Argument()
            arg_trA.name = "trans_a"
            arg_trA.i = 1
            caffe2_node.arg.extend([arg_trA])

        if nnef_node.parameters['transposeB']:
            arg_trB = caffe2_pb2.Argument()
            arg_trB.name = "trans_b"
            arg_trB.i = 1
            caffe2_node.arg.extend([arg_trB])

        self.predict_graph.op.extend([caffe2_node])

    def export_max(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Max"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')
        self.addInput(caffe2_node, nnef_node, 'y')

        self.predict_graph.op.extend([caffe2_node])

    def export_max_pool(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "MaxPool"
        caffe2_node.output.extend([nnef_node.name])

        self.addInput(caffe2_node, nnef_node, 'input')

        arg_kernels = caffe2_pb2.Argument()
        arg_kernels.name = "kernels"
        arg_kernels.ints.extend(nnef_node.parameters['size'][2:])
        caffe2_node.arg.extend([arg_kernels])

        arg_dilations = caffe2_pb2.Argument()
        arg_dilations.name = "dilations"
        arg_dilations.ints.extend(nnef_node.parameters['dilation'][2:])
        caffe2_node.arg.extend([arg_dilations])

        arg_strides = caffe2_pb2.Argument()
        arg_strides.name = "strides"
        arg_strides.ints.extend(nnef_node.parameters['stride'][2:])
        caffe2_node.arg.extend([arg_strides])

        if not nnef_node.parameters['padding']:
            input_shape = nnef_node.parameters['input'].output_shape
            output_shape = nnef_node.output_shape
            for i in range(len(input_shape)):
                fd = (nnef_node.parameters['size'][i] - 1) * nnef_node.parameters['dilation'][i] + 1
                t = (output_shape[i] - 1) * nnef_node.parameters['stride'][i] + fd - input_shape[i]
                nnef_node.parameters['padding'].append((math.floor(t/2), math.ceil(t/2)))

        arg_pads = caffe2_pb2.Argument()
        arg_pads.name = "pads"
        all_same = True
        for pad in nnef_node.parameters['padding']:
            if pad[0] != pad[1]:
                all_same = False
                break
        if all_same:
            for pad in nnef_node.parameters['padding'][2:]:
                arg_pads.ints.extend([pad[0]])
                arg_pads.ints.extend([pad[1]])
            caffe2_node.arg.extend([arg_pads])
        else:
            if len(nnef_node.output_shape) == 4:
                arg_pads.name = "pad_l"
                arg_pads.i = nnef_node.parameters['padding'][2][0]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_r"
                arg_pads.i = nnef_node.parameters['padding'][2][1]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_t"
                arg_pads.i = nnef_node.parameters['padding'][3][0]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_b"
                arg_pads.i = nnef_node.parameters['padding'][3][1]
                caffe2_node.arg.extend([arg_pads])
            else:
                raise ValueError("Caffe2 cannot handle non-equal padding for tensors that are not 4-dimensional")

        self.predict_graph.op.extend([caffe2_node])

    def export_max_pool_with_index(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "MaxPoolWithIndex"
        caffe2_node.output.extend([nnef_node.name[:nnef_node.name.find(',')]])
        caffe2_node.output.extend([nnef_node.name[nnef_node.name.find(',')+2:]])

        self.addInput(caffe2_node, nnef_node, 'input')

        arg_kernels = caffe2_pb2.Argument()
        arg_kernels.name = "kernels"
        arg_kernels.ints.extend(nnef_node.parameters['size'][2:])
        caffe2_node.arg.extend([arg_kernels])

        arg_dilations = caffe2_pb2.Argument()
        arg_dilations.name = "dilations"
        arg_dilations.ints.extend(nnef_node.parameters['dilation'][2:])
        caffe2_node.arg.extend([arg_dilations])

        arg_strides = caffe2_pb2.Argument()
        arg_strides.name = "strides"
        arg_strides.ints.extend(nnef_node.parameters['stride'][2:])
        caffe2_node.arg.extend([arg_strides])

        if not nnef_node.parameters['padding']:
            input_shape = nnef_node.parameters['input'].output_shape
            output_shape = nnef_node.output_shape
            for i in range(len(input_shape)):
                fd = (nnef_node.parameters['size'][i] - 1) * nnef_node.parameters['dilation'][i] + 1
                t = (output_shape[i] - 1) * nnef_node.parameters['stride'][i] + fd - input_shape[i]
                nnef_node.parameters['padding'].append((math.floor(t/2), math.ceil(t/2)))

        arg_pads = caffe2_pb2.Argument()
        arg_pads.name = "pads"
        all_same = True
        for pad in nnef_node.parameters['padding']:
            if pad[0] != pad[1]:
                all_same = False
                break
        if all_same:
            for pad in nnef_node.parameters['padding'][2:]:
                arg_pads.ints.extend([pad[0]])
                arg_pads.ints.extend([pad[1]])
            caffe2_node.arg.extend([arg_pads])
        else:
            if len(nnef_node.output_shape) == 4:
                arg_pads.name = "pad_l"
                arg_pads.i = nnef_node.parameters['padding'][2][0]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_r"
                arg_pads.i = nnef_node.parameters['padding'][2][1]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_t"
                arg_pads.i = nnef_node.parameters['padding'][3][0]
                caffe2_node.arg.extend([arg_pads])
                arg_pads.name = "pad_b"
                arg_pads.i = nnef_node.parameters['padding'][3][1]
                caffe2_node.arg.extend([arg_pads])
            else:
                raise ValueError("Caffe2 cannot handle non-equal padding for tensors that are not 4-dimensional")

        self.predict_graph.op.extend([caffe2_node])

    def export_mean_reduce(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "ReduceMean"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'input')

        arg_axes = caffe2_pb2.Argument()
        arg_axes.name = "axes"
        arg_axes.ints.extend(nnef_node.parameters['axes'])
        caffe2_node.arg.extend([arg_axes])

        arg_keepdims = caffe2_pb2.Argument()
        arg_keepdims.name = "keepdims"
        arg_keepdims.i = 0
        caffe2_node.arg.extend([arg_keepdims])

        self.predict_graph.op.extend([caffe2_node])

    def export_mul(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Mul"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')
        self.addInput(caffe2_node, nnef_node, 'y')

        self.predict_graph.op.extend([caffe2_node])

    def export_neg(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Negative"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')

        self.predict_graph.op.extend([caffe2_node])

    def export_not(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Not"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')

        self.predict_graph.op.extend([caffe2_node])

    def export_or(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Or"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')
        self.addInput(caffe2_node, nnef_node, 'y')

        self.predict_graph.op.extend([caffe2_node])

    def export_output_val(self, nnef_node):
        return

    def export_relu(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Relu"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')

        self.predict_graph.op.extend([caffe2_node])

    def export_reshape(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Reshape"
        caffe2_node.output.extend([nnef_node.name])
        caffe2_node.output.extend([nnef_node.name + '_shape'])
        self.addInput(caffe2_node, nnef_node, 'input')

        arg_shape = caffe2_pb2.Argument()
        arg_shape.name = "shape"
        arg_shape.ints.extend(nnef_node.parameters['shape'])
        caffe2_node.arg.extend([arg_shape])

        self.predict_graph.op.extend([caffe2_node])

    def export_select(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Conditional"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'condition')
        self.addInput(caffe2_node, nnef_node, 'true_value')
        self.addInput(caffe2_node, nnef_node, 'false_value')

        self.predict_graph.op.extend([caffe2_node])

    def export_sigmoid(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Sigmoid"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')

        self.predict_graph.op.extend([caffe2_node])

    def export_softmax(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Softmax"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')

        if len(nnef_node.parameters['axes']) != 1:
            raise AssertionError("Caffe2 accepts a single axis for op: Softmax")
        else:
            arg_axis = caffe2_pb2.Argument()
            arg_axis.name = "axis"
            arg_axis.i = nnef_node.parameters['axes'][0]
            caffe2_node.arg.extend([arg_axis])

        self.predict_graph.op.extend([caffe2_node])

    def export_softplus(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Softplus"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')

        self.predict_graph.op.extend([caffe2_node])

    def export_split(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Split"
        for output in nnef_node.name[1:-1].split(','):
            caffe2_node.output.extend([output])
        self.addInput(caffe2_node, nnef_node, 'value')

        arg_axis = caffe2_pb2.Argument()
        arg_axis.name = "axis"
        arg_axis.i = nnef_node.parameters['axis']
        caffe2_node.arg.extend([arg_axis])

        arg_split = caffe2_pb2.Argument()
        arg_split.name = "split"
        arg_split.ints.extend(nnef_node.parameters['ratios'])
        caffe2_node.arg.extend([arg_split])

        self.predict_graph.op.extend([caffe2_node])

    def export_sub(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Sub"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')
        self.addInput(caffe2_node, nnef_node, 'y')

        self.predict_graph.op.extend([caffe2_node])

    def export_sum_reduce(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "ReduceSum"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'input')

        arg_axes = caffe2_pb2.Argument()
        arg_axes.name = "axes"
        arg_axes.ints.extend(nnef_node.parameters['axes'])
        caffe2_node.arg.extend([arg_axes])

        arg_keepdims = caffe2_pb2.Argument()
        arg_keepdims.name = "keepdims"
        arg_keepdims.i = 0
        caffe2_node.arg.extend([arg_keepdims])

        self.predict_graph.op.extend([caffe2_node])

    def export_tanh(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Tanh"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'x')

        self.predict_graph.op.extend([caffe2_node])

    def export_transpose(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "Transpose"
        caffe2_node.output.extend([nnef_node.name])
        self.addInput(caffe2_node, nnef_node, 'input')

        arg_axes = caffe2_pb2.Argument()
        arg_axes.name = "axes"
        arg_axes.ints.extend(nnef_node.parameters['axes'])
        caffe2_node.arg.extend([arg_axes])

        self.predict_graph.op.extend([caffe2_node])

    def export_variable(self, nnef_node):
        caffe2_node = caffe2_pb2.OperatorDef()
        caffe2_node.name = ""
        caffe2_node.type = "GivenTensorFill"
        caffe2_node.output.extend([nnef_node.name])

        arg_shape = caffe2_pb2.Argument()
        arg_shape.name = "shape"
        if len(nnef_node.output_shape) == 2 and nnef_node.output_shape[0] == 1:
            arg_shape.ints.extend([nnef_node.output_shape[1]])
        else:
            for shape_val in nnef_node.output_shape:
                arg_shape.ints.extend([shape_val])
        caffe2_node.arg.extend([arg_shape])

        arg_values = caffe2_pb2.Argument()
        arg_values.name = "values"
        np_tensor = nnef_node.tensor.astype(np.float32)
        np_tensor = np.reshape(np_tensor, [-1])
        for val in np_tensor:
            arg_values.floats.extend([val])
        caffe2_node.arg.extend([arg_values])

        self.data_graph.op.extend([caffe2_node])
        self.predict_graph.external_input.extend([caffe2_node.output[0]])
