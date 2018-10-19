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

import collections
import math
import os

import nnef
import numpy as np
import networkx as nx

from .importer_exporter import ImporterExporter
from .nnef_graph import  *
from .nnef_data import *
from . import nnef_node as node

class NNEFImporter(ImporterExporter):

    @staticmethod
    def list_to_string(list_input):
        return "[" + ','.join(str(e) for e in list_input) + "]"

    def __init__(self, input_model):
        super(NNEFImporter, self).__init__()

        self.graph_name = ""
        self.input_model = input_model
        self.nxgraph = nx.OrderedDiGraph()
        self.node_pool = collections.OrderedDict()

    def get_input_nodes(self):
        input_node_list = []
        if 'input' in self.node_pool.keys():
            input_node_list.append(self.node_pool['input'])
        else:
            i = 1
            while 'input' + str(i) in self.node_pool.keys():
                input_node_list.append(self.node_pool['input'+str(i)])
                i += 1

        return input_node_list

    def get_output_nodes(self):
        output_node_list = []
        if 'output' in self.node_pool.keys():
            output_node_list.append(self.node_pool['output'])
        else:
            i = 1
            while 'output' + str(i) in self.node_pool.keys():
                output_node_list.append(self.node_pool['output' + str(i)])
                i += 1

        return output_node_list

    def get_node_from_pool(self, operand, param):
        if isinstance(operand[1][param], str):
            return self.node_pool[operand[1][param]]
        else:
            return operand[1][param]

    def remove_node_from_pool(self, node_name):
        self.node_pool.pop(node_name, None)

    def define_elementwise_binary_output_shape(self, node_x, node_y):
        if isinstance(node_x, Node) and not isinstance(node_y, Node):
            return node_x.output_shape[:]
        elif not isinstance(node_x, Node) and isinstance(node_y, Node):
            return node_y.output_shape[:]
        else:
            y_size = x_size = 1
            for i in node_x.output_shape:
                x_size *= i
            for i in node_y.output_shape:
                y_size *= i
            if x_size >= y_size:
                output_shape = node_x.output_shape[:]
            else:
                output_shape = node_y.output_shape[:]
            return output_shape

    def run(self):
        # Changing dir to NNEF's graph parent folder.
        network_dir, model_filename = os.path.split(self.input_model)
        if network_dir != '':
            prevdir = os.getcwd()
            os.chdir(network_dir)

        try:
            attr, ops = nnef.parse_file(model_filename)
        except Exception as ex:
            print('WARNING: converter ignoring exception:', ex)
            nnef._register_layer_ops()
            try:
                attr, ops = nnef.parse_file(model_filename)
            except Exception as ex:
                raise Exception('failed to open %s: %s' %
                                (self.input_model, ex))

        self.graph_name = attr['graph'].name
        for operand in ops:
            if hasattr(self, "import_" + operand[0][0]):
                func = getattr(self, "import_" + operand[0][0])
                returned_node = func(operand)
                self.node_pool[returned_node.name] = returned_node
            else:
                self.import_UNKNOWN(operand)

        input_nodes = self.get_input_nodes()
        output_nodes = self.get_output_nodes()

        graph = NNEFGraph(os.path.basename(self.input_model).split('.')[0],
                          input_nodes,
                          output_nodes,
                          node_pool=self.node_pool)

        if prevdir is not None:
            os.chdir(prevdir)
        return graph

    def import_abs(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Abs(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_add(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.Add(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_add_n(self, operand):
        node_name = operand[1]['y']
        node_x = []
        for x_names in operand[1]['x']:
            node_x.append(self.node_pool[x_names])
        output_shape = node_x[0].output_shape[:]

        return node.AddN(x=node_x,
                         _uid=node_name,
                         _output_shape=output_shape)

    def import_and(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.And(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_area_downsample(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        factors = operand[1]['factor']

        output_shape = node_input.output_shape[:]
        for i in range(len(factors)):
            output_shape[i+2] = int(output_shape[i+2]/factors[i])

        return node.AreaDownsample(input=node_input,
                                   factor=factors,
                                   _uid=node_name,
                                   _output_shape=output_shape)

    def import_avg_pool(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        size = operand[1]['size']

        output_shape = node_input.output_shape[:]
        if 'border' in operand[1]:
            border = operand[1]['border']
        else:
            border = 'constant'
        if 'padding' in operand[1]:
            padding = operand[1]['padding']
        else:
            padding = []
        if 'stride' in operand[1] and operand[1]['stride'] != []:
            stride = operand[1]['stride']
            stride_values = stride
        else:
            stride = []
            stride_values = [1]*len(output_shape)
        if 'dilation' in operand[1] and operand[1]['dilation'] != []:
            dilation = operand[1]['dilation']
            dilation_values = dilation
        else:
            dilation = []
            dilation_values = [1]*len(output_shape)

        for i in range(len(output_shape)):
            if padding == []:
                output_shape[i] = math.ceil(output_shape[i]/stride_values[i])
            else:
                fd = (size[i] - 1)* dilation_values[i] + 1
                output_shape[i] = math.floor((output_shape[i] + padding[i][0] +
                                              padding[i][1] - fd) / stride_values[i]) + 1

        return node.AvgPool(input=node_input,
                            size=size,
                            border=border,
                            padding=padding,
                            stride=stride,
                            dilation=dilation,
                            _uid=node_name,
                            _output_shape=output_shape)

    def import_batch_normalization(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        node_mean = self.get_node_from_pool(operand, 'mean')
        node_variance = self.get_node_from_pool(operand, 'variance')
        node_offset = self.get_node_from_pool(operand, 'offset')
        node_scale = self.get_node_from_pool(operand, 'scale')
        node_epsilon = self.get_node_from_pool(operand, 'epsilon')
        output_shape = node_input.output_shape[:]

        return node.BatchNormalization(input=node_input,
                                       mean=node_mean,
                                       variance=node_variance,
                                       offset=node_offset,
                                       scale=node_scale,
                                       epsilon=node_epsilon,
                                       _uid=node_name,
                                       _output_shape=output_shape)

    def import_ceil(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Ceil(x=node_x,
                         _uid=node_name,
                         _output_shape=output_shape)

    def import_concat(self, operand):
        node_name = operand[1]['value']
        values = []
        for val in operand[1]['values']:
            values.append(self.node_pool[val])
        axis = operand[1]['axis']

        output_shape = values[0].output_shape[:]
        for nnef_node in values[1:]:
            output_shape[axis] += nnef_node.output_shape[axis]

        return node.Concat(values=values,
                           axis=axis,
                           _uid=node_name,
                           _output_shape=output_shape)

    def import_constant(self, operand):
        node_name = operand[1]['output']
        shape = operand[1]['shape']
        value = operand[1]['value']

        return node.Constant(shape=shape,
                             value=value,
                             _uid=node_name,
                             _output_shape=shape)

    def import_conv(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        node_filter = self.get_node_from_pool(operand, 'filter')

        output_shape = node_input.output_shape[:]
        if 'bias' in operand[1]:
            bias = self.get_node_from_pool(operand, 'bias')
        else:
            bias = 0.0
        if 'border' in operand[1]:
            border = operand[1]['border']
        else:
            border = 'constant'
        if 'padding' in operand[1]:
            padding = operand[1]['padding']
        else:
            padding = []
        if 'stride' in operand[1] and operand[1]['stride'] != []:
            stride = operand[1]['stride']
            stride_values = stride
        else:
            stride = []
            stride_values = [1]*(len(output_shape)-2)
        if 'dilation' in operand[1] and operand[1]['dilation'] != []:
            dilation = operand[1]['dilation']
            dilation_values = dilation
        else:
            dilation = []
            dilation_values = [1]*(len(output_shape)-2)
        if 'groups' in operand[1]:
            groups = operand[1]['groups']
        else:
            groups = 1

        conv_filter = node_filter.output_shape
        output_shape[1] = conv_filter[0]

        for i in range(2, len(output_shape)):
            if padding == []:
                output_shape[i] = math.ceil(output_shape[i]/stride_values[i-2])
            else:
                fd = (conv_filter[i] - 1) * dilation_values[i-2] + 1
                pad = padding[i-2][0] + padding[i-2][1]
                output_shape[i] = math.floor((output_shape[i] + pad - fd) / stride_values[i-2]) + 1

        return node.Conv(input=node_input,
                         filter=node_filter,
                         bias=bias,
                         border=border,
                         padding=padding,
                         stride=stride,
                         dilation=dilation,
                         groups=groups,
                         _uid=node_name,
                         _output_shape=output_shape)

    def import_deconv(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        node_filter = self.get_node_from_pool(operand, 'filter')

        output_shape = node_input.output_shape[:]
        if 'bias' in operand[1]:
            bias = self.get_node_from_pool(operand, 'bias')
        else:
            bias = 0.0
        if 'border' in operand[1]:
            border = operand[1]['border']
        else:
            border = 'constant'
        if 'padding' in operand[1]:
            padding = operand[1]['padding']
        else:
            padding = []
        if 'stride' in operand[1] and operand[1]['stride'] != []:
            stride = operand[1]['stride']
            stride_values = stride
        else:
            stride = []
            stride_values = [1]*(len(output_shape)-2)
        if 'dilation' in operand[1] and operand[1]['dilation'] != []:
            dilation = operand[1]['dilation']
            dilation_values = dilation
        else:
            dilation = []
            dilation_values = [1]*(len(output_shape)-2)
        if 'groups' in operand[1]:
            groups = operand[1]['groups']
        else:
            groups = 1

        conv_filter = node_filter.output_shape
        output_shape[1] = conv_filter[1]

        for i in range(2, len(output_shape)):
            fd = (conv_filter[i] - 1) * dilation_values[i-2] + 1
            if padding == []:
                pad = 0
            else:
                pad = padding[i-2][0] + padding[i-2][1]
            output_shape[i] = (output_shape[i] - 1) * stride_values[i-2] + fd - pad

        return node.Deconv(input=node_input,
                           filter=node_filter,
                           bias=bias,
                           border=border,
                           padding=padding,
                           stride=stride,
                           dilation=dilation,
                           groups=groups,
                           _uid=node_name,
                           _output_shape=output_shape)

    def import_div(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.Div(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_elu(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Elu(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_eq(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.EQ(x=node_x,
                       y=node_y,
                       _uid=node_name,
                       _output_shape=output_shape)

    def import_exp(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Exp(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_external(self, operand):
        node_name = operand[1]['output']
        nnef_shape = operand[1]['shape']

        return node.External(shape=nnef_shape,
                             _uid=node_name,
                             _output_shape=nnef_shape)

    def import_floor(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Floor(x=node_x,
                          _uid=node_name,
                          _output_shape=output_shape)

    def import_ge(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.GE(x=node_x,
                       y=node_y,
                       _uid=node_name,
                       _output_shape=output_shape)

    def import_gt(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.GT(x=node_x,
                       y=node_y,
                       _uid=node_name,
                       _output_shape=output_shape)

    def import_le(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.LE(x=node_x,
                       y=node_y,
                       _uid=node_name,
                       _output_shape=output_shape)

    def import_linear(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        node_filter = self.get_node_from_pool(operand, 'filter')
        node_bias = self.get_node_from_pool(operand, 'bias')

        output_shape = [node_input.output_shape[0], node_filter.output_shape[0]]

        return node.Linear(input=node_input,
                           filter=node_filter,
                           bias=node_bias,
                           _uid=node_name,
                           _output_shape=output_shape)

    def import_local_response_normalization(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        size = operand[1]['size']
        if 'alpha' in operand[1]:
            alpha = operand[1]['alpha']
        else:
            alpha = 1.0
        if 'beta' in operand[1]:
            beta = operand[1]['beta']
        else:
            beta = 0.5
        if 'bias' in operand[1]:
            bias = operand[1]['bias']
        else:
            bias = 1.0

        output_shape = node_input.output_shape[:]

        return node.LocalResponseNormalization(input=node_input,
                                               size=size,
                                               alpha=alpha,
                                               beta=beta,
                                               bias=bias,
                                               _uid=node_name,
                                               _output_shape=output_shape)

    def import_log(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Log(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_lt(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.LT(x=node_x,
                       y=node_y,
                       _uid=node_name,
                       _output_shape=output_shape)

    def import_l2_normalization(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        axes = operand[1]['axes']
        if 'bias' in operand[1]:
            bias = operand[1]['bias']
        else:
            bias = 0.0

        output_shape = node_input.output_shape[:]

        return node.L2Normalization(input=node_input,
                                    axes=axes,
                                    bias=bias,
                                    _uid=node_name,
                                    _output_shape=output_shape)

    def import_matmul(self, operand):
        node_name = operand[1]['C']
        node_A = self.get_node_from_pool(operand, 'A')
        node_B = self.get_node_from_pool(operand, 'B')
        trA = operand[1]['transposeA']
        trB = operand[1]['transposeB']

        output_shape = []
        if trA:
            output_shape.append(node_A.output_shape[-1])
        else:
            output_shape.append(node_A.output_shape[0])

        if trB:
            output_shape.append(node_B.output_shape[0])
        else:
            output_shape.append(node_B.output_shape[-1])

        return node.Matmul(A=node_A,
                           B=node_B,
                           transposeA=trA,
                           transposeB=trB,
                           _uid=node_name,
                           _output_shape=output_shape)

    def import_max(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.Max(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_max_pool(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        size = operand[1]['size']

        output_shape = node_input.output_shape[:]
        if 'border' in operand[1]:
            border = operand[1]['border']
        else:
            border = 'constant'
        if 'padding' in operand[1]:
            padding = operand[1]['padding']
        else:
            padding = []
        if 'stride' in operand[1] and operand[1]['stride'] != []:
            stride = operand[1]['stride']
            stride_values = stride
        else:
            stride = []
            stride_values = [1]*len(output_shape)
        if 'dilation' in operand[1] and operand[1]['dilation'] != []:
            dilation = operand[1]['dilation']
            dilation_values = dilation
        else:
            dilation = []
            dilation_values = [1]*len(output_shape)

        for i in range(len(output_shape)):
            if padding == []:
                output_shape[i] = math.ceil(output_shape[i]/stride_values[i])
            else:
                fd = (size[i] - 1)* dilation_values[i] + 1
                output_shape[i] = math.floor((output_shape[i] + padding[i][0] +
                                              padding[i][1] - fd) / stride_values[i]) + 1

        return node.MaxPool(input=node_input,
                            size=size,
                            border=border,
                            padding=padding,
                            stride=stride,
                            dilation=dilation,
                            _uid=node_name,
                            _output_shape=output_shape)

    def import_max_pool_with_index(self, operand):
        node_name = operand[1]['output'] + ', ' + operand[2]['index']
        node_input = self.get_node_from_pool(operand, 'input')
        size = operand[1]['size']

        output_shape = node_input.output_shape[:]
        if 'border' in operand[1]:
            border = operand[1]['border']
        else:
            border = 'constant'
        if 'padding' in operand[1]:
            padding = operand[1]['padding']
        else:
            padding = []
        if 'stride' in operand[1] and operand[1]['stride'] != []:
            stride = operand[1]['stride']
            stride_values = stride
        else:
            stride = []
            stride_values = [1]*len(output_shape)
        if 'dilation' in operand[1] and operand[1]['dilation'] != []:
            dilation = operand[1]['dilation']
            dilation_values = dilation
        else:
            dilation = []
            dilation_values = [1]*len(output_shape)

        for i in range(len(output_shape)):
            if padding == []:
                output_shape[i] = math.ceil(output_shape[i]/stride_values[i])
            else:
                fd = (size[i] - 1)* dilation_values[i] + 1
                output_shape[i] = math.floor((output_shape[i] + padding[i][0] +
                                              padding[i][1] - fd) / stride_values[i]) + 1

        base_node = node.MaxPoolWithIndex(input=node_input,
                                          size=size,
                                          border=border,
                                          padding=padding,
                                          stride=stride,
                                          dilation=dilation,
                                          _uid=node_name,
                                          _output_shape=output_shape)

        node_maxpool = node.OutputVal(base_node=base_node,
                                      base_index=0,
                                      _uid=operand[2]['output'],
                                      _output_shape=output_shape)

        self.node_pool[node_maxpool.name] = node_maxpool

        node_index = node.OutputVal(base_node=base_node,
                                    base_index=1,
                                    _uid=operand[2]['index'],
                                    _output_shape=output_shape)

        self.node_pool[node_index.name] = node_index

        return base_node

    def import_max_reduce(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        axes = operand[1]['axes']

        output_shape = []
        for i in range(len(node_input.output_shape)):
            if i not in axes:
                output_shape.append(node_input.output_shape[i])

        return node.MaxReduce(input=node_input,
                              axes=axes,
                              _uid=node_name,
                              _output_shape=output_shape)

    def import_mean_reduce(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        axes = operand[1]['axes']

        output_shape = []
        for i in range(len(node_input.output_shape)):
            if i not in axes:
                output_shape.append(node_input.output_shape[i])

        return node.MeanReduce(input=node_input,
                               axes=axes,
                               _uid=node_name,
                               _output_shape=output_shape)

    def import_min(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.Min(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_mul(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.Mul(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_multilinear_upsample(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        factors = operand[1]['factor']
        if 'method' in operand[1]:
            method = operand[1]['method']
        else:
            method = 'symmetric'
        if 'border' in operand[1]:
            border = operand[1]['border']
        else:
            border = 'replicate'

        output_shape = node_input.output_shape[:]
        for i in range(len(factors)):
            output_shape[i+2] = int(output_shape[i+2]*factors[i])

        return node.MultilinearUpsample(input=node_input,
                                        factor=factors,
                                        method=method,
                                        border=border,
                                        _uid=node_name,
                                        _output_shape=output_shape)

    def import_ne(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.NE(x=node_x,
                       y=node_y,
                       _uid=node_name,
                       _output_shape=output_shape)

    def import_nearest_downsample(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        factors = operand[1]['factor']

        output_shape = node_input.output_shape[:]
        for i in range(len(factors)):
            output_shape[i+2] = int(output_shape[i+2]/factors[i])

        return node.NearestDownsample(input=node_input,
                                      factor=factors,
                                      _uid=node_name,
                                      _output_shape=output_shape)

    def import_nearest_upsample(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        factors = operand[1]['factor']

        output_shape = node_input.output_shape[:]
        for i in range(len(factors)):
            output_shape[i+2] = int(output_shape[i+2]*factors[i])

        return node.NearestUpsample(input=node_input,
                                    factor=factors,
                                    _uid=node_name,
                                    _output_shape=output_shape)

    def import_neg(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Neg(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_not(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Not(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_or(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.Or(x=node_x,
                       y=node_y,
                       _uid=node_name,
                       _output_shape=output_shape)

    def import_planewise_conv(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        node_filter = self.get_node_from_pool(operand, 'filter')

        output_shape = node_input.output_shape[:]
        if 'bias' in operand[1]:
            bias = self.get_node_from_pool(operand, 'bias')
        else:
            bias = 0.0
        if 'border' in operand[1]:
            border = operand[1]['border']
        else:
            border = 'constant'
        if 'padding' in operand[1]:
            padding = operand[1]['padding']
        else:
            padding = []
        if 'stride' in operand[1] and operand[1]['stride'] != []:
            stride = operand[1]['stride']
            stride_values = stride
        else:
            stride = []
            stride_values = [1]*(len(output_shape)-2)
        if 'dilation' in operand[1] and operand[1]['dilation'] != []:
            dilation = operand[1]['dilation']
            dilation_values = dilation
        else:
            dilation = []
            dilation_values = [1]*(len(output_shape)-2)

        conv_filter = node_filter.output_shape
        output_shape[1] = conv_filter[0]

        for i in range(2, len(output_shape)):
            if padding == []:
                output_shape[i] = math.ceil(output_shape[i]/stride_values[i-2])
            else:
                fd = (conv_filter[i] - 1) * dilation_values[i-2] + 1
                pad = padding[i-2][0] + padding[i-2][1]
                output_shape[i] = math.floor((output_shape[i] + pad - fd) / stride_values[i-2]) + 1

        return node.PlanewiseConv(input=node_input,
                                  filter=node_filter,
                                  bias=bias,
                                  border=border,
                                  padding=padding,
                                  stride=stride,
                                  dilation=dilation,
                                  _uid=node_name,
                                  _output_shape=output_shape)

    def import_pow(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = node_x.output_shape[:]

        return node.Pow(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_rcp(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Rcp(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_relu(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Relu(x=node_x,
                         _uid=node_name,
                         _output_shape=output_shape)

    def import_reshape(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        shape = operand[1]['shape']

        return node.Reshape(input=node_input,
                            shape=shape,
                            _uid=node_name,
                            _output_shape=shape)

    def import_round(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Round(x=node_x,
                          _uid=node_name,
                          _output_shape=output_shape)

    def import_rsqrt(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Rsqrt(x=node_x,
                          _uid=node_name,
                          _output_shape=output_shape)

    def import_select(self, operand):
        node_name = operand[1]['output']
        node_cond = self.get_node_from_pool(operand, 'condition')
        node_true = self.get_node_from_pool(operand, 'true_value')
        node_false = self.get_node_from_pool(operand, 'false_value')

        output_shape = self.define_elementwise_binary_output_shape(node_true, node_false)

        return node.Select(condition=node_cond,
                           true_value=node_true,
                           false_value=node_false,
                           _uid=node_name,
                           _output_shape=output_shape)

    def import_separable_conv(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        node_plane_filter = self.get_node_from_pool(operand, 'plane_filter')
        node_point_filter = self.get_node_from_pool(operand, 'point_filter')

        output_shape = node_input.output_shape[:]
        if 'bias' in operand[1]:
            bias = self.get_node_from_pool(operand, 'bias')
        else:
            bias = 0.0
        if 'border' in operand[1]:
            border = operand[1]['border']
        else:
            border = 'constant'
        if 'padding' in operand[1]:
            padding = operand[1]['padding']
        else:
            padding = []
        if 'stride' in operand[1] and operand[1]['stride'] != []:
            stride = operand[1]['stride']
            stride_values = stride
        else:
            stride = []
            stride_values = [1]*(len(output_shape)-2)
        if 'dilation' in operand[1] and operand[1]['dilation'] != []:
            dilation = operand[1]['dilation']
            dilation_values = dilation
        else:
            dilation = []
            dilation_values = [1]*(len(output_shape)-2)
        if 'groups' in operand[1]:
            groups = operand[1]['groups']
        else:
            groups = 1

        conv_filter = node_plane_filter.output_shape
        output_shape[1] = conv_filter[0]

        for i in range(2, len(output_shape)):
            if padding == []:
                output_shape[i] = math.ceil(output_shape[i]/stride_values[i-2])
            else:
                fd = (conv_filter[i] - 1) * dilation_values[i-2] + 1
                pad = padding[i-2][0] + padding[i-2][1]
                output_shape[i] = math.floor((output_shape[i] + pad - fd) / stride_values[i-2]) + 1

        conv_filter = node_point_filter.output_shape
        output_shape[1] = conv_filter[0]

        for i in range(2, len(output_shape)):
            if padding == []:
                output_shape[i] = math.ceil(output_shape[i]/stride_values[i-2])
            else:
                fd = (conv_filter[i] - 1) * dilation_values[i-2] + 1
                pad = padding[i-2][0] + padding[i-2][1]
                output_shape[i] = math.floor((output_shape[i] + pad - fd) / stride_values[i-2]) + 1

        return node.SeparableConv(input=node_input,
                                  plane_filter=node_plane_filter,
                                  point_filter=node_point_filter,
                                  bias=bias,
                                  border=border,
                                  padding=padding,
                                  stride=stride,
                                  dilation=dilation,
                                  groups=groups,
                                  _uid=node_name,
                                  _output_shape=output_shape)

    def import_sigmoid(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Sigmoid(x=node_x,
                            _uid=node_name,
                            _output_shape=output_shape)

    def import_sign(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Sign(x=node_x,
                         _uid=node_name,
                         _output_shape=output_shape)

    def import_slice(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')

        axes = operand[1]['axes']
        begin = operand[1]['begin']
        end = operand[1]['end']

        input_shape = node_input.output_shape
        output_shape = input_shape[:]
        for i in range(len(axes)):
            if i in axes:
                if begin[i] == -1 and end[i] == 0:
                    output_shape[i] = 1
                elif end[i] == 0:
                    output_shape[i] = int(input_shape[i]-begin[i])
                else:
                    output_shape[i] = int(end[i]-begin[i])

        return node.Slice(input=node_input,
                          axes=axes,
                          begin=begin,
                          end=end,
                          _uid=node_name,
                          _output_shape=output_shape)

    def import_softmax(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if 'axes' in operand[1]:
            axes = operand[1]['axes']
        else:
            axes = [1]

        output_shape = node_x.output_shape[:]

        return node.Softmax(x=node_x,
                            axes=axes,
                            _uid=node_name,
                            _output_shape=output_shape)

    def import_softplus(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Softplus(x=node_x,
                             _uid=node_name,
                             _output_shape=output_shape)

    #Not in current NNEF Documentation
    def import_softsign(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Softsign(x=node_x,
                             _uid=node_name,
                             _output_shape=output_shape)

    def import_split(self, operand):
        node_name = '['
        for val_name in operand[1]['values']:
            node_name += val_name + ', '
        node_name = node_name[:-2] + ']'

        node_value = self.get_node_from_pool(operand, 'value')
        axis = operand[1]['axis']
        ratios = operand[1]['ratios']
        input_shape = node_value.output_shape[:]
        total_ratio = 0
        for ratio in ratios:
            total_ratio += ratio
        mu = int(input_shape[axis]/total_ratio)

        base_node = node.Split(value=node_value,
                               axis=axis,
                               ratios=ratios,
                               _uid=node_name,
                               _output_shape=input_shape)

        for i in range(len(operand[1]['values'])):
            output_shape = input_shape[:]
            output_shape[axis] = ratios[i] * mu
            node_split = node.OutputVal(base_node=base_node,
                                        base_index=i,
                                        _uid=operand[1]['values'][i],
                                        _output_shape=output_shape)
            self.node_pool[node_split.name] = node_split

        return base_node

    def import_sqr(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Sqr(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_sqrt(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Sqrt(x=node_x,
                         _uid=node_name,
                         _output_shape=output_shape)

    def import_stack(self, operand):
        node_name = operand[1]['value']
        values = []
        for val in operand[1]['values']:
            values.append(self.node_pool[val])

        axis = operand[1]['axis']

        output_shape = values[0].output_shape[:]
        output_shape.insert(axis, len(values))

        return node.Stack(values=values,
                          axis=axis,
                          _uid=node_name,
                          _output_shape=output_shape)

    def import_sub(self, operand):
        node_name = operand[1]['z']
        node_x = self.get_node_from_pool(operand, 'x')
        node_y = self.get_node_from_pool(operand, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.Sub(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_sum_reduce(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        axes = operand[1]['axes']
        if 'normalize' in operand[1]:
            normalize = operand[1]['normalize']
        else:
            normalize = False

        output_shape = []
        for i in range(len(node_input.output_shape)):
            if i not in axes:
                output_shape.append(node_input.output_shape[i])

        return node.SumReduce(input=node_input,
                              axes=axes,
                              normalize=normalize,
                              _uid=node_name,
                              _output_shape=output_shape)

    def import_tanh(self, operand):
        node_name = operand[1]['y']
        node_x = self.get_node_from_pool(operand, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Tanh(x=node_x,
                         _uid=node_name,
                         _output_shape=output_shape)

    def import_transpose(self, operand):
        node_name = operand[1]['output']
        node_input = self.get_node_from_pool(operand, 'input')
        perm = operand[1]['axes']

        output_shape = []
        for i in range(len(node_input.output_shape)):
            output_shape.append(node_input.output_shape[perm[i]])

        return node.Transpose(input=node_input,
                              axes=perm,
                              _uid=node_name,
                              _output_shape=output_shape)

    def import_update(self, operand):
        node_name = operand[1]['result']
        node_variable = self.get_node_from_pool(operand, 'variable')
        node_value = self.get_node_from_pool(operand, 'value')

        return node.Update(variable=node_variable,
                           value=node_value,
                           _uid=node_name,
                           _output_shape=node_value.output_shape[:])

    def import_variable(self, operand):
        node_name = operand[1]['output']
        label = operand[1]['label']
        shape = operand[1]['shape']

        no_datfile = False

        try:
            tdf = TensorDataFile()
            tdf.read_from_disk(label + '.dat')

            [nnef_tensor, nnef_type] = tdf.get_data().get_array()

        except IOError:
            nnef_tensor = np.random.rand(*shape).astype(np.float32)
            nnef_type = nnef_tensor.dtype
            no_datfile = True

        nnef_node = node.Variable(label=label,
                                  shape=shape,
                                  _np_dtype=nnef_type,
                                  _np_tensor=nnef_tensor,
                                  _uid=node_name,
                                  _output_shape=shape)

        if no_datfile:
            nnef_node.tensor_data_file.write_to_disk(nnef_node.parameters['label'] + '.dat')

        return nnef_node

    def import_UNKNOWN(self, op):
        assert False, "Missing implementation for node op: %s"%op[0]

class NNEFExporter(ImporterExporter):
    def __init__(self, output_model):
        super(NNEFExporter, self).__init__()
        self.output_model = output_model

    def run(self, nnef_graph):
        self.generate_nnef(nnef_graph)

    def generate_nnef(self, nnef_graph):
        nnef_graph.save_to_file(self.output_model)
