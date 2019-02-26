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
from .nnef_graph import *
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
        self.with_weights = False

    def get_input_nodes(self):
        input_node_list = []
        if 'input' in self.node_pool.keys():
            input_node_list.append(self.node_pool['input'])
        else:
            i = 1
            while 'input' + str(i) in self.node_pool.keys():
                input_node_list.append(self.node_pool['input' + str(i)])
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

    def get_node_from_pool(self, operation, param):
        if isinstance(operation.inputs[param], str):
            return self.node_pool[operation.inputs[param]]
        else:
            return operation.inputs[param]

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
        try:
            self.with_weights = os.path.isdir(self.input_model)
            if not self.with_weights:
                print("Importing without weights (specify a directory as input-model to import with weights)")
            parser_graph = nnef.load_model(self.input_model)

        except Exception as ex:
            raise Exception('failed to open %s: %s' %
                            (self.input_model, ex))

        self.graph_name = parser_graph.name
        for operation in parser_graph.operations:
            if hasattr(self, "import_" + operation.name):
                func = getattr(self, "import_" + operation.name)
                returned_node = func(operation, parser_graph.tensors)
                self.node_pool[returned_node.name] = returned_node
            else:
                self.import_UNKNOWN(operation, parser_graph.tensors)

        input_nodes = self.get_input_nodes()
        output_nodes = self.get_output_nodes()

        graph = NNEFGraph(os.path.basename(self.input_model).split('.')[0],
                          input_nodes,
                          output_nodes,
                          node_pool=self.node_pool)

        return graph

    def import_abs(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Abs(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_add(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.Add(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_add_n(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = []
        for x_names in operation.inputs['x']:
            node_x.append(self.node_pool[x_names])
        output_shape = node_x[0].output_shape[:]

        return node.AddN(x=node_x,
                         _uid=node_name,
                         _output_shape=output_shape)

    def import_and(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.And(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_area_downsample(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        factors = operation.attribs['factor']

        output_shape = node_input.output_shape[:]
        for i in range(len(factors)):
            output_shape[i + 2] = int(output_shape[i + 2] / factors[i])

        return node.AreaDownsample(input=node_input,
                                   factor=factors,
                                   _uid=node_name,
                                   _output_shape=output_shape)

    def import_avg_pool(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        size = operation.attribs['size']

        output_shape = node_input.output_shape[:]
        if 'border' in operation.attribs:
            border = operation.attribs['border']
        else:
            border = 'constant'
        if 'padding' in operation.attribs:
            padding = operation.attribs['padding']
        else:
            padding = []
        if 'stride' in operation.attribs and operation.attribs['stride'] != []:
            stride = operation.attribs['stride']
        else:
            stride = [1] * len(output_shape)
        if 'dilation' in operation.attribs and operation.attribs['dilation'] != []:
            dilation = operation.attribs['dilation']
        else:
            dilation = [1] * len(output_shape)

        for i in range(len(output_shape)):
            if padding == []:
                output_shape[i] = math.ceil(output_shape[i] / stride[i])
            else:
                fd = (size[i] - 1) * dilation[i] + 1
                output_shape[i] = math.floor((output_shape[i] + padding[i][0] +
                                              padding[i][1] - fd) / stride[i]) + 1

        return node.AvgPool(input=node_input,
                            size=size,
                            border=border,
                            padding=padding,
                            stride=stride,
                            dilation=dilation,
                            _uid=node_name,
                            _output_shape=output_shape)

    def import_batch_normalization(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        node_mean = self.get_node_from_pool(operation, 'mean')
        node_variance = self.get_node_from_pool(operation, 'variance')
        node_offset = self.get_node_from_pool(operation, 'offset')
        node_scale = self.get_node_from_pool(operation, 'scale')
        node_epsilon = self.get_node_from_pool(operation, 'epsilon')
        output_shape = node_input.output_shape[:]

        return node.BatchNormalization(input=node_input,
                                       mean=node_mean,
                                       variance=node_variance,
                                       offset=node_offset,
                                       scale=node_scale,
                                       epsilon=node_epsilon,
                                       _uid=node_name,
                                       _output_shape=output_shape)

    def import_ceil(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Ceil(x=node_x,
                         _uid=node_name,
                         _output_shape=output_shape)

    def import_concat(self, operation, tensors):
        node_name = operation.outputs['value']
        values = []
        for val in operation.inputs['values']:
            values.append(self.node_pool[val])
        axis = operation.attribs['axis']

        output_shape = values[0].output_shape[:]
        for nnef_node in values[1:]:
            output_shape[axis] += nnef_node.output_shape[axis]

        return node.Concat(values=values,
                           axis=axis,
                           _uid=node_name,
                           _output_shape=output_shape)

    def import_constant(self, operation, tensors):
        node_name = operation.outputs['output']
        shape = operation.attribs['shape']
        value = operation.attribs['value']

        return node.Constant(shape=shape,
                             value=value,
                             _uid=node_name,
                             _output_shape=shape)

    def import_conv(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        node_filter = self.get_node_from_pool(operation, 'filter')

        output_shape = node_input.output_shape[:]
        if 'bias' in operation.inputs:
            bias = self.get_node_from_pool(operation, 'bias')
        else:
            bias = 0.0
        if 'border' in operation.attribs:
            border = operation.attribs['border']
        else:
            border = 'constant'
        if 'padding' in operation.attribs:
            padding = operation.attribs['padding']
        else:
            padding = []
        if 'stride' in operation.attribs and operation.attribs['stride'] != []:
            stride = operation.attribs['stride']
        else:
            stride = [1] * (len(output_shape) - 2)
        if 'dilation' in operation.attribs and operation.attribs['dilation'] != []:
            dilation = operation.attribs['dilation']
        else:
            dilation = [1] * (len(output_shape) - 2)
        if 'groups' in operation.attribs:
            groups = operation.attribs['groups']
        else:
            groups = 1

        conv_filter = node_filter.output_shape
        output_shape[1] = conv_filter[0]

        for i in range(2, len(output_shape)):
            if padding == []:
                output_shape[i] = math.ceil(output_shape[i] / stride[i - 2])
            else:
                fd = (conv_filter[i] - 1) * dilation[i - 2] + 1
                pad = padding[i - 2][0] + padding[i - 2][1]
                output_shape[i] = math.floor((output_shape[i] + pad - fd) / stride[i - 2]) + 1

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

    def import_deconv(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        node_filter = self.get_node_from_pool(operation, 'filter')

        output_shape = node_input.output_shape[:]
        if 'bias' in operation.inputs:
            bias = self.get_node_from_pool(operation, 'bias')
        else:
            bias = 0.0
        if 'border' in operation.attribs:
            border = operation.attribs['border']
        else:
            border = 'constant'
        if 'padding' in operation.attribs:
            padding = operation.attribs['padding']
        else:
            padding = []
        if 'stride' in operation.attribs and operation.attribs['stride'] != []:
            stride = operation.attribs['stride']
        else:
            stride = [1] * (len(output_shape) - 2)
        if 'dilation' in operation.attribs and operation.attribs['dilation'] != []:
            dilation = operation.attribs['dilation']
        else:
            dilation = [1] * (len(output_shape) - 2)
        if 'groups' in operation.attribs:
            groups = operation.attribs['groups']
        else:
            groups = 1

        conv_filter = node_filter.output_shape
        output_shape[1] = conv_filter[1]

        for i in range(2, len(output_shape)):
            fd = (conv_filter[i] - 1) * dilation[i - 2] + 1
            if padding == []:
                pad = 0
            else:
                pad = padding[i - 2][0] + padding[i - 2][1]
            output_shape[i] = (output_shape[i] - 1) * stride[i - 2] + fd - pad

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

    def import_div(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.Div(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_elu(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Elu(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_eq(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.EQ(x=node_x,
                       y=node_y,
                       _uid=node_name,
                       _output_shape=output_shape)

    def import_exp(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Exp(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_external(self, operation, tensors):
        node_name = operation.outputs['output']
        nnef_shape = operation.attribs['shape']

        return node.External(shape=nnef_shape,
                             _uid=node_name,
                             _output_shape=nnef_shape)

    def import_floor(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Floor(x=node_x,
                          _uid=node_name,
                          _output_shape=output_shape)

    def import_ge(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.GE(x=node_x,
                       y=node_y,
                       _uid=node_name,
                       _output_shape=output_shape)

    def import_gt(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.GT(x=node_x,
                       y=node_y,
                       _uid=node_name,
                       _output_shape=output_shape)

    def import_le(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.LE(x=node_x,
                       y=node_y,
                       _uid=node_name,
                       _output_shape=output_shape)

    def import_linear(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        node_filter = self.get_node_from_pool(operation, 'filter')
        node_bias = self.get_node_from_pool(operation, 'bias')

        output_shape = [node_input.output_shape[0], node_filter.output_shape[0]]

        return node.Linear(input=node_input,
                           filter=node_filter,
                           bias=node_bias,
                           _uid=node_name,
                           _output_shape=output_shape)

    def import_local_response_normalization(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        size = operation.attribs['size']
        if 'alpha' in operation.attribs:
            alpha = operation.attribs['alpha']
        else:
            alpha = 1.0
        if 'beta' in operation.attribs:
            beta = operation.attribs['beta']
        else:
            beta = 0.5
        if 'bias' in operation.attribs:
            bias = operation.attribs['bias']
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

    def import_log(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Log(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_lt(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.LT(x=node_x,
                       y=node_y,
                       _uid=node_name,
                       _output_shape=output_shape)

    def import_l2_normalization(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        axes = operation.attribs['axes']
        if 'bias' in operation.attribs:
            bias = operation.attribs['bias']
        else:
            bias = 0.0

        output_shape = node_input.output_shape[:]

        return node.L2Normalization(input=node_input,
                                    axes=axes,
                                    bias=bias,
                                    _uid=node_name,
                                    _output_shape=output_shape)

    def import_matmul(self, operation, tensors):
        node_name = operation.outputs['C']
        node_A = self.get_node_from_pool(operation, 'A')
        node_B = self.get_node_from_pool(operation, 'B')
        trA = operation.attribs['transposeA']
        trB = operation.attribs['transposeB']

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

    def import_max(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.Max(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_max_pool(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        size = operation.attribs['size']

        output_shape = node_input.output_shape[:]
        if 'border' in operation.attribs:
            border = operation.attribs['border']
        else:
            border = 'constant'
        if 'padding' in operation.attribs:
            padding = operation.attribs['padding']
        else:
            padding = []
        if 'stride' in operation.attribs and operation.attribs['stride'] != []:
            stride = operation.attribs['stride']
        else:
            stride = [1] * len(output_shape)
        if 'dilation' in operation.attribs and operation.attribs['dilation'] != []:
            dilation = operation.attribs['dilation']
        else:
            dilation = [1] * len(output_shape)

        for i in range(len(output_shape)):
            if padding == []:
                output_shape[i] = math.ceil(output_shape[i] / stride[i])
            else:
                fd = (size[i] - 1) * dilation[i] + 1
                output_shape[i] = math.floor((output_shape[i] + padding[i][0] +
                                              padding[i][1] - fd) / stride[i]) + 1

        return node.MaxPool(input=node_input,
                            size=size,
                            border=border,
                            padding=padding,
                            stride=stride,
                            dilation=dilation,
                            _uid=node_name,
                            _output_shape=output_shape)

    def import_max_pool_with_index(self, operation, tensors):
        node_name = operation.outputs['output'] + ', ' + operation[2]['index']
        node_input = self.get_node_from_pool(operation, 'input')
        size = operation.attribs['size']

        output_shape = node_input.output_shape[:]
        if 'border' in operation.attribs:
            border = operation.attribs['border']
        else:
            border = 'constant'
        if 'padding' in operation.attribs:
            padding = operation.attribs['padding']
        else:
            padding = []
        if 'stride' in operation.attribs and operation.attribs['stride'] != []:
            stride = operation.attribs['stride']
        else:
            stride = [1] * len(output_shape)
        if 'dilation' in operation.attribs and operation.attribs['dilation'] != []:
            dilation = operation.attribs['dilation']
        else:
            dilation = [1] * len(output_shape)

        for i in range(len(output_shape)):
            if padding == []:
                output_shape[i] = math.ceil(output_shape[i] / stride[i])
            else:
                fd = (size[i] - 1) * dilation[i] + 1
                output_shape[i] = math.floor((output_shape[i] + padding[i][0] +
                                              padding[i][1] - fd) / stride[i]) + 1

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
                                      _uid=operation[2]['output'],
                                      _output_shape=output_shape)

        self.node_pool[node_maxpool.name] = node_maxpool

        node_index = node.OutputVal(base_node=base_node,
                                    base_index=1,
                                    _uid=operation[2]['index'],
                                    _output_shape=output_shape)

        self.node_pool[node_index.name] = node_index

        return base_node

    def import_max_reduce(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        axes = operation.attribs['axes']

        output_shape = []
        for i in range(len(node_input.output_shape)):
            if i not in axes:
                output_shape.append(node_input.output_shape[i])

        return node.MaxReduce(input=node_input,
                              axes=axes,
                              _uid=node_name,
                              _output_shape=output_shape)

    def import_mean_reduce(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        axes = operation.attribs['axes']

        output_shape = []
        for i in range(len(node_input.output_shape)):
            if i not in axes:
                output_shape.append(node_input.output_shape[i])

        return node.MeanReduce(input=node_input,
                               axes=axes,
                               _uid=node_name,
                               _output_shape=output_shape)

    def import_min(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.Min(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_mul(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.Mul(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_multilinear_upsample(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        factors = operation.attribs['factor']
        if 'method' in operation.attribs:
            method = operation.attribs['method']
        else:
            method = 'symmetric'
        if 'border' in operation.attribs:
            border = operation.attribs['border']
        else:
            border = 'replicate'

        output_shape = node_input.output_shape[:]
        for i in range(len(factors)):
            output_shape[i + 2] = int(output_shape[i + 2] * factors[i])

        return node.MultilinearUpsample(input=node_input,
                                        factor=factors,
                                        method=method,
                                        border=border,
                                        _uid=node_name,
                                        _output_shape=output_shape)

    def import_ne(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.NE(x=node_x,
                       y=node_y,
                       _uid=node_name,
                       _output_shape=output_shape)

    def import_nearest_downsample(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        factors = operation.attribs['factor']

        output_shape = node_input.output_shape[:]
        for i in range(len(factors)):
            output_shape[i + 2] = int(output_shape[i + 2] / factors[i])

        return node.NearestDownsample(input=node_input,
                                      factor=factors,
                                      _uid=node_name,
                                      _output_shape=output_shape)

    def import_nearest_upsample(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        factors = operation.attribs['factor']

        output_shape = node_input.output_shape[:]
        for i in range(len(factors)):
            output_shape[i + 2] = int(output_shape[i + 2] * factors[i])

        return node.NearestUpsample(input=node_input,
                                    factor=factors,
                                    _uid=node_name,
                                    _output_shape=output_shape)

    def import_neg(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Neg(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_not(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Not(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_or(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.Or(x=node_x,
                       y=node_y,
                       _uid=node_name,
                       _output_shape=output_shape)

    def import_planewise_conv(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        node_filter = self.get_node_from_pool(operation, 'filter')

        output_shape = node_input.output_shape[:]
        if 'bias' in operation.inputs:
            bias = self.get_node_from_pool(operation, 'bias')
        else:
            bias = 0.0
        if 'border' in operation.attribs:
            border = operation.attribs['border']
        else:
            border = 'constant'
        if 'padding' in operation.attribs:
            padding = operation.attribs['padding']
        else:
            padding = []
        if 'stride' in operation.attribs and operation.attribs['stride'] != []:
            stride = operation.attribs['stride']
        else:
            stride = [1] * (len(output_shape) - 2)
        if 'dilation' in operation[1] and operation.attribs['dilation'] != []:
            dilation = operation.attribs['dilation']
        else:
            dilation = [1] * (len(output_shape) - 2)

        conv_filter = node_filter.output_shape
        output_shape[1] = conv_filter[0]

        for i in range(2, len(output_shape)):
            if padding == []:
                output_shape[i] = math.ceil(output_shape[i] / stride[i - 2])
            else:
                fd = (conv_filter[i] - 1) * dilation[i - 2] + 1
                pad = padding[i - 2][0] + padding[i - 2][1]
                output_shape[i] = math.floor((output_shape[i] + pad - fd) / stride[i - 2]) + 1

        return node.PlanewiseConv(input=node_input,
                                  filter=node_filter,
                                  bias=bias,
                                  border=border,
                                  padding=padding,
                                  stride=stride,
                                  dilation=dilation,
                                  _uid=node_name,
                                  _output_shape=output_shape)

    def import_pow(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = node_x.output_shape[:]

        return node.Pow(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_rcp(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Rcp(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_relu(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Relu(x=node_x,
                         _uid=node_name,
                         _output_shape=output_shape)

    def import_reshape(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        shape = operation.attribs['shape']

        return node.Reshape(input=node_input,
                            shape=shape,
                            _uid=node_name,
                            _output_shape=shape)

    def import_round(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Round(x=node_x,
                          _uid=node_name,
                          _output_shape=output_shape)

    def import_rsqrt(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Rsqrt(x=node_x,
                          _uid=node_name,
                          _output_shape=output_shape)

    def import_select(self, operation, tensors):
        node_name = operation.outputs['output']
        node_cond = self.get_node_from_pool(operation, 'condition')
        node_true = self.get_node_from_pool(operation, 'true_value')
        node_false = self.get_node_from_pool(operation, 'false_value')

        output_shape = self.define_elementwise_binary_output_shape(node_true, node_false)

        return node.Select(condition=node_cond,
                           true_value=node_true,
                           false_value=node_false,
                           _uid=node_name,
                           _output_shape=output_shape)

    def import_separable_conv(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        node_plane_filter = self.get_node_from_pool(operation, 'plane_filter')
        node_point_filter = self.get_node_from_pool(operation, 'point_filter')

        output_shape = node_input.output_shape[:]
        if 'bias' in operation.inputs:
            bias = self.get_node_from_pool(operation, 'bias')
        else:
            bias = 0.0
        if 'border' in operation.attribs:
            border = operation.attribs['border']
        else:
            border = 'constant'
        if 'padding' in operation.attribs:
            padding = operation.attribs['padding']
        else:
            padding = []
        if 'stride' in operation.attribs and operation.attribs['stride'] != []:
            stride = operation.attribs['stride']
        else:
            stride = [1] * (len(output_shape) - 2)
        if 'dilation' in operation.attribs and operation.attribs['dilation'] != []:
            dilation = operation.attribs['dilation']
        else:
            dilation = [1] * (len(output_shape) - 2)
        if 'groups' in operation.attribs:
            groups = operation.attribs['groups']
        else:
            groups = 1

        conv_filter = node_plane_filter.output_shape
        output_shape[1] = conv_filter[0]

        for i in range(2, len(output_shape)):
            if padding == []:
                output_shape[i] = math.ceil(output_shape[i] / stride[i - 2])
            else:
                fd = (conv_filter[i] - 1) * dilation[i - 2] + 1
                pad = padding[i - 2][0] + padding[i - 2][1]
                output_shape[i] = math.floor((output_shape[i] + pad - fd) / stride[i - 2]) + 1

        conv_filter = node_point_filter.output_shape
        output_shape[1] = conv_filter[0]

        for i in range(2, len(output_shape)):
            if padding == []:
                output_shape[i] = math.ceil(output_shape[i] / stride[i - 2])
            else:
                fd = (conv_filter[i] - 1) * dilation[i - 2] + 1
                pad = padding[i - 2][0] + padding[i - 2][1]
                output_shape[i] = math.floor((output_shape[i] + pad - fd) / stride[i - 2]) + 1

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

    def import_sigmoid(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Sigmoid(x=node_x,
                            _uid=node_name,
                            _output_shape=output_shape)

    def import_sign(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Sign(x=node_x,
                         _uid=node_name,
                         _output_shape=output_shape)

    def import_slice(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')

        axes = operation.attribs['axes']
        begin = operation.attribs['begin']
        end = operation.attribs['end']

        input_shape = node_input.output_shape
        output_shape = input_shape[:]
        for i in range(len(axes)):
            if i in axes:
                if begin[i] == -1 and end[i] == 0:
                    output_shape[i] = 1
                elif end[i] == 0:
                    output_shape[i] = int(input_shape[i] - begin[i])
                else:
                    output_shape[i] = int(end[i] - begin[i])

        return node.Slice(input=node_input,
                          axes=axes,
                          begin=begin,
                          end=end,
                          _uid=node_name,
                          _output_shape=output_shape)

    def import_softmax(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if 'axes' in operation.attribs:
            axes = operation.attribs['axes']
        else:
            axes = [1]

        output_shape = node_x.output_shape[:]

        return node.Softmax(x=node_x,
                            axes=axes,
                            _uid=node_name,
                            _output_shape=output_shape)

    def import_softplus(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Softplus(x=node_x,
                             _uid=node_name,
                             _output_shape=output_shape)

    # Not in current NNEF Documentation
    def import_softsign(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Softsign(x=node_x,
                             _uid=node_name,
                             _output_shape=output_shape)

    def import_split(self, operation, tensors):
        node_name = '['
        for val_name in operation.outputs['values']:
            node_name += val_name + ', '
        node_name = node_name[:-2] + ']'

        node_value = self.get_node_from_pool(operation, 'value')
        axis = operation.attribs['axis']
        ratios = operation.attribs['ratios']
        input_shape = node_value.output_shape[:]
        total_ratio = 0
        for ratio in ratios:
            total_ratio += ratio
        mu = int(input_shape[axis] / total_ratio)

        base_node = node.Split(value=node_value,
                               axis=axis,
                               ratios=ratios,
                               _uid=node_name,
                               _output_shape=input_shape)

        for i in range(len(operation.outputs['values'])):
            output_shape = input_shape[:]
            output_shape[axis] = ratios[i] * mu
            node_split = node.OutputVal(base_node=base_node,
                                        base_index=i,
                                        _uid=operation.outputs['values'][i],
                                        _output_shape=output_shape)
            self.node_pool[node_split.name] = node_split

        return base_node

    def import_sqr(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Sqr(x=node_x,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_sqrt(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Sqrt(x=node_x,
                         _uid=node_name,
                         _output_shape=output_shape)

    def import_stack(self, operation, tensors):
        node_name = operation.outputs['value']
        values = []
        for val in operation.inputs['values']:
            values.append(self.node_pool[val])

        axis = operation.attribs['axis']

        output_shape = values[0].output_shape[:]
        output_shape.insert(axis, len(values))

        return node.Stack(values=values,
                          axis=axis,
                          _uid=node_name,
                          _output_shape=output_shape)

    def import_sub(self, operation, tensors):
        node_name = operation.outputs['z']
        node_x = self.get_node_from_pool(operation, 'x')
        node_y = self.get_node_from_pool(operation, 'y')
        output_shape = self.define_elementwise_binary_output_shape(node_x, node_y)

        return node.Sub(x=node_x,
                        y=node_y,
                        _uid=node_name,
                        _output_shape=output_shape)

    def import_sum_reduce(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        axes = operation.attribs['axes']
        if 'normalize' in operation.attribs:
            normalize = operation.attribs['normalize']
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

    def import_tanh(self, operation, tensors):
        node_name = operation.outputs['y']
        node_x = self.get_node_from_pool(operation, 'x')
        if isinstance(node_x, Node):
            output_shape = node_x.output_shape
        else:
            output_shape = list(np.shape(node_x))

        return node.Tanh(x=node_x,
                         _uid=node_name,
                         _output_shape=output_shape)

    def import_transpose(self, operation, tensors):
        node_name = operation.outputs['output']
        node_input = self.get_node_from_pool(operation, 'input')
        perm = operation.attribs['axes']

        output_shape = []
        for i in range(len(node_input.output_shape)):
            output_shape.append(node_input.output_shape[perm[i]])

        return node.Transpose(input=node_input,
                              axes=perm,
                              _uid=node_name,
                              _output_shape=output_shape)

    def import_update(self, operation, tensors):
        node_name = operation.outputs['result']
        node_variable = self.get_node_from_pool(operation, 'variable')
        node_value = self.get_node_from_pool(operation, 'value')

        return node.Update(variable=node_variable,
                           value=node_value,
                           _uid=node_name,
                           _output_shape=node_value.output_shape[:])

    def import_variable(self, operation, tensors):
        node_name = operation.outputs['output']
        label = operation.attribs['label']
        shape = operation.attribs['shape']

        if self.with_weights:
            tensor = tensors[node_name].data
            assert tensor is not None
        else:
            nnef_dtype = tensors[node_name].dtype
            if nnef_dtype == "integer":
                tensor = np.random.randint(0, 255, dtype=np.int32)
            elif nnef_dtype == "logical":
                tensor = np.random.rand(*shape).astype(np.float32) > 0.5
            else:
                assert nnef_dtype == "scalar", "Unknown nnef dtype: {}".format(nnef_dtype)
                tensor = np.random.rand(*shape).astype(np.float32)

        nnef_node = node.Variable(label=label,
                                  shape=shape,
                                  _np_dtype=tensor.dtype,
                                  _np_tensor=tensor,
                                  _uid=node_name,
                                  _output_shape=shape)

        return nnef_node

    def import_UNKNOWN(self, operation, tensors):
        assert False, "Missing implementation for node op: %s" % operation.name


class NNEFExporter(ImporterExporter):
    def __init__(self, output_model):
        super(NNEFExporter, self).__init__()
        self.output_model = output_model

    def run(self, nnef_graph):
        self.generate_nnef(nnef_graph)

    def generate_nnef(self, nnef_graph):
        nnef_graph.save_to_file(self.output_model)
