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

import numpy as np


class Node(object):
    type_name = set([
        "tensor",
        "extent",
        "scalar",
        "logical",
        "string"
    ])

    type_nnef_primitive = set([
        'tensor',
        'variable',
        'external',
        'constant',
        'concat',
        'split',
        'reshape',
        'transpose',
        'idn',
        'add',
        'sub',
        'mul',
        'div',
        'pow',
        'and',
        'or',
        'not',
        'neg',
        'abs',
        'sign',
        'exp',
        'log',
        'sqrt',
        'rsqrt',
        'sqr',
        'floor',
        'ceil',
        'round',
        'gt',
        'ge',
        'lt',
        'le',
        'eq',
        'ne',
        'min',
        'max',
        'select',
        'update',
        'matmul',
        'sum_reduce',
        'mean_reduce',
        'max_reduce',
        'sigmoid',
        'tanh',
        'elu',
        'relu',
        'leaky_relu',
        'stack',
        'shape_of',
        'slice',
        'softsign',
        'softplus',
        'softmax',
        'conv',
        'deconv',
        'planewise_conv',
        'max_pool',
        'max_pool_with_index',
        'avg_pool',
        'local_response_normalization',
        'batch_normalization',
        'l2_normalization',
        'linear',
        'add_n',
        'multilinear_upsample',
        'nearest_upsample',
        'nearest_downsample',
        'area_downsample',
        'gru',
        'pad',
        'output_val'])

    array_type_spec = set([])

    def __init__(self, nnef_primitive, **kwargs):
        assert '_uid' in kwargs
        assert '/' not in kwargs['_uid'], "Node uid shouldn't contain '/' characters"
        assert ':' not in kwargs['_uid'], "Node uid shouldn't contain ':' characters"

        self.unique_name = kwargs['_uid']
        self.parameters_ = {}
        self.output_shape = None
        self.tensor = None
        self.output_value_ = None
        if not hasattr(self, 'defaults'):
            self.defaults = {}

        if nnef_primitive in self.type_nnef_primitive:
            self.nnef_primitive_ = nnef_primitive
        else:
            print("Creating Node with unsupported primitive: %s"%nnef_primitive)
            assert False

        for key, value in kwargs.items():
            if key not in ['_uid', '_output_shape', '_np_tensor']:
                self.set_parameter(key, value)
            elif key == '_output_shape':
                self.output_shape = value

        for key, value in self.defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value

    def list_to_string(self, list_):
        return "[" + ','.join(str(e) for e in list_) + "]"

    def nnef_node_definition(self):
        param_list = ""
        for key, val in self.parameters.items():

            if key in self.defaults and val == self.defaults[key]:
                continue
            if key.startswith("_"):
                continue

            # Reformat as strings
            if isinstance(val, Node):
                val = val.name

            if key == 'label':
                val = "'" + val + "'"
            elif key == 'shape' or \
                 key == 'size' or \
                 key == 'perm' or \
                 key == 'axis' or \
                 key == 'axes':
                if isinstance(val, list):
                    val = self.list_to_string(val)
                else:
                    val = str(val)
            elif key == 'value':
                if isinstance(self.parameters['value'], Node):
                    val = self.parameters['value'].name
                else:
                    val = str(self.parameters['value'])
            elif isinstance(self.parameters[key], list) and isinstance(self.parameters[key][0], Node):
                new_val = '['
                for value in val:
                    new_val += value.name + ', '
                val = new_val[:-2] + ']'
            elif key == 'scope':
                val = "'" + val + "'"

            if isinstance(val, bool):
                val = 'true' if val else 'false'

            if self.nnef_primitive_ == 'shape_of':
                param_list += '{v}, '.format(v=val)
            else:
                param_list += '{k} = {v}, '.format(k=key, v=val)

        if len(param_list) > 2:
            param_list = param_list[:-2]

        return '{n} = {self.nnef_primitive_}({params})'.format(self=self, n=self.unique_name.lower(), params=param_list)

    def get_value(self):
        print("Trying to get value of unsupported op: %s"%(self.__class__.__name__))
        input()

    @property
    def nnef_version(self):
        return '1.0-provisional'

    @property
    def keyword(self):
        return 'node'

    @property
    def op(self):
        return self.nnef_primitive_

    @property
    def output_value(self):
        return self.output_value_

    @property
    def name(self):
        return self.unique_name

    @property
    def parameters(self):
        return self.parameters_

    def set_name(self, name):
        self.unique_name = name

    def print_parameters(self):
        print("DBG: Printing from parameters_ dict")
        for key, val in self.parameters_.items():
            print(key, ":", val)

    def set_parameter(self, key, param):
        self.parameters_[key] = param

    # This function gets called if no compile implementation is found for nodes
    # Nodes with compile() implementations should check for their callback fct...
    def compile(self, nx_graph):
        if '_compile_callback' in self.parameters:
            self.parameters['_compile_callback'](nx_graph, self)
            return

    def add_edges(self, nx_graph, param_inputs):
        for param in param_inputs:
            if isinstance(self.parameters[param], Node):
                if self.parameters[param].__class__.__name__ == 'Idn':
                    self.parameters[param] = self.parameters[param].parameters['x']
                nx_graph.add_edge(self.parameters[param].name, self.name)
            elif isinstance(self.parameters[param], list):
                for i in range(len(self.parameters[param])):
                    if isinstance(self.parameters[param][i], Node):
                        if self.parameters[param][i].__class__.__name__ == 'Idn':
                            self.parameters[param][i] = self.parameters[param][i].parameters['x']
                        nx_graph.add_edge(self.parameters[param][i].name, self.name)


'''
     NNEF 1.0 provisional operations:
     Framework specific/'vendor specific' operations to be implemented by subclassing Node
     into the framework specific converter file.
'''


class Abs(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value" % (self.__class__.__name__)
        super(Abs, self).__init__('abs', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Add(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(Add, self).__init__('add', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, **kwargs):
        import tensorflow as tf
        nx_graph = kwargs['nx_graph']
        x = tf.constant(nx_graph.node[self.parameters['x'].name]['node'].output_value)
        y = tf.constant(nx_graph.node[self.parameters['y'].name]['node'].output_value)
        self.output_value_ = tf.add(x, y).numpy()
        return self.output_value_


class AddN(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        super(AddN, self).__init__('add_n', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class And(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(And, self).__init__('and', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, nx_graph):
        self.output_value_ = None


class AreaDownsample(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'factor' in kwargs, "NNEF %s node is missing 'factor' value"%(self.__class__.__name__)
        super(AreaDownsample, self).__init__('area_downsample', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None


class AvgPool(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'size' in kwargs, "NNEF %s node is missing 'size' value"%(self.__class__.__name__)
        self.defaults = {'border': 'constant', 'padding': [],
                         'stride': [], 'dilation': []}
        super(AvgPool, self).__init__('avg_pool', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None


class BatchNormalization(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'mean' in kwargs, "NNEF %s node is missing 'mean' value"%(self.__class__.__name__)
        assert 'variance' in kwargs, "NNEF %s node is missing 'variance' value"%(self.__class__.__name__)
        assert 'offset' in kwargs, "NNEF %s node is missing 'offset' value"%(self.__class__.__name__)
        assert 'scale' in kwargs, "NNEF %s node is missing 'scale' value"%(self.__class__.__name__)
        assert 'epsilon' in kwargs, "NNEF %s node is missing 'epsilon' value"%(self.__class__.__name__)
        super(BatchNormalization, self).__init__('batch_normalization', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input', 'mean', 'variance', 'offset', 'scale', 'epsilon'])

    def run(self, nx_graph):
        self.output_value_ = None


class Ceil(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value" % (self.__class__.__name__)
        super(Ceil, self).__init__('ceil', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Concat(Node):
    def __init__(self, **kwargs):
        assert 'values' in kwargs, "NNEF %s node is missing 'values' value"%(self.__class__.__name__)
        assert 'axis' in kwargs, "NNEF %s node is missing 'axis' value"%(self.__class__.__name__)
        assert isinstance(kwargs['values'], list), "NNEF %s node, 'values' value is not of 'list' type"%(self.__class__.__name__)
        super(Concat, self).__init__('concat', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['values'])

    def run(self, **kwargs):
        self.output_value_ = None


class Constant(Node):
    def __init__(self, **kwargs):
        assert 'value' in kwargs, "NNEF %s node is missing 'value' value"%(self.__class__.__name__)
        assert 'shape' in kwargs, "NNEF %s node is missing 'shape' value"%(self.__class__.__name__)
        assert isinstance(kwargs['shape'], list), "NNEF %s node, 'shape' value is not of 'list' type"%(self.__class__.__name__)
        super(Constant, self).__init__('constant', **kwargs)

    def run(self, **kwargs):
        self.output_value_ = self.parameters['value']
        return self.output_value_

    def get_value(self):
        if len(self.parameters['shape']) == 2 and self.parameters['shape'][0] == 1:
            return np.reshape(np.asarray(self.parameters['value']), self.parameters['shape'][1:])
        else:
            return np.reshape(np.asarray(self.parameters['value']), self.parameters['shape'])


class Conv(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'filter' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        self.defaults = {'bias': 0.0, 'border': 'constant', 'padding': [],
                         'stride': [], 'dilation': [], 'groups': 1}
        super(Conv, self).__init__('conv', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        if 'bias' in self.parameters and isinstance(self.parameters['bias'], Node):
            self.add_edges(nx_graph, ['input', 'filter', 'bias'])
        else:
            self.add_edges(nx_graph, ['input', 'filter'])

    def run(self, nx_graph):
        self.output_value_ = None


class Deconv(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'filter' in kwargs, "NNEF %s node is missing 'filter' value"%(self.__class__.__name__)
        self.defaults = {'bias': 0.0, 'border': 'constant', 'padding': [],
                         'stride': [], 'dilation': [], 'groups': 1}
        super(Deconv, self).__init__('deconv', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        if 'bias' in self.parameters and isinstance(self.parameters['bias'], Node):
            self.add_edges(nx_graph, ['input', 'filter', 'bias'])
        else:
            self.add_edges(nx_graph, ['input', 'filter'])

    def run(self, nx_graph):
        self.output_value_ = None


class Div(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(Div, self).__init__('div', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, nx_graph):
        self.output_value_ = None


class Elu(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        super(Elu, self).__init__('elu', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class EQ(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(EQ, self).__init__('eq', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, nx_graph):
        self.output_value_ = None


class Exp(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value" % (self.__class__.__name__)
        super(Exp, self).__init__('exp', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class External(Node):
    def __init__(self, **kwargs):
        assert 'shape' in kwargs, "NNEF External node is missing 'shape' value"
        assert isinstance(kwargs['shape'], list), "NNEF External node, 'shape' must be a list"
        for shape_item in kwargs['shape']:
            assert shape_item > 0, "NNEF External node, 'shape' values must be positive"
        super(External, self).__init__('external', **kwargs)

    def run(self, **kwargs):
        self.output_value_ = kwargs['external_tensor']
        return self.output_value_


class Floor(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value" % (self.__class__.__name__)
        super(Floor, self).__init__('floor', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class GE(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(GE, self).__init__('ge', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, nx_graph):
        self.output_value_ = None


class Gru(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'channels' in kwargs, "NNEF %s node is missing 'channels' value"%(self.__class__.__name__)
        assert 'scope' in kwargs, "NNEF %s node is missing 'scope' value"%(self.__class__.__name__)
        super(Gru, self).__init__('gru', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None


class GT(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(GT, self).__init__('gt', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, nx_graph):
        self.output_value_ = None


class Idn(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        super(Idn, self).__init__('idn', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)

    def run(self, nx_graph):
        self.output_value_ = None


class LE(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(LE, self).__init__('le', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, nx_graph):
        self.output_value_ = None


class LeakyRelu(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'alpha' in kwargs, "NNEF %s node is missing 'alpha' value"%(self.__class__.__name__)
        super(LeakyRelu, self).__init__('leaky_relu', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Linear(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'filter' in kwargs, "NNEF %s node is missing 'filter' value"%(self.__class__.__name__)
        self.defaults = {'bias': 0.0}
        super(Linear, self).__init__('linear', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        if 'bias' in self.parameters and isinstance(self.parameters['bias'], Node):
            self.add_edges(nx_graph, ['input', 'filter', 'bias'])
        else:
            self.add_edges(nx_graph, ['input', 'filter'])

    def run(self, nx_graph):
        self.output_value_ = None


class LocalResponseNormalization(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'size' in kwargs, "NNEF %s node is missing 'size' value"%(self.__class__.__name__)
        self.defaults = {'alpha': 1.0, 'beta': 0.5, 'bias': 1.0}
        super(LocalResponseNormalization, self).__init__('local_response_normalization', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input', 'size', 'alpha', 'beta', 'bias'])

    def run(self, nx_graph):
        self.output_value_ = None


class Log(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value" % (self.__class__.__name__)
        super(Log, self).__init__('log', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class LT(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(LT, self).__init__('lt', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, nx_graph):
        self.output_value_ = None


class L2Normalization(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'axes' in kwargs, "NNEF %s node is missing 'axes' value"%(self.__class__.__name__)
        self.defaults = {'bias': 0.0, 'epsilon': 0.0}
        super(L2Normalization, self).__init__('l2_normalization', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        if 'bias' in self.parameters and isinstance(self.parameters['bias'], Node):
            self.add_edges(nx_graph, ['input', 'bias'])
        else:
            self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None


class Matmul(Node):
    def __init__(self, **kwargs):
        assert 'A' in kwargs, "NNEF %s node is missing 'A' value"%(self.__class__.__name__)
        assert 'B' in kwargs, "NNEF %s node is missing 'B' value"%(self.__class__.__name__)
        if 'transposeA' in kwargs:
            assert isinstance(kwargs['transposeA'], bool)
        if 'transposeB' in kwargs:
            assert isinstance(kwargs['transposeB'], bool)
        self.defaults = {'transposeA': False, 'transposeB': False}
        super(Matmul, self).__init__('matmul', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['A', 'B'])

    def run(self, nx_graph):
        self.output_value_ = None


class Max(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(Max, self).__init__('max', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, nx_graph):
        self.output_value_ = None


class MaxPool(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'size' in kwargs, "NNEF %s node is missing 'size' value"%(self.__class__.__name__)
        self.defaults = {'border': 'constant', 'padding': [], 'stride': [], 'dilation': []}
        super(MaxPool, self).__init__('max_pool', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None


class MaxPoolWithIndex(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'size' in kwargs, "NNEF %s node is missing 'size' value"%(self.__class__.__name__)
        self.defaults = {'border': 'constant', 'padding': [], 'stride': [], 'dilation': []}
        super(MaxPoolWithIndex, self).__init__('max_pool_with_index', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None


class MaxReduce(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value" % (self.__class__.__name__)
        assert 'axes' in kwargs, "NNEF %s node is missing 'axes' value" % (self.__class__.__name__)
        super(MaxReduce, self).__init__('max_reduce', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None


class MeanReduce(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value" % (self.__class__.__name__)
        assert 'axes' in kwargs, "NNEF %s node is missing 'axes' value" % (self.__class__.__name__)
        super(MeanReduce, self).__init__('mean_reduce', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None


class Min(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(Min, self).__init__('min', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, nx_graph):
        self.output_value_ = None


class Mul(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(Mul, self).__init__('mul', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, nx_graph):
        self.output_value_ = None


class MultilinearUpsample(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'factor' in kwargs, "NNEF %s node is missing 'factor' value"%(self.__class__.__name__)
        self.defaults = {'method': 'symmetric', 'border': 'replicate'}
        super(MultilinearUpsample, self).__init__('multilinear_upsample', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None


class NE(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(NE, self).__init__('ne', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, nx_graph):
        self.output_value_ = None


class NearestDownsample(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'factor' in kwargs, "NNEF %s node is missing 'factor' value"%(self.__class__.__name__)
        super(NearestDownsample, self).__init__('nearest_downsample', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None


class NearestUpsample(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'factor' in kwargs, "NNEF %s node is missing 'factor' value"%(self.__class__.__name__)
        super(NearestUpsample, self).__init__('nearest_upsample', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None


class Neg(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value" % (self.__class__.__name__)
        super(Neg, self).__init__('neg', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Not(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value" % (self.__class__.__name__)
        super(Not, self).__init__('not', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Or(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(Or, self).__init__('or', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, nx_graph):
        self.output_value_ = None


class OutputVal(Node):
    def __init__(self, **kwargs):
        super(OutputVal, self).__init__('output_val', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['base_node'])


class Pad(Node):
    def __init__(self, **kwargs):
        super(Pad, self).__init__('pad', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)


class PlanewiseConv(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value" % (self.__class__.__name__)
        assert 'filter' in kwargs, "NNEF %s node is missing 'input' value" % (self.__class__.__name__)
        self.defaults = {'bias': 0.0, 'border': 'constant', 'padding': [], 'stride': [], 'dilation': []}
        super(PlanewiseConv, self).__init__('planewise_conv', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        if 'bias' in self.parameters and isinstance(self.parameters['bias'], Node):
            self.add_edges(nx_graph, ['input', 'filter', 'bias'])
        else:
            self.add_edges(nx_graph, ['input', 'filter'])

    def run(self, nx_graph):
        self.output_value_ = None


class Pow(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(Pow, self).__init__('pow', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, nx_graph):
        self.output_value_ = None


class Rcp(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value" % (self.__class__.__name__)
        super(Rcp, self).__init__('rcp', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Relu(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        super(Relu, self).__init__('relu', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Reshape(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'shape' in kwargs, "NNEF %s node is missing 'shape' value"%(self.__class__.__name__)
        super(Reshape, self).__init__('reshape', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None


class Round(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value" % (self.__class__.__name__)
        super(Round, self).__init__('round', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Rsqrt(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        super(Rsqrt, self).__init__('rsqrt', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Select(Node):
    def __init__(self, **kwargs):
        assert 'condition' in kwargs, "NNEF %s node is missing 'condition' value"%(self.__class__.__name__)
        assert 'true_value' in kwargs, "NNEF %s node is missing 'true_value' value"%(self.__class__.__name__)
        assert 'false_value' in kwargs, "NNEF %s node is missing 'false_value' value"%(self.__class__.__name__)
        super(Select, self).__init__('select', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['condition', 'true_value', 'false_value'])

    def run(self, nx_graph):
        self.output_value_ = None


class SeparableConv(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'plane_filter' in kwargs, "NNEF %s node is missing 'plane_filter' value"%(self.__class__.__name__)
        assert 'point_filter' in kwargs, "NNEF %s node is missing 'point_filter' value"%(self.__class__.__name__)
        self.defaults = {'bias': 0.0, 'border': 'constant', 'padding': [],
                         'stride': [], 'dilation': [], 'groups': 1}
        super(SeparableConv, self).__init__('separable_conv', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        if 'bias' in self.parameters and isinstance(self.parameters['bias'], Node):
            self.add_edges(nx_graph, ['input', 'plane_filter', 'point_filter', 'bias'])
        else:
            self.add_edges(nx_graph, ['input', 'plane_filter', 'point_filter'])

    def run(self, nx_graph):
        self.output_value_ = None


class ShapeOf(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s noe is missing 'x' value"%(self.__class__.__name__)
        super(ShapeOf, self).__init__('shape_of', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None

    def get_value(self):
        return self.output_shape


class Sigmoid(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is imissing 'x' value"%(self.__class__.__name__)
        super(Sigmoid, self).__init__('sigmoid', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Sign(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        super(Sign, self).__init__('sign', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Slice(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'axes' in kwargs, "NNEF %s node is missing 'axes' value"%(self.__class__.__name__)
        assert 'begin' in kwargs, "NNEF %s node is missing 'begin' value"%(self.__class__.__name__)
        assert 'end' in kwargs, "NNEF %s node is missing 'end' value"%(self.__class__.__name__)
        super(Slice, self).__init__('slice', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None

    def get_value(self):
        init_val = self.parameters['input'].get_value()
        output_val = init_val[:]
        axes = self.parameters['axes']
        for i in range(len(axes)):
            if i in axes:
                if i == 0:
                    output_val = output_val[int(self.parameters['begin'][axes.index(i)]):int(self.parameters['end'][axes.index(i)])]
                elif i == 1:
                    output_val = output_val[:][int(self.parameters['begin'][axes.index(i)]):int(self.parameters['end'][axes.index(i)])]
                elif i == 2:
                    output_val = output_val[:][:][int(self.parameters['begin'][axes.index(i)]):int(self.parameters['end'][axes.index(i)])]
                elif i == 3:
                    output_val = output_val[:][:][:][int(self.parameters['begin'][axes.index(i)]):int(self.parameters['end'][axes.index(i)])]
                elif i == 4:
                    output_val = output_val[:][:][:][:][int(self.parameters['begin'][axes.index(i)]):int(self.parameters['end'][axes.index(i)])]
                else:
                    raise ValueError("Axes goes too high, currently not handled")

        return output_val


class Softmax(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        self.defaults = {'axes': [1]}
        super(Softmax, self).__init__('softmax', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Softplus(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        super(Softplus, self).__init__('softplus', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Softsign(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        super(Softsign, self).__init__('softsign', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Split(Node):
    def __init__(self, **kwargs):
        assert 'value' in kwargs, "NNEF %s node is missing 'value' value"%(self.__class__.__name__)
        assert 'axis' in kwargs, "NNEF %s node is missing 'axis' value"%(self.__class__.__name__)
        assert 'ratios' in kwargs, "NNEF %s node is missing 'ratios' value"%(self.__class__.__name__)
        super(Split, self).__init__('split', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['value'])

    def run(self, nx_graph):
        self.output_value_ = None


class Sqr(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        super(Sqr, self).__init__('sqr', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Sqrt(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        super(Sqrt, self).__init__('sqrt', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Stack(Node):
    def __init__(self, **kwargs):
        assert 'values' in kwargs, "NNEF %s node is missing 'values' value"%(self.__class__.__name__)
        assert 'axis' in kwargs, "NNEF %s node is missing 'axis' value"%(self.__class__.__name__)
        super(Stack, self).__init__('stack', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['values'])

    def run(self, nx_graph):
        self.output_value_ = None

    def get_value(self):
        values = []
        for i in range(len(self.parameters['values'])):
            values.append(self.parameters['values'][i].get_value())

        return np.stack(values, self.parameters['axis'])


class Sub(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        assert 'y' in kwargs, "NNEF %s node is missing 'y' value"%(self.__class__.__name__)
        super(Sub, self).__init__('sub', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x', 'y'])

    def run(self, nx_graph):
        self.output_value_ = None

    def get_value(self):
        x_val = self.parameters['x'].get_value()
        y_val = self.parameters['y'].get_value()

        return np.subtract(x_val, y_val)


class SumReduce(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value" % (self.__class__.__name__)
        assert 'axes' in kwargs, "NNEF %s node is missing 'axes' value" % (self.__class__.__name__)
        self.defaults = {'normalize': False}
        super(SumReduce, self).__init__('sum_reduce', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None


class Tanh(Node):
    def __init__(self, **kwargs):
        assert 'x' in kwargs, "NNEF %s node is missing 'x' value"%(self.__class__.__name__)
        super(Tanh, self).__init__('tanh', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['x'])

    def run(self, nx_graph):
        self.output_value_ = None


class Transpose(Node):
    def __init__(self, **kwargs):
        assert 'input' in kwargs, "NNEF %s node is missing 'input' value"%(self.__class__.__name__)
        assert 'axes' in kwargs, "NNEF %s node is missing 'perm' value"%(self.__class__.__name__)
        super(Transpose, self).__init__('transpose', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['input'])

    def run(self, nx_graph):
        self.output_value_ = None


class Update(Node):
    def __init__(self, **kwargs):
        assert 'variable' in kwargs, "NNEF %s node is missing 'variable' value"%(self.__class__.__name__)
        assert 'value' in kwargs, "NNEF %s node is missing 'value' value"%(self.__class__.__name__)
        super(Update, self).__init__('update', **kwargs)

    def compile(self, nx_graph):
        Node.compile(self, nx_graph)
        self.add_edges(nx_graph, ['variable', 'value'])

    def run(self, nx_graph):
        self.output_value_ = None


class Variable(Node):
    def __init__(self, **kwargs):
        assert 'label' in kwargs, "NNEF %s node is missing 'label' value"%(self.__class__.__name__)
        assert 'shape' in kwargs, "NNEF %s node is missing 'shape' value"%(self.__class__.__name__)
        assert isinstance(kwargs['label'], str), "NNEF %s node, 'label' value is not of 'str' type"%(self.__class__.__name__)
        assert isinstance(kwargs['shape'], list), "NNEF %s node, 'shape' value is not of 'list' type"%(self.__class__.__name__)
        super(Variable, self).__init__('variable', **kwargs)
        self.modified = False

        if isinstance(kwargs['_np_tensor'], bytes):
            self.tensor = np.frombuffer(kwargs['_np_tensor'], dtype = kwargs['_np_dtype'])
            self.tensor = np.reshape(self.tensor, kwargs['shape'])
        else:
            self.tensor = kwargs['_np_tensor']

        self.output_shape = kwargs['shape']

    def run(self, **kwargs):
        self.output_value_ = self.tensor
        return self.output_value_
