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

from __future__ import division, print_function, absolute_import

import os
import unittest

import numpy as np
import onnx
from caffe2.python import model_helper

from nnef_tools.io.onnx import onnx_io
from nnef_tools.io.onnx.onnx_graph import *
from tests.conversion.onnx_test_runner import ONNXTestRunner

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


class ONNXLayerTestCases(ONNXTestRunner):

    # Tests:

    def test_Abs(self):
        self._test_from_caffe2(self.get_unary_network_function('Abs'))

    # def test_Acos(self):
    #     pass
    #
    # def test_Acosh(self):
    #     pass

    def test_Add(self):
        self._test_from_caffe2(self.get_binary_network_function('Add'))

    def test_And(self):
        self._test_from_caffe2(self.get_binary_network_function('And', dtype=onnx.TensorProto.BOOL))

    @staticmethod
    def ArgMax_network():
        value_info = {'x': (onnx.TensorProto.FLOAT, [1, 1, 5, 5])}
        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')
        model.net.ArgMax('x', 'y', axis=2)
        model.net.AddExternalOutput(model.net.GetBlobRef('y'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    def test_ArgMax(self):
        self._test_from_caffe2(self.ArgMax_network)

    @staticmethod
    def ArgMin_network():
        value_info = {'x': (onnx.TensorProto.FLOAT, [1, 1, 5, 5])}
        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')
        model.net.ArgMin('x', 'y', axis=2)
        model.net.AddExternalOutput(model.net.GetBlobRef('y'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    def test_ArgMin(self):
        self._test_from_caffe2(self.ArgMin_network)

    # def test_Asin(self):
    #     pass
    #
    # def test_Asinh(self):
    #     pass
    #
    # def test_Atan(self):
    #     pass
    #
    # def test_Atanh(self):
    #     pass

    def test_AveragePool(self):
        self._test_from_caffe2(self.get_unary_network_function('AveragePool',
                                                               kwargs=dict(kernel=3, stride=2, pad=1)))
        self._test_from_caffe2(self.get_unary_network_function('AveragePool',
                                                               kwargs=dict(kernels=(3, 2), strides=(2, 3),
                                                                           pads=(1, 0, 0, 1))))

    @staticmethod
    def BatchNormalization_network():
        value_info = {'x': (onnx.TensorProto.FLOAT, [2, 2, 5, 5]),
                      'scale': (onnx.TensorProto.FLOAT, [2]),
                      'bias': (onnx.TensorProto.FLOAT, [2]),
                      'mean': (onnx.TensorProto.FLOAT, [2]),
                      'var': (onnx.TensorProto.FLOAT, [2])}

        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')
        model.net.AddExternalInput('scale')
        model.net.AddExternalInput('bias')
        model.net.AddExternalInput('mean')
        model.net.AddExternalInput('var')

        model.net.SpatialBN(['x', 'scale', 'bias', 'mean', 'var'], 'y', is_test=True)
        # is_test=False not really supported
        model.net.AddExternalOutput(model.net.GetBlobRef('y'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    def test_BatchNormalization(self):
        self._test_from_caffe2(self.BatchNormalization_network)

    def test_Cast(self):
        def test(from_, to, name):
            g = ONNXGraph('test_network')
            x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype=from_)
            y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 5], dtype=to)
            ONNXOperation(graph=g, name='Cast', inputs=x, attribs=dict(to=onnx_io.build_dtype(to)), outputs=y)
            g.inputs = (x,)
            g.outputs = (y,)
            self._test_from_onnx_graph(g, name)

        test('BOOL', 'BOOL', "Cast0")
        test('FLOAT', 'FLOAT', "Cast1")
        test('INT64', 'INT64', "Cast2")
        test('BOOL', 'FLOAT', "Cast3")
        test('FLOAT', 'BOOL', "Cast4")
        test('BOOL', 'INT64', "Cast5")

    def test_Ceil(self):
        self._test_from_caffe2(self.get_unary_network_function('Ceil'))

    def test_Clip(self):
        self._test_from_caffe2(self.get_unary_network_function('Clip', kwargs=dict(min=0.4, max=0.6)))

    # def test_Compress(self):
    #     pass

    @staticmethod
    def Concat_network():
        value_info = {'x': (onnx.TensorProto.FLOAT, [2, 2, 5, 5]),
                      'y': (onnx.TensorProto.FLOAT, [2, 3, 5, 5]),
                      'z': (onnx.TensorProto.FLOAT, [2, 3, 5, 5])}

        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')
        model.net.AddExternalInput('y')
        model.net.AddExternalInput('z')

        model.net.Concat(['x', 'y', 'z'], ['a', 'b'], axis=1)
        model.net.AddExternalOutput(model.net.GetBlobRef('a'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    def test_Concat(self):
        self._test_from_caffe2(self.Concat_network)

    def test_Constant(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 32, 2, 2], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 32, 2, 2], dtype='FLOAT')
        c = ONNXTensor(graph=g, name='c', shape=[2], dtype='FLOAT', data=[1.0, 2.0])
        ONNXOperation(graph=g, name='Add', inputs=(x, c), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'Constant')

    def test_ConstantOfShape(self):
        # It fails in python2 (but it fails in onnx the onnx backend itself)
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 32, 2, 2], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 32, 2, 2], dtype='FLOAT')
        value = ONNXTensor(graph=g, name='value', shape=[], dtype='FLOAT', data=[1.0])
        constantOfShape = ONNXTensor(graph=g, name='constantOfShape', shape=[1, 32, 2, 2], dtype='FLOAT')
        shape = ONNXTensor(graph=g, name='shape', shape=[4], dtype='INT64', data=[1, 32, 2, 2])
        ONNXOperation(graph=g, name='ConstantOfShape', inputs=(shape, value), outputs=constantOfShape)
        ONNXOperation(graph=g, name='Add', inputs=(x, constantOfShape), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'ConstantOfShape')

    @staticmethod
    def Conv_network():
        value_info = {'x': (onnx.TensorProto.FLOAT, [1, 1, 5, 5])}

        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')
        model.param_init_net.Const(np.ones([4, 1, 3, 3]) * 2 - 1, 'y', dtype=np.float32)
        model.param_init_net.Const(np.ones([4]), 'b', dtype=np.float32)

        model.net.Conv(['x', 'y', 'b'], 'z', kernel=3)
        model.net.AddExternalOutput(model.net.GetBlobRef('z'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    def test_Conv(self):
        self._test_from_caffe2(self.Conv_network)

    @staticmethod
    def ConvTranspose_network():
        value_info = {'x': (onnx.TensorProto.FLOAT, [1, 4, 5, 5])}

        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')
        model.param_init_net.Const(np.ones([4, 1, 3, 3]) * 2 - 1, 'y', dtype=np.float32)
        model.param_init_net.Const(np.ones([1]), 'b', dtype=np.float32)

        model.net.ConvTranspose(['x', 'y', 'b'], 'z', kernel=3, pads=[0, 1, 0, 1])
        model.net.AddExternalOutput(model.net.GetBlobRef('z'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    def test_ConvTranspose(self):
        self._test_from_caffe2(self.ConvTranspose_network)

    # def test_Cosh(self):
    #     pass
    #

    def test_DepthToSpace(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 32, 2, 2], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 8, 8], dtype='FLOAT')
        ONNXOperation(graph=g, name='DepthToSpace', inputs=x, attribs=dict(blocksize=4), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'DepthToSpace', run=False)  # Not implemented in caffe2

    def test_Div(self):
        self._test_from_caffe2(self.get_binary_network_function('Div'))

    @staticmethod
    def Dropout_network():
        value_info = {'x': (onnx.TensorProto.FLOAT, [1, 4, 5, 5])}

        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')

        model.net.Dropout('x', 'y', ratio=0.5, is_test=True)
        model.net.AddExternalOutput(model.net.GetBlobRef('y'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    def test_Dropout(self):
        self._test_from_caffe2(self.Dropout_network)

    def test_Elu(self):
        self._test_from_caffe2(self.get_unary_network_function('Elu'))

    def test_Equal(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 1], dtype='FLOAT')
        a = ONNXTensor(graph=g, name='a', shape=[1, 2, 5, 5], dtype='BOOL')
        b = ONNXTensor(graph=g, name='b', shape=[1, 2, 5, 5], dtype='BOOL')
        ONNXOperation(graph=g, name='Equal', inputs=(x, y), outputs=a)
        ONNXOperation(graph=g, name='Equal', inputs=(x, x), outputs=b)
        g.inputs = (x, y)
        g.outputs = (a, b)
        self._test_from_onnx_graph(g, 'Equal')

    # def test_Erf(self):
    #     pass

    def test_Exp(self):
        self._test_from_caffe2(self.get_unary_network_function('Exp'))

    def test_Expand(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 3], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[5, 6, 2, 3], dtype='FLOAT')
        shape = ONNXTensor(graph=g, name='shape', shape=[4], data=[5, 6, 2, 3], dtype='INT64')
        ONNXOperation(graph=g, name='Expand', inputs=(x, shape), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'Expand')

    # def test_EyeLike(self):
    #     pass

    @staticmethod
    def Flatten_network():
        value_info = {'x': (onnx.TensorProto.FLOAT, [1, 4, 5, 5])}

        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')

        model.net.Flatten('x', 'y', axis=2)
        model.net.AddExternalOutput(model.net.GetBlobRef('y'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    def test_Flatten(self):
        self._test_from_caffe2(self.Flatten_network)

    def test_Floor(self):
        self._test_from_caffe2(self.get_unary_network_function('Floor'))

    # def test_GRU(self):
    #     pass
    #
    # def test_Gather(self):
    #     pass

    def test_Gemm(self):
        g = ONNXGraph('test_network')
        A = ONNXTensor(graph=g, name='A', shape=[5, 10], dtype='FLOAT')
        B = ONNXTensor(graph=g, name='B', shape=[10, 2], dtype='FLOAT')
        C = ONNXTensor(graph=g, name='C', shape=[5, 2], dtype='FLOAT')
        D = ONNXTensor(graph=g, name='D', shape=[5, 2], dtype='FLOAT')
        ONNXOperation(graph=g, name='Gemm', inputs=(A, B, C), outputs=D)
        g.inputs = (A, B, C)
        g.outputs = (D,)
        self._test_from_onnx_graph(g, 'Gemm')

    def test_GlobalAveragePool(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 1, 1], dtype='FLOAT')
        ONNXOperation(graph=g, name='GlobalAveragePool', inputs=x, outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'GlobalAveragePool')

    def test_GlobalLpPool(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 1, 1], dtype='FLOAT')
        ONNXOperation(graph=g, name='GlobalLpPool', inputs=x, outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'GlobalLpPool', run=False)

    def test_GlobalMaxPool(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 1, 1], dtype='FLOAT')
        ONNXOperation(graph=g, name='GlobalMaxPool', inputs=x, outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'GlobalMaxPool')

    def test_Greater(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 1], dtype='FLOAT')
        a = ONNXTensor(graph=g, name='a', shape=[1, 2, 5, 5], dtype='BOOL')
        ONNXOperation(graph=g, name='Greater', inputs=(x, y), outputs=a)
        g.inputs = (x, y)
        g.outputs = (a,)
        self._test_from_onnx_graph(g, 'Greater')

    def test_HardSigmoid(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='HardSigmoid', inputs=x, attribs=dict(alpha=0.25, beta=0.6), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'HardSigmoid')

    # def test_HardMax(self):
    #     pass

    def test_Identity(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='Identity', inputs=x, outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'Identity')

    # def test_If(self):
    #     pass

    def test_InstanceNormalization(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        scale = ONNXTensor(graph=g, name='scale', shape=[2], dtype='FLOAT')
        bias = ONNXTensor(graph=g, name='bias', shape=[2], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='InstanceNormalization',
                      inputs=(x, scale, bias),
                      attribs=dict(epsilon=1e-4),
                      outputs=y)
        g.inputs = (x, scale, bias)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'InstanceNormalization')

    # def test_IsNan(self):
    #     pass

    @staticmethod
    def LRN_network():
        value_info = {'x': (onnx.TensorProto.FLOAT, [1, 4, 5, 5])}

        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')

        model.net.LRN('x', 'y', alpha=0.001, beta=0.7, bias=0.9, size=3)
        model.net.AddExternalOutput(model.net.GetBlobRef('y'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    def test_LRN(self):
        self._test_from_caffe2(self.LRN_network)

    # def test_LSTM(self):
    #     pass

    def test_LeakyRelu(self):
        self._test_from_caffe2(self.get_unary_network_function('LeakyRelu', kwargs=dict(alpha=0.01)))

    def test_Less(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 1], dtype='FLOAT')
        a = ONNXTensor(graph=g, name='a', shape=[1, 2, 5, 5], dtype='BOOL')
        ONNXOperation(graph=g, name='Less', inputs=(x, y), outputs=a)
        g.inputs = (x, y)
        g.outputs = (a,)
        self._test_from_onnx_graph(g, 'Less')

    def test_Log(self):
        self._test_from_caffe2(self.get_unary_network_function('Log'))

    def test_LogSoftmax(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[5, 6, 2, 3], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[5, 6, 2, 3], dtype='FLOAT')
        ONNXOperation(graph=g, name='LogSoftmax', inputs=x, outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'LogSoftmax')

    #
    # def test_Loop(self):
    #     pass

    def test_LpNormalization(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 1, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 1, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='LpNormalization', inputs=x, outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'LpNormalization', run=False)

    # def test_LpPool(self):
    #     pass

    @staticmethod
    def Matmul_network():
        value_info = {'x': (onnx.TensorProto.FLOAT, [5, 6]),
                      'y': (onnx.TensorProto.FLOAT, [6, 7])}

        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')

        model.net.MatMul(['x', 'y'], 'z')
        model.net.AddExternalOutput(model.net.GetBlobRef('z'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    def test_MatMul(self):
        self._test_from_caffe2(self.Matmul_network)

    def test_Max(self):
        self._test_from_caffe2(self.get_ternary_network_function('Max'))

    def test_MaxPool(self):
        self._test_from_caffe2(self.get_unary_network_function('MaxPool', kwargs=dict(kernels=(3, 2), strides=(1, 2))))

    # def test_MaxRoiPool(self):
    #     pass

    def test_MaxUnpool(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 1, 2, 2], dtype='FLOAT')
        i = ONNXTensor(graph=g, name='i', shape=[1, 1, 2, 2], dtype='INT64', data=[2, 1, 0, 3])
        y = ONNXTensor(graph=g, name='y', shape=[1, 1, 4, 4], dtype='FLOAT')
        ONNXOperation(graph=g, name='MaxUnpool', inputs=(x, i), attribs=dict(kernel_shape=[2, 2],
                                                                             pads=[0, 0, 0, 0],
                                                                             strides=[2, 2]), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'MaxUnpool', run=False)

    def test_Mean(self):
        self._test_from_caffe2(self.get_ternary_network_function('Mean'))

    def test_Min(self):
        self._test_from_caffe2(self.get_ternary_network_function('Min'))

    def test_Mul(self):
        self._test_from_caffe2(self.get_binary_network_function('Mul'))

    # def test_Multinomial(self):
    #     pass

    def test_Neg(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='Neg', inputs=x, outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'Neg')

    def test_Not(self):
        self._test_from_caffe2(self.get_unary_network_function('Not', dtype=onnx.TensorProto.BOOL))

    # def test_OneHot(self):
    #     pass

    def test_Or(self):
        self._test_from_caffe2(self.get_binary_network_function('Or', dtype=onnx.TensorProto.BOOL))

    @staticmethod
    def PRelu_network():
        value_info = {'x': (onnx.TensorProto.FLOAT, [1, 3, 5, 3]),
                      'y': (onnx.TensorProto.FLOAT, [3])}

        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')
        model.net.AddExternalInput('y')

        model.net.PRelu(['x', 'y'], 'z')
        model.net.AddExternalOutput(model.net.GetBlobRef('z'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    def test_PRelu(self):
        self._test_from_caffe2(self.PRelu_network)

    def test_Pad(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 3, 6, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='Pad', inputs=x, attribs=dict(pads=[0, 1, 0, 0,
                                                                        0, 0, 1, 0]), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'Pad', run=False)

    def test_Pow(self):
        # Caffe2 pow can not broadcast
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 5], dtype='FLOAT')
        z = ONNXTensor(graph=g, name='z', shape=[1, 2, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='Pow',
                      inputs=(x, y),
                      outputs=z)
        g.inputs = (x, y)
        g.outputs = (z,)
        self._test_from_onnx_graph(g, 'Pow')

    # def test_RNN(self):
    #     pass
    #
    # def test_RandomNormal(self):
    #     pass
    #
    # def test_RandomNormalLike(self):
    #     pass
    #
    # def test_RandomUniform(self):
    #     pass
    #
    # def test_RandomUniformLike(self):
    #     pass

    def test_Reciprocal(self):
        self._test_from_caffe2(self.get_unary_network_function('Reciprocal'))

    def test_ReduceL1(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 1, 1], dtype='FLOAT')
        ONNXOperation(graph=g, name='ReduceL1',
                      inputs=x,
                      attribs=dict(axes=[2, 3]),
                      outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'ReduceL1')

    def test_ReduceL2(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 1, 1], dtype='FLOAT')
        ONNXOperation(graph=g, name='ReduceL2',
                      inputs=x,
                      attribs=dict(axes=[2, 3]),
                      outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'ReduceL2')

    def test_ReduceLogSum(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 1, 1], dtype='FLOAT')
        ONNXOperation(graph=g, name='ReduceLogSum',
                      inputs=x,
                      attribs=dict(axes=[2, 3]),
                      outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'ReduceLogSum', run=False)

    def test_ReduceLogSumExp(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 1, 1], dtype='FLOAT')
        ONNXOperation(graph=g, name='ReduceLogSumExp',
                      inputs=x,
                      attribs=dict(axes=[2, 3]),
                      outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'ReduceLogSumExp', run=False)

    def test_ReduceMax(self):
        self._test_from_caffe2(self.get_unary_network_function('ReduceMax', kwargs=dict(axes=[1, 2])))
        self._test_from_caffe2(self.get_unary_network_function('ReduceMax', kwargs=dict(axes=[1, 2], keepdims=0)))

    def test_ReduceMean(self):
        self._test_from_caffe2(self.get_unary_network_function('ReduceMean', kwargs=dict(axes=[1, 2])))
        self._test_from_caffe2(self.get_unary_network_function('ReduceMean', kwargs=dict(axes=[1, 2], keepdims=0)))

    def test_ReduceMin(self):
        self._test_from_caffe2(self.get_unary_network_function('ReduceMin', kwargs=dict(axes=[1, 2])))
        self._test_from_caffe2(self.get_unary_network_function('ReduceMin', kwargs=dict(axes=[1, 2], keepdims=0)))

    # def test_ReduceProd(self):
    #     pass

    def test_ReduceSum(self):
        self._test_from_caffe2(self.get_unary_network_function('ReduceSum', kwargs=dict(axes=[1, 2])))
        self._test_from_caffe2(self.get_unary_network_function('ReduceSum', kwargs=dict(axes=[1, 2], keepdims=0)))

    def test_ReduceSumSquare(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 1, 1], dtype='FLOAT')
        ONNXOperation(graph=g, name='ReduceSumSquare',
                      inputs=x,
                      attribs=dict(axes=[2, 3]),
                      outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'ReduceSumSquare', run=False)

    def test_Relu(self):
        self._test_from_caffe2(self.get_unary_network_function('Relu'))

    @staticmethod
    def Reshape_network():
        value_info = {'x': (onnx.TensorProto.FLOAT, [1, 3, 5, 3])}

        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')

        model.net.Reshape(['x'], ['z', 'w'], shape=[0, 5, -1, 1])
        model.net.AddExternalOutput(model.net.GetBlobRef('z'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    @staticmethod
    def Reshape2_network():
        value_info = {'x': (onnx.TensorProto.FLOAT, [1, 3, 5, 3]),
                      'y': (onnx.TensorProto.FLOAT, [45])}

        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')
        model.net.AddExternalInput('y')
        model.net.Shape('y', 'sy')
        model.net.Reshape(['x', 'sy'], ['z', 'w'])
        model.net.AddExternalOutput(model.net.GetBlobRef('z'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    def test_Reshape(self):
        self._test_from_caffe2(self.Reshape_network)
        self._test_from_caffe2(self.Reshape2_network)

    # def test_Scan(self):
    #     pass
    #
    # def test_Scatter(self):
    #     pass
    #
    # def test_Selu(self):
    #     pass

    def test_Shape(self):
        self._test_from_caffe2(self.get_unary_network_function('Shape'))

    # def test_Shrink(self):
    #     pass

    def test_Sigmoid(self):
        self._test_from_caffe2(self.get_unary_network_function('Sigmoid'))

    def test_Sign(self):
        self._test_from_caffe2(self.get_unary_network_function('Sign'))

    # def test_Sinh(self):
    #     pass

    def test_Size(self):
        self._test_from_caffe2(self.get_unary_network_function('Size'))

    def test_Slice(self):
        self._test_from_caffe2(self.get_unary_network_function('Slice',
                                                               shape=[1, 3, 5, 5],
                                                               kwargs=dict(starts=[0, 1, 0, 0], ends=[-1, 3, -1, -1])))

    def test_Softmax(self):
        self._test_from_caffe2(self.get_unary_network_function('Softmax'))

    def test_Softplus(self):
        self._test_from_caffe2(self.get_unary_network_function('Softplus'))

    # def test_Softsign(self):
    #     pass

    # def test_SpaceToDepth(self):
    #     pass

    @staticmethod
    def Split_network():
        value_info = {'x': (onnx.TensorProto.FLOAT, [1, 3, 5, 3])}

        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')
        model.net.Split('x', ['a', 'b', 'c'], axis=2, split=[2, 1, 2])
        model.net.AddExternalOutput(model.net.GetBlobRef('a'))
        model.net.AddExternalOutput(model.net.GetBlobRef('b'))
        model.net.AddExternalOutput(model.net.GetBlobRef('c'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    def test_Split(self):
        self._test_from_caffe2(self.Split_network)

    def test_Sqrt(self):
        self._test_from_caffe2(self.get_unary_network_function('Sqrt'))

    def test_Squeeze(self):
        self._test_from_caffe2(self.get_unary_network_function('Squeeze', shape=[1, 2, 1, 5], kwargs=dict(dims=[0, 2])))

    def test_Sub(self):
        self._test_from_caffe2(self.get_binary_network_function('Sub'))

    def test_Sum(self):
        self._test_from_caffe2(self.get_ternary_network_function('Sum'))

    # def test_Tan(self):
    #     pass

    def test_Tanh(self):
        self._test_from_caffe2(self.get_unary_network_function('Tanh'))

    def test_Tile(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        repeats = ONNXTensor(graph=g, name='repeats', shape=[4], dtype='INT64', data=[1, 2, 2, 1])
        z = ONNXTensor(graph=g, name='z', shape=[1, 4, 10, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='Tile',
                      inputs=(x, repeats),
                      outputs=z)
        g.inputs = (x,)
        g.outputs = (z,)
        self._test_from_onnx_graph(g, 'Tile')

    # def test_TopK(self):
    #     pass

    def test_Transpose(self):
        self._test_from_caffe2(self.get_unary_network_function('Transpose'))
        self._test_from_caffe2(self.get_unary_network_function('Transpose', kwargs=dict(axes=[3, 2, 0, 1])))

    def test_Unsqueeze(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 1, 2, 5, 5, 1], dtype='FLOAT')
        ONNXOperation(graph=g, name='Unsqueeze', inputs=x, attribs=dict(axes=[0, 5]), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'Unsqueeze0')

    def test_Upsample(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        m = ONNXTensor(graph=g, name='m', shape=[4], dtype='FLOAT', data=[1.0, 1.0, 2.0, 2.0])
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 10, 10], dtype='FLOAT')
        ONNXOperation(graph=g, name='Upsample', inputs=(x, m), attribs=dict(mode='linear'), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'Upsample0')
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        m = ONNXTensor(graph=g, name='m', shape=[4], dtype='FLOAT', data=[1.0, 1.0, 2.0, 2.0])
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 10, 10], dtype='FLOAT')
        ONNXOperation(graph=g, name='Upsample', inputs=(x, m), attribs=dict(mode='nearest'), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'Upsample1')

    @staticmethod
    def Where_network():
        value_info = {'x': (onnx.TensorProto.BOOL, [1, 3, 5, 3]),
                      'y': (onnx.TensorProto.FLOAT, [1, 3, 5, 3]),
                      'z': (onnx.TensorProto.FLOAT, [1, 3, 5, 3])}

        model = model_helper.ModelHelper(name='test_network')
        model.net.AddExternalInput('x')
        model.net.AddExternalInput('y')
        model.net.AddExternalInput('z')
        model.net.Where(['x', 'y', 'z'], 'a')
        model.net.AddExternalOutput(model.net.GetBlobRef('a'))

        return model.net.Proto(), model.param_init_net.Proto(), value_info

    def test_Where(self):
        self._test_from_caffe2(self.Where_network)

    def test_Xor(self):
        self._test_from_caffe2(self.get_binary_network_function('Xor', dtype=onnx.TensorProto.BOOL))

    def test_ImageScaler(self):
        g = ONNXGraph('test_network')
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='ImageScaler', inputs=x, attribs=dict(bias=[7.0, 3.0], scale=2.0), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_onnx_graph(g, 'ImageScaler', run=False)


if __name__ == '__main__':
    unittest.main()
