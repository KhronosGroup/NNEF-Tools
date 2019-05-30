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

from nnef_tools.core import utils
from nnef_tools.io.onnx import onnx_io
from nnef_tools.io.onnx.onnx_graph import *
from nnef_tests.conversion.onnx_test_runner import ONNXTestRunner

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


class ONNXLayerTestCases(ONNXTestRunner):

    def test_Abs(self):
        self._test_from_graph(self._unary('Abs'))

    def test_Add(self):
        self._test_from_graph(self._binary('Add'))

    def test_And(self):
        self._test_from_graph(self._binary('And', dtype='BOOL', out_dtype='BOOL'))

    def test_ArgMax(self):
        g = ONNXGraph(self._graph_name('ArgMax'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 5, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 1, 5, 5], dtype='INT64')
        ONNXOperation(graph=g, name='ArgMax', inputs=(x,), outputs=(y,), attribs=dict(axis=1, keepdims=1))
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_ArgMin(self):
        g = ONNXGraph(self._graph_name('ArgMin'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 5, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 1, 5, 5], dtype='INT64')
        ONNXOperation(graph=g, name='ArgMin', inputs=(x,), outputs=(y,), attribs=dict(axis=1, keepdims=1))
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_AveragePool(self):
        g = ONNXGraph(self._graph_name('AveragePool'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 1, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 1, 3, 3], dtype='FLOAT')
        ONNXOperation(graph=g, name='AveragePool', inputs=(x,), outputs=(y,),
                      attribs=dict(kernel_shape=(3, 3), strides=(2, 2), pads=[1, 1, 1, 1]))
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

        g = ONNXGraph(self._graph_name('AveragePool'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 1, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 1, 2, 2], dtype='FLOAT')
        ONNXOperation(graph=g, name='AveragePool', inputs=(x,), outputs=(y,),
                      attribs=dict(kernel_shape=(3, 2), strides=(2, 3), pads=[1, 0, 0, 1]))
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_BatchNormalization(self):
        g = ONNXGraph(self._graph_name('BatchNormalization'))
        x = ONNXTensor(graph=g, name='x', shape=[2, 2, 5, 5], dtype='FLOAT')
        scale = ONNXTensor(graph=g, name='scale', shape=[2], dtype='FLOAT')
        bias = ONNXTensor(graph=g, name='bias', shape=[2], dtype='FLOAT')
        mean = ONNXTensor(graph=g, name='mean', shape=[2], dtype='FLOAT')
        var = ONNXTensor(graph=g, name='var', shape=[2], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[2, 2, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='BatchNormalization', inputs=(x, scale, bias, mean, var), outputs=(y,),
                      attribs=dict(epsilon=1e-6))
        g.inputs = (x, scale, bias, mean, var)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_Cast(self):
        def test(from_, to):
            g = ONNXGraph(self._graph_name('Cast'))
            x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype=from_)
            y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 5], dtype=to)
            ONNXOperation(graph=g, name='Cast', inputs=x, attribs=dict(to=onnx_io.build_dtype(to)), outputs=y)
            g.inputs = (x,)
            g.outputs = (y,)
            self._test_from_graph(g)

        test('BOOL', 'BOOL')
        test('FLOAT', 'FLOAT')
        test('INT64', 'INT64')
        test('BOOL', 'FLOAT')
        test('FLOAT', 'BOOL')
        test('BOOL', 'INT64')

    def test_Ceil(self):
        self._test_from_graph(self._unary('Ceil'))

    def test_Clip(self):
        self._test_from_graph(self._unary('Clip', attribs=dict(min=0.4, max=0.6)))

    def test_Concat(self):
        g = ONNXGraph(self._graph_name('Concat'))
        x = ONNXTensor(graph=g, name='x', shape=[2, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[2, 3, 5, 5], dtype='FLOAT')
        z = ONNXTensor(graph=g, name='z', shape=[2, 3, 5, 5], dtype='FLOAT')
        w = ONNXTensor(graph=g, name='w', shape=[2, 8, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='Concat', inputs=(x, y, z), outputs=(w,),
                      attribs=dict(axis=1))
        g.inputs = (x, y, z)
        g.outputs = (w,)
        self._test_from_graph(g)

    def test_Constant(self):
        g = ONNXGraph(self._graph_name('Constant'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 32, 2, 2], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 32, 2, 2], dtype='FLOAT')
        c = ONNXTensor(graph=g, name='c', shape=[2], dtype='FLOAT', data=[1.0, 2.0])
        ONNXOperation(graph=g, name='Add', inputs=(x, c), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_ConstantOfShape(self):
        # It fails in python2 (in the onnx backend itself)
        g = ONNXGraph(self._graph_name('ConstantOfShape'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 32, 2, 2], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 32, 2, 2], dtype='FLOAT')
        value = ONNXTensor(graph=g, name='value', shape=[], dtype='FLOAT', data=[1.0])
        constant_of_shape = ONNXTensor(graph=g, name='constant_of_shape', shape=[1, 32, 2, 2], dtype='FLOAT')
        shape = ONNXTensor(graph=g, name='shape', shape=[4], dtype='INT64', data=[1, 32, 2, 2])
        ONNXOperation(graph=g, name='ConstantOfShape', inputs=(shape, value), outputs=constant_of_shape)
        ONNXOperation(graph=g, name='Add', inputs=(x, constant_of_shape), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_Conv(self):
        g = ONNXGraph(self._graph_name('Conv'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 1, 5, 5], dtype='FLOAT')
        w = ONNXTensor(graph=g, name='w', shape=[4, 1, 3, 3], dtype='FLOAT')
        b = ONNXTensor(graph=g, name='b', shape=[4], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 4, 3, 3], dtype='FLOAT')

        ONNXOperation(graph=g, name='Conv', inputs=(x, w, b), outputs=(y,),
                      attribs=dict(kernel_shape=(3, 3)))
        g.inputs = (x, w, b)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_ConvTranspose(self):
        g = ONNXGraph(self._graph_name('ConvTranspose'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 4, 5, 5], dtype='FLOAT')
        w = ONNXTensor(graph=g, name='w', shape=[4, 1, 3, 3], dtype='FLOAT')
        b = ONNXTensor(graph=g, name='b', shape=[1], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 1, 7, 5], dtype='FLOAT')

        ONNXOperation(graph=g, name='ConvTranspose', inputs=(x, w, b), outputs=(y,),
                      attribs=dict(kernel_shape=(3, 3), pads=[0, 1, 0, 1]))
        g.inputs = (x, w, b)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_DepthToSpace(self):
        g = ONNXGraph(self._graph_name('DepthToSpace'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 32, 2, 2], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 8, 8], dtype='FLOAT')
        ONNXOperation(graph=g, name='DepthToSpace', inputs=x, attribs=dict(blocksize=4), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g, run=False)  # Not implemented in caffe2

    def test_Div(self):
        self._test_from_graph(self._binary('Div'))

    def test_Dropout(self):
        g = ONNXGraph(self._graph_name('Dropout'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 4, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 4, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='Dropout', inputs=x, outputs=y, attribs=dict(ratio=0.5))
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_Elu(self):
        self._test_from_graph(self._unary('Elu'))

    def test_Equal(self):
        self._test_from_graph(self._binary('Equal', dtype='FLOAT', out_dtype='BOOL'))

    def test_Exp(self):
        self._test_from_graph(self._unary('Exp'))

    def test_Expand(self):
        g = ONNXGraph(self._graph_name('Expand'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 3], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[5, 6, 2, 3], dtype='FLOAT')
        shape = ONNXTensor(graph=g, name='shape', shape=[4], data=[5, 6, 2, 3], dtype='INT64')
        ONNXOperation(graph=g, name='Expand', inputs=(x, shape), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_Flatten(self):
        g = ONNXGraph(self._graph_name('Flatten'))
        x = ONNXTensor(graph=g, name='x', shape=[2, 4, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[8, 25], dtype='FLOAT')
        ONNXOperation(graph=g, name='Flatten', inputs=x, outputs=y, attribs=dict(axis=2))
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_Floor(self):
        self._test_from_graph(self._unary('Floor'))

    def test_Gemm(self):
        g = ONNXGraph(self._graph_name('Gemm'))
        A = ONNXTensor(graph=g, name='A', shape=[5, 10], dtype='FLOAT')
        B = ONNXTensor(graph=g, name='B', shape=[10, 2], dtype='FLOAT')
        C = ONNXTensor(graph=g, name='C', shape=[5, 2], dtype='FLOAT')
        D = ONNXTensor(graph=g, name='D', shape=[5, 2], dtype='FLOAT')
        ONNXOperation(graph=g, name='Gemm', inputs=(A, B, C), outputs=D)
        g.inputs = (A, B, C)
        g.outputs = (D,)
        self._test_from_graph(g)

    def test_GlobalAveragePool(self):
        g = ONNXGraph(self._graph_name('GlobalAveragePool'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 1, 1], dtype='FLOAT')
        ONNXOperation(graph=g, name='GlobalAveragePool', inputs=x, outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_GlobalLpPool(self):
        g = ONNXGraph(self._graph_name('GlobalLpPool'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 1, 1], dtype='FLOAT')
        ONNXOperation(graph=g, name='GlobalLpPool', inputs=x, outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g, run=False)

    def test_GlobalMaxPool(self):
        g = ONNXGraph(self._graph_name('GlobalMaxPool'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 1, 1], dtype='FLOAT')
        ONNXOperation(graph=g, name='GlobalMaxPool', inputs=x, outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_Greater(self):
        g = ONNXGraph(self._graph_name('Greater'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 1], dtype='FLOAT')
        a = ONNXTensor(graph=g, name='a', shape=[1, 2, 5, 5], dtype='BOOL')
        ONNXOperation(graph=g, name='Greater', inputs=(x, y), outputs=a)
        g.inputs = (x, y)
        g.outputs = (a,)
        self._test_from_graph(g)

    def test_HardSigmoid(self):
        g = ONNXGraph(self._graph_name('HardSigmoid'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='HardSigmoid', inputs=x, attribs=dict(alpha=0.25, beta=0.6), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_Identity(self):
        g = ONNXGraph(self._graph_name('Identity'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='Identity', inputs=x, outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_InstanceNormalization(self):
        g = ONNXGraph(self._graph_name('InstanceNormalization'))
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
        self._test_from_graph(g)

    def test_LRN(self):
        g = ONNXGraph(self._graph_name('LRN'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 4, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 4, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='LRN', inputs=x, outputs=y, attribs=dict(alpha=0.001, beta=0.7, bias=0.9, size=3))
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_LeakyRelu(self):
        self._test_from_graph(self._unary('LeakyRelu', attribs=dict(alpha=0.01)))

    def test_Less(self):
        self._test_from_graph(self._binary('Less', dtype='FLOAT', out_dtype='BOOL'))

    def test_Log(self):
        self._test_from_graph(self._unary('Log'))

    def test_LogSoftmax(self):
        g = ONNXGraph(self._graph_name('LogSoftmax'))
        x = ONNXTensor(graph=g, name='x', shape=[5, 6, 2, 3], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[5, 6, 2, 3], dtype='FLOAT')
        ONNXOperation(graph=g, name='LogSoftmax', inputs=x, outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_LpNormalization(self):
        g = ONNXGraph(self._graph_name('LpNormalization'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 1, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 1, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='LpNormalization', inputs=x, outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g, run=False)

    def test_LpPool(self):
        g = ONNXGraph(self._graph_name('LpPool'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 2, 2], dtype='FLOAT')
        ONNXOperation(graph=g, name='LpPool', inputs=x, outputs=y, attribs=dict(kernel_shape=[3, 3],
                                                                                strides=[2, 2],
                                                                                p=1))
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g, run=False)

    def test_MatMul(self):
        g = ONNXGraph(self._graph_name('MatMul'))
        A = ONNXTensor(graph=g, name='A', shape=[5, 6], dtype='FLOAT')
        B = ONNXTensor(graph=g, name='B', shape=[6, 7], dtype='FLOAT')
        C = ONNXTensor(graph=g, name='C', shape=[5, 7], dtype='FLOAT')
        ONNXOperation(graph=g, name='MatMul', inputs=(A, B), outputs=C)
        g.inputs = (A, B)
        g.outputs = (C,)
        self._test_from_graph(g)

    def test_Max(self):
        self._test_from_graph(self._ternary('Max'))

    def test_MaxPool(self):
        g = ONNXGraph(self._graph_name('MaxPool'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 1, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 1, 3, 2], dtype='FLOAT')
        ONNXOperation(graph=g, name='MaxPool', inputs=(x,), outputs=(y,),
                      attribs=dict(kernel_shape=(3, 2), strides=(1, 2)))
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_MaxUnpool(self):
        g = ONNXGraph(self._graph_name('MaxUnpool'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 1, 2, 2], dtype='FLOAT')
        i = ONNXTensor(graph=g, name='i', shape=[1, 1, 2, 2], dtype='INT64', data=[2, 1, 0, 3])
        y = ONNXTensor(graph=g, name='y', shape=[1, 1, 4, 4], dtype='FLOAT')
        ONNXOperation(graph=g, name='MaxUnpool', inputs=(x, i), attribs=dict(kernel_shape=[2, 2],
                                                                             pads=[0, 0, 0, 0],
                                                                             strides=[2, 2]), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g, run=False)

    def test_Mean(self):
        self._test_from_graph(self._ternary('Mean'))

    def test_Min(self):
        self._test_from_graph(self._ternary('Min'))

    def test_Mul(self):
        self._test_from_graph(self._binary('Mul'))

    def test_Neg(self):
        self._test_from_graph(self._unary('Neg'))

    def test_Not(self):
        self._test_from_graph(self._unary('Not', dtype='BOOL'))

    def test_Or(self):
        self._test_from_graph(self._binary('Or', dtype='BOOL', out_dtype='BOOL'))

    def test_PRelu(self):
        g = ONNXGraph(self._graph_name('PRelu'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 3, 5, 3], dtype='FLOAT')
        slope = ONNXTensor(graph=g, name='slope', shape=[3], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 3, 5, 3], dtype='FLOAT')
        ONNXOperation(graph=g, name='PRelu', inputs=(x, slope), outputs=y)
        g.inputs = (x, slope)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_Pad(self):
        g = ONNXGraph(self._graph_name('Pad'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 3, 6, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='Pad', inputs=x, attribs=dict(pads=[0, 1, 0, 0,
                                                                        0, 0, 1, 0]), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g, run=False)

    def test_Pow(self):
        # Caffe2 pow can not broadcast
        g = ONNXGraph(self._graph_name('Pow'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 5], dtype='FLOAT')
        z = ONNXTensor(graph=g, name='z', shape=[1, 2, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='Pow',
                      inputs=(x, y),
                      outputs=z)
        g.inputs = (x, y)
        g.outputs = (z,)
        self._test_from_graph(g)

    def test_Reciprocal(self):
        self._test_from_graph(self._unary('Reciprocal'))

    def test_ReduceL1(self):
        self._test_from_graph(self._unary('ReduceL1', out_shape=[1, 1, 1, 5], attribs=dict(axes=[1, 2])), run=False)
        self._test_from_graph(self._unary('ReduceL1', out_shape=[1, 5], attribs=dict(axes=[1, 2], keepdims=0)),
                              run=False)

    def test_ReduceL2(self):
        self._test_from_graph(self._unary('ReduceL2', out_shape=[1, 1, 1, 5], attribs=dict(axes=[1, 2])), run=False)
        self._test_from_graph(self._unary('ReduceL2', out_shape=[1, 5], attribs=dict(axes=[1, 2], keepdims=0)),
                              run=False)

    def test_ReduceLogSum(self):
        self._test_from_graph(self._unary('ReduceLogSum', out_shape=[1, 1, 1, 5], attribs=dict(axes=[1, 2])), run=False)
        self._test_from_graph(self._unary('ReduceLogSum', out_shape=[1, 5], attribs=dict(axes=[1, 2], keepdims=0)),
                              run=False)

    def test_ReduceLogSumExp(self):
        self._test_from_graph(self._unary('ReduceLogSumExp', out_shape=[1, 1, 1, 5], attribs=dict(axes=[1, 2])),
                              run=False)
        self._test_from_graph(self._unary('ReduceLogSumExp', out_shape=[1, 5], attribs=dict(axes=[1, 2], keepdims=0)),
                              run=False)

    def test_ReduceMax(self):
        self._test_from_graph(self._unary('ReduceMax', out_shape=[1, 1, 1, 5], attribs=dict(axes=[1, 2])))
        self._test_from_graph(self._unary('ReduceMax', out_shape=[1, 5], attribs=dict(axes=[1, 2], keepdims=0)))

    def test_ReduceMean(self):
        self._test_from_graph(self._unary('ReduceMean', out_shape=[1, 1, 1, 5], attribs=dict(axes=[1, 2])))
        self._test_from_graph(self._unary('ReduceMean', out_shape=[1, 5], attribs=dict(axes=[1, 2], keepdims=0)))

    def test_ReduceMin(self):
        self._test_from_graph(self._unary('ReduceMin', out_shape=[1, 1, 1, 5], attribs=dict(axes=[1, 2])))
        self._test_from_graph(self._unary('ReduceMin', out_shape=[1, 5], attribs=dict(axes=[1, 2], keepdims=0)))

    def test_ReduceSum(self):
        self._test_from_graph(self._unary('ReduceSum', out_shape=[1, 1, 1, 5], attribs=dict(axes=[1, 2])))
        self._test_from_graph(self._unary('ReduceSum', out_shape=[1, 5], attribs=dict(axes=[1, 2], keepdims=0)))

    def test_ReduceSumSquare(self):
        self._test_from_graph(self._unary('ReduceSumSquare', out_shape=[1, 1, 1, 5], attribs=dict(axes=[1, 2])),
                              run=False)
        self._test_from_graph(self._unary('ReduceSumSquare', out_shape=[1, 5], attribs=dict(axes=[1, 2], keepdims=0)),
                              run=False)

    def test_Relu(self):
        self._test_from_graph(self._unary('Relu'))

    def test_Reshape(self):
        g = ONNXGraph(self._graph_name('Reshape'))
        input = ONNXTensor(graph=g, name='input', shape=[1, 3, 5, 3], dtype='FLOAT')
        shape = ONNXTensor(graph=g, name='shape', shape=[4], data=[0, 5, -1, 1], dtype='INT64')
        output = ONNXTensor(graph=g, name='output', shape=[1, 5, 9, 1], dtype='FLOAT')
        ONNXOperation(graph=g, name='Reshape', inputs=(input, shape), outputs=output)
        g.inputs = (input,)
        g.outputs = (output,)
        self._test_from_graph(g)

        g = ONNXGraph(self._graph_name('Reshape'))
        input = ONNXTensor(graph=g, name='input', shape=[1, 3, 5, 3], dtype='FLOAT')
        shape = ONNXTensor(graph=g, name='shape', shape=[1], data=[45], dtype='INT64')
        output = ONNXTensor(graph=g, name='output', shape=[45], dtype='FLOAT')
        ONNXOperation(graph=g, name='Reshape', inputs=(input, shape), outputs=output)
        g.inputs = (input,)
        g.outputs = (output,)
        self._test_from_graph(g)

    def test_Shape(self):
        self._test_from_graph(self._unary('Shape', shape=[1, 2, 3, 4], out_shape=[4], dtype='FLOAT', out_dtype='INT64'))

    def test_Sigmoid(self):
        self._test_from_graph(self._unary('Sigmoid'))

    def test_Sign(self):
        self._test_from_graph(self._unary('Sign'))

    def test_Size(self):
        self._test_from_graph(self._unary('Size', shape=[1, 2, 3, 4], out_shape=[], dtype='FLOAT', out_dtype='INT64'))

    def test_Slice(self):
        g = ONNXGraph(self._graph_name('Slice'))
        input = ONNXTensor(graph=g, name='input', shape=[1, 3, 5, 3], dtype='FLOAT')
        output = ONNXTensor(graph=g, name='output', shape=[1, 2, 5, 3], dtype='FLOAT')
        ONNXOperation(graph=g, name='Slice', inputs=input, outputs=output,
                      attribs=dict(starts=[0, 1, 0, 0], ends=[utils.INT32_MAX, 3, utils.INT32_MAX, utils.INT32_MAX]))
        g.inputs = (input,)
        g.outputs = (output,)
        self._test_from_graph(g)

    def test_Softmax(self):
        self._test_from_graph(self._unary('Softmax'))

    def test_Softplus(self):
        self._test_from_graph(self._unary('Softplus'))

    def test_Split(self):
        g = ONNXGraph(self._graph_name('Split'))
        input = ONNXTensor(graph=g, name='input', shape=[1, 3, 5, 3], dtype='FLOAT')
        a = ONNXTensor(graph=g, name='a', shape=[1, 3, 2, 3], dtype='FLOAT')
        b = ONNXTensor(graph=g, name='b', shape=[1, 3, 1, 3], dtype='FLOAT')
        c = ONNXTensor(graph=g, name='c', shape=[1, 3, 2, 3], dtype='FLOAT')
        ONNXOperation(graph=g, name='Split', inputs=input, outputs=[a, b, c], attribs=dict(axis=2, split=[2, 1, 2]))
        g.inputs = (input,)
        g.outputs = (a, b, c)
        self._test_from_graph(g)

    def test_Sqrt(self):
        self._test_from_graph(self._unary('Sqrt'))

    def test_Squeeze(self):
        self._test_from_graph(self._unary('Squeeze', shape=[1, 2, 1, 5], out_shape=[2, 5], attribs=dict(axes=[0, 2])))

    def test_Sub(self):
        self._test_from_graph(self._binary('Sub'))

    def test_Sum(self):
        self._test_from_graph(self._ternary('Sum'))

    def test_Tanh(self):
        self._test_from_graph(self._unary('Tanh'))

    def test_Tile(self):
        g = ONNXGraph(self._graph_name('Tile'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        repeats = ONNXTensor(graph=g, name='repeats', shape=[4], dtype='INT64', data=[1, 2, 2, 1])
        z = ONNXTensor(graph=g, name='z', shape=[1, 4, 10, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='Tile',
                      inputs=(x, repeats),
                      outputs=z)
        g.inputs = (x,)
        g.outputs = (z,)
        self._test_from_graph(g)

    def test_Transpose(self):
        self._test_from_graph(self._unary('Transpose', shape=[1, 2, 3, 4], out_shape=[4, 3, 2, 1]))
        self._test_from_graph(
            self._unary('Transpose', shape=[1, 2, 3, 4], out_shape=[4, 3, 1, 2], attribs=dict(perm=[3, 2, 0, 1])))

    def test_Unsqueeze(self):
        g = ONNXGraph(self._graph_name('Unsqueeze'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 1, 2, 5, 5, 1], dtype='FLOAT')
        ONNXOperation(graph=g, name='Unsqueeze', inputs=x, attribs=dict(axes=[0, 5]), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_Upsample(self):
        g = ONNXGraph(self._graph_name('Upsample'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        m = ONNXTensor(graph=g, name='m', shape=[4], dtype='FLOAT', data=[1.0, 1.0, 2.0, 2.0])
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 10, 10], dtype='FLOAT')
        ONNXOperation(graph=g, name='Upsample', inputs=(x, m), attribs=dict(mode='linear'), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)
        g = ONNXGraph(self._graph_name('Upsample'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        m = ONNXTensor(graph=g, name='m', shape=[4], dtype='FLOAT', data=[1.0, 1.0, 2.0, 2.0])
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 10, 10], dtype='FLOAT')
        ONNXOperation(graph=g, name='Upsample', inputs=(x, m), attribs=dict(mode='nearest'), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g)

    def test_Where(self):
        g = ONNXGraph(self._graph_name('Where'))
        cond = ONNXTensor(graph=g, name='cond', shape=[1, 3, 5, 3], dtype='BOOL')
        x = ONNXTensor(graph=g, name='x', shape=[1, 3, 5, 3], dtype='FLOAT', data=[1.0, 1.0, 2.0, 2.0])
        y = ONNXTensor(graph=g, name='y', shape=[1, 3, 5, 3], dtype='FLOAT')
        output = ONNXTensor(graph=g, name='output', shape=[1, 3, 5, 3], dtype='FLOAT')
        ONNXOperation(graph=g, name='Where', inputs=(cond, x, y), outputs=output)
        g.inputs = (cond, x, y)
        g.outputs = (output,)

    def test_Xor(self):
        self._test_from_graph(self._binary('Xor', dtype='BOOL'))

    def test_ImageScaler(self):
        g = ONNXGraph(self._graph_name('ImageScaler'))
        x = ONNXTensor(graph=g, name='x', shape=[1, 2, 5, 5], dtype='FLOAT')
        y = ONNXTensor(graph=g, name='y', shape=[1, 2, 5, 5], dtype='FLOAT')
        ONNXOperation(graph=g, name='ImageScaler', inputs=x, attribs=dict(bias=[7.0, 3.0], scale=2.0), outputs=y)
        g.inputs = (x,)
        g.outputs = (y,)
        self._test_from_graph(g, run=False)

    # Cannot test ConstantFill, not supported in current versions


if __name__ == '__main__':
    unittest.main()
