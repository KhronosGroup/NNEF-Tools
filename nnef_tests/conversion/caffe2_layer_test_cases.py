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

from nnef_tests.conversion.caffe2_test_runner import Caffe2TestRunner, Input

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')

DTYPE_ID_FLOAT = 1
DTYPE_ID_INT32 = 2
DTYPE_ID_BOOL = 5
DTYPE_ID_INT64 = 10

LEGACY_PAD_NOTSET = 0
LEGACY_PAD_VALID = 1
LEGACY_PAD_SAME = 2
LEGACY_PAD_CAFFE_LEGACY_POOLING = 3


# Tested with torch 1.1
class Caffe2LayerTestCases(Caffe2TestRunner):

    def _test_unary(self, _op_name, _input_dtype=DTYPE_ID_FLOAT, **kwargs):
        self._test_layer(_op_name, [
            Input('x', [1, 2, 3, 4], _input_dtype)
        ], **kwargs)

    def _test_binary(self, _op_name, _input_dtype=DTYPE_ID_FLOAT, **kwargs):
        self._test_layer(_op_name, [
            Input('x', [1, 2, 3, 4], _input_dtype),
            Input('y', [1, 2, 3, 4], _input_dtype),
        ], **kwargs)

        self._test_layer(_op_name, [
            Input('x', [1, 2, 3, 4], _input_dtype),
            Input('y', [4], _input_dtype),
        ], broadcast=1, **kwargs)

        # can only broadcast in the leading and trailing dimensions, so it wouldn't work with y=[3, 1, 3, 1]
        self._test_layer(_op_name, [
            Input('x', [3, 2, 3, 4], _input_dtype),
            Input('y', [1, 2, 3, 1], _input_dtype),
        ], broadcast=1, axis=0, **kwargs)

        self._test_layer(_op_name, [
            Input('x', [1, 2, 3, 4], _input_dtype),
            Input('y', [3, 4], _input_dtype),
        ], broadcast=1, **kwargs)

        self._test_layer(_op_name, [
            Input('x', [1, 2, 3, 4], _input_dtype),
            Input('y', [1, 4], _input_dtype),
        ], broadcast=1, **kwargs)

        self._test_layer(_op_name, [
            Input('x', [1, 2, 3, 4], _input_dtype),
            Input('y', [], _input_dtype),
        ], broadcast=1, **kwargs)

        self._test_layer(_op_name, [
            Input('x', [1, 2, 3, 4], _input_dtype),
            Input('y', [2, 3], _input_dtype),
        ], broadcast=1, axis=1, **kwargs)

        self._test_layer(_op_name, [
            Input('x', [1, 2, 3, 4], _input_dtype),
            Input('y', [2, 3], _input_dtype),
        ], broadcast=1, axis_str='H', order='NHWC', **kwargs)

    def _test_variadic(self, op_name):
        def model_fun(model):
            return getattr(model.net, op_name)(['x', 'y', 'z'], 'x'),

        self._test_model_fun(op_name, model_fun, [
            Input('x', [2, 3]),
            Input('y', [2, 3]),
            Input('z', [2, 3]),
        ])

        def model_fun2(model):
            return getattr(model.net, op_name)(['x', 'y', 'z'], 'a'),

        self._test_model_fun(op_name, model_fun2, [
            Input('x', [2, 3]),
            Input('y', [2, 3]),
            Input('z', [2, 3]),
        ])

        def model_fun3(model):
            return getattr(model.net, op_name)(['x'], 'x')

        self._test_model_fun(op_name, model_fun3, [
            Input('x', [1, 2, 3, 4]),
        ])

        def model_fun3(model):
            return getattr(model.net, op_name)(['x', 'y'], 'z')

        self._test_model_fun(op_name, model_fun3, [
            Input('x', [1, 2, 3, 4]),
            Input('y', [1, 2, 3, 4]),
        ])

    def test_abs(self):
        self._test_unary('Abs')

    def test_add(self):
        self._test_binary('Add')

    def test_and(self):
        self._test_binary('And', DTYPE_ID_BOOL)

    def test_argmax(self):
        self._test_layer('ArgMax', [
            Input('x', [2, 2, 2, 2]),
        ])

        self._test_layer('ArgMax', [
            Input('x', [2, 2, 2, 2]),
        ], axis=0, keepdims=1)

        self._test_layer('ArgMax', [
            Input('x', [2, 2, 2, 2]),
        ], axis=3, keepdims=0)

    def test_argmin(self):
        self._test_layer('ArgMin', [
            Input('x', [2, 2, 2, 2]),
        ])

        self._test_layer('ArgMin', [
            Input('x', [2, 2, 2, 2]),
        ], axis=0, keepdims=1)

        self._test_layer('ArgMin', [
            Input('x', [2, 2, 2, 2]),
        ], axis=3, keepdims=0)

    def test_average_pool(self):
        # Pooling does not support dilation

        self._test_layer('AveragePool', [  # seems like ignore border is used
            Input('x', [1, 1, 5, 5]),
        ], kernels=[2, 2], strides=[1, 1], pads=[1, 1, 1, 1], _feed_dict_override={
            'x': np.ones([1, 1, 5, 5], dtype=np.float32)
        })
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[3, 3], strides=[2, 2])
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[3, 3], legacy_pad=LEGACY_PAD_VALID)
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[3, 3], legacy_pad=LEGACY_PAD_SAME)
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[3, 3], strides=[3, 3], legacy_pad=LEGACY_PAD_SAME)
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[3, 3], strides=[3, 3], legacy_pad=LEGACY_PAD_CAFFE_LEGACY_POOLING)
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[3, 3], strides=[3, 3], legacy_pad=LEGACY_PAD_NOTSET)
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[2, 2], strides=[2, 2], legacy_pad=LEGACY_PAD_CAFFE_LEGACY_POOLING)
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[2, 2], strides=[2, 2], legacy_pad=LEGACY_PAD_NOTSET)
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[4, 4], strides=[3, 3], pads=[0, 0, 0, 0], legacy_pad=LEGACY_PAD_CAFFE_LEGACY_POOLING)
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[4, 4], strides=[3, 3], pads=[0, 0, 0, 0], legacy_pad=LEGACY_PAD_NOTSET)
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[4, 4], strides=[3, 3], pads=[1, 1, 0, 0], legacy_pad=LEGACY_PAD_CAFFE_LEGACY_POOLING)
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[4, 4], strides=[3, 3], pads=[1, 1, 0, 0], legacy_pad=LEGACY_PAD_NOTSET)
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernel=4, stride=3, pad=1)
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernel_h=3, kernel_w=4, stride_h=2, stride_w=3, pad_t=0, pad_l=0, pad_b=2, pad_r=1)
        self._test_layer('AveragePool', [
            Input('x', [1, 3, 5, 5]),
        ], kernel_h=3, kernel_w=4, stride_h=2, stride_w=3, pad_t=2, pad_l=1, pad_b=0, pad_r=0)
        # NHWC unsupported
        # self._test_layer('AveragePool', [
        #     Input('x', [1, 5, 5, 1]),
        # ], kernels=[5, 5], order='NHWC')
        self._test_layer('AveragePool', [
            Input('x', [1, 1, 5, 5]),
        ], kernels=[5, 5], order='nchw')
        self._test_layer('AveragePool', [
            Input('x', [1, 1, 5, 5]),
        ], kernels=[5, 5])
        self._test_layer('AveragePool1D', [
            Input('x', [1, 1, 5]),
        ], kernels=[2])
        self._test_layer('AveragePool2D', [
            Input('x', [1, 1, 5, 5]),
        ], kernels=[2, 3])
        self._test_layer('AveragePool3D', [
            Input('x', [1, 1, 5, 5, 5]),
        ], kernels=[2, 3, 4])
        self._test_layer('AveragePool3D', [
            Input('x', [1, 1, 5, 5, 5]),
        ], global_pooling=1)

    def test_batch_matmul(self):
        self._test_layer('BatchMatMul', [
            Input('x', [1, 2, 3, 2]),
            Input('y', [1, 2, 2, 4]),
        ])

        self._test_layer('BatchMatMul', [
            Input('x', [1, 2, 3, 2]),
            Input('y', [1, 1, 2, 4]),
        ], broadcast=1)

        self._test_layer('BatchMatMul', [
            Input('x', [1, 2, 2, 3]),
            Input('y', [1, 2, 2, 4]),
        ], trans_a=1)

        self._test_layer('BatchMatMul', [
            Input('x', [1, 2, 3, 2]),
            Input('y', [1, 2, 4, 2]),
        ], trans_b=1)

        self._test_layer('BatchMatMul', [
            Input('x', [1, 2, 3, 2]),
            Input('y', [2, 4]),
        ], broadcast=1)

        self._test_layer('BatchMatMul', [
            Input('x', [2]),  # (1x)2
            Input('y', [2]),  # 2(x1)
        ], broadcast=1)  # -> 1

        self._test_layer('BatchMatMul', [
            Input('x', [2]),  # (1x)2
            Input('y', [2, 3]),
        ], broadcast=1)  # -> 3

        self._test_layer('BatchMatMul', [
            Input('x', [3, 2]),
            Input('y', [2]),  # 2(x1)
        ], broadcast=1)  # -> 3

        self._test_layer('BatchMatMul', [
            Input('x', [5, 3, 2]),
            Input('y', [2]),  # 2(x1)
        ], broadcast=1)  # -> 5x3

        self._test_layer('BatchMatMul', [
            Input('x', [2]),  # (1x)2
            Input('y', [5, 2, 3]),
        ], broadcast=1)  # -> 5x3

        self._test_layer('BatchMatMul', [
            Input('x', [3, 2]),  # 1x2
            Input('y', [2, 3]),  # 2x3
        ], broadcast=1)  # -> 3x3

        self._test_layer('BatchMatMul', [
            Input('x', [2]),
            Input('y', [2]),
        ], broadcast=1, trans_a=1)  # -> 1

        self._test_layer('BatchMatMul', [
            Input('x', [2]),
            Input('y', [2]),
        ], broadcast=1, trans_b=1)  # -> 1

    def test_cast(self):
        self._test_layer('Cast', [
            Input('x', [1, 2, 3, 4], DTYPE_ID_FLOAT),
        ], to=DTYPE_ID_BOOL)

        self._test_layer('Cast', [
            Input('x', [1, 2, 3, 4], DTYPE_ID_BOOL),
        ], to=DTYPE_ID_FLOAT)

        self._test_layer('Cast', [
            Input('x', [1, 2, 3, 4], DTYPE_ID_FLOAT),
        ], to=DTYPE_ID_FLOAT)

        self._test_layer('Cast', [
            Input('x', [1, 2, 3, 4], DTYPE_ID_INT32),
        ], to=DTYPE_ID_INT32)

    def test_ceil(self):
        self._test_unary('Ceil')

    def test_channel_shuffle(self):
        self._test_layer('ChannelShuffle', [
            Input('x', [1, 12, 1, 1]),
        ])  # group=1
        self._test_layer('ChannelShuffle', [
            Input('x', [1, 12, 1, 1]),
        ], group=6)
        self._test_layer('ChannelShuffle', [
            Input('x', [1, 12]),
        ], group=6, order='NCHW')
        self._test_layer('ChannelShuffle', [
            Input('x', [1, 1, 1, 12]),
        ], group=6, order='NHWC')
        self._test_layer('ChannelShuffle', [
            Input('x', [1, 1, 1, 1, 12]),
        ], group=6, order='nhwc')

    def test_clip(self):
        self._test_unary('Clip')  # don't clip
        self._test_unary('Clip', min=0.0)
        self._test_unary('Clip', min=0.0, max=float('inf'))
        self._test_unary('Clip', max=6.0)
        self._test_unary('Clip', min=float('-inf'), max=6.0)
        self._test_unary('Clip', min=0.0, max=6.0)

    def test_concat(self):
        # Append is inplace only
        # NHWC is only supported in 4D

        def model_fun(model):
            return model.net.Append(['x', 'y'], 'x')

        self._test_model_fun('Append', model_fun, [
            Input('x', [10, 2, 3, 4]),
            Input('y', [2, 2, 3, 4]),
        ])

        self._test_layer('DepthConcat', [
            Input('x', [1, 2, 3]),
            Input('y', [1, 3, 3]),
        ], 2)

        self._test_layer('Concat', [
            Input('x', [1, 2, 3]),
            Input('y', [1, 3, 3]),
        ], 2)

        self._test_layer('Concat', [
            Input('x', [1, 2, 3, 4]),
            Input('y', [1, 2, 3, 3]),
        ], 2, order='NHWC')

        self._test_layer('Concat', [
            Input('x', [1, 2, 3]),
            Input('y', [2, 2, 3]),
        ], 2, axis=0)

        self._test_layer('Concat', [
            Input('x', [1, 2, 3]),
            Input('y', [2, 2, 3]),
        ], 2, axis=-3)

    def test_conditional(self):
        self._test_layer('Conditional', [
            Input('x', [5], DTYPE_ID_BOOL),
            Input('y', [5, 1, 2, 3], DTYPE_ID_FLOAT),
            Input('z', [5, 1, 2, 3], DTYPE_ID_FLOAT),
        ])

    def test_conv(self):
        self._test_layer('Conv', [  # border = constant
            Input('input', [1, 1, 5, 5]),
            Input('filter', [1, 1, 2, 2]),
        ], kernels=[2, 2], pads=[1, 1, 1, 1], _feed_dict_override={
            'input': np.ones([1, 1, 5, 5], dtype=np.float32),
            'filter': np.ones([1, 1, 2, 2], dtype=np.float32)
        })

        self._test_layer('Conv', [
            Input('input', [1, 3, 15, 15]),
            Input('filter', [6, 3, 2, 2]),
        ], kernels=[2, 2], pads=[0, 1, 1, 0])

        self._test_layer('Conv', [
            Input('input', [1, 3, 15, 15]),
            Input('filter', [6, 3, 2, 2]),
        ], kernels=[2, 2], strides=[3, 2], dilations=[2, 3], pads=[0, 1, 1, 0])

        # this would create negative padding
        self._test_layer('Conv', [
            Input('input', [1, 3, 15, 15]),
            Input('filter', [6, 3, 2, 2]),
        ], kernels=[2, 2], strides=[3, 2], legacy_pad=LEGACY_PAD_SAME)

        self._test_layer('Conv', [
            Input('input', [1, 3, 15, 15]),
            Input('filter', [6, 3, 2, 2]),
        ], kernels=[2, 2], strides=[3, 2], legacy_pad=LEGACY_PAD_VALID)

        self._test_layer('Conv', [
            Input('input', [1, 3, 15, 15]),
            Input('filter', [6, 3, 2, 2]),
            Input('bias', [6]),
        ], kernels=[2, 2], strides=[3, 2], dilations=[2, 3], pads=[0, 1, 1, 0])

        self._test_layer('Conv', [
            Input('input', [1, 3, 15, 15]),
            Input('filter', [6, 1, 2, 2]),
            Input('bias', [6]),
        ], kernels=[2, 2], strides=[3, 2], dilations=[2, 3], pads=[0, 1, 1, 0], group=3)

        self._test_layer('Conv', [
            Input('input', [1, 6, 15, 15]),
            Input('filter', [8, 3, 2, 2]),
            Input('bias', [8]),
        ], kernels=[2, 2], strides=[3, 2], dilations=[2, 3], pads=[0, 1, 1, 0], group=2)

        self._test_layer('Conv1D', [
            Input('input', [1, 3, 15]),
            Input('filter', [6, 3, 2]),
            Input('bias', [6]),
        ], kernels=[2], strides=[3], dilations=[2], pads=[0, 1])

        # NHWC unsupported
        # self._test_layer('Conv', [
        #     Input('input', [1, 15, 15, 3]),
        #     Input('filter', [6, 2, 2, 3]),
        #     Input('bias', [6]),
        # ], kernels=[2, 2], strides=[3, 2], dilations=[2, 3], pads=[0, 1, 1, 0], order='NHWC')
        #
        # self._test_layer('Conv', [
        #     Input('input', [1, 15, 3]),
        #     Input('filter', [6, 2, 1]),
        #     Input('bias', [6]),
        # ], kernels=[2], strides=[3], dilations=[2], pads=[0, 1], order='NHWC', group=3)

    def test_conv_transpose(self):
        # Dilation not supported
        # Only 4D is supported
        self._test_layer('ConvTranspose', [
            Input('input', [1, 3, 15, 15]),
            Input('filter', [3, 6, 2, 2]),
        ], kernels=[2, 2], pads=[0, 1, 1, 0])

        self._test_layer('ConvTranspose', [
            Input('input', [1, 3, 15, 15]),
            Input('filter', [3, 6, 2, 2]),
        ], kernels=[2, 2], strides=[3, 2], legacy_pad=LEGACY_PAD_SAME)

        self._test_layer('ConvTranspose', [
            Input('input', [1, 3, 15, 15]),
            Input('filter', [3, 6, 2, 2]),
        ], kernels=[2, 2], legacy_pad=LEGACY_PAD_VALID)

        self._test_layer('ConvTranspose', [
            Input('input', [1, 3, 15, 15]),
            Input('filter', [3, 6, 2, 2]),
        ], kernels=[2, 2], strides=[3, 2], pads=[0, 1, 1, 0])

        self._test_layer('ConvTranspose', [
            Input('input', [1, 3, 15, 15]),
            Input('filter', [3, 6, 2, 2]),
        ], kernels=[2, 2], strides=[3, 2], pads=[0, 1, 1, 0], adjs=[1, 1])

        self._test_layer('ConvTranspose', [
            Input('input', [1, 3, 15, 15]),
            Input('filter', [3, 6, 6, 6]),
        ], kernels=[6, 6], strides=[4, 4], pads=[0, 1, 1, 0], adjs=[3, 2])

        self._test_layer('ConvTranspose', [
            Input('input', [1, 3, 15, 15]),
            Input('filter', [3, 6, 2, 2]),
            Input('bias', [6]),
        ], kernels=[2, 2], strides=[3, 2], pads=[0, 1, 1, 0])

        self._test_layer('ConvTranspose', [
            Input('input', [1, 6, 15, 15]),
            Input('filter', [6, 1, 2, 2]),
            Input('bias', [3]),
        ], kernels=[2, 2], strides=[3, 2], pads=[0, 1, 1, 0], group=3)

        self._test_layer('ConvTranspose', [
            Input('input', [1, 8, 15, 15]),
            Input('filter', [8, 3, 2, 2]),
            Input('bias', [6]),
        ], kernels=[2, 2], strides=[3, 2], pads=[0, 1, 1, 0], group=2)

        # NHWC unsupported
        # self._test_layer('ConvTranspose', [
        #     Input('input', [1, 15, 15, 6]),
        #     Input('filter', [6, 2, 2, 3]),
        #     Input('bias', [3]),
        # ], kernels=[2, 2], strides=[3, 2], pads=[0, 1, 1, 0], order='NHWC')
        #
        # self._test_layer('ConvTranspose', [
        #     Input('input', [1, 15, 15, 8]),
        #     Input('filter', [8, 2, 2, 3]),
        #     Input('bias', [6]),
        # ], kernels=[2, 2], strides=[3, 2], pads=[0, 1, 1, 0], group=2, order='NHWC')

    def test_copy(self):
        self._test_unary('Copy')

        self._test_unary('CopyFromCPUInput')

        self._test_layer('CopyOnDeviceLike', [
            Input('x', [1, 2, 3, 4]),
            Input('y', [1, 2, 3, 4]),
        ])

        self._test_unary('EnsureCPUOutput')

        self._test_layer('StopGradient', [
            Input('x', [1, 2, 3, 4])
        ])

    def test_cos(self):
        self._test_unary('Cos', _can_convert=False)

    def test_div(self):
        self._test_binary('Div')

    def test_dot_product(self):
        # Elementwise product
        self._test_layer('DotProduct', [
            Input('x', [3]),
            Input('y', [3]),
        ])

        # Elementwise product and SumReduce(axes=[1], keepdims=False)
        self._test_layer('DotProduct', [
            Input('x', [3, 2]),
            Input('y', [3, 2]),
        ])

        # Elementwise product and SumReduce(axes=[1...], keepdims=False)
        self._test_layer('DotProduct', [
            Input('x', [3, 2, 2]),
            Input('y', [3, 2, 2]),
        ], _feed_dict_override={
            'x': np.ones([3, 2, 2], dtype=np.float32) * np.array([[[1]], [[2]], [[3]]], dtype=np.float32),
            'y': np.ones([3, 2, 2], dtype=np.float32)
        })

    def test_dropout(self):
        # Not deterministic in train mode
        # Can't read mask in test mode
        def model_fun(model):
            x, mask = model.net.Dropout(['x'], ['x', 'mask'], is_test=True)
            return x

        self._test_model_fun('Dropout', model_fun)

        def model_fun2(model):
            x, mask = model.net.Dropout('x', ['y', 'mask'], ratio=0.5, is_test=True)
            return x

        self._test_model_fun('Dropout', model_fun2)

    def test_eq(self):
        self._test_binary('EQ')

    def test_elementwise_linear(self):
        self._test_layer('ElementwiseLinear', [
            Input('x', [3, 4]),
            Input('w', [4]),
            Input('b', [4]),
        ])
        self._test_layer('ElementwiseLinear', [
            Input('x', [3, 3, 3, 3]),
            Input('w', [9]),
            Input('b', [9]),
        ], axis=2)

    def test_elu(self):
        self._test_unary('Elu')

    def test_exp(self):
        self._test_unary('Exp')

    def test_expand_dims(self):
        self._test_layer('ExpandDims', [
            Input('x', [2, 3, 4])
        ], dims=[0])

        self._test_layer('ExpandDims', [
            Input('x', [2, 1])
        ], dims=[0, 2])

    def test_fc(self):
        self._test_layer('FC', [
            Input('X', [5, 6]),
            Input('W', [3, 6]),
            Input('b', [3]),
        ])

        self._test_layer('FC', [
            Input('X', [5, 1, 3, 2]),
            Input('W', [2, 2, 3, 2]),
            Input('b', [4]),
        ], axis=2, axis_w=2)

        self._test_layer('FC', [
            Input('X', [2]),
            Input('W', [2]),
            Input('b', [1]),
        ], axis=0, axis_w=0)  # -> [1]

        self._test_layer('FC', [
            Input('X', [2]),
            Input('W', [1, 2]),
            Input('b', [1]),
        ], axis=0, axis_w=1)

        self._test_layer('FC', [
            Input('X', [1, 2]),
            Input('W', [2]),
            Input('b', [1]),
        ], axis=1, axis_w=0)

        self._test_layer('FC', [
            Input('X', [2]),
            Input('W', [3, 2]),
            Input('b', [3]),
        ], axis=0, axis_w=1)

        self._test_layer('FC', [
            Input('X', [3, 2]),
            Input('W', [2]),
            Input('b', [1]),
        ], axis=1, axis_w=0)

    def test_fc_transposed(self):
        self._test_layer('FCTransposed', [
            Input('X', [5, 6]),
            Input('W', [6, 3]),
            Input('b', [3]),
        ])

        self._test_layer('FCTransposed', [
            Input('X', [5, 1, 3, 2]),
            Input('W', [3, 2, 2, 2]),
            Input('b', [4]),
        ], axis=2, axis_w=2)

        self._test_layer('FCTransposed', [
            Input('X', [2]),
            Input('W', [2, 1]),
            Input('b', [1]),
        ], axis=0, axis_w=1)

        self._test_layer('FCTransposed', [
            Input('X', [2]),
            Input('W', [2, 3]),
            Input('b', [3]),
        ], axis=0, axis_w=1)

    def test_flatten(self):  # to 2d
        # negative axis not supported
        self._test_layer('Flatten', [
            Input('x', [1, 2, 3, 4])
        ])

        self._test_layer('Flatten', [
            Input('x', [1, 2, 3, 4])
        ], axis=2)

        self._test_layer('Flatten', [
            Input('x', [2])
        ])

        self._test_layer('Flatten', [
            Input('x', [2])
        ], axis=0)

    def test_flatten_to_vec(self):  # to 1d
        self._test_layer('FlattenToVec', [
            Input('x', [1, 2, 3, 4])
        ])
        self._test_layer('FlattenToVec', [
            Input('x', [1])
        ])

    def test_floor(self):
        self._test_unary('Floor')

    def test_ge(self):
        self._test_binary('GE')

    def test_gt(self):
        self._test_binary('GT')

    def test_instance_norm(self):
        self._test_layer('InstanceNorm', [
            Input('input', [3, 2, 3, 4]),
            Input('scale', [2]),
            Input('bias', [2]),
        ])  # epsilon = 1e-5

        # nhwc not supported
        def model_fun(model):
            return model.net.InstanceNorm(['input', 'scale', 'bias'],
                                          ['output', 'saved_mean', 'saved_inv_stdev'],
                                          epsilon=1e-4, order='NHWC')[0]

        self._test_model_fun('InstanceNorm', model_fun, [
            Input('input', [3, 2, 3, 4]),
            Input('scale', [4]),
            Input('bias', [4]),
        ], can_convert=False)

    def test_l1_distance(self):
        # Elementwise subtract + abs
        self._test_layer('L1Distance', [
            Input('x', [5]),
            Input('y', [5]),
        ])
        # Elementwise subtract + abs and SumReduce(axes=[1...], keepdims=False)
        self._test_layer('L1Distance', [
            Input('x', [5, 6]),
            Input('y', [5, 6]),
        ])

    def test_le(self):
        self._test_binary('LE')

    def test_lrn(self):
        # only 4D supported, only odd kernel size
        self._test_layer('LRN', [
            Input('x', [2, 6, 3, 4]),
        ], size=5, alpha=0.1, beta=0.2)
        self._test_layer('LRN', [
            Input('x', [2, 3, 3, 6]),
        ], size=5, alpha=0.1, beta=0.2, order='NHWC')
        self._test_layer('LRN', [
            Input('x', [2, 6, 3, 4]),
        ], size=5, alpha=0.1, beta=0.2, bias=1.1)
        self._test_layer('LRN', [  # scale not supported
            Input('x', [2, 6, 3, 4]),
        ], 2, size=5, alpha=0.1, beta=0.2, bias=1.1, _can_convert=False)

    def test_lt(self):
        self._test_binary('LT')

    def test_layer_norm(self):
        self._test_layer('LayerNorm', [
            Input('x', [2, 6, 3, 4]),
        ], 3)

        self._test_layer('LayerNorm', [
            Input('x', [2, 6, 3, 4]),
        ], 3, axis=3)

        self._test_layer('LayerNorm', [
            Input('x', [2, 6, 3, 4]),
        ], 3, axis=2, epsilon=0.01)

        self._test_layer('LayerNorm', [  # elementwise_affine unsupported
            Input('x', [2, 6, 3, 4]),
            Input('scale', [3, 4]),
            Input('bias', [3, 4]),
        ], 3, axis=2, epsilon=0.01, elementwise_affine=1, _can_convert=False)

    def test_leaky_relu(self):
        self._test_unary('LeakyRelu')
        self._test_unary('LeakyRelu', alpha=0.02)

    def test_log(self):
        self._test_layer('Log', [
            Input('x', [1, 2, 3, 4]),
        ], _feed_dict_override={'x': 0.01 + np.random.random([1, 2, 3, 4]).astype(np.float32)})

    def test_logit(self):
        self._test_unary('Logit')
        self._test_unary('Logit', eps=1e-5)

    def test_lp_norm(self):
        self._test_layer('LpNorm', [
            Input('x', [2]),
        ], p=1, average=1)
        self._test_layer('LpNorm', [
            Input('x', [2, 6, 3, 4]),
        ], p=1, average=0)
        self._test_layer('LpNorm', [
            Input('x', [2]),
        ], p=2, average=1)

    def test_lp_pool(self):
        self._test_layer('LpPool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[3, 3], strides=[2, 2])  # p = 2.0
        self._test_layer('LpPool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[3, 3], strides=[2, 2], p=1.0)
        self._test_layer('LpPool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[3, 3], strides=[2, 2], p=2.0)
        self._test_layer('LpPool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[3, 3], strides=[2, 2], p=3.1416)

    def test_matmul(self):
        self._test_layer('MatMul', [
            Input('x', [3, 4]),
            Input('y', [4, 5]),
        ])
        self._test_layer('MatMul', [
            Input('x', [4, 3]),
            Input('y', [4, 5]),
        ], trans_a=1)
        self._test_layer('MatMul', [
            Input('x', [3, 4]),
            Input('y', [5, 4]),
        ], trans_b=1)
        self._test_layer('MatMul', [
            Input('x', [2, 2, 1, 3]),
            Input('y', [1, 1, 5, 2, 2]),
        ], trans_a=1, axis_a=2, trans_b=1, axis_b=3)

        self._test_layer('MatMul', [
            Input('x', [5, 1, 3, 2]),
            Input('y', [3, 2, 2, 2]),
        ], axis_a=2, axis_b=2)

        self._test_layer('MatMul', [
            Input('x', [2]),
            Input('y', [2, 1]),
        ], axis_a=0, axis_b=1)

        self._test_layer('MatMul', [
            Input('x', [2]),
            Input('y', [2, 3]),
        ], axis_a=0, axis_b=1)

    def test_max(self):
        self._test_variadic('Max')

    def test_max_pool(self):
        self._test_layer('MaxPool', [  # seems like ignore border is used
            Input('x', [1, 1, 5, 5]),
        ], kernels=[2, 2], strides=[1, 1], pads=[1, 1, 1, 1], _feed_dict_override={
            'x': -np.ones([1, 1, 5, 5], dtype=np.float32)
        })
        self._test_layer('MaxPool', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[3, 3], strides=[2, 2])
        self._test_layer('MaxPool2D', [
            Input('x', [1, 3, 5, 5]),
        ], kernels=[3, 3], strides=[2, 2])
        self._test_layer('MaxPool3D', [
            Input('x', [1, 1, 5, 5, 5]),
        ], global_pooling=1)

    def test_max_pool_with_index(self):
        self._test_layer('MaxPoolWithIndex', [
            Input('x', [1, 3, 5, 5]),
        ], 2, kernels=[3, 3], strides=[2, 2], _can_run=False)  # This op only works on GPU

    def test_mean(self):
        self._test_variadic('Mean')

    def test_merge_dim(self):
        self._test_layer('MergeDim', [
            Input('x', [2, 3, 4])
        ])
        self._test_layer('MergeDim', [
            Input('x', [2, 3])
        ])

    def test_min(self):
        self._test_variadic('Min')

    def test_mul(self):
        self._test_binary('Mul')

    def test_ne(self):
        self._test_binary('NE')

    def test_negative(self):
        self._test_unary('Negative')

    def test_normalize(self):
        # L2 normalization
        self._test_layer('Normalize', [
            Input('x', [1, 3]),
        ], axis=0, _feed_dict_override={
            'x': np.ones([1, 3], dtype=np.float32)
        })
        self._test_layer('Normalize', [
            Input('x', [3, 1]),
        ], axis=0, _feed_dict_override={
            'x': np.ones([3, 1], dtype=np.float32)
        })
        self._test_layer('Normalize', [
            Input('x', [2, 6, 3, 4]),
        ])  # axis=-1
        self._test_layer('Normalize', [
            Input('x', [2, 6, 3, 4]),
        ], axis=0)
        self._test_layer('Normalize', [
            Input('x', [2, 6, 3, 4]),
        ], axis=1)

    def test_normalize_l1(self):
        # L1 normalization
        self._test_layer('NormalizeL1', [
            Input('x', [2, 6, 3, 4]),
        ])  # axis=-1
        self._test_layer('NormalizeL1', [
            Input('x', [2, 6, 3, 4]),
        ], axis=0)
        self._test_layer('NormalizeL1', [
            Input('x', [2, 6, 3, 4]),
        ], axis=1)

    def test_not(self):
        self._test_unary('Not', DTYPE_ID_BOOL)

    def test_or(self):
        self._test_binary('Or', DTYPE_ID_BOOL)

    def test_prelu(self):
        self._test_layer('PRelu', [
            Input('x', [2, 6, 3, 4]),
            Input('y', [6]),
        ])

        self._test_layer('PRelu', [
            Input('x', [2, 6, 3, 4]),
            Input('y', [1]),
        ])

        self._test_layer('PRelu', [
            Input('x', [2, 6, 3, 4]),
            Input('y', [4]),
        ], order='NHWC')

        self._test_layer('PRelu', [
            Input('x', [2, 3, 3, 2, 4]),
            Input('y', [4]),
        ], order='NHWC')

    def test_pad_image(self):
        self._test_layer('PadImage', [
            Input('x', [1, 2, 3, 4])
        ], _can_convert=False)

        self._test_layer('PadImage', [
            Input('x', [1, 2, 3, 4])
        ], pads=[0, 1, 2, 3], _can_convert=False)

        self._test_layer('PadImage', [
            Input('x', [1, 2, 3, 4])
        ], pads=[0, 1, 2, 3], order="NHWC", _can_convert=False)

    def test_pow(self):
        self._test_layer('Pow', [
            Input('x', [1, 2, 3, 4]),
            Input('y', [1, 2, 3, 4]),
        ], _feed_dict_override={'x': 0.01 + np.random.random([1, 2, 3, 4]).astype(np.float32),
                                'y': -0.5 + np.random.random([1, 2, 3, 4]).astype(np.float32)})

        self._test_layer('Pow', [
            Input('x', [1, 2, 3, 4]),
            Input('y', [1]),
        ], broadcast=1, _feed_dict_override={'x': 0.01 + np.random.random([1, 2, 3, 4]).astype(np.float32),
                                             'y': -0.5 + np.random.random([1]).astype(np.float32)})

        self._test_layer('Pow', [
            Input('x', [1, 2, 3, 4]),
        ], exponent=2.3, _feed_dict_override={'x': 0.01 + np.random.random([1, 2, 3, 4]).astype(np.float32)})

    def test_prepend_dim(self):
        self._test_layer('PrependDim', [
            Input('x', [6, 2, 2])
        ], dim_size=3)
        self._test_layer('PrependDim', [
            Input('x', [6])
        ], dim_size=3)

    def test_range(self):
        def model_fun(model):
            model.param_init_net.GivenTensorFill([], 'start', values=[0.0], shape=[])
            model.param_init_net.GivenTensorFill([], 'stop', values=[16.0], shape=[])
            model.param_init_net.GivenTensorFill([], 'step', values=[1.0], shape=[])
            range = model.net.Range(['start', 'stop', 'step'], ['range'])
            return model.net.Add([range, 'input'], 'output', broadcast=1, axis=0)

        self._test_model_fun('Range', model_fun, [Input('input', [1])])

        def model_fun2(model):
            model.param_init_net.GivenTensorFill([], 'start', values=[16.0], shape=[])
            model.param_init_net.GivenTensorFill([], 'stop', values=[0.0], shape=[])
            model.param_init_net.GivenTensorFill([], 'step', values=[-0.5], shape=[])
            range = model.net.Range(['start', 'stop', 'step'], ['range'])
            return model.net.Add([range, 'input'], 'output', broadcast=1, axis=0)

        self._test_model_fun('Range', model_fun2, [Input('input', [1])])

        def model_fun3(model):
            model.param_init_net.GivenTensorFill([], 'stop', values=[16.0], shape=[])
            range = model.net.Range(['stop'], ['range'])
            return model.net.Add([range, 'input'], 'output', broadcast=1, axis=0)

        self._test_model_fun('Range', model_fun3, [Input('input', [1])])

    def test_reduce_min(self):
        self._test_layer('ReduceMin', [
            Input('x', [2, 2, 2, 2]),
        ])

        self._test_layer('ReduceMin', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2])

        self._test_layer('ReduceMin', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2], keepdims=0)

    def test_reduce_max(self):
        self._test_layer('ReduceMax', [
            Input('x', [2, 2, 2, 2]),
        ])

        self._test_layer('ReduceMax', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2])

        self._test_layer('ReduceMax', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2], keepdims=0)

        self._test_layer('ReduceFrontMax', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2], keepdims=0)

        self._test_layer('ReduceBackMax', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2], keepdims=0)

        self._test_layer('ColwiseMax', [
            Input('x', [2, 3, 4]),
        ])

        self._test_layer('RowwiseMax', [
            Input('x', [2, 3, 4]),
        ])

    def test_reduce_sum(self):
        self._test_layer('ReduceSum', [
            Input('x', [2, 2, 2, 2]),
        ])

        self._test_layer('ReduceSum', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2])

        self._test_layer('ReduceSum', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2], keepdims=0)

        self._test_layer('ReduceFrontSum', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2], keepdims=0)

        self._test_layer('ReduceBackSum', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2], keepdims=0)

        self._test_layer('ReduceTailSum', [
            Input('x', [2, 3, 4, 5]),
        ])

        self._test_layer('SumElements', [
            Input('x', [2, 3, 4]),
        ])

        self._test_layer('SumElements', [
            Input('x', [2, 3, 4]),
        ], average=True)

    def test_reduce_mean(self):
        self._test_layer('ReduceMean', [
            Input('x', [2, 2, 2, 2]),
        ])

        self._test_layer('ReduceMean', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2])

        self._test_layer('ReduceMean', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2], keepdims=0)

        self._test_layer('ReduceFrontMean', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2], keepdims=0)

        self._test_layer('ReduceBackMean', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2], keepdims=0)

    def test_reduce_l1(self):
        self._test_layer('ReduceL1', [
            Input('x', [2, 2, 2, 2]),
        ])

        self._test_layer('ReduceL1', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2])

        self._test_layer('ReduceL1', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2], keepdims=0)

    def test_reduce_l2(self):
        self._test_layer('ReduceL2', [
            Input('x', [2, 2, 2, 2]),
        ])

        self._test_layer('ReduceL2', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2])

        self._test_layer('ReduceL2', [
            Input('x', [2, 2, 2, 2]),
        ], axes=[1, 2], keepdims=0)

    def test_relu(self):
        self._test_unary('Relu')

    def test_reshape(self):
        self._test_layer('Reshape', [
            Input('x', [1, 2, 3, 4]),
        ], 2, shape=[2, 3, 4])
        self._test_layer('Reshape', [
            Input('x', [1, 2, 3, 4]),
        ], 2, shape=[0, 0, -1])

        def model_fun(model):
            model.param_init_net.GivenTensorInt64Fill([], 'shape', values=[2, 3, 4], shape=[3])
            return model.net.Reshape(['input', 'shape'], ['output', 'old_shape'])

        self._test_model_fun('Reshape', model_fun, [
            Input('input', [1, 2, 3, 4])
        ])

        def model_fun2(model):
            model.param_init_net.GivenTensorInt64Fill([], 'shape', values=[0, 0, -1], shape=[3])
            return model.net.Reshape(['input', 'shape'], ['output', 'old_shape'])

        self._test_model_fun('Reshape', model_fun2, [
            Input('input', [1, 2, 3, 4])
        ])

        def model_fun2(model):
            model.param_init_net.GivenTensorInt64Fill([], 'shape', values=[0, 0, -1], shape=[3])
            reshaped, old_shape = model.net.Reshape(['input', 'shape'], ['reshaped', 'old_shape'])
            return model.net.Reshape([reshaped, old_shape], ['reshaped2', 'old_shape2'])[0]

        self._test_model_fun('Reshape', model_fun2, [
            Input('input', [1, 2, 3, 4])
        ])

    def test_resize_like(self):
        self._test_layer('ResizeLike', [
            Input('x', [1, 2, 3, 4]),
            Input('y', [24]),
        ])
        self._test_layer('ResizeLike', [
            Input('x', [1, 2, 3, 4]),
            Input('y', [2, 3, 4]),
        ])

    def test_resize_nearest(self):
        self._test_layer('ResizeNearest', [
            Input('x', [1, 2, 4, 4]),
        ], width_scale=1.0, height_scale=1.0)
        self._test_layer('ResizeNearest', [
            Input('x', [1, 2, 4, 4]),
        ], width_scale=2.0, height_scale=2.0)
        self._test_layer('ResizeNearest', [
            Input('x', [1, 2, 4, 4]),
        ], width_scale=0.5, height_scale=0.5)
        self._test_layer('ResizeNearest', [
            Input('x', [1, 2, 4, 4]),
        ], width_scale=0.5, height_scale=2.0)
        self._test_layer('ResizeNearest', [
            Input('x', [1, 2, 4, 4]),
        ], width_scale=2.0, height_scale=0.5)
        self._test_layer('ResizeNearest', [  # only integer resize factor is supported
            Input('x', [1, 2, 12, 12]),
        ], width_scale=0.2, height_scale=2.3, _can_convert=False)

    def test_roi_align(self):
        # sampling_ratio=-1 unsupported in Caffe2
        self._test_layer('RoIAlign', [
            Input('x', [2, 2, 32, 32]),
            Input('rois', [3, 5]),
        ], spatial_scale=2.0, pooled_h=4, pooled_w=4, sampling_ratio=4, _feed_dict_override={
            'rois': np.array([
                # batch_index, x1, y1, x2, y2
                [0, 3, 3, 10, 10],
                [1, 4, 4, 20, 20],
                [1, 5, 5, 9, 9],
            ], dtype=np.float32)
        }, _can_convert=False)
        self._test_layer('RoIAlign', [
            Input('x', [1, 2, 32, 32]),
            Input('rois', [3, 4]),
        ], spatial_scale=2.0, pooled_h=4, pooled_w=4, sampling_ratio=4, _feed_dict_override={
            'rois': np.array([
                # x1, y1, x2, y2
                [3, 3, 10, 10],
                [4, 4, 20, 20],
                [5, 5, 9, 9],
            ], dtype=np.float32)
        }, _can_convert=False)
        self._test_layer('RoIAlign', [
            Input('x', [1, 32, 32, 2]),
            Input('rois', [3, 4]),
        ], spatial_scale=2.0, pooled_h=4, pooled_w=4, sampling_ratio=4, order='NHWC', _feed_dict_override={
            'rois': np.array([
                # x1, y1, x2, y2
                [3, 3, 10, 10],
                [4, 4, 20, 20],
                [5, 5, 9, 9],
            ], dtype=np.float32)
        }, _can_convert=False)

    def test_roi_pool(self):
        # NHWC is unsupported in Caffe2

        self._test_layer('RoIPool', [
            Input('x', [2, 2, 32, 32]),
            Input('rois', [3, 5]),
        ], 1, spatial_scale=2.0, pooled_h=4, pooled_w=4, is_test=True, _feed_dict_override={
            'rois': np.array([
                # batch_index, x1, y1, x2, y2
                [0, 3, 3, 10, 10],
                [1, 4, 4, 20, 20],
                [1, 5, 5, 9, 9],
            ], dtype=np.float32)
        }, _can_convert=False)

    def test_row_mul(self):
        self._test_layer('RowMul', [
            Input('x', [3, 5]),
            Input('y', [3]),
        ])

        self._test_layer('RowMul', [
            Input('x', [3, 5, 6]),
            Input('y', [3]),
        ])

        self._test_layer('RowMul', [
            Input('x', [3]),
            Input('y', [3]),
        ])

        self._test_layer('RowMul', [
            Input('x', [12, 5]),
            Input('y', [3, 2, 2]),
        ])

    def test_scale(self):
        self._test_layer('Scale', [
            Input('x', [2, 3, 4]),
        ], scale=-3.4)

        self._test_layer('Scale', [
            Input('x', [2, 3, 4]),
        ])  # scale=1.0

    def test_selu(self):
        self._test_unary('Selu', alpha=1.7, scale=1.06)
        self._test_unary('Selu')  # alpha=1.6732632423543772848170429916717f, scale=1.0507009873554804934193349852946f

    def test_shape(self):
        self._test_layer('Shape', [
            Input('x', [1, 2, 3, 4]),
        ])

        self._test_layer('Shape', [
            Input('x', [1, 2]),
        ])

        self._test_layer('Shape', [
            Input('x', []),
        ])

        def model_fun(model):
            reference_shape = model.net.Shape(['reference'], ['reference_shape'])
            return model.net.Reshape(['input', reference_shape], ['output', 'old_shape'])[0]

        self._test_model_fun('Reshape', model_fun, [
            Input('input', [1, 2, 3, 4]),
            Input('reference', [6, 4])
        ])

    def test_sigmoid(self):
        self._test_unary('Sigmoid')

    def test_sign(self):
        self._test_unary('Sign')

    def test_sin(self):
        self._test_unary('Sin', _can_convert=False)

    def test_size(self):
        self._test_layer('Size', [
            Input('x', [1, 2, 3, 4]),
        ])

        self._test_layer('Size', [
            Input('x', [1, 2]),
        ])

        self._test_layer('Size', [
            Input('x', []),
        ])

    def test_slice(self):
        # Can only slice in 1 dimension

        self._test_layer('Slice', [
            Input('x', [3, 4, 5]),
        ], starts=[0, 1, 0], ends=[3, 2, -1])

        self._test_layer('Slice', [
            Input('x', [3, 4, 5]),
        ], starts=[0, 1, 0], ends=[3, -2, -1])

        def model_fun(model):
            model.param_init_net.GivenTensorInt64Fill([], 'starts', values=[0, 1, 0], shape=[3])
            model.param_init_net.GivenTensorInt64Fill([], 'ends', values=[3, 2, -1], shape=[3])
            return model.net.Slice(['input', 'starts', 'ends'], ['sliced'])

        self._test_model_fun('Slice', model_fun, [
            Input('input', [3, 4, 5]),
        ])

    def test_softmax(self):
        # axis works like for mat mul
        # axis must be < input.rank

        self._test_layer('Softmax', [
            Input('x', [3, 4, 5, 2]),
        ])  # axis=1

        self._test_layer('Softmax', [
            Input('x', [3, 4, 5, 2]),
        ], axis=2)

    def test_softplus(self):
        self._test_unary('Softplus')

    def test_softsign(self):
        self._test_unary('Softsign')

    def test_spatial_bn(self):
        self._test_layer('SpatialBN', [
            Input('x', [1, 2, 3, 4]),
            Input('scale', [2]),
            Input('bias', [2]),
            Input('mean', [2]),
            Input('var', [2]),
        ], is_test=True, epsilon=0.01, order='NCHW', _feed_dict_override={
            'scale': np.array([0.2, 0.3], dtype=np.float32),
            'var': np.array([0.5, 0.4], dtype=np.float32),
        })

        self._test_layer('SpatialBN', [
            Input('x', [1, 3, 4, 2]),
            Input('scale', [2]),
            Input('bias', [2]),
            Input('mean', [2]),
            Input('var', [2]),
        ], is_test=True, epsilon=0.01, order='NHWC', _feed_dict_override={
            'scale': np.array([0.2, 0.3], dtype=np.float32),
            'var': np.array([0.5, 0.4], dtype=np.float32),
        })

        self._test_layer('SpatialBN', [
            Input('x', [1, 2, 3, 4]),
            Input('scale', [2]),
            Input('bias', [2]),
            Input('mean', [2]),
            Input('var', [2]),
            Input('sum', [2]),
            Input('sumsq', [2]),
        ], is_test=True, epsilon=0.01, order='NCHW', momentum=0.9, num_batches=5, _feed_dict_override={
            'scale': np.array([0.2, 0.3], dtype=np.float32),
            'var': np.array([0.5, 0.4], dtype=np.float32),
        })

    def test_split(self):
        self._test_layer('Split', [
            Input('x', [2, 6, 4, 5]),
        ], 3, split=[1, 2, 3])

        self._test_layer('DepthSplit', [
            Input('x', [2, 6, 4, 5]),
        ], 3, split=[1, 2, 3])

        self._test_layer('Split', [
            Input('x', [2, 6, 4, 5]),
        ], 3, axis=2, split=[1, 2, 1])

        self._test_layer('Split', [
            Input('x', [2, 6, 4, 5]),
        ], 1, axis=2)

        self._test_layer('Split', [
            Input('x', [2, 6, 4, 5]),
        ], 2, axis=2)

        self._test_layer('Split', [
            Input('x', [2, 6, 4, 5]),
        ], 5, order='NHWC')

        self._test_layer('Split', [
            Input('x', [2, 6, 4, 5]),
        ], 2, order='NHWC', split=[1, 4])

        def model_fun(model):
            model.param_init_net.GivenTensorIntFill([], 'split', values=[1, 2, 2], shape=[3])
            return model.net.Split(['input', 'split'], ['s1', 's2', 's3'], axis=2)

        self._test_model_fun('Split', model_fun, [
            Input('input', [1, 2, 5, 4])
        ])

        def model_fun2(model):
            ab, split_info = model.net.Concat(['a', 'b'], ['ab', 'split_info'])
            return model.net.Split([ab, split_info], ['out_a', 'out_b'])

        self._test_model_fun('Split', model_fun2, [
            Input('a', [1, 2, 3]),
            Input('b', [1, 3, 3]),
        ])

    def test_sqr(self):
        self._test_unary('Sqr')

    def test_sqrt(self):
        self._test_layer('Sqrt', [
            Input('x', [1, 2, 3, 4])
        ], _feed_dict_override={
            'x': np.random.random([1, 2, 3, 4]).astype(np.float32)  # non-neg
        })

    def test_squared_l2_distance(self):
        # (x-y)^2 / 2
        self._test_layer('SquaredL2Distance', [
            Input('x', [5]),
            Input('y', [5]),
        ])
        # (x-y)^2 / 2 and SumReduce(axes=[1...], keepdims=False)
        self._test_layer('SquaredL2Distance', [
            Input('x', [5, 6]),
            Input('y', [5, 6]),
        ])

    def test_squeeze(self):
        self._test_layer('Squeeze', [
            Input('x', [1, 2, 3, 4])
        ], dims=[0])

        self._test_layer('Squeeze', [
            Input('x', [1, 2, 1, 1])
        ], dims=[0, 2])

    def test_stump_func(self):
        # out[i] = low_value if in[i] <= threshold else high_value
        self._test_layer('StumpFunc', [
            Input('x', [1, 2, 3, 4])
        ])  # all params 0

        self._test_layer('StumpFunc', [
            Input('x', [1, 2, 3, 4])
        ], threshold=3.0, low_value=-1.0, value=1.0)

    def test_sub(self):
        self._test_binary('Sub')

    def test_sum(self):
        self._test_variadic('Sum')

    def test_sum_sqr_elements(self):
        self._test_layer('SumSqrElements', [
            Input('x', [2, 3, 4]),
        ])

        self._test_layer('SumSqrElements', [
            Input('x', [2, 3, 4]),
        ], average=True)

    def test_sum_reduce_like(self):
        self._test_layer('SumReduceLike', [
            Input('x', [2, 3, 4]),
            Input('y', [2, 3, 4]),
        ])

        self._test_layer('SumReduceLike', [
            Input('x', [2, 3, 4]),
            Input('y', [3, 4]),
        ])

        self._test_layer('SumReduceLike', [
            Input('x', [2, 3, 4]),
            Input('y', [4]),
        ])

        self._test_layer('SumReduceLike', [
            Input('x', [2, 3, 4]),
            Input('y', []),
        ])

        self._test_layer('SumReduceLike', [
            Input('x', [2, 3, 4]),
            Input('y', [3, 4]),
        ], axis=-1)

        self._test_layer('SumReduceLike', [
            Input('x', [2, 3, 4]),
            Input('y', []),
        ], axis=3)

        self._test_layer('SumReduceLike', [
            Input('x', [2, 3, 4]),
            Input('y', [3]),
        ], axis_str='C')

        self._test_layer('SumReduceLike', [
            Input('x', [2, 3, 4]),
            Input('y', [3, 4]),
        ], axis_str='H', order='NHWC')

    def test_summarize(self):
        self._test_layer('Summarize', [
            Input('x', [1, 2, 3, 4])
        ])
        self._test_layer('Summarize', [  # does not have output
            Input('x', [1, 2, 3, 4])
        ], 0, to_file=1, _can_convert=False)

    def test_swish(self):
        self._test_unary('Swish')

    def test_tanh(self):
        self._test_unary('Tanh')

    def test_thresholded_relu(self):
        # Yvec = (Xvec > alpha_).select(Xvec, 0.f)
        self._test_unary('ThresholdedRelu')
        self._test_unary('ThresholdedRelu', alpha=2.0)

    def test_tile(self):
        self._test_layer('Tile', [
            Input('x', [1, 2, 3, 4])
        ], axis=1, tiles=3, _can_convert=False)

        self._test_layer('Tile', [
            Input('x', [1, 2, 3, 4])
        ], axis=-1, tiles=3, _can_convert=False)

        def model_fun(model):
            model.param_init_net.GivenTensorIntFill([], 'tiles', values=[3], shape=[1])
            model.param_init_net.GivenTensorIntFill([], 'axis', values=[1], shape=[1])
            return model.net.Tile(['input', 'tiles', 'axis'], ['output'])

        self._test_model_fun('Tile', model_fun, [
            Input('input', [1, 2, 3, 4])
        ], can_convert=False)

    def test_transpose(self):
        self._test_layer('Transpose', [
            Input('x', [1, 2, 3, 4])
        ])  # revert dims
        self._test_layer('Transpose', [
            Input('x', [1, 2, 3, 4])
        ], axes=[0, 2, 3, 1])
        # works for rank >= 3
        self._test_layer('NCHW2NHWC', [
            Input('x', [1, 2, 3, 4, 5])
        ])
        self._test_layer('NCHW2NHWC', [
            Input('x', [1, 2, 3, 4])
        ])
        self._test_layer('NCHW2NHWC', [
            Input('x', [1, 2, 3])
        ])
        self._test_layer('NHWC2NCHW', [
            Input('x', [1, 2, 3, 4, 5])
        ])
        self._test_layer('NHWC2NCHW', [
            Input('x', [1, 2, 3, 4])
        ])
        self._test_layer('NHWC2NCHW', [
            Input('x', [1, 2, 3])
        ])

    def test_weighted_sum(self):
        # We don't support this
        self._test_layer('WeightedSum', [
            Input('x0', [1, 2, 3, 4]),
            Input('w0', []),
            Input('x1', [1, 2, 3, 4]),
            Input('w1', []),
            Input('x2', [1, 2, 3, 4]),
            Input('w2', []),
        ], _can_convert=False)
        self._test_layer('WeightedSum', [
            Input('x0', [1, 2, 3, 4]),
            Input('w0', []),
        ], _can_convert=False)

    def test_where(self):
        self._test_layer('Where', [
            Input('x', [5, 1, 2, 3], DTYPE_ID_BOOL),
            Input('y', [5, 1, 2, 3], DTYPE_ID_FLOAT),
            Input('z', [5, 1, 2, 3], DTYPE_ID_FLOAT),
        ])

    def test_xor(self):
        self._test_binary('Xor', DTYPE_ID_BOOL)


if __name__ == '__main__':
    unittest.main()
