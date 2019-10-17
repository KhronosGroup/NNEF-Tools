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

import unittest

from nnef_tools.backend.pytorch.operations import *


class FunctionsTestCases(unittest.TestCase):
    def test_desample(self):
        input = torch.rand(1, 3, 5, 5)
        pooled, index = nnef_max_pool_with_index(input, size=[1, 1, 2, 2])
        unpooled = nnef_desample(pooled, index, size=[1, 1, 2, 2], output_shape=list(input.shape))
        pooled2, index2 = nnef_max_pool_with_index(unpooled, size=[1, 1, 2, 2])
        self.assertEqual(unpooled.shape, input.shape)
        self.assertTrue(np.all(np.equal(pooled2.numpy(), pooled.numpy())))
        self.assertTrue(np.all(np.equal(index2.numpy(), index2.numpy())))

    def test_desample2(self):
        input = torch.rand(1, 3, 5, 5)
        size = [1, 1, 2, 2]
        padding = [(0, 0), (0, 0), (0, 1), (1, 0)]
        pooled, index = nnef_max_pool_with_index(input, size=size, padding=padding)
        unpooled = nnef_desample(pooled, index, size=size, padding=padding, output_shape=list(input.shape))
        pooled2, index2 = nnef_max_pool_with_index(unpooled, size=size, padding=padding)
        self.assertEqual(unpooled.shape, input.shape)
        self.assertTrue(np.all(np.equal(pooled2.numpy(), pooled.numpy())))
        self.assertTrue(np.all(np.equal(index2.numpy(), index2.numpy())))

    def test_desample3(self):
        input = torch.rand(1, 3, 5, 5)
        pooled, index = nnef_max_pool_with_index(input, size=[1, 1, 2, 2])
        unpooled = nnef_desample(pooled, index, size=[1, 1, 2, 2])
        pooled2, index2 = nnef_max_pool_with_index(unpooled, size=[1, 1, 2, 2])
        self.assertEqual(unpooled.shape, input.shape)
        self.assertTrue(np.all(np.equal(pooled2.numpy(), pooled.numpy())))
        self.assertTrue(np.all(np.equal(index2.numpy(), index2.numpy())))

    def test_desample4(self):
        input = torch.rand(1, 3, 5, 5)
        size = [1, 1, 2, 2]
        padding = [(0, 0), (0, 0), (0, 1), (1, 0)]
        stride = [1, 1, 2, 2]
        pooled, index = nnef_max_pool_with_index(input, size=size, stride=stride, padding=padding)
        unpooled = nnef_desample(pooled, index, size=size, stride=stride, padding=padding,
                                 output_shape=list(input.shape))
        pooled2, index2 = nnef_max_pool_with_index(unpooled, size=size, stride=stride, padding=padding)
        self.assertEqual(unpooled.shape, input.shape)
        self.assertTrue(np.all(np.equal(pooled2.numpy(), pooled.numpy())))
        self.assertTrue(np.all(np.equal(index2.numpy(), index2.numpy())))

    def test_desample5(self):
        input = torch.rand(1, 3, 10, 10)
        size = [1, 1, 3, 3]
        padding = [(0, 0), (0, 0), (0, 0), (0, 0)]
        stride = [1, 1, 1, 3]
        pooled, index = nnef_max_pool_with_index(input, size=size, stride=stride, padding=padding)
        unpooled = nnef_desample(pooled, index, size=size, stride=stride, padding=padding,
                                 output_shape=list(input.shape))
        pooled2, index2 = nnef_max_pool_with_index(unpooled, size=size, stride=stride, padding=padding)
        self.assertEqual(unpooled.shape, input.shape)
        self.assertTrue(np.all(np.equal(pooled2.numpy(), pooled.numpy())))
        self.assertTrue(np.all(np.equal(index2.numpy(), index2.numpy())))

    def test_desample6(self):
        input = torch.rand(2, 3, 5, 5)
        pooled, index = nnef_max_pool_with_index(input, size=[1, 1, 2, 2])
        unpooled = nnef_desample(pooled, index, size=[1, 1, 2, 2], output_shape=[1, 3, 5, 5])
        pooled2, index2 = nnef_max_pool_with_index(unpooled, size=[1, 1, 2, 2])
        self.assertEqual(unpooled.shape, input.shape)
        self.assertTrue(np.all(np.equal(pooled2.numpy(), pooled.numpy())))
        self.assertTrue(np.all(np.equal(index2.numpy(), index2.numpy())))

    def test_deconv(self):
        input = torch.rand(1, 3, 5, 5)
        filter = torch.rand(6, 3, 3, 3)
        bias = torch.zeros(tuple())
        conv = nnef_conv(input, filter, bias)
        deconv = nnef_deconv(conv, filter, bias, output_shape=list(input.shape))
        self.assertEqual(deconv.shape, input.shape)

    def test_deconv2(self):
        input = torch.rand(1, 3, 5, 5)
        filter = torch.rand(6, 3, 3, 3)
        bias = torch.zeros(tuple())
        conv = nnef_conv(input, filter, bias)
        deconv = nnef_deconv(conv, filter, bias)
        self.assertEqual(deconv.shape, input.shape)

    def test_deconv3(self):
        input = torch.rand(1, 3, 5, 5)
        filter = torch.rand(6, 3, 3, 3)
        bias = torch.zeros(tuple())
        padding = [(0, 1), (2, 1)]
        conv = nnef_conv(input, filter, bias, padding=padding)
        deconv = nnef_deconv(conv, filter, bias, padding=padding)
        self.assertEqual(deconv.shape, input.shape)

    def test_deconv4(self):
        input = torch.rand(1, 3, 10, 10)
        filter = torch.rand(6, 3, 3, 3)
        bias = torch.zeros(tuple())
        padding = [(0, 0), (0, 0)]
        stride = [1, 3]
        conv = nnef_conv(input, filter, bias, padding=padding, stride=stride)
        deconv = nnef_deconv(conv, filter, bias, padding=padding, stride=stride)
        self.assertNotEqual(deconv.shape, input.shape)
        deconv = nnef_deconv(conv, filter, bias, padding=padding, stride=stride, output_shape=list(input.shape))
        self.assertEqual(deconv.shape, input.shape)

    def test_deconv5(self):
        input = torch.rand(2, 3, 5, 5)
        filter = torch.rand(6, 3, 3, 3)
        bias = torch.zeros(tuple())
        conv = nnef_conv(input, filter, bias)
        deconv = nnef_deconv(conv, filter, bias, output_shape=[1, 3, 5, 5])
        self.assertEqual(deconv.shape, input.shape)

    def test_box(self):
        input = torch.rand(5, 5, 5, 5, 1, 1, 1)
        output = nnef_box(input, size=[2, 1, 2, 1, 1, 1, 1])
        self.assertEqual(output.shape, input.shape)

    def test_box2(self):
        input = torch.rand(5, 5, 5, 5, 1, 1, 1)
        output = nnef_box(input, size=[2, 1, 2, 1, 1, 1, 1], stride=[2, 1, 2, 2, 1, 1, 1])
        self.assertEqual([3, 5, 3, 3, 1, 1, 1], list(output.shape))

    def test_box3(self):
        input = torch.rand(5)
        output = nnef_box(input, size=[3])
        self.assertEqual(input.shape, output.shape)

    def test_box4(self):
        input = torch.rand(1, 3, 5, 5)
        output = nnef_box(input, size=[1, 1, 2, 2])
        self.assertEqual(input.shape, output.shape)

    def test_box5(self):
        input = torch.rand(1, 3, 5, 5)
        output = nnef_box(input, size=[1, 1, 1, 1], padding=[(0, 0), (0, 0), (10, 11), (10, 11)])
        self.assertEqual([1, 3, 26, 26], list(output.shape))

    def test_max_pool(self):
        input = torch.rand(5, 5, 5, 5, 1, 1, 1)
        output = nnef_max_pool(input, size=[2, 1, 2, 1, 1, 1, 1])
        self.assertEqual(output.shape, input.shape)

    def test_max_pool2(self):
        input = torch.rand(5, 5, 5, 5, 1, 1, 1)
        output = nnef_max_pool(input, size=[2, 1, 2, 1, 1, 1, 1], stride=[2, 1, 2, 2, 1, 1, 1])
        self.assertEqual([3, 5, 3, 3, 1, 1, 1], list(output.shape))

    def test_max_pool3(self):
        input = torch.rand(5)
        output = nnef_max_pool(input, size=[3])
        self.assertEqual(input.shape, output.shape)

    def test_max_pool4(self):
        input = torch.rand(1, 3, 5, 5)
        output = nnef_max_pool(input, size=[1, 1, 2, 2])
        self.assertEqual(input.shape, output.shape)

    def test_reshape(self):
        input = torch.rand(2, 3, 5, 5)
        output = nnef_reshape(input, shape=[0, 3, 25])
        self.assertEqual([2, 3, 25], list(output.shape))

    def test_reshape_2(self):
        input = torch.rand(2, 3, 5, 5)
        output = nnef_reshape(input, shape=[25], axis_start=2, axis_count=2)
        self.assertEqual([2, 3, 25], list(output.shape))

    def test_debox(self):
        input = torch.full(size=(1, 3, 6, 6), fill_value=3.14)
        box = nnef_box(input, size=[1, 1, 3, 3], padding=[(0, 0), (0, 0), (0, 0), (0, 0)], normalize=False)
        debox = nnef_debox(box, size=[1, 1, 3, 3], padding=[(0, 0), (0, 0), (0, 0), (0, 0)], normalize=False)
        self.assertEqual(input.shape, debox.shape)

    def test_debox2(self):
        input = torch.full(size=(1, 3, 6, 6), fill_value=3.14)
        box = nnef_box(input, size=[1, 1, 3, 3], stride=[1, 1, 2, 2], normalize=True)
        debox = nnef_debox(box, size=[1, 1, 3, 3], stride=[1, 1, 2, 2], normalize=True, output_shape=list(input.shape))
        self.assertEqual(input.shape, debox.shape)

    def test_avg_pool(self):
        input = torch.rand(1, 3, 5, 5)
        output = nnef_avg_pool(input, size=[1, 1, 2, 2], border='ignore')
        self.assertEqual(input.shape, output.shape)

    def test_upsample(self):
        input = torch.from_numpy(np.array([[[[1.0, 2.0],
                                             [1.0, 2.0]]]], dtype=np.float32))
        output = nnef_multilinear_upsample(input, factor=[2, 2], method='symmetric', border='constant')
        self.assertTrue(np.allclose([[[[0.5625, 0.9375, 1.3125, 1.1250],
                                       [0.7500, 1.2500, 1.7500, 1.5000],
                                       [0.7500, 1.2500, 1.7500, 1.5000],
                                       [0.5625, 0.9375, 1.3125, 1.1250]]]], output, rtol=0, atol=0.001))
        output = nnef_multilinear_upsample(input, factor=[2, 2], method='asymmetric', border='constant')
        self.assertTrue(np.allclose([[[[1.0000, 1.5000, 2.0000, 1.0000],
                                       [1.0000, 1.5000, 2.0000, 1.0000],
                                       [1.0000, 1.5000, 2.0000, 1.0000],
                                       [0.5000, 0.7500, 1.0000, 0.5000]]]], output, rtol=0, atol=0.001))
        output = nnef_multilinear_upsample(input, factor=[2, 2], method='symmetric', border='replicate')
        self.assertTrue(np.allclose([[[[1.0000, 1.2500, 1.7500, 2.0000],
                                       [1.0000, 1.2500, 1.7500, 2.0000],
                                       [1.0000, 1.2500, 1.7500, 2.0000],
                                       [1.0000, 1.2500, 1.7500, 2.0000]]]], output, rtol=0, atol=0.001))
        output = nnef_multilinear_upsample(input, factor=[2, 2], method='asymmetric', border='replicate')
        self.assertTrue(np.allclose([[[[1.0000, 1.5000, 2.0000, 2.0000],
                                       [1.0000, 1.5000, 2.0000, 2.0000],
                                       [1.0000, 1.5000, 2.0000, 2.0000],
                                       [1.0000, 1.5000, 2.0000, 2.0000]]]], output, rtol=0, atol=0.001))
        output = nnef_multilinear_upsample(input, factor=[2, 2], method='aligned', border='(anything)')
        self.assertTrue(np.allclose([[[[1.0000, 1.3333, 1.6667, 2.0000],
                                       [1.0000, 1.3333, 1.6667, 2.0000],
                                       [1.0000, 1.3333, 1.6667, 2.0000],
                                       [1.0000, 1.3333, 1.6667, 2.0000]]]], output, rtol=0, atol=0.001))

        input = torch.from_numpy(np.array([[[1.0, 2.0]]], dtype=np.float32))
        output = nnef_multilinear_upsample(input, factor=[2], method='symmetric', border='constant')
        self.assertTrue(np.allclose([[[0.7500, 1.2500, 1.7500, 1.5000]]], output, rtol=0, atol=0.001))
        output = nnef_multilinear_upsample(input, factor=[2], method='asymmetric', border='constant')
        self.assertTrue(np.allclose([[[1.0000, 1.5000, 2.0000, 1.0000]]], output, rtol=0, atol=0.001))
        output = nnef_multilinear_upsample(input, factor=[2], method='symmetric', border='replicate')
        self.assertTrue(np.allclose([[[1.0000, 1.2500, 1.7500, 2.0000]]], output, rtol=0, atol=0.001))
        output = nnef_multilinear_upsample(input, factor=[2], method='asymmetric', border='replicate')
        self.assertTrue(np.allclose([[[1.0000, 1.5000, 2.0000, 2.0000]]], output, rtol=0, atol=0.001))
        output = nnef_multilinear_upsample(input, factor=[2], method='aligned', border='(anything)')
        self.assertTrue(np.allclose([[[1.0000, 1.3333, 1.6667, 2.0000]]], output, rtol=0, atol=0.001))

    def test_max_pool_1x1_stride_2x2(self):
        input = torch.from_numpy(np.array([[[[1.0, 2.0],
                                             [3.0, 4.0]]]], dtype=np.float32))
        output = nnef_max_pool(input, size=[1, 1, 1, 1], stride=[1, 1, 2, 2])
        self.assertTrue(np.allclose([[[[1.]]]], output, rtol=0, atol=0.001))
