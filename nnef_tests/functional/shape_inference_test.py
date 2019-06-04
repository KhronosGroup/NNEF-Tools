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

from nnef_tools.shape_inference import shape_inference as infer


class TestShapeInference(unittest.TestCase):
    def test_singleton(self):
        self.assertEqual(infer.singleton(0), [])
        self.assertEqual(infer.singleton(1), [1])
        self.assertEqual(infer.singleton(4), [1, 1, 1, 1])

    def test_copy(self):
        a = [1, 2, 3]
        b = infer.copy(a)
        self.assertEqual(a, b)
        self.assertIsNot(a, b)

    def test_elementwise(self):
        a = [1, 2, 3]
        b = [1, 2, 3]
        c = infer.elementwise([a, b], infer.Broadcast.NONE)
        self.assertEqual(c, a)
        self.assertIsNot(c, a)
        self.assertIsNot(c, b)

        a = [1, 2, 3]
        b = [3]
        ab = [a, b]
        c = infer.elementwise([a, b], infer.Broadcast.FROM_RIGHT)
        self.assertEqual([1, 2, 3], c)
        self.assertEqual([1, 2, 3], a)
        self.assertEqual([3], b)
        self.assertEqual([a, b], ab)

        with self.assertRaises(AssertionError):
            infer.elementwise([[1, 2], [1]], infer.Broadcast.NONE)
        with self.assertRaises(AssertionError):
            infer.elementwise([[1, 2], [1, 3]], infer.Broadcast.NONE)

        self.assertEqual([2, 4, 3], infer.elementwise([[1, 4, 1], [2, 1, 3]], infer.Broadcast.SAME_RANK))
        with self.assertRaises(AssertionError):
            infer.elementwise([[1, 2], [1]], infer.Broadcast.SAME_RANK)

        self.assertEqual([4, 2, 3], infer.elementwise([[1, 2, 3], [4, 2]], infer.Broadcast.FROM_LEFT))
        with self.assertRaises(AssertionError):
            infer.elementwise([[2, 3], [3]], infer.Broadcast.FROM_LEFT)

        self.assertEqual([1, 2, 3], infer.elementwise([[1, 1, 3], [2, 3]], infer.Broadcast.FROM_RIGHT))
        with self.assertRaises(AssertionError):
            infer.elementwise([[2, 3], [2]], infer.Broadcast.FROM_RIGHT)

    def test_sliding_window(self):
        with self.assertRaises(AssertionError):
            infer.sliding_window(input=[1], filter=[], padding=[(1, 1)], stride=[1], dilation=[1])

        with self.assertRaises(AssertionError):
            infer.sliding_window(input=[1], filter=[1], padding=[], stride=[1], dilation=[1])

        with self.assertRaises(AssertionError):
            infer.sliding_window(input=[1], filter=[1], padding=[(1, 1)], stride=[], dilation=[1])

        with self.assertRaises(AssertionError):
            infer.sliding_window(input=[1], filter=[1], padding=[(1, 1)], stride=[1], dilation=[])

        with self.assertRaises(AssertionError):
            infer.sliding_window(input=[], filter=[1], padding=[(1, 1)], stride=[1], dilation=[1])

        self.assertEqual([10, 30, 30, 3], infer.sliding_window(input=[10, 32, 32, 3],
                                                               filter=[1, 3, 3, 1],
                                                               padding=[(0, 0)] * 4,
                                                               stride=[1, 1, 1, 1],
                                                               dilation=[1, 1, 1, 1]))

        self.assertEqual([10, 32, 32, 3], infer.sliding_window(input=[10, 30, 30, 3],
                                                               filter=[1, 3, 3, 1],
                                                               padding=[(0, 0)] * 4,
                                                               stride=[1, 1, 1, 1],
                                                               dilation=[1, 1, 1, 1],
                                                               upscale=True))

        self.assertEqual([10, 28, 26, 3], infer.sliding_window(input=[10, 32, 32, 3],
                                                               filter=[1, 3, 3, 1],
                                                               padding=[(0, 0)] * 4,
                                                               stride=[1, 1, 1, 1],
                                                               dilation=[1, 2, 3, 1]))

        self.assertEqual([10, 15, 32, 3], infer.sliding_window(input=[10, 32, 32, 3],
                                                               filter=[1, 3, 1, 1],
                                                               padding=[(0, 0)] * 4,
                                                               stride=[1, 2, 1, 1],
                                                               dilation=[1, 1, 1, 1]))

        self.assertEqual([10, 16, 32, 3], infer.sliding_window(input=[10, 32, 32, 3],
                                                               filter=[1, 3, 1, 1],
                                                               padding=[(0, 0)] * 4,
                                                               stride=[1, 2, 1, 1],
                                                               dilation=[1, 1, 1, 1],
                                                               ceil=True))

    def test_valid_padding(self):
        self.assertEqual([], infer.valid_padding(0))
        self.assertEqual([(0, 0)], infer.valid_padding(1))
        self.assertEqual([(0, 0), (0, 0), (0, 0), (0, 0)], infer.valid_padding(4))

    def test_same_padding(self):
        self.assertEqual([(0, 1)], infer.same_padding(upscaled_input=[32], filter=[3], stride=[2], dilation=[1]))

        self.assertEqual([(1, 0)], infer.same_padding(upscaled_input=[32],
                                                      filter=[3],
                                                      stride=[2],
                                                      dilation=[1],
                                                      left_bigger=True))

        self.assertEqual([(2, 3)], infer.same_padding(upscaled_input=[32], filter=[3], stride=[2], dilation=[3]))

    def test_concat(self):
        self.assertEqual([3, 2, 3, 4], infer.concat([[1, 2, 3, 4], [2, 2, 3, 4]], 0))
        self.assertEqual([3, 2, 3, 4], infer.concat([[1, 2, 3, 4], [2, 2, 3, 4]], -4))
        self.assertEqual([1, 12, 3, 4], infer.concat([[1, 2, 3, 4], [1, 10, 3, 4]], 1))
        self.assertEqual([1, 2, 3, 9], infer.concat([[1, 2, 3, 4], [1, 2, 3, 5]], -1))

    def test_split(self):
        self.assertEqual([[1, 2, 2], [1, 2, 2], [1, 2, 2]], infer.split([1, 2, 6], -1, num=3))
        self.assertEqual([[1, 2, 2], [1, 2, 4]], infer.split([1, 2, 6], -1, ratios=[1, 2]))
        self.assertEqual([[1, 2, 2], [1, 2, 4]], infer.split([1, 2, 6], -1, sizes=[2, 4]))
        self.assertEqual([[1, 2, 2], [1, 2, 4]], infer.split([1, 2, 6], -1, split_points=[2]))
        self.assertEqual([[1, 2, 2], [1, 2, 1], [1, 2, 2], [1, 2, 1]],
                         infer.split([1, 2, 6], -1, split_points=[2, 3, 5]))

    def test_conv(self):
        self.assertEqual([10, 30, 30, 16], infer.conv(input=[10, 32, 32, 3],
                                                      filter=[3, 3],
                                                      padding=infer.Padding.VALID,
                                                      stride=[1, 1],
                                                      dilation=[1, 1],
                                                      groups=1,
                                                      spatial_begin=infer.spatial_begin(infer.Format.NHWC),
                                                      channel_axis=infer.channel_axis(infer.Format.NHWC),
                                                      output_channels=16))

        self.assertEqual([10, 16, 30, 30], infer.conv(input=[10, 3, 32, 32],
                                                      filter=[3, 3],
                                                      padding=[(0, 0), (0, 0)],
                                                      stride=[1, 1],
                                                      dilation=[1, 1],
                                                      groups=1,
                                                      spatial_begin=infer.spatial_begin(infer.Format.NCHW),
                                                      channel_axis=infer.channel_axis(infer.Format.NCHW),
                                                      output_channels=16))

        self.assertEqual([10, 3, 32, 32], infer.conv(input=[10, 16, 30, 30],
                                                     filter=[3, 3],
                                                     padding=[(0, 0), (0, 0)],
                                                     stride=[1, 1],
                                                     dilation=[1, 1],
                                                     groups=1,
                                                     format=infer.Format.NCHW,
                                                     output_channels=3,
                                                     deconv=True))

        self.assertEqual([10, 3, 32, 32], infer.conv(input=[10, 16, 32, 32],
                                                     filter=[3, 3],
                                                     padding=infer.Padding.SAME_UPPER,
                                                     stride=[1, 1],
                                                     dilation=[1, 1],
                                                     groups=1,
                                                     format=infer.Format.NCHW,
                                                     output_channels=3))

        self.assertEqual([10, 6, 32, 32], infer.conv(input=[10, 3, 32, 32],
                                                     filter=[3, 3],
                                                     padding=infer.Padding.SAME_LOWER,
                                                     stride=[1, 1],
                                                     dilation=[1, 1],
                                                     groups=0,
                                                     format=infer.Format.NCHW,
                                                     output_channels=6))

        self.assertEqual([10, 16, 32, 32], infer.conv(input=[10, 3, 32, 32],
                                                      filter=[3, 3],
                                                      padding=infer.Padding.SAME_UPPER,
                                                      stride=[1, 1],
                                                      dilation=[1, 1],
                                                      groups=1,
                                                      format=infer.Format.NCHW,
                                                      output_channels=16,
                                                      deconv=True))

        self.assertEqual([10, 16, 64, 64], infer.conv(input=[10, 3, 32, 32],
                                                      filter=[3, 3],
                                                      padding=infer.Padding.SAME_UPPER,
                                                      stride=[2, 2],
                                                      dilation=[1, 1],
                                                      groups=1,
                                                      format=infer.Format.NCHW,
                                                      output_channels=16,
                                                      deconv=True))

        self.assertEqual([10, 16, 65, 65], infer.conv(input=[10, 3, 32, 32],
                                                      filter=[3, 3],
                                                      padding=infer.Padding.SAME_UPPER,
                                                      stride=[2, 2],
                                                      dilation=[1, 1],
                                                      groups=1,
                                                      format=infer.Format.NCHW,
                                                      output_channels=16,
                                                      output_padding=[(0, 1), (0, 1)],
                                                      deconv=True))

    def test_squeeze(self):
        self.assertEqual([3, 2], infer.squeeze([3, 2, 1], [-1]))
        self.assertEqual([2, 3], infer.squeeze([1, 2, 3], [0]))
        self.assertEqual([2, 2], infer.squeeze([2, 1, 2, 1], [1, -1]))
        self.assertEqual([], infer.squeeze([1, 1], [0, -1]))

        self.assertEqual([3, 2], infer.squeeze([1, 3, 1, 1, 2, 1, 1]))

    def test_unsqueeze(self):
        self.assertEqual([1, 2, 3], infer.unsqueeze([2, 3], [0]))
        self.assertEqual([2, 3, 1], infer.unsqueeze([2, 3], [-1]))
        self.assertEqual([2, 1, 3], infer.unsqueeze([2, 3], [-2]))
        self.assertEqual([1, 2, 3], infer.unsqueeze([2, 3], [-3]))
        self.assertEqual([1, 2, 1], infer.unsqueeze([1, 2], [2]))
        self.assertEqual([2, 1, 3], infer.unsqueeze([2, 3], [1]))
        self.assertEqual([1, 1, 1, 2, 1, 1], infer.unsqueeze([1, 2], [0, 2, 4, 5]))
        self.assertEqual([1], infer.unsqueeze([], [0]))

    def test_matmul(self):
        self.assertEqual([3, 5], infer.matmul([3, 4], [4, 5]))
        self.assertEqual([3, 5], infer.matmul([4, 3], [4, 5], transpose_a=True))
        self.assertEqual([3, 5], infer.matmul([3, 4], [5, 4], transpose_b=True))
        self.assertEqual([10, 20, 3, 5], infer.matmul([10, 20, 4, 3],
                                                      [10, 20, 5, 4],
                                                      transpose_a=True,
                                                      transpose_b=True))
        self.assertEqual([10, 20, 3, 5], infer.matmul([10, 20, 4, 3],
                                                      [1, 20, 5, 4],
                                                      transpose_a=True,
                                                      transpose_b=True))
        self.assertEqual([10, 20, 3, 5], infer.matmul([10, 20, 4, 3],
                                                      [1, 1, 5, 4],
                                                      transpose_a=True,
                                                      transpose_b=True))

    def test_reduce(self):
        self.assertEqual([1, 1, 1, 1, 5], infer.reduce([1, 2, 3, 4, 5], [1, 2, 3]))
        self.assertEqual([1, 5], infer.reduce([1, 2, 3, 4, 5], [1, 2, 3], squeeze=True))
        self.assertEqual([], infer.reduce([5], [-1], squeeze=True))
        self.assertEqual([1, 6], infer.reduce([5, 6], [-2]))

    def test_stack(self):
        self.assertEqual([1, 2, 3], infer.stack([[1, 2], [1, 2], [1, 2]], 2))
        self.assertEqual([1, 2, 3], infer.stack([[1, 2], [1, 2], [1, 2]], -1))
        self.assertEqual([3, 1, 2], infer.stack([[1, 2], [1, 2], [1, 2]], 0))
        self.assertEqual([1, 3, 2], infer.stack([[1, 2], [1, 2], [1, 2]], 1))

    def test_unstack(self):
        self.assertEqual([[1, 2], [1, 2], [1, 2]], infer.unstack([1, 2, 3], 2))
        self.assertEqual([[1, 2], [1, 2], [1, 2]], infer.unstack([1, 2, 3], -1))
        self.assertEqual([[2, 3]], infer.unstack([1, 2, 3], 0))
        self.assertEqual([[1, 3], [1, 3]], infer.unstack([1, 2, 3], 1))
        s1, s2 = infer.unstack([1, 2, 3], 1)
        self.assertIsNot(s1, s2)

    def test_pad(self):
        self.assertEqual([2, 7, 12], infer.pad([1, 2, 3], [(0, 1), (2, 3), (4, 5)]))

    def test_reshape(self):
        self.assertEqual([4, 6], infer.reshape([4, 2, 3], [4, 6]))
        self.assertEqual([4, 6], infer.reshape([4, 2, 3], [4, -1]))
        self.assertEqual([24], infer.reshape([4, 2, 3], [24]))
        self.assertEqual([24], infer.reshape([4, 2, 3], [-1]))
        self.assertEqual([4, 2, 1, 3], infer.reshape([4, 2, 3], [4, 2, -1, 3]))
        self.assertEqual([4, 6, 1], infer.reshape([4, 2, 3], [4, -1, 1]))
        self.assertEqual([4, 6, 1], infer.reshape([4, 2, 3], [4, -1, 1]))
        self.assertEqual([1], infer.reshape([], [1]))
        self.assertEqual([1], infer.reshape([], [-1]))
        self.assertEqual([], infer.reshape([1], []))
        self.assertEqual([], infer.reshape([1, 1, 1], []))
        self.assertEqual([0, 1], infer.reshape([0], [0, 1]))
        self.assertEqual([1, 2, 3, 4], infer.reshape(input=[1, 3, 2, 4], shape=[2, 3], offset=1, count=2))
        self.assertEqual([1, 2, 3, 4], infer.reshape(input=[1, 24], shape=[2, 3, 4], offset=1, count=-1))
        with self.assertRaises(AssertionError):
            infer.reshape([0], [0, -1])
        self.assertEqual([1, 2, 1, 3], infer.reshape([1, 2, 3], [0, 0, 1, -1], zero_means_same=True))
        self.assertEqual([1], infer.reshape([1], [0], zero_means_same=True))
        with self.assertRaises(AssertionError):
            infer.reshape([], [0], zero_means_same=True)
        with self.assertRaises(AssertionError):
            infer.reshape([1], [1, 0], zero_means_same=True)

    def test_flatten(self):
        self.assertEqual([1, 1], infer.flatten([]))
        self.assertEqual([2, 1], infer.flatten([2]))
        self.assertEqual([2, 3], infer.flatten([2, 3]))
        self.assertEqual([2, 12], infer.flatten([2, 3, 4]))
        self.assertEqual([0, 1], infer.flatten([0]))
        self.assertEqual([0, 3], infer.flatten([0, 3]))
        self.assertEqual([0, 12], infer.flatten([0, 3, 4]))
        self.assertEqual([0, 0], infer.flatten([0, 3, 4, 0]))
        self.assertEqual([1, 0], infer.flatten([1, 3, 4, 0]))

    def test_resize(self):
        self.assertEqual([10, 16, 64, 3], infer.resize([10, 32, 32, 3], [16, 64],
                                                       spatial_begin=infer.spatial_begin(infer.Format.NHWC)))
        self.assertEqual([10, 16, 64], infer.resize([10, 32, 32], [16, 64], format=infer.Format.NHWC))
        self.assertEqual([10, 3, 16, 64], infer.resize([10, 3, 32, 32], [16, 64], format=infer.Format.NCHW))

    def test_downsample(self):
        self.assertEqual([10, 16, 16, 3],
                         infer.downsample([10, 32, 32, 3], [2, 2],
                                          spatial_begin=infer.spatial_begin(infer.Format.NHWC)))
        self.assertEqual([10, 16, 8], infer.downsample([10, 32, 32], [2, 4], format=infer.Format.NHWC))
        self.assertEqual([10, 3, 8, 16], infer.downsample([10, 3, 32, 32], [4, 2], format=infer.Format.NCHW))

    def test_upsample(self):
        self.assertEqual([10, 64, 64, 3], infer.upsample([10, 32, 32, 3], [2, 2],
                                                         spatial_begin=infer.spatial_begin(infer.Format.NHWC)))
        self.assertEqual([10, 64, 128], infer.upsample([10, 32, 32], [2, 4], format=infer.Format.NHWC))
        self.assertEqual([10, 3, 128, 64], infer.upsample([10, 3, 32, 32], [4, 2], format=infer.Format.NCHW))

    def test_transpose(self):
        self.assertEqual([], infer.transpose([]))
        self.assertEqual([1, 2, 3], infer.transpose([3, 2, 1]))
        self.assertEqual([10, 3, 32, 16], infer.transpose([10, 32, 16, 3], [0, 3, 1, 2]))

    def test_slice(self):
        self.assertEqual([1, 1, 1, 2], infer.slice(input=[1, 2, 3, 4], begin=[0, 1, 2, 2], size=[1, 1, 1, 2]))
        self.assertEqual([1, 1, 1, 2], infer.slice(input=[1, 2, 3, 4], begin=[0, 1, 2, 2], size=[-1, -1, -1, -1]))
        self.assertEqual([1, 1, 1, 2], infer.slice(input=[1, 2, 3, 4],
                                                   begin=[0, 1, 2, 2],
                                                   size=[0, 0, 0, 0],
                                                   zero_means_all=True))
        self.assertEqual([0, 0, 0, 0], infer.slice([1, 2, 3, 4], begin=[0, 1, 2, 2], size=[0, 0, 0, 0]))
        self.assertEqual([2, 4, 6, 36], infer.slice(input=[10, 20, 30, 40], begin=[1, 2, 3, 4], size=[2, 4, 6, -1]))

        self.assertEqual([1, 1, 1, 2], infer.slice(input=[1, 2, 3, 4], begin=[0, 1, 2, 2], end=[1, 2, 3, 4]))
        self.assertEqual([1, 1, 1, 2], infer.slice(input=[1, 2, 3, 4],
                                                   begin=[0, 1, 2, 2],
                                                   end=[0, 0, 0, 0],
                                                   zero_means_all=True))
        self.assertEqual([0, 0, 0, 0], infer.slice([1, 2, 3, 4], begin=[0, 1, 2, 2], end=[0, 1, 2, 2]))
        self.assertEqual([2, 4, 6, 36], infer.slice(input=[10, 20, 30, 40],
                                                    begin=[1, 2, 3, 4],
                                                    end=[3, 6, 9, 0],
                                                    zero_means_all=True))
        self.assertEqual([10, 32, 32, 1], infer.slice(input=[10, 32, 32, 3], axes=[3], begin=[1], end=[2]))
        self.assertEqual([10, 32, 32, 0], infer.slice(input=[10, 32, 32, 3], axes=[3], begin=[1], end=[1]))
        self.assertEqual([10, 32, 32, 2], infer.slice(input=[10, 32, 32, 3],
                                                      axes=[3],
                                                      begin=[1],
                                                      end=[0],
                                                      zero_means_all=True))
        self.assertEqual([10, 32, 32, 1], infer.slice(input=[10, 32, 32, 3], axes=[3], begin=[1], size=[1]))
        self.assertEqual([10, 32, 32, 2], infer.slice(input=[10, 32, 32, 3], axes=[3], begin=[1], size=[-1]))
        self.assertEqual([10, 32, 32, 0], infer.slice(input=[10, 32, 32, 3], axes=[-1], begin=[1], size=[0]))
        self.assertEqual([10, 32, 32, 2], infer.slice(input=[10, 32, 32, 3],
                                                      axes=[-1],
                                                      begin=[1],
                                                      size=[0],
                                                      zero_means_all=True))

        self.assertEqual([1, 2, 1, 2], infer.slice(input=[10, 32, 32, 3],
                                                   axes=[-1, 2, -3, 0],
                                                   begin=[1, 2, 3, 0],
                                                   end=[3, 3, 5, 1]))

        self.assertEqual([1, 2, 1, 2], infer.slice(input=[10, 32, 32, 3],
                                                   axes=[-1, 2, -3, 0],
                                                   begin=[1, 2, 3, 0],
                                                   size=[-1, 1, 2, 1]))

        self.assertEqual([10, 5, 3, 2], infer.slice(input=[10, 20, 30, 40],
                                                    begin=[0, 0, 0, 0],
                                                    size=[10, 10, 10, 10],
                                                    stride=[1, 2, 3, 4]))

        self.assertEqual([1, 14, 25, 35], infer.slice(input=[10, 20, 30, 40],
                                                      begin=[5, 5, 5, 5],
                                                      end=[6, -1, 30, 999]))

    def test_bit_mask_to_array(self):
        self.assertEqual([0, 1, 1, 0], infer.bit_mask_to_array(6, 4))
        self.assertEqual([1, 1, 1, 0, 0], infer.bit_mask_to_array(7, 5))
        self.assertEqual([1], infer.bit_mask_to_array(1, 1))
        self.assertEqual([0], infer.bit_mask_to_array(0, 1))
        self.assertEqual([], infer.bit_mask_to_array(0, 0))

    def test_decompose_strided_slice(self):
        anything = 0xdeadbeef
        ssl_begin, ssl_end, ssl_stride, ssl_shape, reshape_shape = \
            infer.decompose_strided_slice(input=[10, 32, 32, 3],
                                          begin=[1, anything, 2, 2, 0],
                                          end=[3, anything, 5, 3, 3],
                                          stride=[1, anything, 1, 1, 1],
                                          new_axis_mask=[0, 1, 0, 0, 0],
                                          shrink_axis_mask=[0, 0, 0, 1, 0],
                                          begin_mask=[0, 0, 0, 0, 0],
                                          end_mask=[0, 0, 0, 0, 0],
                                          ellipsis_mask=[0, 0, 0, 0, 0])
        self.assertEqual([1, 2, 2, 0], ssl_begin)
        self.assertEqual([3, 5, 3, 3], ssl_end)
        self.assertEqual([1, 1, 1, 1], ssl_stride)
        self.assertEqual([2, 3, 1, 3], ssl_shape)
        self.assertEqual([2, 1, 3, 3], reshape_shape)

        ssl_begin, ssl_end, ssl_stride, ssl_shape, reshape_shape = \
            infer.decompose_strided_slice(input=[10, 32, 32, 3],
                                          begin=[anything, anything, anything, anything],
                                          end=[anything, anything, anything, anything],
                                          stride=[1, -1, 3, -2],
                                          new_axis_mask=[0, 0, 0, 0],
                                          shrink_axis_mask=[0, 0, 0, 0],
                                          begin_mask=[1, 1, 1, 1],
                                          end_mask=[1, 1, 1, 1],
                                          ellipsis_mask=[0, 0, 0, 0])
        self.assertEqual([0, 0, 0, 0], ssl_begin)
        self.assertEqual([10, 32, 32, 3], ssl_end)
        self.assertEqual([1, -1, 3, -2], ssl_stride)
        self.assertEqual([10, 32, 10, 1], ssl_shape)
        self.assertEqual([10, 32, 10, 1], reshape_shape)
        self.assertIsNot(ssl_shape, reshape_shape)

        decomposed = infer.decompose_strided_slice(input=[10, 32, 32, 3],
                                                   begin=[0, anything, 0],
                                                   end=[1, anything, 1],
                                                   stride=[1, anything, 1],
                                                   new_axis_mask=[0, 0, 0],
                                                   shrink_axis_mask=[0, 0, 0],
                                                   begin_mask=[0, 0, 0],
                                                   end_mask=[0, 0, 0],
                                                   ellipsis_mask=[0, 1, 0])
        self.assertEqual([0, 0, 0, 0], decomposed.ssl_begin)
        self.assertEqual([1, 32, 32, 1], decomposed.ssl_end)
        self.assertEqual([1, 1, 1, 1], decomposed.ssl_stride)
        self.assertEqual([1, 32, 32, 1], decomposed.ssl_shape)
        self.assertEqual([1, 32, 32, 1], decomposed.reshape_shape)
        self.assertIsNot(decomposed.ssl_shape, decomposed.reshape_shape)

    def test_strided_slice(self):
        anything = 0xcafecafe
        self.assertEqual([2, 1, 3, 3], infer.strided_slice(input=[10, 32, 32, 3],
                                                           begin=[1, anything, 2, 2, 0],
                                                           end=[3, anything, 5, 3, 3],
                                                           stride=[1, anything, 1, 1, 1],
                                                           new_axis_mask=[0, 1, 0, 0, 0],
                                                           shrink_axis_mask=[0, 0, 0, 1, 0],
                                                           begin_mask=[0, 0, 0, 0, 0],
                                                           end_mask=[0, 0, 0, 0, 0],
                                                           ellipsis_mask=[0, 0, 0, 0, 0]))
        self.assertEqual([10, 32, 10, 1], infer.strided_slice(input=[10, 32, 32, 3],
                                                              begin=[anything, anything, anything, anything],
                                                              end=[anything, anything, anything, anything],
                                                              stride=[1, -1, 3, -2],
                                                              new_axis_mask=[0, 0, 0, 0],
                                                              shrink_axis_mask=[0, 0, 0, 0],
                                                              begin_mask=[1, 1, 1, 1],
                                                              end_mask=[1, 1, 1, 1],
                                                              ellipsis_mask=[0, 0, 0, 0]))

        self.assertEqual([1, 32, 32, 1], infer.strided_slice(input=[10, 32, 32, 3],
                                                             begin=[0, anything, 0],
                                                             end=[1, anything, 1],
                                                             stride=[1, anything, 1],
                                                             new_axis_mask=[0, 0, 0],
                                                             shrink_axis_mask=[0, 0, 0],
                                                             begin_mask=[0, 0, 0],
                                                             end_mask=[0, 0, 0],
                                                             ellipsis_mask=[0, 1, 0]))

        self.assertEqual([1, 32, 32, 1], infer.strided_slice(input=[10, 32, 32, 3],
                                                             begin=[0, anything, 0],
                                                             end=[1, anything, 1],
                                                             stride=[1, anything, 1],
                                                             new_axis_mask=0,
                                                             shrink_axis_mask=0,
                                                             begin_mask=0,
                                                             end_mask=0,
                                                             ellipsis_mask=2))

    def test_get_deconv_output_padding(self):
        self.assertEqual([(0, 0), (0, 0)], infer.get_deconv_output_padding(output=[1, 6, 15, 15],
                                                                           input=[1, 3, 15, 15],
                                                                           filter=[2, 2],
                                                                           padding=[(0, 1), (1, 0)],
                                                                           stride=[1, 1],
                                                                           dilation=[1, 1],
                                                                           groups=1,
                                                                           format=infer.Format.NCHW))

        self.assertEqual([(0, 3), (0, 2)], infer.get_deconv_output_padding(output=[1, 6, 64, 63],
                                                                           input=[1, 3, 15, 15],
                                                                           filter=[6, 6],
                                                                           padding=[(0, 1), (1, 0)],
                                                                           stride=[4, 4],
                                                                           dilation=[1, 1],
                                                                           groups=1,
                                                                           format=infer.Format.NCHW))

    def test_tile(self):
        self.assertEqual([4, 6, 6, 4], infer.tile(input=[1, 2, 3, 4], repeat=[4, 3, 2, 1]))


if __name__ == '__main__':
    unittest.main()
