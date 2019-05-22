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

import math
import os
import unittest

import numpy as np

from nnef_tools.core import utils
from nnef_tests.conversion.caffe_test_runner import CaffeTestRunner

with utils.EnvVars(GLOG_minloglevel=3):
    import caffe
    from caffe import layers as L
    from caffe import params as P

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


def make_shape(list_):
    return dict(dim=list_)


def make_weight_filler():
    return dict(type='uniform', min=-0.1, max=0.1)


def make_bias_filler():
    return dict(type='uniform', min=-0.05, max=0.05)


def set_random_weights(net):
    # type: (caffe.Net) -> None

    for name, variables in net.params.items():
        for variable in variables:
            if variable.data.size == 1:
                variable.data[...] = 1.0
            else:
                variable.data[...] = 0.1 + 0.1 * np.random.random(list(variable.data.shape))


class CaffeLayerTestCases(CaffeTestRunner):
    def _netspec_to_model(self, netspec, name, random_weights=True):
        caffemodel_path = os.path.join('out', 'caffe-orig', name, name + '.caffemodel')
        prototxt_path = os.path.join('out', 'caffe-orig', name, 'deploy.prototxt')

        utils.makedirs(os.path.dirname(caffemodel_path), exist_ok=True)
        utils.makedirs(os.path.dirname(prototxt_path), exist_ok=True)

        with open(prototxt_path, "w") as f:
            f.write(str(netspec.to_proto()))

        net = caffe.Net(prototxt_path, caffe.TEST)
        if random_weights:
            set_random_weights(net)
        net.save(caffemodel_path)

        return prototxt_path, caffemodel_path

    def test_convolution(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.conv1 = L.Convolution(n.input1,
                                num_output=10,
                                kernel_size=5,
                                pad=10,
                                stride=2,
                                group=2,
                                weight_filler=make_weight_filler(),
                                bias_filler=make_bias_filler())
        self._test_model(*self._netspec_to_model(n, 'convolution'))

    def test_convolution2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.conv1 = L.Convolution(n.input1,
                                num_output=10,
                                kernel_h=5,
                                kernel_w=3,
                                pad_h=10,
                                pad_w=5,
                                weight_filler=make_weight_filler(),
                                bias_filler=make_bias_filler())
        self._test_model(*self._netspec_to_model(n, 'convolution2'))

    def test_convolution3(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.conv1 = L.Convolution(n.input1,
                                num_output=10,
                                kernel_size=5,
                                dilation=[2, 3],
                                bias_term=False,
                                weight_filler=make_weight_filler(),
                                bias_filler=make_bias_filler())
        self._test_model(*self._netspec_to_model(n, 'convolution3'))

    def test_convolution4(self):
        CAFFE_ENGINE = 1
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 6, 6, 6]))
        n.conv1 = L.Convolution(n.input1,
                                num_output=10,
                                kernel_size=5,
                                pad=10,
                                stride=2,
                                group=2,
                                weight_filler=make_weight_filler(),
                                bias_filler=make_bias_filler(),
                                engine=CAFFE_ENGINE)
        self._test_model(*self._netspec_to_model(n, 'convolution4'))

    def test_group_convolution(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([1, 8, 32, 32]))
        n.conv1 = L.Convolution(n.input1,
                                num_output=16,
                                kernel_size=5,
                                group=4,
                                weight_filler=make_weight_filler(),
                                bias_filler=make_bias_filler())
        self._test_model(*self._netspec_to_model(n, 'group_convolution'))

    def test_pooling(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
        n.pooling = L.Pooling(n.input1,
                              kernel_size=10,
                              pad=5,
                              stride=3)
        self._test_model(*self._netspec_to_model(n, 'pooling'))

    def test_pooling2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
        n.pooling1 = L.Pooling(n.input1,
                               kernel_h=10,
                               kernel_w=10,
                               pad_h=5,
                               pad_w=3,
                               stride=2,
                               pool=P.Pooling.AVE)
        self._test_model(*self._netspec_to_model(n, 'pooling2'))

    def test_pooling3(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
        n.pooling1 = L.Pooling(n.input1,
                               kernel_size=10,
                               pad_h=5,
                               pad_w=3,
                               stride_h=2,
                               stride_w=3,
                               pool=P.Pooling.MAX)
        self._test_model(*self._netspec_to_model(n, 'pooling3'))

    def test_deconvolution(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.deconv1 = L.Deconvolution(n.input1,
                                    convolution_param=dict(num_output=10,
                                                           kernel_size=5,
                                                           pad=10,
                                                           stride=2,
                                                           group=2,
                                                           weight_filler=make_weight_filler(),
                                                           bias_filler=make_bias_filler()))
        self._test_model(*self._netspec_to_model(n, 'deconvolution'))

    def test_deconvolution2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.deconv1 = L.Deconvolution(n.input1,
                                    convolution_param=dict(num_output=10,
                                                           kernel_size=5,
                                                           bias_term=False,
                                                           pad_h=10,
                                                           pad_w=5,
                                                           weight_filler=make_weight_filler(),
                                                           bias_filler=make_bias_filler()))
        self._test_model(*self._netspec_to_model(n, 'deconvolution2'))

    def test_deconvolution3(self):
        CAFFE_ENGINE = 1
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 6, 6, 6]))
        n.deconv1 = L.Deconvolution(n.input1,
                                    convolution_param=dict(num_output=10,
                                                           kernel_size=5,
                                                           pad=1,
                                                           stride=2,
                                                           group=2,
                                                           weight_filler=make_weight_filler(),
                                                           bias_filler=make_bias_filler(),
                                                           engine=CAFFE_ENGINE))
        self._test_model(*self._netspec_to_model(n, 'deconvolution3'))

    def test_elu(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.elu1 = L.ELU(n.input1, alpha=1.0)
        self._test_model(*self._netspec_to_model(n, 'elu'))

    def test_elu2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.elu1 = L.ELU(n.input1, alpha=2.0)
        self._test_model(*self._netspec_to_model(n, 'elu2'))

    def test_threshold(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.abs1 = L.Threshold(n.input1, threshold=0.0)
        self._test_model(*self._netspec_to_model(n, 'threshold'))

    def test_threshold2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.abs1 = L.Threshold(n.input1, threshold=1.5)
        self._test_model(*self._netspec_to_model(n, 'threshold2'))

    def test_relu(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.relu1 = L.ReLU(n.input1)
        self._test_model(*self._netspec_to_model(n, 'relu'))

    def test_relu2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.relu1 = L.ReLU(n.input1, negative_slope=0.1)
        self._test_model(*self._netspec_to_model(n, 'relu2'))

    def test_prelu(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.relu1 = L.PReLU(n.input1)
        self._test_model(*self._netspec_to_model(n, 'prelu'))

    def test_prelu2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.relu1 = L.PReLU(n.input1, channel_shared=True)
        self._test_model(*self._netspec_to_model(n, 'prelu2'))

    def test_concat(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.input2 = L.Input(shape=make_shape([10, 6, 64, 64]))
        n.input3 = L.Input(shape=make_shape([10, 8, 64, 64]))
        n.concat1 = L.Concat(n.input1, n.input2, n.input3)
        self._test_model(*self._netspec_to_model(n, 'concat'))

    def test_concat2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.input2 = L.Input(shape=make_shape([10, 6, 64, 64]))
        n.input3 = L.Input(shape=make_shape([10, 8, 64, 64]))
        n.concat1 = L.Concat(n.input1, n.input2, n.input3, axis=1)
        self._test_model(*self._netspec_to_model(n, 'concat2'))

    def test_concat3(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.input2 = L.Input(shape=make_shape([8, 4, 64, 64]))
        n.input3 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.concat1 = L.Concat(n.input1, n.input2, n.input3, axis=0)
        self._test_model(*self._netspec_to_model(n, 'concat3'))

    def test_concat4(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.input2 = L.Input(shape=make_shape([8, 4, 64, 64]))
        n.input3 = L.Input(shape=make_shape([10, 4, 64, 64]))
        n.concat1 = L.Concat(n.input1, n.input2, n.input3, concat_dim=0)
        self._test_model(*self._netspec_to_model(n, 'concat4'))

    def test_flatten(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.flatten1 = L.Flatten(n.input1)
        self._test_model(*self._netspec_to_model(n, 'flatten'))

    def test_flatten2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.flatten1 = L.Flatten(n.input1, axis=-3, end_axis=-2)
        self._test_model(*self._netspec_to_model(n, 'flatten2'))

    def test_softmax(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.softmax1 = L.Softmax(n.input1)
        self._test_model(*self._netspec_to_model(n, 'softmax'))

    def test_softmax2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.softmax1 = L.Softmax(n.input1, axis=2)
        self._test_model(*self._netspec_to_model(n, 'softmax2'))

    def test_sigmoid(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.sigmoid1 = L.Sigmoid(n.input1)
        self._test_model(*self._netspec_to_model(n, 'sigmoid'))

    def test_sigmoid2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
        n.conv1 = L.Convolution(n.input1,
                                num_output=10,
                                kernel_size=5,
                                weight_filler=make_weight_filler(),
                                bias_filler=make_bias_filler())
        n.sigmoid1 = L.Sigmoid(n.conv1)
        self._test_model(*self._netspec_to_model(n, 'sigmoid2'))

    def test_absval(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.abs1 = L.AbsVal(n.input1)
        self._test_model(*self._netspec_to_model(n, 'absval'))

    def test_tanh(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.tanh1 = L.TanH(n.input1)
        self._test_model(*self._netspec_to_model(n, 'tanh'))

    def test_eltwise_sum(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.input2 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.sum1 = L.Eltwise(n.input1, n.input2)
        self._test_model(*self._netspec_to_model(n, 'eltwise_sum'))

    def test_eltwise_prod(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.input2 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.prod1 = L.Eltwise(n.input1, n.input2, operation=P.Eltwise.PROD)
        self._test_model(*self._netspec_to_model(n, 'eltwise_prod'))

    def test_eltwise_max(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.input2 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.max1 = L.Eltwise(n.input1, n.input2, operation=P.Eltwise.MAX)
        self._test_model(*self._netspec_to_model(n, 'eltwise_max'))

    def test_eltwise_sum2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.input2 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.input3 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.sum1 = L.Eltwise(n.input1, n.input2, n.input3, coeff=[1.1, 2.2, 3.3])
        self._test_model(*self._netspec_to_model(n, 'eltwise_sum2'))

    # only testing
    def test_batchnorm(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.batch_norm1 = L.BatchNorm(n.input1, eps=1e-6)
        self._test_model(*self._netspec_to_model(n, 'batchnorm'))

    def test_batchnorm2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.batch_norm1 = L.BatchNorm(n.input1)
        self._test_model(*self._netspec_to_model(n, 'batchnorm2'))

    def test_scale(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.scale1 = L.Scale(n.input1)
        self._test_model(*self._netspec_to_model(n, 'scale'))

    def test_scale2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.scale1 = L.Scale(n.input1, bias_term=True, bias_filler=make_bias_filler())
        self._test_model(*self._netspec_to_model(n, 'scale2'))

    def test_scale3(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.input2 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.scale1 = L.Scale(n.input1, n.input2, bias_term=True, axis=0)
        self._test_model(*self._netspec_to_model(n, 'scale3'))

    def test_scale4(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.scale1 = L.Scale(n.input1, bias_term=True, bias_filler=make_bias_filler(), axis=0)
        self._test_model(*self._netspec_to_model(n, 'scale4'))

    def test_scale5(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.scale1 = L.Scale(n.input1, bias_term=True, bias_filler=make_bias_filler(), axis=2)
        self._test_model(*self._netspec_to_model(n, 'scale5'))

    def test_scale6(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.input2 = L.Input(shape=make_shape([64, 64]))
        n.scale1 = L.Scale(n.input1, n.input2, bias_term=True, axis=2)
        self._test_model(*self._netspec_to_model(n, 'scale6'))

    def test_scale7(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.input2 = L.Input(shape=make_shape([64, 64]))
        n.scale1 = L.Scale(n.input1, n.input2, axis=2)
        self._test_model(*self._netspec_to_model(n, 'scale7'))

    def test_bias(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.bias1 = L.Bias(n.input1)
        self._test_model(*self._netspec_to_model(n, 'bias'))

    def test_bias2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.input2 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.bias1 = L.Bias(n.input1, n.input2, axis=0)
        self._test_model(*self._netspec_to_model(n, 'bias2'))

    def test_bias3(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.bias1 = L.Bias(n.input1, axis=0)
        self._test_model(*self._netspec_to_model(n, 'bias3'))

    def test_bias4(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.bias1 = L.Bias(n.input1, axis=2)
        self._test_model(*self._netspec_to_model(n, 'bias4'))

    def test_bias5(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.input2 = L.Input(shape=make_shape([4, 64, 64]))
        n.bias1 = L.Bias(n.input1, n.input2, axis=1)
        self._test_model(*self._netspec_to_model(n, 'bias5'))

    def test_bias6(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.input2 = L.Input(shape=make_shape([64, 64]))
        n.bias1 = L.Bias(n.input1, n.input2, axis=2)
        self._test_model(*self._netspec_to_model(n, 'bias5'))

    def test_split(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.output1, n.output2 = L.Split(n.input1, ntop=2)
        self._test_model(*self._netspec_to_model(n, 'split'))

    def test_slice(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.output1, n.output2, n.output3 = L.Slice(n.input1, ntop=3, axis=1, slice_point=[1, 3])
        self._test_model(*self._netspec_to_model(n, 'slice'))

    def test_slice2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.output1, n.output2, n.output3, n.output4 = L.Slice(n.input1, ntop=4, axis=0, slice_point=[1, 3, 5])
        self._test_model(*self._netspec_to_model(n, 'slice2'))

    def test_power(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.pow1 = L.Power(n.input1, power=2.0, scale=0.5, shift=0.01)
        self._test_model(*self._netspec_to_model(n, 'power'))

    def test_power2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        # These two powers can be united
        n.pow1 = L.Power(n.input1, scale=0.3)
        n.pow2 = L.Power(n.pow1, power=2)
        self._test_model(*self._netspec_to_model(n, 'power2'))

    def test_power3(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        # These two powers can not be united
        n.pow1 = L.Power(n.input1, power=2.0)
        n.pow2 = L.Power(n.pow1, scale=0.3)
        self._test_model(*self._netspec_to_model(n, 'power3'))

    def test_exp(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.exp1 = L.Exp(n.input1, base=-1.0, scale=0.5, shift=0.01)
        self._test_model(*self._netspec_to_model(n, 'exp'))

    def test_exp2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.exp1 = L.Exp(n.input1, base=2.0, scale=0.5, shift=0.01)
        self._test_model(*self._netspec_to_model(n, 'exp2'))

    def test_exp3(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.exp1 = L.Exp(n.input1, scale=0.5, shift=0.01)
        self._test_model(*self._netspec_to_model(n, 'exp3'))

    def test_exp4(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.exp1 = L.Exp(n.input1, base=2.0, scale=0.5)
        self._test_model(*self._netspec_to_model(n, 'exp4'))

    def test_log(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.log1 = L.Log(n.input1, base=-1.0, scale=0.5, shift=0.01)
        self._test_model(*self._netspec_to_model(n, 'log'), nonnegative_input=True)

    def test_log2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.log1 = L.Log(n.input1, base=2.0, scale=0.5, shift=0.01)
        self._test_model(*self._netspec_to_model(n, 'log2'), nonnegative_input=True)

    def test_log3(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.log1 = L.Log(n.input1, scale=0.5, shift=0.01)
        self._test_model(*self._netspec_to_model(n, 'log3'), nonnegative_input=True)

    def test_log4(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.log1 = L.Log(n.input1, base=4.0, scale=0.5)
        self._test_model(*self._netspec_to_model(n, 'log4'), nonnegative_input=True)

    def test_dropout(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.dropout1 = L.Dropout(n.input1, dropout_ratio=0.6)
        n.conv1 = L.Convolution(n.dropout1,
                                num_output=10,
                                kernel_size=5,
                                weight_filler=make_weight_filler(),
                                bias_filler=make_bias_filler())
        self._test_model(*self._netspec_to_model(n, 'dropout'))

    # fully connected
    def test_inner_product(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.full1 = L.InnerProduct(n.input1, num_output=20)
        self._test_model(*self._netspec_to_model(n, 'inner_product'))

    def test_inner_product2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.inner_product1 = L.InnerProduct(n.input1, num_output=20, bias_term=False)
        self._test_model(*self._netspec_to_model(n, 'inner_product2'))

    def test_inner_product3(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([2, 2, 16, 16]))
        n.inner_product1 = L.InnerProduct(n.input1, num_output=20, bias_term=True, axis=0)
        self._test_model(*self._netspec_to_model(n, 'inner_product3'))

    def test_inner_product4(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([2, 2, 16, 16]))
        n.inner_product1 = L.InnerProduct(n.input1, num_output=20, bias_term=True, axis=2)
        self._test_model(*self._netspec_to_model(n, 'inner_product4'))

    def test_bnll(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.bnll1 = L.BNLL(n.input1)
        self._test_model(*self._netspec_to_model(n, 'bnll'))

    def test_lrn(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.bnll1 = L.LRN(n.input1, local_size=7, alpha=1.1, beta=0.8)
        self._test_model(*self._netspec_to_model(n, 'lrn'))

    def test_lrn2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.bnll1 = L.LRN(n.input1, local_size=7, alpha=1.1, beta=0.8, norm_region=P.LRN.WITHIN_CHANNEL)
        self._test_model(*self._netspec_to_model(n, 'lrn2'))

    def test_mvn(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.bnll1 = L.MVN(n.input1, eps=0.01)
        self._test_model(*self._netspec_to_model(n, 'mvn'))

    def test_mvn2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.bnll1 = L.MVN(n.input1, normalize_variance=False, across_channels=True, eps=0.01)
        self._test_model(*self._netspec_to_model(n, 'mvn2'))

    def test_reshape(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 32, 64]))
        n.reshape1 = L.Reshape(n.input1, shape=make_shape([6, 4, 32 * 64]))
        self._test_model(*self._netspec_to_model(n, 'reshape'))

    def test_reshape2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 32, 64]))
        n.reshape1 = L.Reshape(n.input1, shape=make_shape([0, 0, -1]))
        self._test_model(*self._netspec_to_model(n, 'reshape2'))

    def test_argmax(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.argmax1 = L.ArgMax(n.input1)
        self._test_model(*self._netspec_to_model(n, 'argmax'))

    def test_argmax2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.argmax1 = L.ArgMax(n.input1, axis=-1)
        self._test_model(*self._netspec_to_model(n, 'argmax2'))

    def test_crop(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.reference1 = L.Input(shape=make_shape([1, 1, 32, 32]))
        n.crop1 = L.Crop(n.input1, n.reference1, axis=2, offset=[16, 16])
        n.conv1 = L.Convolution(n.crop1,
                                num_output=10,
                                kernel_size=5,
                                weight_filler=make_weight_filler(),
                                bias_filler=make_bias_filler())
        self._test_model(*self._netspec_to_model(n, 'crop'))

    # Transformations:
    def test_deconvolution_multilinear_upsample(self):
        factor = 2
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
        n.deconv1 = L.Deconvolution(n.input1,
                                    convolution_param=dict(kernel_size=2 * factor - factor % 2,
                                                           stride=factor,
                                                           num_output=3,
                                                           pad=int(math.ceil((factor - 1) / 2.0)),
                                                           weight_filler=dict(type="bilinear"),
                                                           bias_term=False,
                                                           group=3))
        self._test_model(*self._netspec_to_model(n, 'deconvolution_multilinear_upsample', random_weights=False))

    def test_merge_batchnorm_operations(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.batch_norm1 = L.BatchNorm(n.input1)
        n.scale1 = L.Scale(n.batch_norm1, bias_term=True)
        self._test_model(*self._netspec_to_model(n, 'merge_batchnorm_operations'))

    def test_convert_scale_bias_to_mul_add(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.scale1 = L.Scale(n.input1)
        self._test_model(*self._netspec_to_model(n, 'convert_scale_bias_to_mul_add'))

    def test_convert_scale_bias_to_mul_add2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
        n.scale1 = L.Scale(n.input1, bias_term=True)
        self._test_model(*self._netspec_to_model(n, 'convert_scale_bias_to_mul_add2'))

    def test_convert_global_pooling_to_reduce(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
        n.pooling1 = L.Pooling(n.input1,
                               global_pooling=True,
                               pool=P.Pooling.MAX)
        self._test_model(*self._netspec_to_model(n, 'convert_global_pooling_to_reduce'))

    def test_convert_global_pooling_to_reduce2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
        n.pooling1 = L.Pooling(n.input1,
                               global_pooling=True,
                               pool=P.Pooling.AVE)
        self._test_model(*self._netspec_to_model(n, 'convert_global_pooling_to_reduce2'))

    def test_reduce(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
        n.pooling1 = L.Reduction(n.input1, operation=P.Reduction.SUM, axis=0)
        self._test_model(*self._netspec_to_model(n, 'reduce'))

    def test_reduce2(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
        n.pooling1 = L.Reduction(n.input1, operation=P.Reduction.ASUM, axis=1, coeff=1.1)
        self._test_model(*self._netspec_to_model(n, 'reduce2'))

    def test_reduce3(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
        n.pooling1 = L.Reduction(n.input1, operation=P.Reduction.SUMSQ, axis=2, coeff=1.2)
        self._test_model(*self._netspec_to_model(n, 'reduce3'))

    def test_reduce4(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
        n.pooling1 = L.Reduction(n.input1, operation=P.Reduction.MEAN, axis=3, coeff=1.3)
        self._test_model(*self._netspec_to_model(n, 'reduce3'))

    @unittest.skip("This is not supported in Caffe 1.0, only on the master branch.")
    def test_pooling_round_mode(self):
        n = caffe.NetSpec()
        n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
        n.pooling = L.Pooling(n.input1, kernel_size=10, pad=5, stride=3, round_mode=P.Pooling.FLOOR)
        self._test_model(*self._netspec_to_model(n, 'pooling_round_mode'))


if __name__ == '__main__':
    unittest.main()
