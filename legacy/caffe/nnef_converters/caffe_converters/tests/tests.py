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

from __future__ import division, print_function

import inspect
import math
import os
import shutil
import sys
import traceback
import types

import numpy as np

os.environ['GLOG_minloglevel'] = '3'  # suppress Caffe verbose/warning logs

import caffe
from caffe import layers as L
from caffe import params as P

from .. import caffe_to_nnef
from .. import nnef_to_caffe
from ...common import utils
from ...common import nnef_parser_config


def get_functions(prefix="", module=None):
    # type: (str, types.ModuleType)->list[function]

    if module is None:
        caller_frame = inspect.stack()[1]
        module = inspect.getmodule(caller_frame[0])

    return sorted(
        [
            obj
            for name, obj in inspect.getmembers(module)
            if inspect.isfunction(obj) and name.startswith(prefix)
        ],
        key=lambda f: inspect.getsourcelines(f)[1]
    )


def get_summary(output_dict):
    # type: (dict[str, np.ndarray]) -> list[tuple]

    return [(('name', k),
             ('shape', output_dict[k].shape),
             ('min', output_dict[k].min()),
             ('max', output_dict[k].max()))
            for k in sorted(output_dict.keys())]


def get_file_names_rec(path, ext):
    # type: (str, str)-> list[str]

    return [os.path.join(dp, f)
            for dp, dn, filenames in os.walk(path)
            for f in filenames
            if os.path.splitext(f)[1] == '.' + ext]


def make_shape(list_):
    # type: ()->dict[str, object]

    return dict(dim=list_)


def make_weight_filler():
    # type: ()->dict[str, object]

    return dict(type='uniform', min=-0.1, max=0.1)


def make_bias_filler():
    # type: ()->dict[str, object]

    return dict(type='uniform', min=-0.05, max=0.05)


def set_random_weights(net):
    # type: (caffe.Net) -> None

    def get_seed(name_):
        import hashlib
        return int(hashlib.md5(name_.encode()).hexdigest()[:8], 16)

    for name, variables in net.params.items():
        np.random.seed(get_seed(name))
        for variable in variables:
            if variable.data.size == 1:
                variable.data[...] = 1.0
            else:
                variable.data[...] = 0.1 + 0.1 * np.random.random(list(variable.data.shape))


def test_1():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
    n.conv1 = L.Convolution(n.input1,
                            num_output=10,
                            kernel_size=5,
                            weight_filler=make_weight_filler(),
                            bias_filler=make_bias_filler())
    n.input2 = L.Input(shape=make_shape([10, 3, 64, 64]))
    n.conv2 = L.Convolution(n.input2,
                            num_output=10,
                            kernel_size=5,
                            weight_filler=make_weight_filler(),
                            bias_filler=make_bias_filler())
    n.add1 = L.Eltwise(n.conv1,
                       n.conv2,
                       operation=P.Eltwise.SUM)
    return n


def test_input():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))

    n.conv1 = L.Convolution(n.input1,
                            num_output=10,
                            kernel_size=5,
                            weight_filler=make_weight_filler(),
                            bias_filler=make_bias_filler())
    return n


# Before enabling this test you have to provide cifar10_test_lmdb
def disabled_test_data():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.data1 = L.Data(source="../cifar10_test_lmdb", batch_size=128, backend=P.Data.LMDB)
    n.conv1 = L.Convolution(n.data1,
                            num_output=10,
                            kernel_size=5,
                            weight_filler=make_weight_filler(),
                            bias_filler=make_bias_filler())
    return n


def test_python():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.python1 = L.Python(module="custom_python_layer",
                         layer="CustomPythonLayer",
                         param_str=str({"shape": [10, 3, 64, 64], "value": 5.0}))
    n.conv1 = L.Convolution(n.python1,
                            num_output=10,
                            kernel_size=5,
                            weight_filler=make_weight_filler(),
                            bias_filler=make_bias_filler())
    return n


def test_lrn():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
    n.lrn1 = L.LRN(n.input1)

    return n


# only 4d is supported
def test_convolution():
    # type: ()->caffe.NetSpec

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
    return n


def test_convolution2():
    # type: ()->caffe.NetSpec

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
    return n


def test_convolution3():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
    n.conv1 = L.Convolution(n.input1,
                            num_output=10,
                            kernel_size=5,
                            dilation=[2, 3],
                            bias_term=False,
                            weight_filler=make_weight_filler(),
                            bias_filler=make_bias_filler())
    return n


def test_group_convolution():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([1, 8, 32, 32]))
    n.conv1 = L.Convolution(n.input1,
                            num_output=16,
                            kernel_size=5,
                            group=4,
                            weight_filler=make_weight_filler(),
                            bias_filler=make_bias_filler())
    return n


# only 4d is supported
# stochastic not supported
def test_pooling():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
    n.pooling = L.Pooling(n.input1,
                          kernel_size=10,
                          pad=5,
                          stride=3)
    return n


def test_pooling2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
    n.pooling1 = L.Pooling(n.input1,
                           kernel_h=10,
                           kernel_w=10,
                           pad_h=5,
                           pad_w=3,
                           stride=2,
                           pool=P.Pooling.AVE)
    return n


def test_pooling3():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
    n.pooling1 = L.Pooling(n.input1,
                           kernel_size=10,
                           pad_h=5,
                           pad_w=3,
                           stride_h=2,
                           stride_w=3,
                           pool=P.Pooling.MAX)
    return n


# only 4d is supported
def test_deconvolution():
    # type: ()->caffe.NetSpec

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
    return n


def test_deconvolution2():
    # type: ()->caffe.NetSpec

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
    return n


# alpha != 1 not supported yet
def test_elu():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
    n.elu1 = L.ELU(n.input1, alpha=1.0)
    return n


def test_relu():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
    n.relu1 = L.ReLU(n.input1)
    return n


def test_relu2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
    n.relu1 = L.ReLU(n.input1, negative_slope=0.1)
    return n


def test_prelu():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
    n.relu1 = L.PReLU(n.input1)
    return n


def test_prelu2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
    n.relu1 = L.PReLU(n.input1, channel_shared=True)
    return n


def test_concat():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
    n.input2 = L.Input(shape=make_shape([10, 6, 64, 64]))
    n.input3 = L.Input(shape=make_shape([10, 8, 64, 64]))
    n.concat1 = L.Concat(n.input1, n.input2, n.input3)
    return n


def test_concat2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 4, 64, 64]))
    n.input2 = L.Input(shape=make_shape([10, 6, 64, 64]))
    n.input3 = L.Input(shape=make_shape([10, 8, 64, 64]))
    n.concat1 = L.Concat(n.input1, n.input2, n.input3, axis=1)
    return n


def test_concat3():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.input2 = L.Input(shape=make_shape([8, 4, 64, 64]))
    n.input3 = L.Input(shape=make_shape([10, 4, 64, 64]))
    n.concat1 = L.Concat(n.input1, n.input2, n.input3, axis=0)
    return n


def test_concat4():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.input2 = L.Input(shape=make_shape([8, 4, 64, 64]))
    n.input3 = L.Input(shape=make_shape([10, 4, 64, 64]))
    n.concat1 = L.Concat(n.input1, n.input2, n.input3, concat_dim=0)
    return n


def test_flatten():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.flatten1 = L.Flatten(n.input1)
    return n


def test_flatten2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.flatten1 = L.Flatten(n.input1, axis=-3, end_axis=-2)
    return n


def test_softmax():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.softmax1 = L.Softmax(n.input1)
    return n


def test_softmax2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.softmax1 = L.Softmax(n.input1, axis=2)
    return n


def test_sigmoid():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.sigmoid1 = L.Sigmoid(n.input1)
    return n


def test_sigmoid2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
    n.conv1 = L.Convolution(n.input1,
                            num_output=10,
                            kernel_size=5,
                            weight_filler=make_weight_filler(),
                            bias_filler=make_bias_filler())
    n.sigmoid1 = L.Sigmoid(n.conv1)
    return n


def test_absval():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.abs1 = L.AbsVal(n.input1)
    return n


def test_tanh():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.tanh1 = L.TanH(n.input1)
    return n


# only 2 input tensors are supported
# coeffs are not supported
def test_eltwise_sum():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.input2 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.sum1 = L.Eltwise(n.input1, n.input2)
    return n


def test_eltwise_prod():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.input2 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.prod1 = L.Eltwise(n.input1, n.input2, operation=P.Eltwise.PROD)
    return n


def test_eltwise_max():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.input2 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.max1 = L.Eltwise(n.input1, n.input2, operation=P.Eltwise.MAX)
    return n


# only testing
def test_batchnorm():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.batch_norm1 = L.BatchNorm(n.input1, eps=1e-6)
    return n


def test_batchnorm2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.batch_norm1 = L.BatchNorm(n.input1)
    return n


# only 1-input version is supported
def test_scale():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.scale1 = L.Scale(n.input1)
    return n


def test_scale2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.scale1 = L.Scale(n.input1, bias_term=True, bias_filler=make_bias_filler())
    return n


def test_power():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.pow1 = L.Power(n.input1, power=2.0, scale=0.5, shift=0.01)
    return n


def test_power2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    # These two powers can be united
    n.pow1 = L.Power(n.input1, scale=0.3)
    n.pow2 = L.Power(n.pow1, power=2)
    return n


def test_power3():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    # These two powers can not be united
    n.pow1 = L.Power(n.input1, power=2.0)
    n.pow2 = L.Power(n.pow1, scale=0.3)
    return n


def test_dropout():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.dropout1 = L.Dropout(n.input1, dropout_ratio=0.6)
    n.conv1 = L.Convolution(n.dropout1,
                            num_output=10,
                            kernel_size=5,
                            weight_filler=make_weight_filler(),
                            bias_filler=make_bias_filler())
    return n


# fully connected
# axis, transpose not supported
def test_inner_product():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.full1 = L.InnerProduct(n.input1, num_output=20)
    return n


def test_inner_product2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.inner_product1 = L.InnerProduct(n.input1, num_output=20, bias_term=False)
    return n


def test_bnll():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.bnll1 = L.BNLL(n.input1)
    return n


def test_reshape():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 32, 64]))
    n.reshape1 = L.Reshape(n.input1, shape=make_shape([6, 4, 32 * 64]))
    return n


def test_reshape2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 32, 64]))
    n.reshape1 = L.Reshape(n.input1, shape=make_shape([0, 0, -1]))
    return n


def test_argmax():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.argmax1 = L.ArgMax(n.input1)
    return n


def test_argmax2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.argmax1 = L.ArgMax(n.input1, axis=-1)
    return n


# it is a passthrough now, but it is strange that reference1 is preserved
def test_crop():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.reference1 = L.Input(shape=make_shape([1, 1, 32, 32]))
    n.crop1 = L.Crop(n.input1, n.reference1, axis=2, offset=[16, 16])
    n.conv1 = L.Convolution(n.crop1,
                            num_output=10,
                            kernel_size=5,
                            weight_filler=make_weight_filler(),
                            bias_filler=make_bias_filler())
    return n


# Transformations:
def test_deconvolution_multilinear_upsample():
    # type: ()->caffe.NetSpec

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
    return n


def test_merge_batchnorm_operations():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.batch_norm1 = L.BatchNorm(n.input1)
    n.scale1 = L.Scale(n.batch_norm1, bias_term=True)
    return n


def test_convert_scale_bias_to_mul_add():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.scale1 = L.Scale(n.input1)
    return n


def test_convert_scale_bias_to_mul_add2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([6, 4, 64, 64]))
    n.scale1 = L.Scale(n.input1, bias_term=True)
    return n


def test_convert_global_pooling_to_reduce():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
    n.pooling1 = L.Pooling(n.input1,
                           global_pooling=True,
                           pool=P.Pooling.MAX)
    return n


def test_convert_global_pooling_to_reduce2():
    # type: ()->caffe.NetSpec

    n = caffe.NetSpec()
    n.input1 = L.Input(shape=make_shape([10, 3, 64, 64]))
    n.pooling1 = L.Pooling(n.input1,
                           global_pooling=True,
                           pool=P.Pooling.AVE)
    return n


def do_test(prototxt_path, test_path):
    # type: (str, str)->bool

    utils.reset_error()
    # noinspection PyBroadException
    try:
        test_path = utils.without_slash(test_path)
        utils.ensure_dir(test_path)

        caffemodel_path = test_path + ".caffemodel"
        nnef_path = test_path + "/graph.nnef"
        test_path_caffe = test_path + "_caffe"
        activations_path2 = test_path_caffe + "/activations"
        prototxt_path2 = test_path_caffe + "/graph.prototxt"
        caffemodel_path2 = test_path_caffe + "/graph.caffemodel"

        net = caffe.Net(prototxt_path, caffe.TEST)
        if (test_path != "test_deconvolution_multilinear_upsample"
                and ("alexnet" in test_path or "resnet10" in test_path or "gt_mobilenet" in test_path)):
            set_random_weights(net)
        net.save(caffemodel_path)

        caffe_to_nnef.prototxt_to_nnef(prototxt_path, nnef_path,
                                       caffemodel_file_name=caffemodel_path, with_activations=True)

        nnef_parser_config.load_graph(nnef_path)

        nnef_to_caffe.convert_internal(test_path, test_path_caffe)

        os.chdir(test_path_caffe)
        try:
            caffe_to_nnef.export_activations("../" + prototxt_path2, "../" + activations_path2,
                                             caffemodel_file_name="../" + caffemodel_path2)
            pass
        finally:
            os.chdir("..")

        utils.raise_if_had_error()

        no_heatmap_test = [
            # the export is not equivalent for this
            "test_python",
            # heatmap testing is done by name, so this won't work
            "test_replace_forbidden_characters",
            # they explode with our initialization
            "vgg",
            "resnet50"
        ]

        if test_path in no_heatmap_test:
            return True

        return utils.compare_activation_dirs_np(test_path + "/activations", test_path_caffe + "/activations")
    except Exception:
        utils.print_error(traceback.format_exc())
        return False


def main():
    # type: ()->None

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

    # We write everything to stdout, to prevent lines from appearing out of order
    utils.error_file = sys.stdout
    utils.warning_file = sys.stdout

    tmp_dir = "tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    os.chdir(tmp_dir)
    fails = []
    test_funs = get_functions("test_")
    for fun in test_funs:
        path = fun.__name__
        print("\nTESTING", path)
        prototxt_path = path + ".prototxt"
        net_spec = fun()  # type: caffe.NetSpec
        with open(prototxt_path, "w") as f:
            print(net_spec.to_proto(), file=f)
        success = do_test(prototxt_path, path)
        if success:
            print(path + " succeeded")
        else:
            utils.print_error(path + " failed")
            fails.append(path)

    prototxt_paths = get_file_names_rec("../nnef_converters/caffe_converters/tests/sample-networks", "prototxt")
    for prototxt_path in prototxt_paths:
        path = os.path.basename(prototxt_path)[:-len(".prototxt")]
        print("\nTESTING", path)
        success = do_test(prototxt_path, path)
        if success:
            print(path + " succeeded")
        else:
            utils.print_error(path + " failed")
            fails.append(path)

    os.chdir("..")

    if fails:
        print("Failed:", fails)
    else:
        print("All good!")


if __name__ == "__main__":
    main()
