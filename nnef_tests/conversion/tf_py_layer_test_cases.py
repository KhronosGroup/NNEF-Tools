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
import tensorflow as tf
from tensorflow.contrib import slim as tf_slim
from tensorflow.contrib.layers.python.layers import layers as tf_layers

from nnef_tools.io.tensorflow.tf_py.tf_py_compat import tf_internal
from nnef_tests.conversion.tf_py_test_runner import TFPyTestRunner

if not os.path.exists('nnef_tools') and os.path.exists('../../nnef_tools'):
    os.chdir('../..')


def meshgrid4(w, h):
    return [tf.ones((1, h, 1, 1), dtype=tf.float32) * tf.constant(np.arange(w, dtype=np.float32)[None, None, :, None]),
            tf.constant(np.arange(h, dtype=np.float32)[None, :, None, None]) * tf.ones((1, 1, w, 1), dtype=tf.float32)]


def get_uvd_grid(d0, d1, stride, roffset, eps=1e-3):
    sh = np.array([roffset, 0.5])

    px, py = meshgrid4(d0.shape.as_list()[-2], d0.shape.as_list()[-3])

    py = (tf.zeros_like(d0) + py + sh[0]) * stride

    px = d0 + (px + sh[1]) * stride

    points = tf.concat((px, py, tf.nn.relu(d0 - d1) + eps), axis=-1)
    return points


def reproject_uvdgrid_to3d_notranspose(uvd_grid, Q):
    # Projects 'uvd' points to 3D space
    shape = uvd_grid.shape.as_list()[:-1] + [4]
    XYZ_grid = tf.matmul(Q, tf.reshape(
        tf.concat((uvd_grid, tf.ones(uvd_grid.shape.as_list()[:-1] + [1], dtype=tf.float32)), axis=-1),
        (-1, shape[1] * shape[2], 4)), transpose_b=True)
    XYZ_grid = XYZ_grid[..., :3, :] / tf.reshape(XYZ_grid[:, 3, :],
                                                 [shape[0], 1, -1])
    return tf.reshape(XYZ_grid, np.array(uvd_grid.shape.as_list())[[0, 3, 1, 2]])


def reproject_uvdgrid_to3d_notranspose2(uvd_grid, Q):
    # Projects 'uvd' points to 3D space
    shape = uvd_grid.shape.as_list()[:-1] + [4]  # nhwc
    ones = tf.ones(uvd_grid.shape.as_list()[:-1] + [1], dtype=tf.float32)  # nhwc
    concat = tf.concat((uvd_grid, ones), axis=-1)  # nhwc

    reshape = tf.reshape(concat, (-1, shape[1] * shape[2], 4))  # n, h*w, c

    XYZ_grid = tf.matmul(Q, reshape, transpose_b=True)  # n, c, h*w

    part = XYZ_grid[:, 3, :]  # n, h*w

    reshape2 = tf.reshape(part, [shape[0], 1, -1])  # n, 1, h*w
    # reshape2 = XYZ_grid[:, 3:4, :]

    part2 = XYZ_grid[:, :3, :]  # n, c, h*w

    XYZ_grid = part2 / reshape2  # n, c, h*w

    return tf.reshape(XYZ_grid, np.array(uvd_grid.shape.as_list())[[0, 3, 1, 2]])  # n, c, h, w


def network_naming():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")
    return {'y': -(x * x)}


def network_naming2():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")
    y = -(x * x)
    return y, y


def network_naming3():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3])
    y = tf.placeholder(tf.float32, shape=[10, 64, 64, 3])
    a = x + y
    b = x - y
    return a, b


def network_naming4():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3])
    return -x


def network_my_reshape():
    input_ = tf.placeholder(tf.float32, shape=[1], name="input")
    var = tf.get_variable(dtype=tf.float32, shape=[1], name="var1")

    return input_ + var + tf.reshape(0.0, shape=[1]) + tf.reshape(0.0, shape=[1])


def network_float():
    input_ = tf.placeholder(tf.float32, shape=[], name="input")
    return input_ + tf.constant(1.0)


def network_meshgrid():
    # Adds the i and j coordinates as new channels
    input_ = tf.placeholder(tf.float32, shape=[8, 12, 16, 3], name="input")
    i, j = tf.meshgrid(tf.range(float(12)), tf.range(float(16)), indexing="ij")
    mesh = tf.stack([i, j], axis=2)
    mesh = mesh[tf.newaxis, ...]
    mesh = tf.tile(mesh, [8, 1, 1, 1])
    return tf.concat([input_, mesh], axis=3)


def network_name_starting_with_num():
    input_ = tf.placeholder(tf.float32, shape=[10, 16, 16, 3], name="1.input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 8], name="2.filter")

    conv = tf.nn.conv2d(input=input_, filter=filter_, strides=[1, 1, 1, 1], padding='VALID')
    bias = tf.get_variable(dtype=tf.float32, shape=[8], name="bias")

    return tf.nn.bias_add(conv, bias)


def network_conv_grad_input():
    input_ = tf.placeholder(tf.float32, shape=[10, 16, 16, 3], name="1.input")
    filter_ = tf.constant(value=1.0, dtype=tf.float32, shape=[4, 4, 3, 8], name="filter")

    conv = tf.nn.conv2d(input=input_, filter=filter_, strides=[1, 1, 1, 1], padding='VALID')
    bias = tf.get_variable(dtype=tf.float32, shape=[8], name="bias")

    return tf.nn.bias_add(conv, bias)


def network_optimizer_no_io_transpose():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")
    input_ = tf.reshape(input_, input_.shape)

    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 8], name="filter")
    conv1 = tf.nn.conv2d(input_, filter_, strides=[1, 2, 2, 1], padding='VALID')

    relu = tf.nn.relu(conv1)

    filter2 = tf.get_variable(dtype=tf.float32, shape=[4, 4, 8, 16], name="filter2")
    conv2 = tf.nn.conv2d(relu, filter2, strides=[1, 1, 1, 1], padding='VALID')

    leaky_relu = tf.nn.leaky_relu(conv2, alpha=0.3)

    filter3 = tf.get_variable(dtype=tf.float32, shape=[4, 4, 16, 16], name="filter3")
    conv3 = tf.nn.conv2d(leaky_relu, filter3, strides=[1, 1, 1, 1], padding='VALID')

    return tf.reshape(conv3, conv3.shape)


def network_optimizer_manual_output_transpose():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")
    return tf.transpose(input_, perm=[0, 3, 1, 2])


def network_optimizer1():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")

    return tf.transpose(-input_, perm=[0, 3, 1, 2])


def network_optimizer2():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")

    trans = tf.transpose(-input_, perm=[0, 2, 3, 1])
    return tf.transpose(trans, perm=[0, 3, 1, 2])


def network_optimizer3():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

    return tf.nn.conv2d(input_, filter_, strides=[1, 2, 2, 1], padding='VALID')


def network_optimizer3b():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

    return -tf.nn.conv2d(input_, filter_, strides=[1, 2, 2, 1], padding='VALID')


def network_optimizer3c():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 5], name="filter")

    return tf.nn.depthwise_conv2d(input_, filter_, strides=[1, 2, 2, 1], padding='VALID')


def network_optimizer3d():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

    conv = tf.nn.conv2d(input_, filter_, strides=[1, 2, 2, 1], padding='VALID')
    return {"conv": conv, "flat": tf.reshape(conv, shape=[-1])}


def network_optimizer4():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

    conv1 = tf.nn.conv2d(input_, filter_, strides=[1, 2, 2, 1], padding='VALID')

    input2_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input2")
    filter2_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter2")

    conv2 = tf.nn.conv2d(input2_, filter2_, strides=[1, 2, 2, 1], padding='VALID')

    return conv1 + conv2


def network_optimizer5():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

    conv = tf.nn.conv2d(input_, filter_, strides=[1, 2, 2, 1], padding='VALID')

    bias = tf.get_variable(dtype=tf.float32, shape=[16], name="bias")
    return tf.nn.bias_add(conv, bias)


def network_optimizer6():
    input_ = tf.placeholder(tf.float32, shape=[10, 3, 64, 64], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

    return tf.nn.conv2d(input_, filter_, strides=[1, 1, 2, 2], padding='VALID', data_format="NCHW")


def network_optimizer7():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")

    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 8], name="filter")
    conv1 = tf.nn.conv2d(input_, filter_, strides=[1, 2, 2, 1], padding='VALID')

    relu = tf.nn.relu(conv1)

    filter2 = tf.get_variable(dtype=tf.float32, shape=[4, 4, 8, 16], name="filter2")
    conv2 = tf.nn.conv2d(relu, filter2, strides=[1, 1, 1, 1], padding='VALID')

    leaky_relu = tf.nn.leaky_relu(conv2, alpha=0.3)

    filter3 = tf.get_variable(dtype=tf.float32, shape=[4, 4, 16, 16], name="filter3")
    conv3 = tf.nn.conv2d(leaky_relu, filter3, strides=[1, 1, 1, 1], padding='VALID')

    reshape = tf.reshape(conv3, shape=[10, 10000])

    weights_ = tf.get_variable(dtype=tf.float32, shape=[10000, 25], name="weights")
    matmul = tf.matmul(reshape, weights_)

    return matmul


def network_optimizer8():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")

    return tf.nn.max_pool(input_, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')


def network_optimizer9():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")

    return tf.nn.max_pool_with_argmax(input_, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')


def network_optimizer_broadcast1():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[64, 3], name="b")
    return a + b


def network_optimizer10():
    d0 = tf.placeholder(tf.float32, shape=[10, 16, 16, 1], name='d0')
    d1 = tf.placeholder(tf.float32, shape=[10, 16, 16, 1], name='d1')
    Q = tf.placeholder(tf.float32, shape=[10, 4, 4], name='Q')

    stride = 1
    roffset = 0.1
    grid = get_uvd_grid(d0, d1, stride, roffset)

    grid = reproject_uvdgrid_to3d_notranspose(grid, Q)

    return grid


def network_optimizer11():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 8], name="filter")

    conv1 = tf.nn.conv2d(input_, filter_, strides=[1, 2, 2, 1], padding='VALID')

    relu = tf.nn.relu(conv1)

    filter2 = tf.get_variable(dtype=tf.float32, shape=[4, 4, 8, 1], name="filter2")

    conv2 = tf.nn.conv2d(relu, filter2, strides=[1, 1, 1, 1], padding='VALID')

    Q = tf.placeholder(tf.float32, shape=[10, 4, 4], name='Q')

    stride = 1
    roffset = 0.1
    grid = get_uvd_grid(conv2, -conv2, stride, roffset)

    return reproject_uvdgrid_to3d_notranspose2(grid, Q)


def network_dtype_int32():
    a = tf.placeholder(dtype=tf.int32, shape=[1, 2, 3], name="a")
    b = tf.get_variable(dtype=tf.int32, shape=[1, 2, 3], name="b")
    c = tf.constant(dtype=tf.int32, shape=[1, 2, 3], name="c", value=1)

    return tf.concat([a, b, c], axis=0)


def network_dtype_float64():
    a = tf.placeholder(dtype=tf.float64, shape=[1, 2, 3], name="a")
    b = tf.get_variable(dtype=tf.float64, shape=[1, 2, 3], name="b")
    c = tf.constant(dtype=tf.float64, shape=[1, 2, 3], name="c", value=1)

    return tf.concat([a, b, c], axis=0)


def network_dtype_bool():
    a = tf.placeholder(dtype=tf.bool, shape=[1, 2, 3], name="a")
    b = tf.get_variable(dtype=tf.bool, shape=[1, 2, 3], name="b")
    c = tf.constant(dtype=tf.bool, shape=[1, 2, 3], name="c", value=True)

    return tf.concat([a, b, c], axis=0)


def network_for_gradients1():
    x = tf.placeholder(tf.float32, shape=[1, 2, 3], name='x')
    a = tf.get_variable('a', shape=[1, 2, 3])
    b = tf.get_variable('b', shape=[1, 2, 3])
    c = tf.constant(value=13.0, dtype=tf.float32, shape=[1, 2, 3])
    y = tf.placeholder(tf.float32, shape=[1, 2, 3], name='y')

    yy = (tf.exp(x) + a) * b + c

    return tf.reduce_mean(tf.square(y - yy))


def network_for_gradients2():
    x = tf.placeholder(tf.float32, shape=[1, 2, 3], name='x')
    a = tf.get_variable('a', shape=[1, 2, 3])
    b = tf.get_variable('b', shape=[1, 2, 3])
    c = tf.constant(value=13.0, dtype=tf.float32, shape=[1, 2, 3])
    y = tf.placeholder(tf.float32, shape=[3, 1, 2, 3], name='y')

    yy = tf.stack([(tf.exp(x) + a) * b + c, a, b])

    return tf.reduce_mean(tf.square(y - yy))


def network_for_gradients3():
    tan_pitch1 = tf.placeholder(name='tan_pitch', shape=[1], dtype=tf.float32)
    return tf.pow(tan_pitch1, y=2.0)


def network_for_gradients4():
    tan_pitch1 = tf.placeholder(name='tan_pitch', shape=[1], dtype=tf.float32)
    return tf.sqrt(tan_pitch1)


def network_for_gradients5():
    input1 = tf.placeholder(dtype=tf.float32, shape=[100, 2], name='input1')
    input2 = tf.placeholder(dtype=tf.float32, shape=[100, 2], name='input2')
    input3 = tf.placeholder(dtype=tf.float32, shape=[3, 3], name='input3')
    input4 = tf.placeholder(dtype=tf.float32, shape=[3, 3], name='input4')
    input5 = tf.placeholder(dtype=tf.float32, shape=[3], name='input5')
    input6 = tf.placeholder(dtype=tf.float32, shape=[3, 3], name='input6')
    tan_pitch1 = tf.get_variable(name='tan_pitch', shape=[1], dtype=tf.float32)
    tan_roll1 = tf.get_variable(name='tan_roll', shape=[1], dtype=tf.float32)
    height1 = tf.get_variable(name='height', shape=[1], dtype=tf.float32)
    unstack1, unstack2 = tf.unstack(input1, num=2, axis=1)
    stack1 = tf.stack([unstack1, unstack2], axis=1)
    unstack3, unstack4 = tf.unstack(input2, num=2, axis=1)
    stack2 = tf.stack([unstack3, unstack4], axis=1)
    pow1 = tf.pow(tan_pitch1, y=2.0)
    add1 = tf.add(x=1.0, y=pow1)
    pow2 = tf.pow(tan_roll1, y=2.0)
    add2 = tf.add(add1, pow2)
    sqrt1 = tf.sqrt(add2)
    slice1 = tf.slice(stack1, begin=[0, 1], size=[-1, 1])
    reshape1 = tf.reshape(slice1, shape=[100])
    slice2 = tf.slice(input3, begin=[1, 2], size=[1, 1])
    reshape2 = tf.reshape(slice2, shape=[1])
    sub1 = tf.subtract(reshape1, reshape2)
    slice3 = tf.slice(input3, begin=[1, 1], size=[1, 1])
    reshape3 = tf.reshape(slice3, shape=[1])
    div1 = tf.divide(sub1, reshape3)
    slice4 = tf.slice(stack1, begin=[0, 0], size=[-1, 1])
    reshape4 = tf.reshape(slice4, shape=[100])
    slice5 = tf.slice(input3, begin=[0, 2], size=[1, 1])
    reshape5 = tf.reshape(slice5, shape=[1])
    sub2 = tf.subtract(reshape4, reshape5)
    slice6 = tf.slice(input3, begin=[0, 0], size=[1, 1])
    reshape6 = tf.reshape(slice6, shape=[1])
    div2 = tf.divide(sub2, reshape6)
    mul1 = tf.multiply(height1, sqrt1)
    sub3 = tf.subtract(div1, tan_pitch1)
    mul2 = tf.multiply(tan_roll1, div2)
    sub4 = tf.subtract(sub3, mul2)
    div3 = tf.divide(mul1, sub4)
    mul3 = tf.multiply(div3, div2)
    mul4 = tf.multiply(tan_pitch1, div3)
    mul5 = tf.multiply(tan_roll1, mul3)
    add3 = tf.add(mul4, mul5)
    mul6 = tf.multiply(height1, sqrt1)
    add4 = tf.add(add3, mul6)
    stack3 = tf.stack([mul3, add4, div3], axis=1)
    pow3 = tf.pow(tan_pitch1, y=2.0)
    add5 = tf.add(x=1.0, y=pow3)
    pow4 = tf.pow(tan_roll1, y=2.0)
    add6 = tf.add(add5, pow4)
    sqrt2 = tf.sqrt(add6)
    slice7 = tf.slice(stack2, begin=[0, 1], size=[-1, 1])
    reshape7 = tf.reshape(slice7, shape=[100])
    slice8 = tf.slice(input4, begin=[1, 2], size=[1, 1])
    reshape8 = tf.reshape(slice8, shape=[1])
    sub5 = tf.subtract(reshape7, reshape8)
    slice9 = tf.slice(input4, begin=[1, 1], size=[1, 1])
    reshape9 = tf.reshape(slice9, shape=[1])
    div4 = tf.divide(sub5, reshape9)
    slice10 = tf.slice(stack2, begin=[0, 0], size=[-1, 1])
    reshape10 = tf.reshape(slice10, shape=[100])
    slice11 = tf.slice(input4, begin=[0, 2], size=[1, 1])
    reshape11 = tf.reshape(slice11, shape=[1])
    sub6 = tf.subtract(reshape10, reshape11)
    slice12 = tf.slice(input4, begin=[0, 0], size=[1, 1])
    reshape12 = tf.reshape(slice12, shape=[1])
    div5 = tf.divide(sub6, reshape12)
    mul7 = tf.multiply(height1, sqrt2)
    sub7 = tf.subtract(div4, tan_pitch1)
    mul8 = tf.multiply(tan_roll1, div5)
    sub8 = tf.subtract(sub7, mul8)
    div6 = tf.divide(mul7, sub8)
    mul9 = tf.multiply(div6, div5)
    mul10 = tf.multiply(tan_pitch1, div6)
    mul11 = tf.multiply(tan_roll1, mul9)
    add7 = tf.add(mul10, mul11)
    mul12 = tf.multiply(height1, sqrt2)
    add8 = tf.add(add7, mul12)
    stack4 = tf.stack([mul9, add8, div6], axis=1)
    trans1 = tf.transpose(stack4, perm=[1, 0])
    trans2 = tf.transpose(input6, perm=[1, 0])
    matmul1 = tf.matmul(trans2, trans1, transpose_a=False, transpose_b=False)
    trans3 = tf.transpose(matmul1, perm=[1, 0])
    sub9 = tf.subtract(trans3, input5)
    trans4 = tf.transpose(stack3, perm=[1, 0])
    trans5 = tf.transpose(sub9, perm=[1, 0])
    sub10 = tf.subtract(trans4, trans5)
    sqr1 = tf.square(sub10)
    reduce1 = tf.reduce_sum(sqr1, axis=[0], keep_dims=True)
    reshape13 = tf.reshape(reduce1, shape=[100])
    add9 = tf.add(reshape13, y=0.01)
    reduce2 = tf.reduce_mean(add9, axis=[0], keep_dims=True)
    reshape14 = tf.reshape(reduce2, shape=[1])
    output = tf.add(x=0.0, y=reshape14)

    return output


def network_strided_slice1():
    x = tf.placeholder(tf.float32, shape=[10, 10, 10], name='x')
    return x[0:1, :, 2:-2]


def network_strided_slice2():
    x = tf.placeholder(tf.float32, shape=[10, 10, 10], name='x')
    return x[0:1, 1, 1]


def network_strided_slice3():
    x = tf.placeholder(tf.float32, shape=[10, 10, 10], name='x')
    return x[1, 1, 1]


def network_strided_slice4():
    x = tf.placeholder(tf.float32, shape=[10, 10, 10], name='x')
    return x[:, 1:3, tf.newaxis, :]


def network_strided_slice5():
    x = tf.placeholder(tf.float32, shape=[10, 10, 10, 10], name='x')
    return x[1:4, ..., 1]


def network_strided_slice6():
    x = tf.placeholder(tf.float32, shape=[10, 4, 20], name='x')
    return x[:, 3, :]


def network_placeholder1():
    return -tf.placeholder(tf.float32, shape=[1, 2, 3, 1], name="placeholder1")


def network_placeholder2():
    return -tf.placeholder(tf.float64, shape=[1], name="input")


def network_variable1():
    ph = tf.placeholder(tf.float32, shape=[2, 2], name="ph")

    var = tf.Variable(initial_value=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), name="var1")

    return ph + var


def network_variable2():
    ph = tf.placeholder(tf.float32, shape=[1, 2, 3], name="ph")

    var = tf.get_variable('var1', shape=[1, 2, 3])

    return ph + var


def network_variable3():
    ph = tf.placeholder(tf.float32, shape=[1], name="ph")

    var = tf.get_variable('var1', initializer=1.0)

    return ph + var


def network_variable4():
    ph = tf.placeholder(tf.float32, shape=[2, 3, 2], name="ph")

    var = tf.get_variable('var1', initializer=np.array([
        [[1, 3], [1, 4], [1, 5]],
        [[1, 3], [1, 4], [1, 5]]
    ], dtype=np.float32))

    return ph + var


def network_variable_reuse():
    with tf.variable_scope("scope1", reuse=tf.AUTO_REUSE):
        a1 = tf.get_variable('a', shape=[1])
        ph = tf.placeholder(tf.float32, shape=[1], name="ph")
        b = a1 + ph
        a2 = tf.get_variable('a', shape=[1])
        return a1 + a2 + b


def network_constant1():
    ph = tf.placeholder(tf.float32, shape=[2, 2], name="ph")

    const = tf.constant(np.array([[1, 2], [3, 4.0]], dtype=np.float32))

    return ph + const


def network_constant2():
    ph = tf.placeholder(tf.float32, shape=[1], name="ph")

    const = tf.constant(np.array([1.0], dtype=np.float32))

    return ph + const


def network_constant3():
    ph = tf.placeholder(tf.float32, shape=[1], name="ph")

    const = tf.constant(1.0)

    return ph + const


def network_conv1d():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 3, 16], name="filter")

    return tf.nn.conv1d(input_, filter_, 1, 'VALID')


def network_conv2d():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

    return tf.nn.conv2d(input_, filter_, [1, 1, 1, 1], 'VALID')


def network_conv3d():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 4, 3, 16], name="filter")

    return tf.nn.conv3d(input_, filter_, [1, 1, 1, 1, 1], 'VALID')


def network_conv2d_transpose():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 16, 3], name="filter")

    return tf.nn.conv2d_transpose(input_, filter_, output_shape=[1, 64, 64, 16], strides=[1, 1, 1, 1], padding='SAME')


def network_conv3d_transpose():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 4, 16, 3], name="filter")

    return tf.nn.conv3d_transpose(input_, filter_, output_shape=[1, 64, 64, 64, 16], strides=[1, 1, 1, 1, 1],
                                  padding='SAME')


def network_depthwise_conv2d():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 2], name="filter")

    return tf.nn.depthwise_conv2d(input_, filter_, [1, 1, 1, 1], 'VALID')


def network_depthwise_conv2d_native():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 2], name="filter")

    return tf.nn.depthwise_conv2d_native(input_, filter_, [1, 1, 1, 1], 'VALID')


def network_separable_conv2d():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    depthwise_filter = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 2], name="dw_filter")
    pointwise_filter = tf.get_variable(dtype=tf.float32, shape=[1, 1, 6, 7], name="pw_filter")

    return tf.nn.separable_conv2d(input_, depthwise_filter, pointwise_filter, [1, 1, 1, 1], 'VALID')


def network_atrous_conv2d_1():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

    return tf.nn.atrous_conv2d(input_, filter_, rate=1, padding='VALID')


def network_atrous_conv2d_2():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

    return tf.nn.atrous_conv2d(input_, filter_, rate=2, padding='VALID')


def network_atrous_conv2d_transpose1():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 16, 3], name="filter")

    return tf.nn.atrous_conv2d_transpose(input_, filter_, output_shape=[1, 64, 64, 16], rate=1, padding='SAME')


def network_atrous_conv2d_transpose2():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 16, 3], name="filter")

    return tf.nn.atrous_conv2d_transpose(input_, filter_, output_shape=[1, 64, 64, 16], rate=2, padding='SAME')


def network_convolution1():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 4, 3, 16], name="filter")

    return tf.nn.convolution(input_, filter_, 'SAME')


def network_convolution2():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[5, 5, 3, 16], name="filter")

    return tf.nn.convolution(input_, filter_, padding='SAME', strides=[2, 2])


def network_convolution_generated_1():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[1, 1, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='SAME', strides=[1, 1],
                             dilation_rate=[1, 1])


def network_convolution_generated_2():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[1, 1, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='SAME', strides=[1, 1],
                             dilation_rate=[2, 2])


def network_convolution_generated_3():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[1, 1, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='VALID', strides=[1, 1],
                             dilation_rate=[1, 1])


def network_convolution_generated_4():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[1, 1, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='VALID', strides=[1, 1],
                             dilation_rate=[2, 2])


def network_convolution_generated_5():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[1, 1, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='SAME', strides=[2, 2],
                             dilation_rate=[1, 1])


def network_convolution_generated_6():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[1, 1, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='VALID', strides=[2, 2],
                             dilation_rate=[1, 1])


def network_convolution_generated_7():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[3, 3, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='SAME', strides=[1, 1],
                             dilation_rate=[1, 1])


def network_convolution_generated_8():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[3, 3, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='SAME', strides=[1, 1],
                             dilation_rate=[2, 2])


def network_convolution_generated_9():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[3, 3, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='VALID', strides=[1, 1],
                             dilation_rate=[1, 1])


def network_convolution_generated_10():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[3, 3, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='VALID', strides=[1, 1],
                             dilation_rate=[2, 2])


def network_convolution_generated_11():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[3, 3, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='SAME', strides=[2, 2],
                             dilation_rate=[1, 1])


def network_convolution_generated_12():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[3, 3, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='VALID', strides=[2, 2],
                             dilation_rate=[1, 1])


def network_convolution_generated_13():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[5, 5, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='SAME', strides=[1, 1],
                             dilation_rate=[1, 1])


def network_convolution_generated_14():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[5, 5, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='SAME', strides=[1, 1],
                             dilation_rate=[2, 2])


def network_convolution_generated_15():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[5, 5, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='VALID', strides=[1, 1],
                             dilation_rate=[1, 1])


def network_convolution_generated_16():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[5, 5, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='VALID', strides=[1, 1],
                             dilation_rate=[2, 2])


def network_convolution_generated_17():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[5, 5, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='SAME', strides=[2, 2],
                             dilation_rate=[1, 1])


def network_convolution_generated_18():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[5, 5, 3, 16], name="filter")
    return tf.nn.convolution(input_, filter_, padding='VALID', strides=[2, 2],
                             dilation_rate=[1, 1])


def network_pool1():
    ph = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="placeholder1")
    return tf.nn.max_pool(ph, ksize=[1, 6, 8, 1], strides=[1, 2, 3, 1], padding='VALID')


def network_pool2():
    ph = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="placeholder1")
    return tf.nn.max_pool(ph, ksize=[1, 6, 8, 1], strides=[1, 2, 3, 1], padding='SAME')


def network_pool3():
    ph = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="placeholder1")
    output, _argmax = tf.nn.max_pool_with_argmax(ph, ksize=[1, 6, 8, 1], strides=[1, 2, 3, 1], padding='SAME')
    return output


def network_pool4():
    ph = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="placeholder1")
    output, argmax = tf.nn.max_pool_with_argmax(ph, ksize=[1, 6, 8, 1], strides=[1, 2, 3, 1], padding='SAME')
    return output, argmax


def network_pool5():
    ph = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="placeholder1")
    return tf.nn.avg_pool(ph, ksize=[1, 6, 8, 1], strides=[1, 2, 3, 1], padding='SAME')


def network_activation_elu():
    ph = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="placeholder1")
    return tf.nn.elu(ph)


def network_activation_relu():
    ph = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="placeholder1")
    return tf.nn.relu(ph)


def network_activation_relu6():
    ph = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="placeholder1")
    return tf.nn.relu6(ph)


def network_leaky_relu_literal1():
    ph = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="features")
    return tf.nn.leaky_relu(ph, alpha=0.3)


def network_activation_softplus():
    ph = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="placeholder1")
    return tf.nn.softplus(ph)


def network_unary_negative():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.negative(floats)


def network_unary_abs():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.abs(floats)


def network_unary_sign():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.sign(floats)


def network_unary_exp():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.exp(floats)


def network_unary_log():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.log(floats)


def network_unary_sqrt():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.sqrt(floats)


def network_unary_rsqrt():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.rsqrt(floats)


def network_unary_square():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.square(floats)


def network_unary_floor():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.floor(floats)


def network_unary_ceil():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.ceil(floats)


def network_unary_round():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.round(floats)


def network_unary_sigmoid():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.sigmoid(floats)


def network_unary_nn_sigmoid():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.nn.sigmoid(floats)


def network_unary_tanh():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.tanh(floats)


def network_unary_nn_tanh():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf.nn.tanh(floats)


def network_unary_logical_not():
    a = tf.placeholder(tf.float32, shape=[2, 2], name="a")
    b = tf.placeholder(tf.float32, shape=[2, 2], name="b")

    v = tf.Variable(initial_value=np.array([[1.0, 2], [3, 4]], dtype=np.float32), name="v")
    w = tf.Variable(initial_value=np.array([[5.0, 6], [7, 8]], dtype=np.float32), name="w")

    cond = tf.logical_not(tf.equal(a, b))
    return tf.where(cond, v, w)


def network_unary_ops_logical_not():
    a = tf.placeholder(tf.float32, shape=[2, 2], name="a")
    b = tf.placeholder(tf.float32, shape=[2, 2], name="b")

    v = tf.Variable(initial_value=np.array([[1.0, 2], [3, 4]], dtype=np.float32), name="v")
    w = tf.Variable(initial_value=np.array([[5.0, 6], [7, 8]], dtype=np.float32), name="w")

    cond = tf_internal.logical_not(tf.equal(a, b))
    return tf.where(cond, v, w)


def network_unary_ops_reciprocal():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf_internal.reciprocal(floats)


def network_unary_ops_sign():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf_internal.sign(floats)


def network_unary_ops_exp():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf_internal.exp(floats)


def network_unary_ops_log():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf_internal.log(floats)


def network_unary_ops_square():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf_internal.square(floats)


def network_unary_ops_floor():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf_internal.floor(floats)


def network_unary_ops_ceil():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf_internal.ceil(floats)


def network_unary_ops_round():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf_internal.round(floats)


def network_unary_ops_sqrt():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf_internal.sqrt(floats)


def network_unary_ops_rsqrt():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf_internal.rsqrt(floats)


def network_unary_ops__abs():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf_internal.abs(floats)


def network_unary_ops_neg():
    floats = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="floats")
    return tf_internal.neg(floats)


def network_binary_add():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="b")
    return tf.add(a, b)


def network_binary_subtract():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="b")
    return tf.subtract(a, b)


def network_binary_multiply():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="b")
    return tf.multiply(a, b)


def network_binary_divide():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="b")
    return tf.divide(a, b)


def network_binary_pow():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="b")
    return tf.pow(a, b)


def network_binary_squared_difference():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="b")
    return tf.squared_difference(a, b)


def network_binary_minimum():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="b")
    return tf.minimum(a, b)


def network_binary_maximum():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="b")
    return tf.maximum(a, b)


def network_binary2():
    a = tf.placeholder(tf.float32, shape=[2, 2], name="a")
    b = tf.placeholder(tf.float32, shape=[2, 2], name="b")
    c = tf.placeholder(tf.float32, shape=[2, 2], name="c")

    cond1 = tf.logical_and(tf.equal(a, b), tf.equal(b, c))
    cond2 = tf.logical_or(tf.equal(a, b), tf.equal(b, c))

    cond3 = tf.greater(a, b)
    cond4 = tf.greater_equal(a, b)
    cond5 = tf.less(a, b)
    cond6 = tf.less_equal(a, b)
    cond7 = tf.equal(a, b)
    cond8 = tf.not_equal(a, b)

    v = tf.Variable(initial_value=np.array([[1.0, 2], [3, 4]], dtype=np.float32), name="v")
    w = tf.Variable(initial_value=np.array([[5.0, 6], [7, 8]], dtype=np.float32), name="w")

    return (
        tf.where(cond1, v, w),
        tf.where(cond2, v, w),
        tf.where(cond3, v, w),
        tf.where(cond4, v, w),
        tf.where(cond5, v, w),
        tf.where(cond6, v, w),
        tf.where(cond7, v, w),
        tf.where(cond8, v, w)
    )


def network_binary_ops_add():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="b")
    return tf_internal.add(a, b)


def network_binary_ops_div():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="b")
    return tf_internal.div(a, b)


def network_binary_ops__pow():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="b")
    return tf_internal.pow(a, b)


def network_binary4():
    a = tf.placeholder(tf.float32, shape=[2, 2], name="a")
    b = tf.placeholder(tf.float32, shape=[2, 2], name="b")
    c = tf.placeholder(tf.float32, shape=[2, 2], name="c")

    cond1 = tf_internal.logical_and(tf.equal(a, b), tf.equal(b, c))
    cond2 = tf_internal.logical_or(tf.equal(a, b), tf.equal(b, c))

    cond3 = tf_internal.greater(a, b)
    cond4 = tf_internal.greater_equal(a, b)
    cond5 = tf_internal.less(a, b)
    cond6 = tf_internal.less_equal(a, b)
    cond7 = tf_internal.equal(a, b)
    cond8 = tf_internal.not_equal(a, b)

    v = tf.Variable(initial_value=np.array([[1.0, 2], [3, 4]], dtype=np.float32), name="v")
    w = tf.Variable(initial_value=np.array([[5.0, 6], [7, 8]], dtype=np.float32), name="w")

    return (
        tf.where(cond1, v, w),
        tf.where(cond2, v, w),
        tf.where(cond3, v, w),
        tf.where(cond4, v, w),
        tf.where(cond5, v, w),
        tf.where(cond6, v, w),
        tf.where(cond7, v, w),
        tf.where(cond8, v, w)
    )


def network_binary_ops_sub():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="b")
    return tf_internal.sub(a, b)


def network_binary_ops_mul():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="b")
    return tf_internal.mul(a, b)


def network_binary_ops_real_div():
    a = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="b")
    return tf_internal.real_div(a, b)


def network_where1():
    a = tf.placeholder(tf.float32, shape=[2, 2], name="a")
    b = tf.placeholder(tf.float32, shape=[2, 2], name="b")

    v = tf.Variable(initial_value=np.array([[1.0, 2], [3, 4]], dtype=np.float32), name="v")
    w = tf.Variable(initial_value=np.array([[5.0, 6], [7, 8]], dtype=np.float32), name="w")

    return tf.where(tf.equal(a, b), v, w)


def network_reduce_mean1():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.reduce_mean(input_tensor, keep_dims=True)


def network_reduce_mean2():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.reduce_mean(input_tensor, axis=[0], keep_dims=True)


def network_reduce_mean3():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.reduce_mean(input_tensor, axis=[-4, -3], keep_dims=True)


def network_reduce_mean4():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.reduce_mean(input_tensor, keepdims=True)


def network_reduce_sum1():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.reduce_sum(input_tensor, keep_dims=True)


def network_reduce_sum2():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.reduce_sum(input_tensor, axis=[0], keep_dims=True)


def network_reduce_sum3():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.reduce_sum(input_tensor, axis=[-4, -3], keep_dims=True)


def network_reduce_min():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64], name="input_tensor")
    return tf.reduce_min(input_tensor, keepdims=True)


def network_argmax():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.argmax(input_tensor, axis=0)


def network_argmin():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.argmin(input_tensor, axis=1)


def network_reduce_max1():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64], name="input_tensor")
    return tf.reduce_max(input_tensor, keep_dims=True)


def network_reduce_max2():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64], name="input_tensor")
    return tf.reduce_max(input_tensor, keep_dims=False)


def network_reduce_max3():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.reduce_max(input_tensor, axis=[0], keep_dims=True)


def network_reduce_max4():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.reduce_max(input_tensor, axis=[0], keep_dims=False)


def network_reduce_max5():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.reduce_max(input_tensor, axis=[-3, -2], keep_dims=True)


def network_reduce_max6():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.reduce_max(input_tensor, axis=[-3, -2], keep_dims=False)


def network_reduce_max7():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.reduce_max(input_tensor, axis=[1], keep_dims=True)


def network_reduce_max8():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")
    return tf.reduce_max(input_tensor, axis=[1], keep_dims=False)


def network_lrn1():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")

    return (
        tf.nn.lrn(input_tensor, 5, 1.1, 1.1, 0.6),
        tf.nn.local_response_normalization(input_tensor, 5, 1.1, 1.1, 0.6)
    )


def network_lrn2():
    input_tensor = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input_tensor")

    return (
        tf.nn.lrn(input_tensor),
        tf.nn.local_response_normalization(input_tensor)
    )


def network_batch_normalization1():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")
    offset = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="offset")
    scale = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="scale")
    variance_epsilon = 0.01

    mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)

    return tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon)


def network_batch_normalization2():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")
    offset = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="offset")
    scale = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="scale")
    variance_epsilon = 0.01

    mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=False)

    return tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon)


def network_l2_normalization1():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")

    return tf.nn.l2_normalize(x, axis=0)


def network_l2_normalization2():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")

    return tf.nn.l2_normalize(x, dim=0)


def network_l2_normalization3():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")

    return tf.nn.l2_normalize(x, axis=[0, 1])


def network_l2_normalization4():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")

    return tf.nn.l2_normalize(x, dim=[0, 1])


def network_l2_normalization5():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")

    return tf.nn.l2_normalize(x, axis=[0, 1], epsilon=0.001)


def network_matmul1():
    a = tf.placeholder(tf.float32, shape=[10, 20], name="a")
    b = tf.placeholder(tf.float32, shape=[20, 30], name="b")

    return tf.matmul(a, b)


def network_matmul2():
    a = tf.placeholder(tf.float32, shape=[20, 10], name="a")
    b = tf.placeholder(tf.float32, shape=[20, 30], name="b")

    return tf.matmul(a, b, transpose_a=True)


def network_matmul3():
    a = tf.placeholder(tf.float32, shape=[10, 20], name="a")
    b = tf.placeholder(tf.float32, shape=[30, 20], name="b")

    return tf.matmul(a, b, transpose_b=True)


def network_matmul4():
    a = tf.placeholder(tf.float32, shape=[20, 10], name="a")
    b = tf.placeholder(tf.float32, shape=[30, 20], name="b")

    return tf.matmul(a, b, transpose_a=True, transpose_b=True)


def network_assign1():
    var1 = tf.get_variable(shape=[2, 2], dtype=tf.float32, name="var1")
    placeholder1 = tf.placeholder(tf.float32, shape=[2, 2], name="placeholder1")

    return tf.assign(var1, placeholder1)


def network_add_n1():
    ph = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="ph")
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")

    return ph + tf.add_n([x])


def network_add_n2():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")
    y = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="y")
    z = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="z")

    return tf.add_n([x, y, z])


def network_bias_add1():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")
    bias = tf.placeholder(tf.float32, shape=[3], name="bias")

    return tf.nn.bias_add(x, bias)


def network_concat1():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")
    y = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="y")

    return tf.concat([x, y], 3)


def network_concat2():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64], name="x")
    y = tf.placeholder(tf.float32, shape=[10, 64, 64], name="y")

    return tf.concat([x, y], -3)


def network_concat3():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64], name="x")

    return tf.concat([x], 0)


def network_split1():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")

    return tf.split(x, 3)


def network_split2():
    x = tf.placeholder(tf.float32, shape=[6, 12, 64, 3], name="x")

    return tf.split(x, [2, 4, 6], axis=1)


def network_split3():
    x = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="x")
    a, b = tf.split(x, 2)
    return a


def network_softmax1():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")

    return tf.nn.softmax(x)


def network_softmax2_old():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")

    return tf.nn.softmax(x, dim=1)


def network_softmax2():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")

    return tf.nn.softmax(x, axis=1)


def network_softmax3():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")

    return tf_layers.softmax(x)


def network_moments1():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")

    return tf.nn.moments(x, [1, 2])


def network_reshape1():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")

    return tf.reshape(x, x.shape)


def network_reshape2():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")

    return tf.reshape(x, [73728])


def network_reshape3():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")

    return tf.reshape(x, [4096, 18])


def network_reshape4():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")

    return tf.reshape(x, [6, 4096, 3])


def network_flatten1():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")

    return tf_layers.flatten(x)


def network_expand_dims1():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")

    return tf.expand_dims(x, axis=4)


def network_expand_dims2():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")

    return tf.expand_dims(x, dim=0)


def network_squeeze1():
    x = tf.placeholder(tf.float32, shape=[1, 2, 1, 2], name="x")

    return tf.squeeze(x)


def network_squeeze2():
    x = tf.placeholder(tf.float32, shape=[1, 2, 1, 2], name="x")
    return tf.squeeze(x, [2])


def network_squeeze3():
    x = tf.placeholder(tf.float32, shape=[10, 1, 1, 1], name="x")

    return -tf.squeeze(x, [1, 2, 3])


def network_squeeze4():
    x = tf.placeholder(tf.float32, shape=[10, 1, 1, 1], name="x")

    return -tf.squeeze(x, [1, 2])


def network_transpose1():
    x = tf.placeholder(tf.float32, shape=[10, 20], name="x")

    return tf.transpose(x)


def network_transpose2():
    x = tf.placeholder(tf.float32, shape=[1, 2, 3, 4], name="x")

    return tf.transpose(x)


def network_transpose3():
    x = tf.placeholder(tf.float32, shape=[1, 2, 3, 4], name="x")

    return tf.transpose(x, [3, 0, 1, 2])


def network_resize_images_nn_down():
    images = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="images")

    return tf.image.resize_images(images, [16, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def network_resize_images_area_down():
    images = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="images")

    return tf.image.resize_images(images, [16, 16], method=tf.image.ResizeMethod.AREA)


def network_resize_images_nn_up():
    images = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="images")

    return tf.image.resize_images(images, [64, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def network_resize_images_bilinear_up():
    images = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="images")

    return tf.image.resize_images(images, [64, 64], method=tf.image.ResizeMethod.BILINEAR)


def network_resize_images_bilinear_same():
    images = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="images")

    return tf.image.resize_images(-images, [32, 32], method=tf.image.ResizeMethod.BILINEAR)


def network_resize_bilinear1():
    images = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="images")

    return tf.image.resize_bilinear(images, [64, 64])


def network_resize_bilinear2():
    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

    conv1 = tf.nn.conv2d(input_, filter_, strides=[1, 1, 1, 1], padding='SAME')

    input2_ = tf.image.resize_bilinear(conv1, [128, 128])

    filter2_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 16, 24], name="filter2")

    conv2 = tf.nn.conv2d(input2_, filter2_, strides=[1, 1, 1, 1], padding='VALID')

    return conv2


def network_resize_nearest1():
    images = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="images")

    return tf.image.resize_nearest_neighbor(images, [64, 64])


def network_resize_area1():
    images = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="images")

    return tf.image.resize_area(images, [16, 16])


def network_passthrough_identity():
    x = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="x")

    return -tf.identity(x)


def network_passthrough_stop_gradient():
    x = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="x")

    return -tf.stop_gradient(x)


def network_passthrough_cast():
    x = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="x")

    return -tf.cast(x, tf.float64)


def network_passthrough_dropout():
    x = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="x")

    return -tf.nn.dropout(x, 0.5)


def network_passthrough_as_result():
    x = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="x")

    return tf.identity(-x)


def network_cast_bool_to_float():
    x = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="x")
    y = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="y")

    smaller = x < y

    cast = tf.cast(smaller, tf.float32)

    return tf.reduce_sum(cast)


def network_cast_float_to_bool():
    x = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="x")
    y = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="y")

    cond = tf.cast(x, tf.bool)

    return tf.where(cond, x, y)


def network_add_conv_transform1():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")

    return tf_slim.conv2d(input_, num_outputs=4, kernel_size=[4, 4], rate=4, padding='SAME',
                          biases_initializer=tf.random_normal_initializer)  # just to test with nonzero value


def network_add_conv_transform2():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")

    return tf_slim.conv2d(input_, num_outputs=4, kernel_size=[4, 4], rate=4, padding='VALID',
                          biases_initializer=tf.random_normal_initializer)


def network_add_conv_transform3():
    input_ = tf.placeholder(tf.float32, shape=[10, 16, 16, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 8], name="filter")

    conv = tf.nn.conv2d(input=input_, filter=filter_, strides=[1, 1, 1, 1], padding='VALID')
    bias = tf.placeholder(tf.float32, shape=[8], name="bias")

    return tf.nn.bias_add(conv, bias)


def network_pad_transform_pool_constant():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")

    padded_input = tf.pad(input_, paddings=[[1, 1], [2, 2], [3, 3], [4, 4]], mode='CONSTANT')
    return tf.nn.max_pool(padded_input, ksize=[1, 6, 8, 1], strides=[1, 2, 3, 1], padding='VALID')


def network_pad_transform_conv_constant():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

    padded_input = tf.pad(input_, paddings=[[0, 0], [2, 2], [3, 3], [0, 0]], mode='CONSTANT')
    return tf.nn.conv2d(padded_input, filter_, [1, 1, 1, 1], 'VALID')


def network_pad_transform_conv_symmetric():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

    padded_input = tf.pad(input_, paddings=[[0, 0], [2, 2], [3, 3], [0, 0]], mode='SYMMETRIC')
    return tf.nn.conv2d(padded_input, filter_, [1, 1, 1, 1], 'VALID')


def network_pad_transform_conv_reflect():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

    padded_input = tf.pad(input_, paddings=[[0, 0], [2, 2], [3, 3], [0, 0]], mode='REFLECT')
    return tf.nn.conv2d(padded_input, filter_, [1, 1, 1, 1], 'VALID')


def network_fused_batch_norm_transform1():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")
    scale = tf.placeholder(tf.float32, shape=[3], name="scale")
    offset = tf.placeholder(tf.float32, shape=[3], name="offset")

    y, batch_mean, batch_var = tf.nn.fused_batch_norm(x, scale, offset)
    return y, batch_mean, batch_var


def network_fused_batch_norm_transform2():
    x = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="x")
    scale = tf.placeholder(tf.float32, shape=[3], name="scale")
    offset = tf.placeholder(tf.float32, shape=[3], name="offset")

    mean, variance = tf.nn.moments(x, axes=[0, 1, 2])

    # probably we should not use _batch_mean, _batch_var if not training
    y, _batch_mean, _batch_var = tf.nn.fused_batch_norm(x, scale, offset, mean=mean, variance=variance,
                                                        is_training=False)
    return y


def network_fused_batch_norm_transform3():
    x = tf.placeholder(tf.float32, shape=[10, 3, 64, 64], name="x")
    scale = tf.placeholder(tf.float32, shape=[3], name="scale")
    offset = tf.placeholder(tf.float32, shape=[3], name="offset")

    y, batch_mean, batch_var = tf.nn.fused_batch_norm(x, scale, offset, data_format='NCHW')
    return y, batch_mean, batch_var


def network_complex1():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")
    bias = tf.get_variable(dtype=tf.float32, shape=[16], name="bias")
    padded_input = tf.pad(input_, paddings=[[0, 0], [2, 2], [3, 3], [0, 0]])
    return tf.nn.conv2d(padded_input, filter_, [1, 1, 1, 1], 'VALID') + bias


def network_filter_matmul1():
    input_ = tf.placeholder(tf.float32, shape=[2, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")
    conv = tf.nn.conv2d(input_, filter_, [1, 1, 1, 1], 'VALID')
    flat = tf_layers.flatten(conv)
    weights = tf.get_variable(dtype=tf.float32, shape=[59536, 30], name="weights")
    return tf.matmul(flat, weights)


def network_filter_matmul2():
    input_ = tf.placeholder(tf.float32, shape=[2, 3, 64, 64], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")
    conv = tf.nn.conv2d(input_, filter_, [1, 1, 1, 1], 'VALID', data_format="NCHW")
    flat = tf.reshape(conv, [2, -1])
    weights = tf.get_variable(dtype=tf.float32, shape=[59536, 30], name="weights")
    return tf.matmul(flat, weights)


def network_filter_matmul3():
    input_ = tf.placeholder(tf.float32, shape=[2, 3, 64, 64], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")
    conv = tf.nn.conv2d(input_, filter_, [1, 1, 1, 1], 'VALID', data_format="NCHW")
    relu = tf.nn.relu(conv)
    flat = tf_layers.flatten(relu)
    weights = tf.get_variable(dtype=tf.float32, shape=[59536, 30], name="weights")
    return tf.matmul(flat, weights)


def network_filter_matmul4():
    input_ = tf.placeholder(tf.float32, shape=[2, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")
    conv = tf.nn.conv2d(input_, filter_, [1, 1, 1, 1], 'SAME')
    flat = tf.reshape(conv, [-1, 65536])
    weights = tf.get_variable(dtype=tf.float32, shape=[65536, 30], name="weights")
    return tf.matmul(flat, weights)


def network_format_rank1():
    ph = tf.placeholder(tf.float32, shape=[1], name="ph")
    var = tf.get_variable('var1', shape=[1])
    return var + ph


def network_format_rank2():
    ph = tf.placeholder(tf.float32, shape=[1, 2], name="ph")
    var = tf.get_variable('var1', shape=[1, 2])
    return var + ph


def network_format_rank3():
    ph = tf.placeholder(tf.float32, shape=[1, 2, 3], name="ph")
    var = tf.get_variable('var1', shape=[1, 2, 3])
    return var + ph


def network_format_rank4():
    ph = tf.placeholder(tf.float32, shape=[1, 2, 3, 4], name="ph")
    var = tf.get_variable('var1', shape=[1, 2, 3, 4])
    return var + ph


def network_optimizer_conv_input_nhwc_filter_hwcn():
    input_ = tf.placeholder(tf.float32, shape=[1, 2, 3, 4], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[1, 1, 4, 16], name="filter")

    return tf.nn.conv2d(input_, filter_, [1, 1, 1, 1], 'VALID', data_format='NHWC')


def network_optimizer_conv_input_nchw_filter_hwcn():
    input_ = tf.placeholder(tf.float32, shape=[1, 2, 3, 4], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[1, 1, 2, 16], name="filter")

    return tf.nn.conv2d(input_, filter_, [1, 1, 1, 1], 'VALID', data_format='NCHW')


def network_optimizer_conv_input_nchw_filter_hwcm():
    input_ = tf.placeholder(tf.float32, shape=[1, 2, 3, 4], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[1, 1, 2, 5], name="filter")

    return tf.nn.depthwise_conv2d(input_, filter_, [1, 1, 1, 1], 'VALID', data_format='NCHW')


def network_optimizer_conv_input_nchw_filter_hwcn_bias_c():
    input_ = tf.placeholder(tf.float32, shape=[1, 2, 3, 4], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[1, 1, 2, 16], name="filter")

    conv = tf.nn.conv2d(input_, filter_, [1, 1, 1, 1], 'VALID', data_format='NCHW')

    # test case: bias is after conv
    bias = tf.placeholder(tf.float32, shape=[16], name="bias")

    return tf.nn.bias_add(conv, bias, data_format='NCHW')


def network_optimizer_conv_input_nchw_filter_const_hwcn():
    input_ = tf.placeholder(tf.float32, shape=[1, 2, 3, 4], name="input")
    filter_ = tf.constant(np.array([[[
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [100, 200, 300, 400, 500],
        [1000, 2000, 3000, 4000, 5000]]]], dtype=np.float32))

    return tf.nn.conv2d(input_, filter_, [1, 1, 1, 1], 'VALID', data_format='NHWC')


def network_optimizer_max_pool():
    ph = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="placeholder1")
    return tf.nn.max_pool(ph, ksize=[1, 6, 6, 1], strides=[1, 2, 3, 1], padding='VALID')


def network_optimizer_max_pool_nchw():
    ph = tf.placeholder(tf.float32, shape=[10, 3, 64, 64], name="placeholder1")
    return tf.nn.max_pool(ph, ksize=[1, 1, 6, 6], strides=[1, 1, 2, 3], padding='VALID', data_format='NCHW')


def network_optimizer_conv_nchw_hwcn_padding_dilations():
    input_ = tf.placeholder(tf.float32, shape=[1, 3, 64, 64], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")

    padded_input = tf.pad(input_, paddings=[[0, 0], [0, 0], [2, 2], [3, 3]])
    return tf.nn.conv2d(padded_input, filter_, strides=[1, 1, 1, 1], dilations=[1, 1, 3, 3],
                        padding='VALID', data_format='NCHW')


def network_optimizer_conv_transpose_nchw_hwcn():
    input_ = tf.placeholder(tf.float32, shape=[1, 3, 64, 64], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 16, 3], name="filter")

    return tf.nn.conv2d_transpose(input_, filter_, output_shape=[1, 16, 64, 64], strides=[1, 1, 1, 1],
                                  padding='SAME', data_format='NCHW')


def network_returning_input():
    return tf.placeholder(tf.float32, shape=[1, 2, 3, 1], name="placeholder1")


def network_clip_by_value1():
    input_ = tf.placeholder(tf.float32, shape=[2, 3, 64, 64], name="input")
    return tf.clip_by_value(input_, 2.5, 2.6)


def network_clip_by_value2():
    input_ = tf.placeholder(tf.float32, shape=[2, 3, 64, 64], name="input")
    return tf.clip_by_value(input_, 2.5, 3)


def network_clip_by_value3():
    input_ = tf.placeholder(tf.float32, shape=[2, 3, 64, 64], name="input")
    a = tf.placeholder(tf.float32, shape=[2, 3, 64, 64], name="a")
    b = tf.placeholder(tf.float32, shape=[2, 3, 64, 64], name="b")
    return tf.clip_by_value(input_, a, b)


def network_zeros():
    input_ = tf.placeholder(tf.float32, shape=[2, 64, 64, 3], name="input")
    return input_ + tf.zeros([2, 64, 64, 3])


def network_zeros_like():
    input_ = tf.placeholder(tf.float32, shape=[2, 64, 64, 3], name="input")
    return input_ + tf.zeros_like(input_)


def network_ones():
    input_ = tf.placeholder(tf.float32, shape=[2, 64, 64, 3], name="input")
    return input_ + tf.ones([2, 64, 64, 3])


def network_ones_like():
    input_ = tf.placeholder(tf.float32, shape=[2, 64, 64, 3], name="input")
    return input_ + tf.ones_like(input_)


def network_slice1():
    input_ = tf.placeholder(tf.float32, shape=[2, 64, 64, 3], name="input")
    return tf.slice(input_, begin=[0, 20, 20, 1], size=[1, 30, 30, 2])


def network_slice2():
    input_ = tf.placeholder(tf.float32, shape=[2, 64, 64, 3], name="input")
    return tf.slice(input_, begin=[0, 20, 20, 1], size=[-1, 30, 30, -1])


def network_stack1():
    a = tf.placeholder(tf.float32, shape=[2, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[2, 64, 64, 3], name="b")
    c = tf.placeholder(tf.float32, shape=[2, 64, 64, 3], name="c")
    return tf.stack([a, b, c])


def network_stack2():
    a = tf.placeholder(tf.float32, shape=[2, 64, 64, 3], name="a")
    b = tf.placeholder(tf.float32, shape=[2, 64, 64, 3], name="b")
    c = tf.placeholder(tf.float32, shape=[2, 64, 64, 3], name="c")
    return tf.stack([a, b, c], axis=-1)


def network_unstack1():
    a = tf.placeholder(tf.float32, shape=[3, 2, 64, 64, 3], name="a")
    return tf.unstack(a)


def network_unstack2():
    a = tf.placeholder(tf.float32, shape=[2, 64, 64, 3, 3], name="a")
    return tf.unstack(a, axis=3)


def network_norm_euclidean():
    x = tf.placeholder(tf.float32, shape=[1, 2, 3], name='x')
    return tf.norm(x)


def network_norm_fro():
    x = tf.placeholder(tf.float32, shape=[2, 2], name='x')
    return tf.norm(x, axis=(0, 1), ord='fro')


def network_norm_1():
    x = tf.placeholder(tf.float32, shape=[1, 2, 3], name='x')
    return tf.norm(x, ord=1)


def network_norm_2():
    x = tf.placeholder(tf.float32, shape=[1, 2, 3], name='x')
    return tf.norm(x, ord=2)


def network_norm_inf():
    x = tf.placeholder(tf.float32, shape=[1, 2, 3], name='x')
    return tf.norm(x, ord=np.inf)


def network_output_shape1():
    input_ = tf.placeholder(tf.float32, shape=[1, 32, 32, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 16, 3], name="filter")

    return tf.nn.conv2d_transpose(input_, filter_, output_shape=[1, 64, 64, 16], strides=[1, 2, 2, 1], padding='SAME')


def network_shape_of__output_shape1():
    input_ = tf.placeholder(tf.float32, shape=[1, 32, 32, 3], name="input1")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 16, 3], name="filter")
    other_tensor = tf.placeholder(tf.float32, shape=[1, 64, 64, 16], name="input2")
    transposed = tf.nn.conv2d_transpose(input_, filter_, output_shape=tf.shape(other_tensor), strides=[1, 2, 2, 1],
                                        padding='SAME')

    return transposed, -other_tensor


def network_shape_of__output_shape2():
    input_ = tf.placeholder(tf.float32, shape=[1, 3, 32, 32], name="input1")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 16, 3], name="filter")
    other_tensor = tf.placeholder(tf.float32, shape=[1, 16, 64, 64], name="input2")
    transposed = tf.nn.conv2d_transpose(input_, filter_, output_shape=tf.shape(other_tensor), strides=[1, 1, 2, 2],
                                        padding='SAME', data_format="NCHW")

    return transposed, -other_tensor


def network_shape_of__reshape():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")
    y = tf.placeholder(tf.float32, shape=[6, 64 * 64 * 3], name="y")

    return tf.reshape(x, tf.shape(y)) + y


def network_no_shape_of__output_shape1():
    input_ = tf.placeholder(tf.float32, shape=[1, 32, 32, 3], name="input1")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 16, 3], name="filter")
    other_tensor = tf.placeholder(tf.float32, shape=[1, 64, 64, 16], name="input2")
    transposed = tf.nn.conv2d_transpose(input_, filter_, output_shape=tf.shape(other_tensor), strides=[1, 2, 2, 1],
                                        padding='SAME')

    return transposed, -other_tensor


def network_no_shape_of__output_shape2():
    input_ = tf.placeholder(tf.float32, shape=[1, 3, 32, 32], name="input1")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 16, 3], name="filter")
    other_tensor = tf.placeholder(tf.float32, shape=[1, 16, 64, 64], name="input2")
    transposed = tf.nn.conv2d_transpose(input_, filter_, output_shape=tf.shape(other_tensor), strides=[1, 1, 2, 2],
                                        padding='SAME', data_format="NCHW")

    return transposed, -other_tensor


def network_no_shape_of__reshape():
    x = tf.placeholder(tf.float32, shape=[6, 64, 64, 3], name="x")
    y = tf.placeholder(tf.float32, shape=[6, 64 * 64 * 3], name="y")

    return tf.reshape(x, tf.shape(y)) + y


def network_range1():
    x = tf.placeholder(tf.float32, shape=[12], name="x")
    y = tf.range(12.0)
    return x + y


def network_range2():
    x = tf.placeholder(tf.float32, shape=[12], name="x")
    y = tf.range(0, 6, delta=0.5)
    return x + y


def network_range3():
    x = tf.placeholder(tf.int32, shape=[12], name="x")
    y = tf.range(12, dtype=tf.int32)
    return tf.concat([x, y], axis=0)


def network_pad():
    input_shape = [1, 224, 224, 3]
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=input_shape)
    return tf.pad(input_, paddings=[[0, 0], [5, 10], [5, 10], [0, 0]])


def network_tile1():
    input_shape = [1, 32, 32, 1]
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=input_shape)
    return tf.tile(input_, multiples=[5, 1, 1, 4])


def network_tile2():
    input_shape = [1, 32, 32, 3]
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=input_shape)
    return tf.tile(input_, multiples=[1, 3, 2, 2])


def network_tile3():
    input_shape = [1, 32, 32, 3]
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=input_shape)
    return tf.tile(input_, multiples=[5, 3, 2, 2])


def network_resize_bicubic1():
    images = tf.placeholder(tf.float32, shape=[100, 32, 32, 3], name="images")

    return tf.image.resize_bicubic(images, [64, 64])


class TFPyLayerTestCases(TFPyTestRunner):

    def test_naming(self):
        self._test(network_naming)

    def test_naming2(self):
        self._test(network_naming2)

    def test_naming3(self):
        self._test(network_naming3)

    def test_naming4(self):
        self._test(network_naming4)

    def test_my_reshape(self):
        self._test(network_my_reshape)

    def test_float(self):
        self._test(network_float)

    def test_meshgrid(self):
        self._test(network_meshgrid)

    def test_name_starting_with_num(self):
        self._test(network_name_starting_with_num)

    def test_conv_grad_input(self):
        self._test(network_conv_grad_input)

    def test_optimizer_no_io_transpose(self):
        self._test(network_optimizer_no_io_transpose)

    def test_optimizer_manual_output_transpose(self):
        self._test(network_optimizer_manual_output_transpose)

    def test_optimizer1(self):
        self._test(network_optimizer1)

    def test_optimizer2(self):
        self._test(network_optimizer2)

    def test_optimizer3(self):
        self._test(network_optimizer3)

    def test_optimizer3b(self):
        self._test(network_optimizer3b)

    def test_optimizer3c(self):
        self._test(network_optimizer3c)

    def test_optimizer3d(self):
        self._test(network_optimizer3d)

    def test_optimizer4(self):
        self._test(network_optimizer4)

    def test_optimizer5(self):
        self._test(network_optimizer5)

    def test_optimizer6(self):
        self._test(network_optimizer6)

    def test_optimizer7(self):
        self._test(network_optimizer7)

    def test_optimizer8(self):
        self._test(network_optimizer8)

    def test_optimizer9(self):
        self._test(network_optimizer9)

    def test_optimizer_broadcast1(self):
        self._test(network_optimizer_broadcast1)

    def test_optimizer10(self):
        self._test(network_optimizer10)

    def test_optimizer11(self):
        self._test(network_optimizer11)

    def test_dtype_int32(self):
        self._test(network_dtype_int32)

    def test_dtype_float64(self):
        self._test(network_dtype_float64)

    def test_dtype_bool(self):
        self._test(network_dtype_bool)

    def test_for_gradients1(self):
        self._test(network_for_gradients1)

    def test_for_gradients2(self):
        self._test(network_for_gradients2)

    def test_for_gradients3(self):
        self._test(network_for_gradients3)

    def test_for_gradients4(self):
        self._test(network_for_gradients4)

    def test_for_gradients5(self):
        self._test(network_for_gradients5)

    def test_strided_slice1(self):
        self._test(network_strided_slice1)

    def test_strided_slice2(self):
        self._test(network_strided_slice2)

    def test_strided_slice3(self):
        self._test(network_strided_slice3)

    def test_strided_slice4(self):
        self._test(network_strided_slice4)

    def test_strided_slice5(self):
        self._test(network_strided_slice5)

    def test_strided_slice6(self):
        self._test(network_strided_slice6)

    def test_placeholder1(self):
        self._test(network_placeholder1)

    def test_placeholder2(self):
        self._test(network_placeholder2)

    def test_variable1(self):
        self._test(network_variable1)

    def test_variable2(self):
        self._test(network_variable2)

    def test_variable3(self):
        self._test(network_variable3)

    def test_variable4(self):
        self._test(network_variable4)

    def test_variable_reuse(self):
        self._test(network_variable_reuse)

    def test_constant1(self):
        self._test(network_constant1)

    def test_constant2(self):
        self._test(network_constant2)

    def test_constant3(self):
        self._test(network_constant3)

    def test_conv1d(self):
        self._test(network_conv1d)

    def test_conv2d(self):
        self._test(network_conv2d)

    def test_conv3d(self):
        self._test(network_conv3d)

    def test_conv2d_transpose(self):
        self._test(network_conv2d_transpose)

    def test_conv3d_transpose(self):
        self._test(network_conv3d_transpose)

    def test_depthwise_conv2d(self):
        self._test(network_depthwise_conv2d)

    def test_depthwise_conv2d_native(self):
        self._test(network_depthwise_conv2d_native)

    def test_separable_conv2d(self):
        self._test(network_separable_conv2d)

    def test_atrous_conv2d_1(self):
        self._test(network_atrous_conv2d_1)

    def test_atrous_conv2d_2(self):
        self._test(network_atrous_conv2d_2)

    def test_atrous_conv2d_transpose1(self):
        self._test(network_atrous_conv2d_transpose1)

    def test_atrous_conv2d_transpose2(self):
        self._test(network_atrous_conv2d_transpose2)

    def test_convolution1(self):
        self._test(network_convolution1)

    def test_convolution2(self):
        self._test(network_convolution2)

    def test_convolution_generated_1(self):
        self._test(network_convolution_generated_1)

    def test_convolution_generated_2(self):
        self._test(network_convolution_generated_2)

    def test_convolution_generated_3(self):
        self._test(network_convolution_generated_3)

    def test_convolution_generated_4(self):
        self._test(network_convolution_generated_4)

    def test_convolution_generated_5(self):
        self._test(network_convolution_generated_5)

    def test_convolution_generated_6(self):
        self._test(network_convolution_generated_6)

    def test_convolution_generated_7(self):
        self._test(network_convolution_generated_7)

    def test_convolution_generated_8(self):
        self._test(network_convolution_generated_8)

    def test_convolution_generated_9(self):
        self._test(network_convolution_generated_9)

    def test_convolution_generated_10(self):
        self._test(network_convolution_generated_10)

    def test_convolution_generated_11(self):
        self._test(network_convolution_generated_11)

    def test_convolution_generated_12(self):
        self._test(network_convolution_generated_12)

    def test_convolution_generated_13(self):
        self._test(network_convolution_generated_13)

    def test_convolution_generated_14(self):
        self._test(network_convolution_generated_14)

    def test_convolution_generated_15(self):
        self._test(network_convolution_generated_15)

    def test_convolution_generated_16(self):
        self._test(network_convolution_generated_16)

    def test_convolution_generated_17(self):
        self._test(network_convolution_generated_17)

    def test_convolution_generated_18(self):
        self._test(network_convolution_generated_18)

    def test_pool1(self):
        self._test(network_pool1)

    def test_pool2(self):
        self._test(network_pool2)

    def test_pool3(self):
        self._test(network_pool3)

    def test_pool4(self):
        self._test(network_pool4)

    def test_pool5(self):
        self._test(network_pool5)

    def test_activation_elu(self):
        self._test(network_activation_elu)

    def test_activation_relu(self):
        self._test(network_activation_relu)

    def test_activation_relu6(self):
        self._test(network_activation_relu6)

    def test_leaky_relu_literal1(self):
        self._test(network_leaky_relu_literal1)

    def test_activation_softplus(self):
        self._test(network_activation_softplus)

    def test_unary_negative(self):
        self._test(network_unary_negative)

    def test_unary_abs(self):
        self._test(network_unary_abs)

    def test_unary_sign(self):
        self._test(network_unary_sign)

    def test_unary_exp(self):
        self._test(network_unary_exp)

    def test_unary_log(self):
        self._test(network_unary_log)

    def test_unary_sqrt(self):
        self._test(network_unary_sqrt)

    def test_unary_rsqrt(self):
        self._test(network_unary_rsqrt)

    def test_unary_square(self):
        self._test(network_unary_square)

    def test_unary_floor(self):
        self._test(network_unary_floor)

    def test_unary_ceil(self):
        self._test(network_unary_ceil)

    def test_unary_round(self):
        self._test(network_unary_round)

    def test_unary_sigmoid(self):
        self._test(network_unary_sigmoid)

    def test_unary_nn_sigmoid(self):
        self._test(network_unary_nn_sigmoid)

    def test_unary_tanh(self):
        self._test(network_unary_tanh)

    def test_unary_nn_tanh(self):
        self._test(network_unary_nn_tanh)

    def test_unary_logical_not(self):
        self._test(network_unary_logical_not)

    def test_unary_ops_logical_not(self):
        self._test(network_unary_ops_logical_not)

    def test_unary_ops_reciprocal(self):
        self._test(network_unary_ops_reciprocal)

    def test_unary_ops_sign(self):
        self._test(network_unary_ops_sign)

    def test_unary_ops_exp(self):
        self._test(network_unary_ops_exp)

    def test_unary_ops_log(self):
        self._test(network_unary_ops_log)

    def test_unary_ops_square(self):
        self._test(network_unary_ops_square)

    def test_unary_ops_floor(self):
        self._test(network_unary_ops_floor)

    def test_unary_ops_ceil(self):
        self._test(network_unary_ops_ceil)

    def test_unary_ops_round(self):
        self._test(network_unary_ops_round)

    def test_unary_ops_sqrt(self):
        self._test(network_unary_ops_sqrt)

    def test_unary_ops_rsqrt(self):
        self._test(network_unary_ops_rsqrt)

    def test_unary_ops__abs(self):
        self._test(network_unary_ops__abs)

    def test_unary_ops_neg(self):
        self._test(network_unary_ops_neg)

    def test_binary_add(self):
        self._test(network_binary_add)

    def test_binary_subtract(self):
        self._test(network_binary_subtract)

    def test_binary_multiply(self):
        self._test(network_binary_multiply)

    def test_binary_divide(self):
        self._test(network_binary_divide)

    def test_binary_pow(self):
        self._test(network_binary_pow)

    def test_binary_squared_difference(self):
        self._test(network_binary_squared_difference)

    def test_binary_minimum(self):
        self._test(network_binary_minimum)

    def test_binary_maximum(self):
        self._test(network_binary_maximum)

    def test_binary2(self):
        self._test(network_binary2)

    def test_binary_ops_add(self):
        self._test(network_binary_ops_add)

    def test_binary_ops_div(self):
        self._test(network_binary_ops_div)

    def test_binary_ops__pow(self):
        self._test(network_binary_ops__pow)

    def test_binary4(self):
        self._test(network_binary4)

    def test_binary_ops_sub(self):
        self._test(network_binary_ops_sub)

    def test_binary_ops_mul(self):
        self._test(network_binary_ops_mul)

    def test_binary_ops_real_div(self):
        self._test(network_binary_ops_real_div)

    def test_where1(self):
        self._test(network_where1)

    def test_reduce_mean1(self):
        self._test(network_reduce_mean1)

    def test_reduce_mean2(self):
        self._test(network_reduce_mean2)

    def test_reduce_mean3(self):
        self._test(network_reduce_mean3)

    def test_reduce_mean4(self):
        self._test(network_reduce_mean4)

    def test_reduce_sum1(self):
        self._test(network_reduce_sum1)

    def test_reduce_sum2(self):
        self._test(network_reduce_sum2)

    def test_reduce_sum3(self):
        self._test(network_reduce_sum3)

    def test_reduce_min(self):
        self._test(network_reduce_min)

    def test_argmax(self):
        self._test(network_argmax)

    def test_argmin(self):
        self._test(network_argmin)

    def test_reduce_max1(self):
        self._test(network_reduce_max1)

    def test_reduce_max2(self):
        self._test(network_reduce_max2)

    def test_reduce_max3(self):
        self._test(network_reduce_max3)

    def test_reduce_max4(self):
        self._test(network_reduce_max4)

    def test_reduce_max5(self):
        self._test(network_reduce_max5)

    def test_reduce_max6(self):
        self._test(network_reduce_max6)

    def test_reduce_max7(self):
        self._test(network_reduce_max7)

    def test_reduce_max8(self):
        self._test(network_reduce_max8)

    def test_lrn1(self):
        self._test(network_lrn1)

    def test_lrn2(self):
        self._test(network_lrn2)

    def test_batch_normalization1(self):
        self._test(network_batch_normalization1)

    def test_batch_normalization2(self):
        self._test(network_batch_normalization2)

    def test_l2_normalization1(self):
        self._test(network_l2_normalization1)

    def test_l2_normalization2(self):
        self._test(network_l2_normalization2)

    def test_l2_normalization3(self):
        self._test(network_l2_normalization3)

    def test_l2_normalization4(self):
        self._test(network_l2_normalization4)

    def test_l2_normalization5(self):
        self._test(network_l2_normalization5)

    def test_matmul1(self):
        self._test(network_matmul1)

    def test_matmul2(self):
        self._test(network_matmul2)

    def test_matmul3(self):
        self._test(network_matmul3)

    def test_matmul4(self):
        self._test(network_matmul4)

    def test_assign1(self):
        self._test(network_assign1)

    def test_add_n1(self):
        self._test(network_add_n1)

    def test_add_n2(self):
        self._test(network_add_n2)

    def test_bias_add1(self):
        self._test(network_bias_add1)

    def test_concat1(self):
        self._test(network_concat1)

    def test_concat2(self):
        self._test(network_concat2)

    def test_concat3(self):
        self._test(network_concat3)

    def test_split1(self):
        self._test(network_split1)

    def test_split2(self):
        self._test(network_split2)

    def test_split3(self):
        self._test(network_split3)

    def test_softmax1(self):
        self._test(network_softmax1)

    def test_softmax2_old(self):
        self._test(network_softmax2_old)

    def test_softmax2(self):
        self._test(network_softmax2)

    def test_softmax3(self):
        self._test(network_softmax3)

    def test_moments1(self):
        self._test(network_moments1)

    def test_reshape1(self):
        self._test(network_reshape1)

    def test_reshape2(self):
        self._test(network_reshape2)

    def test_reshape3(self):
        self._test(network_reshape3)

    def test_reshape4(self):
        self._test(network_reshape4)

    def test_flatten1(self):
        self._test(network_flatten1)

    def test_expand_dims1(self):
        self._test(network_expand_dims1)

    def test_expand_dims2(self):
        self._test(network_expand_dims2)

    def test_squeeze1(self):
        self._test(network_squeeze1)

    def test_squeeze2(self):
        self._test(network_squeeze2)

    def test_squeeze3(self):
        self._test(network_squeeze3)

    def test_squeeze4(self):
        self._test(network_squeeze4)

    def test_transpose1(self):
        self._test(network_transpose1)

    def test_transpose2(self):
        self._test(network_transpose2)

    def test_transpose3(self):
        self._test(network_transpose3)

    def test_resize_images_nn_down(self):
        self._test(network_resize_images_nn_down)

    def test_resize_images_area_down(self):
        self._test(network_resize_images_area_down)

    def test_resize_images_nn_up(self):
        self._test(network_resize_images_nn_up)

    def test_resize_images_bilinear_up(self):
        self._test(network_resize_images_bilinear_up)

    def test_resize_images_bilinear_same(self):
        self._test(network_resize_images_bilinear_same)

    def test_resize_bilinear1(self):
        self._test(network_resize_bilinear1)

    def test_resize_bilinear2(self):
        self._test(network_resize_bilinear2)

    def test_resize_nearest1(self):
        self._test(network_resize_nearest1)

    def test_resize_area1(self):
        self._test(network_resize_area1)

    def test_passthrough_identity(self):
        self._test(network_passthrough_identity)

    def test_passthrough_stop_gradient(self):
        self._test(network_passthrough_stop_gradient)

    def test_passthrough_cast(self):
        self._test(network_passthrough_cast)

    def test_passthrough_dropout(self):
        self._test(network_passthrough_dropout, cmp=False)  # Without dropout it will not compare to the same

    def test_passthrough_as_result(self):
        self._test(network_passthrough_as_result)

    def test_cast_bool_to_float(self):
        self._test(network_cast_bool_to_float)

    def test_cast_float_to_bool(self):
        self._test(network_cast_float_to_bool)

    def test_add_conv_transform1(self):
        self._test(network_add_conv_transform1)

    def test_add_conv_transform2(self):
        self._test(network_add_conv_transform2)

    def test_add_conv_transform3(self):
        self._test(network_add_conv_transform3)

    def test_pad_transform_pool_constant(self):
        self._test(network_pad_transform_pool_constant)

    def test_pad_transform_conv_constant(self):
        self._test(network_pad_transform_conv_constant)

    def test_pad_transform_conv_symmetric(self):
        self._test(network_pad_transform_conv_symmetric)

    def test_pad_transform_conv_reflect(self):
        self._test(network_pad_transform_conv_reflect)

    def test_fused_batch_norm_transform1(self):
        self._test(network_fused_batch_norm_transform1)

    def test_fused_batch_norm_transform2(self):
        self._test(network_fused_batch_norm_transform2)

    def test_fused_batch_norm_transform3(self):
        self._test(network_fused_batch_norm_transform3)

    def test_complex1(self):
        self._test(network_complex1)

    def test_filter_matmul1(self):
        self._test(network_filter_matmul1)

    def test_filter_matmul2(self):
        self._test(network_filter_matmul2)

    def test_filter_matmul3(self):
        self._test(network_filter_matmul3)

    def test_filter_matmul4(self):
        self._test(network_filter_matmul4)

    def test_format_rank1(self):
        self._test(network_format_rank1)

    def test_format_rank2(self):
        self._test(network_format_rank2)

    def test_format_rank3(self):
        self._test(network_format_rank3)

    def test_format_rank4(self):
        self._test(network_format_rank4)

    def test_optimizer_conv_input_nhwc_filter_hwcn(self):
        self._test(network_optimizer_conv_input_nhwc_filter_hwcn)

    def test_optimizer_conv_input_nchw_filter_hwcn(self):
        self._test(network_optimizer_conv_input_nchw_filter_hwcn)

    def test_optimizer_conv_input_nchw_filter_hwcm(self):
        self._test(network_optimizer_conv_input_nchw_filter_hwcm)

    def test_optimizer_conv_input_nchw_filter_hwcn_bias_c(self):
        self._test(network_optimizer_conv_input_nchw_filter_hwcn_bias_c)

    def test_optimizer_conv_input_nchw_filter_const_hwcn(self):
        self._test(network_optimizer_conv_input_nchw_filter_const_hwcn)

    def test_optimizer_max_pool(self):
        self._test(network_optimizer_max_pool)

    def test_optimizer_max_pool_nchw(self):
        self._test(network_optimizer_max_pool_nchw)

    def test_optimizer_conv_nchw_hwcn_padding_dilations(self):
        self._test(network_optimizer_conv_nchw_hwcn_padding_dilations)

    def test_optimizer_conv_transpose_nchw_hwcn(self):
        self._test(network_optimizer_conv_transpose_nchw_hwcn)

    def test_returning_input(self):
        self._test(network_returning_input)

    def test_clip_by_value1(self):
        self._test(network_clip_by_value1)

    def test_clip_by_value2(self):
        self._test(network_clip_by_value2)

    def test_clip_by_value3(self):
        self._test(network_clip_by_value3)

    def test_zeros(self):
        self._test(network_zeros)

    def test_zeros_like(self):
        self._test(network_zeros_like)

    def test_ones(self):
        self._test(network_ones)

    def test_ones_like(self):
        self._test(network_ones_like)

    def test_slice1(self):
        self._test(network_slice1)

    def test_slice2(self):
        self._test(network_slice2)

    def test_stack1(self):
        self._test(network_stack1)

    def test_stack2(self):
        self._test(network_stack2)

    def test_unstack1(self):
        self._test(network_unstack1)

    def test_unstack2(self):
        self._test(network_unstack2)

    def test_norm_euclidean(self):
        self._test(network_norm_euclidean)

    def test_norm_fro(self):
        self._test(network_norm_fro)

    def test_norm_1(self):
        self._test(network_norm_1)

    def test_norm_2(self):
        self._test(network_norm_2)

    def test_norm_inf(self):
        self._test(network_norm_inf)

    def test_output_shape1(self):
        self._test(network_output_shape1)

    def test_shape_of__output_shape1(self):
        self._test(network_shape_of__output_shape1)

    def test_shape_of__output_shape2(self):
        self._test(network_shape_of__output_shape2)

    def test_shape_of__reshape(self):
        self._test(network_shape_of__reshape)

    def test_no_shape_of__output_shape1(self):
        self._test(network_no_shape_of__output_shape1)

    def test_no_shape_of__output_shape2(self):
        self._test(network_no_shape_of__output_shape2)

    def test_no_shape_of__reshape(self):
        self._test(network_no_shape_of__reshape)

    def test_range1(self):
        self._test(network_range1)

    def test_range2(self):
        self._test(network_range2)

    def test_range3(self):
        self._test(network_range3)

    def test_pad(self):
        self._test(network_pad)

    def test_tile1(self):
        self._test(network_tile1)

    def test_tile2(self):
        self._test(network_tile2)

    def test_tile3(self):
        self._test(network_tile3)

    def test_resize_bicubic1(self):
        self._test(network_resize_bicubic1, cmp=False)  # Imprecise conversion


if __name__ == "__main__":
    unittest.main()
