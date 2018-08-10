import tensorflow as tf


def small_net1():
    # NHWC

    input_ = tf.placeholder(tf.float32, shape=[10, 64, 64, 3], name="input")

    filter1 = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 8], name="filter")
    conv1 = tf.nn.conv2d(input_, filter1, strides=[1, 2, 2, 1], padding='VALID')

    relu = tf.nn.relu(conv1)

    filter2 = tf.get_variable(dtype=tf.float32, shape=[4, 4, 8, 16], name="filter2")
    conv2 = tf.nn.conv2d(relu, filter2, strides=[1, 1, 1, 1], padding='VALID')

    return {"conv2": conv2}


def small_net2():
    # NCHW

    input_ = tf.placeholder(tf.float32, shape=[10, 3, 64, 64], name="input")

    filter1 = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 8], name="filter")
    conv1 = tf.nn.conv2d(input_, filter1, strides=[1, 1, 2, 2], padding='VALID', data_format='NCHW')

    relu = tf.nn.relu(conv1)

    filter2 = tf.get_variable(dtype=tf.float32, shape=[4, 4, 8, 16], name="filter2")
    conv2 = tf.nn.conv2d(relu, filter2, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')

    return {"conv2": conv2}
