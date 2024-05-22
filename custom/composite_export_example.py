import numpy as np
import src.nnef_tools.io.tf.graphdef as graphdef
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf


# define composite operators as decorated python functions

@graphdef.composite_function
def lp_norm(x, p=2, axis=None, keepdims=False, name=None):
    return tf.pow(tf.reduce_sum(tf.pow(tf.abs(x), p), axis=axis, keepdims=keepdims), 1 / p)


@graphdef.composite_function
def sum_pool2d(input, ksize, strides, padding, data_format='NHWC', name=None):
    pooled = tf.nn.avg_pool2d(input, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
    return pooled * float(np.prod(ksize))


# reset tractking of composite functions

graphdef.reset_composites()


# define the TF graph

x = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32, name='input')
w = tf.get_variable('w', shape=(5, 5, 3, 16), dtype=tf.float32, initializer=tf.zeros_initializer)
x = tf.nn.conv2d(x, w, strides=1, padding='SAME')
x = sum_pool2d(x, ksize=(1, 3, 3, 1), strides=1, padding='SAME')
x = lp_norm(x, axis=3, keepdims=True)


# export the graph to protobuf

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    graphdef.save_default_graph('test.pb', session=sess, outputs={x: 'output'},
                                input_shapes={'input': (1, 32, 32, 3)})
