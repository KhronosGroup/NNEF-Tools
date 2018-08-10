import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets.alexnet import alexnet_v2 as _alexnet_v2
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_50 as _resnet_v2_50
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3 as _inception_v3
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16 as _vgg_16


def alexnet_v2():
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=[1, 224, 224, 3])
    output, _end_points = _alexnet_v2(input_, num_classes=1000, is_training=False)
    return {"output": output}


def resnet_v2_50():
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=[1, 224, 224, 3])
    output, _end_points = _resnet_v2_50(input_, num_classes=1000, is_training=False)
    return {"output": output}


def inception_v3():
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=[1, 224, 224, 3])
    output, _end_points = _inception_v3(input_, num_classes=1000, is_training=False)
    return {"output": output}


def vgg16():
    input_ = tf.placeholder(dtype=tf.float32, name='input', shape=[1, 224, 224, 3])
    output, _end_points = _vgg_16(input_, num_classes=1000, is_training=False)
    return {"output": output}
