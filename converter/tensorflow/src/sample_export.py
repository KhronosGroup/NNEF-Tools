# Copyright (c) 2012-2017 The Khronos Group Inc.
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


import tensorflow as tf
import numpy as np
import tf2nnef      # this is required for exporting to NNEF

from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_50     # only needed for the example


input_shape = [1, 224, 224, 3]


# define network here
# function must not take any params
# inputs must be defined via placeholders
# outputs must be returned as a tuple
# only parts of the graph which lead to returned outputs will be exported
def network():
    input = tf.placeholder(dtype=tf.float32, name='input', shape=input_shape)
    net, end_points = resnet_v2_50(input, num_classes=1000, is_training=False)
    return net


checkpoint = None   # replace with checkpoint file name if you also want to export weights and activations

# provide network builder method and checkpoint to exporter
# returns a list of tensors that may be further used
converter = tf2nnef.export_network(network, checkpoint)

if checkpoint is not None:
    # must feed palceholders created in network definition with some input data
    tf2nnef.export_activations(converter, checkpoint, feed_dict={'input:0': np.random.random(input_shape)})
