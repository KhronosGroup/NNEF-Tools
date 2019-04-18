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

from functools import partial

import nnef_tools.shape_inference.shape_inference as shapes
from nnef_tools.core import utils


def pairs(items):
    return list(zip(items, items))


def convolution_shape(op, transposed):
    return shapes.conv(input=op.inputs[0].shape, filter=op.attribs['kernel_size'],
                       padding=pairs(op.attribs['pad']), stride=op.attribs['stride'],
                       dilation=op.attribs['dilation'], groups=op.attribs['group'],
                       output_channels=op.attribs['num_output'],
                       deconv=transposed, format=shapes.Format.NCHW)


def pooling_shape(op):
    if op.attribs['global_pooling']:
        return op.inputs[0].shape[:2] + [1] * len(op.inputs[0].shape[2:])

    rank = op.inputs[0].rank
    return shapes.sliding_window(input=op.inputs[0].shape, filter=(1, 1) + op.attribs['kernel_size'],
                                 padding=pairs((0, 0) + op.attribs['pad']), stride=(1, 1) + op.attribs['stride'],
                                 dilation=[1] * rank,
                                 ceil=True)


def concat_shape(op):
    return shapes.concat(inputs=[input.shape for input in op.inputs], axis=op.attribs['axis'])


def forward_shape(op):
    return shapes.copy(op.inputs[0].shape)


def input_shape(op):
    shapes = op.attribs['shape']
    if len(shapes) == 1:
        return shapes[0]
    else:
        return shapes


def reshape_shape(op):
    return shapes.reshape(op.inputs[0].shape, shape=op.attribs['shape'],
                          offset=op.attribs['axis'], count=op.attribs['num_axes'],
                          zero_means_same=True)


def flatten_shape(op):
    return shapes.flatten(op.inputs[0].shape)


def argmax_shape(op):
    if op.attribs['axis'] is None:
        return [op.inputs[0].shape[0], op.attribs['top_k']]
    else:
        shape = list(op.inputs[0].shape)
        shape[op.attribs['axis']] = op.attribs['top_k']
        return shape


def reduction_shape(op):
    axis = op.attribs['axis']
    rank = len(op.inputs[0].shape)
    axes = list(range(axis + rank if axis < 0 else axis, rank))
    return shapes.reduce(op.inputs[0].shape, axes=axes, squeeze=True)


def tile_shape(op):
    rank = op.inputs[0].shape
    axis = op.attrib['axis']
    return shapes.tile(op.inputs[0].shape, repeat=[op.attrib['tiles'] if i == axis else 1 for i in range(rank)])


def batch_matmul_shape(op):
    return op.inputs[0].shape[:op.attribs['axis']] + [op.attribs['num_output']]


def matmul_shape(op):
    return [op.inputs[0].shape[0], op.attribs['num_output']]


def slice_shape(op):
    return shapes.split(op.inputs[0].shape, axis=op.attribs['axis'], split_points=op.attribs['slice_point'])


def split_shape(op):
    return [op.inputs[0].shape] * len(op.outputs)


def crop_shape(op):
    input = op.inputs[0].shape
    reference = op.inputs[1].shape
    axis = op.attribs['axis']
    return input[:axis] + reference[axis:]


def data_shape(op, labelled=False):
    batch = op.attribs['batch_size']
    size = op.attribs['crop_size']
    shape = [batch, 3, size, size]  # can we put None instead of 3?
    return [shape, [batch]] if labelled else shape


def infer_shape(op, custom_shapes):
    func = _DefaultShapeFuncs.get(op.name)
    if func is None:
        func = custom_shapes.get(op.name)
    if not func:
        raise utils.NNEFToolsException("shape inference not defined for operation '{}'".format(op.name))
    shapes = func(op)
    if len(op.outputs) == 1:
        op.outputs[0].shape = shapes
    else:
        for tensor, shape in zip(op.outputs, shapes):
            tensor.shape = shape


_DefaultShapeFuncs = {
    'AbsVal': forward_shape,
    'Accuracy': forward_shape,
    'ArgMax': argmax_shape,
    'BatchNorm': forward_shape,
    'Bias': forward_shape,
    'BNLL': forward_shape,
    'Clip': forward_shape,
    'Concat': concat_shape,
    'ContrastiveLoss': None,
    'Convolution': partial(convolution_shape, transposed=False),
    'Crop': crop_shape,
    'Data': partial(data_shape, labelled=True),
    'Deconvolution': partial(convolution_shape, transposed=True),
    'Dropout': forward_shape,
    'DummyData': data_shape,
    'Eltwise': forward_shape,
    'ELU': forward_shape,
    'Embed': None,
    'Exp': forward_shape,
    'Flatten': flatten_shape,
    'HDF5Data': data_shape,
    'HDF5Output': None,
    'HingeLoss': None,
    'ImageData': data_shape,
    'InfogainLoss': None,
    'InnerProduct': batch_matmul_shape,
    'Input': input_shape,
    'Log': forward_shape,
    'LRN': forward_shape,
    'LSTM': matmul_shape,
    'MemoryData': data_shape,
    'MVN': forward_shape,
    'Parameter': input_shape,
    'Pooling': pooling_shape,
    'Power': forward_shape,
    'PReLU': forward_shape,
    'Recurrent': matmul_shape,
    'Reduction': reduction_shape,
    'ReLU': forward_shape,
    'Reshape': reshape_shape,
    'RNN': matmul_shape,
    'Scale': forward_shape,
    'Sigmoid': forward_shape,
    'Softmax': forward_shape,
    'SoftmaxWithLoss': forward_shape,
    'Split': split_shape,
    'SPP': None,
    'Slice': slice_shape,
    'Swish': forward_shape,
    'TanH': forward_shape,
    'Threshold': forward_shape,
    'Tile': tile_shape,
    'WindowData': data_shape,
    '_Padding': forward_shape,  # We will merge it into the next op, so no need to calculate the padded shape
}
