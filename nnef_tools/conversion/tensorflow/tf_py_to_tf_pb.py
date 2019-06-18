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

import typing
from functools import partial

import numpy as np
import six

from nnef_tools.conversion.tensorflow import tf_pb_to_tf_py
from nnef_tools.core import utils
from nnef_tools.io.tensorflow.tf_graph import *
from nnef_tools.shape_inference import shape_inference as infer

# noinspection PyProtectedMember
_tf_py_dtype_to_tf_pb_dtype = utils.key_value_swapped(tf_pb_to_tf_py._tf_py_dtype_by_tf_pb_dtype)


def convert(tf_graph):
    # type: (TFGraph)->None

    for tensor in tf_graph.tensors:
        if tensor.is_variable:
            if tensor.data.dtype == np.int64:
                tensor.data = tensor.data.astype(np.int32)
                tensor.dtype = "int32"

            if tensor.data.dtype == np.float64:
                tensor.data = tensor.data.astype(np.float32)
                tensor.dtype = "float32"

    for op in list(tf_graph.operations):
        if op.name == "tf.nn.softmax":
            expand_softmax(tf_graph, op)

    for op in list(tf_graph.operations):
        assert op.name in _DefaultConverters, "No tf_py_to_tf_pb converter for {}".format(op.name)
        _DefaultConverters[op.name](op)

    for tensor in tf_graph.tensors:
        tensor.dtype = _tf_py_dtype_to_tf_pb_dtype[tensor.dtype]

    for op in tf_graph.operations:
        if op.name not in ['LogicalAnd', 'LogicalNot', 'LogicalOr']:
            if op.name in ['Select', 'Conv2DBackpropInput', 'Conv3DBackpropInputV2']:
                op.attribs['T'] = op.inputs[1].dtype
            else:
                op.attribs['T'] = op.inputs[0].dtype

        if op.name == 'MaxPoolWithArgmax':
            op.attribs['Targmax'] = 'DT_INT64'

    tf_graph.generate_missing_names()


def expand_softmax(tf_graph, tf_op):
    assert tf_op.input.rank != 0

    axis = tf_op.attribs.get('axis')
    if axis is None:
        axis = -1
    if axis < 0:
        axis += tf_op.input.rank

    tf_op.attribs['axis'] = -1

    if tf_op.input.rank == 2 and axis == 1:
        return

    if axis != tf_op.input.rank - 1:
        perm = utils.without(range(tf_op.input.rank), axis) + [axis]
        perm_inv = utils.inverse_permutation(perm)
        transpose = TFOperation(graph=tf_graph,
                                name="tf.transpose",
                                inputs=tf_op.input,
                                attribs=dict(perm=perm),
                                outputs=TFTensor(graph=tf_graph,
                                                 name=None,
                                                 shape=infer.transpose(input=tf_op.input.shape, axes=perm),
                                                 dtype=tf_op.input.dtype))
        tf_op.inputs = transpose.output
        old_output = tf_op.output
        tf_op.outputs = TFTensor(graph=tf_graph,
                                 name=None,
                                 shape=tf_op.input.shape,
                                 dtype=tf_op.input.dtype)
        TFOperation(graph=tf_graph,
                    name="tf.transpose",
                    inputs=tf_op.output,
                    attribs=dict(perm=perm_inv),
                    outputs=old_output)

    if tf_op.input.rank != 2:
        shape = [-1, tf_op.input.shape[-1]]
        reshape = TFOperation(graph=tf_graph,
                              name="tf.reshape",
                              inputs=tf_op.input,
                              attribs=dict(shape=shape),
                              outputs=TFTensor(graph=tf_graph,
                                               name=None,
                                               shape=infer.reshape(input=tf_op.input.shape, shape=shape),
                                               dtype=tf_op.input.dtype))
        tf_op.inputs = reshape.output
        old_output = tf_op.output
        tf_op.outputs = TFTensor(graph=tf_graph, name=None, shape=list(tf_op.input.shape), dtype=tf_op.input.dtype)
        TFOperation(graph=tf_graph,
                    name="tf.reshape",
                    inputs=tf_op.output,
                    attribs=dict(shape=old_output.shape),
                    outputs=old_output)


def create_constant_tensor(graph, value, np_dtype=None):
    if np_dtype is not None:
        arr = np.array(value, dtype=np_dtype)
    else:
        arr = np.array(value)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        elif arr.dtype == np.int64:
            arr = arr.astype(np.int32)  # Constants must be int32 at most places
    return TFTensor(graph=graph, name=None, shape=list(arr.shape), dtype=str(arr.dtype), data=arr.flatten().tolist())


def generic_converter(op,  # type: TFOperation
                      target_name,  # type: str
                      revert_inputs=False,  # type: bool
                      attrib_name_dict=None,  # type: typing.Optional[typing.Dict[str, str]]
                      attrib_to_input_dict=None,  # type: typing.Optional[typing.Dict[str, int]]
                      attribs_to_remove=None  # type: typing.Optional[typing.List[str]]
                      ):
    op.name = target_name

    if revert_inputs:
        op.inputs = tuple(reversed(op.inputs))

    if attrib_name_dict:
        attribs = {}

        for k, v in six.iteritems(op.attribs):
            if k in attrib_name_dict:
                attribs[attrib_name_dict[k]] = v
            else:
                attribs[k] = v
        op.attribs = attribs

    if attrib_to_input_dict:
        inputs = list(op.inputs)
        attribs_inputs = sorted([(a, i) for a, i in six.iteritems(attrib_to_input_dict)], key=lambda t: t[1])
        for attrib_name, input_id in attribs_inputs:
            if input_id < 0:
                input_id += len(inputs) + 1
            inputs.insert(input_id, create_constant_tensor(op.graph, op.attribs[attrib_name]))
            del op.attribs[attrib_name]
        op.inputs = inputs

    if attribs_to_remove:
        for attrib_name in attribs_to_remove:
            del op.attribs[attrib_name]


def generate_back_converters(converters):
    back_converters = {}
    for target_name, converter in six.iteritems(converters):
        if isinstance(converter, partial) and converter.func == tf_pb_to_tf_py.generic_converter:
            from_name = converter.keywords['target_name']
            revert_inputs = converter.keywords.get('revert_inputs', False)
            attrib_name_dict = utils.key_value_swapped(converter.keywords.get('attrib_name_dict', {}))
            attrib_to_input_dict = utils.key_value_swapped(converter.keywords.get('input_to_attrib_dict', {}))
            attribs_to_remove = list(six.iterkeys(converter.keywords.get('new_attribs', {})))
            back_converters[from_name] = partial(generic_converter,
                                                 target_name=target_name,
                                                 revert_inputs=revert_inputs,
                                                 attrib_name_dict=attrib_name_dict,
                                                 attrib_to_input_dict=attrib_to_input_dict,
                                                 attribs_to_remove=attribs_to_remove)
    return back_converters


def convert_batch_normalization(op):
    # type: (TFOperation)->None
    op.name = "FusedBatchNorm"
    input, mean, variance, offset, scale = op.inputs

    is_nhwc = not (mean.rank >= 2 and mean.shape[1] > 1)

    def make_1d(tensor):
        # type: (TFTensor)->TFTensor

        if tensor.rank == 1:
            return tensor

        return TFOperation(name='Reshape',
                           graph=tensor.graph,
                           inputs=(tensor, create_constant_tensor(graph=tensor.graph, value=[tensor.count])),
                           outputs=TFTensor(graph=tensor.graph, shape=[tensor.count], dtype=tensor.dtype)).output

    op.inputs = (input, make_1d(scale), make_1d(offset), make_1d(mean), make_1d(variance))
    op.attribs['is_training'] = False
    op.attribs['epsilon'] = op.attribs['variance_epsilon']
    op.attribs['data_format'] = 'NHWC' if is_nhwc else 'NCHW'
    del op.attribs['variance_epsilon']


def convert_flatten(op):
    # type: (TFOperation)->None
    op.name = "Reshape"
    op.inputs = tuple(op.inputs) + (create_constant_tensor(op.graph, list(op.output.shape)),)


def convert_split(op):  # TODO what if split has -1, and fix split
    # type: (TFOperation)->None
    if isinstance(op.attribs['num_or_size_splits'], (list, tuple)):
        op.name = 'SplitV'
        op.inputs = (op.input,
                     create_constant_tensor(op.graph, op.attribs['num_or_size_splits'], np_dtype=np.int64),
                     create_constant_tensor(op.graph, op.attribs['axis']))
        op.attribs['num_split'] = len(op.attribs['num_or_size_splits'])
        del op.attribs['num_or_size_splits']
        del op.attribs['axis']
    else:
        op.name = 'Split'
        op.inputs = (create_constant_tensor(op.graph, op.attribs['axis']), op.input)
        op.attribs['num_split'] = op.attribs['num_or_size_splits']
        del op.attribs['num_or_size_splits']
        del op.attribs['axis']


def postconvert_concat(op):
    # type: (TFOperation)->None
    op.attribs['N'] = len(op.inputs) - 1


def postconvert_slice(op):
    # type: (TFOperation)->None
    op.attribs['Index'] = 'DT_INT32'


def convert_cast(op):
    # type: (TFOperation)->None
    op.name = 'Cast'
    op.attribs['SrcT'] = _tf_py_dtype_to_tf_pb_dtype[op.input.dtype]
    op.attribs['DstT'] = _tf_py_dtype_to_tf_pb_dtype[op.output.dtype]


def converter_sequence(fun1, fun2):
    def f(op):
        fun1(op)
        fun2(op)

    return f


# noinspection PyProtectedMember
_DefaultConverters = generate_back_converters(
    tf_pb_to_tf_py._DefaultConverters
)  # type: typing.Dict[str, typing.Callable[[TFOperation], None]]

_DefaultConverters.update({
    "tf.nn.batch_normalization": convert_batch_normalization,
    "tf.layers.flatten": convert_flatten,
    'tf.concat': converter_sequence(_DefaultConverters['tf.concat'], postconvert_concat),
    'tf.nn.depthwise_conv2d': _DefaultConverters['tf.nn.depthwise_conv2d_native'],
    'tf.split': convert_split,
    'tf.slice': converter_sequence(_DefaultConverters['tf.slice'], postconvert_slice),
    'tf.cast': convert_cast,
})
