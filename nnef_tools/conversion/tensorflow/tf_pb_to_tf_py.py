from __future__ import division, print_function, absolute_import

import typing
from functools import partial

import numpy as np
import six

from nnef_tools.core import utils
from nnef_tools.io.tensorflow.tf_graph import *
from nnef_tools.io.tensorflow.tf_pb import tf_pb_eval, tf_pb_shape_inference

_tf_py_dtype_by_tf_pb_dtype = {
    'DT_INVALID': None,
    'DT_HALF': 'float16',
    'DT_FLOAT': 'float32',
    'DT_DOUBLE': 'float64',
    'DT_INT8': 'int8',
    'DT_INT16': 'int16',
    'DT_INT32': 'int32',
    'DT_INT64': 'int64',
    'DT_UINT8': 'uint8',
    'DT_UINT16': 'uint16',
    'DT_UINT32': 'uint32',
    'DT_UINT64': 'uint64',
    'DT_BOOL': 'bool',
    'DT_STRING': 'string',
    'DT_COMPLEX64': 'complex64',
    'DT_COMPLEX128': 'complex128',
}


def _evaluate_constant(tf_tensor):
    # type: (TFTensor)->np.ndarray

    # noinspection PySimplifyBooleanCheck
    if tf_tensor.data == []:
        return np.array([], dtype=np.dtype(tf_tensor.dtype))

    value = np.array(tf_tensor.data, dtype=np.dtype(tf_tensor.dtype))
    last_val = value.flat[-1]
    value2 = np.full(shape=tf_tensor.shape, fill_value=last_val, dtype=np.dtype(tf_tensor.dtype))
    value2.flat[:value.size] = value.flat
    return value2


# noinspection PyProtectedMember
def evaluate_and_convert(tf_graph, source_shapes=None, source_dtypes=None):
    # type: (TFGraph, typing.Dict[str, typing.List[int]], typing.Dict[str, str])->None
    if source_shapes is None:
        source_shapes = {}
    if source_dtypes is None:
        source_dtypes = {}

    source_shapes = {(k + ':0' if ':' not in k else k): v for k, v in six.iteritems(source_shapes)}
    source_dtypes = {(k + ':0' if ':' not in k else k): v for k, v in six.iteritems(source_dtypes)}

    tf_graph.sort()

    for tensor in tf_graph.tensors:
        tensor.dtype = _tf_py_dtype_by_tf_pb_dtype.get(tensor.dtype, None)

        if tensor.name and tensor.name in source_shapes:
            if not _is_compatible(tensor.shape, source_shapes[tensor.name]):
                raise utils.NNEFToolsException(
                    "The specified shape is incompatible with the original shape for {}. {} vs. {}".format(
                        tensor.name, source_shapes[tensor.name], tensor.shape))
            tensor.shape = source_shapes[tensor.name]
        if tensor.name and tensor.name in source_dtypes:
            if not (tensor.dtype is None or tensor.dtype == source_dtypes[tensor.name]):
                raise utils.NNEFToolsException(
                    "The specified dtype is incompatible with the original dtype for {}. {} vs. {}".format(
                        tensor.name, source_dtypes[tensor.name], tensor.dtype))

            tensor.dtype = source_dtypes[tensor.name]
        if len(tensor.producers) == 0 and (tensor.shape is None or -1 in tensor.shape or tensor.dtype is None):
            raise utils.NNEFToolsException(
                "Source tensor '{}' has incomplete dtype or shape: {} {}\n"
                "Please specify it in --input-shape or through the corresponding API.".format(
                    tensor.name, tensor.dtype, tensor.shape))

    const_value_by_tensor = {}

    for tensor in tf_graph.tensors:
        if tensor.is_constant:
            const_value_by_tensor[tensor] = tf_pb_eval._evaluate_constant(tensor)
        elif tensor.is_variable:
            const_value_by_tensor[tensor] = tensor.data

    for op in tf_graph.operations:
        # Shape prop
        assert op.name in tf_pb_shape_inference._DefaultPropagators, "No shape propagator for {}".format(op.name)
        propagated_shapes, propagated_dtypes = tf_pb_shape_inference._DefaultPropagators[op.name](op,
                                                                                                  const_value_by_tensor)
        assert not utils.has_le_0(propagated_shapes)
        assert len(propagated_shapes) == len(propagated_dtypes) == len(op.outputs)
        for new_shape, new_dtype, tensor in zip(propagated_shapes, propagated_dtypes, op.outputs):
            assert _is_compatible(tensor.shape, new_shape)
            tensor.shape = new_shape
            assert tensor.dtype is None or tensor.dtype == new_dtype

        # Evaluation
        if op.name in tf_pb_eval._DefaultOpEvaluators:
            tf_pb_eval._DefaultOpEvaluators[op.name](op, const_value_by_tensor)

        # Conversion
        assert op.name in _DefaultConverters, "No tf_pb_to_tf_py converter for {}".format(op.name)
        _DefaultConverters[op.name](op, const_value_by_tensor)

    for tensor in tf_graph.tensors:
        if tensor.is_variable:
            label = tensor.name
            if label is not None:
                if label.endswith(':0'):
                    label = label[:-2]
                label = label.replace(':', '_')
            tensor.label = label


def _is_compatible(orig_shape, shape):
    if orig_shape is None:
        return True
    if len(orig_shape) != len(shape):
        return False
    for o, s in zip(orig_shape, shape):
        if o != s and o != -1:
            return False
    return True


def fix_types(list_):
    # type: (typing.Any)->typing.Any
    if isinstance(list_, list) and len(list_) >= 1 and utils.is_anyint(list_[0]):
        list_ = [utils.anyint_to_int(i) for i in list_]
    return list_


def generic_converter(op,  # type: TFOperation
                      const_value_by_tensor,  # type: typing.Dict[TFTensor, np.ndarray]
                      target_name,  # type: str
                      attrib_name_dict=None,  # type: typing.Optional[typing.Dict[str, str]]
                      input_to_attrib_dict=None,  # type: typing.Optional[typing.Dict[int, str]]
                      revert_inputs=False,  # type: bool
                      new_attribs=None,  # type: typing.Optional[typing.Dict[str, typing.Any]]
                      ):
    # type: (...)->None

    op.name = target_name
    if attrib_name_dict:
        attribs = {}

        for k, v in six.iteritems(op.attribs):
            if k in attrib_name_dict:
                attribs[attrib_name_dict[k]] = v
            else:
                attribs[k] = v
        op.attribs = attribs
    if input_to_attrib_dict:
        inputs = []
        for i in range(len(op.inputs)):
            if i in input_to_attrib_dict:
                assert "{}.{} not evaluated to constant".format(op.name, input_to_attrib_dict[i])
                op.attribs[input_to_attrib_dict[i]] = fix_types(const_value_by_tensor[op.inputs[i]].tolist())
            elif (i - len(op.inputs)) in input_to_attrib_dict:
                assert "{}.{} not evaluated to constant".format(op.name, input_to_attrib_dict[i - len(op.inputs)])
                op.attribs[input_to_attrib_dict[i - len(op.inputs)]] = fix_types(
                    const_value_by_tensor[op.inputs[i]].tolist())
            else:
                inputs.append(op.inputs[i])
        op.inputs = tuple(inputs)
    if revert_inputs:
        op.inputs = tuple(reversed(op.inputs))
    if new_attribs:
        op.attribs.update(new_attribs)


# See: https://www.tensorflow.org/api_docs/cc/
_DefaultConverters = {
    # attribless:
    "Abs": partial(generic_converter, target_name="tf.abs"),
    "Add": partial(generic_converter, target_name="tf.add"),
    "BiasAdd": partial(generic_converter, target_name="tf.nn.bias_add"),
    "Ceil": partial(generic_converter, target_name="tf.ceil"),
    "Elu": partial(generic_converter, target_name="tf.nn.elu"),
    "Equal": partial(generic_converter, target_name="tf.equal"),
    "Exp": partial(generic_converter, target_name="tf.exp"),
    "Floor": partial(generic_converter, target_name="tf.floor"),
    "Greater": partial(generic_converter, target_name="tf.greater"),
    "GreaterEqual": partial(generic_converter, target_name="tf.greater_equal"),
    "Identity": partial(generic_converter, target_name="tf.identity"),
    "Less": partial(generic_converter, target_name="tf.less"),
    "LessEqual": partial(generic_converter, target_name="tf.less_equal"),
    "Log": partial(generic_converter, target_name="tf.log"),
    "LogicalAnd": partial(generic_converter, target_name="tf.logical_and"),
    "LogicalNot": partial(generic_converter, target_name="tf.logical_not"),
    "LogicalOr": partial(generic_converter, target_name="tf.logical_or"),
    "Maximum": partial(generic_converter, target_name="tf.maximum"),
    "Minimum": partial(generic_converter, target_name="tf.minimum"),
    "Mul": partial(generic_converter, target_name="tf.multiply"),
    "Neg": partial(generic_converter, target_name="tf.negative"),
    "NotEqual": partial(generic_converter, target_name="tf.not_equal"),
    "Pow": partial(generic_converter, target_name="tf.pow"),
    "RealDiv": partial(generic_converter, target_name="tf.divide"),
    "Relu": partial(generic_converter, target_name="tf.nn.relu"),
    "Relu6": partial(generic_converter, target_name="tf.nn.relu6"),
    "Round": partial(generic_converter, target_name="tf.round"),
    "Rsqrt": partial(generic_converter, target_name="tf.rsqrt"),
    "Sigmoid": partial(generic_converter, target_name="tf.nn.sigmoid"),
    "Sign": partial(generic_converter, target_name="tf.sign"),
    "Softmax": partial(generic_converter, target_name="tf.nn.softmax", new_attribs={'axis': -1}),
    "Softplus": partial(generic_converter, target_name="tf.nn.softplus"),
    "Softsign": partial(generic_converter, target_name="tf.nn.softsign"),
    "Sqrt": partial(generic_converter, target_name="tf.sqrt"),
    "Square": partial(generic_converter, target_name="tf.square"),
    "Sub": partial(generic_converter, target_name="tf.subtract"),
    "Tanh": partial(generic_converter, target_name="tf.nn.tanh"),
    "Select": partial(generic_converter, target_name="tf.where"),
    'ClipByValue': partial(generic_converter, target_name='tf.clip_by_value'),

    # more complex:
    "AvgPool": partial(generic_converter, target_name="tf.nn.avg_pool"),
    "Conv2D": partial(generic_converter, target_name="tf.nn.conv2d"),
    "Conv3D": partial(generic_converter, target_name="tf.nn.conv3d"),
    "Conv2DBackpropInput": partial(generic_converter,
                                   target_name="tf.nn.conv2d_transpose",
                                   input_to_attrib_dict={0: "output_shape"},
                                   revert_inputs=True),
    "Conv3DBackpropInputV2": partial(generic_converter,
                                     target_name="tf.nn.conv3d_transpose",
                                     input_to_attrib_dict={0: "output_shape"},
                                     revert_inputs=True),
    # "CudnnRNN": None,
    "DepthwiseConv2dNative": partial(generic_converter, target_name="tf.nn.depthwise_conv2d_native"),
    "FusedBatchNorm": partial(generic_converter, target_name="tf.nn.fused_batch_norm"),
    "LRN": partial(generic_converter, target_name="tf.nn.lrn"),
    "MatMul": partial(generic_converter, target_name="tf.matmul"),
    "MaxPool": partial(generic_converter, target_name="tf.nn.max_pool"),
    "MaxPoolWithArgmax": partial(generic_converter, target_name="tf.nn.max_pool_with_argmax"),
    "Pack": partial(generic_converter, target_name="tf.stack"),
    # "Placeholder": None,
    # "PlaceholderWithDefault": None,
    "Shape": partial(generic_converter, target_name="tf.shape"),
    "Squeeze": partial(generic_converter, target_name="tf.squeeze", attrib_name_dict={"squeeze_dims": "axis"}),

    # even more complex:
    "ExpandDims": partial(generic_converter, target_name="tf.expand_dims", input_to_attrib_dict={1: "axis"}),
    "ArgMin": partial(generic_converter, target_name="tf.argmin", input_to_attrib_dict={1: "axis"}),
    "ArgMax": partial(generic_converter, target_name="tf.argmax", input_to_attrib_dict={1: "axis"}),
    "Max": partial(generic_converter, target_name="tf.reduce_max", attrib_name_dict={"keep_dims": "keepdims"},
                   input_to_attrib_dict={1: "axis"}),
    "Min": partial(generic_converter, target_name="tf.reduce_min", attrib_name_dict={"keep_dims": "keepdims"},
                   input_to_attrib_dict={1: "axis"}),
    "Mean": partial(generic_converter, target_name="tf.reduce_mean", attrib_name_dict={"keep_dims": "keepdims"},
                    input_to_attrib_dict={1: "axis"}),
    "ConcatV2": partial(generic_converter, target_name="tf.concat", input_to_attrib_dict={-1: "axis"}),
    "Pad": partial(generic_converter,
                   target_name="tf.pad",
                   input_to_attrib_dict={1: "paddings"},
                   new_attribs={'mode': 'CONSTANT',
                                'constant_values': 0.0}),
    "Reshape": partial(generic_converter, target_name="tf.reshape", input_to_attrib_dict={1: "shape"}),
    "ResizeArea": partial(generic_converter, target_name="tf.image.resize_area", input_to_attrib_dict={1: "size"}),
    "ResizeBilinear": partial(generic_converter,
                              target_name="tf.image.resize_bilinear",
                              input_to_attrib_dict={1: "size"}),
    "ResizeNearestNeighbor": partial(generic_converter,
                                     target_name="tf.image.resize_nearest_neighbor",
                                     input_to_attrib_dict={1: "size"}),
    "Slice": partial(generic_converter, target_name="tf.slice", input_to_attrib_dict={1: "begin", 2: "size"}),
    "Split": partial(generic_converter,
                     target_name="tf.split",
                     attrib_name_dict={'num_split': 'num_or_size_splits'},
                     input_to_attrib_dict={0: "axis"}),
    "SplitV": partial(generic_converter,
                      target_name="tf.split",
                      input_to_attrib_dict={1: "num_or_size_splits", 2: "axis"}),
    "StridedSlice": partial(generic_converter,
                            target_name="tf.strided_slice",
                            input_to_attrib_dict={1: "begin", 2: "end", 3: "strides"}),
    "Sum": partial(generic_converter, target_name="tf.reduce_sum", input_to_attrib_dict={1: "axis"},
                   attrib_name_dict={"keep_dims": "keepdims"}),
    "Transpose": partial(generic_converter, target_name="tf.transpose", input_to_attrib_dict={1: "perm"}),
}
