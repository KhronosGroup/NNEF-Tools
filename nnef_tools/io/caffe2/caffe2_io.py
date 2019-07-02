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
import typing

import numpy as np
import six

from nnef_tools.core import utils, graph_utils, json_utils
from nnef_tools.io.caffe2 import caffe2_pb, caffe2_shapes
from nnef_tools.io.caffe2.caffe2_graph import *

OUTPUT_DEFAULT_NETWORK_NAME = 'network'

LEGACY_PAD_NOTSET = 0
LEGACY_PAD_VALID = 1
LEGACY_PAD_SAME = 2
LEGACY_PAD_CAFFE_LEGACY_POOLING = 3


class ReadException(utils.NNEFToolsException):
    pass


class WriteException(utils.NNEFToolsException):
    pass


NumpyDTypeByCaffe2DType = {
    'UNDEFINED': None,
    'FLOAT': np.float32,
    'BYTE': np.uint8,
    'UINT8': np.uint8,
    'INT8': np.int8,
    'UINT16': np.uint16,
    'INT16': np.int16,
    'INT32': np.int32,
    'INT64': np.int64,
    'STRING': np.str,
    'BOOL': np.bool,
    'FLOAT16': np.float16,
    'DOUBLE': np.float64,
}

Caffe2DTypeByInitializer = {
    'GivenTensorFill': 'FLOAT',
    'GivenTensorDoubleFill': 'DOUBLE',
    'GivenTensorBoolFill': 'BOOL',
    'GivenTensorInt16Fill': 'INT16',
    'GivenTensorIntFill': 'INT32',
    'GivenTensorInt64Fill': 'INT64',
    'GivenTensorStringFill': 'STRING',
    'Int8GivenTensorFill': 'INT8',
    'Int8GivenIntTensorFill': 'INT32',
}

InitializerByCaffe2DType = {
    'FLOAT': 'GivenTensorFill',
    'DOUBLE': 'GivenTensorDoubleFill',
    'BOOL': 'GivenTensorBoolFill',
    'INT16': 'GivenTensorInt16Fill',
    'INT32': 'GivenTensorIntFill',
    'INT64': 'GivenTensorInt64Fill',
    'STRING': 'GivenTensorStringFill',
}

QuantizedInitializerByCaffe2DType = {
    'INT8': 'Int8GivenTensorFill',
    'INT32': 'Int8GivenIntTensorFill',
}

# https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h
# attributes:
# global_pooling,
# kernels (kernel, kernel_h, kernel_w),
# dilations (dilation, dilation_h, dilation_w),
# strides (stride, stride_h, stride_w),
# pads (pad, pad_t, pad_l, pad_b, pad_r, legacy_pad),
# group,
# order,
# float16_compute,
# shared_buffer
ConvPoolBaseOps = {
    'AveragePool', 'AveragePool1D', 'AveragePool2D', 'AveragePool3D',
    'AveragePoolGradient', 'AveragePool1DGradient', 'AveragePool2DGradient', 'AveragePool3DGradient',
    'Conv', 'Conv1D', 'Conv2D', 'Conv3D',
    'ConvGradient', 'Conv1DGradient', 'Conv2DGradient', 'Conv3DGradient',
    'MaxPool', 'MaxPool1D', 'MaxPool2D', 'MaxPool3D',
    'MaxPoolGradient', 'MaxPool1DGradient', 'MaxPool2DGradient', 'MaxPool3DGradient',
    'MaxPoolWithIndex', 'MaxPoolWithIndexGradient',
    'PadImage',
    'LpPool', 'LpPoolGradient',
}

# https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_transpose_unpool_op_base.h
# attributes:
# kernels (kernel, kernel_h, kernel_w),
# strides (stride, stride_h, stride_w),
# pads (pad, pad_t, pad_l, pad_b, pad_r, legacy_pad),
# adjs (adj, adj_h, adj_w),
# group,
# order,
# shared_buffer
ConvTransposeUnpoolBaseOps = {
    'ConvTranspose', 'ConvTransposeGradient',
}

ReductionOps = {
    'ReduceMin', 'ReduceMax', 'ReduceSum', 'ReduceMean',
    'ReduceFrontMin', 'ReduceFrontMax', 'ReduceFrontSum', 'ReduceFrontMean',
    'ReduceBackMin', 'ReduceBackMax', 'ReduceBackSum', 'ReduceBackMean',
    'ReduceTailSum',
    'ReduceMinGradient', 'ReduceMaxGradient', 'ReduceSumGradient', 'ReduceMeanGradient',
    'ReduceFrontMinGradient', 'ReduceFrontMaxGradient', 'ReduceFrontSumGradient', 'ReduceFrontMeanGradient',
    'ReduceBackMinGradient', 'ReduceBackMaxGradient', 'ReduceBackSumGradient', 'ReduceBackMeanGradient',
    'ReduceL1', 'ReduceL2', 'ReduceL1Gradient', 'ReduceL2Gradient',
    'ColwiseMax', 'RowwiseMax', 'ColwiseMaxGradient', 'RowwiseMaxGradient',
    'SumElements', 'SumElementsGradient', 'SumElementsInt', 'SumReduceLike',
}

CopyOps = {
    'Copy',
    'CopyFromCPUInput',
    'CopyOnDeviceLike',
    'EnsureCPUOutput',
    'StopGradient',
}

BinaryOps = {
    'Add', 'And', 'Div', 'EQ', 'NE', 'GE', 'GT', 'LE', 'LT', 'Mul', 'Or', 'Sub', 'Xor',
}


def fixint(i):
    return utils.anyint_to_int(i) if i is not None else None


def fixstr(s):
    return utils.anystr_to_str(s) if s is not None else None


def get_field(proto, name, default=None):
    return getattr(proto, name) if proto.HasField(name) else default


# From https://caffe2.ai/doxygen-c/html/conv__pool__op__base_8h_source.html
def compute_pad(in_size, stride, kernel, dilation, legacy_pad, pad_head=0, pad_tail=0, legacy_in_effect=None):
    CAFFE2_PAD_HEAD_MORE = False
    dilated_kernel = dilation * (kernel - 1) + 1
    if legacy_pad == LEGACY_PAD_NOTSET:
        return pad_head, pad_tail
    elif legacy_pad == LEGACY_PAD_VALID:
        return 0, 0
    elif legacy_pad == LEGACY_PAD_SAME:
        legacy_target_size = (in_size + stride - 1) // stride
        pad_needed = (legacy_target_size - 1) * stride + dilated_kernel - in_size
        if CAFFE2_PAD_HEAD_MORE:
            pad_head = (pad_needed + 1) // 2
        else:
            pad_head = pad_needed // 2
        pad_tail = pad_needed - pad_head
        return pad_head, pad_tail
    elif legacy_pad == LEGACY_PAD_CAFFE_LEGACY_POOLING:
        out_size = 1 + (in_size + pad_head * 2 - dilated_kernel + stride - 1) // stride
        if pad_head > 0 and (out_size - 1) * stride >= in_size + pad_head:
            out_size -= 1
        standard_out_size = 1 + (in_size + pad_head * 2 - dilated_kernel) // stride
        assert out_size >= standard_out_size
        if out_size > standard_out_size and legacy_in_effect is not None:
            legacy_in_effect[0] = True
        # Original line from caffe2 code gives too much padding (pad >= kernel):
        # pad_tail = pad_head + stride * (out_size - standard_out_size)
        pad_tail = stride * (out_size - 1) + dilated_kernel - (in_size + pad_head)
        return pad_head, pad_tail
    else:
        assert False


def unify_hw_attrib(op, new_attribs, name_base, length, default):
    old_attribs = op.attribs
    name = name_base
    names = name_base + 's'
    name_h = name_base + '_h'
    name_w = name_base + '_w'

    if old_attribs.get(names):
        new_attribs[names] = old_attribs[names]
    elif name in old_attribs:
        new_attribs[names] = [old_attribs[name]] * length
    elif name_h in old_attribs and name_w in old_attribs:
        new_attribs[names] = [old_attribs[name_h], old_attribs[name_w]]
    else:
        if default is None:
            raise ReadException("{} missing {} argument".format(op.name, names))
        new_attribs[names] = [default] * length


def unify_conv_pool_base_ops(op, tensor_names):
    # type: (Caffe2Operation, typing.Set[str])->None

    is_nhwc = op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    is_gradient = op.name.endswith('Gradient')
    spatial_dims = op.inputs[0].rank - 2
    new_attribs = {}

    # unify op name (remove 1D, 2D, 3D, etc.)
    if op.name.startswith('AveragePool'):
        op.name = 'AveragePool' + ('Gradient' if is_gradient else '')
    elif op.name.startswith('MaxPool') and not op.name.startswith('MaxPoolWithIndex'):
        op.name = 'MaxPool' + ('Gradient' if is_gradient else '')
    elif op.name.startswith('Conv') and not op.name.startswith('ConvTranspose'):
        op.name = 'Conv' + ('Gradient' if is_gradient else '')
    elif op.name.startswith('MaxPool') and not op.name.startswith('MaxPoolWithIndex'):
        op.name = 'MaxPool' + ('Gradient' if is_gradient else '')

    # unify order
    new_attribs['order'] = 'NHWC' if is_nhwc else 'NCHW'

    # unify global_pooling
    if 'Pool' in op.name:
        new_attribs['global_pooling'] = op.attribs.get('global_pooling', 0)

    # unify kernels
    if 'Pool' in op.name or 'Conv' in op.name:
        if op.attribs.get('global_pooling', 0):
            new_attribs['kernels'] = op.inputs[0].shape[1:-1] if is_nhwc else op.inputs[0].shape[2:]
        else:
            unify_hw_attrib(op, new_attribs, 'kernel', length=spatial_dims, default=None)

    # unify strides
    if 'Pool' in op.name or 'Conv' in op.name:
        unify_hw_attrib(op, new_attribs, 'stride', length=spatial_dims, default=1)

    # unify dilations
    if 'Pool' in op.name or 'Conv' in op.name:
        if 'Transpose' in op.name:
            assert not any((k.startswith('dilation')) for k in six.iterkeys(op.attribs))
        else:
            unify_hw_attrib(op, new_attribs, 'dilation', length=spatial_dims, default=1)

    # unify pads
    if op.attribs.get('pads'):
        new_attribs['pads'] = op.attribs['pads']
    elif 'pad' in op.attribs:
        new_attribs['pads'] = [op.attribs['pad']] * (2 * spatial_dims)
    elif 'pad_t' in op.attribs and 'pad_l' in op.attribs and 'pad_b' in op.attribs and 'pad_r' in op.attribs:
        new_attribs['pads'] = [op.attribs['pad_t'], op.attribs['pad_l'], op.attribs['pad_b'], op.attribs['pad_r']]
    else:
        new_attribs['pads'] = [0] * (2 * spatial_dims)

    # resolve legacy padding
    if 'Conv' in op.name or 'Pool' in op.name:
        legacy_pad = op.attribs.get('legacy_pad', LEGACY_PAD_NOTSET)
        if (legacy_pad in [LEGACY_PAD_VALID, LEGACY_PAD_SAME]
                and (any(k.startswith('pad') for k in six.iterkeys(op.attribs)))):
            raise ReadException("Padding should not be specified when legacy_pad is VALID or SAME")

        if legacy_pad != LEGACY_PAD_NOTSET:
            if 'Transpose' in op.name:
                assert legacy_pad in [LEGACY_PAD_SAME, LEGACY_PAD_VALID]
                # from https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_transpose_unpool_op_base.h
                new_attribs['pads'] = [0] * (2 * spatial_dims)  # They do it for same too
            else:
                padding = caffe2_shapes.caffe2_pads_to_nnef_padding(new_attribs['pads'])
                legacy_in_effect = [False]
                padding = [compute_pad(in_size, stride, kernel, dilation, legacy_pad, p, q, legacy_in_effect)
                           for in_size, stride, kernel, dilation, (p, q)
                           in zip(op.inputs[0].shape[1 if is_nhwc else 2:],
                                  new_attribs['strides'],
                                  new_attribs['kernels'],
                                  new_attribs['dilations'],
                                  padding)]
                new_attribs['pads'] = [max(0, p) for p in caffe2_shapes.nnef_padding_to_caffe2_pads(padding)]
                if legacy_in_effect[0]:
                    print('Legacy Caffe size calculation is in effect for {} = {}()'
                          .format(', '.join(t.name for t in op.outputs), op.name))

    # unify adjs (adjustment a.k.a. output padding)
    if 'Transpose' in op.name:
        unify_hw_attrib(op, new_attribs, 'adj', length=spatial_dims, default=0)

    # unify group
    if 'Conv' in op.name:
        new_attribs['group'] = op.attribs.get('group', 1)

    # unify mode
    if op.name == 'PadImage':
        new_attribs['mode'] = op.attribs.get('mode', 'constant')
        for k, v in six.iteritems(op.attribs):
            if 'stride' in k or 'dilation' in k:
                assert v == 1 or all(vv == 1 for vv in v), "Padding cannot have stride/dilation"
    # unify p
    if op.name == 'LpPool':
        new_attribs['p'] = float(op.attribs.get('p', 2.0))

    # set unified attribs
    op.attribs = new_attribs


def unify_reduce_ops(op, tensor_names):
    # type: (Caffe2Operation, typing.Set[str])->None

    new_attribs = {}
    is_gradient = 'Gradient' in op.name

    if op.name.startswith('ColwiseMax'):
        op.name = 'ReduceMaxGradient' if is_gradient else 'ReduceMax'
        new_attribs['axes'] = [1]
        if not is_gradient:
            new_attribs['keepdims'] = False
    elif op.name.startswith('RowwiseMax'):
        op.name = 'ReduceMaxGradient' if is_gradient else 'ReduceMax'
        new_attribs['axes'] = [2]
        if not is_gradient:
            new_attribs['keepdims'] = False
    elif op.name.startswith('SumElements'):
        if op.attribs.get('average', False):
            op.name = 'ReduceMeanGradient' if is_gradient else 'ReduceMean'
        else:
            op.name = 'ReduceSumGradient' if is_gradient else 'ReduceSum'
        new_attribs['axes'] = list(range(op.inputs[0].rank))
        if not is_gradient:
            new_attribs['keepdims'] = False
    elif op.name == 'SumReduceLike':
        input, reference = op.inputs
        is_nhwc = op.attribs.get('order', 'NCHW').upper() == 'NHWC'
        if op.attribs.get('axis_str'):
            axis = ('NHWC' if is_nhwc else 'NCHW').index(op.attribs['axis_str'].upper())
        else:
            axis = op.attribs.get('axis', -1)
        if axis == -1:
            axis = input.rank - reference.rank
        if axis < 0:
            axis += input.rank
        new_attribs['axis'] = axis
    else:
        is_front = 'Front' in op.name
        is_back = 'Back' in op.name
        is_tail = 'Tail' in op.name
        op.name = op.name.replace('Front', '').replace('Back', '').replace('Tail', '')

        if is_front:
            n = op.attribs.get('num_reduce_dim', 1)
            new_attribs['axes'] = list(range(n))
            if not is_gradient:
                new_attribs['keepdims'] = False
        elif is_back:
            n = op.attribs.get('num_reduce_dim', 1)
            new_attribs['axes'] = list(range(op.inputs[0].rank - n, op.inputs[0].rank))
            if not is_gradient:
                new_attribs['keepdims'] = False
        elif is_tail:
            new_attribs['axes'] = list(range(1, op.inputs[0].rank))
            if not is_gradient:
                new_attribs['keepdims'] = False
        else:
            if op.attribs.get('axes'):
                new_attribs['axes'] = op.attribs['axes']
            else:
                new_attribs['axes'] = list(range(op.inputs[0].rank))

            if not is_gradient:
                new_attribs['keepdims'] = op.attribs.get('keepdims', True)
    op.attribs = new_attribs


def unify_op(op, tensor_names):
    # type: (Caffe2Operation, typing.Set[str])->None
    # From https://caffe2.ai/doxygen-c/html/conv__pool__op__base_8h_source.html

    if op.name in ConvPoolBaseOps or op.name in ConvTransposeUnpoolBaseOps:
        unify_conv_pool_base_ops(op, tensor_names)
    elif op.name in ('Concat', 'DepthConcat'):
        op.name = 'Concat'
        assert not ('order' in op.attribs and 'axis' in op.attribs)

        is_nhwc = op.attribs.get('order', 'NCHW').upper() == 'NHWC'
        axis = op.attribs['axis'] if 'axis' in op.attribs else (-1 if is_nhwc else 1)
        axis = op.inputs[0].rank + axis if axis < 0 else axis

        op.attribs = {
            'axis': axis,
            'add_axis': op.attribs.get('add_axis', 0),
        }
    elif op.name == 'Append':
        op.name = 'Concat'
        op.attribs = {
            'axis': 0,
            'add_axis': 0,
        }
        split_info = Caffe2Tensor(graph=op.graph, name=utils.get_numbered_name('split_info', tensor_names))
        tensor_names.add(split_info.name)
        op.outputs = (op.output, split_info)
    elif op.name in ('Split', 'DepthSplit'):
        op.name = 'Split'
        assert not ('order' in op.attribs and 'axis' in op.attribs)

        is_nhwc = op.attribs.get('order', 'NCHW').upper() == 'NHWC'
        axis = op.attribs['axis'] if 'axis' in op.attribs else (-1 if is_nhwc else 1)
        axis = op.inputs[0].rank + axis if axis < 0 else axis

        new_attribs = {'axis': axis}
        if len(op.inputs) == 1:
            if 'split' in op.attribs:
                new_attribs['split'] = op.attribs['split']
            else:
                assert op.inputs[0].shape[axis] % len(op.outputs) == 0
                new_attribs['split'] = [op.inputs[0].shape[axis] // len(op.outputs)] * len(op.outputs)

        op.attribs = new_attribs
    elif op.name in ('GenerateProposals', 'GenerateProposalsCPP'):
        op.name = 'GenerateProposals'
    elif op.name in ReductionOps:
        unify_reduce_ops(op, tensor_names)
    elif op.name in CopyOps:
        op.name = 'Copy'
        op.inputs = (op.inputs[0],)
    elif op.name in BinaryOps or (op.name == 'Pow' and len(op.inputs) == 2):
        a, b = op.inputs
        order = op.attribs.get('order', 'NCHW').upper()
        axis = -1
        if 'axis' in op.attribs:
            axis = op.attribs['axis']
        elif op.attribs.get('axis_str'):
            axis = order.index(op.attribs['axis_str'].upper())
        if axis == -1:
            axis = a.rank - b.rank
        if axis < 0:
            axis += a.rank
        if axis == 0 and op.inputs[0].shape == op.inputs[1].shape:
            op.attribs = {
                'broadcast': 0,
            }
        else:
            op.attribs = {
                'broadcast': 1,
                'axis': axis
            }

    elif op.name == 'NCHW2NHWC':
        op.name = 'Transpose'
        axes = list(range(op.input.rank))
        op.attribs = {
            'axes': axes[0:1] + axes[2:] + axes[1:2]
        }
    elif op.name == 'NHWC2NCHW':
        op.name = 'Transpose'
        axes = list(range(op.input.rank))
        op.attribs = {
            'axes': axes[0:1] + axes[-1:] + axes[1:-1]
        }


def get_operation(op_def, graph, tensor_by_name):
    inputs = [fixstr(input) for input in op_def.input]
    outputs = [fixstr(output) for output in op_def.output]
    name = fixstr(get_field(op_def, 'name'))
    domain = fixstr(get_field(op_def, 'domain'))
    type = fixstr(op_def.type)

    if domain == "caffe2":
        domain = None
    if domain:
        type = domain + "::" + type

    for input in inputs:
        if input not in tensor_by_name:
            print("Creating tensor: ", input)
            tensor_by_name[input] = Caffe2Tensor(graph=graph, name=input)

    for output in outputs:
        if output not in tensor_by_name:
            tensor_by_name[output] = Caffe2Tensor(graph=graph, name=output)

    attribs = {}
    for arg in op_def.arg:
        k, v = get_attribute(type, arg)
        attribs[k] = v

    return Caffe2Operation(graph=graph,
                           name=type,
                           label=name if name else None,
                           inputs=[tensor_by_name[input] for input in inputs],
                           outputs=[tensor_by_name[output] for output in outputs],
                           attribs=attribs)


def get_attribute(op_type, arg):
    name = fixstr(arg.name)

    if arg.HasField('f'):
        value = float(arg.f)
    elif arg.HasField('i'):
        value = fixint(arg.i)
    elif arg.HasField('s'):
        if op_type in ["Int8GivenTensorFill", "GivenTensorStringFill"] and name == 'values':
            value = np.frombuffer(arg.s, dtype=np.int8)
        else:
            value = fixstr(arg.s)
    elif arg.HasField('t'):
        raise ReadException("Tensor arg not supported")
    elif arg.HasField('n'):
        raise ReadException("Networks with subnetworks (If, While, etc.) are not supported")
    elif arg.floats:
        value = list(arg.floats)
    elif arg.ints:
        value = [fixint(i) for i in arg.ints]
    elif arg.strings:
        value = [fixstr(s) for s in arg.strings]
    elif arg.tensors:
        raise ReadException("Tensor list arg not supported")
    elif arg.nets:
        raise ReadException("Networks with subnetworks (If, While, etc.) are not supported")
    elif getattr(arg, 'qtensors', None):
        raise ReadException("Quantized Tensor list arg not supported")
    else:
        value = []

    if name == "values":
        if op_type == "GivenTensorFill":
            value = np.array(value, dtype=np.float32)
        elif op_type == "GivenTensorDoubleFill":
            value = np.array(value, dtype=np.float64)
        elif op_type == "GivenTensorBoolFill":
            value = np.array(value, dtype=np.bool)
        elif op_type == "GivenTensorInt16Fill":
            value = np.array(value, dtype=np.int16)
        elif op_type == "GivenTensorIntFill":
            value = np.array(value, dtype=np.int32)
        elif op_type == "GivenTensorInt64Fill":
            value = np.array(value, dtype=np.int64)
        elif op_type == "GivenTensorStringFill":
            value = np.array(value, dtype=np.int8)
        elif op_type == "Int8GivenTensorFill":
            value = np.array(value, dtype=np.int8)
        elif op_type == "Int8GivenIntTensorFill":
            value = np.array(value, dtype=np.int32)  # np.int32 not typo

    return name, value


def get_device_option(device_option):
    d = {}
    for field in device_option.DESCRIPTOR.fields:
        if field.label != field.LABEL_REPEATED and device_option.HasField(field.name):
            d[fixstr(field.name)] = getattr(device_option, field.name)
        elif field.label == field.LABEL_REPEATED and getattr(device_option, field.name):
            d[fixstr(field.name)] = list(getattr(device_option, field.name))
    return d


def get_graph_from_net_def(net_def):
    graph = Caffe2Graph(name=fixstr(net_def.name) if net_def.name else None)
    tensor_by_name = {}
    input_names = [fixstr(name) for name in net_def.external_input]
    output_names = [fixstr(name) for name in net_def.external_output]
    graph.inputs = [Caffe2Tensor(graph=graph, name=input_name) for input_name in input_names]
    graph.outputs = [Caffe2Tensor(graph=graph, name=output_name) for output_name in output_names]
    tensor_by_name.update({t.name: t for t in graph.tensors})

    for op_def in net_def.op:
        get_operation(op_def, graph, tensor_by_name)

    def duplicate(tensor):
        tensor = Caffe2Tensor(graph=graph, name=utils.get_numbered_name(tensor.name, tensor_by_name))
        tensor_by_name[tensor.name] = tensor
        return tensor

    graph_utils.resolve_tensor_overwrite(graph, duplicate)
    return graph


def get_graph_from_value_info(dict_):
    graph = Caffe2Graph()
    for tensor_name, [dtype, shape] in six.iteritems(dict_):
        Caffe2Tensor(graph=graph, name=fixstr(tensor_name), shape=list(shape), dtype=caffe2_pb.dtype_id_to_name(dtype))
    graph.inputs = tuple(graph.tensors)
    return graph


def combine_graphs_into_predict_graph(predict_graph, init_graph, value_info_graph):
    # type: (Caffe2Graph, Caffe2Graph, Caffe2Graph)->None
    input_names = {input.name for input in value_info_graph.inputs}
    predict_tensor_by_name = {tensor.name: tensor for tensor in predict_graph.tensors}
    in_init_but_not_in_predict = []
    for init_tensor in init_graph.tensors:
        if init_tensor.name not in input_names:
            if not init_tensor.producer:
                raise ReadException("Tensor '{}' does not have a producer in the init-net".format(init_tensor.name))
            init_op = init_tensor.producer
            if init_op.name not in Caffe2DTypeByInitializer:
                raise ReadException("Initializer '{}' is not supported".format(init_op.name))
            if init_tensor.name not in predict_tensor_by_name:
                in_init_but_not_in_predict.append(init_tensor.name)
                continue
            predict_tensor = predict_tensor_by_name[init_tensor.name]  # type: Caffe2Tensor
            assert predict_tensor.producer is None and predict_tensor.shape is None
            predict_tensor.shape = list(init_op.attribs['shape'])
            predict_tensor.dtype = Caffe2DTypeByInitializer[init_op.name]
            predict_tensor.data = np.reshape(init_op.attribs['values'], init_op.attribs['shape'])
            if init_op.name in ('Int8GivenTensorFill', 'Int8GivenIntTensorFill'):
                predict_tensor.quantization = Caffe2Quantization(init_op.attribs['Y_scale'],
                                                                 init_op.attribs['Y_zero_point'])
    input_tensors = []
    for value_info_tensor in sorted(value_info_graph.tensors, key=lambda t: t.name):
        if value_info_tensor.name not in predict_tensor_by_name:
            possible_input_names = ' or '.join(sorted(['"' + t.name + '"'
                                                       for t in predict_graph.tensors
                                                       if t.producer is None and (t.data is None or t.data.size <= 1)]))
            raise ReadException("Tensor '{}' is in value-info but not in predict-net.\n"
                                "Possible input tensors: {}".format(value_info_tensor.name, possible_input_names))
        input_tensor = predict_tensor_by_name[value_info_tensor.name]
        assert input_tensor.producer is None and input_tensor.shape is None and input_tensor.data is None
        input_tensor.shape = value_info_tensor.shape
        input_tensor.dtype = value_info_tensor.dtype
        input_tensors.append(input_tensor)
    predict_graph.inputs = input_tensors
    if not predict_graph.outputs:
        predict_graph.outputs = [t for t in predict_graph.tensors if not t.consumers]
    graph_utils.remove_unreachable(predict_graph)
    for tensor in predict_graph.tensors:
        if (not tensor.producers
                and not tensor.is_variable
                and not tensor.is_constant
                and tensor not in predict_graph.inputs):
            raise ReadException(
                "Tensor '{}' has no initializer but is not listed as input in value-info.".format(tensor.name))

    if in_init_but_not_in_predict:
        print("Warning: There were tensors in the init-net that are not present in the predict-net: {}".format(
            in_init_but_not_in_predict))
    for tensor in predict_graph.tensors:
        if tensor.data is not None and tensor.data.size == 0:
            print("Warning: Tensor '{}' possibly missing from value_info.json (has zero size)".format(tensor.name))


def write_predict_net(graph, filename):
    # type: (Caffe2Graph, str)->None

    net_def = caffe2_pb.NetDef()
    net_def.name = graph.name if graph.name else OUTPUT_DEFAULT_NETWORK_NAME

    net_def.external_input.extend([tensor.name for tensor in ([tensor for tensor in graph.inputs]
                                                              + [tensor for tensor in graph.tensors if
                                                                 tensor.is_variable or tensor.is_constant])])

    net_def.external_output.extend([tensor.name for tensor in graph.outputs])

    for op in graph.operations:
        op_def = net_def.op.add()
        build_op_def(op, op_def)

    with open(filename, 'wb') as file:
        file.write(net_def.SerializeToString())


def write_init_net(graph, filename):
    # type: (Caffe2Graph, str)->None

    net_def = caffe2_pb.NetDef()
    net_def.name = graph.name if graph.name else OUTPUT_DEFAULT_NETWORK_NAME

    for tensor in graph.tensors:
        if tensor.is_constant or tensor.is_variable:
            initializer_by_dtype = (QuantizedInitializerByCaffe2DType if tensor.quantization
                                    else InitializerByCaffe2DType)
            if tensor.dtype not in initializer_by_dtype:
                raise WriteException("No initializer for dtype: {}, quantization={}"
                                     .format(tensor.dtype, tensor.quantization))
            initializer = initializer_by_dtype[tensor.dtype]
            if tensor.is_variable:
                values = tensor.data
            else:
                assert tensor.is_constant
                if len(tensor.data) == tensor.count:
                    values = np.array(tensor.data).reshape(tensor.shape)
                else:
                    if len(tensor.data) != 1:
                        raise WriteException("Constant tensor: len(tensor.data) must be in 1 or tensor.count")
                    values = np.array(tensor.data * tensor.count).reshape(tensor.shape)

            if initializer in ("Int8GivenTensorFill", "GivenTensorStringFill"):
                values = np.array(values, dtype=np.int8).tobytes()

            attribs = dict(values=values, shape=tensor.shape)
            if tensor.quantization:
                attribs['Y_scale'] = tensor.quantization.scale
                attribs['Y_zero_point'] = tensor.quantization.zero_point
            op = Caffe2Operation(graph=graph,
                                 name=initializer,
                                 attribs=attribs,
                                 outputs=tensor)
            op_def = net_def.op.add()
            build_op_def(op, op_def)
            graph.remove_operation(op, unlink=True)

    with open(filename, 'wb') as file:
        file.write(net_def.SerializeToString())


def build_op_def(op, op_def):
    # type: (Caffe2Operation, caffe2_pb.OperatorDef)->None

    op_def.input.extend([input.name for input in op.inputs])
    op_def.output.extend([output.name for output in op.outputs])

    if op.label:
        op_def.name = op.label

    if "::" in op.name:
        op_def.domain, op_def.type = op.name.split('::')
    else:
        op_def.type = op.name

    for k, v in six.iteritems(op.attribs):
        argument = op_def.arg.add()
        build_attribute(k, v, argument)


def build_attribute(key, value, argument):
    # type: (str, typing.Any, caffe2_pb.Argument)->None

    argument.name = key

    if isinstance(value, np.ndarray):
        value = value.flatten().tolist()

    if isinstance(value, float):
        argument.f = value
    elif isinstance(value, int):
        argument.i = value
    elif isinstance(value, bytes):
        argument.s = value
    elif isinstance(value, str):
        argument.s = value.encode('utf-8')
    elif isinstance(value, Caffe2Tensor):
        raise WriteException("Argument type Tensor is unsupported")
    elif isinstance(value, Caffe2Graph):
        raise WriteException("Argument type NetDef is unsupported")
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            pass
        else:
            if isinstance(value[0], float):
                argument.floats.extend(value)
            elif isinstance(value[0], int):
                argument.ints.extend(value)
            elif isinstance(value[0], bytes):
                argument.strings.extend(value)
            elif isinstance(value[0], str):
                argument.strings.extend([v.encode('utf-8') for v in value])
            elif isinstance(value[0], Caffe2Tensor):
                raise WriteException("Argument type Tensor list is unsupported")
            elif isinstance(value[0], Caffe2Graph):
                raise WriteException("Argument type NetDef list is unsupported")
            else:
                assert False, \
                    "Unsupported attribute: {}: {} of type: List[{}]".format(key, value, type(value[0]).__name__)

    else:
        assert False, "Unsupported attribute: {}: {} of type: {}".format(key, value, type(value).__name__)


def write_value_info(graph, filename):
    # type: (Caffe2Graph, str)->None
    json_utils.dump({input.name: [caffe2_pb.dtype_name_to_id(input.dtype), input.shape] for input in graph.inputs},
                    filename, indent=False)


class Reader(object):

    def __init__(self, custom_shapes=None):
        if custom_shapes is None:
            custom_shapes = {}
        self.custom_shapes = custom_shapes

    def __call__(self, predict_net_path, init_net_path, value_info_path):
        # type: (str, str, str)->Caffe2Graph

        net_def = caffe2_pb.NetDef()
        with open(predict_net_path, 'rb') as f:
            net_def.ParseFromString(f.read())
        predict_graph = get_graph_from_net_def(net_def)

        net_def = caffe2_pb.NetDef()
        with open(init_net_path, 'rb') as f:
            net_def.ParseFromString(f.read())
        init_graph = get_graph_from_net_def(net_def)

        value_info_graph = get_graph_from_value_info(json_utils.load(value_info_path))

        combine_graphs_into_predict_graph(predict_graph, init_graph, value_info_graph)

        tensor_names = {tensor.name for tensor in predict_graph.tensors}
        for op in list(predict_graph.operations):
            unify_op(op, tensor_names)
            caffe2_shapes.infer_shape(op, self.custom_shapes)
        predict_graph.generate_missing_names()
        graph_utils.remove_unreachable(predict_graph)

        return predict_graph


class Writer(object):

    def __call__(self, graph, directory):
        if not graph.is_sorted():
            graph.sort()
        graph.generate_missing_names(labels_too=False)

        utils.makedirs(directory, exist_ok=True)
        predict_net_path = os.path.join(directory, "predict_net.pb")
        init_net_path = os.path.join(directory, "init_net.pb")
        value_info_path = os.path.join(directory, "value_info.json")
        write_predict_net(graph, predict_net_path)
        write_init_net(graph, init_net_path)
        write_value_info(graph, value_info_path)
        return None


__all__ = [
    'Reader',
    'Writer',
]
