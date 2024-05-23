# Copyright (c) 2020 The Khronos Group Inc.
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

import onnx
from onnx.shape_inference import infer_shapes
import caffe2.python.onnx.frontend
from caffe2.proto import caffe2_pb2
from ..onnx.reader import onnx_model_to_graph
from ...utils.types import as_str
from google.protobuf import text_format
from collections.abc import Sequence
from caffe2.python.workspace import GlobalInit
import json
import sys
import os

try:
    import caffe
except ImportError:
    from . import caffe
    sys.modules['caffe'] = caffe

from caffe2.python.caffe_translator import TranslateModel, TranslatorRegistry, ConvertTensorProtosToInitNet, \
    AddArgument, BaseTranslate


@TranslatorRegistry.Register("ArgMax")
def TranslateArgmax(layer, pretrained_blobs, is_test, **kwargs):
    param = layer.argmax_param
    if param.top_k != 1:
        raise ValueError("Unsupported attribute value 'top_k = {}' in ArgMax layer".format(param.top_k))
    if param.out_max_val:
        raise ValueError("Conversion of ArgMax layer with 'out_max_val = True' is not supported")

    if param.HasField("axis"):
        caffe_op = BaseTranslate(layer, "ArgMax")
        AddArgument(caffe_op, "keepdims", True)
        AddArgument(caffe_op, "axis", param.axis)
        return caffe_op, []
    else:
        flatten_op = caffe2_pb2.OperatorDef()
        flatten_op.type = "Flatten"
        flatten_op.input.extend(layer.bottom)
        flatten_op.output.append(layer.bottom[0] + "_flattened")

        argmax_op = caffe2_pb2.OperatorDef()
        argmax_op.type = "ArgMax"
        argmax_op.input.append(layer.bottom[0] + "_flattened")
        argmax_op.output.extend(layer.top)
        AddArgument(argmax_op, "keepdims", True)
        AddArgument(argmax_op, "axis", 1)

        return [flatten_op, argmax_op], []


@TranslatorRegistry.Register("ReLU")
def TranslateRelu(layer, pretrained_blobs, is_test, **kwargs):
    param = layer.relu_param
    if param.HasField("negative_slope") and param.negative_slope:
        relu_op = BaseTranslate(layer, "LeakyRelu")
        AddArgument(relu_op, "alpha", param.negative_slope)
        return relu_op, []
    else:
        return BaseTranslate(layer, "Relu"), []


def _HookedTranslateLayer(layer, pretrained_blobs, is_test, **kwargs):
    _pre_translate(layer, pretrained_blobs)
    ops, params = _TranslateLayer(layer, pretrained_blobs, is_test, **kwargs)
    _post_translate(layer, pretrained_blobs, ops, params)
    return ops, params


_TranslateLayer = TranslatorRegistry.TranslateLayer
TranslatorRegistry.TranslateLayer = _HookedTranslateLayer


GlobalInit(['caffe2', '--caffe2_log_level=2'])


_UnrecognizedAttribs = {'ws_nbytes_limit'}


def _caffe_to_caffe2(prototxt, caffemodel):
    if prototxt.layer[0].type == 'Input':
        input_names = list(prototxt.layer[0].top)
        input_shapes = list(item.dim for item in prototxt.layer[0].input_param.shape)
    else:
        input_names = list(item for item in prototxt.input)
        input_shapes = list(item.dim for item in prototxt.input_shape)
        if len(input_shapes) == 0:
            input_dims = prototxt.input_dim
            assert len(input_dims) == 4 * len(input_names)
            input_shapes = [None] * len(input_names)
            for i in range(len(input_names)):
                input_shapes[i] = input_dims[4 * i: 4 * (i+1)]

    predict_net, params = TranslateModel(prototxt, caffemodel, is_test=True, remove_legacy_pad=False, input_dims=[])

    # Assume there is one input and one output
    external_input = predict_net.op[0].input[0]
    external_output = predict_net.op[-1].output[0]

    predict_net.external_input.extend([external_input])
    predict_net.external_input.extend([param.name for param in params.protos])
    predict_net.external_output.extend([external_output])
    init_net = ConvertTensorProtosToInitNet(params, external_input)

    value_info = {name: (onnx.TensorProto.FLOAT, shape) for name, shape in zip(input_names, input_shapes)}

    return predict_net, init_net, value_info


_DeconvGroups = {}


def _pre_translate(layer, blobs):
    if layer.type == 'Convolution':
        _fix_conv_pool_param(layer.convolution_param)
    elif layer.type == 'Deconvolution':
        _fix_conv_pool_param(layer.convolution_param)
        if layer.convolution_param.group != 1:
            _DeconvGroups[layer.name] = layer.convolution_param.group
            layer.convolution_param.group = 0   # to trick the conversion check
    elif layer.type == 'Pooling':
        _fix_conv_pool_param(layer.pooling_param)
    elif layer.type == 'Eltwise':
        _fix_eltwise_param(layer.eltwise_param)
    elif layer.type == 'BatchNorm':
        _fix_batch_norm_param(layer.batch_norm_param, blobs)


def _post_translate(layer, blobs, ops, params):
    if layer.type == 'Deconvolution' and layer.convolution_param.group == 0:
        AddArgument(ops[0], 'group', _DeconvGroups[layer.name])


def _fix_conv_pool_param(param):
    if isinstance(param.kernel_size, Sequence) and len(param.kernel_size) == 2:
        param.kernel_h = param.kernel_size[0]
        param.kernel_w = param.kernel_size[1]
        del param.kernel_size[1]
        del param.kernel_size[0]
    if isinstance(param.stride, Sequence) and len(param.stride) == 2:
        param.stride_h = param.stride[0]
        param.stride_w = param.stride[1]
        del param.stride[1]
        del param.stride[0]
    if isinstance(param.pad, Sequence) and len(param.pad) == 2:
        param.pad_h = param.pad[0]
        param.pad_w = param.pad[1]
        del param.pad[1]
        del param.pad[0]


def _fix_eltwise_param(param):
    if len(param.coeff) > 0 and all(c == 1 for c in param.coeff):
        for i in reversed(range(len(param.coeff))):
            del param.coeff[i]


def _fix_batch_norm_param(param, blobs):
    if len(blobs) > 2:
        if blobs[2].data[0] == 0:
            blobs[2].data[0] = 1


def _caffe2_net_to_onnx_model(predict_net, init_net, value_info):
    graph = caffe2.python.onnx.frontend.caffe2_net_to_onnx_graph(predict_net, init_net, value_info)
    if not graph.name:
        graph.name = 'Graph'

    opset_id = onnx.OperatorSetIdProto()
    opset_id.domain = ''
    opset_id.version = 11
    model = onnx.helper.make_model(graph, opset_imports=[opset_id])
    onnx.checker.check_model(model)
    return model


def _remove_unrecognized_attributes(net_def):
    for op_def in net_def.op:
        for idx in reversed(range(len(op_def.arg))):
            name = as_str(op_def.arg[idx].name)
            if name in _UnrecognizedAttribs:
                del op_def.arg[idx]


def load_caffe_model(path):
    base, ext = os.path.splitext(path)
    assert ext == '.prototxt'

    with open(path) as file:
        prototxt = caffe.proto.caffe_pb2.NetParameter()
        text_format.Merge(file.read(), prototxt)
    with open(base + '.caffemodel', 'rb') as file:
        caffemodel = caffe.proto.caffe_pb2.NetParameter()
        caffemodel.ParseFromString(file.read())

    return prototxt, caffemodel


def load_caffe_model_as_onnx(path):
    prototxt, caffemodel = load_caffe_model(path)
    predict_net, init_net, value_info = _caffe_to_caffe2(prototxt, caffemodel)
    _remove_unrecognized_attributes(predict_net)
    return _caffe2_net_to_onnx_model(predict_net, init_net, value_info)


def load_caffe2_model(folder):
    predict_net = caffe2_pb2.NetDef()
    with open(os.path.join(folder, 'predict_net.pb'), 'rb') as file:
        predict_net.ParseFromString(file.read())

    init_net = caffe2_pb2.NetDef()
    with open(os.path.join(folder, 'init_net.pb'), 'rb') as file:
        init_net.ParseFromString(file.read())

    with open(os.path.join(folder, 'value_info.json')) as file:
        value_info = json.load(file)

    return predict_net, init_net, value_info


def load_caffe2_model_as_onnx(folder):
    predict_net, init_net, value_info = load_caffe2_model(folder)
    _remove_unrecognized_attributes(predict_net)
    return _caffe2_net_to_onnx_model(predict_net, init_net, value_info)


class Reader:

    def __init__(self, legacy=False):
        self._legacy = legacy

    def __call__(self, path):
        onnx_model = load_caffe_model_as_onnx(path) if self._legacy else load_caffe2_model_as_onnx(path)
        onnx.checker.check_model(onnx_model)
        onnx_model = infer_shapes(onnx_model)
        return onnx_model_to_graph(onnx_model)
