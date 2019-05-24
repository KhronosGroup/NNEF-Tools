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

import copy
import os
from collections import OrderedDict

import google.protobuf.text_format as protobuf_text_format
import numpy as np

from nnef_tools.core import graph_utils
from nnef_tools.core import utils
from nnef_tools.io.caffe import caffe_shapes as shapes
from nnef_tools.io.caffe.caffe_graph import *
from nnef_tools.io.caffe.caffe_pb import caffe__pb2 as caffe_pb

_layer_type_param_fields = {
    'AbsVal': '',
    'Accuracy': 'accuracy_param',
    'ArgMax': 'argmax_param',
    'BatchNorm': 'batch_norm_param',
    'Bias': 'bias_param',
    'BNLL': '',
    'Clip': 'clip_param',
    'Concat': 'concat_param',
    'ContrastiveLoss': 'contrastive_loss_param',
    'Convolution': 'convolution_param',
    'Crop': 'crop_param',
    'Data': 'data_param',
    'Deconvolution': 'convolution_param',
    'Dropout': 'dropout_param',
    'DummyData': 'dummy_data_param',
    'Eltwise': 'eltwise_param',
    'ELU': 'elu_param',
    'Embed': 'embed_param',
    'Exp': 'exp_param',
    'Flatten': 'flatten_param',
    'HDF5Data': 'hdf5_data_param',
    'HDF5Output': 'hdf5_output_param',
    'HingeLoss': 'hinge_loss_param',
    'ImageData': 'image_data_param',
    'InfogainLoss': 'infogain_loss_param',
    'InnerProduct': 'inner_product_param',
    'Input': 'input_param',
    'Log': 'log_param',
    'LRN': 'lrn_param',
    'LSTM': 'recurrent_param',
    'MemoryData': 'memory_data_param',
    'MVN': 'mvn_param',
    'Parameter': 'parameter_param',
    'Pooling': 'pooling_param',
    'Power': 'power_param',
    'PReLU': 'prelu_param',
    'Python': 'python_param',
    'Recurrent': 'recurrent_param',
    'Reduction': 'reduction_param',
    'ReLU': 'relu_param',
    'Reshape': 'reshape_param',
    'RNN': 'recurrent_param',
    'Scale': 'scale_param',
    'Sigmoid': 'sigmoid_param',
    'Softmax': 'softmax_param',
    'SoftmaxWithLoss': 'softmax_param',
    'Split': '',
    'SPP': 'spp_param',
    'Slice': 'slice_param',
    'Swish': 'swish_param',
    'TanH': 'tanh_param',
    'Threshold': 'threshold_param',
    'Tile': 'tile_param',
    'WindowData': 'window_data_param',
}

# int to str mapping
_layer_type_by_v1_layer_type = [
    None,
    'Accuracy',
    'BNLL',
    'Concat',
    'Convolution',
    'Data',
    'Dropout',
    'EuclideanLoss',
    'Flatten',
    'HDF5Data',
    'HDF5Output',
    'Im2Col',
    'ImageData',
    'InfogainLoss',
    'InnerProduct',
    'LRN',
    'MultinomialLogisticLoss',
    'Pooling',
    'ReLU',
    'Sigmoid',
    'Softmax',
    'SoftmaxLoss',
    'Split',
    'TanH',
    'WindowData',
    'Eltwise',
    'Power',
    'SigmoidCrossEntropyLoss',
    'HingeLoss',
    'MemoryData',
    'ArgMax',
    'Threshold',
    'DummyData',
    'Slice',
    'MVN',
    'AbsVal',
    'Silence',
    'ContrastiveLoss',
    'Exp',
    'Deconvolution',
]

_layer_type_by_v0_layer_type = {
    "accuracy": "Accuracy",
    "bnll": "BNLL",
    "concat": "Concat",
    "conv": "Convolution",
    "data": "Data",
    "dropout": "Dropout",
    "euclidean_loss": "EuclideanLoss",
    "flatten": "Flatten",
    "hdf5_data": "HDF5Data",
    "hdf5_output": "HDF5Output",
    "im2col": "Im2Col",
    "images": "ImageData",
    "infogain_loss": "InfogainLoss",
    "innerproduct": "InnerProduct",
    "lrn": "LRN",
    "multinomial_logistic_loss": "MultinomialLogisticLoss",
    "pool": "Pooling",
    "relu": "ReLU",
    "sigmoid": "Sigmoid",
    "softmax": "Softmax",
    "softmax_loss": "SoftmaxLoss",
    "split": "Split",
    "tanh": "TanH",
    "window_data": "WindowData",
    "padding": "_Padding",
}

# from: https://github.com/BVLC/caffe/blob/master/src/caffe/util/upgrade_proto.cpp
_attrib_updater_by_layer_type_v0_to_v2 = {
    "accuracy": lambda a: dict(),
    "bnll": lambda a: dict(),
    "concat": lambda a: dict(axis=a['concat_dim']),
    "conv": lambda a: dict(bias_term=a['biasterm'],
                           num_output=a['num_output'],
                           pad=a['pad'],
                           kernel_size=a['kernelsize'],
                           group=a['group'],
                           stride=a['stride'],
                           dilation=1),
    "data": lambda a: dict(crop_size=a['cropsize'], batch_size=a['batchsize']),
    "dropout": lambda a: dict(dropout_ratio=a['dropout_ratio']),
    "euclidean_loss": lambda a: dict(),
    "flatten": lambda a: dict(axis=1, end_axis=-1),
    "hdf5_data": lambda a: dict(crop_size=a['cropsize'], batch_size=a['batchsize']),
    "hdf5_output": lambda a: dict(),
    "im2col": lambda a: dict(),
    "images": lambda a: dict(crop_size=a['cropsize'], batch_size=a['batchsize']),
    "infogain_loss": lambda a: dict(source=a['source'], axis=1),
    "innerproduct": lambda a: dict(num_output=a['num_output'], bias_term=a['biasterm'], axis=1, transpose=False),
    "lrn": lambda a: dict(local_size=a['local_size'], alpha=a['alpha'], beta=a['beta'], norm_region=0, k=a['k']),
    "multinomial_logistic_loss": lambda a: dict(),
    "pool": lambda a: dict(pad=a['pad'],
                           kernel_size=a['kernelsize'],
                           stride=a['stride'],
                           global_pooling=False,
                           round_mode=0,
                           pool=a['pool']),
    "relu": lambda a: dict(negative_slope=0.0),
    "sigmoid": lambda a: dict(),
    "softmax": lambda a: dict(axis=1),
    "softmax_loss": lambda a: dict(),
    "split": lambda a: dict(),
    "tanh": lambda a: dict(),
    "window_data": lambda a: dict(crop_size=a['cropsize'], batch_size=a['batchsize']),
    "padding": lambda a: dict(pad=a['pad']),
}


def _read_net_from_prototxt(filename, net_param):
    with open(filename, 'rb') as f:
        text = f.read()
    protobuf_text_format.Merge(text, net_param)


def _read_net_from_protobuf(filename, net_param):
    with open(filename, mode='rb') as f:
        return net_param.ParseFromString(f.read())


def _decode_graph(net_param, custom_shapes):
    version = _get_version(net_param)

    graph = CaffeGraph(utils.unify_int_and_str_types(net_param.name) if net_param.name else None)

    tensors = {}

    # Handle deprecated input parameter
    if len(net_param.input) > 0:
        input_names = list(net_param.input)
        if len(net_param.input_shape) == len(input_names):
            input_shapes = [list(s.dim) for s in net_param.input_shape]
        elif len(net_param.input_dim) == 4 * len(input_names):
            input_shapes = list(utils.batches(net_param.input_dim, size=4))
        else:
            raise utils.NNEFToolsException("Inputs cannot be determined from the network!")
        input_names, input_shapes = utils.unify_int_and_str_types((input_names, input_shapes))
        input_tensors = [CaffeTensor(graph=graph, name=name, shape=list(shape))
                         for name, shape in zip(input_names, input_shapes)]
        tensors.update({t.name: t for t in input_tensors})
        CaffeOperation(graph=graph, name='Input', outputs=input_tensors, attribs=dict(shape=input_shapes), label=None)

    layers = net_param.layer if version >= 2 else net_param.layers
    for layer in layers:
        op = _decode_operation(layer, graph, tensors, version)

        if version == 0 and len(op.inputs) >= 1:
            for producer in op.inputs[0].producers:
                if producer.name == '_Padding':
                    op.attribs['pad'] = list(producer.attribs['pad'])

        shapes.infer_shape(op, custom_shapes)

    def duplicate(tensor):
        tensor = CaffeTensor(graph=graph,
                             name=utils.get_numbered_name(tensor.name, tensors),
                             shape=copy.copy(tensor.shape))
        tensors[tensor.name] = tensor
        return tensor

    graph_utils.resolve_tensor_overwrite(graph, duplicate)

    graph.inputs = OrderedDict([(op.output.name, op.output) for op in graph.operations if op.name == 'Input'])
    graph.outputs = OrderedDict([(tensor.name, tensor) for tensor in graph.tensors if not tensor.consumers])

    if version == 0:
        graph_utils.remove_passthroughs(graph, lambda op: op.name == "_Padding")

    return graph


def _build_caffemodel_from_graph(graph, net_param, with_weights=True):
    if graph.name:
        net_param.name = graph.name

    for operation in graph.operations:
        layer = net_param.layer.add()
        _build_operation(operation, layer)
        if with_weights:
            for tensor in operation.inputs:
                if tensor.data is not None:
                    blob = layer.blobs.add()
                    _build_blob(tensor, blob)


def _change_blob_to_rank(tensor, rank):
    if tensor.rank != rank:
        assert all(dim == 1 for dim in tensor.shape[:-rank])
        tensor.shape = tensor.shape[-rank:]
        tensor.data = tensor.data.reshape(tensor.shape)


def _fix_legacy_blob_ranks(op_name, attribs, inputs, blobs):
    if op_name in ("BatchNorm", "PReLU"):
        for blob in blobs:
            _change_blob_to_rank(blob, rank=1)
    elif op_name in ("Convolution", "Deconvolution"):
        if len(blobs) >= 2:
            _change_blob_to_rank(blobs[1], rank=1)
    elif op_name in ("InnerProduct",):
        if len(blobs) >= 1:
            _change_blob_to_rank(blobs[0], rank=2)
        if len(blobs) >= 2:
            _change_blob_to_rank(blobs[1], rank=1)
    elif op_name in ("Bias", "Scale"):
        rank = inputs[1].rank if len(inputs) >= 2 else attribs['num_axes']
        for blob in blobs:
            _change_blob_to_rank(blob, rank=rank)


def _decode_operation(layer, graph, tensors, version):
    if version == 0:
        layer_type = _layer_type_by_v0_layer_type[layer.layer.type]
    elif version == 1:
        layer_type = _layer_type_by_v1_layer_type[layer.type]
    elif version == 2:
        layer_type = utils.unify_int_and_str_types(layer.type)
    else:
        assert False

    for top in layer.top:
        if top not in tensors:
            tensors[top] = CaffeTensor(graph, utils.unify_int_and_str_types(top))

    inputs = [tensors[bottom] for bottom in layer.bottom]
    outputs = [tensors[top] for top in layer.top]
    blobs = layer.blobs if version >= 1 else layer.layer.blobs
    blobs = [_decode_blob(layer.name, idx, blob, graph) for idx, blob in enumerate(blobs)]

    if version == 0:
        attribs = _decode_attributes(layer.layer)
        attribs = _unify_attribs_v0(layer.layer.type, attribs, rank=len(inputs[0].shape) if inputs else 0)
    else:
        attribs = {}
        field = _layer_type_param_fields.get(layer_type)
        if field:
            params = getattr(layer, field)
            assert params is not None
            attribs = _decode_attributes(params)
            attribs = _unify_attribs_v1_v2(params, attribs, rank=len(inputs[0].shape) if inputs else 0)
        elif field is None:
            print("Warning: could not decode params for layer of type '{}'".format(layer_type))

        transform = layer.transform_param
        if transform:
            if transform.crop_size:
                attribs['crop_size'] = transform.crop_size

    _fix_legacy_blob_ranks(layer_type, attribs, inputs, blobs)
    layer_name = layer.name if version >= 1 else layer.layer.name
    return CaffeOperation(graph=graph,
                          name=utils.unify_int_and_str_types(layer_type),
                          inputs=inputs + blobs,
                          outputs=outputs,
                          attribs=utils.unify_int_and_str_types(attribs),
                          label=utils.unify_int_and_str_types(layer_name))


def _build_operation(operation, layer):
    layer.type = operation.name
    layer.name = operation.label

    layer.bottom.extend([tensor.name for tensor in operation.inputs if tensor.data is None])
    layer.top.extend([tensor.name for tensor in operation.outputs])

    field = _layer_type_param_fields.get(operation.name)
    if field:
        params = getattr(layer, field)
        assert params is not None
        _build_attributes(operation.attribs, params)
    elif field is None:
        print("Warning: could not build params for layer of type '{}'".format(operation.name))


def _decode_attributes(params):
    attribs = {}
    for field in params.DESCRIPTOR.fields:
        if field.label != field.LABEL_REPEATED and not params.HasField(field.name) and not field.has_default_value:
            attribs[field.name] = None
        else:
            value = getattr(params, field.name)
            if field.type != field.TYPE_MESSAGE:
                attribs[field.name] = list(value) if field.label == field.LABEL_REPEATED else value
            elif field.message_type.name == 'BlobShape':
                attribs[field.name] = ([list(v.dim) for v in value]
                                       if field.label == field.LABEL_REPEATED
                                       else list(value.dim))
            elif field.message_type.name == 'FillerParameter':
                attribs[field.name] = value.type

    return attribs


def _build_attributes(attribs, params):
    for field in params.DESCRIPTOR.fields:
        value = attribs.get(field.name)
        if value is not None:
            # Make caffe 1.0 friendly, if possible
            if isinstance(params, caffe_pb.PoolingParameter) and field.name == 'round_mode' and value == 0:
                continue

            if field.type != field.TYPE_MESSAGE:
                if field.label == field.LABEL_REPEATED:
                    getattr(params, field.name).extend(value)
                else:
                    setattr(params, field.name, value)
            elif field.message_type.name == 'BlobShape':
                if field.label == field.LABEL_REPEATED:
                    for shape in value:
                        blob_shape = getattr(params, field.name).add()
                        blob_shape.dim.extend(shape)
                else:
                    getattr(params, field.name).dim.extend(value)
            elif field.message_type.name == 'FillerParameter':
                getattr(params, field.name).type = value


def _read_blobs_from_caffemodel(filename, graph):
    net_param = caffe_pb.NetParameter()
    _read_net_from_protobuf(filename, net_param)
    version = _get_version(net_param)

    operations = {op.label: op for op in graph.operations}

    layers_not_in_prototxt = []
    layers = net_param.layer if version >= 2 else net_param.layers
    for layer in layers:
        layer = layer if version >= 1 else layer.layer
        if layer.blobs:
            if layer.name in operations:
                blobs = [_decode_blob(layer.name, idx, blob, graph) for idx, blob in enumerate(layer.blobs)]
                op = operations[layer.name]
                _fix_legacy_blob_ranks(op.name, op.attribs, op.inputs, blobs)
                op.inputs = list(op.inputs) + blobs
            else:
                layers_not_in_prototxt.append(layer.name)

    if layers_not_in_prototxt:
        print("Warning: Weights were found for layers that are not present in the prototxt: {}".format(
            utils.unify_int_and_str_types(layers_not_in_prototxt)))


def _to_numpy(repeated_scalar_container, dtype):
    # Creating a list before conversion makes it much faster
    return np.array(list(repeated_scalar_container), dtype=dtype)


def _decode_blob(name, idx, blob, graph):
    shape = list(blob.shape.dim) if blob.HasField('shape') else [blob.num, blob.channels, blob.height, blob.width]
    data = _to_numpy(blob.data if blob.data else blob.double_data, dtype=np.float32 if blob.data else np.float64)
    if data.size == 1 and shape == [0, 0, 0, 0]:
        shape = []
    return CaffeTensor(graph=graph,
                       name=utils.unify_int_and_str_types(name + '$blob' + str(idx + 1)),
                       shape=utils.unify_int_and_str_types(shape),
                       data=data.reshape(shape))


def _build_blob(tensor, blob):
    data = tensor.data.reshape(-1).tolist()
    if tensor.data.dtype == np.float32:
        blob.data.extend(data)
    elif tensor.data.dtype == np.float64:
        blob.double_data.extend(data)
    else:
        assert False, "Unexpected blob type: {}".format(tensor.data.dtype)
    blob.shape.dim.extend(list(tensor.shape))


def _merge_deprecated(params, attribs, old_name, new_name):
    if params.HasField(old_name) and not params.HasField(new_name):
        attribs[new_name] = attribs[old_name]
    del attribs[old_name]


def _merge_hw_attribs(params, attribs, base, name):
    h_name, w_name = base + '_h', base + '_w'
    h, w = attribs.get(h_name), attribs.get(w_name)

    field = params.DESCRIPTOR.fields_by_name[name]
    repeated = field.label == field.LABEL_REPEATED
    defined = len(attribs[name]) != 0 if repeated else params.HasField(name)

    if not defined and params.HasField(h_name) and params.HasField(w_name):
        attribs[name] = (h, w)

    if h is not None and w is not None:
        del attribs[h_name]
        del attribs[w_name]


def _fix_attrib_rank(attribs, name, rank, default):
    value = attribs.get(name)
    if value is not None:
        if isinstance(value, list):
            if len(value) == 0:
                attribs[name] = (default,) * rank
            elif len(value) == 1:
                attribs[name] = (value[0],) * rank
            elif len(value) != rank:
                raise ValueError("length of '{}' must be {}, found {}".format(name, rank, value))
        elif not isinstance(value, tuple):
            attribs[name] = (value,) * rank


def _unify_attribs_v1_v2(params, attribs, rank):
    if isinstance(params, (caffe_pb.ConvolutionParameter, caffe_pb.PoolingParameter)):
        _merge_hw_attribs(params, attribs, 'pad', 'pad')
        _merge_hw_attribs(params, attribs, 'stride', 'stride')
        _merge_hw_attribs(params, attribs, 'kernel', 'kernel_size')
        if attribs.get('global_pooling') and 'kernel_size' in attribs:
            del attribs['kernel_size']

        _fix_attrib_rank(attribs, 'pad', rank - 2, default=0)
        _fix_attrib_rank(attribs, 'stride', rank - 2, default=1)
        _fix_attrib_rank(attribs, 'dilation', rank - 2, default=1)
        _fix_attrib_rank(attribs, 'kernel_size', rank - 2, default=1)
    elif isinstance(params, caffe_pb.ConcatParameter):
        _merge_deprecated(params, attribs, 'concat_dim', 'axis')
    elif isinstance(params, caffe_pb.SliceParameter):
        _merge_deprecated(params, attribs, 'slice_dim', 'axis')

    return attribs


def _unify_attribs_v0(layer_type_v0, attribs, rank):
    attribs = _attrib_updater_by_layer_type_v0_to_v2[layer_type_v0](attribs)

    if layer_type_v0 in ('conv', 'pool'):
        if attribs.get('global_pooling') and 'kernel_size' in attribs:
            del attribs['kernel_size']

        attribs['pad'] = (attribs['pad'],) * (rank - 2)
        attribs['stride'] = (attribs['stride'],) * (rank - 2)
        if 'kernel_size' in attribs:
            attribs['kernel_size'] = (attribs['kernel_size'],) * (rank - 2)
        if 'dilation' in attribs:
            attribs['dilation'] = (attribs['dilation'],) * (rank - 2)
    elif layer_type_v0 == 'padding':
        attribs['pad'] = (attribs['pad'],) * (rank - 2)

    return attribs


def _split_hw_attribs(attribs, base, name):
    value = attribs.get(name)
    if value is not None and isinstance(value, (list, tuple)):
        assert len(value) == 2, "length of '{}' must be 2, found: {}".format(name, len(value))
        del attribs[name]
        attribs[base + '_h'], attribs[base + '_w'] = value


def _normalize_attribs(op, attribs):
    if op == 'Pooling':
        _split_hw_attribs(attribs, 'pad', 'pad')
        _split_hw_attribs(attribs, 'stride', 'stride')
        if not attribs.get('global_pooling'):
            _split_hw_attribs(attribs, 'kernel', 'kernel_size')


def _get_version(net_param):
    if any(layer.HasField('layer') for layer in net_param.layers):
        return 0
    elif len(net_param.layers) > 0:
        return 1
    else:
        return 2


class Reader(object):

    def __init__(self, custom_shapes=None):
        if custom_shapes is None:
            custom_shapes = {}
        self._custom_shapes = custom_shapes

    def __call__(self, graph_filename, model_filename=None):
        net_param = caffe_pb.NetParameter()
        _read_net_from_prototxt(graph_filename, net_param)
        graph = _decode_graph(net_param, self._custom_shapes)
        if model_filename:
            _read_blobs_from_caffemodel(model_filename, graph)
        return graph


class Writer(object):

    def __init__(self):
        pass

    def __call__(self, graph, filename):
        graph.generate_missing_names()
        if not graph.is_sorted():
            graph.sort()

        if filename.endswith('.prototxt'):
            model_filename = filename[:-len('.prototxt')] + '.caffemodel'
        else:
            model_filename = filename + '.caffemodel'

        for operation in graph.operations:
            _normalize_attribs(operation.name, operation.attribs)

        utils.makedirs(os.path.dirname(model_filename), exist_ok=True)

        net_param = caffe_pb.NetParameter()
        _build_caffemodel_from_graph(graph, net_param)
        with open(model_filename, 'wb') as file:
            file.write(net_param.SerializeToString())

        net_param = caffe_pb.NetParameter()
        _build_caffemodel_from_graph(graph, net_param, with_weights=False)
        with open(filename, 'w') as file:
            file.write(str(net_param))
