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


import os
from abstractnet import *

GPU_MODE = False

def getparams(self, proto):
    self.top = []
    self.top.append(proto.top[0])
    self.bottom = []
    for b in proto.bottom:
        self.bottom.append(b)
    self.name = proto.name


def getDilation(self, proto):
    d = proto.convolution_param.dilation
    if not d:
        d = [1]
    self.dilation = [int(d[0])]

def getPadding(self, proto):
    pad_w = proto.convolution_param.pad_w
    pad_h = proto.convolution_param.pad_h
    if not pad_w and not pad_h:
        pad_w = proto.convolution_param.pad
        pad_h = proto.convolution_param.pad
    if not pad_w and not pad_h:
        pad_w = proto.pooling_param.pad
        pad_h = proto.pooling_param.pad
    if not pad_w and not pad_h:
        pad_w = [0]
        pad_h = [0]
    if isinstance(pad_w, int):
        pad_w = [pad_w]
        pad_h = [pad_h]
    self.padding = [
        0,
        0,
        int(pad_h[0]),
        int(pad_w[0])
    ]


def getPads(operation, proto, bigger, smaller, with_dilation = False):
    getPadding(operation, proto)
    pad_left = operation.padding[-1]
    pad_top = operation.padding[-2]
    stride = operation.stride[-1]
    size_w = int(operation.size[-1]) - stride
    size_h = int(operation.size[-2]) - stride
    dilation_extra_w = 0
    dilation_extra_h = 0
    if with_dilation:
        getDilation(operation,proto)
        dilation_extra_w = (int(operation.size[-1])-1)*(operation.dilation[0]-1)
        dilation_extra_h = (int(operation.size[-2])-1)*(operation.dilation[0]-1)
    pad_right = smaller.data.shape[-1]*stride - bigger.data.shape[-1] + size_w - pad_left + dilation_extra_w
    pad_bottom = smaller.data.shape[-2]*stride - bigger.data.shape[-2] + size_h - pad_top + dilation_extra_h
    operation.pads = [int(pad_top), int(pad_bottom), int(pad_left), int(pad_right)]


def getStride(self, stride):
    if not stride:
        stride = [1]
    if not isinstance(stride, int):
        stride = stride[0]
    self.stride = [
        1,
        1,
        int(stride),
        int(stride)
    ]


Operation.getparams = getparams
Operation.getPadding = getPadding
Operation.getStride = getStride


def createMerge(proto, net, n_instance):
    s = MergeOperation()
    getparams(s, proto)
    s.axis = 2
    net.operations.append(s)


def createEltwise(proto, net, n_instance):
    s = AddOperation()
    getparams(s, proto)
    net.operations.append(s)


def createLRN(proto, net, n_instance):
    s = LrnOperation()
    getparams(s, proto)
    s.size = [1, proto.lrn_param.local_size]
    s.bias = 1.0
    s.alpha = proto.lrn_param.alpha
    s.beta = proto.lrn_param.beta
    net.operations.append(s)


def createElu(proto, net, n_instance):
    s = ELUOperation()
    getparams(s, proto)
    s.alpha = proto.elu_param.alpha
    net.operations.append(s)


def createRelu(proto, net, n_instance):
    s = ReLUOperation()
    getparams(s, proto)
    s.negative_slope = proto.relu_param.negative_slope
    net.operations.append(s)


def createPool(proto, net, n_instance):
    s = PoolOperation()
    getparams(s, proto)
    s.size = [
        1,
        1,
        int(proto.pooling_param.kernel_size),
        int(proto.pooling_param.kernel_size)
    ]
    getStride(s, proto.pooling_param.stride)
    getPads(s,proto,n_instance.blobs[s.bottom[0]],n_instance.blobs[s.top[0]])
    pool_types = ["max", "avg"]
    s.pool = pool_types[proto.pooling_param.pool]
    net.operations.append(s)


def createDeconvInterp(proto, net, n_instance):
    s = InterpOperation()
    getparams(s, proto)
    s.upsample_stride = int(proto.convolution_param.stride[0])
    net.operations.append(s)


def createDeconv(proto, net, n_instance, deconv_as_resamp):
    if deconv_as_resamp and (not proto.convolution_param.bias_term) and proto.convolution_param.weight_filler.type == "bilinear":
        createDeconvInterp(proto, net, n_instance)
        return
    s = DeconvOperation()
    getparams(s, proto)
    s.use_bias = proto.convolution_param.bias_term
    bottomsize = n_instance.blobs[s.bottom[0]].data.shape[1]
    topsize = proto.convolution_param.num_output
    s.size = [
        int(topsize),
        int(bottomsize),
        int(proto.convolution_param.kernel_size[0]),
        int(proto.convolution_param.kernel_size[0])
    ]
    group = proto.convolution_param.group
    if group:
        s.groups = int(group)
    getStride(s, proto.convolution_param.stride)
    getPads(s,proto,n_instance.blobs[s.top[0]],n_instance.blobs[s.bottom[0]],True)
    net.operations.append(s)


def createInnerProduct(proto, net, n_instance):
    s = ConvOperation()
    getparams(s, proto)
    s.use_bias = proto.inner_product_param.bias_term
    bottomsize = n_instance.blobs[s.bottom[0]].data.shape[1]
    w = 1
    h = 1
    if len(n_instance.blobs[s.bottom[0]].data.shape) == 4:
        w = n_instance.blobs[s.bottom[0]].data.shape[3]
        h = n_instance.blobs[s.bottom[0]].data.shape[2]
    topsize = proto.inner_product_param.num_output
    s.size = [
        int(topsize),
        int(bottomsize),
        int(h),
        int(w)
    ]
    s.stride = [1, 1, 1, 1]
    s.padding = [0, 0, 0, 0]
    s.pads = s.padding
    net.operations.append(s)


def createInterp(proto, net, n_instance):
    s = InterpOperation()
    getparams(s, proto)
    bottom_size = n_instance.blobs[s.bottom[0]].data.shape[-1]
    top_size = n_instance.blobs[s.top[0]].data.shape[-1]
    s.upsample_stride = int(top_size/bottom_size)
    net.operations.append(s)


def createBatchNorm(proto, net, n_instance):
    s = ScaleOperation()
    getparams(s, proto)
    s.channels = int(n_instance.params[s.name][0].data.shape[0])
    s.use_bias = True
    s._caffe_batchnorm_convert = True
    net.operations.append(s)


def createScale(proto, net, n_instance):
    s = ScaleOperation()
    getparams(s, proto)
    s.channels = int(n_instance.params[s.name][0].data.shape[0])
    s.use_bias = proto.scale_param.bias_term
    net.operations.append(s)


def createPower(proto, net, n_instance):
    s = PowerOperation()
    getparams(s, proto)
    s.power = proto.power_param.power
    s.scale = proto.power_param.scale
    s.shift = proto.power_param.shift
    net.operations.append(s)


def createBNLL(proto, net, n_instance):
    s = BNLLOperation()
    getparams(s, proto)
    net.operations.append(s)


def createConv(proto, net, n_instance):
    s = ConvOperation()
    getparams(s, proto)
    s.use_bias = proto.convolution_param.bias_term
    bottomsize = n_instance.blobs[s.bottom[0]].data.shape[1]
    h = 1
    w = 1
    if len(proto.convolution_param.kernel_size) != 0:
        h = proto.convolution_param.kernel_size[0]
        w = proto.convolution_param.kernel_size[0]
    else:
        h = proto.convolution_param.kernel_h
        w = proto.convolution_param.kernel_w
    topsize = proto.convolution_param.num_output
    s.size = [
        int(topsize),
        int(bottomsize),
        int(h),
        int(w)
    ]
    getStride(s, proto.convolution_param.stride)
    getPads(s,proto,n_instance.blobs[s.bottom[0]],n_instance.blobs[s.top[0]],True)
    group = proto.convolution_param.group
    if group:
        s.groups = int(group)
    net.operations.append(s)


def createTanH(proto, net, n_instance):
    s = TanhOperation()
    getparams(s, proto)
    net.operations.append(s)


def createSoftmax(proto, net, n_instance):
    s = SoftmaxOperation()
    getparams(s, proto)
    net.operations.append(s)


def createReshape(proto, net, n_instance):
    s = ReshapeOperation()
    getparams(s, proto)
    dim = proto.reshape_param.shape.dim
    s.shape = [int(dim[0]), int(dim[1]), int(dim[2]), int(dim[3])]
    net.operations.append(s)


def createAbsVal(proto, net, n_instance):
    s = AbsOperation()
    getparams(s, proto)
    net.operations.append(s)


def createSigmoid(proto, net, n_instance):
    s = SigmoidOperation()
    getparams(s, proto)
    net.operations.append(s)


def createFlatten(proto, net, n_instance):
    s = ConvOperation()
    getparams(s, proto)
    bottomsize = n_instance.blobs[s.bottom[0]].data.shape[1]
    topsize = bottomsize
    s.size = [
        int(topsize),
        int(bottomsize),
        int(net.data[s.bottom[0]][2]),
        int(net.data[s.bottom[0]][3])
    ]
    s.stride = [1, 1, 1, 1]
    s.padding = [0, 0, 0, 0]
    net.operations.append(s)


def createInput(proto, net, n_instance):
    if len(proto.top) > 0:
        s = InputOperation()
        s.top = []
        s.top.append(proto.top[0])
        s.bottom = []
        s.name = proto.name
        s.size = n_instance.blobs[proto.top[0]].data.shape
        net.operations.append(s)

def createArgmax(proto, net, n_instance):
    if proto.argmax_param.top_k == 1 and proto.argmax_param.out_max_val == False:
        s = ArgmaxOperation()
        getparams(s, proto)
        axis = int(proto.argmax_param.axis)
        bottomsize = n_instance.blobs[s.bottom[0]].data.shape
        s.size = []
        for i in range(len(bottomsize)):
            if i == axis:
                s.size.append(int(bottomsize[i]))
            else:
                s.size.append(1)
        net.operations.append(s)
    else:
        log("Argmax params not supported yet")


import subprocess as _subprocess
from tempfile import NamedTemporaryFile as _NamedTemporaryFile
import google.protobuf.text_format as _gprototext
import caffe.proto.caffe_pb2 as _caffe_pb2
import caffe
import numpy as np
import os.path

def read_heatmaps_data(net, n):
    names = []
    for op in net.operations:
        for b in op.top + op.bottom:
            if not b in names:
                net.heatmap_data[b] = n.blobs[b].data.copy()
                names.append(b)
    for h in net.heatmap_data.keys():
        log(h + " " + str(net.heatmap_data[h].shape))

def read_weights(net, n):
    net.weight_data = True
    for op in net.operations:
        if isinstance(op, ConvOperation) or isinstance(op, ScaleOperation) or isinstance(op, DeconvOperation):
            if op._caffe_batchnorm_convert:
                eps = 1e-8
                var = n.params[op.name][1].data
                mean = n.params[op.name][0].data
                norm = 1 / n.params[op.name][2].data[0]
                var_normed = var * norm + eps
                mean_normed = mean * norm
                op.weight_data['filter'] = 1 / np.sqrt(var_normed)
                if op.use_bias:
                    op.weight_data['bias'] = - mean_normed / np.sqrt(var_normed)
            else:
                op.weight_data['filter'] = n.params[op.name][0].data
                if op.use_bias:
                    op.weight_data['bias'] = n.params[op.name][1].data
        if isinstance(op, ConvOperation) and len(op.weight_data['filter'].shape) != 4:  # InnerProduct
            op.weight_data['filter'] = op.weight_data['filter'].reshape(op.size)

def replace_buffer_names(net):
    for key in net.buffer_names_to_replace.keys():
        for op in net.operations:
            for i in range(len(op.bottom)):
                if str(op.bottom[i]) == key:
                    op.bottom[i] = net.buffer_names_to_replace[key]
            for i in range(len(op.top)):
                if str(op.top[i]) == key:
                    op.top[i] = net.buffer_names_to_replace[key]


def buildNet(fname_prototxt, fname_caffemodel, deconv_as_resamp=True, forward=False):
    fname = fname_prototxt.split(".prototxt")[0]
    if os.path.isfile(fname_caffemodel):
        n_instance = caffe.Net(fname_prototxt, fname_caffemodel, caffe.TEST)
    else:
        n_instance = caffe.Net(fname_prototxt, caffe.TEST)
        n_instance.save(fname+".caffemodel")
    if forward:
        n_instance.forward()
    upgrade_bin = "/usr/bin/upgrade_net_proto_text"
    if not os.path.exists(upgrade_bin):
        upgrade_bin = os.path.join(os.environ['CAFFE_BIN_FOLDER'], "upgrade_net_proto_text.bin")
    with _NamedTemporaryFile(mode='r', suffix='.prototxt') as tmpfile:
        _subprocess.check_call([upgrade_bin,
                                fname + ".prototxt",
                                tmpfile.name])
        text = tmpfile.read()
    n = _caffe_pb2.NetParameter()
    _gprototext.Merge(text, n)
    layers = n.ListFields()[-1][1]
    net = AbstractNet(fname.split("/")[-1])
    net.buffer_names_to_replace = {}
    for i in range(0, len(layers)):
        l = layers[i]
        log(l.type + " " + l.name)
        if l.type == "Input" or l.type == "Python" or l.type == "Data":
            createInput(l, net, n_instance)
        elif l.type == "Interp":
            createInterp(l, net, n_instance)
        elif l.type == "LRN":
            createLRN(l, net, n_instance)
        elif l.type == "Convolution":
            createConv(l, net, n_instance)
        elif l.type == "Pooling":
            createPool(l, net, n_instance)
        elif l.type == "Deconvolution":
            createDeconv(l, net, n_instance, deconv_as_resamp)
        elif l.type == "ELU":
            createElu(l, net, n_instance)
        elif l.type == "ReLU":
            createRelu(l, net, n_instance)
        elif l.type == "Concat":
            createMerge(l, net, n_instance)
        elif l.type == "Flatten":
            createFlatten(l, net, n_instance)
        elif l.type == "Softmax":
            createSoftmax(l, net, n_instance)
        elif l.type == "Sigmoid":
            createSigmoid(l, net, n_instance)
        elif l.type == "AbsVal":
            createAbsVal(l, net, n_instance)
        elif l.type == "TanH":
            createTanH(l, net, n_instance)
        elif l.type == "Eltwise":
            createEltwise(l, net, n_instance)
        elif l.type == "BatchNorm":
            createBatchNorm(l, net, n_instance)
        elif l.type == "Scale":
            createScale(l, net, n_instance)
        elif l.type == "Power":
            createPower(l, net, n_instance)
        elif l.type == "Dropout":
            log("Dropout ignored")
        elif l.type == "InnerProduct":
            createInnerProduct(l, net, n_instance)
        elif l.type == "BNLL":
            createBNLL(l, net, n_instance)
        elif l.type == "Reshape":
            createReshape(l, net, n_instance)
        elif l.type == "ArgMax":
            createArgmax(l, net, n_instance)
        elif l.type == "Crop":
            net.buffer_names_to_replace[l.top[0]] = l.bottom[0]
        else:
            log("============= NOT IMPLEMENTED YET ==============")
            log(l.type)
            log(l.argmax_param)
    read_heatmaps_data(net, n_instance)
    read_weights(net, n_instance)
    replace_buffer_names(net)
    return net


if GPU_MODE:
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    log("not_gpu")
    caffe.set_mode_cpu()
