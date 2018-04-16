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


from export_from_caffe import *
import struct
import tarfile
import shutil

def prettyprint(self, variable):
    param = getattr(self,variable)
    if isinstance(param, basestring):
        param = "'"+param+"'"
    else:
        param = str(param).lower()
    return variable + " = " + param

def nnef_variables(self):
    return ""

def nnef_variable_signature(name, label, shape):
    return name + " = variable(label = '" + label + "', shape = " + str(shape) + ")"

def nnef_weight_variables_signature(op_name, names_shapes_dict):
    h = ""
    for key in names_shapes_dict.keys():
        l = []
        for s in names_shapes_dict[key]:
            l.append(int(s))
        h = h + nnef_variable_signature(op_name + "_" + key, op_name + "/" + key, l) + "\r\n"
    return h[:-2]

def nnef_signature(self, output, variables, variables_to_pretty):
    sig = self.nnef_signature_name() + "("
    for variable in variables:
        if isinstance(variable, list):
            sig = sig + "["
            for v in variable:
                sig = sig + v + ", "
            sig = sig[:-2]
            sig = sig + "], "
        else:
            sig = sig + variable + ", "
    for variable in variables_to_pretty:
        sig = sig + self.prettyprint(variable) + ", "
    sig = sig[:-2] + ")"
    return output+" = "+sig

def nnef_signature_name(self):
    return ""

def nnef_standard_InputOperation(self):
    return "data = external(shape="+str(list(self.size))+")"

def nnef_signature_name_SplitOperation(self):
    return "part"
def nnef_standard_SplitOperation(self):
    self.axis = 1
    return self.nnef_signature(self.top, [self.bottom[0]], ["axis"])

def nnef_signature_name_InterpOperation(self):
    return "multilinear_upsample"
def nnef_standard_InterpOperation(self):
    self.factor = [self.upsample_stride, self.upsample_stride]
    self.method="symmetric"
    self.border="constant"
    return self.nnef_signature(self.top[0], [self.bottom[0]], ["factor", "method", "border"])

def nnef_signature_name_RescaleOperation(self):
    return "rescale"
def nnef_standard_RescaleOperation(self):
    self.strides = [self.stride, self.stride]
    return self.nnef_signature(self.top[0], [self.bottom[0]], ["strides"])

def nnef_signature_name_LrnOperation(self):
    return "local_response_normalization"
def nnef_standard_LrnOperation(self):
    return self.nnef_signature(self.top[0], [self.bottom[0]], ["size", "alpha", "beta", "bias"])

def nnef_signature_name_BatchNormOperation(self):
    return "batch_normalization"
def nnef_variables_BatchNormOperation(self):
    size = [1,self.weight_data["mean"].shape[0]]
    d = {
        "variance": size,
        "mean": size,
        "offset": size,
        "scale": size,
    }
    return nnef_weight_variables_signature(self.name, d)
def nnef_standard_BatchNormOperation(self):
    self.epsilon = 0.00001
    self.scope = self.name
    header = self.nnef_signature(self.top[0], [self.bottom[0], self.name + "_mean", self.name + "_variance", self.name + "_offset", self.name + "_scale"], ["epsilon"])
    return header

def nnef_signature_name_ScaleOperation(self):
    return ""
def nnef_variables_ScaleOperation(self):
    size = [1,self.channels]
    d = {"filter": size}
    if self.use_bias:
        d["bias"] = size
    return nnef_weight_variables_signature(self.name, d)
def nnef_standard_ScaleOperation(self):
    self.scope = self.name
    if not self.use_bias:
        header = self.top[0] + " = " + self.bottom[0] + " * " + self.name + "_filter"
    else:
        header = self.top[0] + " = " + self.bottom[0] + " * " + self.name + "_filter" + " + " + self.name + "_bias"
    return header

def nnef_signature_name_PowerOperation(self):
    return "power_scale_shift"
def nnef_standard_PowerOperation(self):
    return self.nnef_signature(self.top[0], [self.bottom[0]], ["power", "scale", "shift"])

def nnef_signature_name_DeconvOperation(self):
    return "deconv"
def nnef_standard_DeconvOperation(self):
    self.scope = self.name
    self.channels = self.size[0]
    self.size = self.size[2:]
    self.padding = [(self.pads[0],self.pads[1]),(self.pads[2],self.pads[3])]
    self.stride = self.stride[2:]
    self.padding = [(self.pads[0], self.pads[1]), (self.pads[2], self.pads[3])]
    header = self.nnef_signature(self.top[0], [self.bottom[0], self.name + "_filter"], ["stride", "padding", "groups"])
    if self.use_bias:
        header = self.nnef_signature(self.top[0], [self.bottom[0], self.name + "_filter", self.name + "_bias"], ["stride", "padding", "groups"])
    return header
def nnef_variables_DeconvOperation(self):
    size = self.size
    size = [size[0],size[1]/self.groups,size[2],size[3]]
    bsize = [1,self.size[0]]
    d = {"filter": size}
    if self.use_bias:
        d["bias"] = bsize
    return nnef_weight_variables_signature(self.name, d)

def nnef_signature_name_ConvOperation(self):
    return "conv"
def nnef_standard_ConvOperation(self):
    self.scope = self.name
    self.channels = self.size[0]
    self.size = self.size[2:]
    self.padding = [(self.pads[0],self.pads[1]),(self.pads[2],self.pads[3])]
    self.stride = self.stride[2:]
    header = self.nnef_signature(self.top[0], [self.bottom[0], self.name + "_filter"], [ "stride", "padding", "groups"])
    if self.use_bias:
        header = self.nnef_signature(self.top[0], [self.bottom[0], self.name + "_filter", self.name + "_bias"], [ "stride", "padding", "groups"])
    return header
def nnef_variables_ConvOperation(self):
    size = self.size
    size = [size[0],size[1]/self.groups,size[2],size[3]]
    bsize = [1,self.size[0]]
    d = {"filter": size}
    if self.use_bias:
        d["bias"] = bsize
    return nnef_weight_variables_signature(self.name, d)


def nnef_signature_name_PoolOperation(self):
    return self.pool+"_pool"
def nnef_standard_PoolOperation(self):
    if hasattr(self,'padding'):
        self.padding = [(0,0),(0,0),(self.pads[0],self.pads[1]),(self.pads[2],self.pads[3])]
        return self.nnef_signature(self.top[0], [self.bottom[0]], ["size", "stride", "padding"])
    return self.nnef_signature(self.top[0], [self.bottom[0]], ["size", "stride"])

def nnef_signature_name_ReLUOperation(self):
    return "relu" if (not self.leaky() ) else "leaky_relu"
def nnef_standard_ReLUOperation(self):
    if not self.leaky():
        return self.nnef_signature(self.top[0], [self.bottom[0]], [])
    return self.nnef_signature(self.top[0], [self.bottom[0]], ["negative_slope"])

def nnef_signature_name_BNLLOperation(self):
    return "binomial_normal_log_likelihood"
def nnef_standard_BNLLOperation(self):
    return self.nnef_signature(self.top[0], [self.bottom[0]], [])

def nnef_signature_name_ReshapeOperation(self):
    return "reshape"
def nnef_standard_ReshapeOperation(self):
    return self.nnef_signature(self.top[0], [self.bottom[0]], ["shape"])

def nnef_signature_name_SoftmaxOperation(self):
    return "softmax"
def nnef_standard_SoftmaxOperation(self):
    return self.nnef_signature(self.top[0], [self.bottom[0]], [])

def nnef_signature_name_ArgmaxOperation(self):
    return "argmax"
def nnef_standard_ArgmaxOperation(self):
    return self.nnef_signature(self.top[0], [self.bottom[0]], ["size"])

def nnef_signature_name_TanhOperation(self):
    return "tanh"
def nnef_standard_TanhOperation(self):
    return self.nnef_signature(self.top[0], [self.bottom[0]], [])

def nnef_signature_name_AbsOperation(self):
    return "abs"
def nnef_standard_AbsOperation(self):
    return self.nnef_signature(self.top[0], [self.bottom[0]], [])

def nnef_signature_name_SigmoidOperation(self):
    return "sigmoid"
def nnef_standard_SigmoidOperation(self):
    return self.nnef_signature(self.top[0], [self.bottom[0]], [])

def nnef_signature_name_AddOperation(self):
    return "add"
def nnef_standard_AddOperation(self):
    return self.nnef_signature(self.top[0], self.bottom, [])
def nnef_variables_AddOperation(self):
    d = {}
    for key in self.weight_data:
        d[key] = [1,self.weight_data[key].shape[0]]
    return nnef_weight_variables_signature(self.name, d)

def nnef_signature_name_MulOperation(self):
    return "mul"
def nnef_standard_MulOperation(self):
    return self.nnef_signature(self.top[0], self.bottom, [])
def nnef_variables_MulOperation(self):
    d = {}
    for key in self.weight_data:
        d[key] = [1,self.weight_data[key].shape[0]]
    return nnef_weight_variables_signature(self.name, d)

def nnef_signature_name_MergeOperation(self):
    return "concat"
def nnef_standard_MergeOperation(self):
    self.axis = 1
    bottoms = "["
    for b in self.bottom[:-1]:
        bottoms = bottoms + b + ","
    bottoms = bottoms + self.bottom[-1] + "]"
    return self.nnef_signature(self.top[0], [bottoms], ["axis"])

def nnef_standard_operations(self):
    header = ""
    for op in self.operations:
        header = header + "\r\n" + op.nnef_standard()
    return header
def nnef_standard(self, outputs=None):
    # build ops
    s = ""
    for f in self.operations:
        v = f.nnef_variables()
        if len(v) > 3:
            s = s + v + "\r\n"
    s = s + self.nnef_standard_operations()
    tops = []
    all_tops = []
    # SEARCH FOR TOPS
    for op in self.operations:
        for ftop in op.top:
            all_tops.append(ftop)
            output = True
            for op in self.operations:
                output = output and not (ftop in op.bottom)
            if output:
                tops.append(ftop)
    topstring = tops[0].replace("/","_")
    if len(tops) > 1:
        for top in tops[1:]:
            topstring = topstring + ", " + top.replace("/","_")
    s = s.replace("\r\n","\r\n    ")
    if outputs:
        for o in outputs:
            if not o in all_tops:
                log( "Invalid output: " + o)
                exit()
        outputstring = outputs[0]
        for i in range(len(outputs)-1):
            outputstring += ", " + outputs[i+1]
    else:
        outputstring = self.operations[-1].top[0]
    s = "version 1.0\r\n\r\ngraph "+self.name+"( "+self.operations[0].top[0]+" ) -> ( "+outputstring+" )\r\n{\r\n    "+s+"\r\n}"
    return s

def save_nnef_tensor(filename, tensor):
    file = open(filename, "wb")
    file.write(struct.pack('c', b'N'))
    file.write(struct.pack('c', b'\xEF')) # MAGIC NUMBER
    file.write(struct.pack("B", 1))
    file.write(struct.pack("B", 0)) # VERSION
    data_offset = 12 + len(tensor.shape) * 4 + 4
    file.write(struct.pack('i', data_offset))
    file.write(struct.pack('i', len(tensor.shape)))
    for s in tensor.shape:
        file.write(struct.pack('i', s))
    file.write(struct.pack("B",0))
    file.write(struct.pack("B",32))
    file.write(struct.pack("B",0))
    file.write(struct.pack("B",0))
    file.write(tensor.flatten())
    file.close()

def save_operation_to_nnef_bin(self, operation):
    dirname = self.name + "/" + operation.name
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    wd = operation.weight_data
    for item in wd.items():
        filename = dirname + "/" + item[0] + ".dat"
        param = item[1]
        log(filename)
        log(param.shape)
        if len(param.shape) == 2:
            param = param.reshape(param.shape[0], param.shape[1], 1, 1)
        if item[0] == "bias" or isinstance(operation, ScaleOperation) or isinstance(operation, BatchNormOperation):
            param = param.reshape(1, param.shape[0])
        save_nnef_tensor(filename, param)

def save_nnef_bins_weights(self):
    for f in self.operations:
        if isinstance(f, WeightedOperation):
            self.save_operation_to_nnef_bin(f)

def save_nnef_bins_heatmaps(self):
    dirname = self.name + "/heatmaps"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for h in self.heatmap_data.keys():
        hs = h.replace("/","_")
        save_nnef_tensor(self.name + "/heatmaps/" + hs + ".dat", self.heatmap_data[h])

def dir_to_targz(output_path):
    filename = output_path + '.tgz'
    tar = tarfile.open(filename, 'w:gz')
    for file in os.listdir(output_path):
        tar.add(output_path + '/' + file, file)
    tar.close()
    shutil.rmtree(output_path)


AbstractNet.nnef_standard_operations = nnef_standard_operations
AbstractNet.nnef_standard = nnef_standard
AbstractNet.save_operation_to_nnef_bin = save_operation_to_nnef_bin
AbstractNet.save_nnef_bins_weights = save_nnef_bins_weights
AbstractNet.save_nnef_bins_heatmaps = save_nnef_bins_heatmaps


Operation.prettyprint = prettyprint
Operation.nnef_signature = nnef_signature
Operation.nnef_variables =  nnef_variables
Operation.nnef_signature_name =  nnef_signature_name

InputOperation.nnef_standard =  nnef_standard_InputOperation
SplitOperation.nnef_standard =  nnef_standard_SplitOperation
SplitOperation.nnef_signature_name =  nnef_signature_name_SplitOperation
InterpOperation.nnef_standard =  nnef_standard_InterpOperation
InterpOperation.nnef_signature_name =  nnef_signature_name_InterpOperation
RescaleOperation.nnef_standard =  nnef_standard_RescaleOperation
RescaleOperation.nnef_signature_name =  nnef_signature_name_RescaleOperation
LrnOperation.nnef_standard =  nnef_standard_LrnOperation
LrnOperation.nnef_signature_name =  nnef_signature_name_LrnOperation
BatchNormOperation.nnef_variables =  nnef_variables_BatchNormOperation
BatchNormOperation.nnef_standard =  nnef_standard_BatchNormOperation
BatchNormOperation.nnef_signature_name =  nnef_signature_name_BatchNormOperation
ScaleOperation.nnef_variables =  nnef_variables_ScaleOperation
ScaleOperation.nnef_standard =  nnef_standard_ScaleOperation
ScaleOperation.nnef_signature_name =  nnef_signature_name_ScaleOperation
PowerOperation.nnef_standard =  nnef_standard_PowerOperation
PowerOperation.nnef_signature_name =  nnef_signature_name_PowerOperation
ConvOperation.nnef_variables =  nnef_variables_ConvOperation
ConvOperation.nnef_standard =  nnef_standard_ConvOperation
ConvOperation.nnef_signature_name =  nnef_signature_name_ConvOperation
DeconvOperation.nnef_variables =  nnef_variables_DeconvOperation
DeconvOperation.nnef_standard =  nnef_standard_DeconvOperation
DeconvOperation.nnef_signature_name =  nnef_signature_name_DeconvOperation
PoolOperation.nnef_standard =  nnef_standard_PoolOperation
PoolOperation.nnef_signature_name =  nnef_signature_name_PoolOperation
ReLUOperation.nnef_standard =  nnef_standard_ReLUOperation
ReLUOperation.nnef_signature_name =  nnef_signature_name_ReLUOperation
SoftmaxOperation.nnef_standard =  nnef_standard_SoftmaxOperation
SoftmaxOperation.nnef_signature_name =  nnef_signature_name_SoftmaxOperation
ArgmaxOperation.nnef_standard =  nnef_standard_ArgmaxOperation
ArgmaxOperation.nnef_signature_name =  nnef_signature_name_ArgmaxOperation
TanhOperation.nnef_standard =  nnef_standard_TanhOperation
TanhOperation.nnef_signature_name =  nnef_signature_name_TanhOperation
AbsOperation.nnef_standard =  nnef_standard_AbsOperation
AbsOperation.nnef_signature_name =  nnef_signature_name_AbsOperation
ReshapeOperation.nnef_standard =  nnef_standard_ReshapeOperation
ReshapeOperation.nnef_signature_name =  nnef_signature_name_ReshapeOperation
SigmoidOperation.nnef_standard =  nnef_standard_SigmoidOperation
SigmoidOperation.nnef_signature_name =  nnef_signature_name_SigmoidOperation
BNLLOperation.nnef_standard =  nnef_standard_BNLLOperation
BNLLOperation.nnef_signature_name =  nnef_signature_name_BNLLOperation
AddOperation.nnef_standard =  nnef_standard_AddOperation
AddOperation.nnef_signature_name =  nnef_signature_name_AddOperation
AddOperation.nnef_variables =  nnef_variables_AddOperation
MulOperation.nnef_standard =  nnef_standard_MulOperation
MulOperation.nnef_signature_name =  nnef_signature_name_MulOperation
MulOperation.nnef_variables =  nnef_variables_MulOperation
MergeOperation.nnef_standard =  nnef_standard_MergeOperation
MergeOperation.nnef_signature_name =  nnef_signature_name_MergeOperation
