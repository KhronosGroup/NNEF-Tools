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


import re
import sys


if(sys.version_info[0] > 2):
    basestring = str

def log(s):
    print(s)

class AbstractNet:
    def __init__(self, name):
        self.operations = []
        self.name = name
        self.weight_data = False
        self.heatmap_data = {}
    def clone(self):
        result = AbstractNet(self.name)
        result.weight_data = self.weight_data
        result.heatmap_data = self.heatmap_data
        for op in self.operations:
            result.operations.append(op)
        return result
    def replace_forbidden_characters(self):
        self.name = re.sub(r'\W', '_', self.name)
        for op in self.operations:
            if hasattr(op,'name'):
                op.name = re.sub(r'\W', '_', op.name)
            for i in range(0,len(op.bottom)):
                op.bottom[i] = re.sub(r'\W', '_', op.bottom[i])
            for i in range(0,len(op.top)):
                op.top[i] = re.sub(r'\W', '_', op.top[i])
    def resolve_inplace_operations(self):
        for i in range(len(self.operations)):
            if self.operations[i].in_place():
                old_name = self.operations[i].top[0]
                new_name = self.operations[i].top[0] + ""
                conflict = True
                while conflict:
                    new_name = new_name + "_tmp"
                    conflict = False
                    for o in self.operations:
                        if new_name in o.bottom or new_name in o.top:
                            conflict = True
                self.operations[i].top[0] = new_name
                for j in range(i+1, len(self.operations)):
                    for k in range(len(self.operations[j].top)):
                        if self.operations[j].top[k] == old_name:
                            self.operations[j].top[k] = new_name
                    for k in range(len(self.operations[j].bottom)):
                        if self.operations[j].bottom[k] == old_name:
                            self.operations[j].bottom[k] = new_name
    def merge_batchnorm_operations(self):
        for op in self.operations:
            if isinstance(op, ScaleOperation):
                op.mergeToBatchNorm(self)
    def convert_scalebias_to_mul_add(self):
        for op in self.operations:
            if isinstance(op, ScaleOperation):
                op.convertToMulAdd(self)
    def convert_global_pooling_to_reduce(self):
        for op in self.operations:
            if isinstance(op, PoolOperation):
                op.convertToReduce(self)



class Operation:
    def __init__(self):
        self._caffe_batchnorm_convert = False
        self.groups = 1
        self.top = []
        self.bottom = []
        self.name = ""
    def copyTo(self, result):
        result.groups = self.groups
        result.top.append(self.top[0])
        for b in self.bottom:
            result.bottom.append(b)
        result.name = self.name
    def in_place(self):
        if len(self.bottom) > 0 and len(self.top) > 0:
            return self.bottom[0] == self.top[0]
        return False

class WeightedOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
        self.weight_data = {}
        self.size = []
        self.stride = []
        self.padding = []
        self.pads = []
        self.bias = False
        self.use_bias = False
    def copyTo(self, result):
        Operation.copyTo(self, result)
        result.size = []
        for s in self.size:
            result.size.append(s)
        result.stride = []
        for s in self.stride:
            result.stride.append(s)
        result.padding = []
        for s in self.padding:
            result.padding.append(s)
        result.pads = self.pads
        result.bias = self.bias
        result.use_bias = self.use_bias
        result.weight_data = {}
        for w in self.weight_data:
            result.weight_data[w] = self.weight_data[w].copy()


class InputOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
        self.size = []
    def copy(self):
        result = InputOperation()
        result.size = self.size
        Operation.copyTo(self, result)
        return result

class SplitOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
        self.axis = 1
    def copy(self):
        result = SplitOperation()
        Operation.copyTo(self, result)
        result.axis = self.axis
        return result

class InterpOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
        self.pads = []
        self.stride = -1
        self.upsample_stride = -1
    def copy(self):
        result = InterpOperation()
        Operation.copyTo(self, result)
        result.pads = self.pads
        result.stride = self.stride
        result.upsample_stride = self.upsample_stride
        return result

class RescaleOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
        self.upsample_stride = -1
    def copy(self):
        result = RescaleOperation()
        Operation.copyTo(self, result)
        result.upsample_stride = self.upsample_stride
        return result

class LrnOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
        self.size = []
        self.alpha = 1
        self.beta = 1
        self.bias = 1
    def copy(self):
        result = LrnOperation()
        Operation.copyTo(self, result)
        for s in self.size:
            result.size.append(s)
        result.alpha = self.alpha
        result.beta = self.beta
        result.bias = self.bias
        return result

class BatchNormOperation(WeightedOperation):
    def __init__(self):
        WeightedOperation.__init__(self)
        self.epsilon = 1e-8
    def copyTo(self, result):
        WeightedOperation.copyTo(self, result)
        return result
    def copy(self):
        result = BatchNormOperation()
        self.copyTo(result)
        return result

class PowerOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
        self.power = 1.0
        self.scale = 1.0
        self.shift = 0.0
    def copyTo(self, result):
        Operation.copyTo(self, result)
        result.power = self.power
        result.scale = self.scale
        result.shift = self.shift
    def copy(self):
        result = PowerOperation()
        self.copyto(result)
        return result

class ScaleOperation(WeightedOperation):
    def __init__(self):
        WeightedOperation.__init__(self)
        self.channels=1
    def copyTo(self, result):
        WeightedOperation.copyTo(self, result)
        result._caffe_batchnorm_convert = self._caffe_batchnorm_convert
        result.channels=self.channels
        return result
    def copy(self):
        result = ScaleOperation()
        self.copyTo(result)
        return result
    def mergeToBatchNorm(self, net):
        if self._caffe_batchnorm_convert:
            for i in range(len(net.operations)):
                op = net.operations[i]
                if isinstance(op, ScaleOperation) and not op._caffe_batchnorm_convert and self.top[0] == op.bottom[0]:
                    new_op = BatchNormOperation()
                    new_op.name = self.name
                    new_op.bottom = self.bottom
                    new_op.top = self.top
                    new_op.weight_data["scale"] = op.weight_data["filter"]
                    new_op.weight_data["offset"] = op.weight_data["bias"]
                    new_op.weight_data["mean"] = -1 * self.weight_data["bias"] / self.weight_data["filter"]
                    new_op.weight_data["variance"] = 1 / self.weight_data["filter"]
                    net.operations[i] = new_op
                    net.operations.remove(self)
                    return
    def convertToMulAdd(self, net):
        index = net.operations.index(self)
        net.operations.remove(self)
        if self.use_bias:
            add = AddOperation()
            add.name = self.name + "_add"
            add.top.append(self.top[0])
            add.bottom.append(self.top[0])
            add.bottom.append(add.name + "_bias")
            add.weight_data["bias"] = self.weight_data["bias"]
            net.operations.insert(index,add)
        mul = MulOperation()
        mul.name = self.name + "_mul"
        mul.top.append(self.top[0])
        mul.bottom.append(self.bottom[0])
        mul.bottom.append(mul.name + "_filter")
        mul.weight_data["filter"] = self.weight_data["filter"]
        net.operations.insert(index,mul)

class ConvOperation(WeightedOperation):
    def __init__(self):
        WeightedOperation.__init__(self)
    def copyTo(self, result):
        WeightedOperation.copyTo(self, result)
        return result
    def copy(self):
        result = ConvOperation()
        result.use_bias = self.use_bias
        result.stride = self.stride
        result.padding = self.padding
        result.size = self.size
        result.dilation = self.dilation
        self.copyTo(result)
        return result

class DeconvOperation(WeightedOperation):
    def __init__(self):
        WeightedOperation.__init__(self)
    def copy(self):
        result = DeconvOperation()
        result.dilation = self.dilation
        WeightedOperation.copy(self, result)
        return result

class ReduceOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
        self.op = "mean"
        self.axes = [2,3]
    def copy(self):
        result = MeanReduceOperation()
        Operation.copyTo(self, result)
        result.op = self.op
        self.axes = [2,3]

class PoolOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
        self.size = []
        self.stride = []
        self.padding = []
        self.pads = []
        self.global_receptive_field = False
    def copy(self):
        result = PoolOperation()
        Operation.copyTo(self, result)
        result.pool = self.pool
        for s in self.size:
            result.size.append(s)
        for s in self.stride:
            result.stride.append(s)
        result.pads = self.pads
        result.padding = self.padding
        result.global_receptive_field = self.global_receptive_field
        return result
    def convertToReduce(self, net):
        if self.global_receptive_field:
            new_op = ReduceOperation()
            new_op.name = self.name
            new_op.bottom = self.bottom
            new_op.top = self.top
            if self.pool == "avg":
                new_op.op = "mean"
            elif self.pool == "max":
                new_op.op = "max"
            index = net.operations.index(self)
            net.operations[index] = new_op

class ELUOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
        self.alpha = 1
    def copy(self):
        result = ELUOperation()
        Operation.copyTo(self, result)
        result.alpha = self.alpha
        return result

class ReLUOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
        self.negative_slope = 0
    def copy(self):
        result = ReLUOperation()
        Operation.copyTo(self, result)
        result.negative_slope = self.negative_slope
        return result
    def leaky(self):
        return self.negative_slope != 0

class SoftmaxOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
    def copy(self):
        result = SoftmaxOperation()
        Operation.copyTo(self, result)
        return result

class ArgmaxOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
        self.size = []
    def copy(self):
        result = ArgmaxOperation()
        Operation.copyTo(self, result)
        result.size = self.size
        return result

class BNLLOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
    def copy(self):
        result = BNLLOperation()
        Operation.copyTo(self, result)
        return result

class ReshapeOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
        self.shape = []
    def copy(self):
        result = ReshapeOperation()
        Operation.copyTo(self, result)
        result.shape = self.shape
        return result

class TanhOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
    def copy(self):
        result = TanhOperation()
        Operation.copyTo(self, result)
        return result

class AbsOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
    def copy(self):
        result = AbsOperation()
        Operation.copyTo(self, result)
        return result

class SigmoidOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
    def copy(self):
        result = SigmoidOperation()
        Operation.copyTo(self, result)
        return result

class AddOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
    def copy(self):
        result = AddOperation()
        Operation.copyTo(self, result)
        return result

class MulOperation(WeightedOperation):
    def __init__(self):
        WeightedOperation.__init__(self)
    def copy(self):
        result = MulOperation()
        WeightedOperation.copyTo(self, result)
        return result

class MergeOperation(Operation):
    def __init__(self):
        Operation.__init__(self)
        self.axis = 2
    def copy(self):
        result = MergeOperation()
        Operation.copyTo(self, result)
        result.axis = self.axis
        return result
