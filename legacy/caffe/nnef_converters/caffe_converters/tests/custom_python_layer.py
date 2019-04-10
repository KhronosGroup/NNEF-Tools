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

from __future__ import division, print_function

import caffe
import numpy as np


# noinspection PyMethodOverriding,PyUnusedLocal
class CustomPythonLayer(caffe.Layer):
    # noinspection PyAttributeOutsideInit
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.shape = params['shape']
        self.value = params['value']

        if len(bottom) != 0:
            raise Exception("This layer needs 0 inputs.")

        for i in range(len(top)):
            top[i].reshape(*self.shape)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for i in range(len(top)):
            top[i].data[...] = np.full(shape=self.shape, fill_value=self.value)

    def backward(self, top, propagate_down, bottom):
        pass
