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

import numpy as np


class Transform(object):
    def apply_np(self, arr):
        assert False, "Unimplemented"


class Transpose(Transform):
    def __init__(self, axes):
        self.axes = axes

    def is_identity(self):
        return self.axes == list(range(len(self.axes)))

    def copy(self):
        return Transpose(self.axes)

    def apply_np(self, arr):
        return np.transpose(arr, self.axes)

    def __repr__(self):
        return "Transpose({})".format(self.axes)


class Squeeze(Transform):
    def __init__(self, axes):
        self.axes = axes

    def is_identity(self):
        return not self.axes

    def copy(self):
        return Squeeze(self.axes)

    def apply_np(self, arr):
        return np.squeeze(arr, self.axes)

    def __repr__(self):
        return "Squeeze({})".format(self.axes)


class Unsqueeze(Transform):
    def __init__(self, axes):
        self.axes = axes

    def is_identity(self):
        return not self.axes

    def copy(self):
        return Unsqueeze(self.axes)

    def apply_np(self, arr):
        return np.reshape(arr, unsqueezed_shape(list(arr.shape), self.axes))

    def __repr__(self):
        return "Unsqueeze({})".format(self.axes)


class Reshape(Transform):
    def __init__(self, shape):
        self.shape = shape

    def is_identity(self, input_shape):
        return self.shape == input_shape

    def copy(self):
        return Reshape(self.shape)

    def apply_np(self, arr):
        return np.reshape(arr, self.shape)

    def __repr__(self):
        return "Reshape({})".format(self.shape)


def _unsqueezed_shape2(shape, axes, i, n):
    return ([] if i == n
            else ([1] + _unsqueezed_shape2(shape, axes, i + 1, n) if i in axes
                  else [shape[0]] + _unsqueezed_shape2(shape[1:], axes, i + 1, n)))


# Works as in NNEF: axes correspond to output dims
def unsqueezed_shape(shape, axes):
    return _unsqueezed_shape2(shape, axes, 0, len(shape) + len(axes))


def squeezed_shape(shape, axes, can_squeeze_not_one=False):
    assert can_squeeze_not_one or all(dim == 1 for axis, dim in enumerate(shape) if axis in axes)
    return [dim for axis, dim in enumerate(shape) if axis not in axes]
