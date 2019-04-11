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

import nnef
import numpy as np
import six

from nnef_tools.conversion import conversion_info


class _ActivationPair(object):
    def __init__(self, from_, to, trafos, short_from, short_to):
        self.short_from = short_from
        self.short_to = short_to
        self.from_ = from_  # type: str
        self.to = to  # type: str
        self.trafos = trafos  # type: typing.List[conversion_info.Transform]


_nnef_dtype_by_np_dtype = None


def _np_load_dtype_table():
    import numpy as np

    global _nnef_dtype_by_np_dtype

    if _nnef_dtype_by_np_dtype is None:
        _nnef_dtype_by_np_dtype = {
            np.dtype(np.float16): 'scalar',
            np.dtype(np.float32): 'scalar',
            np.dtype(np.float64): 'scalar',
            np.dtype(np.int8): 'integer',
            np.dtype(np.uint8): 'integer',
            np.dtype(np.int16): 'integer',
            np.dtype(np.uint16): 'integer',
            np.dtype(np.int32): 'integer',
            np.dtype(np.uint32): 'integer',
            np.dtype(np.int64): 'integer',
            np.dtype(np.uint64): 'integer',
            np.dtype(np.bool_): 'logical'
        }


def _is_np_dtype_exportable_to_nnef(dtype):
    import numpy as np
    _np_load_dtype_table()

    return np.dtype(dtype) in _nnef_dtype_by_np_dtype


def _try_np_dtype_to_nnef(dtype):
    import numpy as np
    _np_load_dtype_table()

    return _nnef_dtype_by_np_dtype.get(np.dtype(dtype))


def _are_np_dtypes_compatible_in_nnef(t1, t2):
    t1 = _try_np_dtype_to_nnef(t1)
    t2 = _try_np_dtype_to_nnef(t2)
    return t1 is not None and t2 is not None and t1 == t2


def _read_nnef_tensor(filename):
    with open(filename, "rb") as file:
        return nnef.read_tensor(file)[0]


# noinspection PyProtectedMember
def _are_activations_close(activation_pair, verbose=False, allowed_bad_pixel_ratio=0.0):
    # type: (_ActivationPair, bool, float)->bool

    arr1 = _read_nnef_tensor(activation_pair.from_)  # type:np.ndarray
    arr2 = _read_nnef_tensor(activation_pair.to)  # type: np.ndarray
    for trafo in activation_pair.trafos:
        arr1 = trafo.apply_np(arr1)

    if arr1.dtype != arr2.dtype:
        print("Warning: different dtypes: {} != {} for {}, {}"
              .format(arr1.dtype.name, arr2.dtype.name, activation_pair.short_from, activation_pair.short_to))

    if not _are_np_dtypes_compatible_in_nnef(arr1.dtype, arr2.dtype):
        print("Error: incompatible dtypes: {} != {} for {}, {}"
              .format(arr1.dtype.name, arr2.dtype.name, activation_pair.short_from, activation_pair.short_to))
        return False
    elif arr1.shape != arr2.shape:
        print("Error: shape error {} != {} for {}, {}"
              .format(arr1.shape, arr2.shape, activation_pair.short_from, activation_pair.short_to))
        return False
    else:
        if arr1.dtype == np.dtype(np.bool_):
            differences = (arr1 != arr2).astype(np.int8).sum(dtype=np.int64)
            if differences > 0:  # Maybe too strict
                print("Error: Bool tensors different at {} places: {}, {}"
                      .format(differences, activation_pair.short_from, activation_pair.short_to))
                return False
            return True
        else:
            error_rate1 = np.abs(arr2 - arr1) / np.maximum(np.abs(arr1), 1e-32)
            error_rate2 = np.abs(arr2 - arr1) / np.maximum(np.abs(arr2), 1e-32)
            error_rate = np.maximum(error_rate1, error_rate2)
            max_error_rate1 = np.max(error_rate1)
            max_error_rate2 = np.max(error_rate2)
            max_error_rate = max(max_error_rate1, max_error_rate2)
            max_diff = np.max(np.abs(arr2 - arr1))
            max_value = min(np.max(np.abs(arr1)), np.max(np.abs(arr2)))

            err_thresh = 1e-4

            if max_value < err_thresh:
                print("Info: max value: {} for {}, {}"
                      .format(max_value, activation_pair.short_from, activation_pair.short_to))

            bad_pixel_ratio = np.count_nonzero(error_rate > err_thresh) / error_rate.size
            if max_error_rate > err_thresh and max_diff > err_thresh:
                if bad_pixel_ratio > allowed_bad_pixel_ratio:
                    print("Error: max error rate: {} max diff: {} for {}, {}"
                          .format(max_error_rate, max_diff, activation_pair.short_from, activation_pair.short_to))
                    print("Info: Bad pixel ratio: {}%".format(100 * bad_pixel_ratio))
                    # print(arr1, arr2)
                    return False
                else:
                    print("Warning: max error rate: {} max diff: {} for {}, {}"
                          .format(max_error_rate, max_diff, activation_pair.short_from, activation_pair.short_to))
                    print("Info: Bad pixel ratio: {}%".format(100 * bad_pixel_ratio))
                    # print(arr1, arr2)
                    return True
            elif max_diff != 0 or verbose:
                print("Info: max error rate: {} max diff: {} for {}, {}"
                      .format(max_error_rate, max_diff, activation_pair.short_from, activation_pair.short_to))
            return True


def transform_feed_dict(feed_dict, conv_info):
    # type: (typing.Dict[str, np.ndarray], conversion_info.ConversionInfo)->typing.Dict[str, np.ndarray]
    tensor_info_by_source_name = {tensor_info.source_name: tensor_info for tensor_info in conv_info.tensors}
    feed_dict2 = {}
    for name, value in six.iteritems(feed_dict):
        assert isinstance(name, str)
        assert name in tensor_info_by_source_name
        tensor_info = tensor_info_by_source_name[name]  # type: conversion_info.TensorInfo
        new_name = tensor_info.target_name
        new_value = value
        for trafo in tensor_info.transforms:
            new_value = trafo.apply_np(new_value)
        assert list(new_value.shape) == tensor_info.target_shape
        feed_dict2[new_name] = new_value
    return feed_dict2


def compare_activation_dirs(dir0, dir1, conv_info, verbose=False, allowed_bad_pixel_ratio=0.0):
    # type: (str, str, conversion_info.ConversionInfo, bool, float)->None

    fnset0 = set(os.listdir(dir0))
    fnset1 = set(os.listdir(dir1))

    activation_pairs = []

    used_source_names = set()
    used_target_names = set()
    for tensor in conv_info.tensors:
        source = tensor.source_name + '.dat'
        target = tensor.target_name + '.dat'
        if source in fnset0 and target in fnset1:
            activation_pairs.append(_ActivationPair(from_=os.path.join(dir0, source),
                                                    to=os.path.join(dir1, target),
                                                    trafos=tensor.transforms,
                                                    short_from=source,
                                                    short_to=target))
            used_source_names.add(source)
            used_target_names.add(target)
    if verbose:
        for fn in fnset0:
            if fn not in used_source_names:
                print("Warning: Unused source file: {}".format(fn))
        if len(used_source_names) != len(fnset0):
            print("Info: Used source files {} / {}".format(len(used_source_names), len(fnset0)))

    assert all([_are_activations_close(p, verbose=verbose, allowed_bad_pixel_ratio=allowed_bad_pixel_ratio)
                for p in activation_pairs]), "Some activation pairs differ"
