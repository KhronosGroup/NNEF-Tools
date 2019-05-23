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

import numpy as np
import six

from nnef_tools.conversion import conversion_info
from nnef_tools.core import utils
from nnef_tools.io.nnef.nnef_io import write_nnef_tensor

with utils.EnvVars(GLOG_minloglevel=3):
    import caffe


def _get_input_dtypes_and_shapes(net):
    return {i: (np.float32, list(net.blobs[i].shape)) for i in net.inputs}


def export(prototxt_path,
           caffemodel_path,
           input_sources,
           conversion_info_path,
           output_path):
    conv_info = conversion_info.load(conversion_info_path)
    tensor_info_by_caffe_name = {t.source_name: t for t in conv_info.tensors}
    np.random.seed(0)
    net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
    net.forward(**input_sources.create_feed_dict(_get_input_dtypes_and_shapes(net)))
    has_error = False

    for k, v in six.iteritems(net.blobs):
        tensor_info = tensor_info_by_caffe_name.get(k)
        if tensor_info is not None:
            filename = os.path.join(output_path, tensor_info.target_name + ".dat")
            arr = v.data.copy()
            if np.isnan(arr).any():
                print("Error: '{}' has nan's".format(tensor_info.target_name))
                has_error = True
            elif not np.isfinite(arr).all():
                print("Error: '{}' has inf's".format(tensor_info.target_name))
                has_error = True
            try:
                for transform in tensor_info.transforms:
                    arr = transform.apply_np(arr)
                write_nnef_tensor(filename, np.asarray(arr, order='C'))
            except ValueError as e:
                print("Error: Can not export '{}': {}".format(tensor_info.target_name, e))
    if has_error:
        raise utils.NNEFToolsException("There were errors!")
