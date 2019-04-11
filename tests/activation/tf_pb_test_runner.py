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
import shutil

from nnef_tools.activation_export import activation_test
from nnef_tools.conversion import conversion_info
from nnef_tools.conversion.tensorflow import tf_pb_all_in_one


def test_tf_pb(filename, source_shapes, feed_dict, prefer_nhwc, network_name, export_io_only, delete_after_each):
    dir = os.path.join('out', network_name + ('_nhwc' if prefer_nhwc else '_nchw'))

    if os.path.exists(dir):
        shutil.rmtree(dir)

    tf_pb_all_in_one.convert_tf_pb_to_nnef(
        file_name=filename,
        output_directory=os.path.join(dir, 'nnef'),
        network_name=network_name,
        optimization_level=tf_pb_all_in_one.OptimizationLevel.FULL,
        io_transform=tf_pb_all_in_one.IOTransform.SMART_TF_NHWC_TO_NCHW,
        source_shapes=source_shapes,
        activation_export_feed_dict=feed_dict,
        activation_export_io_only=export_io_only)

    conv_info1 = conversion_info.load(os.path.join(dir, 'nnef', 'conversion.json'))

    tf_pb_all_in_one.convert_nnef_to_tf_pb(
        nnef_tgz_or_dir_path=os.path.join(dir, 'nnef', network_name + '_nnef'),
        output_directory=os.path.join(dir, 'tf_pb'),
        optimization_level=tf_pb_all_in_one.OptimizationLevel.FULL,
        io_transform=(tf_pb_all_in_one.IOTransform.SMART_NCHW_TO_TF_NHWC
                      if prefer_nhwc else tf_pb_all_in_one.IOTransform.SMART_NCHW_TO_TF_NCHW),
        prefer_nhwc=prefer_nhwc)
    conv_info2 = conversion_info.load(os.path.join(dir, 'tf_pb', 'conversion.json'))

    feed_dict2 = (activation_test.transform_feed_dict(feed_dict=feed_dict,
                                                      conv_info=conversion_info.compose(conv_info1, conv_info2))
                  if feed_dict is not None else None)

    tf_pb_all_in_one.convert_tf_pb_to_nnef(
        file_name=os.path.join(dir, 'tf_pb', 'graph.pb'),
        output_directory=os.path.join(dir, 'nnef2'),
        network_name=network_name,
        optimization_level=tf_pb_all_in_one.OptimizationLevel.FULL,
        io_transform=(tf_pb_all_in_one.IOTransform.SMART_TF_NHWC_TO_NCHW
                      if prefer_nhwc else tf_pb_all_in_one.IOTransform.SMART_TF_NCHW_TO_NCHW),
        activation_export_feed_dict=feed_dict2,
        activation_export_io_only=export_io_only)

    conv_info3 = conversion_info.load(os.path.join(dir, 'nnef2', 'conversion.json'))

    if feed_dict is not None:
        activation_test.compare_activation_dirs(
            os.path.join(dir, 'nnef', 'activations'),
            os.path.join(dir, 'nnef2', 'activations'),
            conv_info=conversion_info.compose(conv_info2, conv_info3),
            verbose=False,
            allowed_bad_pixel_ratio=0.01 if not prefer_nhwc and not export_io_only else 0.0)

    if delete_after_each:
        shutil.rmtree(dir)
