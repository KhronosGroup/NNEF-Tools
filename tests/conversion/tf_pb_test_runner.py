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
import unittest

from nnef_tools import convert
from nnef_tools.activation_export import activation_test
from nnef_tools.conversion import conversion_info
from nnef_tools.core import utils


def get_placeholders():
    import tensorflow as tf
    return [t for op in tf.get_default_graph().get_operations()
            for t in op.values() if
            'Placeholder' in op.node_def.op]


def save_protobuf(filename, output_node_names, sess, recreate):
    import tensorflow as tf

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    output_graph = filename + "test.pb"

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.isfile(filename + "test.pbtxt") or recreate:
        tf.train.write_graph(input_graph_def, "", filename + "test.pbtxt")

    if not os.path.isfile(output_graph) or recreate:
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def,
                                                                        output_node_names.split(","))
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        tf.train.write_graph(output_graph_def, "", filename + "test.pbtxt")


class TFPbTestRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if int(os.environ.get('NNEF_ACTIVATION_TESTING', '1')):
            import numpy as np
            import tensorflow as tf

            np.random.seed(0)
            tf.set_random_seed(0)

        if os.path.exists('out'):
            shutil.rmtree('out')

    @staticmethod
    def _test_network(path, source_shape=1):
        assert utils.is_anyint(source_shape) or isinstance(source_shape, (list, tuple))

        network = os.path.basename(path.rsplit('.', 1)[0])
        command = """
        ./nnef_tools/convert.py --input-format=tensorflow-pb \\
                                --input-model={path} \\
                                --output-format=nnef \\
                                --output-model=out/nnef/{network}.nnef.tgz \\
                                --compress \\
                                --conversion-info \\
                                --input-shape="{shape}" """.format(path=path, network=network, shape=source_shape)
        print(command)
        convert.convert_using_command(command)

        for prefer_nchw in [False, True]:
            print("Prefer NCHW" if prefer_nchw else "Prefer NHWC")

            prefer_nchw_opt = "--prefer-nchw" if prefer_nchw else ""
            prefer_nchw_str = "_prefer_nchw" if prefer_nchw else ""

            command = """
            ./nnef_tools/convert.py --input-format=nnef \\
                                    --input-model=out/nnef/{network}.nnef.tgz \\
                                    --output-format=tensorflow-pb \\
                                    --output-model=out/tensorflow-pb{nchw_str}/{network}.pb \\
                                    --conversion-info \\
                                    {nchw_opt}""".format(network=network,
                                                         nchw_str=prefer_nchw_str,
                                                         nchw_opt=prefer_nchw_opt)
            print(command)
            convert.convert_using_command(command)

            command = """
            ./nnef_tools/convert.py --input-format=tensorflow-pb \\
                                    --input-model=out/tensorflow-pb{nchw_str}/{network}.pb \\
                                    --output-format=nnef \\
                                    --output-model=out/nnef2{nchw_str}/{network}.nnef.tgz \\
                                    --compress \\
                                    --conversion-info""".format(network=network, nchw_str=prefer_nchw_str)
            print(command)
            convert.convert_using_command(command)

            activation_testing = int(os.environ.get('NNEF_ACTIVATION_TESTING', '1'))
            print("Activation testing is", "ON" if activation_testing else "OFF")
            if activation_testing:
                import numpy as np
                import tensorflow as tf
                from nnef_tools import export_activation
                from nnef_tools.activation_export.tensorflow import tf_activation_exporter

                def normalize_shape(shape, default=1):
                    return [int(dim.value) if dim.value is not None else default for dim in shape.dims]

                tf.reset_default_graph()
                export_activation.tf_set_default_graph_from_pb(path)

                if isinstance(source_shape, (list, tuple)):
                    feed_dict = {placeholder.name: np.random.random(source_shape)
                                 for placeholder in get_placeholders()}
                else:
                    feed_dict = {placeholder.name: np.random.random(normalize_shape(placeholder.shape,
                                                                                    default=source_shape))
                                 for placeholder in get_placeholders()}

                conv_info_tf_to_nnef = conversion_info.load(
                    os.path.join("out", 'nnef', network + ".nnef.tgz.conversion.json"))

                tf_activation_exporter.export(
                    output_path=os.path.join("out", 'nnef', network + ".nnef.tgz.activations"),
                    feed_dict=feed_dict,
                    conversion_info=conv_info_tf_to_nnef,
                    input_output_only=True,
                    verbose=False)

                conv_info_nnef_to_tf = conversion_info.load(
                    os.path.join('out', 'tensorflow-pb'+prefer_nchw_str, network + ".pb.conversion.json"))

                conv_info_tf_to_tf = conversion_info.compose(conv_info_tf_to_nnef, conv_info_nnef_to_tf)

                feed_dict2 = activation_test.transform_feed_dict(feed_dict, conv_info_tf_to_tf)

                conv_info_tf_to_nnef2 = conversion_info.load(
                    os.path.join('out', 'nnef2'+prefer_nchw_str, network + ".nnef.tgz.conversion.json"))
                conv_info_nnef_to_nnef = conversion_info.compose(conv_info_nnef_to_tf, conv_info_tf_to_nnef2)

                tf.reset_default_graph()
                export_activation.tf_set_default_graph_from_pb(
                    os.path.join('out', 'tensorflow-pb'+prefer_nchw_str, network + ".pb"))

                tf_activation_exporter.export(
                    output_path=os.path.join("out", 'nnef2'+prefer_nchw_str, network + ".nnef.tgz.activations"),
                    feed_dict=feed_dict2,
                    conversion_info=conv_info_tf_to_nnef2,
                    input_output_only=True,
                    verbose=False)

                activation_test.compare_activation_dirs(
                    os.path.join("out", 'nnef', network + ".nnef.tgz.activations"),
                    os.path.join("out", 'nnef2'+prefer_nchw_str, network + ".nnef.tgz.activations"),
                    conv_info_nnef_to_nnef,
                    verbose=False)

    @staticmethod
    def _test_layer(output_name, output_nodes, recreate=True):
        import tensorflow as tf

        pb_path = os.path.join('out', 'pb', output_name)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            save_protobuf(pb_path, output_nodes, sess, recreate)
            print("Wrote", os.path.abspath(pb_path))
            sess.close()

        TFPbTestRunner._test_network(os.path.join(pb_path, 'test.pb'))
