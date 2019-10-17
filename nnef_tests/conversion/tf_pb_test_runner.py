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

import six

from nnef_tools import convert
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

    def _test_network(self, path, source_shape=1, ignore_extra_outputs=False):
        assert utils.is_anyint(source_shape) or isinstance(source_shape, (list, tuple))

        network = os.path.basename(path.rsplit('/', 2)[1])
        command = """
        ./nnef_tools/convert.py --input-format=tensorflow-pb \\
                                --input-model={path} \\
                                --output-format=nnef \\
                                --output-model=out/nnef/{network}.nnef.tgz \\
                                --compress \\
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
                                    {nchw_opt}""".format(network=network,
                                                         nchw_str=prefer_nchw_str,
                                                         nchw_opt=prefer_nchw_opt)
            print(command)
            convert.convert_using_command(command)

            activation_testing = int(os.environ.get('NNEF_ACTIVATION_TESTING', '1'))
            print("Activation testing is", "ON" if activation_testing else "OFF")
            if activation_testing:
                import numpy as np
                import tensorflow as tf

                def normalize_shape(shape, default=1):
                    return [int(dim.value) if dim.value is not None else default for dim in shape.dims]

                tf.reset_default_graph()
                self._set_default_graph_from_pb(path)

                if isinstance(source_shape, (list, tuple)):
                    feed_dict = {placeholder.name: np.random.random(source_shape)
                                 for placeholder in get_placeholders()}
                else:
                    feed_dict = {placeholder.name: np.random.random(normalize_shape(placeholder.shape,
                                                                                    default=source_shape))
                                 for placeholder in get_placeholders()}
                old_names = [placeholder.name for placeholder in get_placeholders()]

                outputs = self._run_tfpb(path, feed_dict)

                tf.reset_default_graph()
                path2 = os.path.join('out', 'tensorflow-pb' + prefer_nchw_str, network + ".pb")
                self._set_default_graph_from_pb(path2)

                feed_dict2 = {placeholder.name: feed_dict[old_names[i]]
                              for i, placeholder in enumerate(get_placeholders())}

                outputs2 = self._run_tfpb(path2, feed_dict2)

                if ignore_extra_outputs:
                    outputs2 = outputs2[-len(outputs):]

                self.assertTrue(len(outputs) == len(outputs2))
                for a, b in zip(outputs, outputs2):
                    if a.dtype == np.bool:
                        self.assertTrue(np.all(a == b))
                    else:
                        self.assertTrue(np.all(np.isfinite(a)))
                        self.assertTrue(np.all(np.isfinite(b)))
                        self.assertTrue(np.allclose(a, b, atol=1e-5))

    def _test_layer(self, output_name, output_nodes, recreate=True, ignore_extra_outputs=False):
        import tensorflow as tf

        pb_path = os.path.join('out', 'pb', output_name)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            save_protobuf(pb_path, output_nodes, sess, recreate)
            print("Wrote", os.path.abspath(pb_path))
            sess.close()

        self._test_network(os.path.join(pb_path, 'test.pb'), ignore_extra_outputs=True)

    @staticmethod
    def _run_tfpb(path, feed_dict):
        import tensorflow as tf

        feed_dict = {k + ':0' if ':' not in k else k: v for k, v in six.iteritems(feed_dict)}

        tf.reset_default_graph()
        with tf.gfile.GFile(path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        with tf.Session() as sess:
            outputs = []
            ops = tf.get_default_graph().get_operations()
            for op in ops:
                if all(len(t.consumers()) == 0 for t in op.outputs):
                    if op.op_def.name == "FusedBatchNorm":
                        outputs.append(op.outputs[0])
                    else:
                        for t in op.outputs:
                            outputs.append(t)
            return [output for output in sess.run(outputs, feed_dict)]

    @staticmethod
    def _set_default_graph_from_pb(frozen_graph_filename):
        import tensorflow as tf
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
