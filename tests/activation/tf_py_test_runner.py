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

import shutil
import fnmatch
import os
import sys
import unittest

import numpy as np
import tensorflow as tf

from nnef_tools import convert
from nnef_tools.activation_export import activation_test
from nnef_tools.activation_export.tensorflow import tf_activation_exporter
from nnef_tools.conversion import conversion_info
from nnef_tools.core import utils
from nnef_tools.io.tensorflow import tf_py_io

DELETE_DATS_AND_CHECKPOINTS = True  # Uses less disk space


def get_placeholders():
    tensors = []
    for op in tf.get_default_graph().get_operations():
        if "Placeholder" in op.node_def.op:
            tensors.append(op.outputs[0])
    return sorted(tensors, key=lambda t: t.name)


def get_feed_dict():
    placeholders = get_placeholders()
    feed_dict = {}
    for p in placeholders:
        feed_dict[utils.anystr_to_str(p.name)] = np.random.random(p.shape)
    return feed_dict


def get_variables():
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)


def save_random_checkpoint(network_outputs, checkpoint_path, feed_dict):
    variables = get_variables()

    init = None
    saver = None
    if variables:
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session() as sess:
        if variables:
            sess.run(init)

        sess.run(network_outputs, feed_dict=feed_dict)

        if variables:
            if not os.path.exists(os.path.dirname(checkpoint_path)):
                os.makedirs(os.path.dirname(checkpoint_path))

            save_path = saver.save(sess, os.path.relpath(checkpoint_path))
            assert save_path == checkpoint_path
            return checkpoint_path
        else:
            return None


def recursive_glob(dir, glob):
    matches = []
    for root, dir_names, file_names in os.walk(dir):
        for filename in fnmatch.filter(file_names, glob):
            matches.append(os.path.join(root, filename))
    return matches


def tf_call_silently(fun, *args):
    if 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
        old_value = os.environ['TF_CPP_MIN_LOG_LEVEL']
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        retval = fun(*args)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = old_value
        return retval
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        retval = fun(*args)
        del os.environ['TF_CPP_MIN_LOG_LEVEL']
        return retval


def tf_has_cuda_gpu():
    return tf_call_silently(tf.test.is_gpu_available, True)


class TFPyTestRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.path.exists('out'):
            shutil.rmtree('out')

        print("PYTHON", sys.version)
        print("TENSORFLOW", tf.__version__)
        print("GPU ENABLED" if tf_has_cuda_gpu() else "!!! GPU DISABLED")

    def _test(self,
              fun,
              cmp=True,
              custom_tf_to_nnef_converters="",
              custom_nnef_to_tf_converters="",
              test_module="tests.activation.tf_py_layer_test_cases"):
        out_dir = os.path.join("out", fun.__name__)
        try:
            tf.reset_default_graph()
            tf.set_random_seed(0)

            network_outputs = fun()
            feed_dict = get_feed_dict()
            checkpoint_path = os.path.join("out", fun.__name__, "orig_checkpoint", fun.__name__ + ".ckpt")
            checkpoint_path = save_random_checkpoint(network_outputs, checkpoint_path, feed_dict)

            tf.reset_default_graph()
            tf.set_random_seed(0)

            compress_nnef = False
            command = """
                ./nnef_tools/convert.py --input-framework=tensorflow-py \\
                                        --output-framework=nnef \\
                                        --input-model={module}.{network}{checkpoint} \\
                                        --output-directory=out/{network}/nnef \\
                                        --custom-converters="{custom}" \\
                                        --permissive \\
                                        --io-transformation=SMART_TF_NHWC_TO_NCHW \\
                                        {compress}
                """.format(checkpoint=':' + checkpoint_path if checkpoint_path else "",
                           network=fun.__name__,
                           custom=custom_tf_to_nnef_converters,
                           compress="--compress" if compress_nnef else "",
                           module=test_module)

            convert.convert_using_command(command)

            open(os.path.join("out", fun.__name__, "__init__.py"), "w").close()

            tf.reset_default_graph()
            tf.set_random_seed(0)
            fun()
            conv_info = conversion_info.load(os.path.join("out", fun.__name__, "nnef", "conversion.json"))
            tf_activation_exporter.export(output_path=os.path.join("out", fun.__name__, "nnef", "activations"),
                                          feed_dict=feed_dict,
                                          conversion_info=conv_info,
                                          checkpoint_path=checkpoint_path,
                                          verbose=False)

            prefer_nhwc_options = [True]
            if tf_has_cuda_gpu():
                prefer_nhwc_options += [False]
            for prefer_nhwc in prefer_nhwc_options:
                print("Converting to TensorFlow {}".format("NHWC" if prefer_nhwc else "NCHW"))
                data_format_str = ("nhwc" if prefer_nhwc else "nchw")
                tf_output_dir = os.path.join("out", fun.__name__, "tf_" + data_format_str)
                command = """
                    ./nnef_tools/convert.py --input-framework=nnef \\
                                            --output-framework=tensorflow-py \\
                                            --input-model=out/{network}/nnef/model{tgz} \\
                                            --output-directory={output} \\
                                            --io-transformation=SMART_NCHW_TO_TF_NHWC \\
                                            --custom-converters="{custom}" \\
                                            --permissive
                    """.format(network=fun.__name__,
                               custom=custom_nnef_to_tf_converters,
                               tgz=".nnef.tgz" if compress_nnef else "",
                               output=tf_output_dir)
                convert.convert_using_command(command)

                open(os.path.join(tf_output_dir, "__init__.py"), "w").close()
                open(os.path.join(tf_output_dir, "model", "__init__.py"), "w").close()

                with open(os.path.join(tf_output_dir, "model", "model.py")) as f:
                    tf_src = f.read()

                # noinspection PyProtectedMember
                new_net_fun = tf_py_io._tfsource_to_function(tf_src, fun.__name__)

                conv_info_tf_to_nnef = conversion_info.load(os.path.join(out_dir, "nnef", "conversion.json"))
                conv_info_nnef_to_tf = conversion_info.load(os.path.join(tf_output_dir, "conversion.json"))
                conv_info_tf_to_tf = conversion_info.compose(conv_info_tf_to_nnef, conv_info_nnef_to_tf)

                conversion_info.dump(conv_info_tf_to_tf, os.path.join(tf_output_dir, "conv_info_tf_to_tf.json"))

                feed_dict2 = activation_test.transform_feed_dict(feed_dict, conv_info_tf_to_tf)
                nnef2_out_dir = os.path.join(out_dir, "nnef_from_tf_" + data_format_str)

                tf.reset_default_graph()
                tf.set_random_seed(0)

                command = """
                    ./nnef_tools/convert.py --input-framework=tensorflow-py \\
                                            --output-framework=nnef \\
                                            --input-model={input}{checkpoint} \\
                                            --output-directory={output} \\
                                            --custom-converters="{custom}" \\
                                            --permissive \\
                                            --io-transformation=SMART_TF_NHWC_TO_NCHW \\
                                            {compress}
                    """.format(checkpoint=(':' + (os.path.join(tf_output_dir, "model", "checkpoint", "model.ckpt")
                                                                            if checkpoint_path else "")),
                               input=tf_output_dir.replace('/', '.') + ".model.model." + fun.__name__,
                               custom=custom_tf_to_nnef_converters,
                               compress="--compress" if compress_nnef else "",
                               output=nnef2_out_dir)

                convert.convert_using_command(command)

                conv_info_tf_to_nnef2 = conversion_info.load(os.path.join(out_dir,
                                                                          "nnef_from_tf_" + data_format_str,
                                                                          "conversion.json"))
                conv_info_nnef_to_nnef = conversion_info.compose(conv_info_nnef_to_tf, conv_info_tf_to_nnef2)
                conversion_info.dump(conv_info_nnef_to_nnef, os.path.join(nnef2_out_dir,
                                                                          "conv_info_nnef_to_nnef.json"))

                tf.reset_default_graph()
                tf.set_random_seed(0)
                new_net_fun()
                tf_activation_exporter.export(
                    output_path=os.path.join(nnef2_out_dir, "activations"),
                    feed_dict=feed_dict2,
                    conversion_info=conv_info_tf_to_nnef2,
                    checkpoint_path=(os.path.join(tf_output_dir, "model", "checkpoint", "model.ckpt")
                                     if checkpoint_path else None),
                    verbose=False)

                if cmp:
                    activation_test.compare_activation_dirs(
                        os.path.join(out_dir, "nnef", "activations"),
                        os.path.join(out_dir, "nnef_from_tf_" + data_format_str, "activations"),
                        conv_info_nnef_to_nnef,
                        verbose=False)
        finally:
            if DELETE_DATS_AND_CHECKPOINTS:
                dat_files = recursive_glob(out_dir, "*.dat")
                checkpoints = recursive_glob(out_dir, "*ckpt*")
                for file_name in set(dat_files + checkpoints):
                    os.remove(file_name)
