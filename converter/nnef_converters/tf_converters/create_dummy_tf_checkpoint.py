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

import argparse
import importlib
import os
import sys


def ensure_dir(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise Exception("{} is not a directory".format(path))
    else:
        os.makedirs(path)


def main():
    try:
        from nnef_converters.version import __version__ as version
        version = "create_dummy_tf_checkpoint: nnef_converters {}".format(version)
    except ImportError:
        version = "unknown"

    parser = argparse.ArgumentParser(description="Create dummy Tensorflow checkpoint")
    parser.add_argument("network_function",
                        help="the fully qualified name of a ()->Dict[str, tf.Tensor] function, "
                             "e.g. module.function or package.module.function")
    parser.add_argument("-p", "--pythonpath", help="this path is added to PYTHONPATH "
                                                   "when loading the module of network_function")
    parser.add_argument("-o", "--output_path", default=".",
                        help="target directory path, default: current directory")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print debug info to the console")
    parser.add_argument("--version", action="version", version=version)
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        import tensorflow as tf
    except ImportError:
        print("Error: package 'tensorflow' not found / can not be imported")
        exit(1)

    module_name = function_name = module = function_ = None
    try:
        module_name, function_name = args.network_function.rsplit('.', 1)
    except ValueError:
        print("Error: network_function must consist of at least two parts: module.function")
        exit(1)

    try:
        sys.path.insert(0, '.')
        if args.pythonpath:
            sys.path.insert(0, args.pythonpath)
        module = importlib.import_module(module_name)
        if args.pythonpath:
            sys.path = sys.path[1:]
        sys.path = sys.path[1:]
    except ImportError:
        print("Error: Can not import module {}".format(module_name))
        if not args.pythonpath:
            print(
                "If the package or module is not in the current directory, try setting the -p/--pythonpath parameter.")
        exit(1)

    try:
        function_ = getattr(module, function_name)
        function_.__name__ = function_name
    except AttributeError:
        print("Error: Function {} not found in module {}".format(function_name, module_name))
        exit(1)

    ensure_dir(args.output_path)

    function_()

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        saver.save(sess, os.path.join(args.output_path, function_name + "_ckpt", function_name + ".ckpt"))

    # prevent tf bug
    del saver
    del sess


if __name__ == "__main__":
    main()
