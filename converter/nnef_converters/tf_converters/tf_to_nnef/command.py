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
import sys
import os


def main():
    try:
        from nnef_converters.version import __version__ as version
        version = "tf_to_nnef: nnef_converters {}".format(version)
    except ImportError:
        version = "unknown"

    parser = argparse.ArgumentParser(description="Tensorflow (source) to NNEF converter")
    parser.add_argument("network_function",
                        help="the fully qualified name of a ()->Dict[str, tf.Tensor] function, "
                             "e.g. module.function or package.module.function")
    parser.add_argument("-p", "--pythonpath", help="this path is added to PYTHONPATH "
                                                   "when loading the module of network_function")
    parser.add_argument("-m", "--model",
                        help="checkpoint file or directory path, used to export weights")
    parser.add_argument("-o", "--output_path", default=".",
                        help="target directory path, default: current directory")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print debug info to the console and the resulting NNEF file")
    parser.add_argument("--version", action="version", version=version)
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        import nnef
    except ImportError:
        print("Error: package 'nnef' not found / can not be imported")
        exit(1)

    try:
        import numpy
    except ImportError:
        print("Error: package 'numpy' not found / can not be imported")
        exit(1)

    try:
        import tensorflow
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

    from nnef_converters.tf_converters.tf_to_nnef import convert
    from nnef_converters.common.utils import ConversionException

    try:
        convert(network_function=function_,
                checkpoint_path=args.model,
                output_path=args.output_path,
                verbose=args.verbose)
    except ConversionException:
        print("Error: There were conversion errors!")
        exit(1)


if __name__ == "__main__":
    main()
