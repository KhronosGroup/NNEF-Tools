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
import os


def main():
    try:
        from nnef_converters.version import __version__ as version
        version = "nnef_to_caffe: nnef_converters {}".format(version)
    except ImportError:
        version = "unknown"

    parser = argparse.ArgumentParser(description="NNEF to Caffe converter")
    parser.add_argument("nnef_path",
                        help="Path of NNEF archive, file or directory, e.g. graph.nnef.tgz, graph.nnef or graph_dir")
    parser.add_argument("-o", "--output_path", default=".",
                        help="target directory path, default: current directory")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print debug info to the console")
    parser.add_argument("--version", action="version", version=version)
    args = parser.parse_args()

    os.environ['GLOG_minloglevel'] = '3'

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
        import caffe
    except ImportError:
        print("Error: You have to install Caffe and set the PYTHONPATH variable to include its python folder")
        exit(1)

    from nnef_converters.caffe_converters.nnef_to_caffe import convert
    from nnef_converters.common.utils import ConversionException

    try:
        convert(nnef_path=args.nnef_path, output_path=args.output_path, verbose=args.verbose)
    except ConversionException:
        print("Error: There were conversion errors!")
        exit(1)


if __name__ == "__main__":
    main()
