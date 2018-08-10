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
import sys
import os


def main():
    try:
        from nnef_converters.version import __version__ as version
        version = "caffe_to_nnef: nnef_converters {}".format(version)
    except ImportError:
        version = "unknown"

    parser = argparse.ArgumentParser(description="Caffe to NNEF converter")
    parser.add_argument("prototxt_path",
                        help="The path of the Caffe prototxt file that you want to convert to NNEF")
    parser.add_argument("-p", "--pythonpath", help="this path is added to PYTHONPATH "
                                                   "when loading the prototxt file")
    parser.add_argument("-m", "--model",
                        help="caffemodel file path, used to export weights")
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

    updater_path = "/usr/bin/upgrade_net_proto_text"
    if os.path.exists(updater_path):
        if args.verbose:
            print("Detected prototxt updater: " + updater_path)
    elif 'CAFFE_BIN_FOLDER' in os.environ:
        updater_path = os.path.join(os.environ['CAFFE_BIN_FOLDER'], "upgrade_net_proto_text")
        if os.path.exists(updater_path) or os.path.exists(updater_path + ".exe"):
            if args.verbose:
                print("Used prototxt updater: " + updater_path)
        else:
            print("Error: upgrade_net_proto_text is not present in CAFFE_BIN_FOLDER")
            exit(1)
    elif 'CAFFE_BIN_FOLDER' not in os.environ:
        if os.name == "nt":
            caffe_bin_folder = os.path.normpath(os.path.join(os.path.dirname(caffe.__file__),
                                                             "..", "..", "build", "tools", "Release"))
        else:
            caffe_bin_folder = os.path.normpath(os.path.join(os.path.dirname(caffe.__file__),
                                                             "..", "..", "build", "tools"))
        updater_path = os.path.join(caffe_bin_folder, "upgrade_net_proto_text")
        if os.path.exists(updater_path) or os.path.exists(updater_path + ".exe"):
            os.environ["CAFFE_BIN_FOLDER"] = caffe_bin_folder
            if args.verbose:
                print("Detected prototxt updater: " + updater_path)
        else:
            print("Error: The location of upgrade_net_proto_text could not be detected, "
                  "please set the CAFFE_BIN_FOLDER environment variable to its folder")
            exit(1)

    from nnef_converters.caffe_converters.caffe_to_nnef import convert
    from nnef_converters.common.utils import ConversionException

    sys.path.insert(0, '.')
    if args.pythonpath:
        sys.path.insert(0, args.pythonpath)
    try:
        convert(prototxt_path=args.prototxt_path,
                caffemodel_path=args.model,
                output_path=args.output_path,
                verbose=args.verbose)
    except ConversionException:
        print("Error: There were conversion errors!")
        exit(1)


if __name__ == "__main__":
    main()
