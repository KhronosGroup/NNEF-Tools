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


def ensure_dir(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise Exception("{} is not a directory".format(path))
    else:
        os.makedirs(path)


def main():
    try:
        from nnef_converters.version import __version__ as version
        version = "create_dummy_caffe_model: nnef_converters {}".format(version)
    except ImportError:
        version = "unknown"

    parser = argparse.ArgumentParser(description="Create dummy Caffe model")
    parser.add_argument("prototxt_path",
                        help="The path of the Caffe prototxt file that you want to convert to NNEF")
    parser.add_argument("-p", "--pythonpath", help="this path is added to PYTHONPATH "
                                                   "when loading the prototxt file")
    parser.add_argument("-o", "--output_path", default=".",
                        help="target directory path, default: current directory")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print debug info to the console")
    parser.add_argument("--version", action="version", version=version)
    args = parser.parse_args()

    os.environ['GLOG_minloglevel'] = '3'

    ensure_dir(args.output_path)

    try:
        import caffe
    except ImportError:
        print("Error: You have to install Caffe and set the PYTHONPATH variable to include its python folder")
        exit(1)

    sys.path.insert(0, '.')
    if args.pythonpath:
        sys.path.insert(0, args.pythonpath)

    net_name = os.path.splitext(os.path.basename(args.prototxt_path))[0]
    net = caffe.Net(args.prototxt_path, caffe.TEST)
    net.save(os.path.join(args.output_path, net_name + ".caffemodel"))


if __name__ == "__main__":
    main()
