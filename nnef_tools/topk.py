#!/usr/bin/env python

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

import sys

# Python2: Ensure that load from current directory is enabled, but load from the directory of the script is disabled
if len(sys.path) == 0:
    sys.path.append('')
if sys.path[0] != '':
    sys.path[0] = ''

import argparse

import numpy as np
import os
import nnef

from nnef_tools.io.nnef.nnef_io import read_nnef_tensor
from nnef_tools.core import utils


def topk(data, axis, k):
    indices = np.flip(np.argsort(data, axis=axis), axis=axis).take(indices=range(k), axis=axis)
    values = np.take_along_axis(data, indices, axis=axis)
    return values, indices


def get_args():
    parser = argparse.ArgumentParser(
        description="NNEF-Tools/topk.py: Print top k elements of a tensor along the given axis",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('tensor_path',
                        nargs='?',
                        help="By default the standard output is used, "
                             "but only if something is piped or redirected to it.")
    parser.add_argument('-k', '--k', default=5,
                        help="""The number of printed items. If k <= 0 it prints the whole tensor. Default: 5.""")
    parser.add_argument('-a', '--axis', default=1, help="""The axis for doing topk. Default: 1""")
    return parser.parse_args()


def tensor_name(path):
    name = os.path.basename(path)
    if name.endswith('.dat'):
        name = name[:-len('.dat')]
    return name


def main():
    np.set_printoptions(precision=3, suppress=True, floatmode='fixed', threshold=sys.maxsize)
    args = get_args()

    if args.tensor_path is None:
        utils.set_stdin_to_binary()

        if sys.stdin.isatty():
            print("No input provided!", file=sys.stderr)
            exit(1)

        tensor = nnef.read_tensor(sys.stdin)[0]
    else:
        tensor = read_nnef_tensor(args.tensor_path)

    print('Shape:', list(tensor.shape))

    if int(args.k) <= 0:
        print(tensor)
    else:
        axis = int(args.axis)
        k = min(int(args.k), tensor.shape[axis])

        if axis >= len(tensor.shape) or axis < -len(tensor.shape):
            print("axis={} is outside the supported range for this tensor.".format(axis))
            exit(1)

        values, indices = topk(tensor, axis=axis, k=k)
        values = np.squeeze(values)
        indices = np.squeeze(indices)
        print("TopK({}, k={}, axis={}):\nValues:\n{}\nIndices:\n{}".format(
            tensor_name(args.tensor_path) if args.tensor_path is not None else "stdin", k, axis, values, indices))


if __name__ == '__main__':
    main()
