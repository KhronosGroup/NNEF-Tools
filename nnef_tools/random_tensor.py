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
import nnef

from nnef_tools.io.input_source import RandomInput, create_input
from nnef_tools.io.nnef.nnef_io import write_nnef_tensor
from nnef_tools.core import utils


def get_args(argv):
    parser = argparse.ArgumentParser(
        description="NNEF-Tools/random_tensor.py: Create a random tensor",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('params',
                        nargs='*',
                        default=["uniform", "0", "1"],
                        help="The parameters of the random generator. Possible parametrizations:\n"
                             "uniform MIN MAX (default dtype: float32, range: [MIN, MAX])\n"
                             "normal MIN MAX (default dtype: float32)\n"
                             "binomial NUM TRUE_PROB (default dtype: int32, range: [0, NUM])\n"
                             "bernoulli TRUE_PROB (default dtype: bool)\n"
                             "Default: uniform 0 1")

    parser.add_argument('--output', required=False,
                        help="The path of the output tensor, e.g. tensor.dat.\n"
                             "By default the standard output is used, but only if the command is piped or redirected.")

    parser.add_argument("--shape",
                        nargs='*',
                        required=True,
                        type=int,
                        help="Target shape\n"
                             "E.g. 1 3 224 224")

    parser.add_argument("--seed",
                        required=False,
                        default=-1,
                        type=int,
                        help="Seed to use for random generation.\n"
                             "Default: -1 (Get the seed from /dev/urandom or the clock)")

    parser.add_argument("--dtype",
                        required=False,
                        help="Numpy dtype of the generated tensor. For the default, see params:")

    return parser.parse_args(args=argv[1:])


def main():
    try:
        args = get_args(sys.argv)

        if not args.output:
            if sys.stdout.isatty():
                raise utils.NNEFToolsException("No output provided.")
            utils.set_stdout_to_binary()

        if args.dtype is None:
            distribution = args.params[0]
            if distribution == 'binomial':
                args.dtype = "int32"
            elif distribution == 'bernoulli':
                args.dtype = "bool"
            else:
                args.dtype = "float32"

        args.params[1:] = [float(param) for param in args.params[1:]]

        if args.seed != -1:
            np.random.seed(args.seed)

        random_input = RandomInput(*args.params)
        arr = create_input(random_input, np_dtype=np.dtype(args.dtype), shape=args.shape)

        if args.output:
            write_nnef_tensor(args.output, arr)
        else:
            nnef.write_tensor(sys.stdout, arr)
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)


if __name__ == '__main__':
    main()
