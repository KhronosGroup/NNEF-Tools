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
import os
import nnef

import numpy as np

from nnef_tools.io.input_source import InputSources
from nnef_tools.core import utils
from nnef_tools.io.nnef import nnef_io
from nnef_tools.io.nnef.parser_config import NNEFParserConfig


def _evaluate_input_shape(shape):
    try:
        if shape is None:
            return None
        elif len(shape) == 1 and not shape[0].isnumeric():
            return eval(shape[0])
        else:
            return [int(dim) for dim in shape]
    except Exception:
        raise utils.NNEFToolsException("Can not evaluate --input-shape {}".format(' '.join(shape)))


def get_args(argv):
    parser = argparse.ArgumentParser(
        description="NNEF-Tools/generate_inputs.py: Generate inputs for an NNEF network.\n",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Tips:
- If you refer to a Python package or module that is not in the current directory,
please add its location to PYTHONPATH.
- Quote parameters if they contain spaces or special characters.
""")
    parser.add_argument("network", help="Path of an NNEF model.\n")

    parser.add_argument("--params",
                        default=None,
                        required=False,
                        help="""An input_source or a tensor_name->input_source dictionary in Python syntax.
The dictionary can also have "*" or a tuple of names as key, "*" meaning "for all other". 
Please quote this parameter.

The following input sources are supported:

Random(algo, *args):
    Random(uniform, min, max) for int and float, range: [min, max]
    Random(normal, mean, std) for float
    Random(binomial, num, true_prob) for int, range: [0, num]
    Random(bernoulli, true_prob) for bool
    Keyword arguments can not be used with Random.
    
Tensor('tensor.dat'):
    The argument must be the path of an NNEF tensor file (.dat).
    
Default: 
  scalar: Random('normal', 0.0, 1.0) 
  integer: Random('binomial', 255, 0.5) 
  logical: Random('bernoulli', 0.5)""")

    parser.add_argument("--seed",
                        required=False,
                        default=-1,
                        type=int,
                        help="Seed to use for random generation.\n"
                             "Default: -1 (Get the seed from /dev/urandom or the clock)")

    parser.add_argument('--custom-operations',
                        nargs='*',
                        help="""Custom modules: e.g. package.module1 package.module2""")

    parser.add_argument('--shape',
                        nargs="+",
                        required=False,
                        help="""Use this to override input shapes.

    Set all input shapes to [10, 224, 224, 3]:
    --shape 10 224 224 3

    Set different input shapes for each input:
    --shape "{'input_name_1': [10, 224, 224, 3], 'input_name_2': [10, 299, 299, 3]}" """)

    parser.add_argument('--output', required=False,
                        help="The path of the output directory.\n"
                             "By default the standard output is used, but only if the command is piped or redirected.")

    args = parser.parse_args(args=argv[1:])

    args.shape = _evaluate_input_shape(args.shape)

    return args


def main():
    try:
        args = get_args(sys.argv)

        if not args.output:
            if sys.stdout.isatty():
                raise utils.NNEFToolsException("No output provided.")
            utils.set_stdout_to_binary()

        args.params = InputSources(args.params)

        if args.seed != -1:
            np.random.seed(args.seed)

        parser_configs = NNEFParserConfig.load_configs(args.custom_operations, load_standard=True)
        reader = nnef_io.Reader(parser_configs=parser_configs, input_shape=args.shape)

        # read without weights
        graph = reader(os.path.join(args.network, 'graph.nnef') if os.path.isdir(args.network) else args.network)

        inputs = tuple(args.params.create_input(name=input.name,
                                                np_dtype=input.get_numpy_dtype(),
                                                shape=input.shape,
                                                allow_bigger_batch=True) for input in graph.inputs)

        if args.output:
            for tensor, array in zip(graph.inputs, inputs):
                nnef_io.write_nnef_tensor(os.path.join(args.output, tensor.name + '.dat'), array)
        else:
            for array in inputs:
                nnef.write_tensor(sys.stdout, array)
    except Exception as e:
        print('Error: {}'.format(e), file=sys.stderr)
        exit(1)


if __name__ == '__main__':
    main()
