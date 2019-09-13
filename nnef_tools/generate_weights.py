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
import tempfile
import shutil

import numpy as np

from nnef_tools.io.input_source import InputSources
from nnef_tools.core import utils
from nnef_tools.io.nnef import nnef_io
from nnef_tools.io.nnef.parser_config import NNEFParserConfig
from nnef_tools.io.nnef.nnef_graph import *


def get_args(argv):
    parser = argparse.ArgumentParser(
        description="NNEF-Tools/generate_weights.py: Generate weights for an NNEF network.\n"
                    "Existing weights are not overwritten.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("network", help="Path of an NNEF model or a text file.\n"
                                        "Given my_network.nnef (or my_network.txt), it generates my_network.nnef.tgz.")

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

    return parser.parse_args(args=argv[1:])


def generate_weights(g, nnef_path, output_dir, input_sources):
    # type: (NNEFGraph, str, str, InputSources)->bool
    did_write = False
    warned = False
    for tensor in g.tensors:
        if tensor.is_variable:
            dat_path = os.path.join(output_dir, tensor.label + '.dat')
            if os.path.exists(dat_path):
                if not warned:
                    print("Warning: leaving existing weights unchanged".format(tensor.name, tensor.label))
                    warned = True
            else:
                array = input_sources.create_input(tensor.name, np_dtype=tensor.get_numpy_dtype(), shape=tensor.shape)
                nnef_io.write_nnef_tensor(dat_path, array)
                did_write = True
    if not os.path.isdir(nnef_path):
        shutil.copy(nnef_path, os.path.join(output_dir, 'graph.nnef'))
        did_write = True
    return did_write


def main():
    try:
        args = get_args(sys.argv)

        args.params = InputSources(args.params)

        if args.seed != -1:
            np.random.seed(args.seed)

        parent_dir_of_input_model = os.path.dirname(utils.path_without_trailing_separator(args.network))

        tmp_dir = None
        if args.network.endswith('.tgz'):
            nnef_path = tmp_dir = tempfile.mkdtemp(prefix="nnef_", dir=parent_dir_of_input_model)
            utils.tgz_extract(args.network, nnef_path)
        else:
            nnef_path = args.network

        try:
            parser_configs = NNEFParserConfig.load_configs(args.custom_operations, load_standard=True)
            reader = nnef_io.Reader(parser_configs=parser_configs)

            # read without weights
            graph = reader(os.path.join(nnef_path, 'graph.nnef') if os.path.isdir(nnef_path) else nnef_path)
            if os.path.isdir(nnef_path):
                output_path = nnef_path
            elif nnef_path.endswith('.nnef') or nnef_path.endswith('.txt'):
                output_path = tmp_dir = tempfile.mkdtemp(prefix="nnef_", dir=parent_dir_of_input_model)
            else:
                assert False

            did_generate_weights = generate_weights(graph,
                                                    nnef_path,
                                                    output_path,
                                                    input_sources=args.params)
            nnef_path = output_path

            if tmp_dir and did_generate_weights:
                if args.network.endswith('.tgz'):
                    print("Info: Changing input archive")
                    shutil.move(args.network, args.network + '.nnef-tools-backup')
                    utils.tgz_compress(dir_path=nnef_path, file_path=args.network)
                    os.remove(args.network + '.nnef-tools-backup')
                else:
                    output_path = args.network.rsplit('.', 1)[0] + '.nnef.tgz'
                    backup_path = output_path + '.nnef-tools-backup'
                    if os.path.exists(output_path):
                        shutil.move(output_path, backup_path)
                    utils.tgz_compress(dir_path=nnef_path, file_path=output_path)
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir)
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)


if __name__ == '__main__':
    main()
