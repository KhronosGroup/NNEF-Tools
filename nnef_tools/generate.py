# Copyright (c) 2020 The Khronos Group Inc.
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

import argparse
import numpy as np
import nnef
import sys
import os


_nnef_dtype_to_numpy = {
    'scalar': np.float32,
    'integer': np.int32,
    'bool': np.bool,
}


def _is_lambda(v):
    LAMBDA = lambda: 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


def uniform(min=0, max=1):
    return lambda shape: np.random.uniform(min, max, shape)


def normal(mean=0, std=1):
    return lambda shape: np.random.normal(mean, std, shape)


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)

    try:
        distribution = eval(args.random)
        if not _is_lambda(distribution):
            distribution = distribution()
    except Exception as e:
        print("Could not evaluate distribution: " + str(e), file=sys.stderr)
        return -1

    graph = nnef.parse_file(os.path.join(args.model, 'graph.nnef'))

    for op in graph.operations:
        if args.weights and op.name == 'variable':
            label = op.attribs['label']
            shape = op.attribs['shape']
            data = distribution(shape).astype(_nnef_dtype_to_numpy[op.dtype])
            filename = os.path.join(args.model, label + '.dat')

            os.makedirs(os.path.split(filename)[0], exist_ok=True)
            with open(filename, 'wb') as file:
                nnef.write_tensor(file, data)

            if args.verbose:
                print("Generated weight '{}'".format(filename))

        if args.inputs and op.name == 'external':
            name = op.outputs['output']
            shape = op.attribs['shape']
            data = distribution(shape).astype(_nnef_dtype_to_numpy[op.dtype])
            filename = os.path.join(args.model, args.inputs, name + '.dat')

            os.makedirs(os.path.split(filename)[0], exist_ok=True)
            with open(filename, 'wb') as file:
                nnef.write_tensor(file, data)

            if args.verbose:
                print("Generated input '{}'".format(filename))

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='The model to generate')
    parser.add_argument('--random', type=str, required=True,
                        help='Random distribution for input generation')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for input generation')
    parser.add_argument('--weights', action='store_true',
                        help='Generate weights')
    parser.add_argument('--inputs', type=str, nargs='?', default=None, const='.',
                        help='Generate inputs')
    parser.add_argument('--verbose', action='store_true',
                        help='Weather to print generated file names')
    exit(main(parser.parse_args()))
