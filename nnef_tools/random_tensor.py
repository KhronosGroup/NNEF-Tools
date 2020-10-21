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

from .utils import stdio
import numpy as np
import argparse
import nnef
import sys


def _is_lambda(v):
    LAMBDA = lambda: 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


def uniform(min=0, max=1):
    return lambda shape: np.random.uniform(min, max, shape)


def normal(mean=0, std=1):
    return lambda shape: np.random.normal(mean, std, shape)


def bernoulli(prob=0.5):
    return lambda shape: np.random.uniform(0, 1, shape) < prob


def main(args):
    if args.output is None:
        if not stdio.is_stdout_piped():
            print("Output must be piped", file=sys.stderr)
            return -1
        stdio.set_stdout_to_binary()

    try:
        distribution = eval(args.distribution)
        if not _is_lambda(distribution):
            distribution = distribution()
    except Exception as e:
        print("Could not evaluate distribution: " + str(e), file=sys.stderr)
        return -1

    tensor = distribution(args.shape).astype(np.dtype(args.dtype))

    if args.output is not None:
        with open(args.output, 'wb') as file:
            nnef.write_tensor(file, tensor)
    else:
        nnef.write_tensor(sys.stdout, tensor)
        
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('distribution', type=str,
                        help='The distribution to generate values from')
    parser.add_argument('--shape', type=int, nargs='+', required=True,
                        help='The dimensions of the tensor to generate')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='The data-type of the resulting tensor')
    parser.add_argument('--output', type=str, default=None,
                        help='File name to save the result into')
    exit(main(parser.parse_args()))
