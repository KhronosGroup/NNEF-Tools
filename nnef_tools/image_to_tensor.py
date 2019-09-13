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

import numpy as np
import nnef

from nnef_tools.io.input_source import ImageInput, create_input
from nnef_tools.io.nnef.nnef_io import write_nnef_tensor
from nnef_tools.core import utils


def get_args(argv):
    parser = argparse.ArgumentParser(
        description="NNEF-Tools/image_to_tensor.py: Create tensor from (a batch of) image(s)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Image processing steps:
    1. The image is loaded as float32.
    2. The image is resized to the given size if specified.
    3. The image is transformed to RGB or BGR as requested.
    4. The image is transformed to the specified range. 
    5. The image is normalized as follows: image = (image - mean) / std
    6. The image is transformed to NCHW or NHWC as requested. 
    7. The image is casted to the requested dtype.""")

    parser.add_argument('input', nargs='+', help="The path or paths of images, e.g. image.jpg, *.jpg, etc.")
    parser.add_argument('--output', required=False,
                        help="The path of the output tensor, e.g. tensor.dat.\n"
                             "By default the standard output is used, but only if the command is piped or redirected.")

    parser.add_argument("--color",
                        choices=['RGB', 'BGR'],
                        default='RGB',
                        type=str.upper,
                        help="Input color format. Default: RGB")

    parser.add_argument("--format",
                        choices=['NCHW', 'NHWC'],
                        default='NCHW',
                        type=str.upper,
                        help="Input data format. Default: NCHW")

    parser.add_argument("--range",
                        nargs=2,
                        type=float,
                        default=[0, 255],
                        help="Range for representing the image. Default: 0 255")

    parser.add_argument("--mean",
                        nargs='+',
                        type=float,
                        default=[0],
                        help="Mean to subtract from the image. Default: 0\n"
                             "Can be channel-wise, e.g. 127 128 129")

    parser.add_argument("--std",
                        nargs='+',
                        type=float,
                        default=[1],
                        help="Standard deviation to divide the image with. Default: 1\n"
                             "Can be channel-wise, e.g. 127 128 129")

    parser.add_argument("--size",
                        nargs='+',
                        required=False,
                        type=int,
                        help="Target image size: width [height]. Default: The size of the given images.\n"
                             "E.g. 224 or 640 480")

    parser.add_argument("--dtype",
                        default="float32",
                        help="Numpy dtype of the generated tensor (float32, etc.). Default: float32")

    args = parser.parse_args(args=argv[1:])

    if args.size is not None:
        if len(args.size) == 1:
            args.size *= 2
        if len(args.size) != 2:
            raise utils.NNEFToolsException("The --size parameter must have 1 or 2 arguments if given!")

    return args


def main():
    try:
        args = get_args(sys.argv)

        if not args.output:
            if sys.stdout.isatty():
                raise utils.NNEFToolsException("No output provided.")
            utils.set_stdout_to_binary()

        image_input = ImageInput([os.path.join(path, '*') if os.path.isdir(path) else path for path in args.input],
                                 color_format=args.color,
                                 data_format=args.format,
                                 range=args.range,
                                 norm=[args.mean, args.std])
        shape = None
        if args.size is not None:
            shape = [1, 3, args.size[1], args.size[0]] if args.format == 'NCHW' else [1, args.size[1], args.size[0], 3]
        arr = create_input(image_input, np_dtype=np.dtype(args.dtype), shape=shape, allow_bigger_batch=True)

        if args.output:
            write_nnef_tensor(args.output, arr)
        else:
            nnef.write_tensor(sys.stdout, arr)
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)


if __name__ == '__main__':
    main()
