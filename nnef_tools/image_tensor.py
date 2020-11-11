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
import skimage
import skimage.io
import skimage.color
import skimage.transform
import glob
import os


def transform_image(img, color, range, mean, std, size, dtype, data_format):
    img = img.astype(np.float32) / 255.0

    if color.upper() == 'RGB':
        img = img[..., (0, 1, 2)]  # remove alpha channel if present
    else:
        img = img[..., (2, 1, 0)]

    if range is not None:
        min = np.array(range[0], dtype=np.float32)
        max = np.array(range[1], dtype=np.float32)
        img *= max - min
        img += min

    if mean is not None:
        mean = np.array(mean, dtype=np.float32)
        img -= mean

    if std is not None:
        std = np.array(std, dtype=np.float32)
        img /= std

    if size is not None:
        img = skimage.transform.resize(img, size,
                                       preserve_range=True,
                                       anti_aliasing=True,
                                       mode='reflect')
    if dtype is not None:
        img = img.astype(dtype)

    if data_format.upper() == 'NCHW':
        img = img.transpose((2, 0, 1))

    return img


def main(args):
    if args.output is None:
        if not stdio.is_stdout_piped():
            print("Output must be piped", file=sys.stderr)
            return -1
        stdio.set_stdout_to_binary()

    images = []
    for pattern in args.images:
        filenames = sorted(glob.glob(os.path.expanduser(pattern)))
        assert filenames, "No files found for path: {}".format(pattern)
        for filename in filenames:
            img = skimage.img_as_ubyte(skimage.io.imread(filename))
            if len(img.shape) == 2:
                img = skimage.color.gray2rgb(img)

            img = transform_image(img, args.color, args.range, args.mean, args.std, args.size,
                                  np.dtype(args.dtype), args.format)
            images.append(img)

    if not all(img.shape == images[0].shape for img in images):
        print("The size of all images must be the same, or --size must be specified", file=sys.stderr)
        return -1

    tensor = np.stack(images)

    if args.output is not None:
        with open(args.output, 'wb') as file:
            nnef.write_tensor(file, tensor)
    else:
        nnef.write_tensor(sys.stdout, tensor)
    
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images', type=str, nargs='+',
                        help='The path(s) of images to turn into a tensor; may include wildcard expressions')
    parser.add_argument('--size', type=int, nargs=2, default=None,
                        help='The spatial size of the resulting tensor')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='The data-type of the resulting tensor')
    parser.add_argument("--color", type=str.upper, choices=['RGB', 'BGR'], default='RGB',
                        help="The resulting color-format")
    parser.add_argument("--format", type=str.upper, choices=['NCHW', 'NHWC'], default='NCHW',
                        help="The resulting data-format")
    parser.add_argument("--range", type=float, nargs=2, default=[0, 1],
                        help="Resulting range for representing the image")
    parser.add_argument("--mean", type=float, nargs='+', default=None,
                        help="Mean to subtract from the image; may be per-channel")
    parser.add_argument("--std", type=float, nargs='+', default=None,
                        help="Standard deviation to divide the image with; may be per-channel")
    parser.add_argument('--output', type=str, default=None,
                        help='File name to save the result into')
    exit(main(parser.parse_args()))
