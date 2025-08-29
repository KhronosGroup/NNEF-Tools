# Copyright (c) 2017-2025 The Khronos Group Inc.
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
from .optimization.sknd_transposer import NXCtoNCX, NCXtoNXC
from .io.skriptnd import Reader, Writer
import os


def get_transposer(source_format, target_format):
    if source_format == "NXC" and target_format == "NCX":
        return NXCtoNCX()
    elif source_format == "NCX" and target_format == "NXC":
        return NCXtoNXC()
    else:
        return None


def main(args):
    reader = Reader(atomic=True)
    writer = Writer()
    transposer = get_transposer(args.input_format.upper(), args.output_format.upper())

    name, ext = os.path.splitext(args.input_model)
    default_output_model = name + '.' + args.output_format + ext

    model = reader(args.input_model, init_data=False)
    transposer(model, inputs_to_transpose=args.inputs_to_transpose)

    writer(model, args.output_model or default_output_model, include_variables=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-model', type=str, required=True,
                        help='The input model')
    parser.add_argument('--output-model', type=str, default=None,
                        help='The output model')
    parser.add_argument('--input-format', type=str, required=True,
                        choices=['nxc', 'ncx'], help='The data format of the input model')
    parser.add_argument('--output-format', type=str, required=True,
                        choices=['nxc', 'ncx'], help='The data format of the output model')
    parser.add_argument('--inputs-to-transpose', type=int, nargs='+', default=None,
                        help='The indices of inputs that need transposing')
    args = parser.parse_args()

    exit(main(args))
