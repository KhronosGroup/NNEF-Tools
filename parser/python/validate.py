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

import nnef
import argparse


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('path', type=str, help='path to the model to validate')
    ap.add_argument('--stdlib', type=str, help='file name of alternate standard operation definitions '
                                               '(defaults to all-primitive definitions)', default='')
    ap.add_argument('--lower', type=str, help='comma separated list of operations to lower (if defined as compound)',
                    default='')
    ap.add_argument('--shapes', action="store_true", help='perform shape validation as well')
    args = ap.parse_args()

    stdlib = ''
    if args.stdlib:
        try:
            with open(args.stdlib) as file:
                stdlib = file.read()
        except FileNotFoundError as e:
            print('Could not open file: ' + args.stdlib)
            exit(-1)

    try:
        graph = nnef.load_graph(args.path, stdlib=stdlib, lowered=args.lower.split(','))
    except nnef.Error as err:
        print('Parse error: ' + str(err))
        exit(-1)

    if args.shapes:
        try:
            nnef.infer_shapes(graph)
        except nnef.Error as err:
            print('Shape error: ' + str(err))
            exit(-1)

    print(nnef.format_graph(graph.name, graph.inputs, graph.outputs, graph.operations, graph.tensors,
                            annotate_shapes=args.shapes))
    print('Validation succeeded')
