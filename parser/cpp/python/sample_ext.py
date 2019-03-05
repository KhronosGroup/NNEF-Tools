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


def shuffle_shape(op, args, shapes):
    shapes[args['output']] = shapes[args['input']]


nnef._register_custom_ops("shuffle", "fragment shuffle<?>( input: tensor<?>, groups: integer ) -> ( output: tensor<?> );")
nnef._register_custom_shapes({"shuffle": shuffle_shape})


graph = nnef.parse_string(
    """
    version 1.0;
    graph Net( input ) -> ( output )
    {
        input = external(shape = [1,3,224,224]);
        filter = variable(shape = [32,3,5,5], label = 'conv/filter');
        conv = conv(input, filter);
        output = shuffle(conv, groups = 4);
    }
    """
)

print(nnef.format_graph(graph.name, graph.inputs, graph.outputs, graph.operations))
