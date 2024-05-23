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


def shuffle_shape(input, groups):
    assert input[1] % groups == 0, "input channels ({}) is not divisible by groups ({})".format(input[1], groups)
    return input


graph = nnef.parse_string(
    """
    version 1.0;
    extension KHR_enable_fragment_definitions;

    fragment shuffle<?>( input: tensor<?>, groups: integer ) -> ( output: tensor<?> );

    graph Net( input ) -> ( output )
    {
        input = external(shape = [1,3,224,224]);
        filter = variable(shape = [32,3,5,5], label = 'conv/filter');
        conv = conv(input, filter);
        output = shuffle(conv, groups = 4);
    }
    """
)

nnef.infer_shapes(graph, custom_shapes={'shuffle': shuffle_shape})

print(nnef.format_graph(graph.name, graph.inputs, graph.outputs, graph.operations, graph.tensors))
