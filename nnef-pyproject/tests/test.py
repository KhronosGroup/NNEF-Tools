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

import unittest
import nnef


class ParserTest(unittest.TestCase):

    def test_empty_document(self):
        with self.assertRaises(nnef.Error):
            nnef.parse_string("")

    def test_empty_body(self):
        with self.assertRaises(nnef.Error):
            nnef.parse_string("version 1.0; graph G( input ) -> ( output ) {}")

    def test_minimal(self):
        nnef.parse_string("""
            version 1.0;
            graph G( input ) -> ( output )
            {
                input = external(shape = []);
                output = copy(input);
            }
            """)

    def test_empty_input_not_declared(self):
        with self.assertRaises(nnef.Error):
            nnef.parse_string("""
                version 1.0;
                graph G( input ) -> ( output )
                {
                    output = copy(input);
                }
                """)

    def test_input_not_external(self):
        with self.assertRaises(nnef.Error):
            nnef.parse_string("""
                version 1.0;
                graph G( input ) -> ( output )
                {
                    input = constant(shape = [], value = [1.0]);
                    output = copy(input);
                }
                """)

    def test_external_not_input(self):
        with self.assertRaises(nnef.Error):
            nnef.parse_string("""
                version 1.0;
                graph G( input ) -> ( output )
                {
                    input = external(shape = []);
                    other = external(shape = []);
                    output = add(input, other);
                }
                """)

    def test_empty_output_not_declared(self):
        with self.assertRaises(nnef.Error):
            nnef.parse_string("""
                version 1.0;
                graph G( input ) -> ( output )
                {
                    input = external(shape = []);
                }
                """)

    def test_variable_update(self):
        nnef.parse_string("""
            version 1.0;
            graph G( input ) -> ( output )
            {
                input = external(shape = []);
                var = variable(shape = [], label = 'var');
                output = update(var, input);
            }
            """)

    def test_non_variable_update(self):
        with self.assertRaises(nnef.Error):
            nnef.parse_string("""
                version 1.0;
                graph G( input ) -> ( output )
                {
                    input = external(shape = []);
                    output = update(input, input);
                }
                """)

    def test_custom_fragment(self):
        nnef.parse_string("""
            version 1.0;
            extension KHR_enable_fragment_definitions, KHR_enable_operator_expressions;
            
            fragment op( input: tensor<scalar> ) -> ( output: tensor<scalar> )
            {
                output = input;
            }
            
            graph G( input ) -> ( output )
            {
                input = external(shape = []);
                output = op(input);
            }
            """)
    
    def test_reshape(self):
        graph = nnef.parse_string("""
            version 1.0;
            graph G( input ) -> ( output )
            {
                input = external(shape = [1,2,3,4]);
                output = reshape(input, axis_start = 1, axis_count = 2, shape = [6]);
            }
            """)
        nnef.infer_shapes(graph)

    # def test_alexnet(self):
    #     nnef.parse_file("../examples/alexnet.txt")
    #
    # def test_googlenet(self):
    #     nnef.parse_file("../examples/googlenet.txt")
    #
    # def test_resnet(self):
    #     nnef.parse_file("../examples/resnet.txt")
    #
    # def test_vggnet(self):
    #     nnef.parse_file("../examples/vgg.txt")


if __name__ == '__main__':
    unittest.main()
