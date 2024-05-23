/*
 * Copyright (c) 2017 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnef.h"
#include <string>
#include <iostream>


int main( int argc, const char * argv[] )
{
    if ( argc < 2 )
    {
        std::cerr << "Input file name must be provided" << std::endl;
        return -1;
    }
    
    const std::string path = argv[1];
    
    nnef::Graph graph;
    std::string error;
    
    if ( !nnef::load_graph(path, graph, error, "") )
    {
        std::cerr << error << std::endl;
        return -1;
    }
    
    if ( !nnef::infer_shapes(graph, error) )
    {
		std::cerr << error << std::endl;
		return -1;
    }
    
    std::cerr << "Successfully parsed file: " << path << std::endl;
    
    return 0;
}
