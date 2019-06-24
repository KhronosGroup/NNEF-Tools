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
#include <iostream>
#include <fstream>
#include <cstring>


template<typename T>
std::ostream& print( std::ostream& os, const std::vector<T>& v, const char* sep = ", " )
{
	for ( auto it = v.begin(); it != v.end(); ++it )
	{
		if ( it != v.begin() )
		{
			os << sep;
		}
		os << *it;
	}
	return os;
}

std::ostream& print_items( std::ostream& os, const nnef::ValueDict& m )
{
    for ( auto it = m.begin(); it != m.end(); ++it )
    {
        if ( it != m.begin() )
        {
            os << ", ";
        }
        os << it->first << " = " << it->second;
    }
    return os;
}

std::ostream& print_values( std::ostream& os, const nnef::ValueDict& m )
{
    for ( auto it = m.begin(); it != m.end(); ++it )
    {
        if ( it != m.begin() )
        {
            os << ", ";
        }
        os << it->second;
    }
    return os;
}

template<typename T>
std::ostream& operator<<( std::ostream& os, const std::vector<T>& v )
{
	return print(os, v);
}

std::ostream& print_shapes( std::ostream& os, const nnef::Value& result, const nnef::Graph& graph )
{
	if ( result.kind() == nnef::Value::Identifier )
	{
		auto& tensor = graph.tensors.at(result.identifier());
		os << tensor.dtype << "[";
		print(os, tensor.shape, ",");
		os << "]";
	}
	else
	{
		for ( size_t i = 0; i < result.size(); ++i )
		{
			if ( i )
			{
				os << ", ";
			}
			print_shapes(os, result[i], graph);
		}
	}
	return os;
}


int main( int argc, const char * argv[] )
{
    if ( argc < 2 )
    {
        std::cout << "Usage: nnef-validator <path/to/NNEF/folder> (--option)*" << std::endl;
        std::cout << std::endl;
        std::cout << "Description of options:" << std::endl;
        std::cout << "--stdlib <path>: path to alternative stdlib source file" << std::endl;
        std::cout << "--lower <op>: op to lower" << std::endl;
        std::cout << "--shapes: infer tensor shapes" << std::endl;
        return 0;
    }
    
    const std::string path = argv[1];
    
    bool infer_shapes = false;
    std::string stdlib;
    std::set<std::string> lowered;
    for ( int i = 2; i < argc; ++i )
    {
        if ( std::strcmp(argv[i], "--stdlib") == 0 )
        {
            if ( argc > i+1 )
            {
                const char* path = argv[++i];
                std::ifstream is(path);
                if ( !is )
                {
                    std::cerr << "Could not open stdlib file: " << path << std::endl;
                    std::cerr << "Using default implementation as fallback" << std::endl;
                }
                else
                {
                    stdlib = std::string(std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>());
                }
            }
            else
            {
                std::cerr << "Expected path for stdlib file after " << argv[i] << std::endl;
            }
        }
        else if ( std::strcmp(argv[i], "--lower") == 0 )
        {
            if ( argc > i+1 )
            {
                const std::string op = argv[++i];
                lowered.insert(op);
            }
            else
            {
                std::cerr << "Expected op name after " << argv[i] << std::endl;
            }
        }
        else if ( std::strcmp(argv[i], "--shapes") == 0 )
        {
            infer_shapes = true;
        }
        else
        {
            std::cerr << "Unrecognized option: " << argv[i] << std::endl;
        }
    }
    
    nnef::Graph graph;
    std::string error;
    
    if ( !nnef::load_graph(path, graph, error, stdlib, lowered) )
    {
        std::cerr << error << std::endl;
        return -1;
    }
    
    if ( infer_shapes && !nnef::infer_shapes(graph, error) )
    {
		std::cerr << error << std::endl;
		return -1;
    }
    
    std::cout << "graph " << graph.name << "( " << graph.inputs << " ) -> ( " << graph.outputs << " )" << std::endl;
    std::cout << "{" << std::endl;
    for ( const auto& op : graph.operations )
    {
        std::cout << "\t";
		print_values(std::cout, op.outputs);
        std::cout << " = " << op.name;
        if ( !op.dtype.empty() )
        {
            std::cout << "<" << op.dtype << ">";
        }
        std::cout << "(";
        print_values(std::cout, op.inputs);
        if ( !op.inputs.empty() && !op.attribs.empty() )
        {
            std::cout << ", ";
        }
        print_items(std::cout, op.attribs);
        std::cout << ");";
		
		if ( infer_shapes )
		{
			std::cout << "    # ";
			for ( auto it = op.outputs.begin(); it != op.outputs.end(); ++it )
			{
				if ( it != op.outputs.begin() )
				{
					std::cout << ", ";
				}
				print_shapes(std::cout, it->second, graph);
			}
		}
		std::cout << std::endl;
    }
    std::cout << "}" << std::endl;
    std::cerr << "Validation succeeded" << std::endl;
    
    return 0;
}
