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

#include <iostream>
#include <fstream>
#include "nnef.h"


template<typename T>
std::ostream& operator<<( std::ostream& os, const std::vector<T>& v )
{
    for ( auto it = v.begin(); it != v.end(); ++it )
    {
        if ( it != v.begin() )
        {
            os << ", ";
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


int main( int argc, const char * argv[] )
{
    if ( argc < 2 )
    {
        std::cout << "Usage: nnef-validator <network-structure.nnef> (--option)*" << std::endl;
        std::cout << std::endl;
        std::cout << "Description of options:" << std::endl;
        std::cout << "--graph: parse graph file only" << std::endl;
        std::cout << "--custom <fn>: import custom fragments defined in a separate file" << std::endl;
        std::cout << "--atomic <op>: op to treat as atomic" << std::endl;
        std::cout << "--lower <op>: op to lower" << std::endl;
        return 0;
    }
    
    const std::string path = argv[1];
    
    bool graph_only = false;
    std::string customs;
    auto shapeFuncs = nnef::standardShapeFuncs();
    for ( int i = 2; i < argc; ++i )
    {
        if ( std::strcmp(argv[i], "--graph") == 0 )
        {
            graph_only = true;
        }
        else if ( std::strcmp(argv[i], "--custom") == 0 )
        {
            if ( argc > i+1 )
            {
                std::ifstream cis(argv[++i]);
                if ( !cis )
                {
                    std::cerr << "Could not open file: " << argv[i] << std::endl;
                    continue;
                }
                
                std::istreambuf_iterator<char> begin(cis), end;
                customs.insert(customs.end(), begin, end);
            }
            else
            {
                std::cerr << "Expected file name after " << argv[i] << std::endl;
            }
        }
        else if ( std::strcmp(argv[i], "--atomic") == 0 )
        {
            if ( argc > i+1 )
            {
                const std::string op = argv[++i];
                shapeFuncs[op] = nnef::inferShapeTransitive;
            }
            else
            {
                std::cerr << "Expected op name after " << argv[i] << std::endl;
            }
        }
        else if ( std::strcmp(argv[i], "--lower") == 0 )
        {
            if ( argc > i+1 )
            {
                const std::string op = argv[++i];
                shapeFuncs.erase(op);
            }
            else
            {
                std::cerr << "Expected op name after " << argv[i] << std::endl;
            }
        }
        else
        {
            std::cerr << "Unrecognized option: " << argv[i] << std::endl;
        }
    }
    
    nnef::Graph graph;
    std::string error;
    
    bool ok = graph_only ? nnef::parse_graph(path, "", graph, error, customs, shapeFuncs) : nnef::load_model(path, graph, error, customs, shapeFuncs);
    
    if ( !ok )
    {
        std::cout << error << std::endl;
        return -1;
    }
    
    std::cout << "-- Validation succeeded, printing lowered graph --" << std::endl << std::endl;
    
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
        std::cout << ");" << std::endl;
    }
    std::cout << "}" << std::endl;
    
    return 0;
}
