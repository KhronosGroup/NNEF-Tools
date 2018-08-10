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
#include <cstring>
#include <memory>
#include <regex>
#include <map>
#include "flat/flat_parser.h"
#include "comp/comp_parser.h"
#include "flat/quant_parser.h"
#include "comp/layers_source.h"
#include "common/binary.h"


static const size_t MaxDims = 8;


struct PrintCallback : public nnef::Parser::Callback, public nnef::Propagation
{
    std::map<std::string,bool> atomics;
    std::istream& qis;
    const char* qfn;

    nnef::Dictionary<nnef::Dictionary<nnef::Value>> quantizations;

    PrintCallback( const std::map<std::string,bool>& atomics, std::istream& qis, const char* qfn )
    : nnef::Propagation(MaxDims), atomics(atomics), qis(qis), qfn(qfn)
    {
    }

    virtual void beginDocument( const std::string& filename, const nnef::Parser::version_t& version )
    {
        std::cout << "version " << version.first << "." << version.second << ";" << std::endl;
    }

    virtual bool handleExtension( const std::string& ext )
    {
        std::cout << "extension " << ext << ";" << std::endl;
        return false;
    }

    virtual void beginGraph( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Prototype>& fragments )
    {
        if ( qis )
        {
            quantizations = nnef::QuantParser::parse(qis, qfn, fragments);
        }

        std::cout << "graph " << proto.name() << "( ";

        bool first = true;
        for ( size_t i = 0; i < proto.paramCount(); ++i )
        {
            auto& param = proto.param(i);

            if ( !param.type()->isAttribute() )
            {
                if ( !first )
                {
                    std::cout << ", ";
                }
                std::cout << param.name();

                first = false;
            }
        }
        
        std::cout << " ) -> ( ";
        
        for ( size_t i = 0; i < proto.resultCount(); ++i )
        {
            auto& result = proto.result(i);
            
            if ( i )
            {
                std::cout << ", ";
            }
            std::cout << result.name();
        }
        
        std::cout << " )" << std::endl << '{' << std::endl;
    }

    virtual void endGraph( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Typename>& dtypes, const nnef::Dictionary<nnef::Shape>& shapes )
    {
        std::cout << '}' << std::endl;

        for ( auto& it : quantizations )
        {
            if ( !dtypes.contains(it.first) )
            {
                std::cerr << "Warning: quantization info found for undeclared tensor '" << it.first << "'" << std::endl;
            }
        }
    }

    virtual void operation( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Value>& args,
                           const nnef::Dictionary<nnef::Typename>& dtypes, const nnef::Dictionary<nnef::Shape>& shapes )
    {
        std::cout << '\t';

        for ( size_t i = 0; i < proto.resultCount(); ++i )
        {
            auto& result = proto.result(i);

            if ( i )
            {
                std::cout << ", ";
            }
            std::cout << args[result.name()];
        }

        std::cout << " = " << proto.name();

        if ( proto.isGeneric() )
        {
            auto dtype = args["?"];
            if ( dtype )
            {
                std::cout << '<' << dtype.string() << '>';
            }
        }

        std::cout << "(";

        bool named = false;
        for ( size_t i = 0; i < proto.paramCount(); ++i )
        {
            auto& param = proto.param(i);
            named |= param.type()->isAttribute();

            if ( i )
            {
                std::cout << ", ";
            }
            if ( named )
            {
                std::cout << param.name() << " = ";
            }
            std::cout << args[param.name()];
        }

        std::cout << ");" << std::endl;
    }
    
    virtual bool isAtomic( const nnef::Prototype& proto, const nnef::Dictionary<nnef::Value>& args )
    {
        auto it = atomics.find(proto.name());
        if ( it != atomics.end() )
        {
            return it->second;
        }
        return nnef::getPropagationGroup(proto.name()) != nnef::PropagationGroup::Unknown;
    }

    virtual bool shouldDeferShapeOf( const nnef::Prototype& proto, const std::string& param ) const
    {
        return param == "shape" || param.substr(param.length() - 6) == "_shape";
    }
};


static void parseAtomics( const char* str, std::map<std::string,bool>& atomics, bool add_or_remove )
{
    std::regex reg("\\s+");
    std::cregex_token_iterator it(str, str + strlen(str), reg, -1);
    while ( it != std::cregex_token_iterator() )
    {
        auto op = (*it++).str();
        atomics.emplace(op, add_or_remove);
    }
}


int main( int argc, const char * argv[] )
{
    if ( argc < 2 )
    {
        std::cout << "Usage: nnef-validator <network-structure.nnef> (--option)*" << std::endl;
        std::cout << std::endl;
        std::cout << "Description of options:" << std::endl;
        std::cout << "--flat: use flat parser instead of compositional" << std::endl;
        std::cout << "--layers: enable predefined layer level fragments" << std::endl;
        std::cout << "--binary: check binary data files for variables" << std::endl;
        std::cout << "--quant: check quantization file" << std::endl;
        std::cout << "--atomic <op names>: ops to treat as atomic" << std::endl;
        std::cout << "                default list includes standard ops" << std::endl;
        std::cout << "--no-atomic <op names>: ops to treat as non-atomic" << std::endl;
        return 0;
    }
    
    const std::string filename = argv[1];
    
    std::ifstream is(filename.c_str());
    if ( !is )
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        return -1;
    }

    bool flat = false;
    bool layers = false;
    bool binary = false;
    bool quant = false;
    std::map<std::string,bool> atomics;
    for ( int i = 2; i < argc; ++i )
    {
        if ( std::strcmp(argv[i], "--flat") == 0 )
        {
            flat = true;
        }
        else if ( std::strcmp(argv[i], "--layers") == 0 )
        {
            layers = true;
        }
        else if ( std::strcmp(argv[i], "--binary") == 0 )
        {
            binary = true;
        }
        else if ( std::strcmp(argv[i], "--quant") == 0 )
        {
            quant = true;
        }
        else if ( std::strcmp(argv[i], "--atomic") == 0 || std::strcmp(argv[i], "--no-atomic") == 0 )
        {
            bool add_or_remove = std::strncmp(argv[i], "--no", 4) != 0;
            if ( argc > i+1 )
            {
                parseAtomics(argv[++i], atomics, add_or_remove);
            }
            else
            {
                std::cerr << "Expected list of op names after " << argv[i] << std::endl;
            }
        }
        else
        {
            std::cerr << "Unrecognized option: " << argv[i] << std::endl;
        }
    }

    const std::string quantFilename = filename.substr(0, filename.find_last_of(".")) + ".quant";
    std::ifstream qis;
    if ( quant )
    {
        qis.open(quantFilename.c_str());
        if ( !qis )
        {
            std::cerr << "Could not open file: " << quantFilename << std::endl;
            return -1;
        }
    }

    PrintCallback callback(atomics, qis, quantFilename.c_str());

    try
    {
        if ( flat )
        {
            nnef::FlatParser parser(callback);
            parser.parse(is, filename.c_str(), callback);
        }
        else
        {
            nnef::CompParser parser(callback);
            if ( layers )
            {
                parser.import("layers", nnef::layers_source());
            }
            parser.parse(is, filename.c_str(), callback);
        }

        std::cout << "Parse succeeded" << std::endl;
    }
    catch ( nnef::Error e )
    {
        printf("Parse error in file '%s' [%u:%u] %s\n", e.position().filename, e.position().line, e.position().column, e.what());

        auto origin = e.position().origin;
        while ( origin )
        {
            printf("... evaluated from file '%s' [%u:%u]\n", origin->filename, origin->line, origin->column);
            origin = origin->origin;
        }
    }

    if ( binary )
    {
        for ( auto it : callback.variableShapes() )
        {
            const std::string binaryFilename = filename.substr(0, filename.find_last_of('/') + 1) + it.first + ".dat";
            std::ifstream bin(binaryFilename.c_str(), std::ios::binary);
            if ( !bin )
            {
                std::cerr << "Could not open file: " << binaryFilename << std::endl;
                continue;
            }

            nnef::TensorHeader header;
            bin.read((char*)&header, sizeof(header));

            try
            {
                nnef::validate_tensor_header(header);
            }
            catch ( nnef::Error e )
            {
                std::cerr << "Failed to read binary header from file: " << binaryFilename << std::endl;
                std::cerr << e.what() << std::endl;
                continue;
            }

            nnef::Shape shape(header.rank, header.extents);

            if ( shape != it.second )
            {
                std::cerr << "Shape " << shape << " in tensor file '" << binaryFilename << "' does not match shape "
                          << it.second << " defined in network structure" << std::endl;
            }
        }
    }

    return 0;
}
