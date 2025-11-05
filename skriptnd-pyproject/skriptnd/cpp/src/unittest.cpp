/*
 * Copyright (c) 2017-2025 The Khronos Group Inc.
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

#include <fstream>
#include <filesystem>
#include "skriptnd.h"
#include "runtime.h"


int main( int argc, const char * argv[] )
{
    auto error_handler = [&]( const sknd::Position& position, const std::string& message, const sknd::StackTrace& stacktrace, const bool warning )
    {
        std::cout << (warning ? "⚠️ Warning" : "🛑 Error") << " in module '" << position.module << "'";
        if ( position.line )
        {
            std::cout << " [" << position.line << ':' << position.column << "]";
        }
        std::cout << ": " << message << std::endl;
        for ( auto it = stacktrace.rbegin(); it != stacktrace.rend(); ++it )
        {
            auto& [op, pos] = *it;
            std::cout << "\twhile calling operator '" << op << "' in module '" << pos.module;
            std::cout << "' [" << pos.line << ':' << pos.column << "]" << std::endl;
        }
    };
    
    if ( argc < 2 )
    {
        std::cerr << "Test folder must be provided" << std::endl;
        return -1;
    }
    
    const std::string folder = argv[1];
    
    bool atomic = false;
    bool unroll = false;
    for ( size_t i = 2; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if ( arg == "--atomic" )
        {
            atomic = true;
        }
        else if ( arg == "--unroll" )
        {
            unroll = true;
        }
    }
    
    size_t passed = 0;
    size_t failed = 0;
    for ( auto& entry : std::filesystem::recursive_directory_iterator(folder) )
    {
        if ( entry.is_directory() && entry.path().extension() == ".nnef2" )
        {
            auto filename = entry.path().string() + "/main.sknd";
            std::ifstream is(filename);
            if ( !is )
            {
                std::cerr << "Could not open file: " << filename << std::endl;
                return -1;
            }
            
            auto model = sknd::read_model(is, "main", "", "skriptnd/stdlib/", "", error_handler);
            if ( model )
            {
                if ( atomic )
                {
                    sknd::flatten_model(*model, sknd::TrueOperationFilter);
                }
                std::cout << "✅ Succesfully parsed model " << entry.path().filename() << std::endl;
                ++passed;
            }
            else
            {
                std::cout << "❌ Failed to parse " << entry.path().filename() << std::endl;
                ++failed;
            }
        }
    }
    
    if ( failed )
    {
        std::cout << failed << " test cases failed, " << passed << " passed" << std::endl;
    }
    else
    {
        std::cout << "All test cases passed" << std::endl;
    }
    
    return 0;
}
