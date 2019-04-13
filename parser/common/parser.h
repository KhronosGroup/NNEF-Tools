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

#ifndef _NNEF_PARSER_H_
#define _NNEF_PARSER_H_

#include "value.h"
#include "lexer.h"
#include "prototype.h"
#include "dictionary.h"
#include <functional>


namespace nnef
{

    class Parser
    {
    public:

        typedef std::pair<int,int> version_t;
        typedef std::vector<std::string> extensions_t;

        enum Flags { KHR_ENABLE_FRAGMENT_DEFINITIONS = 0x1, KHR_ENABLE_OPERATOR_EXPRESSIONS = 0x2 };

    public:

        struct Callback
        {
            virtual ~Callback() {}
            
            virtual void beginDocument( const std::string& filename, const version_t& version ) {}
            virtual void endDocument( const std::string& filename ) {}

            virtual bool handleExtension( const std::string& extension ) { return false; }

            virtual void beginGraph( const Prototype& proto, const Dictionary<Prototype>& fragments ) {}
            virtual void endGraph( const Prototype& proto, const Dictionary<Typename>& dtypes ) {}

            virtual void operation( const Prototype& proto, const Dictionary<Value>& args, const Dictionary<Typename>& dtypes ) = 0;
        };

    public:
        
        virtual ~Parser() {}

        virtual void parse( std::istream& is, const char* filename, Callback& callback ) = 0;

    protected:

        static Typename getTypename( Lexer& lexer )
        {
            switch ( lexer.token() )
            {
                case Lexer::Integer:
                    return Typename::Integer;
                case Lexer::Scalar:
                    return Typename::Scalar;
                case Lexer::Logical:
                    return Typename::Logical;
                case Lexer::String:
                    return Typename::String;
                case '?':
                    return Typename::Generic;
                default:
                    throw Error(lexer.position(), "expected type name, found '%s'", Lexer::tokenString(lexer.token()).c_str());
            }
        }

        static version_t readVersion( Lexer& lexer )
        {
            lexer.readToken(Lexer::Version);

            if ( lexer.token() != Lexer::Fractional )
            {
                throw Error(lexer.position(), "expected version number");
            }

            auto str = lexer.string();

            const size_t dots = std::count(str.begin(), str.end(), '.');
            bool isdigits = std::all_of(str.begin(), str.end(), []( char ch ){ return std::isdigit(ch) || ch == '.'; });

            if ( !isdigits || dots != 1 )
            {
                throw Error(lexer.position(), "invalid version number format: %s", str.c_str());
            }

            lexer.next();

            auto dot = str.find('.');
            auto major = std::atoi(str.substr(0,dot).c_str());
            auto minor = std::atoi(str.substr(dot+1).c_str());

            static const version_t MaxSupportedVersion(1,0);

            auto version = version_t(major,minor);
            if ( version > MaxSupportedVersion )
            {
                throw Error(lexer.position(), "unsupported version %d.%d; maximum supported version is %d.%d",
                            (int)major, (int)minor, (int)MaxSupportedVersion.first, (int)MaxSupportedVersion.second);
            }

            lexer.readToken(';');

            return version;
        }

        static extensions_t readExtensions( Lexer& lexer, std::function<bool( const std::string& )> handler )
        {
            extensions_t extensions;

            while ( lexer.readIfToken(Lexer::Extension) )
            {
                do
                {
                    auto position = lexer.position();

                    extensions.push_back(lexer.string());
                    lexer.readToken(Lexer::Identifier);

                    if ( !handler(extensions.back()) )
                    {
                        throw Error(position, "could not handle extension '%s'", extensions.back().c_str());
                    }
                }
                while ( lexer.readIfToken(',') );

                lexer.readToken(';');
            }

            return extensions;
        }
    };

}   // namespace nnef


#endif
