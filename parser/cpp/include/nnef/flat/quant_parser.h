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

#ifndef _NNEF_QUANTIZATION_H_
#define _NNEF_QUANTIZATION_H_

#include "../common/lexer.h"
#include "../common/error.h"
#include "../common/prototype.h"
#include "../common/dictionary.h"
#include "flat_parser.h"
#include <iostream>
#include <sstream>


namespace nnef
{

    class QuantParser : public FlatParser
    {
    public:

        static Dictionary<Dictionary<Value>> parse( std::istream& is, const char* filename, const Dictionary<Prototype>& prototypes )
        {
            Lexer lexer(is, filename);
            lexer.next();

            Dictionary<Dictionary<Value>> quantization;

            for ( unsigned line = 0; lexer.token() != Lexer::Eof; ++line )
            {
                const std::string tensor = lexer.string();
                if ( quantization.count(tensor) )
                {
                    throw Error(lexer.position(), "duplicate quantization entries for tensor '%s'", tensor.c_str());
                }

                lexer.readToken(Lexer::Characters);
                lexer.readToken(':');

                auto args = parseInvocation(lexer, prototypes);

                quantization.emplace(tensor, std::move(args));
            }

            return quantization;
        }

    private:

        static Dictionary<Value> parseInvocation( Lexer& lexer, const Dictionary<Prototype>& prototypes )
        {
            Position position = lexer.position();

            const std::string op = lexer.string();
            lexer.readToken(Lexer::Identifier);

            auto it = prototypes.find(op);
            if ( it == prototypes.end() )
            {
                throw Error(position, "undefined quantization operation '%s'", op.c_str());
            }

            auto& proto = it->second;
            if ( !proto.paramCount() )
            {
                throw Error(position, "quantization operation must have at least one parameter");
            }
            if ( proto.param(0).type()->kind() != Type::Tensor )
            {
                throw Error(position, "first parameter of quantization operation must be of type tensor");
            }

            lexer.readToken('(');

            Dictionary<Value> args = parseArguments(proto, lexer, nullptr, nullptr, false, true, true, &proto.param(0));

            lexer.readToken(')');
            lexer.readToken(';');

            args["op-name"] = Value::string(op);

            return args;
        }
    };

}   // namespace nnef


#endif
