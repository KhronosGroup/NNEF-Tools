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

#ifndef _NNEF_FLAT_PARSER_H_
#define _NNEF_FLAT_PARSER_H_

#include "../common/dictionary.h"
#include "../common/typespec.h"
#include "../common/prototype.h"
#include "../common/parser.h"
#include "../common/value.h"
#include "../common/lexer.h"
#include "../common/error.h"
#include "../common/shape.h"
#include "stdlib_protos.h"
#include <cassert>


namespace nnef
{

    class FlatParser : public Parser
    {
        typedef Dictionary<Prototype> Prototypes;
        typedef Dictionary<const Type*> Types;

    public:

        typedef Error::Position Position;

    public:

        virtual void parse( std::istream& is, Callback& callback )
        {
            Lexer lexer(is);
            lexer.next();

            static auto prototypes = stdlib();

            parseVersion(lexer);
            parseGraph(lexer, prototypes, callback);
        }

    private:
        
        static void parseGraph( Lexer& lexer, const Prototypes& prototypes, Callback& callback )
        {
            readToken(Lexer::Graph, lexer);
            
            const std::string name = lexer.string();
            
            readToken(Lexer::Identifier, lexer);
            
            auto params = parseIdentifiers<Param>(lexer);
            
            readToken(Lexer::Arrow, lexer);
            
            auto results = parseIdentifiers<Result>(lexer);
            
            const Prototype graph(name, params, results);
            
            callback.beginGraph(graph);
            
            readToken('{', lexer);
            
            Types declared;
            Dictionary<Shape> shapes;
            
            while ( lexer.token() != '}' )
            {
                parseAssignment(lexer, graph, prototypes, declared, shapes, callback);
            }
            
            checkGraphParamsAssigned(graph, declared, lexer.position());
            
            readToken('}', lexer);
            
            callback.endGraph(graph);
        }
        
        template<typename T>
        static std::vector<T> parseIdentifiers( Lexer& lexer )
        {
            static const PrimitiveType TensorType(Typename::Scalar, true);
            
            std::vector<T> identifiers;
            
            readToken('(', lexer);
            
            do
            {
                const std::string id = lexer.string();
                
                readToken(Lexer::Identifier, lexer);
                
                identifiers.emplace_back(id, &TensorType);
            }
            while ( readIfToken(',', lexer) );
            
            readToken(')', lexer);
            
            return identifiers;
        }

        static void checkGraphParam( const Value& arg, const Prototype& graph, const std::string& target, const Position& position )
        {
            switch ( arg.kind() )
            {
                case Value::Tensor:
                {
                    if ( target == "external" )
                    {
                        if ( !graph.param(arg.tensor().id) )
                        {
                            throw Error(position, "identifier '%s' assigned by operation 'external' must be a graph parameter",
                                             arg.tensor().id.c_str());
                        }
                    }
                    else
                    {
                        if ( graph.param(arg.tensor().id) )
                        {
                            throw Error(position, "graph parameter '%s' can only be assigned by operation 'external'",
                                             arg.tensor().id.c_str());
                        }
                    }
                    break;
                }
                case Value::Array:
                case Value::Tuple:
                {
                    for ( size_t i = 0; i < arg.size(); ++i )
                    {
                        checkGraphParam(arg[i], graph, target, position);
                    }
                    break;
                }
                default:
                {
                    assert(false);
                }
            }
        }
        
        static void checkGraphParamsAssigned( const Prototype& graph, const Types& declared, const Position& position )
        {
            for ( size_t i = 0; i < graph.paramCount(); ++i )
            {
                auto& param = graph.param(i);
                if ( !declared.contains(param.name()) )
                {
                    throw Error(position, "graph parameter '%s' not assigned", param.name().c_str());
                }
            }
            
            for ( size_t i = 0; i < graph.resultCount(); ++i )
            {
                auto& result = graph.result(i);
                if ( !declared.contains(result.name()) )
                {
                    throw Error(position, "graph result '%s' not assigned", result.name().c_str());
                }
            }
        }

    private:

        static void parseAssignment( Lexer& lexer, const Prototype& graph, const Prototypes& prototypes,
                                    Types& declared, Dictionary<Shape>& shapes, Callback& callback )
        {
            auto position = lexer.position();

            const Value results = parseTuple(lexer, nullptr);

            readToken('=', lexer);
            
            const std::string target = lexer.string();

            readToken(Lexer::Identifier, lexer);

            auto it = prototypes.find(target);
            if ( it == prototypes.end() )
            {
                throw Error(lexer.position(), "undefined operation '%s'", target.c_str());
            }
            
            auto& proto = it->second;
            
            checkGraphParam(results, graph, proto.name(), position);

            readToken('(', lexer);

            Dictionary<Value> args = parseArguments(proto, lexer, &declared);

            readToken(')', lexer);
            readIfToken(';', lexer);

            if ( results.size() != proto.resultCount() )
            {
                throw Error(position, "left-hand-side item count must match result count of operation (%s)",
                                 (int)proto.resultCount());
            }

            for ( size_t i = 0; i < proto.resultCount(); ++i )
            {
                declare(results[i], proto.result(i).type(), declared, position);

                args.emplace(proto.result(i).name(), std::move(results[i]));
            }
            
            try
            {
                callback.propagate(proto, args, shapes);
                callback.operation(proto, args, shapes);
            }
            catch ( Error e )
            {
                throw Error(position, e.what());
            }
        }

        static Dictionary<Value> parseArguments( const Prototype& proto, Lexer& lexer, const Types* decls )
        {
            bool expectNamed = false;

            Dictionary<Value> args;

            do
            {
                auto position = lexer.position();

                if ( args.size() > proto.paramCount() )
                {
                    throw Error(position, "too many positional arguments; definition of '%s' has only %d parameters",
                                proto.name().c_str(), (int)proto.paramCount());
                }

                const Param* param = nullptr;
                Value arg = Value::none();

                bool named = false;
                if ( lexer.token() == Lexer::Identifier )
                {
                    auto string = lexer.string();
                    lexer.next();

                    if ( lexer.token() == '=' )
                    {
                        lexer.next();

                        param = proto.param(string);
                        if ( !param )
                        {
                            throw Error(position, "operation '%s' has no parameter called '%s'",
                                        proto.name().c_str(), string.c_str());
                        }

                        arg = parseValue(lexer, decls);
                        named = true;
                    }
                    else
                    {
                        param = &proto.param(args.size());
                        arg = makeIdentifier(string, position, decls);
                    }
                }
                else
                {
                    param = &proto.param(args.size());
                    arg = parseValue(lexer, decls);
                }

                if ( !castable(param->type(), arg, *decls) )
                {
                    throw Error(position, "argument cannot be cast to parameter type '%s'",
                                param->type()->toString().c_str());
                }

                expectNamed |= named || !param->type()->isTensor();
                if ( expectNamed && !named )
                {
                    throw Error(position, "expected named argument");
                }

                if ( args.contains(param->name()) )
                {
                    throw Error(position, "duplicate arguments: parameter '%s' already assigned (%u,%u)",
                                param->name().c_str());
                }

                args.emplace(param->name(), std::move(arg));
            }
            while ( readIfToken(',', lexer) );

            for ( size_t i = 0; i < proto.paramCount(); ++i )
            {
                auto& param = proto.param(i);

                if ( !args.contains(param.name()) )
                {
                    if ( param.defaultValue() )
                    {
                        args[param.name()] = param.defaultValue();
                    }
                    else
                    {
                        throw Error(lexer.position(), "missing argument for fragment '%s'; parameter '%s' not assigned",
                                    proto.name().c_str(), param.name().c_str());
                    }
                }
            }

            return args;
        }

        static void declare( const Value& arg, const Type* type, Types& declared, const Position& position )
        {
            switch ( arg.kind() )
            {
                case Value::Tensor:
                {
                    if ( !type->isPrimitive() )
                    {
                        throw Error(position, "cannot assign result of type '%s' to tensor identifier", type->toString().c_str());
                    }
                    const std::string& id = arg.tensor().id;
                    if ( declared.contains(id) )
                    {
                        throw Error(position, "identifier '%s' already declared", id.c_str());
                    }
                    declared.emplace(id, type);
                    break;
                }
                case Value::Array:
                {
                    if ( !type->isArray() )
                    {
                        throw Error(position, "cannot assign result of type '%s' to array", type->toString().c_str());
                    }
                    auto arrayType = dynamic_cast<const ArrayType*>(type);
                    for ( size_t i = 0; i < arg.size(); ++i )
                    {
                        declare(arg[i], arrayType->itemType(), declared, position);
                    }
                    break;
                }
                case Value::Tuple:
                {
                    if ( !type->isTuple() )
                    {
                        throw Error(position, "cannot assign result of type '%s' to tuple", type->toString().c_str());
                    }
                    auto tupleType = dynamic_cast<const TupleType*>(type);
                    for ( size_t i = 0; i < arg.size(); ++i )
                    {
                        declare(arg[i], tupleType->itemType(i), declared, position);
                    }
                    break;
                }
                default:
                {
                    throw Error(position, "literal expression not allowed in this context");
                }
            }
        }

    private:

        static Value parseValue( Lexer& lexer, const Types* decls )
        {
            switch ( lexer.token() )
            {
                case Lexer::True:
                case Lexer::False:
                {
                    return parseLogical(lexer);
                }
                case '-':
                case Lexer::Real:
                case Lexer::Integer:
                {
                    return parseNumber(lexer);
                }
                case Lexer::Characters:
                {
                    return parseString(lexer);
                }
                case Lexer::Identifier:
                {
                    return parseIdentifier(lexer, decls);
                }
                case '[':
                {
                    return parseArray(lexer, decls);
                }
                case '(':
                {
                    return parseTuple(lexer, decls);
                }
                default:
                {
                    throw Error(lexer.position(), "unexpected token '%s'", Lexer::tokenString(lexer.token()).c_str());
                }
            }
        }
        
        static Value parseNumber( Lexer& lexer )
        {
            bool negative = lexer.token() == '-';
            if ( negative )
            {
                lexer.next();
            }
            if ( lexer.token() == Lexer::Integer )
            {
                return parseInteger(lexer, negative);
            }
            else if ( lexer.token() == Lexer::Real )
            {
                return parseScalar(lexer, negative);
            }
            else
            {
                throw Error(lexer.position(), "expected number");
            }
        }

        static Value parseInteger( Lexer& lexer, bool negative )
        {
            auto value = getIntegerValue(lexer);
            lexer.next();
            return Value::integer(negative ? -value : value);
        }

        static Value parseScalar( Lexer& lexer, bool negative )
        {
            auto value = getScalarValue(lexer);
            lexer.next();
            return Value::scalar(negative ? -value : value);
        }

        static Value parseLogical( Lexer& lexer )
        {
            auto value = lexer.token() == Lexer::True;
            lexer.next();
            return Value::logical(value);
        }

        static Value parseString( Lexer& lexer )
        {
            auto value = lexer.string();
            lexer.next();
            return Value::string(value);
        }

        static Value parseIdentifier( Lexer& lexer, const Types* decls )
        {
            auto value = makeIdentifier(lexer.string(), lexer.position(), decls);
            lexer.next();
            return value;
        }

        static Value makeIdentifier( const std::string& name, const Position& position, const Types* decls )
        {
            if ( decls && !decls->contains(name) )
            {
                throw Error(position, "undeclared identifier '%s'", name.c_str());
            }
            return Value::tensor({ name });
        }

        static Value parseArray( Lexer& lexer, const Types* decls )
        {
            readToken('[', lexer);

            std::vector<Value> items;

            if ( lexer.token() != ']' )
            {
                do
                {
                    auto item = parseValue(lexer, decls);
                    items.push_back(std::move(item));
                }
                while ( readIfToken(',', lexer) );
            }

            readToken(']', lexer);

            return Value::array(std::move(items));
        }

        static Value parseTuple( Lexer& lexer, const Types* decls )
        {
            std::vector<Value> items;

            bool parenthesized = lexer.token() == '(';
            if ( parenthesized )
            {
                lexer.next();

                auto first = parseValue(lexer, decls);
                readToken(',', lexer);

                items.push_back(first);
            }

            do
            {
                auto item = parseValue(lexer, decls);
                items.push_back(std::move(item));
            }
            while ( readIfToken(',', lexer) );

            if ( parenthesized )
            {
                readToken(')', lexer);
            }

            return Value::tuple(std::move(items));
        }

    private:

        static bool castable( const Type* type, const Value& value, const Types& declared )
        {
            if ( type->isPrimitive() )
            {
                auto primitive = dynamic_cast<const PrimitiveType*>(type);
                switch ( value.kind() )
                {
                    case Value::Integer:
                    {
                        return primitive->name() == Typename::Extent;
                    }
                    case Value::Scalar:
                    {
                        return primitive->name() == Typename::Scalar;
                    }
                    case Value::Logical:
                    {
                        return primitive->name() == Typename::Logical;
                    }
                    case Value::String:
                    {
                        return primitive->name() == Typename::String;
                    }
                    case Value::Tensor:
                    {
                        return primitive == declared[value.tensor().id];
                    }
                    default:
                    {
                        return false;
                    }
                }
            }
            else if ( type->isArray() )
            {
                if ( value.kind() != Value::Array )
                {
                    return false;
                }
                auto& array = dynamic_cast<const ArrayType&>(*type);
                for ( size_t i = 0; i < value.size(); ++i )
                {
                    if ( !castable(array.itemType(), value[i], declared) )
                    {
                        return false;
                    }
                }
                return true;
            }
            else if ( type->isTuple() )
            {
                if ( value.kind() != Value::Tuple )
                {
                    return false;
                }
                auto& tuple = dynamic_cast<const TupleType&>(*type);
                for ( size_t i = 0; i < value.size(); ++i )
                {
                    if ( !castable(tuple.itemType(i), value[i], declared) )
                    {
                        return false;
                    }
                }
                return true;
            }
            return false;
        }

        static float getScalarValue( Lexer& lexer )
        {
            return (float)std::atof(lexer.string().c_str());
        }

        static int getIntegerValue( Lexer& lexer )
        {
            return std::atoi(lexer.string().c_str());
        }
        
        static void readToken( int token, Lexer& lexer )
        {
            if ( lexer.token() != token )
            {
                throw Error(lexer.position(), "expected token '%s', found '%s'",
                                 Lexer::tokenString(token).c_str(), Lexer::tokenString(lexer.token()).c_str());
            }
            lexer.next();
        }
        
        static bool readIfToken( int token, Lexer& lexer )
        {
            if ( lexer.token() == token )
            {
                lexer.next();
                return true;
            }
            return false;
        }
        
        static Prototypes stdlib()
        {
            auto stdlib = stdlibPrototypes();
            
            Prototypes prototypes;
            for ( auto& proto : stdlib )
            {
                prototypes.emplace(proto.name(), std::move(proto));
            }
            
            return prototypes;
        }
    };

}   // namespace nnef


#endif
