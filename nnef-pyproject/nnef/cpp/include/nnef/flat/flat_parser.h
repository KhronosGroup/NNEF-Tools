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

#include "../common/prototype.h"
#include "../common/dictionary.h"
#include "../common/typeutils.h"
#include "../common/parser.h"
#include "../common/value.h"
#include "../common/lexer.h"
#include "../common/error.h"
#include "stdlib_protos.h"
#include <cassert>


namespace nnef
{

    class FlatParser : public Parser
    {
    public:

        typedef Error::Position Position;

    public:

        virtual void parse( std::istream& is, const char* filename, Callback& callback )
        {
            Lexer lexer(is, filename);
            lexer.next();

            auto version = readVersion(lexer);

            callback.beginDocument(filename, version);

            auto extensions = readExtensions(lexer, [&]( const std::string& ext )
            {
                return callback.handleExtension(ext);
            });

            static Dictionary<Prototype> prototypes = buildPrototypes();

            parseGraph(lexer, prototypes, callback);

            callback.endDocument(filename);
        }

    private:
        
        void parseGraph( Lexer& lexer, const Dictionary<Prototype>& prototypes, Callback& callback )
        {
            lexer.readToken(Lexer::Graph);
            
            const std::string name = lexer.string();
            
            lexer.readToken(Lexer::Identifier);
            
            auto params = parseIdentifiers<Param>(lexer);
            
            lexer.readToken(Lexer::Arrow);
            
            auto results = parseIdentifiers<Result>(lexer);
            
            const Prototype graph(name, params, results);

            callback.beginGraph(graph, prototypes);
            
            lexer.readToken('{');
            
            Dictionary<Typename> dtypes;
            
            while ( lexer.token() != '}' )
            {
                parseAssignment(lexer, graph, prototypes, dtypes, callback);
            }
            
            checkGraphParamsAssigned(graph, dtypes, lexer.position());
            
            lexer.readToken('}');
            
            callback.endGraph(graph, dtypes);

            lexer.readToken(Lexer::Eof);
        }
        
        template<typename T>
        static std::vector<T> parseIdentifiers( Lexer& lexer )
        {
            std::vector<T> identifiers;
            
            lexer.readToken('(');
            
            do
            {
                const std::string id = lexer.string();
                
                lexer.readToken(Lexer::Identifier);
                
                identifiers.emplace_back(id, tensorType(Typename::Scalar));
            }
            while ( lexer.readIfToken(',') );
            
            lexer.readToken(')');
            
            return identifiers;
        }

        static void checkGraphParam( const Value& arg, const Prototype& graph, const std::string& target, const Position& position )
        {
            switch ( arg.kind() )
            {
                case Value::Identifier:
                {
                    if ( target == "external" )
                    {
                        if ( !graph.param(arg.identifier()) )
                        {
                            throw Error(position, "identifier '%s' assigned by operation 'external' must be a graph parameter",
                                             arg.identifier().c_str());
                        }
                    }
                    else
                    {
                        if ( graph.param(arg.identifier()) )
                        {
                            throw Error(position, "graph parameter '%s' can only be assigned by operation 'external'",
                                             arg.identifier().c_str());
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
        
        static void checkGraphParamsAssigned( const Prototype& graph, const Dictionary<Typename>& declared, const Position& position )
        {
            for ( size_t i = 0; i < graph.paramCount(); ++i )
            {
                auto& param = graph.param(i);
                if ( !declared.count(param.name()) )
                {
                    throw Error(position, "graph parameter '%s' not assigned", param.name().c_str());
                }
            }
            
            for ( size_t i = 0; i < graph.resultCount(); ++i )
            {
                auto& result = graph.result(i);
                if ( !declared.count(result.name()) )
                {
                    throw Error(position, "graph result '%s' not assigned", result.name().c_str());
                }
            }
        }

    private:

        void parseAssignment( Lexer& lexer, const Prototype& graph, const Dictionary<Prototype>& prototypes,
                             Dictionary<Typename>& dtypes, Callback& callback )
        {
            auto position = lexer.position();

            const Value results = parseTuple(lexer, nullptr, false, true);

            lexer.readToken('=');
            
            const std::string target = lexer.string();

            lexer.readToken(Lexer::Identifier);

            auto it = prototypes.find(target);
            if ( it == prototypes.end() )
            {
                throw Error(lexer.position(), "undefined operation '%s'", target.c_str());
            }
            
            auto& proto = it->second;
            
            checkGraphParam(results, graph, proto.name(), position);
            
            const PrimitiveType* dataType = proto.genericParamDefault();
            if ( lexer.readIfToken('<') )
            {
                if ( lexer.token() == '?' )
                {
                    throw Error(lexer.position(), "expected type name");
                }
                
                dataType = primitiveType(getTypename(lexer));
                lexer.next();
                
                lexer.readToken('>');
            }

            lexer.readToken('(');

            Dictionary<Value> args = parseArguments(proto, lexer, &dtypes, dataType, true, false, false);

            lexer.readToken(')');
            lexer.readToken(';');

            if ( results.size() != proto.resultCount() )
            {
                throw Error(position, "left-hand-side item count must match result count of operation (%d)",
                                 (int)proto.resultCount());
            }

            if ( proto.isGeneric() && !dataType && !deduceDataType(proto, args, dtypes, dataType, position) )
            {
                throw Error(position, "could not deduce generic data-type");
            }

            if ( dataType )
            {
                args["?"] = Value::string(dataType->toString());
            }
            
            for ( size_t i = 0; i < proto.resultCount(); ++i )
            {
                auto& result = proto.result(i);
                auto type = dataType ? bindDataType(result.type(), dataType) : result.type();

                declare(results[i], type, dtypes, position);

                args.emplace(result.name(), std::move(results[i]));
            }
            
            callback.operation(proto, args, dtypes);
        }

    protected:

        static Dictionary<Value> parseArguments( const Prototype& proto, Lexer& lexer, const Dictionary<Typename>* decls,
                                                const PrimitiveType* dataType, const bool allowIdentifier, const bool allowArrayToTensor,
                                                bool expectNamed, const Param* exclusion = nullptr )
        {
            Dictionary<Value> args;

            do
            {
                auto position = lexer.position();

                if ( args.size() >= proto.paramCount() )
                {
                    throw Error(position, "too many arguments; definition of '%s' has only %d parameters",
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

                        arg = parseValue(lexer, decls, true, allowIdentifier);
                        named = true;
                    }
                    else if ( allowIdentifier )
                    {
                        param = &proto.param(args.size());
                        arg = makeIdentifier(string, position, decls);
                    }
                    else
                    {
                        throw Error(position, "token 'identifier' not allowed in this context");
                    }
                }
                else
                {
                    param = &proto.param(args.size());
                    arg = parseValue(lexer, decls, true, allowIdentifier);
                }

                auto paramType = dataType ? bindDataType(param->type(), dataType) : param->type();
                auto argType = typeOf(arg, *decls);
                if ( !isCastable(argType, paramType, true, allowArrayToTensor) )
                {
                    throw Error(position, "argument of type '%s' cannot be cast to type '%s' for parameter '%s'",
                                argType->toString().c_str(), paramType->toString().c_str(), param->name().c_str());
                }

                expectNamed |= named || paramType->isAttribute();
                if ( expectNamed && !named )
                {
                    throw Error(position, "expected named argument");
                }

                if ( args.count(param->name()) )
                {
                    throw Error(position, "duplicate arguments: parameter '%s' already assigned",
                                param->name().c_str());
                }
                if ( param == exclusion )
                {
                    throw Error(lexer.position(), "argument '%s' of operation '%s' must not be provided in this context",
                                param->name().c_str(), proto.name().c_str());
                }
                if ( param->type()->kind() == Type::Tensor && isJaggedArray(arg) )
                {
                    throw Error(lexer.position(), "tensor literal argument for argument '%s' must not be jagged nested array",
                                param->name().c_str());
                }

                args.emplace(param->name(), std::move(arg));
            }
            while ( lexer.readIfToken(',') );

            for ( size_t i = 0; i < proto.paramCount(); ++i )
            {
                auto& param = proto.param(i);

                if ( &param != exclusion && !args.count(param.name()) )
                {
                    if ( param.defaultValue() )
                    {
                        if ( param.type()->isGeneric() )
                        {
                            auto valueType = typeOf(param.defaultValue(), *decls);
                            auto paramType = dataType ? bindDataType(param.type(), dataType) : param.type();
                            if ( !isCastable(valueType, paramType, true, allowArrayToTensor) )
                            {
                                throw Error(lexer.position(), "default value type '%s' cannot be cast to type '%s' for parameter '%s'",
                                            valueType->toString().c_str(), paramType->toString().c_str(), param.name().c_str());
                            }
                        }
                        args[param.name()] = param.defaultValue();
                    }
                    else
                    {
                        throw Error(lexer.position(), "missing argument for operation '%s'; parameter '%s' not assigned",
                                    proto.name().c_str(), param.name().c_str());
                    }
                }
            }

            return args;
        }
        
    private:
        
        static bool checkNestedArrayShape( const Value& value, const int* shape, const size_t rank )
        {
            if ( rank == 0 )
            {
                return value.kind() != Value::Array;
            }
            else if ( value.kind() != Value::Array || value.size() != (size_t)*shape )
            {
                return false;
            }
            for ( size_t i = 0; i < value.size(); ++i )
            {
                if ( !checkNestedArrayShape(value[i], shape + 1, rank - 1) )
                {
                    return false;
                }
            }
            return true;
        }
        
        static bool isJaggedArray( const Value& value )
        {
            auto shape = nestedArrayShape(value);
            return !checkNestedArrayShape(value, shape.data(), shape.size());
        }

    private:

        static void declare( const Value& arg, const Type* type, Dictionary<Typename>& dtypes, const Position& position )
        {
            switch ( arg.kind() )
            {
                case Value::Identifier:
                {
                    if ( type->kind() != Type::Tensor )
                    {
                        throw Error(position, "cannot assign result of type '%s' to tensor identifier", type->toString().c_str());
                    }
                    const std::string& id = arg.identifier();
                    if ( dtypes.count(id) )
                    {
                        throw Error(position, "identifier '%s' already declared", id.c_str());
                    }
                    auto dataType = static_cast<const TensorType*>(type)->dataType();
                    assert(dataType->kind() == Type::Primitive);
                    dtypes.emplace(id, static_cast<const PrimitiveType*>(dataType)->name());
                    break;
                }
                case Value::Array:
                {
                    if ( type->kind() != Type::Array )
                    {
                        throw Error(position, "cannot assign result of type '%s' to array", type->toString().c_str());
                    }
                    auto arrayType = static_cast<const ArrayType*>(type);
                    for ( size_t i = 0; i < arg.size(); ++i )
                    {
                        declare(arg[i], arrayType->itemType(), dtypes, position);
                    }
                    break;
                }
                case Value::Tuple:
                {
                    if ( type->kind() != Type::Tuple )
                    {
                        throw Error(position, "cannot assign result of type '%s' to tuple", type->toString().c_str());
                    }
                    auto tupleType = static_cast<const TupleType*>(type);
                    for ( size_t i = 0; i < arg.size(); ++i )
                    {
                        declare(arg[i], tupleType->itemType(i), dtypes, position);
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

        static Value parseValue( Lexer& lexer, const Dictionary<Typename>* decls, bool allowLiteral, bool allowIdentifier )
        {
            switch ( lexer.token() )
            {
                case Lexer::True:
                case Lexer::False:
                {
                    if ( allowLiteral )
                    {
                        return parseLogical(lexer);
                    }
                    break;
                }
                case '-':
                case Lexer::Decimal:
                case Lexer::Fractional:
                {
                    if ( allowLiteral )
                    {
                        return parseNumber(lexer);
                    }
                    break;
                }
                case Lexer::Characters:
                {
                    if ( allowLiteral )
                    {
                        return parseString(lexer);
                    }
                    break;
                }
                case '[':
                {
                    return parseArray(lexer, decls, allowLiteral, allowIdentifier);
                }
                case '(':
                {
                    return parseTuple(lexer, decls, allowLiteral, allowIdentifier);
                }
                case Lexer::Identifier:
                {
                    if ( allowIdentifier )
                    {
                        return parseIdentifier(lexer, decls);
                    }
                    break;
                }
                default:
                {
                    throw Error(lexer.position(), "unexpected token '%s'", Lexer::tokenString(lexer.token()).c_str());
                }
            }
            throw Error(lexer.position(), "token '%s' not allowed in this context", Lexer::tokenString(lexer.token()).c_str());
        }
        
        static Value parseNumber( Lexer& lexer )
        {
            bool negative = lexer.token() == '-';
            if ( negative )
            {
                lexer.next();
            }
            if ( lexer.token() == Lexer::Decimal )
            {
                return parseInteger(lexer, negative);
            }
            else if ( lexer.token() == Lexer::Fractional )
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

        static Value parseIdentifier( Lexer& lexer, const Dictionary<Typename>* decls )
        {
            auto value = makeIdentifier(lexer.string(), lexer.position(), decls);
            lexer.next();
            return value;
        }

        static Value makeIdentifier( const std::string& name, const Position& position, const Dictionary<Typename>* decls )
        {
            if ( decls && !decls->count(name) )
            {
                throw Error(position, "undeclared identifier '%s'", name.c_str());
            }
            return Value::identifier(name);
        }

        static Value parseArray( Lexer& lexer, const Dictionary<Typename>* decls, bool allowLiteral, bool allowIdentifier )
        {
            lexer.readToken('[');

            std::vector<Value> items;

            if ( lexer.token() != ']' )
            {
                do
                {
                    auto item = parseValue(lexer, decls, allowLiteral, allowIdentifier);
                    items.push_back(std::move(item));
                }
                while ( lexer.readIfToken(',') );
            }

            lexer.readToken(']');

            return Value::array(std::move(items));
        }

        static Value parseTuple( Lexer& lexer, const Dictionary<Typename>* decls, bool allowLiteral, bool allowIdentifier )
        {
            std::vector<Value> items;

            bool parenthesized = lexer.token() == '(';
            if ( parenthesized )
            {
                lexer.next();

                auto first = parseValue(lexer, decls, allowLiteral, allowIdentifier);
                lexer.readToken(',');

                items.push_back(first);
            }

            do
            {
                auto item = parseValue(lexer, decls, allowLiteral, allowIdentifier);
                items.push_back(std::move(item));
            }
            while ( lexer.readIfToken(',') );

            if ( parenthesized )
            {
                lexer.readToken(')');
            }

            return Value::tuple(std::move(items));
        }
        
    private:

        static const Type* typeOf( const Value& value, const Dictionary<Typename>& declared )
        {
            switch ( value.kind() )
            {
                case Value::Integer:
                {
                    return primitiveType(Typename::Integer);
                }
                case Value::Scalar:
                {
                    return primitiveType(Typename::Scalar);
                }
                case Value::Logical:
                {
                    return primitiveType(Typename::Logical);
                }
                case Value::String:
                {
                    return primitiveType(Typename::String);
                }
                case Value::Identifier:
                {
                    return tensorType(declared.at(value.identifier()));
                }
                case Value::Array:
                {
                    auto itemType = value.size() ? typeOf(value[0], declared) : nullptr;
                    return arrayType(itemType);
                }
                case Value::Tuple:
                {
                    std::vector<const Type*> itemTypes(value.size());
                    for ( size_t i = 0; i < value.size(); ++i )
                    {
                        itemTypes[i] = typeOf(value[i], declared);
                    }
                    return tupleType(itemTypes);
                }
                case Value::None:
                {
                    return nullptr;
                }
            }
            assert(false);
            return nullptr;
        }

        static bool deduceDataType( const Prototype& proto, const Dictionary<Value>& args, const Dictionary<Typename>& declared,
                                   const PrimitiveType*& dataType, const Position& position )
        {
            Dictionary<const Type*> types;
            for ( auto& arg : args )
            {
                types[arg.first] = typeOf(arg.second, declared);
            }
            for ( size_t i = 0; i < proto.paramCount(); ++i )
            {
                auto& param = proto.param(i);
                if ( !types.count(param.name()) )
                {
                    assert(param.defaultValue());
                    types[param.name()] = typeOf(param.defaultValue(), declared);
                }
            }

            try
            {
                return nnef::deduceDataType(proto, types, dataType);
            }
            catch ( std::pair<Typename,Typename> e )
            {
                throw Error(position, "could not deduce data-type: ambiguous candidates '%s' vs '%s'", toString(e.first), toString(e.second));
            }
        }
        
        static Dictionary<Prototype> buildPrototypes()
        {
            static auto stdlibPrototypes = nnef::stdlibPrototypes();
            
            Dictionary<Prototype> prototypes;
            for ( auto& proto : stdlibPrototypes )
            {
                prototypes.emplace(proto.name(), std::move(proto));
            }
            return prototypes;
        }
    };

}   // namespace nnef


#endif
