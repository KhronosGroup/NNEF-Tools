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

#ifndef _NNEF_COMP_PARSER_H_
#define _NNEF_COMP_PARSER_H_

#include "../common/dictionary.h"
#include "../common/typespec.h"
#include "../common/prototype.h"
#include "../common/parser.h"
#include "../common/value.h"
#include "../common/lexer.h"
#include "../common/error.h"
#include "../common/shape.h"
#include "stdlib_source.h"
#include "expression.h"
#include "evaluation.h"
#include "fragment.h"
#include <cassert>
#include <sstream>
#include <cctype>


namespace nnef
{
    
    class CompParser : public Parser
    {
    public:
        
        typedef Error::Position Position;

    private:

        typedef Dictionary<Shared<Fragment>> Fragments;
        typedef Dictionary<const Type*> Declarations;
        
    public:

        CompParser( bool includeLayers = false )
        : _includeLayers(includeLayers)
        {
        }
        
        virtual void parse( std::istream& is, Callback& callback )
        {
            Lexer lexer(is);
            lexer.next();

            static const Fragments stdlibFragments = parseStdlib();

            Fragments fragments = stdlibFragments;
            if ( _includeLayers )
            {
                parseFragments(stdlib_layers<void>::text, fragments);
            }

            parseVersion(lexer);

            while ( lexer.token() == Lexer::Fragment )
            {
                parseFragment(lexer, fragments, false, true);
            }

            auto graph = parseFragment(lexer, fragments, true, false);

            callback.beginGraph(graph->prototype());

            Dictionary<Value> values;
            Dictionary<Shape> shapes;

            Evaluation evaluation(*graph, fragments);
            for ( size_t i = 0; i < graph->assignmentCount(); ++i )
            {
                auto& assignment = graph->assignment(i);

                if ( assignment.rhs().kind() == Expr::Invocation )
                {
                    auto& target = dynamic_cast<const InvocationExpr&>(assignment.rhs()).target();
                    auto& proto = fragments[target]->prototype();
                    checkGraphParam(assignment.lhs(), graph->prototype(), proto.name());
                }

                const Value context = evaluation.evaluateLvalue(assignment.lhs(), Dictionary<Value>(), true);
                evaluation.evaluateAssign(assignment.lhs(), assignment.rhs(), values, shapes, callback, false, context);
            }

            callback.endGraph(graph->prototype(), shapes);
        }

    private:

        static Fragments parseStdlib()
        {
            Fragments fragments;
            parseFragments(stdlib_source<void>::text, fragments);
            return fragments;
        }

        static void parseFragments( const char* text, Fragments& fragments )
        {
            std::stringstream ss(text);
            Lexer lexer(ss);
            lexer.next();

            while ( lexer.token() == Lexer::Fragment )
            {
                parseFragment(lexer, fragments, false, true);
            }
        }

        static Shared<Prototype> parsePrototype( Lexer& lexer, bool allowTypespec )
        {
            const std::string name = lexer.string();
            readToken(Lexer::Identifier, lexer);

            std::vector<Param> params = parseParams(lexer, allowTypespec);

            readToken(Lexer::Arrow, lexer);

            std::vector<Result> results = parseResults(lexer, allowTypespec);

            return std::make_shared<Prototype>(name, params, results);
        }

        static std::vector<Param> parseParams( Lexer& lexer, bool allowTypespec )
        {
            std::vector<Param> params;

            readToken('(', lexer);

            const Declarations emptyTypes = Declarations();

            do
            {
                auto name = lexer.string();
                readToken(Lexer::Identifier, lexer);

                const Type* type = tensorType(Typename::Scalar);
                if ( allowTypespec )
                {
                    readToken(':', lexer);
                    type = parseTypespec(lexer);
                }

                auto defaultValue = Value::none();
                if ( lexer.token() == '=' )
                {
                    lexer.next();

                    auto expr = fixEmptyArray(parseExpression(lexer, nullptr, &emptyTypes), type);

                    if ( !castable(expr->type(), type) )
                    {
                        throw Error(expr->position(), "default value type '%s' cannot be cast to parameter type '%s'",
                                    expr->type()->toString().c_str(), type->toString().c_str());
                    }

                    defaultValue = Evaluation::evaluateRvalue(*expr);
                }

                params.emplace_back(name, type, defaultValue);
            }
            while ( readIfToken(',', lexer) );

            readToken(')', lexer);

            return params;
        }

        static std::vector<Result> parseResults( Lexer& lexer, bool allowTypespec )
        {
            std::vector<Result> results;

            readToken('(', lexer);

            do
            {
                auto name = lexer.string();
                readToken(Lexer::Identifier, lexer);

                const Type* type = tensorType(Typename::Scalar);
                if ( allowTypespec )
                {
                    readToken(':', lexer);
                    type = parseTypespec(lexer);
                }

                results.emplace_back(name, type);
            }
            while ( readIfToken(',', lexer) );

            readToken(')', lexer);

            return results;
        }

        static Shared<Fragment> parseFragment( Lexer& lexer, Fragments& fragments, bool graph, bool allowPrimitive )
        {
            readToken(graph ? Lexer::Graph : Lexer::Fragment, lexer);

            auto position = lexer.position();

            auto proto = parsePrototype(lexer, !graph);
            if ( fragments.contains(proto->name()) )
            {
                throw Error(position, "operation '%s' already defined", proto->name().c_str());
            }

            checkPrototype(*proto, position);

            auto primitive = std::make_shared<Fragment>(proto);
            if ( !graph )
            {
                fragments[proto->name()] = primitive;
            }
            
            if ( allowPrimitive && lexer.token() != '{' )
            {
                return primitive;
            }

            Declarations decls;
            if ( !graph )
            {
                for ( size_t i = 0; i < proto->paramCount(); ++i )
                {
                    auto& param = proto->param(i);
                    decls[param.name()] = param.type();
                }
            }

            std::vector<Assignment> assignments;

            readToken('{', lexer);

            do
            {
                auto lhs = parseTuple(lexer, nullptr, nullptr);

                readToken('=', lexer);

                auto rhs = parseExpression(lexer, &fragments, &decls);

                readIfToken(';', lexer);

                declare(*lhs, rhs->type(), decls);

                assignments.emplace_back(lhs, rhs);
            }
            while ( lexer.token() != '}' );

            readToken('}', lexer);

            if ( graph )
            {
                for ( size_t i = 0; i < proto->paramCount(); ++i )
                {
                    auto& param = proto->param(i);
                    if ( !decls.contains(param.name()) )
                    {
                        throw Error(lexer.position(), "graph parameter '%s' is not assigned",
                                    param.name().c_str());
                    }
                }
            }

            for ( size_t i = 0; i < proto->resultCount(); ++i )
            {
                auto& result = proto->result(i);
                auto decl = decls[result.name()];
                if ( !decl )
                {
                    throw Error(lexer.position(), "result '%s' of operation '%s' is not assigned",
                                result.name().c_str(), proto->name().c_str());
                }
                else if ( !castable(decl, result.type()) )
                {
                    throw Error(position, "result '%s' of operation '%s' is declared as '%s' but assignment has incompatible type '%s'",
                                result.name().c_str(), proto->name().c_str(), result.type()->toString().c_str(), decl->toString().c_str());
                }
            }

            auto fragment = std::make_shared<Fragment>(proto, assignments);
            if ( !graph )
            {
                fragments[proto->name()] = fragment;
            }
            return fragment;
        }

        static bool checkGraphParam( const Expr& expr, const Prototype& graph, const std::string& target )
        {
            switch ( expr.kind() )
            {
                case Expr::Identifier:
                {
                    auto& identifier = dynamic_cast<const IdentifierExpr&>(expr);

                    if ( target == "external" )
                    {
                        if ( !graph.param(identifier.name()) )
                        {
                            throw Error(identifier.position(), "identifiers assigned by operation 'external' must be graph parameters");
                        }
                    }
                    else
                    {
                        if ( graph.param(identifier.name()) )
                        {
                            throw Error(identifier.position(), "graph parameter '%s' can only be assigned by operation 'external'",
                                  identifier.name().c_str());
                        }
                    }
                    return true;
                }
                case Expr::Array:
                case Expr::Tuple:
                {
                    bool valid = true;
                    auto& items = dynamic_cast<const ItemExpr&>(expr);
                    for ( size_t i = 0; i < items.size(); ++i )
                    {
                        valid &= checkGraphParam(items.item(i), graph, target);
                    }
                    return valid;
                }
                default:
                {
                    assert(false);
                    return false;
                }
            }
        }

    private:

        static const Type* parseArrayTypespec( Lexer& lexer, const Type* type )
        {
            while ( lexer.token() == '[' )
            {
                lexer.next();

                readToken(']', lexer);

                type = arrayType(type);
            }

            return type;
        }

        static const Type* parseTupleTypespec( Lexer& lexer )
        {
            lexer.next();

            std::vector<const Type*> items;
            do
            {
                items.push_back(parseTypespec(lexer));
            }
            while ( readIfToken(',', lexer) );

            readToken(')', lexer);

            return parseArrayTypespec(lexer, tupleType(items));
        }

        static const Type* parseTypespec( Lexer& lexer )
        {
            if ( lexer.token() == '(' )
            {
                return parseTupleTypespec(lexer);
            }

            const Type* type = nullptr;
            if ( lexer.token() == Lexer::Tensor )
            {
                lexer.next();

                Typename dtype = Typename::Scalar;
                if ( lexer.token() == '<' )
                {
                    lexer.next();

                    dtype = getTypename(lexer);
                    lexer.next();

                    readToken('>', lexer);
                }

                type = tensorType(dtype);
            }
            else
            {
                const Typename name = getTypename(lexer);
                lexer.next();

                type = primitiveType(name);
            }

            return parseArrayTypespec(lexer, type);
        }
        
    private:
        
        static Shared<Expr> parseExpression( Lexer& lexer, const Fragments* fragments, const Declarations* decls, bool enableIf = true )
        {
            auto expr = parsePrimary(lexer, fragments, decls);
            if ( expr->kind() != Expr::Literal )
            {
                expr = parseSubscripts(lexer, fragments, decls, expr);
            }
            expr = parseBinary(lexer, fragments, decls, expr);
            if ( lexer.token() == Lexer::If && enableIf )
            {
                expr = parseSelect(lexer, fragments, decls, expr);
            }
            return expr;
        }
        
        static Shared<Expr> parsePrimary( Lexer& lexer, const Fragments* fragments, const Declarations* decls )
        {
            switch ( lexer.token() )
            {
                case Lexer::True:
                case Lexer::False:
                {
                    return parseLogical(lexer);
                }
                case Lexer::Real:
                {
                    return parseScalar(lexer);
                }
                case Lexer::Integer:
                {
                    return parseExtent(lexer);
                }
                case Lexer::Characters:
                {
                    return parseString(lexer);
                }
                case Lexer::Identifier:
                {
                    return parseIdentifier(lexer, fragments, decls);
                }
                case '[':
                {
                    return parseArray(lexer, fragments, decls);
                }
                case '(':
                {
                    return parseTuple(lexer, fragments, decls);
                }
                case '-':
                case '!':
                {
                    return parseUnary(lexer, fragments, decls);
                }
                case Lexer::LengthOf:
                case Lexer::ShapeOf:
                case Lexer::RangeOf:
                case Lexer::Extent:
                case Lexer::Scalar:
                case Lexer::Logical:
                case Lexer::String:
                {
                    return parseBuiltin(lexer, fragments, decls);
                }
                default:
                {
                    throw Error(lexer.position(), "unexpected token '%s'", Lexer::tokenString(lexer.token()).c_str());
                }
            }
        }
        
        static Shared<Expr> parseExtent( Lexer& lexer )
        {
            auto position = lexer.position();
            
            auto value = getIntegerValue(lexer);
            lexer.next();
            
            return std::make_shared<ExtentExpr>(position, value, primitiveType(Typename::Extent));
        }
        
        static Shared<Expr> parseScalar( Lexer& lexer )
        {
            auto position = lexer.position();
            
            auto value = getScalarValue(lexer);
            lexer.next();
            
            return std::make_shared<ScalarExpr>(position, value, primitiveType(Typename::Scalar));
        }
        
        static Shared<Expr> parseLogical( Lexer& lexer )
        {
            auto position = lexer.position();
            
            auto value = lexer.token() == Lexer::True;
            lexer.next();
            
            return std::make_shared<LogicalExpr>(position, value, primitiveType(Typename::Logical));
        }
        
        static Shared<Expr> parseString( Lexer& lexer )
        {
            auto position = lexer.position();
            
            auto value = lexer.string();
            lexer.next();
            
            return std::make_shared<StringExpr>(position, value, primitiveType(Typename::String));
        }
        
        static Shared<Expr> parseIdentifier( Lexer& lexer, const Fragments* fragments, const Declarations* decls )
        {
            auto position = lexer.position();
            auto string = lexer.string();
            lexer.next();
            
            if ( lexer.token() == '(' )
            {
                if ( !fragments )
                {
                    throw Error(lexer.position(), "identifier not allowed in this context");
                }
                return parseInvocation(lexer, fragments, decls, position, string);
            }
            else
            {
                return makeIdentifier(position, string, decls);
            }
        }
        
        static Shared<Expr> makeIdentifier( const Position& position, const std::string& name, const Declarations* decls )
        {
            const Type* type = nullptr;
            if ( decls )
            {
                type = (*decls)[name];
                if ( !type )
                {
                    throw Error(position, "undeclared identifier '%s'", name.c_str());
                }
            }
            return std::make_shared<IdentifierExpr>(position, name, type);
        }
        
        static Shared<Expr> parseArray( Lexer& lexer, const Fragments* fragments, const Declarations* decls )
        {
            auto position = lexer.position();
            lexer.next();
            
            std::vector<Shared<Expr>> items;

            const Type* type = nullptr;
            
            if ( lexer.token() != ']' )
            {
                auto first = parseExpression(lexer, fragments, decls);
                if ( lexer.token() == Lexer::For )
                {
                    return parseComprehension(lexer, fragments, decls, position, first);
                }

                items = { first };
                type = first->type();

                while ( readIfToken(',', lexer) )
                {
                    auto item = parseExpression(lexer, fragments, decls);
                    items.push_back(item);

                    if ( type )
                    {
                        type = commonType(type, item->type());
                        if ( !type )
                        {
                            throw Error(position, "incompatible item types (%s vs %s) in array",
                                        first->type()->toString().c_str(), item->type()->toString().c_str());
                        }
                    }
                }
            }
            
            readToken(']', lexer);
            
            return std::make_shared<ArrayExpr>(position, items, arrayType(type));
        }
        
        static Shared<Expr> parseTuple( Lexer& lexer, const Fragments* fragments, const Declarations* decls )
        {
            auto position = lexer.position();
            
            bool parenthesized = lexer.token() == '(';
            if ( parenthesized )
            {
                lexer.next();
            }

            std::vector<Shared<Expr>> items;
            std::vector<const Type*> types;

            auto first = parseExpression(lexer, fragments, decls);

            if ( lexer.token() == ',' )
            {
                items = { first };
                types = { first->type() };

                while ( readIfToken(',', lexer) )
                {
                    auto item = parseExpression(lexer, fragments, decls);
                    items.push_back(item);
                    types.push_back(item->type());
                }
            }
            
            if ( parenthesized )
            {
                readToken(')', lexer);
            }

            return items.empty() ? first : std::make_shared<TupleExpr>(position, items, tupleType(types));
        }
        
        static Shared<Expr> parseInvocation( Lexer& lexer, const Fragments* fragments, const Declarations* decls,
                                            const Position& position, const std::string& target )
        {
            readToken('(', lexer);
            
            auto& fragment = (*fragments)[target];
            if ( !fragment )
            {
                throw Error(position, "undefined operation '%s'", target.c_str());
            }

            const Prototype& proto = fragment->prototype();

            Dictionary<Shared<Expr>> args;
            
            bool expectNamed = false;
            
            do
            {
                auto position = lexer.position();
                
                if ( args.size() >= proto.paramCount() )
                {
                    throw Error(position, "too many positional arguments; definition of '%s' has only %d parameters",
                                proto.name().c_str(), (int)proto.paramCount());
                }
                
                const Param* param = nullptr;
                Shared<Expr> arg;
                
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
                        
                        arg = parseExpression(lexer, fragments, decls);
                        named = true;
                    }
                    else
                    {
                        param = &proto.param(args.size());
                        if ( lexer.token() == '(' )
                        {
                            arg = parseInvocation(lexer, fragments, decls, position, string);
                        }
                        else
                        {
                            arg = makeIdentifier(position, string, decls);
                        }
                        arg = parseSubscripts(lexer, fragments, decls, arg);
                        arg = parseBinary(lexer, fragments, decls, arg);
                        if ( lexer.token() == Lexer::If )
                        {
                            arg = parseSelect(lexer, fragments, decls, arg);
                        }
                    }
                }
                else
                {
                    param = &proto.param(args.size());
                    arg = parseExpression(lexer, fragments, decls);
                }

                arg = fixEmptyArray(arg, param->type());

                if ( !castable(arg->type(), param->type()) )
                {
                    throw Error(position, "argument of type '%s' cannot be cast to parameter type '%s'",
                                arg->type()->toString().c_str(), param->type()->toString().c_str());
                }
                
                expectNamed |= named || !param->type()->isTensor();
                if ( expectNamed && !named )
                {
                    throw Error(position, "expected named argument");
                }

                auto contained = args[param->name()];
                if ( contained )
                {
                    auto& pos = contained->position();
                    throw Error(position, "duplicate arguments: parameter '%s' already assigned (%u,%u)",
                                param->name().c_str(), pos.line, pos.column);
                }

                args[param->name()] = arg;
            }
            while ( readIfToken(',', lexer) );
            
            for ( size_t i = 0; i < proto.paramCount(); ++i )
            {
                auto& param = proto.param(i);

                if ( !args.contains(param.name()) && !param.defaultValue() )
                {
                    throw Error(lexer.position(), "missing argument for fragment '%s'; parameter '%s' not assigned",
                                    proto.name().c_str(), param.name().c_str());
                }
            }
            
            readToken(')', lexer);

            const Type* type = resultType(proto);

            if ( target == "select" )
            {
                return makeSelectExpr(position, args, type);
            }
            return std::make_shared<InvocationExpr>(position, target, args, type);
        }
        
        static Shared<Expr> parseUnary( Lexer& lexer, const Fragments* fragments, const Declarations* decls )
        {
            auto position = lexer.position();
            int op = lexer.token();
            lexer.next();
            
            auto rhs = parseExpression(lexer, fragments, decls);
            
            auto type = unaryResultType(rhs->type(), op);
            if ( !type )
            {
                throw Error(position, "invalid operand type '%s' for operation '%s'",
                            rhs->type()->toString().c_str(), Lexer::tokenString(op).c_str());
            }
            
            if ( type->isPrimitive() && type->isTensor() )
            {
                auto target = unaryOpName(op);
                auto args = makeUnaryOpArgs(rhs);
                return std::make_shared<InvocationExpr>(position, target, args, type);
            }
            else
            {
                return std::make_shared<UnaryExpr>(position, rhs, op, type);
            }
        }
        
        static Shared<Expr> parseBinary( Lexer& lexer, const Fragments* fragments, const Declarations* decls, Shared<Expr> lhs, int exprPrec = 0 )
        {
            auto position = lhs->position();

            while (true)
            {
                int tokPrec = tokenPrecedence(lexer.token());
                if ( tokPrec < exprPrec )
                {
                    return lhs;
                }
                
                int op = lexer.token();
                lexer.next();
                
                auto rhs = parsePrimary(lexer, fragments, decls);
                rhs = parseSubscripts(lexer, fragments, decls, rhs);
                
                int nextPrec = tokenPrecedence(lexer.token());
                if ( tokPrec < nextPrec )
                {
                    rhs = parseBinary(lexer, fragments, decls, rhs, tokPrec + 1);
                }
                
                auto type = binaryResultType(lhs->type(), rhs->type(), op);
                if ( !type )
                {
                    throw Error(position, "invalid operand types '%s' and '%s' for operation '%s'",
                                lhs->type()->toString().c_str(),  rhs->type()->toString().c_str(),
                                Lexer::tokenString(op).c_str());
                }
                
                if ( type->isPrimitive() && type->isTensor() )
                {
                    auto target = binaryOpName(op);
                    auto args = makeBinaryOpArgs(lhs, rhs);
                    lhs = std::make_shared<InvocationExpr>(position, target, args, type);
                }
                else
                {
                    lhs = std::make_shared<BinaryExpr>(position, lhs, rhs, op, type);
                }
            }
        }
        
        static Shared<Expr> parseBuiltin( Lexer& lexer, const Fragments* fragments, const Declarations* decls )
        {
            auto position = lexer.position();
            int op = lexer.token();
            lexer.next();
            
            readToken('(', lexer);
            
            auto arg = parseExpression(lexer, fragments, decls);
            
            auto type = builtinResultType(op);
            if ( !type )
            {
                throw Error(position, "invalid operand type '%s' for operation '%s'",
                            arg->type()->toString().c_str(), Lexer::tokenString(op).c_str());
            }
            
            readToken(')', lexer);

            if ( op == Lexer::LengthOf )
            {
                if ( !arg->type()->isArray() && arg->type() != primitiveType(Typename::String) )
                {
                    throw Error(position, "argument of length_of() must be an array or string (found %s)", arg->type()->toString().c_str());
                }
            }
            if ( op == Lexer::ShapeOf )
            {
                if ( !arg->type()->isPrimitive() )
                {
                    throw Error(position, "argument of shape_of() must be of primitive type (found %s)",
                                arg->type()->toString().c_str());
                }
            }
            else if ( op == Lexer::RangeOf && arg->type() != primitiveType(Typename::String) )
            {
                if ( !arg->type()->isArray() )
                {
                    throw Error(position, "argument of range_of() must be an array or string (found %s)",
                                arg->type()->toString().c_str());
                }
            }
            else if ( op == Lexer::Extent || op == Lexer::Scalar || op == Lexer::Logical || op == Lexer::String )
            {
                if ( !arg->type()->isPrimitive() || arg->type()->isTensor() )
                {
                    throw Error(position, "argument of %s() must be of primitive type except tensor (found %s)",
                                Lexer::tokenString(op).c_str(), arg->type()->toString().c_str());
                }
            }
            
            return std::make_shared<BuiltinExpr>(position, arg, op, type);
        }

        static Shared<Expr> parseSubscript( Lexer& lexer, const Fragments* fragments, const Declarations* decls, const Shared<Expr> sequence )
        {
            lexer.next();

            Shared<Expr> beg, end;
            const Type* type = nullptr;

            if ( sequence->type()->isTuple() )
            {
                beg = parseExpression(lexer, fragments, decls);
                if ( beg->kind() != Expr::Literal || beg->type() != primitiveType(Typename::Extent) )
                {
                    throw Error(beg->position(), "tuple index must be an extent literal");
                }

                auto idx = dynamic_cast<const ExtentExpr&>(*beg).value();

                type = dynamic_cast<const TupleType*>(sequence->type())->itemType(idx);
            }
            else if ( sequence->type()->isArray() || sequence->type() == primitiveType(Typename::String) )
            {
                if ( lexer.token() != ':' )
                {
                    beg = parseExpression(lexer, fragments, decls);
                    if ( beg->type() != primitiveType(Typename::Extent) )
                    {
                        throw Error(beg->position(), "array index must be of type extent, found '%s'", beg->type()->toString().c_str());
                    }
                }
                bool range = false;
                if ( lexer.token() == ':' )
                {
                    lexer.next();
                    range = true;

                    if ( lexer.token() != ']' )
                    {
                        end = parseExpression(lexer, fragments, decls);
                        if ( end->type() != primitiveType(Typename::Extent) )
                        {
                            throw Error(end->position(), "array index must be of type extent, found '%s'", end->type()->toString().c_str());
                        }
                    }
                }
                else
                {
                    end = beg;
                }

                readToken(']', lexer);

                if ( sequence->type()->isArray() )
                {
                    auto arrayType = dynamic_cast<const ArrayType*>(sequence->type());
                    type = range ? arrayType : arrayType->itemType();
                }
                else
                {
                    type = primitiveType(Typename::String);
                }
            }
            else
            {
                throw Error(sequence->position(), "subscripted expression must be of type array, tuple, or string; found '%s'",
                            sequence->type()->toString().c_str());
            }

            return std::make_shared<SubscriptExpr>(sequence->position(), sequence, beg, end, type);
        }

        static Shared<Expr> parseSubscripts( Lexer& lexer, const Fragments* fragments, const Declarations* decls, Shared<Expr> sequence )
        {
            while ( lexer.token() == '[' )
            {
                sequence = parseSubscript(lexer, fragments, decls, sequence);
            }
            return sequence;
        }

        static Shared<Expr> parseSelect( Lexer& lexer, const Fragments* fragments, const Declarations* decls, Shared<Expr> trueValue )
        {
            readToken(Lexer::If, lexer);

            auto condition = parseExpression(lexer, fragments, decls);
            if ( condition->type() != primitiveType(Typename::Logical) && condition->type() != tensorType(Typename::Logical) )
            {
                throw Error(condition->position(), "condition must be a logical value or a logical tensor");
            }

            readToken(Lexer::Else, lexer);

            auto falseValue = parseExpression(lexer, fragments, decls);

            if ( isEmptyArray(trueValue) && isEmptyArray(falseValue) )
            {
                return trueValue;
            }
            else if ( isEmptyArray(trueValue) )
            {
                trueValue = fixEmptyArray(trueValue, falseValue->type());
            }
            else if ( isEmptyArray(falseValue) )
            {
                falseValue = fixEmptyArray(falseValue, trueValue->type());
            }

            const Type* type = commonType(trueValue->type(), falseValue->type());
            if ( !type )
            {
                throw Error(trueValue->position(), "incompatible types in if-else expression (%s vs %s)",
                            trueValue->type()->toString().c_str(), falseValue->type()->toString().c_str());
            }

            if ( condition->type() == tensorType(Typename::Logical) )
            {
                auto args = makeSelectOpArgs(condition, trueValue, falseValue);
                return std::make_shared<InvocationExpr>(trueValue->position(), "select", args, type);
            }
            else
            {
                return std::make_shared<SelectExpr>(trueValue->position(), condition, trueValue, falseValue, type);
            }
        }

        static Shared<Expr> parseComprehension( Lexer& lexer, const Fragments* fragments, const Declarations* decls,
                                               const Position& position, const Shared<Expr> item )
        {
            readToken(Lexer::For, lexer);

            auto iterator = std::make_shared<IdentifierExpr>(lexer.position(), lexer.string(), nullptr);
            if ( decls && decls->find(iterator->name()) != decls->end() )
            {
                throw Error(iterator->position(), "iterator '%s' hides declared identifier", iterator->name().c_str());
            }

            readToken(Lexer::Identifier, lexer);
            readToken(Lexer::In, lexer);

            auto iterable = parseExpression(lexer, fragments, decls, false);
            if ( !iterable->type()->isArray() )
            {
                throw Error(iterable->position(), "expression not iterable");
            }

            Shared<Expr> condition = nullptr;
            if ( lexer.token() == Lexer::If )
            {
                lexer.next();

                condition = parseExpression(lexer, fragments, decls);
                if ( condition->type() != primitiveType(Typename::Logical) )
                {
                    throw Error(condition->position(), "condition in comprehension expression must be a logical expression");
                }
            }

            const Type* type = arrayType(item->type());

            return std::make_shared<ComprehensionExpr>(position, item, iterator, iterable, condition, type);
        }
        
    private:

        static bool isEmptyArray( const Shared<Expr>& expr )
        {
            return expr->kind() == Expr::Array && dynamic_cast<const ArrayExpr&>(*expr).size() == 0;
        }

        static Shared<Expr> fixEmptyArray( const Shared<Expr>& expr, const Type* type )
        {
            if ( isEmptyArray(expr) )
            {
                return std::make_shared<ArrayExpr>(expr->position(), type);
            }
            return expr;
        }
        
        static float getScalarValue( Lexer& lexer )
        {
            return (float)std::atof(lexer.string().c_str());
        }
        
        static int getIntegerValue( Lexer& lexer )
        {
            return std::atoi(lexer.string().c_str());
        }

        static Typename getTypename( Lexer& lexer )
        {
            switch ( lexer.token() )
            {
                case Lexer::Extent:
                    return Typename::Extent;
                case Lexer::Scalar:
                    return Typename::Scalar;
                case Lexer::Logical:
                    return Typename::Logical;
                case Lexer::String:
                    return Typename::String;
                default:
                    throw Error(lexer.position(), "expected type name");
            }
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

        static void declare( const Expr& expr, const Type* type, Declarations& declared )
        {
            switch ( expr.kind() )
            {
                case Expr::Identifier:
                {
                    auto& identifier = dynamic_cast<const IdentifierExpr&>(expr);
                    if ( declared.contains(identifier.name()) )
                    {
                        throw Error(expr.position(), "identifier '%s' is already declared", identifier.name().c_str());
                    }
                    declared.emplace(identifier.name(), type);
                    break;
                }
                case Expr::Array:
                {
                    if ( !type->isArray() )
                    {
                        throw Error(expr.position(), "cannot assign result of type '%s' to array", type->toString().c_str());
                    }
                    auto& array = dynamic_cast<const ArrayExpr&>(expr);
                    auto arrayType = dynamic_cast<const ArrayType*>(type);
                    for ( size_t i = 0; i < array.size(); ++i )
                    {
                        declare(array.item(i), arrayType->itemType(), declared);
                    }
                    break;
                }
                case Expr::Tuple:
                {
                    if ( !type->isTuple() )
                    {
                        throw Error(expr.position(), "cannot assign result of type '%s' to tuple", type->toString().c_str());
                    }
                    auto& tuple = dynamic_cast<const TupleExpr&>(expr);
                    auto tupleType = dynamic_cast<const TupleType*>(type);
                    if ( tupleType->size() != tuple.size() )
                    {
                        throw Error(expr.position(), "cannot assign result of type '%s' to a tuple of size %d",
                                    type->toString().c_str(), (int)tuple.size());
                    }
                    for ( size_t i = 0; i < tuple.size(); ++i )
                    {
                        declare(tuple.item(i), tupleType->itemType(i), declared);
                    }
                    break;
                }
                default:
                {
                    throw Error(expr.position(), "expression not allowed in this context");
                }
            }
        }

        static void checkPrototype( const Prototype& proto, const Position& position )
        {
            for ( size_t i = 0; i < proto.paramCount(); ++i )
            {
                auto& param = proto.param(i);

                if ( proto.param(param.name()) != &param )
                {
                    throw Error(position, "duplicate parameter definition for fragment '%s'; parameter '%s' is already defined",
                                proto.name().c_str(), param.name().c_str());
                }
            }

            for ( size_t i = 0; i < proto.resultCount(); ++i )
            {
                auto& result = proto.result(i);

                if ( proto.result(result.name()) != &result )
                {
                    throw Error(position, "duplicate result definition for operation '%s'; result '%s' is already defined",
                                proto.name().c_str(), result.name().c_str());
                }
                if ( proto.param(result.name()) )
                {
                    throw Error(position, "invalid result definition for operation '%s'; '%s' is already defined as parameter",
                                proto.name().c_str(), result.name().c_str());
                }
            }
        }

    private:
        
        static const Type* primitiveType( const Typename name )
        {
            static const PrimitiveType types[] =
            {
                PrimitiveType(Typename::Extent, false),
                PrimitiveType(Typename::Scalar, false),
                PrimitiveType(Typename::Logical, false),
                PrimitiveType(Typename::String, false),
            };
            return &types[(size_t)name];
        }

        static const Type* tensorType( const Typename name )
        {
            static const PrimitiveType types[] =
            {
                PrimitiveType(Typename::Extent, true),
                PrimitiveType(Typename::Scalar, true),
                PrimitiveType(Typename::Logical, true),
                PrimitiveType(Typename::String, true),
            };
            return &types[(size_t)name];
        }
        
        static const Type* arrayType( const Type* itemType )
        {
            static std::map<const Type*,const Type*> types;
            
            auto& type = types[itemType];
            if ( !type )
            {
                type = new ArrayType(itemType);
            }
            return type;
        }
        
        static const Type* tupleType( const std::vector<const Type*>& itemTypes )
        {
            static std::map<std::vector<const Type*>,const Type*> types;
            
            auto& type = types[itemTypes];
            if ( !type )
            {
                type = new TupleType(itemTypes);
            }
            return type;
        }

        static bool castable( const Type* type1, const Type* type2 )
        {
            if ( type1->isPrimitive() && type2->isPrimitive() )
            {
                auto prim1 = dynamic_cast<const PrimitiveType*>(type1);
                auto prim2 = dynamic_cast<const PrimitiveType*>(type2);
                if ( !prim1->isTensor() && prim2->isTensor() )
                {
                    return prim1->name() == prim2->name();
                }
            }
            return type1 == type2;
        }

        static const Type* commonType( const Type* type1, const Type* type2 )
        {
            if ( castable(type1, type2) )
            {
                return type2;
            }
            else if ( castable(type2, type1) )
            {
                return type1;
            }
            return nullptr;
        }

        static const Type* resultType( const Prototype& proto )
        {
            if ( proto.resultCount() == 1 )
            {
                return proto.result(0).type();
            }

            std::vector<const Type*> types(proto.resultCount());
            for ( size_t i = 0; i < proto.resultCount(); ++i )
            {
                types[i] = proto.result(i).type();
            }
            return tupleType(types);
        }
        
        static const Type* unaryResultType( const Type* argType, int op )
        {
            switch ( op )
            {
                case '-':
                case '+':
                {
                    if ( argType == primitiveType(Typename::Extent) ||
                        argType == primitiveType(Typename::Scalar) ||
                        argType == tensorType(Typename::Scalar) )
                    {
                        return argType;
                    }
                    break;
                }
                case '!':
                {
                    if ( argType == primitiveType(Typename::Logical) ||
                        argType == tensorType(Typename::Scalar) )
                    {
                        return argType;
                    }
                    break;
                }
            }
            return nullptr;
        }
        
        static const Type* binaryResultType( const Type* lhsType, const Type* rhsType, int op )
        {
            if ( op == Lexer::In && rhsType->isArray() )
            {
                return primitiveType(Typename::Logical);
            }
            else if ( op == '+' && lhsType->isArray() && rhsType == lhsType )
            {
                return lhsType;
            }
            else if ( op == '*' )
            {
                if ( lhsType->isArray() && rhsType == primitiveType(Typename::Extent) )
                {
                    return lhsType;
                }
                if ( rhsType->isArray() && lhsType == primitiveType(Typename::Extent) )
                {
                    return rhsType;
                }
            }
            
            const Type* argType = commonType(lhsType, rhsType);
            
            switch ( op )
            {
                case '<':
                case '>':
                case Lexer::Le:
                case Lexer::Ge:
                case Lexer::Eq:
                case Lexer::Ne:
                {
                    return argType == tensorType(Typename::Scalar) ? tensorType(Typename::Logical) : primitiveType(Typename::Logical);
                }
                case '+':
                case '*':
                {
                    if ( argType == primitiveType(Typename::String) )
                    {
                        return argType;
                    }
                }
                case '-':
                case '/':
                case '^':
                {
                    if ( argType == primitiveType(Typename::Extent) ||
                        argType == primitiveType(Typename::Scalar) ||
                        argType == tensorType(Typename::Scalar) )
                    {
                        return argType;
                    }
                    break;
                }
                case Lexer::And:
                case Lexer::Or:
                {
                    if ( argType == primitiveType(Typename::Logical) ||
                        argType == tensorType(Typename::Scalar) )
                    {
                        return argType;
                    }
                    break;
                }
            }
            return nullptr;
        }
        
        static const Type* builtinResultType( int op )
        {
            switch ( op )
            {
                case Lexer::LengthOf:
                {
                    return primitiveType(Typename::Extent);
                }
                case Lexer::ShapeOf:
                {
                    return arrayType(primitiveType(Typename::Extent));
                }
                case Lexer::RangeOf:
                {
                    return arrayType(primitiveType(Typename::Extent));
                }
                case Lexer::Extent:
                {
                    return primitiveType(Typename::Extent);
                }
                case Lexer::Scalar:
                {
                    return primitiveType(Typename::Scalar);
                }
                case Lexer::String:
                {
                    return primitiveType(Typename::String);
                }
                case Lexer::Logical:
                {
                    return primitiveType(Typename::Logical);
                }
            }
            
            return nullptr;
        }
        
        static int tokenPrecedence( int token )
        {
            static const std::map<int,int> precedence =
            {
                std::make_pair(Lexer::In, 10),
                std::make_pair(Lexer::And, 20),
                std::make_pair(Lexer::Or, 20),
                std::make_pair(Lexer::Le, 30),
                std::make_pair(Lexer::Ge, 30),
                std::make_pair(Lexer::Eq, 30),
                std::make_pair(Lexer::Ne, 30),
                std::make_pair('<', 30),
                std::make_pair('>', 30),
                std::make_pair('+', 40),
                std::make_pair('-', 40),
                std::make_pair('*', 50),
                std::make_pair('/', 50),
                std::make_pair('^', 60),
            };
            
            auto it = precedence.find(token);
            return it != precedence.end() ? it->second : -1;
        }
        
    private:
        
        static const char* unaryOpName( int op )
        {
            switch (op)
            {
                case '+':
                    return "idn";
                case '-':
                    return "neg";
                case '!':
                    return "not";
                default:
                    return nullptr;
            }
        }
        
        static const char* binaryOpName( int op )
        {
            switch (op)
            {
                case '+':
                    return "add";
                case '-':
                    return "sub";
                case '*':
                    return "mul";
                case '/':
                    return "div";
                case '^':
                    return "pow";
                case '<':
                    return "lt";
                case '>':
                    return "gt";
                case Lexer::Le:
                    return "le";
                case Lexer::Ge:
                    return "ge";
                case Lexer::Eq:
                    return "eq";
                case Lexer::Ne:
                    return "ne";
                case Lexer::And:
                    return "and";
                case Lexer::Or:
                    return "or";
                default:
                    return nullptr;
            }
        }
        
        static Dictionary<Shared<Expr>> makeUnaryOpArgs( const Shared<Expr>& right )
        {
            const Dictionary<Shared<Expr>> args =
            {
                { "x", right },
            };
            return args;
        }
        
        static Dictionary<Shared<Expr>> makeBinaryOpArgs( const Shared<Expr> left, const Shared<Expr> right )
        {
            const Dictionary<Shared<Expr>> args =
            {
                { "x", left },
                { "y", right },
            };
            return args;
        }
        
        static Dictionary<Shared<Expr>> makeSelectOpArgs( const Shared<Expr> condition, const Shared<Expr> trueValue, const Shared<Expr> falseValue )
        {
            const Dictionary<Shared<Expr>> args =
            {
                { "condition", condition },
                { "true_value", trueValue },
                { "false_value", falseValue },
            };
            return args;
        }

        static Shared<Expr> makeSelectExpr( const Position& position, Dictionary<Shared<Expr>>& args, const Type* type )
        {
            auto& condition = args["condition"];
            if ( condition->type()->isTensor() )
            {
                return std::make_shared<InvocationExpr>(position, "select", args, type);
            }

            auto& trueValue = args["true_value"];
            auto& falseValue = args["false_value"];
            return std::make_shared<SelectExpr>(position, condition, trueValue, falseValue, type);
        }

    private:

        bool _includeLayers;
    };
    
}   // namespace nnef


#endif
