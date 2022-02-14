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
#include "../common/prototype.h"
#include "../common/typeutils.h"
#include "../common/parser.h"
#include "../common/value.h"
#include "../common/lexer.h"
#include "../common/error.h"
#include "stdlib_source.h"
#include "expression.h"
#include "evaluation.h"
#include "fragment.h"
#include <exception>
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

        typedef Dictionary<Fragment> Fragments;
        typedef Dictionary<Prototype> Prototypes;
        typedef Dictionary<const Type*> Declarations;
        
    public:

        CompParser( const std::string& stdlib, const std::set<std::string>& lowered = {} )
        : _stdlib_source(!stdlib.empty() ? stdlib : stdlib_source()), _lowered(lowered), _flags(0)
        {
        }
        
        virtual void parse( std::istream& is, const char* filename, Callback& callback )
        {
            Lexer lexer(is, filename);
            lexer.next();

            auto version = readVersion(lexer);

            callback.beginDocument(filename, version);

            _flags = 0;
            auto extensions = readExtensions(lexer, [&]( const std::string& ext )
            {
                return callback.handleExtension(ext) || handleExtension(ext);
            });

            Prototypes prototypes;
            Fragments fragments;

            parseFragments(_stdlib_source, "stdlib", prototypes, fragments);

            if ( _flags & KHR_ENABLE_FRAGMENT_DEFINITIONS )
            {
                while ( lexer.token() == Lexer::Fragment )
                {
                    auto fragment = parseFragment(lexer, prototypes, (_flags & KHR_ENABLE_OPERATOR_EXPRESSIONS) != 0);
                    fragments.emplace(fragment.prototype().name(), std::move(fragment));
                }
            }

            lexer.readToken(Lexer::Graph);

            auto graph = parsePrototype(lexer, prototypes, false, true);
            auto assignments = parseAssignments(lexer, graph, prototypes, (_flags & KHR_ENABLE_OPERATOR_EXPRESSIONS) != 0, true);
            
            callback.beginGraph(graph, prototypes);

            Dictionary<Value> values;
            Dictionary<Typename> dtypes;
			std::set<std::string> vars;

            Evaluation evaluation(assignments, fragments, _lowered);
            for ( auto& assignment : assignments )
            {
				checkExternalsAndVariables(assignment.lhs(), assignment.rhs(), graph, vars);
				
                const Value context = evaluation.evaluateLvalue(assignment.lhs(), Dictionary<Value>(), true);
                evaluation.evaluateAssign(assignment.lhs(), assignment.rhs(), values, dtypes, callback, nullptr, context);
            }

            callback.endGraph(graph, dtypes);
            callback.endDocument(filename);

            lexer.readToken(Lexer::Eof);
        }

    private:

        bool handleExtension( const std::string& ext )
        {
            if ( ext == "KHR_enable_fragment_definitions" )
            {
                _flags |= KHR_ENABLE_FRAGMENT_DEFINITIONS;
                return true;
            }
            else if ( ext == "KHR_enable_operator_expressions" )
            {
                _flags |= KHR_ENABLE_OPERATOR_EXPRESSIONS;
                return true;
            }
            return false;
        }

        static void parseFragments( const std::string& text, const char* filename, Prototypes& prototypes, Fragments& fragments )
        {
            std::stringstream ss(text);
            Lexer lexer(ss, filename);
            lexer.next();

            while ( lexer.token() != Lexer::Eof )
            {
                auto fragment = parseFragment(lexer, prototypes, true);
                fragments.emplace(fragment.prototype().name(), std::move(fragment));
            }
        }

        static Prototype parsePrototype( Lexer& lexer, const Prototypes& prototypes, bool allowTypespec, bool graph )
        {
            auto position = lexer.position();

            const std::string name = lexer.string();
            lexer.readToken(Lexer::Identifier);

            if ( prototypes.count(name) )
            {
                throw Error(position, "operation '%s' already defined", name.c_str());
            }

            bool isGenericDecl = false;
            const PrimitiveType* genericParamDefault = nullptr;
            if ( !graph && lexer.readIfToken('<') )
            {
                isGenericDecl = true;

                lexer.readToken('?');
                if ( lexer.readIfToken('=') )
                {
                    genericParamDefault = primitiveType(getTypename(lexer));
                    lexer.next();
                }
                
                lexer.readToken('>');
            }

            std::vector<Param> params = parseParams(lexer, name, allowTypespec, graph);

            lexer.readToken(Lexer::Arrow);

            std::vector<Result> results = parseResults(lexer, name, allowTypespec, !graph);

            for ( auto& result : results )
            {
                if ( std::find_if(params.begin(), params.end(), [&]( const Param& param ){ return param.name() == result.name(); }) != params.end() )
                {
                    throw Error(position, "invalid definition of operation '%s'; '%s' is defined both as parameter and as result",
                                name.c_str(), result.name().c_str());
                }
            }

            bool attribute = results.front().type()->isAttribute();
            for ( size_t i = 1; i < results.size(); ++i )
            {
                if ( results[i].type()->isAttribute() != attribute )
                {
                    throw Error(position, "result types of fragment must be all tensor types or all attribute types");
                }
            }

            auto isGenericTyped = []( const Typed& typed ){ return typed.type()->isGeneric(); };
            bool hasGenericParams = std::any_of(params.begin(), params.end(), isGenericTyped);
            bool hasGenericResults = std::any_of(results.begin(), results.end(), isGenericTyped);
            if ( (hasGenericParams || hasGenericResults) && !isGenericDecl )
            {
                throw Error(position, "fragment with generic parameter or result types must be declared generic using <?>");
            }
            else if ( isGenericDecl && !hasGenericParams && !hasGenericResults )
            {
                throw Error(position, "fragment declared as generic must have at least one generic parameter or result type");
            }

            return Prototype(name, params, results, genericParamDefault);
        }

        static std::vector<Param> parseParams( Lexer& lexer, const std::string& op, bool allowTypespec, bool forceDefaults )
        {
            std::vector<Param> params;

            lexer.readToken('(');

            bool expectAttribute = false;
            do
            {
                auto position = lexer.position();

                auto name = lexer.string();
                lexer.readToken(Lexer::Identifier);

                const Type* type = tensorType();
                if ( allowTypespec )
                {
                    lexer.readToken(':');
                    type = parseTypespec(lexer, true);
                }

                if ( expectAttribute && !type->isAttribute() )
                {
                    throw Error(position, "expected attribute, found parameter of type '%s'", type->toString().c_str());
                }

                expectAttribute |= type->isAttribute();

                auto defaultValue = Value::none();
                if ( lexer.token() == '=' )
                {
                    lexer.next();

                    auto expr = parseExpression(lexer, nullptr, nullptr, true, false, false, false);

                    if ( !isCastable(expr->type(), type) )
                    {
                        throw Error(expr->position(), "default value type '%s' cannot be cast to parameter type '%s'",
                                    expr->type()->toString().c_str(), type->toString().c_str());
                    }

                    defaultValue = Evaluation::evaluateRvalue(*expr);
                }
                else if ( forceDefaults && type->isAttribute() )
                {
                    throw Error(position, "expected default value for parameter '%s'", name.c_str());
                }

                if ( std::find_if(params.begin(), params.end(), [&]( const Param& param ){ return param.name() == name; }) != params.end() )
                {
                    throw Error(position, "duplicate parameter definition for fragment '%s'; parameter '%s' is already defined",
                                op.c_str(), name.c_str());
                }

                params.emplace_back(name, type, defaultValue);
            }
            while ( lexer.readIfToken(',') );

            lexer.readToken(')');

            return params;
        }

        static std::vector<Result> parseResults( Lexer& lexer, const std::string& op, bool allowTypespec, bool allowAttribute )
        {
            std::vector<Result> results;

            lexer.readToken('(');

            do
            {
                auto position = lexer.position();

                auto name = lexer.string();
                lexer.readToken(Lexer::Identifier);

                const Type* type = tensorType();
                if ( allowTypespec )
                {
                    lexer.readToken(':');
                    type = parseTypespec(lexer, false);

                    if ( !allowAttribute && type->isAttribute() )
                    {
                        throw Error(position, "non-tensor type not allowed in this context");
                    }
                }

                if ( std::find_if(results.begin(), results.end(), [&]( const Result& result ){ return result.name() == name; }) != results.end() )
                {
                    throw Error(position, "duplicate result definition for operation '%s'; result '%s' is already defined",
                                op.c_str(), name.c_str());
                }

                results.emplace_back(name, type);
            }
            while ( lexer.readIfToken(',') );

            lexer.readToken(')');

            return results;
        }

        static Fragment parseFragment( Lexer& lexer, Prototypes& prototypes, bool allowOperator )
        {
            lexer.readToken(Lexer::Fragment);

            auto prototype = parsePrototype(lexer, prototypes, true, false);
            auto& proto = prototypes.emplace(prototype.name(), prototype).first->second;

            std::vector<Assignment> assignments;
            if ( !lexer.readIfToken(';') )
            {
                assignments = parseAssignments(lexer, proto, prototypes, allowOperator, false);
            }

            return Fragment(proto, assignments);
        }

        static std::vector<Assignment> parseAssignments( Lexer& lexer, const Prototype& proto, const Prototypes& prototypes, bool allowOperator, bool graph )
        {
            Declarations decls;
            for ( size_t i = 0; i < proto.paramCount(); ++i )
            {
                auto& param = proto.param(i);
                if ( !graph || param.type()->isAttribute() )
                {
                    decls[param.name()] = param.type();
                }
            }

            std::vector<Assignment> assignments;

            lexer.readToken('{');

            do
            {
                auto lhs = parseTuple(lexer, nullptr, nullptr, false, true, false);

                lexer.readToken('=');

                auto rhs = allowOperator ? parseExpression(lexer, &prototypes, &decls, true, true, true) : parseInvocation(lexer, &prototypes, &decls);

                lexer.readToken(';');

                declare(*lhs, rhs->type(), decls);
				if ( !graph )
				{
					checkOperationsAllowed(*rhs);
				}

                assignments.emplace_back(lhs, rhs);
            }
            while ( lexer.token() != '}' );

            if ( graph )
            {
                for ( size_t i = 0; i < proto.paramCount(); ++i )
                {
                    auto& param = proto.param(i);
                    if ( !decls.count(param.name()) )
                    {
                        throw Error(lexer.position(), "graph parameter '%s' is not assigned", param.name().c_str());
                    }
                }
            }

            for ( size_t i = 0; i < proto.resultCount(); ++i )
            {
                auto& result = proto.result(i);
                auto decl = decls[result.name()];
                if ( !decl )
                {
                    throw Error(lexer.position(), "result '%s' of operation '%s' is not assigned",
                                result.name().c_str(), proto.name().c_str());
                }
                else if ( !isCastable(decl, result.type(), true) )
                {
                    throw Error(lexer.position(), "result '%s' of operation '%s' is declared as '%s' but assignment has incompatible type '%s'",
                                result.name().c_str(), proto.name().c_str(), result.type()->toString().c_str(), decl->toString().c_str());
                }
            }

            lexer.readToken('}');

            return assignments;
        }
		
		static void checkOperationsAllowed( const Expr& rhs )
		{
			traverse(rhs, []( const Expr& expr )
			{
				if ( expr.kind() == Expr::Invocation )
				{
					auto& invocation = static_cast<const InvocationExpr&>(expr);
					
					if ( invocation.target() == "external" || invocation.target() == "variable" || invocation.target() == "update" )
					{
						throw Error(invocation.position(), "operation '%s' not allowed inside fragments", invocation.target().c_str());
					}
				}
			});
		}
		
		void checkExternalsAndVariables( const Expr& lhs, const Expr& rhs, const Prototype& graph, std::set<std::string>& vars )
		{
			if ( (lhs.kind() == Expr::Array || lhs.kind() == Expr::Tuple) && rhs.kind() == lhs.kind() )
			{
				auto& left = static_cast<const ItemExpr&>(lhs);
				auto& right = static_cast<const ItemExpr&>(rhs);
				
				for ( size_t i = 0; i < left.size(); ++i )
				{
					checkExternalsAndVariables(left.item(i), right.item(i), graph, vars);
				}
			}
			else if ( rhs.kind() == Expr::Invocation && lhs.kind() == Expr::Identifier )
			{
				auto& identifier = static_cast<const IdentifierExpr&>(lhs);
				auto& invocation = static_cast<const InvocationExpr&>(rhs);
				
				if ( invocation.target() == "external" )
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
				
				if ( invocation.target() == "variable" )
				{
					vars.insert(identifier.name());
				}
				
				if ( invocation.target() == "update" )
				{
					auto& arg = *invocation.arg("variable");
					
					if ( arg.kind() != Expr::Identifier || !vars.count(static_cast<const IdentifierExpr&>(arg).name()) )
					{
						throw Error(arg.position(), "first argument to operation 'update' must be a variable");
					}
				}
			}
		}
		
		static void traverse( const Expr& expr, std::function<void(const Expr&)> func )
		{
			func(expr);
			switch ( expr.kind() )
			{
				case Expr::Literal:
				case Expr::Identifier:
				{
					break;
				}
				case Expr::Builtin:
				{
					auto& builtin = static_cast<const BuiltinExpr&>(expr);
					traverse(builtin.arg(), func);
					break;
				}
				case Expr::Array:
				case Expr::Tuple:
				{
					auto& items = static_cast<const ItemExpr&>(expr);
					for ( size_t i = 0; i < items.size(); ++i )
					{
						traverse(items.item(i), func);
					}
					break;
				}
				case Expr::Subscript:
				{
					auto& subscript = static_cast<const SubscriptExpr&>(expr);
					traverse(subscript.sequence(), func);
					if ( subscript.begin() )
					{
						traverse(*subscript.begin(), func);
					}
					if ( subscript.end() )
					{
						traverse(*subscript.end(), func);
					}
					break;
				}
				case Expr::Comprehension:
				{
					auto& comprehension = static_cast<const ComprehensionExpr&>(expr);
					for ( size_t i = 0; i < comprehension.iteratorCount(); ++i )
					{
						traverse(comprehension.iterator(i), func);
						traverse(comprehension.iterable(i), func);
					}
					if ( comprehension.condition() )
					{
						traverse(*comprehension.condition(), func);
					}
					traverse(comprehension.item(), func);
					break;
				}
				case Expr::Unary:
				{
					auto& unary = static_cast<const UnaryExpr&>(expr);
					traverse(unary.right(), func);
					break;
				}
				case Expr::Binary:
				{
					auto& binary = static_cast<const BinaryExpr&>(expr);
					traverse(binary.left(), func);
					traverse(binary.right(), func);
					break;
				}
				case Expr::Select:
				{
					auto& select = static_cast<const SelectExpr&>(expr);
					traverse(select.condition(), func);
					traverse(select.trueValue(), func);
					traverse(select.falseValue(), func);
					break;
				}
				case Expr::Invocation:
				{
					auto& invocation = static_cast<const InvocationExpr&>(expr);
					for ( auto it = invocation.begin(); it != invocation.end(); ++it )
					{
						traverse(*it->second, func);
					}
					break;
				}
			}
		}

    private:

        static const Type* parseArrayTypespec( Lexer& lexer, const Type* type )
        {
            while ( lexer.readIfToken('[') )
            {
                lexer.readToken(']');

                type = arrayType(type);
            }

            return type;
        }

        static const Type* parseTupleTypespec( Lexer& lexer, bool allowUnboundTensor )
        {
            auto position = lexer.position();

            lexer.next();

            std::vector<const Type*> items;
            do
            {
                items.push_back(parseTypespec(lexer, allowUnboundTensor));
            }
            while ( lexer.readIfToken(',') );

            lexer.readToken(')');

            bool attribute = items.front()->isAttribute();
            for ( size_t i = 1; i < items.size(); ++i )
            {
                if ( items[i]->isAttribute() != attribute )
                {
                    throw Error(position, "item types in tuple type must be all attribute types or all tensor types");
                }
            }

            return parseArrayTypespec(lexer, tupleType(items));
        }

        static const Type* parseTypespec( Lexer& lexer, bool allowUnboundTensor )
        {
            if ( lexer.token() == '(' )
            {
                return parseTupleTypespec(lexer, allowUnboundTensor);
            }

            const Type* type = nullptr;
            if ( lexer.readIfToken(Lexer::Tensor) )
            {
                lexer.readToken('<');

                type = tensorType();

                if ( lexer.token() != '>' )
                {
                    type = tensorType(getTypename(lexer));
                    lexer.next();
                }
                else if ( !allowUnboundTensor )
                {
                    throw Error(lexer.position(), "unbound tensor not allowed in this context");
                }

                lexer.readToken('>');
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
        
        static Shared<Expr> parseExpression( Lexer& lexer, const Prototypes* prototypes, Declarations* decls,
                                            bool allowLiteral, bool allowIdentifier, bool allowOperator,
                                            bool allowSelect = true )
        {
            auto expr = parsePrimary(lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator);
            if ( expr->kind() != Expr::Literal && allowOperator )
            {
                expr = parseSubscripts(lexer, prototypes, decls, expr);
            }
            if ( allowOperator )
            {
                expr = parseBinary(lexer, prototypes, decls, expr);
                if ( lexer.token() == Lexer::If && allowSelect )
                {
                    expr = parseSelect(lexer, prototypes, decls, expr);
                }
            }
            return expr;
        }
        
        static Shared<Expr> parsePrimary( Lexer& lexer, const Prototypes* prototypes, Declarations* decls,
                                         bool allowLiteral, bool allowIdentifier, bool allowOperator )
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
                case Lexer::Fractional:
                {
                    if ( allowLiteral )
                    {
                        return parseScalar(lexer);
                    }
                    break;
                }
                case Lexer::Decimal:
                {
                    if ( allowLiteral )
                    {
                        return parseInteger(lexer);
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
                case Lexer::Identifier:
                {
                    if ( allowIdentifier )
                    {
                        return parseIdentifier(lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator);
                    }
                    break;
                }
                case '[':
                {
                    return parseArray(lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator);
                }
                case '(':
                {
                    return parseTuple(lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator);
                }
                case '-':
                {
                    return parseUnary(lexer, prototypes, decls);
                }
                case '!':
                {
                    if ( allowOperator )
                    {
                        return parseUnary(lexer, prototypes, decls);
                    }
                    break;
                }
                case Lexer::ShapeOf:
                {
                    throw Error(lexer.position(), "the use of operator 'shape_of' is deprecated and is not supported");
                }
                case Lexer::LengthOf:
                case Lexer::RangeOf:
                case Lexer::Integer:
                case Lexer::Scalar:
                case Lexer::Logical:
                case Lexer::String:
                {
                    if ( allowOperator )
                    {
                        return parseBuiltin(lexer, prototypes, decls);
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
        
        static Shared<Expr> parseInteger( Lexer& lexer )
        {
            auto position = lexer.position();
            
            auto value = getIntegerValue(lexer);
            lexer.next();
            
            return std::make_shared<IntegerExpr>(position, value, primitiveType(Typename::Integer));
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
        
        static Shared<Expr> parseIdentifier( Lexer& lexer, const Prototypes* prototypes, Declarations* decls,
                                            bool allowLiteral, bool allowIdentifier, bool allowOperator )
        {
            auto position = lexer.position();
            auto string = lexer.string();

            lexer.readToken(Lexer::Identifier);
            
            if ( lexer.token() == '(' || (lexer.token() == '<' && prototypes && prototypes->count(string)) )
            {
                return parseInvocation(lexer, prototypes, decls, position, string, allowLiteral, allowIdentifier, allowOperator);
            }
            else
            {
                return makeIdentifier(position, string, decls);
            }
        }
        
        static Shared<Expr> makeIdentifier( const Position& position, const std::string& name, Declarations* decls )
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
        
        static Shared<Expr> parseArray( Lexer& lexer, const Prototypes* prototypes, Declarations* decls,
                                       bool allowLiteral, bool allowIdentifier, bool allowOperator )
        {
            auto position = lexer.position();
            lexer.next();
            
            std::vector<Shared<Expr>> items;

            const Type* type = nullptr;
            
            if ( lexer.token() != ']' )
            {
                if ( lexer.token() == Lexer::For )
                {
                    return parseComprehension(lexer, prototypes, decls, position);
                }

                auto first = parseExpression(lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator);
                items = { first };
                type = first->type();

                while ( lexer.readIfToken(',') )
                {
                    auto item = parseExpression(lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator);
                    items.push_back(item);

                    if ( decls )
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
            
            lexer.readToken(']');
            
            return std::make_shared<ArrayExpr>(position, items, arrayType(type));
        }
        
        static Shared<Expr> parseTuple( Lexer& lexer, const Prototypes* prototypes, Declarations* decls,
                                       bool allowLiteral, bool allowIdentifier, bool allowOperator )
        {
            auto position = lexer.position();
            
            bool parenthesized = lexer.token() == '(';
            if ( parenthesized )
            {
                lexer.next();
            }

            std::vector<Shared<Expr>> items;
            std::vector<const Type*> types;

            auto first = parseExpression(lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator);

            if ( lexer.token() == ',' )
            {
                items = { first };
                types = { first->type() };

                while ( lexer.readIfToken(',') )
                {
                    auto item = parseExpression(lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator);
                    items.push_back(item);
                    types.push_back(item->type());
                }
            }
            
            if ( parenthesized )
            {
                lexer.readToken(')');
            }

            return items.empty() ? first : std::make_shared<TupleExpr>(position, items, tupleType(types));
        }

        static Shared<Expr> parseInvocation( Lexer& lexer, const Prototypes* prototypes, Declarations* decls )
        {
            auto position = lexer.position();
            auto string = lexer.string();

            lexer.readToken(Lexer::Identifier);

            if ( lexer.token() != '(' && lexer.token() != '<' )
            {
                throw Error(position, "expected operation invocation");
            }

            return parseInvocation(lexer, prototypes, decls, position, string, true, true, false);
        }
        
        static Shared<Expr> parseInvocation( Lexer& lexer, const Prototypes* prototypes, Declarations* decls,
                                            const Position& position, const std::string& target,
                                            bool allowLiteral, bool allowIdentifier, bool allowOperator )
        {
            auto it = prototypes->find(target);
            if ( it == prototypes->end() )
            {
                throw Error(position, "undefined operation '%s'", target.c_str());
            }

            const Prototype& proto = it->second;
            
            const PrimitiveType* dataType = proto.genericParamDefault();
            if ( lexer.readIfToken('<') )
            {
                dataType = primitiveType(getTypename(lexer));
                lexer.next();
                
                lexer.readToken('>');
            }

            lexer.readToken('(');

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
                    
                    if ( lexer.readIfToken('=') )
                    {
                        param = proto.param(string);
                        if ( !param )
                        {
                            throw Error(position, "operation '%s' has no parameter called '%s'",
                                        proto.name().c_str(), string.c_str());
                        }
                        
                        arg = parseExpression(lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator);
                        named = true;
                    }
                    else
                    {
                        param = &proto.param(args.size());
                        if ( lexer.token() == '(' )
                        {
                            arg = parseInvocation(lexer, prototypes, decls, position, string, allowLiteral, allowIdentifier, allowOperator);
                        }
                        else
                        {
                            arg = makeIdentifier(position, string, decls);
                        }
                        arg = parseSubscripts(lexer, prototypes, decls, arg);
                        arg = parseBinary(lexer, prototypes, decls, arg);
                        if ( lexer.token() == Lexer::If )
                        {
                            arg = parseSelect(lexer, prototypes, decls, arg);
                        }
                    }
                }
                else
                {
                    param = &proto.param(args.size());
                    arg = parseExpression(lexer, prototypes, decls, allowLiteral, allowIdentifier, allowOperator);
                }

                auto paramType = dataType ? bindDataType(param->type(), dataType) : param->type();
                if ( !isCastable(arg->type(), paramType) )
                {
                    throw Error(position, "argument of type '%s' cannot be cast to type '%s' for parameter '%s'",
                                arg->type()->toString().c_str(), paramType->toString().c_str(), param->name().c_str());
                }
                
                expectNamed |= named || paramType->isAttribute();
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
            while ( lexer.readIfToken(',') );
            
            for ( size_t i = 0; i < proto.paramCount(); ++i )
            {
                auto& param = proto.param(i);

                if ( !args.count(param.name()) )
                {
                    if ( !param.defaultValue() )
                    {
                        throw Error(lexer.position(), "missing argument for fragment '%s'; parameter '%s' not assigned",
                                        proto.name().c_str(), param.name().c_str());
                    }
                    else if ( param.type()->isGeneric() )
                    {
                        auto valueType = typeOf(param.defaultValue());
                        auto paramType = dataType ? bindDataType(param.type(), dataType) : param.type();
                        if ( !isCastable(valueType, paramType) )
                        {
                            throw Error(lexer.position(), "default value type '%s' cannot be cast to type '%s' for parameter '%s'",
                                        valueType->toString().c_str(), paramType->toString().c_str(), param.name().c_str());
                        }
                    }
                }
            }
            
            lexer.readToken(')');
            
            if ( proto.isGeneric() && !dataType && !deduceDataType(proto, args, dataType, position) )
            {
                throw Error(position, "could not deduce generic data-type");
            }
            
            const Type* type = resultType(proto, dataType);

            return std::make_shared<InvocationExpr>(position, target, args, type, dataType);
        }
        
        static Shared<Expr> parseUnary( Lexer& lexer, const Prototypes* prototypes, Declarations* decls )
        {
            auto position = lexer.position();
            int op = lexer.token();
            lexer.next();
            
            auto rhs = parseExpression(lexer, prototypes, decls, true, true, true);
            
            auto type = unaryResultType(rhs->type(), op);
            if ( !type )
            {
                throw Error(position, "invalid operand type '%s' for operation '%s'",
                            rhs->type()->toString().c_str(), Lexer::tokenString(op).c_str());
            }
            
            if ( type->kind() == Type::Tensor )
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
        
        static Shared<Expr> parseBinary( Lexer& lexer, const Prototypes* prototypes, Declarations* decls, Shared<Expr> lhs, int exprPrec = 0 )
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
                
                auto rhs = parsePrimary(lexer, prototypes, decls, true, true, true);
                rhs = parseSubscripts(lexer, prototypes, decls, rhs);
                
                int nextPrec = tokenPrecedence(lexer.token());
                if ( tokPrec < nextPrec )
                {
                    rhs = parseBinary(lexer, prototypes, decls, rhs, tokPrec + 1);
                }
                
                auto type = binaryResultType(lhs->type(), rhs->type(), op);
                if ( !type )
                {
                    throw Error(position, "invalid operand types '%s' and '%s' for operation '%s'",
                                lhs->type()->toString().c_str(),  rhs->type()->toString().c_str(),
                                Lexer::tokenString(op).c_str());
                }
                
                if ( type->kind() == Type::Tensor )
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
        
        static Shared<Expr> parseBuiltin( Lexer& lexer, const Prototypes* prototypes, Declarations* decls )
        {
            auto position = lexer.position();
            int op = lexer.token();
            lexer.next();
            
            lexer.readToken('(');
            
            auto arg = parseExpression(lexer, prototypes, decls, true, true, true);
            
            auto type = builtinResultType(op);
            if ( !type )
            {
                throw Error(position, "invalid operand type '%s' for operation '%s'",
                            arg->type()->toString().c_str(), Lexer::tokenString(op).c_str());
            }
            
            lexer.readToken(')');

            if ( op == Lexer::LengthOf )
            {
                if ( arg->type()->kind() != Type::Array && arg->type() != primitiveType(Typename::String) )
                {
                    throw Error(position, "argument of length_of() must be an array or string (found %s)", arg->type()->toString().c_str());
                }
            }
            if ( op == Lexer::ShapeOf )
            {
                if ( arg->type()->kind() != Type::Tensor && arg->type()->kind() != Type::Primitive )
                {
                    throw Error(position, "argument of shape_of() must be of tensor or primitive type (found %s)",
                                arg->type()->toString().c_str());
                }
            }
            else if ( op == Lexer::RangeOf && arg->type() != primitiveType(Typename::String) )
            {
                if ( arg->type()->kind() != Type::Array )
                {
                    throw Error(position, "argument of range_of() must be an array or string (found %s)",
                                arg->type()->toString().c_str());
                }
            }
            else if ( op == Lexer::Integer || op == Lexer::Scalar || op == Lexer::Logical || op == Lexer::String )
            {
                if ( arg->type()->kind() != Type::Primitive )
                {
                    throw Error(position, "argument of %s() must be of non-tensor primitive type (found %s)",
                                Lexer::tokenString(op).c_str(), arg->type()->toString().c_str());
                }
            }
            
            return std::make_shared<BuiltinExpr>(position, arg, op, type);
        }

        static Shared<Expr> parseSubscript( Lexer& lexer, const Prototypes* prototypes, Declarations* decls, const Shared<Expr> sequence )
        {
            lexer.next();

            Shared<Expr> beg, end;
            const Type* type = nullptr;

            if ( sequence->type()->kind() == Type::Tuple )
            {
                beg = parseExpression(lexer, prototypes, decls, true, true, true);
                if ( beg->kind() != Expr::Literal || beg->type() != primitiveType(Typename::Integer) )
                {
                    throw Error(beg->position(), "tuple index must be an integer literal");
                }

                auto idx = static_cast<const IntegerExpr&>(*beg).value();
				
				lexer.readToken(']');

                type = static_cast<const TupleType*>(sequence->type())->itemType(idx);
            }
            else if ( sequence->type()->kind() == Type::Array || sequence->type() == primitiveType(Typename::String) )
            {
                if ( lexer.token() != ':' )
                {
                    beg = parseExpression(lexer, prototypes, decls, true, true, true);
                    if ( beg->type() != primitiveType(Typename::Integer) )
                    {
                        throw Error(beg->position(), "array index must be of type integer, found '%s'", beg->type()->toString().c_str());
                    }
                }
                bool range = false;
                if ( lexer.readIfToken(':') )
                {
                    range = true;

                    if ( lexer.token() != ']' )
                    {
                        end = parseExpression(lexer, prototypes, decls, true, true, true);
                        if ( end->type() != primitiveType(Typename::Integer) )
                        {
                            throw Error(end->position(), "array index must be of type integer, found '%s'", end->type()->toString().c_str());
                        }
                    }
                }
                else
                {
                    end = beg;
                }

                lexer.readToken(']');

                if ( sequence->type()->kind() == Type::Array )
                {
                    auto arrayType = static_cast<const ArrayType*>(sequence->type());
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

        static Shared<Expr> parseSubscripts( Lexer& lexer, const Prototypes* prototypes, Declarations* decls, Shared<Expr> sequence )
        {
            while ( lexer.token() == '[' )
            {
                sequence = parseSubscript(lexer, prototypes, decls, sequence);
            }
            return sequence;
        }

        static Shared<Expr> parseSelect( Lexer& lexer, const Prototypes* prototypes, Declarations* decls, Shared<Expr> trueValue )
        {
            lexer.readToken(Lexer::If);

            auto condition = parseExpression(lexer, prototypes, decls, true, true, true);
            if ( condition->type() != primitiveType(Typename::Logical) )
            {
                throw Error(condition->position(), "condition must be a logical value");
            }

            lexer.readToken(Lexer::Else);

            auto falseValue = parseExpression(lexer, prototypes, decls, true, true, true);

            const Type* type = commonType(trueValue->type(), falseValue->type());
            if ( !type )
            {
                throw Error(trueValue->position(), "incompatible types in if-else expression (%s vs %s)",
                            trueValue->type()->toString().c_str(), falseValue->type()->toString().c_str());
            }

            return std::make_shared<SelectExpr>(trueValue->position(), condition, trueValue, falseValue, type);
        }

        static Shared<Expr> parseComprehension( Lexer& lexer, const Prototypes* prototypes, Declarations* decls,
                                               const Position& position )
        {
            lexer.readToken(Lexer::For);

            std::vector<Shared<Expr>> iterators, iterables;
            
            do
            {
                auto iterator = parseIterator(lexer, decls);
                
                lexer.readToken(Lexer::In);

                auto iterable = parseExpression(lexer, prototypes, decls, true, true, true, false);
                if ( iterable->type()->kind() != Type::Array )
                {
                    throw Error(iterable->position(), "expression not iterable");
                }
                
                iterators.push_back(iterator);
                iterables.push_back(iterable);
                
                auto itemType = static_cast<const ArrayType*>(iterable->type())->itemType();
                declare(*iterator, itemType, *decls);
            }
            while ( lexer.readIfToken(',') );

            Shared<Expr> condition = nullptr;
            if ( lexer.readIfToken(Lexer::If) )
            {
                condition = parseExpression(lexer, prototypes, decls, true, true, true);
                if ( condition->type() != primitiveType(Typename::Logical) )
                {
                    throw Error(condition->position(), "condition in comprehension expression must be a logical expression");
                }
            }

            lexer.readToken(Lexer::Yield);

            auto item = parseExpression(lexer, prototypes, decls, true, true, true);
            const Type* type = arrayType(item->type());

            for ( size_t i = 0; i < iterators.size(); ++i )
            {
                undeclare(*iterators[i], *decls);
            }

            lexer.readToken(']');

            return std::make_shared<ComprehensionExpr>(position, iterators, iterables, condition, item, type);
        }
        
    private:
        
        static Shared<Expr> parseIterator( Lexer& lexer, const Declarations* decls )
        {
            if ( lexer.token() == Lexer::Identifier )
            {
                auto iterator = std::make_shared<IdentifierExpr>(lexer.position(), lexer.string(), nullptr);
                lexer.readToken(Lexer::Identifier);
                
                return iterator;
            }
            
            if ( lexer.token() != '(' )
            {
                throw Error(lexer.position(), "expected tuple or identifier");
            }
            lexer.next();
            
            auto position = lexer.position();
            
            std::vector<Shared<Expr>> items;
            std::vector<const Type*> types;
            
            auto first = parseIterator(lexer, decls);
            
            if ( lexer.token() == ',' )
            {
                items = { first };
                types = { first->type() };
                
                while ( lexer.readIfToken(',') )
                {
                    auto item = parseIterator(lexer, decls);
                    items.push_back(item);
                    types.push_back(item->type());
                }
            }
            
            lexer.readToken(')');
            
            return items.empty() ? first : std::make_shared<TupleExpr>(position, items, tupleType(types));
        }
        
    private:

        static void declare( const Expr& expr, const Type* type, Declarations& declared )
        {
            switch ( expr.kind() )
            {
                case Expr::Identifier:
                {
                    auto& identifier = static_cast<const IdentifierExpr&>(expr);
                    if ( declared.count(identifier.name()) )
                    {
                        throw Error(expr.position(), "identifier '%s' is already declared", identifier.name().c_str());
                    }
                    declared.emplace(identifier.name(), type);
                    break;
                }
                case Expr::Array:
                {
                    if ( type->kind() != Type::Array )
                    {
                        throw Error(expr.position(), "cannot assign result of type '%s' to array", type->toString().c_str());
                    }
                    auto& array = static_cast<const ArrayExpr&>(expr);
                    auto arrayType = static_cast<const ArrayType*>(type);
                    for ( size_t i = 0; i < array.size(); ++i )
                    {
                        declare(array.item(i), arrayType->itemType(), declared);
                    }
                    break;
                }
                case Expr::Tuple:
                {
                    if ( type->kind() != Type::Tuple )
                    {
                        throw Error(expr.position(), "cannot assign result of type '%s' to tuple", type->toString().c_str());
                    }
                    auto& tuple = static_cast<const TupleExpr&>(expr);
                    auto tupleType = static_cast<const TupleType*>(type);
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
        
        static void undeclare( const Expr& expr, Declarations& declared )
        {
            switch ( expr.kind() )
            {
                case Expr::Identifier:
                {
                    auto& identifier = static_cast<const IdentifierExpr&>(expr);
                    declared.erase(identifier.name());
                    break;
                }
                case Expr::Array:
                case Expr::Tuple:
                {
                    auto& items = static_cast<const ItemExpr&>(expr);
                    for ( size_t i = 0; i < items.size(); ++i )
                    {
                        undeclare(items.item(i), declared);
                    }
                    break;
                }
                default:
                {
                    throw Error(expr.position(), "expression not allowed in this context");
                }
            }
        }

    private:

        static bool deduceDataType( const Prototype& proto, const Dictionary<Shared<Expr>>& args, const PrimitiveType*& dataType,
                                   const Position& position )
        {
            Dictionary<const Type*> types;
            for ( auto& arg : args )
            {
                types[arg.first] = arg.second->type();
            }
            for ( size_t i = 0; i < proto.paramCount(); ++i )
            {
                auto& param = proto.param(i);
                if ( !types.count(param.name()) )
                {
                    assert(param.defaultValue());
                    types[param.name()] = typeOf(param.defaultValue());
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

        static const Type* resultType( const Prototype& proto, const PrimitiveType* dataType )
        {
            if ( proto.resultCount() == 1 )
            {
                return dataType ? bindDataType(proto.result(0).type(), dataType) : proto.result(0).type();
            }

            std::vector<const Type*> types(proto.resultCount());
            for ( size_t i = 0; i < proto.resultCount(); ++i )
            {
                types[i] = dataType ? bindDataType(proto.result(i).type(), dataType) : proto.result(i).type();
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
                    if ( argType == primitiveType(Typename::Integer) ||
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
            if ( op == Lexer::In && rhsType->kind() == Type::Array )
            {
                return primitiveType(Typename::Logical);
            }
            else if ( op == '+' && lhsType->kind() == Type::Array && rhsType == lhsType )
            {
                return lhsType;
            }
            else if ( op == '*' )
            {
                if ( lhsType->kind() == Type::Array && rhsType == primitiveType(Typename::Integer) )
                {
                    return lhsType;
                }
                if ( rhsType->kind() == Type::Array && lhsType == primitiveType(Typename::Integer) )
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
                    return argType == tensorType(Typename::Scalar) ? (const Type*)tensorType(Typename::Logical) : (const Type*)primitiveType(Typename::Logical);
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
                    if ( argType == primitiveType(Typename::Integer) ||
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
                    return primitiveType(Typename::Integer);
                }
                case Lexer::ShapeOf:
                {
                    return arrayType(primitiveType(Typename::Integer));
                }
                case Lexer::RangeOf:
                {
                    return arrayType(primitiveType(Typename::Integer));
                }
                case Lexer::Integer:
                {
                    return primitiveType(Typename::Integer);
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

        static const Type* typeOf( const Value& value )
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
                case Value::Array:
                {
                    auto itemType = value.size() ? typeOf(value[0]) : nullptr;
                    return arrayType(itemType);
                }
                case Value::Tuple:
                {
                    std::vector<const Type*> itemTypes(value.size());
                    for ( size_t i = 0; i < value.size(); ++i )
                    {
                        itemTypes[i] = typeOf(value[i]);
                    }
                    return tupleType(itemTypes);
                }
                case Value::Identifier:
                case Value::None:
                {
                    return nullptr;
                }
            }
            assert(false);
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
                    return "copy";
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

    private:

        static bool checkGraphParamType( const Value& value, const Type* type )
        {
            switch ( value.kind() )
            {
                case Value::Integer:
                {
                    return type == primitiveType(Typename::Integer);
                }
                case Value::Scalar:
                {
                    return type == primitiveType(Typename::Scalar);
                }
                case Value::Logical:
                {
                    return type == primitiveType(Typename::Logical);
                }
                case Value::String:
                {
                    return type == primitiveType(Typename::String);
                }
                case Value::Identifier:
                {
                    return type == tensorType();
                }
                case Value::Array:
                {
                    if ( type->kind() != Type::Array )
                    {
                        return false;
                    }
                    auto arrayType = static_cast<const ArrayType*>(type);
                    for ( size_t i = 0; i < value.size(); ++i )
                    {
                        if ( !checkGraphParamType(value[i], arrayType->itemType()) )
                        {
                            return false;
                        }
                    }
                    return true;
                }
                case Value::Tuple:
                {
                    if ( type->kind() != Type::Tuple )
                    {
                        return false;
                    }
                    auto tupleType = static_cast<const TupleType*>(type);
                    for ( size_t i = 0; i < value.size(); ++i )
                    {
                        if ( !checkGraphParamType(value[i], tupleType->itemType(i)) )
                        {
                            return false;
                        }
                    }
                    return true;
                }
                case Value::None:
                {
                    return false;
                }
            }
        }

    private:

        const std::string _stdlib_source;
        const std::set<std::string>& _lowered;
        size_t _flags;
    };
    
}   // namespace nnef


#endif
