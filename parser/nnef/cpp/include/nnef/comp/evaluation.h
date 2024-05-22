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

#ifndef _NNEF_EVALUATION_H_
#define _NNEF_EVALUATION_H_

#include "../common/dictionary.h"
#include "../common/error.h"
#include "../common/value.h"
#include "../common/parser.h"
#include "expression.h"
#include "fragment.h"
#include <cassert>
#include <cmath>
#include <set>


namespace nnef
{

    class Evaluation
    {
        typedef Dictionary<Fragment> Fragments;
        typedef Parser::Callback Callback;

    public:

        Evaluation( const std::vector<Assignment>& assignments, const Fragments& fragments, const std::set<std::string>& lowered )
        : _fragments(fragments), _lowered(lowered)
        {
            for ( auto& assignment : assignments )
            {
                addReservedIdentifiers(assignment.lhs());
            }
        }

    public:
        
        static Value evaluateLvalue( const Expr& expr, const Dictionary<Value>& values, bool fallbackToIds )
        {
            switch ( expr.kind() )
            {
                case Expr::Identifier:
                {
                    auto& identifier = static_cast<const IdentifierExpr&>(expr);
                    
                    auto it = values.find(identifier.name());
                    return it != values.end() ? it->second : (fallbackToIds ? Value::identifier(identifier.name()) : Value::identifier(""));
                }
                case Expr::Array:
                {
                    auto& array = static_cast<const ArrayExpr&>(expr);

                    Value::items_t items(array.size());
                    for ( size_t i = 0; i < array.size(); ++i )
                    {
                        items[i] = evaluateLvalue(array.item(i), values, fallbackToIds);
                    }
                    return Value::array(items);
                }
                case Expr::Tuple:
                {
                    auto& tuple = static_cast<const TupleExpr&>(expr);
                    
                    Value::items_t items(tuple.size());
                    for ( size_t i = 0; i < tuple.size(); ++i )
                    {
                        items[i] = evaluateLvalue(tuple.item(i), values, fallbackToIds);
                    }
                    return Value::tuple(items);
                }
                default:
                {
                    assert(false);
                    return Value::none();
                }
            }
        }
        
        static Value evaluateRvalue( const Expr& expr )
        {
            switch ( expr.kind() )
            {
                case Expr::Literal:
                {
                    return evaluateLiteral(expr);
                }
                case Expr::Array:
                case Expr::Tuple:
                {
                    auto& sequence = static_cast<const ItemExpr&>(expr);
                    
                    Value::items_t items(sequence.size());
                    for ( size_t i = 0; i < sequence.size(); ++i )
                    {
                        items[i] = evaluateRvalue(sequence.item(i));
                    }
                    return expr.kind() == Expr::Array ? Value::array(items) : Value::tuple(items);
                }
                case Expr::Unary:
                {
                    auto& unary = static_cast<const UnaryExpr&>(expr);
                    if ( unary.op() == '-' )
                    {
                        auto arg = evaluateRvalue(unary.right());
                        if ( arg.kind() == Value::Integer )
                        {
                            return Value::integer(-arg.integer());
                        }
                        else if ( arg.kind() == Value::Scalar )
                        {
                            return Value::scalar(-arg.scalar());
                        }
                    }
                }
                default:
                {
                    assert(false);
                    return Value::none();
                }
            }
        }

        void evaluateAssign( const Expr& lhs, const Expr& rhs, Dictionary<Value>& values, Dictionary<Typename>& dtypes,
                            Callback& callback, const PrimitiveType* dtype, const Value& context )
        {
            auto value = evaluate(rhs, values, dtypes, callback, dtype, context);
            assign(lhs, value, values, dtypes, callback);
        }

    private:

        Value evaluate( const Expr& expr, const Dictionary<Value>& values, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, const Value& context = Value::none() )
        {
            switch ( expr.kind() )
            {
                case Expr::Literal:
                {
                    return evaluateLiteral(expr);
                }
                case Expr::Identifier:
                {
                    return evaluate(static_cast<const IdentifierExpr&>(expr), values);
                }
                case Expr::Array:
                {
                    return evaluate(static_cast<const ArrayExpr&>(expr), values, dtypes, callback, dtype, context);
                }
                case Expr::Tuple:
                {
                    return evaluate(static_cast<const TupleExpr&>(expr), values, dtypes, callback, dtype, context);
                }
                case Expr::Subscript:
                {
                    return evaluate(static_cast<const SubscriptExpr&>(expr), values, dtypes, callback, dtype);
                }
                case Expr::Unary:
                {
                    return evaluate(static_cast<const UnaryExpr&>(expr), values, dtypes, callback, dtype);
                }
                case Expr::Binary:
                {
                    return evaluate(static_cast<const BinaryExpr&>(expr), values, dtypes, callback, dtype);
                }
                case Expr::Select:
                {
                    return evaluate(static_cast<const SelectExpr&>(expr), values, dtypes, callback, dtype, context);
                }
                case Expr::Comprehension:
                {
                    return evaluate(static_cast<const ComprehensionExpr&>(expr), values, dtypes, callback, dtype, context);
                }
                case Expr::Builtin:
                {
                    return evaluate(static_cast<const BuiltinExpr&>(expr), values, dtypes, callback, dtype);
                }
                case Expr::Invocation:
                {
                    return evaluate(static_cast<const InvocationExpr&>(expr), values, dtypes, callback, dtype, context);
                }
                default:
                {
                    assert(false);
                    return Value::none();
                }
            }
        }

        static Value evaluateLiteral( const Expr& expr )
        {
            auto type = static_cast<const PrimitiveType&>(*expr.type());
            switch ( type.name() )
            {
                case Typename::Integer:
                {
                    return evaluate(static_cast<const IntegerExpr&>(expr));
                }
                case Typename::Scalar:
                {
                    return evaluate(static_cast<const ScalarExpr&>(expr));
                }
                case Typename::Logical:
                {
                    return evaluate(static_cast<const LogicalExpr&>(expr));
                }
                case Typename::String:
                {
                    return evaluate(static_cast<const StringExpr&>(expr));
                }
                default:
                {
                    assert(false);
                    return Value::none();
                }
            }
        }

        static Value evaluate( const ScalarExpr& scalar )
        {
            return Value::scalar(scalar.value());
        }

        static Value evaluate( const IntegerExpr& integer )
        {
            return Value::integer(integer.value());
        }

        static Value evaluate( const LogicalExpr& logical )
        {
            return Value::logical(logical.value());
        }

        static Value evaluate( const StringExpr& string )
        {
            return Value::string(string.value());
        }

        static Value evaluate( const IdentifierExpr& identifier, const Dictionary<Value>& values )
        {
            if ( !values.count(identifier.name()) )
            {
                throw Error(identifier.position(), "undefined identifier '%s'", identifier.name().c_str());
            }
            return values.at(identifier.name());
        }

        Value evaluate( const ArrayExpr& array, const Dictionary<Value>& values, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, const Value& context )
        {
            Value::items_t items(array.size());
            for ( size_t i = 0; i < array.size(); ++i )
            {
                auto ctx = context.kind() == Value::Array ? context[i] : Value::none();
                items[i] = evaluate(array.item(i), values, dtypes, callback, dtype, ctx);
            }
            return Value::array(items);
        }

        Value evaluate( const TupleExpr& tuple, const Dictionary<Value>& values, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, const Value& context )
        {
            Value::items_t items(tuple.size());
            for ( size_t i = 0; i < tuple.size(); ++i )
            {
                auto ctx = context.kind() == Value::Tuple ? context[i] : Value::none();
                items[i] = evaluate(tuple.item(i), values, dtypes, callback, dtype, ctx);
            }
            return Value::tuple(items);
        }

        Value evaluate( const SubscriptExpr& subscript, const Dictionary<Value>& values, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype )
        {
            Value sequence = evaluate(subscript.sequence(), values, dtypes, callback, dtype);

            if ( subscript.isRange() )
            {
                Value::integer_t i = subscript.begin() ? evaluate(*subscript.begin(), values, dtypes, callback, dtype).integer() : (Value::integer_t)0;
                if ( i < 0 )
                {
                    i += (Value::integer_t)sequence.size();
                }
                if ( i < 0 || i > (Value::integer_t)sequence.size() )
                {
                    throw Error(subscript.position(), "range begin (%d) out of bounds (size = %d)", (int)i, (int)sequence.size());
                }

                Value::integer_t j = subscript.end() ? evaluate(*subscript.end(), values, dtypes, callback, dtype).integer() : (Value::integer_t)sequence.size();
                if ( j < 0 )
                {
                    j += (Value::integer_t)sequence.size();
                }
                if ( j < 0 || j > (Value::integer_t)sequence.size() )
                {
                    throw Error(subscript.position(), "range end (%d) out of bounds (size = %d)", (int)j, (int)sequence.size());
                }

                if ( j < i )
                {
                    throw Error(subscript.position(), "invalid range: %d:%d", (int)i, (int)j);
                }

                if ( sequence.kind() == Value::String )
                {
                    return Value::string(sequence.string().substr(i,j-i));
                }
                else
                {
                    auto it = sequence.items().begin();
                    Value::items_t items(it + i, it + j);
                    return Value::array(items);
                }
            }
            else
            {
                Value::integer_t index = evaluate(*subscript.begin(), values, dtypes, callback, dtype).integer();
                if ( index < 0 )
                {
                    index += (Value::integer_t)sequence.size();
                }
                if ( index < 0 || index >= (Value::integer_t)sequence.size() )
                {
                    throw Error(subscript.position(), "index (%d) out of bounds (size = %d)", (int)index, (int)sequence.size());
                }

                if ( sequence.kind() == Value::String )
                {
                    return Value::string(sequence.string().substr(index,1));
                }
                else
                {
                    return sequence[index];
                }
            }
        }

        Value evaluate( const UnaryExpr& unary, const Dictionary<Value>& values, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype )
        {
            Value right = evaluate(unary.right(), values, dtypes, callback, dtype);

            if ( unary.op() == '!' )
            {
                if ( right.kind() == Value::Logical )
                {
                    return Value::logical(!right.logical());
                }
            }
            else if ( unary.op() == '-' )
            {
                if ( right.kind() == Value::Integer )
                {
                    return Value::integer(-right.integer());
                }
                else if ( right.kind() == Value::Scalar )
                {
                    return Value::scalar(-right.scalar());
                }
            }
            else if ( unary.op() == '+' )
            {
                return right;
            }

            assert(false);
            return Value::none();
        }

        Value evaluate( const BinaryExpr& binary, const Dictionary<Value>& values, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype )
        {
            bool lazy = binary.op() == Lexer::And || binary.op() == Lexer::Or;

            Value left = evaluate(binary.left(), values, dtypes, callback, dtype);
            Value right = lazy ? Value::none() : evaluate(binary.right(), values, dtypes, callback, dtype);

            switch ( binary.op() )
            {
                case '+':
                {
                    if ( left.kind() == Value::String && right.kind() == Value::String )
                    {
                        return Value::string(left.string() + right.string());
                    }
                    else if ( left.kind() == Value::Array && right.kind() == Value::Array )
                    {
                        Value::items_t items = left.array();
                        items.insert(items.end(), right.array().begin(), right.array().end());
                        return Value::array(items);
                    }
                    else
                    {
                        return evaluateBinary<std::plus>(left, right);
                    }
                }
                case '*':
                {
                    if ( left.kind() == Value::String && right.kind() == Value::Integer )
                    {
                        Value::string_t str;
                        for ( size_t i = 0; i < (size_t)right.integer(); ++i )
                        {
                            str += left.string();
                        }
                        return Value::string(str);
                    }
                    else if ( left.kind() == Value::Array && right.kind() == Value::Integer )
                    {
                        Value::items_t items;
                        for ( size_t i = 0; i < (size_t)right.integer(); ++i )
                        {
                            items.insert(items.end(), left.array().begin(), left.array().end());
                        }
                        return Value::array(items);
                    }
                    else
                    {
                        return evaluateBinary<std::multiplies>(left, right);
                    }
                }
                case '-':
                {
                    return evaluateBinary<std::minus>(left, right);
                }
                case '/':
                {
                    return evaluateBinary<std::divides>(left, right);
                }
                case '^':
                {
                    return evaluateBinary<power>(left, right);
                }
                case '<':
                {
                    return evaluateBinary<std::less>(left, right);
                }
                case '>':
                {
                    return evaluateBinary<std::greater>(left, right);
                }
                case Lexer::Le:
                {
                    return evaluateBinary<std::less_equal>(left, right);
                }
                case Lexer::Ge:
                {
                    return evaluateBinary<std::greater_equal>(left, right);
                }
                case Lexer::Eq:
                {
                    return evaluateBinary<std::equal_to>(left, right);
                }
                case Lexer::Ne:
                {
                    return evaluateBinary<std::not_equal_to>(left, right);
                }
                case Lexer::And:
                {
                    return !left.logical() ? left : evaluate(binary.right(), values, dtypes, callback, dtype);
                }
                case Lexer::Or:
                {
                    return left.logical() ? left : evaluate(binary.right(), values, dtypes, callback, dtype);
                }
                case Lexer::In:
                {
                    auto& items = right.array();
                    bool contains = std::find(items.begin(), items.end(), left) != items.end();
                    return Value::logical(contains);
                }
                default:
                {
                    break;
                }
            }

            assert(false);
            return Value::none();
        }

        Value evaluate( const SelectExpr& select, const Dictionary<Value>& values, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, const Value& context )
        {
            Value condition = evaluate(select.condition(), values, dtypes, callback, dtype);
            return condition.logical() ? evaluate(select.trueValue(), values, dtypes, callback, dtype, context) :
                                         evaluate(select.falseValue(), values, dtypes, callback, dtype, context);
        }

        Value evaluate( const ComprehensionExpr& comprehension, const Dictionary<Value>& values, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, const Value& context )
        {
            std::vector<Value> iterables;
            for ( size_t i = 0; i < comprehension.iteratorCount(); ++i )
            {
                auto iterable = evaluate(comprehension.iterable(i), values, dtypes, callback, dtype);
                iterables.push_back(iterable);
            }
            
            const size_t length = iterables.front().size();
            for ( size_t i = 1; i < iterables.size(); ++i )
            {
                if ( iterables[i].size() != length )
                {
                    throw Error(comprehension.position(), "iterables must have the same length in array comprehension");
                }
            }

            Value::items_t items;

            Dictionary<Value> ids = values;
            for ( size_t i = 0; i < length; ++i )
            {
                for ( size_t k = 0; k < iterables.size(); ++k )
                {
                    assign(comprehension.iterator(k), iterables[k][i], ids, dtypes, callback);
                }

                bool accept = comprehension.condition() ? evaluate(*comprehension.condition(), ids, dtypes, callback, dtype).logical() : true;
                if ( accept )
                {
                    auto ctx = context.kind() == Value::Array && items.size() < context.size() ? context[items.size()] : Value::none();
                    auto item = evaluate(comprehension.item(), ids, dtypes, callback, dtype, ctx);
                    items.push_back(item);
                }

                for ( size_t k = 0; k < iterables.size(); ++k )
                {
                    unassign(comprehension.iterator(k), ids);
                }
            }
            return Value::array(items);
        }

        Value evaluate( const InvocationExpr& invocation, const Dictionary<Value>& values, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, const Value& context )
        {
            auto& fragment = _fragments.at(invocation.target());
            auto& proto = fragment.prototype();

            Dictionary<Value> ids;
            for ( size_t i = 0; i < proto.paramCount(); ++i )
            {
                auto& param = proto.param(i);
                auto arg = invocation.arg(param.name());
                ids[param.name()] = arg ? evaluate(*arg, values, dtypes, callback, dtype) : param.defaultValue();
            }

            const PrimitiveType* dataType = invocation.dataType() == primitiveType(Typename::Generic) ? dtype : invocation.dataType();
            if ( dataType )
            {
                ids["?"] = Value::string(dataType->toString());
            }
            
            if ( !invocation.type()->isAttribute() )
            {
                if ( context )
                {
                    checkStructure(context, invocation.type(), invocation.position());
                }
                
                auto& resultType = static_cast<const TupleType&>(*invocation.type());
                if ( proto.resultCount() == 1 )
                {
                    ids[proto.result(0).name()] = getResultValue(context, resultType, proto.name());
                }
                else
                {
                    assert(context.kind() == Value::Tuple);
                    for ( size_t i = 0; i < proto.resultCount(); ++i )
                    {
                        ids[proto.result(i).name()] = getResultValue(context[i], *resultType.itemType(i), proto.name());
                    }
                }
            }
            
            bool lower = fragment.assignmentCount() && _lowered.count(proto.name());
            if ( lower )
            {
                for ( size_t i = 0; i < fragment.assignmentCount(); ++i )
                {
                    auto& assignment = fragment.assignment(i);
                    
                    const Value ctx = evaluateLvalue(assignment.lhs(), ids, false);
                    try
                    {
                        evaluateAssign(assignment.lhs(), assignment.rhs(), ids, dtypes, callback, dataType, ctx);
                    }
                    catch ( const Error& e )
                    {
                        throw Error(chain(e.position(), invocation.position()), e.what());
                    }
                }
            }
            
            Value value;
            if ( proto.resultCount() == 1 )
            {
                value = ids[proto.result(0).name()];
            }
            else
            {
                Value::items_t items(proto.resultCount());
                for ( size_t i = 0; i < proto.resultCount(); ++i )
                {
                    items[i] = ids[proto.result(i).name()];
                }
                value = Value::tuple(items);
            }
            
            if ( hasNone(value) )
            {
                throw Error(invocation.position(), "could not evaluate invocation (possibly unknown result array length)");
            }
            
            if ( !lower )
            {
                declare(value, invocation.type(), dtypes, dtype);
                callback.operation(proto, ids, dtypes);
            }
            
            return value;
        }

        Value evaluate( const BuiltinExpr& builtin, const Dictionary<Value>& values, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype )
        {
            Value arg = evaluate(builtin.arg(), values, dtypes, callback, dtype);

            switch ( builtin.op() )
            {
                case Lexer::LengthOf:
                {
                    auto length = arg.kind() == Value::String ? arg.string().length() : arg.array().size();
                    return Value::integer((Value::integer_t)length);
                }
                case Lexer::RangeOf:
                {
                    auto length = arg.kind() == Value::String ? arg.string().length() : arg.array().size();

                    Value::items_t items(length);
                    for ( size_t i = 0; i < length; ++i )
                    {
                        items[i] = Value::integer((Value::integer_t)i);
                    }
                    return Value::array(items);
                }
                case Lexer::ShapeOf:
                {
                    throw Error(builtin.position(), "the use of operator 'shape_of' is deprecated and is not supported");
                }
                case Lexer::Integer:
                {
                    if ( arg.kind() == Value::Integer )
                    {
                        return arg;
                    }
                    else if ( arg.kind() == Value::Scalar )
                    {
                        return Value::integer((Value::integer_t)arg.scalar());
                    }
                    else if ( arg.kind() == Value::Logical )
                    {
                        return Value::integer((Value::integer_t)arg.logical());
                    }
                    else if ( arg.kind() == Value::String )
                    {
                        char* end;
                        const char* str = arg.string().c_str();

                        auto value = (Value::integer_t)std::strtol(str, &end, 10);
                        if ( end == str )
                        {
                            throw Error(builtin.position(), "cannot convert string '%s' to integer", str);
                        }
                        return Value::integer(value);
                    }
                    break;
                }
                case Lexer::Scalar:
                {
                    if ( arg.kind() == Value::Scalar )
                    {
                        return arg;
                    }
                    else if ( arg.kind() == Value::Integer )
                    {
                        return Value::scalar((Value::scalar_t)arg.integer());
                    }
                    else if ( arg.kind() == Value::Logical )
                    {
                        return Value::scalar((Value::scalar_t)arg.logical());
                    }
                    else if ( arg.kind() == Value::String )
                    {
                        char* end;
                        const char* str = arg.string().c_str();

                        auto value = (Value::scalar_t)std::strtof(str, &end);
                        if ( end == str )
                        {
                            throw Error(builtin.position(), "cannot convert string '%s' to scalar", str);
                        }
                        return Value::scalar(value);
                    }
                    break;
                }
                case Lexer::Logical:
                {
                    if ( arg.kind() == Value::Logical )
                    {
                        return arg;
                    }
                    else if ( arg.kind() == Value::Integer )
                    {
                        return Value::logical(arg.integer() != 0);
                    }
                    else if ( arg.kind() == Value::Scalar )
                    {
                        return Value::logical(arg.scalar() != 0);
                    }
                    else if ( arg.kind() == Value::String )
                    {
                        return Value::logical(!arg.string().empty());
                    }
                    break;
                }
                case Lexer::String:
                {
                    if ( arg.kind() == Value::Logical )
                    {
                        return Value::string(std::to_string(arg.logical()));
                    }
                    else if ( arg.kind() == Value::Integer )
                    {
                        return Value::string(std::to_string(arg.integer()));
                    }
                    else if ( arg.kind() == Value::Scalar )
                    {
                        return Value::string(std::to_string(arg.scalar()));
                    }
                    else if ( arg.kind() == Value::String )
                    {
                        return arg;
                    }
                    break;
                }
                default:
                {
                    break;
                }
            }

            assert(false);
            return Value::none();
        }

    private:

        template<template<typename> class Op>
        static Value evaluateBinary( const Value& left, const Value& right )
        {
            if ( left.kind() == Value::Integer && right.kind() == Value::Integer )
            {
                Op<Value::integer_t> op;
                return Value::make(op(left.integer(), right.integer()));
            }
            else if ( left.kind() == Value::Scalar && right.kind() == Value::Scalar )
            {
                Op<Value::scalar_t> op;
                return Value::make(op(left.scalar(), right.scalar()));
            }
            assert(false);
            return Value::none();
        }
        
        static Typename dtypeOf( const Value& value, const Dictionary<Typename>& dtypes )
        {
            switch ( value.kind() )
            {
                case Value::Scalar:
                    return Typename::Scalar;
                case Value::Integer:
                    return Typename::Integer;
                case Value::Logical:
                    return Typename::Logical;
                case Value::String:
                    return Typename::String;
                case Value::Identifier:
                    return dtypes.at(value.identifier());
                default:
                    assert(false);
                    return Typename::Generic;
            }
        }
        
        void insertCopy( const Value& lvalue, const Value& rvalue, Dictionary<Typename>& dtypes, Callback& callback )
        {
            const Typename dtype = dtypeOf(rvalue, dtypes);
            const Value dvalue = Value::string(toString(dtype));
            
            const Prototype& proto = _fragments.at("copy").prototype();
            const Dictionary<Value> args =
            {
                std::make_pair("x", rvalue),
                std::make_pair("y", lvalue),
                std::make_pair("?", dvalue)
            };
            
            dtypes[lvalue.identifier()] = dtype;
            callback.operation(proto, args, dtypes);
        }

        void assign( const Expr& lhs, const Value& rvalue, Dictionary<Value>& ids, Dictionary<Typename>& dtypes, Callback& callback )
        {
            switch ( lhs.kind() )
            {
                case Expr::Array:
                {
                    auto& array = static_cast<const ArrayExpr&>(lhs);
                    if ( array.size() != rvalue.size() )
                    {
                        throw Error(lhs.position(), "cannot assign array of length %d to array of length %d",
                                    (int)rvalue.size(), (int)array.size());
                    }
                    for ( size_t i = 0; i < array.size(); ++i )
                    {
                        assign(array.item(i), rvalue[i], ids, dtypes, callback);
                    }
                    break;
                }
                case Expr::Tuple:
                {
                    auto& tuple = static_cast<const TupleExpr&>(lhs);
                    assert(tuple.size() == rvalue.size());

                    for ( size_t i = 0; i < tuple.size(); ++i )
                    {
                        assign(tuple.item(i), rvalue[i], ids, dtypes, callback);
                    }
                    break;
                }
                case Expr::Identifier:
                {
                    auto& identifier = static_cast<const IdentifierExpr&>(lhs);
                    auto& lvalue = ids[identifier.name()];

                    if ( lvalue )
                    {
                        if ( lvalue != rvalue )
                        {
                            if ( lvalue.kind() == Value::Array || lvalue.kind() == Value::Tuple )
                            {
                                if ( lvalue.kind() == Value::Array && lvalue.size() != rvalue.size() )
                                {
                                    throw Error(lhs.position(), "cannot assign array of length %d to array of length %d",
                                                (int)rvalue.size(), (int)lvalue.size());
                                }
                                for ( size_t i = 0; i < lvalue.size(); ++i )
                                {
                                    insertCopy(lvalue[i], rvalue[i], dtypes, callback);
                                }
                            }
                            else
                            {
                                assert(lvalue.kind() == Value::Identifier);
                                insertCopy(lvalue, rvalue, dtypes, callback);
                            }
                        }
                    }
                    else
                    {
                        lvalue = rvalue;
                    }

                    break;
                }
                default:
                {
                    assert(false);
                    break;
                }
            }
        }

        void unassign( const Expr& lhs, Dictionary<Value>& ids )
        {
            switch ( lhs.kind() )
            {
                case Expr::Array:
                case Expr::Tuple:
                {
                    auto& items = static_cast<const ItemExpr&>(lhs);
                    for ( size_t i = 0; i < items.size(); ++i )
                    {
                        unassign(items.item(i), ids);
                    }
                    break;
                }
                case Expr::Identifier:
                {
                    auto& identifier = static_cast<const IdentifierExpr&>(lhs);
                    ids.erase(identifier.name());
                    break;
                }
                default:
                {
                    assert(false);
                    break;
                }
            }
        }

        static void declare( const Value& arg, const Type* type, Dictionary<Typename>& dtypes, const PrimitiveType* dtype )
        {
            switch ( arg.kind() )
            {
                case Value::Identifier:
                {
                    assert(type->kind() == Type::Tensor);
                    const std::string& id = arg.identifier();
                    auto tensorType = static_cast<const TensorType*>(type);
                    assert(tensorType->dataType()->kind() == Type::Primitive);
                    auto dataType = static_cast<const PrimitiveType*>(tensorType->dataType());
                    auto name = dataType->name() == Typename::Generic ? dtype->name() : dataType->name();
                    assert(!dtypes.count(id) || dtypes.at(id) == name);
                    dtypes.emplace(id, name);
                    break;
                }
                case Value::Array:
                {
                    assert(type->kind() == Type::Array);
                    auto arrayType = static_cast<const ArrayType*>(type);
                    for ( size_t i = 0; i < arg.size(); ++i )
                    {
                        declare(arg[i], arrayType->itemType(), dtypes, dtype);
                    }
                    break;
                }
                case Value::Tuple:
                {
                    assert(type->kind() == Type::Tuple);
                    auto tupleType = static_cast<const TupleType*>(type);
                    for ( size_t i = 0; i < arg.size(); ++i )
                    {
                        declare(arg[i], tupleType->itemType(i), dtypes, dtype);
                    }
                    break;
                }
                default:
                {
                    break;
                }
            }
        }
        
        static void checkStructure( const Value& value, const Type* type, const Error::Position& position )
        {
            switch ( type->kind() )
            {
                case Type::Primitive:
                case Type::Tensor:
                {
                    if ( value.kind() != Value::Identifier )
                    {
                        throw Error(position, "invocation context mismatch: expected identifier on left hand side to match type '%s'",
                                    type->toString().c_str());
                    }
                    break;
                }
                case Type::Array:
                {
                    if ( value.kind() == Value::Identifier || value.kind() == Value::None )
                    {
                        break;
                    }
                    if ( value.kind() != Value::Array )
                    {
                        throw Error(position, "invocation context mismatch: expected array on left hand side to match type '%s'",
                                    type->toString().c_str());
                    }
                    auto& array = static_cast<const ArrayType&>(*type);
                    for ( size_t i = 0; i < value.size(); ++i )
                    {
                        checkStructure(value[i], array.itemType(), position);
                    }
                    break;
                }
                case Type::Tuple:
                {
                    if ( value.kind() != Value::Tuple )
                    {
                        throw Error(position, "invocation context mismatch: expected tuple on left hand side to match type '%s'",
                                    type->toString().c_str());
                    }
                    auto& tuple = static_cast<const TupleType&>(*type);
                    for ( size_t i = 0; i < value.size(); ++i )
                    {
                        checkStructure(value[i], tuple.itemType(i), position);
                    }
                    break;
                }
            }
        }

    private:

        typedef Error::Position Position;

        Position chain( const Position& position, const Position& origin )
        {
            const Position chained = { position.line, position.column, position.filename, &origin };
            return chained;
        }

    private:

        std::string nextTensorId( const std::string& op )
        {
            return op + std::to_string(++_tensorCounts[op]);
        }

        std::string makeTensorId( const std::string& op )
        {
            std::string id;
            do
            {
                id = nextTensorId(op);
            }
            while ( isReservedId(id) );

            _reservedIds.insert(id);
            return id;
        }

        Value makeResultValue( const std::string& op, size_t idx )
        {
            auto id = makeTensorId(op);
            return Value::identifier(idx ? indexedId(id, idx) : id);
        }
        
        Value getResultValue( const Value& context, const Type& type, const std::string op, size_t idx = 0 )
        {
            if ( !context )
            {
                if ( type.kind() == Type::Array )
                {
                    return Value::none();
                }
                return makeResultValue(op, idx);
            }
            else if ( context.kind() == Value::Identifier )
            {
                if ( type.kind() == Type::Array )
                {
                    return Value::none();
                }
                return context.identifier() != "" ? context : makeResultValue(op, idx);
            }
            else if ( context.kind() == Value::Array )
            {
                std::vector<Value> results(context.size());
                auto& arrayType = static_cast<const ArrayType&>(type);
                for ( size_t i = 0; i < context.size(); ++i )
                {
                    results[i] = getResultValue(context[i], *arrayType.itemType(), op, i + 1);
                }
                return Value::array(results);
            }
            else if ( context.kind() == Value::Tuple )
            {
                std::vector<Value> results(context.size());
                auto& tupleType = static_cast<const TupleType&>(type);
                for ( size_t i = 0; i < context.size(); ++i )
                {
                    results[i] = getResultValue(context[i], *tupleType.itemType(i), op);
                }
                return Value::array(results);
            }
            else
            {
                assert(false);
                return Value();
            }
        }
        
        bool hasNone( const Value& value )
        {
            switch ( value.kind() )
            {
                case Value::None:
                {
                    return true;
                }
                case Value::Tuple:
                case Value::Array:
                {
                    for ( size_t i = 0; i < value.size(); ++i )
                    {
                        if ( hasNone(value[i]) )
                        {
                            return true;
                        }
                    }
                    return false;
                }
                default:
                {
                    return false;
                }
            }
        }

        void addReservedIdentifiers( const Expr& expr )
        {
            switch ( expr.kind() )
            {
                case Expr::Identifier:
                {
                    auto& identifier = static_cast<const IdentifierExpr&>(expr);
                    _reservedIds.insert(identifier.name());
                    break;
                }
                case Expr::Array:
                case Expr::Tuple:
                {
                    auto& items = static_cast<const ItemExpr&>(expr);
                    for ( size_t i = 0; i < items.size(); ++i )
                    {
                        addReservedIdentifiers(items.item(i));
                    }
                    break;
                }
                default:
                {
                    assert(false);
                    break;
                }
            }
        }

        bool isReservedId( const std::string& id )
        {
            return _reservedIds.find(id) != _reservedIds.end();
        }

        bool isReservedId( const std::string& id, const size_t size )
        {
            for ( size_t i = 0; i < size; ++i )
            {
                if ( isReservedId(indexedId(id,i+1)) )
                {
                    return true;
                }
            }
            return false;
        }

        std::string indexedId( const std::string& id, const size_t idx )
        {
            return id + "_" + std::to_string(idx);
        }

    private:

        template<typename T>
        struct power
        {
            T operator()( const T& left, const T& right )
            {
                return (T)std::pow(left, right);
            }
        };

    private:

        const Fragments& _fragments;
        const std::set<std::string>& _lowered;

        Dictionary<size_t> _tensorCounts;
        std::set<std::string> _reservedIds;
    };

}   // namespace nnef


#endif
