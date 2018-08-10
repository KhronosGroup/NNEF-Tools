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

#include "../common/propagation.h"
#include "../common/dictionary.h"
#include "../common/error.h"
#include "../common/value.h"
#include "../common/parser.h"
#include "../common/shape.h"
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

        Evaluation( const std::vector<Assignment>& assignments, const Fragments& fragments, Propagation& propagation, bool deferShapeOf )
        : _fragments(fragments), _propagation(propagation), _deferShapeOf(deferShapeOf)
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
                    auto& identifier = dynamic_cast<const IdentifierExpr&>(expr);
                    
                    auto value = values[identifier.name()];
                    if ( !value && fallbackToIds )
                    {
                        value = Value::identifier(identifier.name());
                    }
                    return value;
                }
                case Expr::Array:
                {
                    auto& array = dynamic_cast<const ArrayExpr&>(expr);

                    Value::items_t items(array.size());
                    for ( size_t i = 0; i < array.size(); ++i )
                    {
                        items[i] = evaluateLvalue(array.item(i), values, fallbackToIds);
                    }
                    return Value::array(items);
                }
                case Expr::Tuple:
                {
                    auto& tuple = dynamic_cast<const TupleExpr&>(expr);
                    
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
                    auto& sequence = dynamic_cast<const ItemExpr&>(expr);
                    
                    Value::items_t items(sequence.size());
                    for ( size_t i = 0; i < sequence.size(); ++i )
                    {
                        items[i] = evaluateRvalue(sequence.item(i));
                    }
                    return expr.kind() == Expr::Array ? Value::array(items) : Value::tuple(items);
                }
                case Expr::Unary:
                {
                    auto& unary = dynamic_cast<const UnaryExpr&>(expr);
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

        void evaluateAssign( const Expr& lhs, const Expr& rhs, Dictionary<Value>& values, Dictionary<Shape>& shapes, Dictionary<Typename>& dtypes,
                            Callback& callback, const PrimitiveType* dtype, bool silent, const Value& context )
        {
            auto value = evaluate(rhs, values, shapes, dtypes, callback, dtype, silent, context);
            assign(lhs, value, values, shapes, dtypes, callback, silent);
            declare(value, rhs.type(), dtypes, dtype);
        }

    private:

        Value evaluate( const Expr& expr, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, bool silent, const Value& context = Value::none() )
        {
            switch ( expr.kind() )
            {
                case Expr::Literal:
                {
                    return evaluateLiteral(expr);
                }
                case Expr::Identifier:
                {
                    return evaluate(dynamic_cast<const IdentifierExpr&>(expr), values);
                }
                case Expr::Array:
                {
                    return evaluate(dynamic_cast<const ArrayExpr&>(expr), values, shapes, dtypes, callback, dtype, silent, context);
                }
                case Expr::Tuple:
                {
                    return evaluate(dynamic_cast<const TupleExpr&>(expr), values, shapes, dtypes, callback, dtype, silent, context);
                }
                case Expr::Subscript:
                {
                    return evaluate(dynamic_cast<const SubscriptExpr&>(expr), values, shapes, dtypes, callback, dtype, silent);
                }
                case Expr::Unary:
                {
                    return evaluate(dynamic_cast<const UnaryExpr&>(expr), values, shapes, dtypes, callback, dtype, silent);
                }
                case Expr::Binary:
                {
                    return evaluate(dynamic_cast<const BinaryExpr&>(expr), values, shapes, dtypes, callback, dtype, silent);
                }
                case Expr::Select:
                {
                    return evaluate(dynamic_cast<const SelectExpr&>(expr), values, shapes, dtypes, callback, dtype, silent, context);
                }
                case Expr::Comprehension:
                {
                    return evaluate(dynamic_cast<const ComprehensionExpr&>(expr), values, shapes, dtypes, callback, dtype, silent, context);
                }
                case Expr::Builtin:
                {
                    return evaluate(dynamic_cast<const BuiltinExpr&>(expr), values, shapes, dtypes, callback, dtype, silent);
                }
                case Expr::Invocation:
                {
                    return evaluate(dynamic_cast<const InvocationExpr&>(expr), values, shapes, dtypes, callback, dtype, silent, context);
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
            auto type = dynamic_cast<const PrimitiveType&>(*expr.type());
            switch ( type.name() )
            {
                case Typename::Integer:
                    return evaluate(dynamic_cast<const IntegerExpr&>(expr));
                case Typename::Scalar:
                    return evaluate(dynamic_cast<const ScalarExpr&>(expr));
                case Typename::Logical:
                    return evaluate(dynamic_cast<const LogicalExpr&>(expr));
                case Typename::String:
                    return evaluate(dynamic_cast<const StringExpr&>(expr));
                case Typename::Generic:
                    assert(false);
                    return Value::none();
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
            if ( !values.contains(identifier.name()) )
            {
                throw Error(identifier.position(), "undefined identifier '%s'", identifier.name().c_str());
            }
            return values[identifier.name()];
        }

        Value evaluate( const ArrayExpr& array, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, bool silent, const Value& context )
        {
            Value::items_t items(array.size());
            for ( size_t i = 0; i < array.size(); ++i )
            {
                auto ctx = context.kind() == Value::Array ? context[i] : Value::none();
                items[i] = evaluate(array.item(i), values, shapes, dtypes, callback, dtype, silent, ctx);
            }
            return Value::array(items);
        }

        Value evaluate( const TupleExpr& tuple, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, bool silent, const Value& context )
        {
            Value::items_t items(tuple.size());
            for ( size_t i = 0; i < tuple.size(); ++i )
            {
                auto ctx = context.kind() == Value::Tuple ? context[i] : Value::none();
                items[i] = evaluate(tuple.item(i), values, shapes, dtypes, callback, dtype, silent, ctx);
            }
            return Value::tuple(items);
        }

        Value evaluate( const SubscriptExpr& subscript, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, bool silent )
        {
            Value sequence = evaluate(subscript.sequence(), values, shapes, dtypes, callback, dtype, silent);

            sequence = evaluateShapeOf(sequence, shapes);

            if ( subscript.isRange() )
            {
                Value::integer_t i = subscript.begin() ? evaluate(*subscript.begin(), values, shapes, dtypes, callback, dtype, silent).integer() : (Value::integer_t)0;
                if ( i < 0 )
                {
                    i += sequence.size();
                }
                if ( i < 0 || i > (Value::integer_t)sequence.size() )
                {
                    throw Error(subscript.position(), "range begin (%d) out of bounds (size = %d)", (int)i, (int)sequence.size());
                }

                Value::integer_t j = subscript.end() ? evaluate(*subscript.end(), values, shapes, dtypes, callback, dtype, silent).integer() : (Value::integer_t)sequence.size();
                if ( j < 0 )
                {
                    j += sequence.size();
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
                    auto it = sequence.array().begin();
                    Value::items_t items(it + i, it + j);
                    return Value::array(items);
                }
            }
            else
            {
                Value::integer_t index = evaluate(*subscript.begin(), values, shapes, dtypes, callback, dtype, silent).integer();
                if ( index < 0 )
                {
                    index += sequence.size();
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

        Value evaluate( const UnaryExpr& unary, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, bool silent )
        {
            Value right = evaluate(unary.right(), values, shapes, dtypes, callback, dtype, silent);

            right = evaluateShapeOf(right, shapes);

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

        Value evaluate( const BinaryExpr& binary, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, bool silent )
        {
            bool lazy = binary.op() == Lexer::And || binary.op() == Lexer::Or;

            Value left = evaluate(binary.left(), values, shapes, dtypes, callback, dtype, silent);
            Value right = lazy ? Value::none() : evaluate(binary.right(), values, shapes, dtypes, callback, dtype, silent);

            left = evaluateShapeOf(left, shapes);
            right = evaluateShapeOf(right, shapes);

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
                    return !left.logical() ? left : evaluate(binary.right(), values, shapes, dtypes, callback, dtype, silent);
                }
                case Lexer::Or:
                {
                    return left.logical() ? left : evaluate(binary.right(), values, shapes, dtypes, callback, dtype, silent);
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

        Value evaluate( const SelectExpr& select, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, bool silent, const Value& context )
        {
            Value condition = evaluate(select.condition(), values, shapes, dtypes, callback, dtype, silent);
            return condition.logical() ? evaluate(select.trueValue(), values, shapes, dtypes, callback, dtype, silent, context) :
                                         evaluate(select.falseValue(), values, shapes, dtypes, callback, dtype, silent, context);
        }

        Value evaluate( const ComprehensionExpr& comprehension, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, bool silent, const Value& context )
        {
            std::vector<Value> iterables;
            for ( size_t i = 0; i < comprehension.iteratorCount(); ++i )
            {
                auto iterable = evaluate(comprehension.iterable(i), values, shapes, dtypes, callback, dtype, silent);

                iterable = evaluateShapeOf(iterable, shapes);

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
                    assign(comprehension.iterator(k), iterables[k][i], ids, shapes, dtypes, callback, silent);
                }

                bool accept = comprehension.condition() ? evaluate(*comprehension.condition(), ids, shapes, dtypes, callback, dtype, silent).logical() : true;
                if ( accept )
                {
                    auto ctx = context.kind() == Value::Array && items.size() < context.size() ? context[items.size()] : Value::none();
                    auto item = evaluate(comprehension.item(), ids, shapes, dtypes, callback, dtype, silent, ctx);
                    items.push_back(item);
                }

                for ( size_t k = 0; k < iterables.size(); ++k )
                {
                    unassign(comprehension.iterator(k), ids);
                }
            }
            return Value::array(items);
        }

        Value evaluate( const InvocationExpr& invocation, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, bool silent, const Value& context )
        {
            auto& fragment = _fragments.at(invocation.target());
            auto& proto = fragment.prototype();

            Dictionary<Value> ids;
            for ( size_t i = 0; i < proto.paramCount(); ++i )
            {
                auto& param = proto.param(i);
                auto arg = invocation.arg(param.name());
                ids[param.name()] = arg ? evaluate(*arg, values, shapes, dtypes, callback, dtype, silent) : param.defaultValue();
            }

            const PrimitiveType* dataType = invocation.dataType() == primitiveType(Typename::Generic) ? dtype : invocation.dataType();
            if ( dataType )
            {
                ids["?"] = Value::string(dataType->toString());
            }
            
            bool atomic = fragment.assignmentCount() == 0 || callback.isAtomic(proto, ids);
            
            if ( atomic )
            {
                evaluateShapeOf(proto, ids, shapes);
            }

            if ( fragment.assignmentCount() == 0 )
            {
                for ( size_t i = 0; i < proto.resultCount(); ++i )
                {
                    const Value hint = context.kind() == Value::Tuple && context.size() == proto.resultCount() ? context[i] : context;

                    auto& result = proto.result(i);
                    if ( result.type()->kind() == Type::Array )
                    {
                        const size_t length = _propagation.resultArrayLength(proto, result.name(), ids, shapes);
                        if ( hint.kind() == Value::Array && hint.size() == length )
                        {
                            ids[result.name()] = hint;
                        }
                        else if ( hint.kind() == Value::Identifier )
                        {
                            ids[result.name()] = makeResultValue(proto.name(), length, hint.identifier());
                        }
                        else
                        {
                            ids[result.name()] = makeResultValue(proto.name(), length);
                        }
                    }
                    else
                    {
                        ids[result.name()] = hint.kind() == Value::Identifier ? hint : makeResultValue(proto.name());
                    }
                }

                try
                {
                    _propagation.propagateShapes(proto, ids, shapes);
                }
                catch ( Error e )
                {
                    throw Error(invocation.position(), e.what());
                }
            }
            else
            {
                for ( size_t i = 0; i < proto.resultCount(); ++i )
                {
                    auto& result = proto.result(i);
                    if ( !result.type()->isAttribute() )
                    {
                        const Value hint = context.kind() == Value::Tuple && context.size() == proto.resultCount() ? context[i] : context;
                        if ( hint )
                        {
                            bool hintMatchesType = (result.type()->kind() == Type::Array && hint.kind() == Value::Array)
                                                || (result.type()->kind() == Type::Tuple && hint.kind() == Value::Tuple)
                                                || (result.type()->kind() == Type::Tensor && hint.kind() == Value::Identifier);
                            if ( hintMatchesType )
                            {
                                ids[result.name()] = hint;
                            }
                        }
                        else if ( atomic )
                        {
                            ids[result.name()] = makeResultValue(proto.name());
                        }
                    }
                }

                for ( size_t i = 0; i < fragment.assignmentCount(); ++i )
                {
                    auto& assignment = fragment.assignment(i);

                    const Value ctx = evaluateLvalue(assignment.lhs(), ids, false);
                    try
                    {
                        evaluateAssign(assignment.lhs(), assignment.rhs(), ids, shapes, dtypes, callback, dataType, silent || atomic, ctx);
                    }
                    catch ( Error e )
                    {
                        throw Error(chain(e.position(), invocation.position()), e.what());
                    }
                }
            }

            if ( atomic && !silent )
            {
                callback.operation(proto, ids, dtypes, shapes);
            }

            if ( proto.resultCount() == 1 )
            {
                return ids[proto.result(0).name()];
            }
            Value::items_t items(proto.resultCount());
            for ( size_t i = 0; i < proto.resultCount(); ++i )
            {
                items[i] = ids[proto.result(i).name()];
            }
            return Value::tuple(items);
        }

        Value evaluate( const BuiltinExpr& builtin, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Dictionary<Typename>& dtypes,
                       Callback& callback, const PrimitiveType* dtype, bool silent )
        {
            Value arg = evaluate(builtin.arg(), values, shapes, dtypes, callback, dtype, silent);

            arg = evaluateShapeOf(arg, shapes);

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
                    if ( arg.kind() != Value::Identifier )
                    {
                        std::vector<Value> items;
                        return Value::array(items);
                    }
                    else if ( _deferShapeOf )
                    {
                        return Value::shape_of(arg.identifier());
                    }
                    else
                    {
                        auto& shape = shapes[arg.identifier()];

                        std::vector<Value> items(shape.rank());
                        for ( size_t i = 0; i < items.size(); ++i )
                        {
                            items[i] = Value::integer(shape[i]);
                        }
                        return Value::array(items);
                    }
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
                        return Value::logical((Value::logical_t)arg.integer());
                    }
                    else if ( arg.kind() == Value::Scalar )
                    {
                        return Value::logical((Value::logical_t)arg.scalar());
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

        void assign( const Expr& lhs, const Value& rvalue, Dictionary<Value>& ids, Dictionary<Shape>& shapes, const Dictionary<Typename>& dtypes,
                    Callback& callback, bool silent = false )
        {
            switch ( lhs.kind() )
            {
                case Expr::Array:
                {
                    auto& array = dynamic_cast<const ArrayExpr&>(lhs);
                    if ( array.size() != rvalue.size() )
                    {
                        throw Error(lhs.position(), "cannot assign array of length %d to array of length %d",
                                    (int)rvalue.size(), (int)array.size());
                    }
                    for ( size_t i = 0; i < array.size(); ++i )
                    {
                        assign(array.item(i), rvalue[i], ids, shapes, dtypes, callback, silent);
                    }
                    break;
                }
                case Expr::Tuple:
                {
                    auto& tuple = dynamic_cast<const TupleExpr&>(lhs);
                    assert(tuple.size() == rvalue.size());

                    for ( size_t i = 0; i < tuple.size(); ++i )
                    {
                        assign(tuple.item(i), rvalue[i], ids, shapes, dtypes, callback, silent);
                    }
                    break;
                }
                case Expr::Identifier:
                {
                    auto& identifier = dynamic_cast<const IdentifierExpr&>(lhs);
                    auto& lvalue = ids[identifier.name()];

                    if ( lvalue )
                    {
                        if ( lvalue != rvalue && !silent )
                        {
                            assert(lvalue.kind() == Value::Identifier);
                            assert(rvalue.kind() == Value::Identifier);

                            const Typename dtype = dtypes[rvalue.identifier()];
                            const Value dvalue = Value::string(toString(dtype));

                            const Prototype& proto = _fragments.at("copy").prototype();
                            const Dictionary<Value> args = { std::make_pair("x", rvalue), std::make_pair("y", lvalue), std::make_pair("?", dvalue) };

                            _propagation.propagateShapes(proto, args, shapes);
                            callback.operation(proto, args, dtypes, shapes);
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
                    auto& items = dynamic_cast<const ItemExpr&>(lhs);
                    for ( size_t i = 0; i < items.size(); ++i )
                    {
                        unassign(items.item(i), ids);
                    }
                    break;
                }
                case Expr::Identifier:
                {
                    auto& identifier = dynamic_cast<const IdentifierExpr&>(lhs);
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
                    auto tensorType = dynamic_cast<const TensorType*>(type);
                    assert(tensorType->dataType()->kind() == Type::Primitive);
                    auto dataType = dynamic_cast<const PrimitiveType*>(tensorType->dataType());
                    auto name = dataType->name() == Typename::Generic ? dtype->name() : dataType->name();
                    assert(!dtypes.contains(id) || dtypes[id] == name);
                    dtypes.emplace(id, name);
                    break;
                }
                case Value::Array:
                {
                    assert(type->kind() == Type::Array);
                    auto arrayType = dynamic_cast<const ArrayType*>(type);
                    for ( size_t i = 0; i < arg.size(); ++i )
                    {
                        declare(arg[i], arrayType->itemType(), dtypes, dtype);
                    }
                    break;
                }
                case Value::Tuple:
                {
                    assert(type->kind() == Type::Tuple);
                    auto tupleType = dynamic_cast<const TupleType*>(type);
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

    private:

        void evaluateShapeOf( const Prototype& proto, Dictionary<Value>& args, const Dictionary<Shape>& shapes )
        {
            for ( auto it = args.begin(); it != args.end(); ++it )
            {
                if ( it->second.kind() == Value::ShapeOf && !_propagation.shouldDeferShapeOf(proto, it->first) )
                {
                    it->second = makeShapeValue(shapes[it->second.shape_of().id]);
                }
            }
        }

        static Value evaluateShapeOf( const Value& value, const Dictionary<Shape>& shapes )
        {
            return value.kind() == Value::ShapeOf ? makeShapeValue(shapes[value.shape_of().id]) : value;
        }

        static Value makeShapeValue( const Shape& shape )
        {
            Value::items_t items(shape.rank());
            for ( size_t i = 0; i < items.size(); ++i )
            {
                items[i] = Value::integer((Value::integer_t)shape[i]);
            }
            return Value::array(items);
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

        Value makeResultValue( const std::string& op )
        {
            return Value::identifier(makeTensorId(op));
        }

        Value makeResultValue( const std::string& op, const size_t size )
        {
            return makeResultValue(op, size, makeTensorId(op));
        }
        
        Value makeResultValue( const std::string& op, const size_t size, const std::string& id )
        {
            Value::items_t items(size);
            for ( size_t i = 0; i < size; ++i )
            {
                items[i] = Value::identifier(indexedId(id,i+1));
            }
            return Value::array(items);
        }

        void addReservedIdentifiers( const Expr& expr )
        {
            switch ( expr.kind() )
            {
                case Expr::Identifier:
                {
                    auto& identifier = dynamic_cast<const IdentifierExpr&>(expr);
                    _reservedIds.insert(identifier.name());
                    break;
                }
                case Expr::Array:
                case Expr::Tuple:
                {
                    auto& items = dynamic_cast<const ItemExpr&>(expr);
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
        Propagation& _propagation;
        const bool _deferShapeOf;

        Dictionary<size_t> _tensorCounts;
        std::set<std::string> _reservedIds;
    };

}   // namespace nnef


#endif
