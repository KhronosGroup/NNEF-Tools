/*
 * Copyright (c) 2012-2017 The Khronos Group Inc.
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
        typedef Dictionary<Shared<Fragment>> Fragments;
        typedef Parser::Callback Callback;

    public:

        Evaluation( const Fragment& graph, const Fragments& fragments )
        : _fragments(fragments)
        {
            for ( size_t i = 0; i < graph.assignmentCount(); ++i )
            {
                addReservedIdentifiers(graph.assignment(i).lhs());
            }
        }

    public:
        
        static Value evaluateLvalue( const Expr& expr, const Type* type, const Dictionary<Value>* values = nullptr )
        {
            switch ( expr.kind() )
            {
                case Expr::Identifier:
                {
                    if ( !type->isTensor() )
                    {
                        return Value::none();
                    }
                    
                    auto& identifier = dynamic_cast<const IdentifierExpr&>(expr);
                    return values ? (*values)[identifier.name()] : Value::tensor({ identifier.name() });
                }
                case Expr::Array:
                {
                    if ( !type->isArray() )
                    {
                        return Value::none();
                    }
                    
                    auto& array = dynamic_cast<const ArrayExpr&>(expr);
                    auto arrayType = dynamic_cast<const ArrayType*>(type);
                    
                    Value::items_t items(array.size());
                    for ( size_t i = 0; i < array.size(); ++i )
                    {
                        items[i] = evaluateLvalue(array.item(i), arrayType->itemType(), values);
                    }
                    return Value::array(items);
                }
                case Expr::Tuple:
                {
                    if ( !type->isTuple() )
                    {
                        return Value::none();
                    }
                    
                    auto& tuple = dynamic_cast<const TupleExpr&>(expr);
                    auto tupleType = dynamic_cast<const TupleType*>(type);
                    
                    Value::items_t items(tuple.size());
                    for ( size_t i = 0; i < tuple.size(); ++i )
                    {
                        items[i] = evaluateLvalue(tuple.item(i), tupleType->itemType(i), values);
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

        void evaluateAssign( const Expr& lhs, const Expr& rhs, Dictionary<Value>& values, Dictionary<Shape>& shapes,
                            Callback& callback, const Value& context )
        {
            assign(lhs, evaluate(rhs, values, shapes, callback, context), values);
        }

    private:

        Value evaluate( const Expr& expr, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Callback& callback,
                       const Value& context = Value::none() )
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
                    return evaluate(dynamic_cast<const ArrayExpr&>(expr), values, shapes, callback);
                }
                case Expr::Tuple:
                {
                    return evaluate(dynamic_cast<const TupleExpr&>(expr), values, shapes, callback);
                }
                case Expr::Subscript:
                {
                    return evaluate(dynamic_cast<const SubscriptExpr&>(expr), values, shapes, callback);
                }
                case Expr::Unary:
                {
                    return evaluate(dynamic_cast<const UnaryExpr&>(expr), values, shapes, callback);
                }
                case Expr::Binary:
                {
                    return evaluate(dynamic_cast<const BinaryExpr&>(expr), values, shapes, callback);
                }
                case Expr::Select:
                {
                    return evaluate(dynamic_cast<const SelectExpr&>(expr), values, shapes, callback);
                }
                case Expr::Comprehension:
                {
                    return evaluate(dynamic_cast<const ComprehensionExpr&>(expr), values, shapes, callback);
                }
                case Expr::Builtin:
                {
                    return evaluate(dynamic_cast<const BuiltinExpr&>(expr), values, shapes, callback);
                }
                case Expr::Invocation:
                {
                    return evaluate(dynamic_cast<const InvocationExpr&>(expr), values, shapes, callback, context);
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
                case Typename::Extent:
                    return evaluate(dynamic_cast<const ExtentExpr&>(expr));
                case Typename::Scalar:
                    return evaluate(dynamic_cast<const ScalarExpr&>(expr));
                case Typename::Logical:
                    return evaluate(dynamic_cast<const LogicalExpr&>(expr));
                case Typename::String:
                    return evaluate(dynamic_cast<const StringExpr&>(expr));
            }
        }

        static Value evaluate( const ScalarExpr& scalar )
        {
            return Value::scalar(scalar.value());
        }

        static Value evaluate( const ExtentExpr& extent )
        {
            return Value::integer(extent.value());
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

        Value evaluate( const ArrayExpr& array, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Callback& callback )
        {
            Value::items_t items(array.size());
            for ( size_t i = 0; i < array.size(); ++i )
            {
                items[i] = evaluate(array.item(i), values, shapes, callback);
            }
            return Value::array(items);
        }

        Value evaluate( const TupleExpr& tuple, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Callback& callback )
        {
            Value::items_t items(tuple.size());
            for ( size_t i = 0; i < tuple.size(); ++i )
            {
                items[i] = evaluate(tuple.item(i), values, shapes, callback);
            }
            return Value::tuple(items);
        }

        Value evaluate( const SubscriptExpr& subscript, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Callback& callback )
        {
            const Value sequence = evaluate(subscript.sequence(), values, shapes, callback);

            if ( subscript.isRange() )
            {
                Value::integer_t i = subscript.begin() ? evaluate(*subscript.begin(), values, shapes, callback).integer() : (Value::integer_t)0;
                if ( i < 0 )
                {
                    i += sequence.size();
                }
                if ( i < 0 || i > sequence.size() )
                {
                    throw Error(subscript.position(), "range begin (%d) out of bounds (size = %d)", (int)i, (int)sequence.size());
                }

                Value::integer_t j = subscript.end() ? evaluate(*subscript.end(), values, shapes, callback).integer() : (Value::integer_t)sequence.size();
                if ( j < 0 )
                {
                    j += sequence.size();
                }
                if ( j < 0 || j > sequence.size() )
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
                Value::integer_t index = evaluate(*subscript.begin(), values, shapes, callback).integer();
                if ( index < 0 )
                {
                    index += sequence.size();
                }
                if ( index < 0 || index >= sequence.size() )
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

        Value evaluate( const UnaryExpr& unary, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Callback& callback )
        {
            const Value right = evaluate(unary.right(), values, shapes, callback);

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

        Value evaluate( const BinaryExpr& binary, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Callback& callback )
        {
            bool lazy = binary.op() == Lexer::And || binary.op() == Lexer::Or;

            const Value left = evaluate(binary.left(), values, shapes, callback);
            Value right = lazy ? Value::none() : evaluate(binary.right(), values, shapes, callback);

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
                        for ( size_t i = 0; i < right.integer(); ++i )
                        {
                            str += left.string();
                        }
                        return Value::string(str);
                    }
                    else if ( left.kind() == Value::Array && right.kind() == Value::Integer )
                    {
                        Value::items_t items;
                        for ( size_t i = 0; i < right.integer(); ++i )
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
                    return !left.logical() ? left : evaluate(binary.right(), values, shapes, callback);
                }
                case Lexer::Or:
                {
                    return left.logical() ? left : evaluate(binary.right(), values, shapes, callback);
                }
                case Lexer::In:
                {
                    auto& items = left.array();
                    bool contains = std::find(items.begin(), items.end(), right) != items.end();
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

        Value evaluate( const SelectExpr& select, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Callback& callback )
        {
            Value condition = evaluate(select.condition(), values, shapes, callback);
            return condition.logical() ? evaluate(select.trueValue(), values, shapes, callback) : evaluate(select.falseValue(), values, shapes, callback);
        }

        Value evaluate( const ComprehensionExpr& comprehension, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Callback& callback )
        {
            const std::string& iterator = dynamic_cast<const IdentifierExpr&>(comprehension.iterator()).name();
            const Value iterable = evaluate(comprehension.iterable(), values, shapes, callback);

            Value::items_t items(iterable.size());

            Dictionary<Value> ids = values;
            for ( size_t i = 0; i < iterable.size(); ++i )
            {
                ids[iterator] = iterable[i];

                if ( comprehension.condition() )
                {
                    const Value condition = evaluate(*comprehension.condition(), ids, shapes, callback);
                    if ( !condition.logical() )
                    {
                        continue;
                    }
                }

                items[i] = evaluate(comprehension.item(), ids, shapes, callback);
            }
            return Value::array(items);
        }

        Value evaluate( const InvocationExpr& invocation, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Callback& callback,
                       const Value& context )
        {
            auto& fragment = *_fragments[invocation.target()];
            auto& proto = fragment.prototype();

            Dictionary<Value> ids;
            for ( size_t i = 0; i < proto.paramCount(); ++i )
            {
                auto& param = proto.param(i);
                auto arg = invocation.arg(param.name());
                ids[param.name()] = arg ? evaluate(arg->value(), values, shapes, callback) : param.defaultValue();
            }

            bool atomic = fragment.assignmentCount() == 0 || callback.isAtomic(proto, ids);

            for ( size_t i = 0; i < proto.resultCount(); ++i )
            {
                const Value hint = proto.resultCount() != 1 && context.kind() == Value::Tuple ? context[i] : context;

                auto& result = proto.result(i);
                if ( result.type()->isArray() )
                {
                    const size_t length = callback.resultArrayLength(proto, ids, i);
                    ids[result.name()] = makeResultValue(proto.name(), length, hint);
                }
                else
                {
                    ids[result.name()] = makeResultValue(proto.name(), hint);
                }
            }

            try
            {
                bool propagated = callback.propagate(proto, ids, shapes);
                if ( atomic )
                {
                    if ( !propagated )
                    {
                        throw Error("shape propagation not defined for operation '%s'", proto.name().c_str());
                    }
                    callback.operation(proto, ids, shapes);
                }
            }
            catch ( Error e )
            {
                throw Error(invocation.position(), e.what());
            }

            if ( !atomic )
            {
                const Dictionary<Value> res = makeResultDict(proto, context);
                for ( size_t i = 0; i < fragment.assignmentCount(); ++i )
                {
                    auto& assignment = fragment.assignment(i);
                    
                    const Value cntxt = evaluateLvalue(assignment.lhs(), assignment.rhs().type(), &res);
                    try
                    {
                        evaluateAssign(assignment.lhs(), assignment.rhs(), ids, shapes, callback, cntxt);
                    }
                    catch ( Error e )
                    {
                        throw Error(chain(e.position(), invocation.position()), e.what());
                    }
                }
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

        Value evaluate( const BuiltinExpr& builtin, const Dictionary<Value>& values, Dictionary<Shape>& shapes, Callback& callback )
        {
            const Value arg = evaluate(builtin.arg(), values, shapes, callback);

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
                    auto& shape = arg.kind() == Value::Tensor ? shapes[arg.tensor().id] : Shape::singleton();

                    auto length = std::max(shape.rank(), (size_t)2);

                    std::vector<Value> items(length);
                    for ( size_t i = 0; i < items.size(); ++i )
                    {
                        items[i] = Value::integer(shape[i]);
                    }
                    return Value::array(items);
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

        static void assign( const Expr& lhs, const Value& rhs, Dictionary<Value>& ids )
        {
            switch ( lhs.kind() )
            {
                case Expr::Array:
                {
                    auto& array = dynamic_cast<const ArrayExpr&>(lhs);
                    if ( array.size() != rhs.size() )
                    {
                        throw Error(lhs.position(), "cannot assign array of length %d to array of length %d",
                                    (int)rhs.size(), (int)array.size());
                    }
                    for ( size_t i = 0; i < array.size(); ++i )
                    {
                        assign(array.item(i), rhs[i], ids);
                    }
                    break;
                }
                case Expr::Tuple:
                {
                    auto& tuple = dynamic_cast<const TupleExpr&>(lhs);
                    assert(tuple.size() == rhs.size());
                    for ( size_t i = 0; i < tuple.size(); ++i )
                    {
                        assign(tuple.item(i), rhs[i], ids);
                    }
                    break;
                }
                case Expr::Identifier:
                {
                    auto& identifier = dynamic_cast<const IdentifierExpr&>(lhs);
                    ids[identifier.name()] = rhs;

                    break;
                }
                default:
                {
                    assert(false);
                    break;
                }
            }
        }

    private:

        typedef Error::Position Position;

        Position chain( const Position& position, const Position& origin )
        {
            return (Position){ position.line, position.column, &origin };
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

        Value makeResultValue( const std::string& op, const Value& hint )
        {
            if ( hint )
            {
                assert(isReservedId(hint.tensor().id));
                return hint;
            }
            return Value::tensor({ makeTensorId(op) });
        }
        
        Value makeResultValue( const std::string& op, const size_t size, const Value& hint )
        {
            const std::string id = hint && hint.kind() == Value::Tensor && !isReservedId(hint.tensor().id, size) ? hint.tensor().id : makeTensorId(op);

            Value::items_t items(size);
            for ( size_t i = 0; i < size; ++i )
            {
                if ( hint && hint.kind() != Value::Tensor )
                {
                    assert(isReservedId(hint[i].tensor().id));
                    items[i] = hint[i];
                }
                else
                {
                    assert(!hint || isReservedId(id));
                    items[i] = Value::tensor({ indexedId(id,i+1) });
                }
            }
            return Value::array(items);
        }

        Dictionary<Value> makeResultDict( const Prototype& proto, const Value& context )
        {
            Dictionary<Value> results;
            if ( context )
            {
                if ( proto.resultCount() == 1 )
                {
                    results[proto.result(0).name()] = context;
                }
                else
                {
                    for ( size_t i = 0; i < proto.resultCount(); ++i )
                    {
                        results[proto.result(i).name()] = context[i];
                    }
                }
            }
            return results;
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

        Dictionary<size_t> _tensorCounts;
        std::set<std::string> _reservedIds;
    };

}   // namespace nnef


#endif
