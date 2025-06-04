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

#ifndef _SKND_SIMPLIFICATION_H_
#define _SKND_SIMPLIFICATION_H_

#include "valuexpr.h"
#include "polynom.h"


namespace sknd
{

    class Simplification
    {
        typedef wchar_t symbol_type;
        
        template<typename T>
        using Poly = polynom<T,symbol_type>;
        
        template<typename T>
        using Dict = std::map<std::string,T>;
        
        struct PolynomContext
        {
            std::map<std::string,symbol_type> symbols;
            std::vector<ValueExpr> exprs;
        };
        
    public:
        
        static void simplify( ValueExpr& expr )
        {
            simplify_heuristic(expr);
            simplify_polynomial(expr);
        }
        
        static ValueExpr simplified( const ValueExpr& expr )
        {
            auto simplified = expr;
            simplify(simplified);
            return simplified;
        }
        
        static ValueExpr simplified( ValueExpr&& expr )
        {
            simplify(expr);
            return expr;
        }
        
        static void resolve( ValueExpr& expr )
        {
            preorder_traverse(expr, []( ValueExpr& x )
            {
                if ( x.is_size_access() )
                {
                    auto& access = x.as_size_access();
                    x = access.pack.canonic_size();
                }
                else if ( x.is_shape_access() )
                {
                    auto& access = x.as_shape_access();
                    if ( access.dim.is_literal() )
                    {
                        auto tensor = access.tensor;
                        if ( access.item == nullptr )
                        {
                            x = tensor.canonic_shape()[access.dim.as_int()];
                        }
                        else if ( access.item.is_literal() )
                        {
                            x = tensor[access.item.as_int()].canonic_shape[access.dim.as_int()];
                        }
                        if ( tensor.packed() && !x.packed() )
                        {
                            x = ValueExpr::uniform(x, tensor.size(), tensor.max_size());
                        }
                    }
                }
                else if ( x.is_reference() )
                {
                    auto& reference = x.as_reference();
                    x = *reference.target;
                    resolve(x);
                }
            });
        }
        
        static ValueExpr resolved( const ValueExpr& expr )
        {
            auto resolved = expr;
            resolve(resolved);
            return resolved;
        }
        
        static ValueExpr resolved( ValueExpr&& expr )
        {
            resolve(expr);
            return expr;
        }
        
        static void canonify( ValueExpr& expr )
        {
            resolve(expr);
            simplify(expr);
        }
        
        static ValueExpr canonical( const ValueExpr& expr )
        {
            auto canonic = expr;
            canonify(canonic);
            return canonic;
        }
        
        static ValueExpr canonical( ValueExpr&& expr )
        {
            canonify(expr);
            return expr;
        }
        
    private:
        
        static void simplify_polynomial( ValueExpr& expr )
        {
            PolynomContext ctx;
            ctx.exprs.push_back(ValueExpr(nullptr));
            
            simplify_polynomial(expr, ctx);
        }
        
        static void simplify_polynomial( ValueExpr& expr, PolynomContext& ctx )
        {
            if ( expr != nullptr && !expr.is_literal() && !expr.is_identifier() &&
                !expr.is_size_access() && !expr.is_shape_access() && !expr.is_tensor_access() )
            {
                auto size = expr.max_size_or_null();
                
                switch ( expr.dtype() )
                {
                    case Typename::Int:
                    {
                        expr = as_expr(as_polynom<int_t>(expr, ctx), size, ctx);
                        break;
                    }
                    case Typename::Real:
                    {
                        expr = as_expr(as_polynom<real_t>(expr, ctx), size, ctx);
                        break;
                    }
                    case Typename::Bool:
                    {
                        expr = as_expr(as_polynom<bool_t>(expr, ctx), size, ctx);
                        break;
                    }
                    default:
                    {
                        assert(false);
                    }
                }
            }
        }
        
    private:
        
        template<typename T>
        static Poly<T> as_polynom( ValueExpr& expr, PolynomContext& ctx )
        {
            switch ( expr.kind() )
            {
                case ValueExpr::Literal:
                {
                    if constexpr( std::is_same_v<T,int_t> )
                    {
                        return Poly<T>::constant(expr.as_int());
                    }
                    else if constexpr( std::is_same_v<T,real_t> )
                    {
                        return Poly<T>::constant(expr.as_real());
                    }
                    else if constexpr( std::is_same_v<T,bool_t> )
                    {
                        return Poly<T>::constant(expr.as_bool());
                    }
                    assert(false);
                }
                case ValueExpr::Identifier:
                {
                    return Poly<T>::monomial(make_symbol(expr, ctx));
                }
                case ValueExpr::Unary:
                {
                    auto& unary = expr.as_unary();
                    auto op = Lexer::operator_value(unary.op);
                    
                    if constexpr( std::is_same_v<T,bool_t> )
                    {
                        if ( op == Lexer::Operator::Not )
                        {
                            auto arg = as_polynom<T>(unary.arg, ctx);
                            return !arg;
                        }
                    }
                    else if ( op == Lexer::Operator::Plus || op == Lexer::Operator::Minus )
                    {
                        auto arg = as_polynom<T>(unary.arg, ctx);
                        
                        switch ( op )
                        {
                            case Lexer::Operator::Plus:
                            {
                                return arg;
                            }
                            case Lexer::Operator::Minus:
                            {
                                return -arg;
                            }
                            default:
                            {
                                break;
                            }
                        }
                    }
                    break;
                }
                case ValueExpr::Binary:
                {
                    auto& binary = expr.as_binary();
                    auto op = Lexer::operator_value(binary.op);
                    
                    if constexpr( std::is_same_v<T,bool_t> )
                    {
                        auto size = expr.max_size_or_null();
                        
                        switch ( op )
                        {
                            case Lexer::Operator::Equal:
                            {
                                return eval_equality(binary.left, binary.right, size, ctx);
                            }
                            case Lexer::Operator::NotEqual:
                            {
                                return !eval_equality(binary.left, binary.right, size, ctx);
                            }
                            case Lexer::Operator::Less:
                            {
                                return eval_inequality(true, binary.left, binary.right, size, ctx);
                            }
                            case Lexer::Operator::LessEqual:
                            {
                                return eval_inequality(false, binary.left, binary.right, size, ctx);
                            }
                            case Lexer::Operator::Greater:
                            {
                                return eval_inequality(true, binary.right, binary.left, size, ctx);
                            }
                            case Lexer::Operator::GreaterEqual:
                            {
                                return eval_inequality(false, binary.right, binary.left, size, ctx);
                            }
                            default:
                            {
                                break;
                            }
                        }
                        
                        if ( op == Lexer::Operator::And || op == Lexer::Operator::Or ||
                            op == Lexer::Operator::Xor || op == Lexer::Operator::Imply )
                        {
                            auto left = as_polynom<T>(binary.left, ctx);
                            auto right = as_polynom<T>(binary.right, ctx);
                            
                            switch ( op )
                            {
                                case Lexer::Operator::And:
                                {
                                    return left && right;
                                }
                                case Lexer::Operator::Or:
                                {
                                    return left || right;
                                }
                                case Lexer::Operator::Xor:
                                {
                                    return left ^ right;
                                }
                                case Lexer::Operator::Imply:
                                {
                                    return !left || right;
                                }
                                default:
                                {
                                    break;
                                }
                            }
                        }
                    }
                    else
                    {
                        if ( op == Lexer::Operator::Plus || op == Lexer::Operator::Minus ||
                            op == Lexer::Operator::Multiply || op == Lexer::Operator::Divide || 
                            op == Lexer::Operator::Power )
                        {
                            auto left = as_polynom<T>(binary.left, ctx);
                            auto right = as_polynom<T>(binary.right, ctx);
                            
                            switch ( op )
                            {
                                case Lexer::Operator::Plus:
                                {
                                    return left + right;
                                }
                                case Lexer::Operator::Minus:
                                {
                                    return left - right;
                                }
                                case Lexer::Operator::Multiply:
                                {
                                    return left * right;
                                }
                                case Lexer::Operator::Divide:
                                {
                                    if ( right.is_constant() && left.is_divisible(right.constant_value()) )
                                    {
                                        left.const_divide(right.constant_value());
                                        return left;
                                    }
                                    if ( left.is_monomial_divisible(right) )
                                    {
                                        left.monomial_divide(right);
                                        return left;
                                    }
                                    break;
                                }
                                case Lexer::Operator::Power:
                                {
                                    if ( right.is_constant() && (int_t)right.constant_value() == right.constant_value() )
                                    {
                                        Poly<T> pow((T)1);
                                        for ( int_t i = 0; i < (int_t)right.constant_value(); ++i )
                                        {
                                            pow *= left;
                                        }
                                        return pow;
                                    }
                                    break;
                                }
                                default:
                                {
                                    break;
                                }
                            }
                        }
                    }
                    break;
                }
                default:
                {
                    break;
                }
            }
            recurse(expr, [&]( ValueExpr& x ){ simplify_polynomial(x, ctx); });
            return Poly<T>(make_symbol(expr, ctx));
        }
        
        template<typename T>
        static ValueExpr as_expr( const Poly<T>& poly, const std::optional<size_t> size, PolynomContext& ctx )
        {
            const Typename type = typename_of<T>::value;
            if ( poly.is_constant() )
            {
                return ValueExpr(poly.constant_value());
            }
            
            ValueExpr expr;
            for ( auto& [symbols, coeff] : poly )
            {
                if ( expr == nullptr )
                {
                    expr = as_expr<T>(symbols, coeff, size, ctx);
                }
                else
                {
                    auto op = coeff < (T)0 ? "-" : "+";
                    expr = ValueExpr::binary(op, expr, as_expr<T>(symbols, std::abs(coeff), size, ctx), type, size);
                }
            }
            if ( poly.constant_value() != (T)0 )
            {
                auto op = poly.constant_value() < (T)0 ? "-" : "+";
                expr = ValueExpr::binary(op, expr, ValueExpr(std::abs(poly.constant_value())), type, size);
            }
            else if ( expr == nullptr )
            {
                expr = (T)0;
            }
            return expr;
        }
        
        template<typename T, typename S, typename C>
        static ValueExpr as_expr( const S& symbols, const C& coeff, const std::optional<size_t> size, PolynomContext& ctx )
        {
            const Typename type = typename_of<T>::value;
            
            ValueExpr expr;
            for ( size_t i = 0; i < symbols.size(); )
            {
                size_t power = 1;
                while ( i + power < symbols.size() && symbols[i+power] == symbols[i] )
                {
                    ++power;
                }
                auto& base = ctx.exprs[symbols[i]];
                auto term = power == 1 ? base : ValueExpr::binary("**", base, ValueExpr((T)power), type, size);
                expr = i == 0 ? term : ValueExpr::binary("*", expr, term, type, size);
                i += power;
            }
            if ( coeff == (T)-1 )
            {
                expr = ValueExpr::unary("-", expr, type, size);
            }
            else if ( coeff != (T)1 )
            {
                expr = ValueExpr::binary("*", ValueExpr(coeff), expr, type, size);
            }
            return expr;
        }
        
        static ValueExpr as_expr( const Poly<int_t>& poly, const std::optional<size_t> size, PolynomContext& ctx )
        {
            return as_expr<int_t>(poly, size, ctx);
        }
        
        static ValueExpr as_expr( const Poly<real_t>& poly, const std::optional<size_t> size, PolynomContext& ctx )
        {
            return as_expr<real_t>(poly, size, ctx);
        }
        
        static ValueExpr as_expr( const Poly<bool_t>& poly, const std::optional<size_t> size, PolynomContext& ctx )
        {
            if ( poly.is_constant() )
            {
                return ValueExpr(poly.constant_value());
            }
            ValueExpr expr;
            for ( auto& symbols : poly )
            {
                if ( expr == nullptr )
                {
                    expr = as_expr(symbols, size, ctx);
                }
                else
                {
                    expr = ValueExpr::binary("^", expr, as_expr(symbols, size, ctx), Typename::Bool, size);
                }
            }
            if ( poly.constant_value() )
            {
                expr = ValueExpr::binary("^", expr, ValueExpr(poly.constant_value()), Typename::Bool, size);
            }
            else if ( expr == nullptr )
            {
                expr = false;
            }
            return expr;
        }
        
        template<typename S>
        static ValueExpr as_expr( const S& symbols, const std::optional<size_t> size, PolynomContext& ctx )
        {
            ValueExpr expr = ctx.exprs[symbols.front()];
            for ( size_t i = 1; i < symbols.size(); ++i )
            {
                expr = ValueExpr::binary("&&", expr, ctx.exprs[symbols[i]], Typename::Bool, size);
            }
            return expr;
        }
        
        template<typename T>
        static Poly<bool_t> eval_equality( const Poly<T>& lhs, const Poly<T>& rhs, const std::optional<size_t> size, PolynomContext& ctx )
        {
            if ( lhs == rhs )
            {
                return Poly<bool_t>(true);
            }
            auto diff = lhs - rhs;
            
            auto leading_value = diff.constant_value() != (T)0 ? diff.constant_value() : diff.begin()->second;
            if ( leading_value < (T)0 )
            {
                diff.negate();
            }
            auto expr = ValueExpr::binary("==", as_expr(diff, size, ctx), ValueExpr((T)0), Typename::Bool, size);
            return Poly<bool_t>(make_symbol(expr, ctx));
        }
        
        static Poly<bool_t> eval_equality( const Poly<bool_t>& lhs, const Poly<bool_t>& rhs, const std::optional<size_t> size, PolynomContext& ctx )
        {
            if ( lhs == rhs )
            {
                return Poly<bool_t>(true);
            }
            auto diff = lhs ^ rhs;
            
            auto expr = ValueExpr::binary("==", as_expr(diff, size, ctx), ValueExpr(false), Typename::Bool, size);
            return Poly<bool_t>(make_symbol(expr, ctx));
        }
        
        template<typename T>
        static Poly<bool_t> eval_inequality( const bool strict, const Poly<T>& lhs, const Poly<T>& rhs, const std::optional<size_t> size,
                                            PolynomContext& ctx )
        {
            auto diff = lhs - rhs;
            
            if ( diff.is_constant() )
            {
                return Poly<bool_t>(strict ? diff.constant_value() < (T)0 : diff.constant_value() <= (T)0);
            }
            else
            {
                auto expr = ValueExpr::binary(strict ? "<" : "<=", as_expr(diff, size, ctx), ValueExpr((T)0), Typename::Bool, size);
                return Poly<bool_t>(make_symbol(expr, ctx));
            }
        }
        
        static Poly<bool_t> eval_equality( ValueExpr& left, ValueExpr& right, const std::optional<size_t> size, PolynomContext& ctx )
        {
            switch ( left.dtype() )
            {
                case Typename::Int:
                {
                    auto lhs = as_polynom<int_t>(left, ctx);
                    auto rhs = as_polynom<int_t>(right, ctx);
                    return eval_equality<int_t>(lhs, rhs, size, ctx);
                }
                case Typename::Real:
                {
                    auto lhs = as_polynom<real_t>(left, ctx);
                    auto rhs = as_polynom<real_t>(right, ctx);
                    return eval_equality<real_t>(lhs, rhs, size, ctx);
                }
                case Typename::Bool:
                {
                    auto lhs = as_polynom<bool_t>(left, ctx);
                    auto rhs = as_polynom<bool_t>(right, ctx);
                    return eval_equality(lhs, rhs, size, ctx);
                }
                default:
                {
                    assert(false);
                    return Poly<bool_t>();
                }
            }
        }
        
        static Poly<bool_t> eval_inequality( const bool strict, ValueExpr& left, ValueExpr& right, const std::optional<size_t> size,
                                            PolynomContext& ctx )
        {
            switch ( left.dtype() )
            {
                case Typename::Int:
                {
                    auto lhs = as_polynom<int_t>(left, ctx);
                    auto rhs = as_polynom<int_t>(right, ctx);
                    return eval_inequality<int_t>(strict, lhs, rhs, size, ctx);
                }
                case Typename::Real:
                {
                    auto lhs = as_polynom<real_t>(left, ctx);
                    auto rhs = as_polynom<real_t>(right, ctx);
                    return eval_inequality<real_t>(strict, lhs, rhs, size, ctx);
                }
                default:
                {
                    assert(false);
                    return Poly<bool_t>();
                }
            }
        }
        
        static symbol_type make_symbol( const ValueExpr& expr, PolynomContext& ctx )
        {
            auto result = ctx.symbols.emplace(str(expr), ctx.exprs.size());
            if ( result.second )
            {
                ctx.exprs.push_back(expr);
            }
            return result.first->second;
        }
        
    private:
        
        template<bool Recursive = true>
        static void simplify_heuristic( ValueExpr& expr )
        {
            if constexpr(Recursive)
            {
                recurse(expr, []( ValueExpr& x ){ simplify_heuristic<true>(x); });
            }
            
            switch ( expr.kind() )
            {
                case ValueExpr::Cast:
                {
                    auto& cast = expr.as_cast();
                    auto& arg = cast.arg.is_uniform() ? cast.arg.as_uniform().value : cast.arg;
                    if ( arg.is_literal() )
                    {
                        switch ( cast.dtype )
                        {
                            case Typename::Real:
                            {
                                expr = arg.dtype() == Typename::Int ? ValueExpr((real_t)(int_t)arg) : arg.dtype() == Typename::Bool ?
                                                                      ValueExpr((real_t)(bool_t)arg) : arg.detach();
                                break;
                            }
                            case Typename::Int:
                            {
                                expr = arg.dtype() == Typename::Real ? ValueExpr((int_t)(real_t)arg) : arg.dtype() == Typename::Bool ?
                                                                       ValueExpr((int_t)(bool_t)arg) : arg.detach();
                                break;
                            }
                            case Typename::Bool:
                            {
                                expr = arg.dtype() == Typename::Real ? ValueExpr((real_t)(bool_t)arg) : arg.dtype() == Typename::Int ?
                                                                       ValueExpr((int_t)(bool_t)arg) : arg.detach();
                                break;
                            }
                            default:
                            {
                                break;
                            }
                        }
                    }
                    else if ( cast.arg.is_cast() )
                    {
                        expr = ValueExpr(ValueExpr::CastExpr{ cast.dtype, cast.arg.as_cast().arg.detach() }, cast.dtype, expr.max_size_or_null());
                    }
                    else if ( cast.arg.dtype() == cast.dtype )
                    {
                        expr = cast.arg.detach();
                    }
                    break;
                }
                case ValueExpr::Unary:
                {
                    auto& unary = expr.as_unary();
                    auto& arg = unary.arg.is_uniform() ? unary.arg.as_uniform().value : unary.arg;
                    auto op = Lexer::operator_value(unary.op);
                    if ( arg.is_literal() )
                    {
                        auto value = unary.op.size() < 3 ? fold_constants_unary(op, arg) :
                                                           fold_constants_unary_func(unary.op, arg);
                        if ( unary.arg.is_uniform() )
                        {
                            expr = ValueExpr::uniform(std::move(value), unary.arg.as_uniform().size.detach(), expr.max_size());
                        }
                        else
                        {
                            expr = std::move(value);
                        }
                    }
                    else
                    {
                        auto value = unary.op.size() < 3 ? simplify_unary(op, unary.arg) : simplify_unary_func(unary.op, unary.arg);
                        if ( value != nullptr )
                        {
                            expr = std::move(value);
                        }
                    }
                    break;
                }
                case ValueExpr::Binary:
                {
                    auto& binary = expr.as_binary();
                    auto& left = binary.left.is_uniform() ? binary.left.as_uniform().value : binary.left;
                    auto& right = binary.right.is_uniform() ? binary.right.as_uniform().value : binary.right;
                    auto op = Lexer::operator_value(binary.op);
                    if ( left.is_literal() && right.is_literal() )
                    {
                        auto value = fold_constants_binary(op, left, right);
                        if ( binary.left.is_uniform() )
                        {
                            expr = ValueExpr::uniform(std::move(value), binary.left.as_uniform().size.detach(), expr.max_size());
                        }
                        else if ( binary.right.is_uniform() )
                        {
                            expr = ValueExpr::uniform(std::move(value), binary.right.as_uniform().size.detach(), expr.max_size());
                        }
                        else
                        {
                            expr = std::move(value);
                        }
                    }
                    else if ( (left.is_shape_access() || right.is_shape_access()) && Lexer::is_comparison(op) )
                    {
                        auto value = simplify_shape_comparison(op, binary.left, binary.right);
                        if ( value != nullptr )
                        {
                            expr = std::move(value);
                        }
                    }
                    else
                    {
                        auto value = simplify_binary(op, binary.left, binary.right);
                        if ( value != nullptr )
                        {
                            expr = std::move(value);
                        }
                    }
                    break;
                }
                case ValueExpr::Select:
                {
                    auto& select = expr.as_select();
                    if ( select.cond.is_literal() )
                    {
                        expr = select.cond.as_bool() ? select.left.detach() : select.right.detach();
                    }
                    else if ( select.cond.is_uniform() && select.cond.as_uniform().value.is_literal() )
                    {
                        expr = select.cond.as_uniform().value.as_bool() ? select.left.detach() : select.right.detach();
                    }
                    else if ( select.cond.is_list() )
                    {
                        auto& list = select.cond.as_list();
                        if ( std::all_of(list.begin(), list.end(), []( const ValueExpr& x ){ return x.is_literal(); }) )
                        {
                            std::vector<ValueExpr> items(list.size());
                            for ( size_t i = 0; i < items.size(); ++i )
                            {
                                items[i] = list[i] ? select.left.at(i) : select.right.at(i);
                            }
                            expr = ValueExpr::list(std::move(items), expr.dtype());
                        }
                    }
                    break;
                }
                case ValueExpr::Fold:
                {
                    auto& fold = expr.as_fold();
                    auto& pack = fold.pack.is_reference() ? *fold.pack.as_reference().target : fold.pack;
                    auto op = Lexer::operator_value(fold.op);
                    if ( pack.is_list() )
                    {
                        auto& items = pack.as_list();
                        if ( std::all_of(items.begin(), items.end(), []( const ValueExpr& x ){ return x.is_literal(); }) )
                        {
                            expr = fold_constants_fold(op, fold.accumulate, pack);
                        }
                    }
                    else if ( pack.is_uniform() )
                    {
                        auto value = simplify_uniform_fold(op, fold.accumulate, pack);
                        if ( value != nullptr )
                        {
                            expr = std::move(value);
                        }
                    }
                    break;
                }
                case ValueExpr::Subscript:
                {
                    auto& subscript = expr.as_subscript();
                    if ( subscript.index.is_literal() )
                    {
                        auto index = (size_t)subscript.index.as_int();
                        expr = subscript.pack.at(index);
                    }
                    else if ( subscript.index.is_uniform() && subscript.index.as_uniform().value.is_literal() )
                    {
                        auto& uniform = subscript.index.as_uniform();
                        auto index = (size_t)uniform.value.as_int();
                        expr = ValueExpr::uniform(subscript.pack.at(index), uniform.size.detach(), subscript.index.max_size());
                    }
                    break;
                }
                case ValueExpr::TensorAccess:
                {
                    auto& access = expr.as_tensor_access();
                    if ( access.item.is_literal() )
                    {
                        access.tensor = &access.tensor[access.item.as_int()];
                        access.item = nullptr;
                    }
                    break;
                }
                case ValueExpr::ShapeAccess:
                {
                    auto& access = expr.as_shape_access();
                    if ( access.dim.is_literal() )
                    {
                        auto& value = access.tensor.shape()[access.dim.as_int()];
                        if ( value.is_literal() )
                        {
                            expr = value;
                        }
                    }
                    break;
                }
                case ValueExpr::SizeAccess:
                {
                    auto& access = expr.as_size_access();
                    auto& value = access.pack.size();
                    if ( value.is_literal() )
                    {
                        expr = value;
                    }
                    break;
                }
                default:
                {
                    break;
                }
            }
        }
        
    protected:
        
        static ValueExpr fold_constants_unary( const Lexer::Operator op, const ValueExpr& arg )
        {
            switch ( op )
            {
                case Lexer::Operator::Plus:
                {
                    return arg;
                }
                case Lexer::Operator::Minus:
                {
                    if ( arg == ValueExpr::positive_infinity<int_t>() )
                    {
                        return ValueExpr::negative_infinity<int_t>();
                    }
                    if ( arg == ValueExpr::negative_infinity<int_t>() )
                    {
                        return ValueExpr::positive_infinity<int_t>();
                    }
                    return arg.dtype() == Typename::Real ? ValueExpr(-(real_t)arg) : ValueExpr(-(int_t)arg);
                }
                case Lexer::Operator::Not:
                {
                    return !(bool_t)arg;
                }
                default:
                {
                    assert(false);
                    return nullptr;
                }
            }
        }
        
        static ValueExpr fold_constants_unary_func( const std::string& name, const ValueExpr& arg )
        {
            if ( arg.dtype() == Typename::Int )
            {
                return eval_unary_func(name, arg.as_int());
            }
            else if ( arg.dtype() == Typename::Real )
            {
                return eval_unary_func(name, arg.as_real());
            }
            else
            {
                assert(false);
                return nullptr;
            }
        }
        
        static ValueExpr fold_constants_binary( const Lexer::Operator op, const ValueExpr& left, const ValueExpr& right )
        {
            switch ( op )
            {
                case Lexer::Operator::Plus:
                {
                    if ( left.dtype() == Typename::Str )
                    {
                        return ValueExpr((str_t)left + (str_t)right);
                    }
                    return left.dtype() == Typename::Real ? ValueExpr((real_t)left + (real_t)right) : ValueExpr((int_t)left + (int_t)right);
                }
                case Lexer::Operator::Minus:
                {
                    return left.dtype() == Typename::Real ? ValueExpr((real_t)left - (real_t)right) : ValueExpr((int_t)left - (int_t)right);
                }
                case Lexer::Operator::Multiply:
                {
                    return left.dtype() == Typename::Real ? ValueExpr((real_t)left * (real_t)right) : ValueExpr((int_t)left * (int_t)right);
                }
                case Lexer::Operator::Divide:
                {
                    return left.dtype() == Typename::Real ? ValueExpr((real_t)left / (real_t)right) : ValueExpr((int_t)left / (int_t)right);
                }
                case Lexer::Operator::CeilDivide:
                {
                    return left.dtype() == Typename::Real ? ValueExpr(ceil_div((real_t)left, (real_t)right)) :
                                                            ValueExpr(ceil_div((int_t)left, (int_t)right));
                }
                case Lexer::Operator::Modulo:
                {
                    return left.dtype() == Typename::Real ? ValueExpr(std::fmod((real_t)left, (real_t)right)) :
                                                            ValueExpr((int_t)left % (int_t)right);
                }
                case Lexer::Operator::Min:
                {
                    return left.dtype() == Typename::Real ? ValueExpr(std::min((real_t)left, (real_t)right)) :
                                                            ValueExpr(std::min((int_t)left, (int_t)right));
                }
                case Lexer::Operator::Max:
                {
                    return left.dtype() == Typename::Real ? ValueExpr(std::max((real_t)left, (real_t)right)) :
                                                            ValueExpr(std::max((int_t)left, (int_t)right));
                }
                case Lexer::Operator::Power:
                {
                    return left.dtype() == Typename::Real ? ValueExpr((real_t)std::pow((real_t)left, (real_t)right)) :
                                                            ValueExpr((int_t)std::pow((int_t)left, (int_t)right));
                }
                case Lexer::Operator::And:
                {
                    return (bool_t)left && (bool_t)right;
                }
                case Lexer::Operator::Or:
                {
                    return (bool_t)left || (bool_t)right;
                }
                case Lexer::Operator::Xor:
                {
                    return (bool_t)left ^ (bool_t)right;
                }
                case Lexer::Operator::Imply:
                {
                    return ValueExpr(!(bool_t)left || (bool_t)right);
                }
                case Lexer::Operator::Less:
                {
                    return left.dtype() == Typename::Real ? (real_t)left < (real_t)right : (int_t)left < (int_t)right;
                }
                case Lexer::Operator::Greater:
                {
                    return left.dtype() == Typename::Real ? (real_t)left > (real_t)right : (int_t)left > (int_t)right;
                }
                case Lexer::Operator::LessEqual:
                {
                    return left.dtype() == Typename::Real ? (real_t)left <= (real_t)right : (int_t)left <= (int_t)right;
                }
                case Lexer::Operator::GreaterEqual:
                {
                    return left.dtype() == Typename::Real ? (real_t)left >= (real_t)right : (int_t)left >= (int_t)right;
                }
                case Lexer::Operator::Equal:
                {
                    return left == right;
                }
                case Lexer::Operator::NotEqual:
                {
                    return left != right;
                }
                default:
                {
                    assert(false);
                    return nullptr;
                }
            }
        }
        
        static ValueExpr simplify_unary( const Lexer::Operator op, ValueExpr& arg )
        {
            return nullptr;
        }
        
        static ValueExpr simplify_unary_func( const std::string& name, ValueExpr& arg )
        {
            if ( (name == "round" || name == "floor" || name == "ceil") && arg.is_cast() )
            {
                auto& cast = arg.as_cast();
                if ( cast.dtype == Typename::Real && cast.arg.dtype() == Typename::Int )
                {
                    return arg.detach();
                }
            }
            return nullptr;
        }
        
        static ValueExpr simplify_binary( const Lexer::Operator op, ValueExpr& left, ValueExpr& right )
        {
            switch ( op )
            {
                case Lexer::Operator::Plus:
                {
                    if ( left == (real_t)0 || left == (int_t)0 )
                    {
                        return right.detach();
                    }
                    if ( right == (real_t)0 || right == (int_t)0 )
                    {
                        return left.detach();
                    }
                    if ( left.is_positive_infinity() || left.is_negative_infinity() )
                    {
                        return left.detach();
                    }
                    if ( right.is_positive_infinity() || right.is_negative_infinity() )
                    {
                        return right.detach();
                    }
                    break;
                }
                case Lexer::Operator::Minus:
                {
                    if ( left == (real_t)0 || left == (int_t)0 )
                    {
                        return -right.detach();
                    }
                    if ( right == (real_t)0 || right == (int_t)0 )
                    {
                        return left.detach();
                    }
                    if ( left.is_positive_infinity() || left.is_negative_infinity() )
                    {
                        return left.detach();
                    }
                    if ( right.is_positive_infinity() || right.is_negative_infinity() )
                    {
                        return -right.detach();
                    }
                    break;
                }
                case Lexer::Operator::Multiply:
                {
                    if ( left == (real_t)1 || left == (int_t)1 )
                    {
                        return right.detach();
                    }
                    if ( right == (real_t)1 || right == (int_t)1 )
                    {
                        return left.detach();
                    }
                    if ( left == (real_t)-1 || left == (int_t)-1 )
                    {
                        return -right.detach();
                    }
                    if ( right == (real_t)-1 || right == (int_t)-1 )
                    {
                        return -left.detach();
                    }
                    if ( left.is_infinity<int_t>() && right.is_literal() )
                    {
                        return right.as_int() > 0 ? left.detach() : -left.detach();
                    }
                    if ( right.is_infinity<int_t>() && left.is_literal() )
                    {
                        return left.as_int() > 0 ? right.detach() : -right.detach();
                    }
                    break;
                }
                case Lexer::Operator::Divide:
                case Lexer::Operator::CeilDivide:
                {
                    if ( right == (real_t)1 || right == (int_t)1 )
                    {
                        return left.detach();
                    }
                    if ( right == (real_t)-1 || right == (int_t)-1 )
                    {
                        return -left.detach();
                    }
                    if ( left == right )
                    {
                        return left.dtype() == Typename::Real ? ValueExpr((real_t)1) : ValueExpr((int_t)1);
                    }
                    if ( left.is_infinity<int_t>() && right.is_literal() )
                    {
                        return right.as_int() > 0 ? left.detach() : -left.detach();
                    }
                    if ( right.is_infinity<int_t>() )
                    {
                        return ValueExpr((int_t)0);
                    }
                    break;
                }
                case Lexer::Operator::Modulo:
                {
                    if ( right == (real_t)1 )
                    {
                        return ValueExpr((real_t)0);
                    }
                    if ( right == (int_t)1 )
                    {
                        return ValueExpr((int_t)0);
                    }
                    break;
                }
                case Lexer::Operator::And:
                {
                    if ( left == true )
                    {
                        return right.detach();
                    }
                    else if ( right == true )
                    {
                        return left.detach();
                    }
                    else if ( right == false )
                    {
                        return right.detach();
                    }
                    break;
                }
                case Lexer::Operator::Or:
                {
                    if ( left == false )
                    {
                        return right.detach();
                    }
                    else if ( right == false )
                    {
                        return left.detach();
                    }
                    else if ( right == true )
                    {
                        return right.detach();
                    }
                    break;
                }
                case Lexer::Operator::Xor:
                {
                    if ( left == right )
                    {
                        return ValueExpr(false);
                    }
                    break;
                }
                case Lexer::Operator::Imply:
                {
                    if ( left == true || right == true )
                    {
                        return right.detach();
                    }
                    else if ( right == false )
                    {
                        return !left.detach();
                    }
                    break;
                }
                case Lexer::Operator::Equal:
                {
                    if ( left == right )
                    {
                        return ValueExpr(true);
                    }
                    break;
                }
                case Lexer::Operator::NotEqual:
                {
                    if ( left == right )
                    {
                        return ValueExpr(false);
                    }
                    break;
                }
                case Lexer::Operator::GreaterEqual:
                {
                    if ( left == right )
                    {
                        return ValueExpr(true);
                    }
                    if ( left.is_positive_infinity() )
                    {
                        return ValueExpr(true);
                    }
                    if ( right.is_positive_infinity() )
                    {
                        return ValueExpr(false);
                    }
                    if ( left.is_negative_infinity() )
                    {
                        return ValueExpr(false);
                    }
                    if ( right.is_negative_infinity() )
                    {
                        return ValueExpr(true);
                    }
                    if ( left.is_shape_access() && right == (int_t)0 )
                    {
                        return ValueExpr(true);
                    }
                    break;
                }
                case Lexer::Operator::LessEqual:
                {
                    if ( left == right )
                    {
                        return ValueExpr(true);
                    }
                    if ( left.is_positive_infinity() )
                    {
                        return ValueExpr(false);
                    }
                    if ( right.is_positive_infinity() )
                    {
                        return ValueExpr(true);
                    }
                    if ( left.is_negative_infinity() )
                    {
                        return ValueExpr(true);
                    }
                    if ( right.is_negative_infinity() )
                    {
                        return ValueExpr(false);
                    }
                    if ( right.is_shape_access() && left == (int_t)0 )
                    {
                        return ValueExpr(true);
                    }
                    break;
                }
                case Lexer::Operator::Greater:
                {
                    if ( left == right )
                    {
                        return ValueExpr(false);
                    }
                    if ( left.is_positive_infinity() )
                    {
                        return ValueExpr(true);
                    }
                    if ( right.is_positive_infinity() )
                    {
                        return ValueExpr(false);
                    }
                    if ( left.is_negative_infinity() )
                    {
                        return ValueExpr(false);
                    }
                    if ( right.is_negative_infinity() )
                    {
                        return ValueExpr(true);
                    }
                    if ( right.is_shape_access() && left == (int_t)0 )
                    {
                        return ValueExpr(false);
                    }
                    break;
                }
                case Lexer::Operator::Less:
                {
                    if ( left == right )
                    {
                        return ValueExpr(false);
                    }
                    if ( left.is_positive_infinity() )
                    {
                        return ValueExpr(false);
                    }
                    if ( right.is_positive_infinity() )
                    {
                        return ValueExpr(true);
                    }
                    if ( left.is_negative_infinity() )
                    {
                        return ValueExpr(true);
                    }
                    if ( right.is_negative_infinity() )
                    {
                        return ValueExpr(false);
                    }
                    if ( left.is_shape_access() && right == (int_t)0 )
                    {
                        return ValueExpr(false);
                    }
                    break;
                }
                case Lexer::Operator::Min:
                {
                    if ( left == right )
                    {
                        return left.detach();
                    }
                    if ( left.is_positive_infinity() )
                    {
                        return right.detach();
                    }
                    if ( right.is_positive_infinity() )
                    {
                        return left.detach();
                    }
                    if ( left.is_negative_infinity() )
                    {
                        return left.detach();
                    }
                    if ( right.is_negative_infinity() )
                    {
                        return right.detach();
                    }
                    if ( left.is_int() && left.as_int() <= 0 && (right.is_shape_access() || right.is_size_access()) )
                    {
                        return left.detach();
                    }
                    if ( right.is_int() && right.as_int() <= 0 && (left.is_shape_access() || left.is_size_access()) )
                    {
                        return right.detach();
                    }
                    break;
                }
                case Lexer::Operator::Max:
                {
                    if ( left == right )
                    {
                        return left.detach();
                    }
                    if ( left.is_positive_infinity() )
                    {
                        return left.detach();
                    }
                    if ( right.is_positive_infinity() )
                    {
                        return right.detach();
                    }
                    if ( left.is_negative_infinity() )
                    {
                        return right.detach();
                    }
                    if ( right.is_negative_infinity() )
                    {
                        return left.detach();
                    }
                    if ( left.is_int() && left.as_int() <= 0 && (right.is_shape_access() || right.is_size_access()) )
                    {
                        return right.detach();
                    }
                    if ( right.is_int() && right.as_int() <= 0 && (left.is_shape_access() || left.is_size_access()) )
                    {
                        return left.detach();
                    }
                    break;
                }
                default:
                {
                    break;
                }
            }
            return nullptr;
        }
        
        static ValueExpr simplify_shape_comparison( const Lexer::Operator op, const ValueExpr& left, const ValueExpr& right )
        {
            return left.dtype() == Typename::Real ? simplify_shape_comparison<real_t>(op, left, right) : simplify_shape_comparison<int_t>(op, left, right);
        }
        
        template<typename T>
        static ValueExpr simplify_shape_comparison( const Lexer::Operator op, const ValueExpr& left, const ValueExpr& right )
        {
            switch ( op )
            {
                case Lexer::Operator::Equal:
                {
                    if ( (left.is_literal() && left.as<T>() < (T)0) || (right.is_literal() && right.as<T>() < (T)0) )
                    {
                        return false;
                    }
                    break;
                }
                case Lexer::Operator::Less:
                {
                    if ( left.is_literal() && left.as<T>() >= (T)0 )
                    {
                        return true;
                    }
                    if ( right.is_literal() && right.as<T>() <= (T)0 )
                    {
                        return false;
                    }
                    break;
                }
                case Lexer::Operator::LessEqual:
                {
                    if ( left.is_literal() && left.as<T>() > (T)0 )
                    {
                        return true;
                    }
                    if ( right.is_literal() && right.as<T>() < (T)0 )
                    {
                        return false;
                    }
                    break;
                }
                case Lexer::Operator::Greater:
                {
                    if ( left.is_literal() && left.as<T>() <= (T)0 )
                    {
                        return false;
                    }
                    if ( right.is_literal() && right.as<T>() >= (T)0 )
                    {
                        return true;
                    }
                    break;
                }
                case Lexer::Operator::GreaterEqual:
                {
                    if ( left.is_literal() && left.as<T>() < (T)0 )
                    {
                        return false;
                    }
                    if ( right.is_literal() && right.as<T>() > (T)0 )
                    {
                        return true;
                    }
                    break;
                }
                default:
                {
                    break;
                }
            }
            return nullptr;
        }
        
        static ValueExpr fold_constants_fold( const Lexer::Operator op, bool accumulate, const ValueExpr& pack )
        {
            auto& items = pack.as_list();
            switch ( op )
            {
                case Lexer::Operator::Plus:
                {
                    return pack.dtype() == Typename::Real ? eval_fold<std::plus,real_t>(accumulate, items, (real_t)0) :
                                                           eval_fold<std::plus,int_t>(accumulate, items, (int_t)0);
                }
                case Lexer::Operator::Multiply:
                {
                    return pack.dtype() == Typename::Real ? eval_fold<std::multiplies,real_t>(accumulate, items, (real_t)1) :
                                                           eval_fold<std::multiplies,int_t>(accumulate, items, (int_t)1);
                }
                case Lexer::Operator::Min:
                {
                    return pack.dtype() == Typename::Real ? eval_fold<minimize,real_t>(accumulate, items) :
                                                           eval_fold<minimize,int_t>(accumulate, items);
                }
                case Lexer::Operator::Max:
                {
                    return pack.dtype() == Typename::Real ? eval_fold<minimize,real_t>(accumulate, items) :
                                                           eval_fold<minimize,int_t>(accumulate, items);
                }
                case Lexer::Operator::And:
                {
                    return eval_fold<std::logical_and,bool_t>(accumulate, items, (bool_t)true);
                }
                case Lexer::Operator::Or:
                {
                    return eval_fold<std::logical_or,bool_t>(accumulate, items, (bool_t)false);
                }
                default:
                {
                    assert(false);
                    return nullptr;
                }
            }
        }
        
        static ValueExpr simplify_uniform_fold( const Lexer::Operator op, bool accumulate, const ValueExpr& pack )
        {
            auto& uniform = pack.as_uniform();
            switch ( op )
            {
                case Lexer::Operator::Plus:
                {
                    if ( !accumulate )
                    {
                        return uniform.size * uniform.value;
                    }
                    else if ( uniform.size.is_literal() )
                    {
                        std::vector<ValueExpr> items(uniform.size.as_int());
                        for ( int_t i = 0; i < items.size(); ++i )
                        {
                            items[i] = i * uniform.value;
                        }
                        return ValueExpr::list(std::move(items), pack.dtype());
                    }
                    else if ( pack.dtype() == Typename::Int )
                    {
                        return ValueExpr(ValueExpr::RangeExpr{ uniform.value, uniform.value + uniform.size * uniform.value, uniform.value },
                                         pack.dtype(), pack.max_size());
                    }
                }
                case Lexer::Operator::Multiply:
                {
                    if ( !accumulate )
                    {
                        return pow(uniform.value, uniform.size);
                    }
                    else if ( uniform.size.is_literal() )
                    {
                        std::vector<ValueExpr> items(uniform.size.as_int());
                        for ( int_t i = 0; i < items.size(); ++i )
                        {
                            items[i] = pow(uniform.value, i);
                        }
                        return ValueExpr::list(std::move(items), pack.dtype());
                    }
                }
                case Lexer::Operator::Min:
                case Lexer::Operator::Max:
                case Lexer::Operator::And:
                case Lexer::Operator::Or:
                {
                    return accumulate ? pack : uniform.value;
                }
                default:
                {
                    return nullptr;
                }
            }
        }
        
        template<template<typename> class F, typename T>
        static ValueExpr eval_fold( bool accumulate, const std::vector<ValueExpr>& items, const std::optional<T> init = std::nullopt )
        {
            F<T> func;
            if ( accumulate )
            {
                std::vector<ValueExpr> values(items.size());
                values.front() = items.front();
                for ( size_t i = 1; i < items.size(); ++i )
                {
                    values[i] = func(values[i-1].as<T>(), items[i].as<T>());
                }
                return ValueExpr::list(std::move(values), typename_of<T>::value);
            }
            else
            {
                auto value = init;
                for ( auto& item : items )
                {
                    value = func(*value, item.as<T>());
                }
                return value ? ValueExpr(*value) : ValueExpr(nullptr);
            }
        }
        
        static ValueExpr eval_unary_func( const std::string& name, const int_t arg )
        {
            typedef int_t (*int_builtin_t)(int_t);
            static const Dict<int_builtin_t> int_builtins =
            {
                { "abs", std::abs },
                { "sign", sknd::sign },
            };
            
            auto func = int_builtins.at(name);
            return func(arg);
        }
        
        static ValueExpr eval_unary_func( const std::string& name, const real_t arg )
        {
            typedef real_t (*real_builtin_t)(real_t);
            static const Dict<real_builtin_t> real_builtins =
            {
                { "abs", std::abs },
                { "sign", sknd::sign },
                { "exp", std::exp },
                { "log", std::log },
                { "sqrt", std::sqrt },
                { "sin", std::sin },
                { "cos", std::cos },
                { "tan", std::tan },
                { "asin", std::asin },
                { "acos", std::acos },
                { "atan", std::atan },
                { "sinh", std::sinh },
                { "cosh", std::cosh },
                { "tanh", std::tanh },
                { "asinh", std::asinh },
                { "acosh", std::acosh },
                { "atanh", std::atanh },
                { "erf", std::erf },
                { "round", std::round },
                { "floor", std::floor },
                { "ceil", std::ceil },
                { "frac", sknd::frac },
            };
            
            auto func = real_builtins.at(name);
            return func(arg);
        }
        
    protected:
        
        static bool is_trigonometric_func( const std::string& func )
        {
            static const std::set<std::string> funcs =
            {
                "sin",
                "cos",
                "tan",
                "asin",
                "acos",
                "atan",
                "sinh",
                "cosh",
                "tanh",
                "asinh",
                "acosh",
                "atanh",
            };
            return funcs.count(func) != 0;
        }
    };

}   // namespace sknd


#endif
