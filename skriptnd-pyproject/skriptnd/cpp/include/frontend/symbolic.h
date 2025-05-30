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

#ifndef _SKND_SYMBOLIC_H_
#define _SKND_SYMBOLIC_H_

#include "astexpr.h"
#include "polynom.h"
#include "lexer.h"
#include <algorithm>
#include <numeric>
#include <cassert>
#include <map>


namespace sknd
{

    typedef std::function<Typename(const std::string&)> SymbolTypes;

    
    template<typename S = char>
    class Symbolic
    {
        using Operator = Lexer::Operator;
        
        template<typename T>
        using Dict = std::map<std::string,T>;
        
    public:
        
        typedef S symbol_type;
        
        template<typename T>
        using poly = polynom<T,symbol_type>;
        
    public:
        
        void clear()
        {
            _symbols.clear();
        }
        
        template<typename T>
        poly<T> eval_polynom( const Expr& expr, const SymbolTypes& types, const Dict<Shared<Expr>>& subs = {} )
        {
            switch ( expr.kind )
            {
                case Expr::Literal:
                {
                    if constexpr( std::is_same_v<T,int_t> )
                    {
                        return poly<T>(as_int(expr).value);
                    }
                    else if constexpr( std::is_same_v<T,real_t> )
                    {
                        return poly<T>(as_real(expr).value);
                    }
                    else if constexpr( std::is_same_v<T,bool_t> )
                    {
                        return poly<T>(as_bool(expr).value);
                    }
                    assert(false);
                    return poly<T>();
                }
                case Expr::Identifier:
                {
                    auto& identifier = as_identifier(expr);
                    auto it = subs.find(identifier.name);
                    if ( it != subs.end() )
                    {
                        return eval_polynom<T>(*it->second, types, subs);
                    }
                    return poly<T>(make_symbol(identifier.name));
                }
                case Expr::Unary:
                {
                    auto& unary = as_unary(expr);
                    auto arg = eval_polynom<T>(*unary.arg, types, subs);
                    
                    if constexpr( std::is_same_v<T,bool_t> )
                    {
                        if ( unary.op == Operator::Not )
                        {
                            return !arg;
                        }
                    }
                    else
                    {
                        switch ( unary.op )
                        {
                            case Operator::Plus:
                            {
                                return arg;
                            }
                            case Operator::Minus:
                            {
                                return -arg;
                            }
                            default:
                            {
                                break;
                            }
                        }
                    }
                    return poly<T>(make_symbol(Lexer::str(unary.op) + str(arg)));
                }
                case Expr::Binary:
                {
                    auto& binary = as_binary(expr);
                    
                    if constexpr( std::is_same_v<T,bool_t> )
                    {
                        if ( binary.op == Operator::Equal )
                        {
                            return eval_equality(*binary.left, *binary.right, types, subs);
                        }
                        else if ( binary.op == Operator::NotEqual )
                        {
                            return !eval_equality(*binary.left, *binary.right, types, subs);
                        }
                        else if ( binary.op == Operator::Less || binary.op == Operator::Greater ||
                                  binary.op == Operator::LessEqual || binary.op == Operator::GreaterEqual )
                        {
                            return eval_inequality(binary.op, *binary.left, *binary.right, types, subs);
                        }
                        
                        auto lhs = eval_polynom<T>(*binary.left, types, subs);
                        auto rhs = eval_polynom<T>(*binary.right, types, subs);
                        switch ( binary.op )
                        {
                            case Operator::And:
                            {
                                return lhs && rhs;
                            }
                            case Operator::Or:
                            {
                                return lhs || rhs;
                            }
                            case Operator::Imply:
                            {
                                return !lhs || rhs;
                            }
                            default:
                            {
                                return poly<T>(make_symbol(str(lhs) + Lexer::str(binary.op) + str(rhs)));
                            }
                        }
                    }
                    else
                    {
                        if ( binary.op == Operator::Min || binary.op == Operator::Max )
                        {
                            auto op = binary.op == Operator::Min ? Lexer::Operator::Less : Lexer::Operator::Greater;
                            auto cond = std::make_shared<BinaryExpr>(binary.position, binary.left, binary.right, op);
                            auto select = std::make_shared<SelectExpr>(binary.position, cond, binary.left, binary.right);
                            return eval_polynom<T>(*select, types, subs);
                        }
                        
                        auto lhs = eval_polynom<T>(*binary.left, types, subs);
                        auto rhs = eval_polynom<T>(*binary.right, types, subs);
                        switch ( binary.op )
                        {
                            case Operator::Plus:
                            {
                                return lhs + rhs;
                            }
                            case Operator::Minus:
                            {
                                return lhs - rhs;
                            }
                            case Operator::Multiply:
                            {
                                return lhs * rhs;
                            }
                            case Operator::Divide:
                            {
                                if ( rhs.is_constant() && lhs.is_divisible(rhs.constant_value()) )
                                {
                                    lhs.const_divide(rhs.constant_value());
                                    return lhs;
                                }
                            }
                            case Operator::Power:
                            {
                                if ( rhs.is_constant() && (int_t)rhs.constant_value() == rhs.constant_value() )
                                {
                                    poly<T> pow((T)1);
                                    for ( int_t i = 0; i < (int_t)rhs.constant_value(); ++i )
                                    {
                                        pow *= lhs;
                                    }
                                    return pow;
                                }
                            }
                            default:
                            {
                                return poly<T>(make_symbol(str(lhs) + Lexer::str(binary.op) + str(rhs)));
                            }
                        }
                    }
                }
                case Expr::Select:
                {
                    auto& select = as_select(expr);
                    auto cond = eval_polynom<bool_t>(*select.cond, types, subs);
                    auto lhs = eval_polynom<T>(*select.left, types, subs);
                    
                    if ( !select.right )
                    {
                        return poly<T>(make_symbol(str(cond) + "?" + str(lhs)));
                    }
                    
                    auto rhs = eval_polynom<T>(*select.right, types, subs);
                    
                    if constexpr( std::is_same_v<T,bool_t> )
                    {
                        return poly<bool_t>((cond && lhs) || (!cond && rhs));
                    }
                    else
                    {
                        if ( !cond.constant_value() )
                        {
                            cond.negate();
                            std::swap(lhs, rhs);
                        }
                        return poly<T>(make_symbol(str(cond) + "?" + str(lhs) + ":" + str(rhs)));
                    }
                }
                case Expr::Coalesce:
                {
                    auto& coalesce = as_coalesce(expr);
                    auto condition = eval_polynom<T>(*coalesce.condition, types, subs);
                    auto alternate = eval_polynom<T>(*coalesce.alternate, types, subs);
                    return poly<T>(make_symbol(str(condition) + "??" + str(alternate)));
                }
                case Expr::List:
                {
                    auto& list = as_list(expr);
                    std::string symbol;
                    
                    symbol += '[';
                    for ( auto& item : list.items )
                    {
                        if ( symbol.length() != 1 )
                        {
                            symbol += ',';
                        }
                        symbol += str(eval_polynom<T>(*item, types, subs));
                    }
                    symbol += ']';
                    
                    return poly<T>(make_symbol(symbol));
                }
                case Expr::Zip:
                {
                    auto& zip = as_zip(expr);
                    std::string symbol;
                    
                    symbol += '(';
                    for ( auto& item : zip.items )
                    {
                        if ( symbol.length() != 1 )
                        {
                            symbol += ',';
                        }
                        symbol += str(eval_polynom<T>(*item, types, subs));
                    }
                    symbol += ')';
                    return poly<T>(make_symbol(symbol));
                }
                case Expr::Expand:
                {
                    auto& expand = as_expand(expr);
                    return eval_polynom<T>(*expand.item, types, subs);
                }
                case Expr::Range:
                {
                    if constexpr( !std::is_same_v<T,bool_t> )
                    {
                        auto& range = as_range(expr);
                        auto first = range.first ? eval_polynom<T>(*range.first, types, subs) : poly<T>((T)0);
                        auto last = eval_polynom<T>(*range.last, types, subs);
                        auto stride = range.stride ? eval_polynom<T>(*range.stride, types, subs) : poly<T>((T)1);
                        return poly<T>(make_symbol(str(first) + ':' + str(last) + ':' + str(stride)));
                    }
                    break;
                }
                case Expr::Fold:
                {
                    auto& fold = as_fold(expr);
                    assert(!fold.cumulative);
                    
                    bool foldable = fold.op == Operator::Plus || fold.op == Operator::Multiply || 
                                    fold.op == Operator::And || fold.op == Operator::Or;
                    
                    if ( foldable && fold.pack->kind == Expr::List )
                    {
                        auto& list = as_list(*fold.pack);
                        
                        poly<T> value;
                        if constexpr( std::is_same_v<T,bool_t> )
                        {
                            value = fold.op == Operator::And;
                        }
                        else
                        {
                            value = fold.op == Operator::Plus ? (T)0 : (T)1;
                        }
                        for ( size_t i = 0; i < list.items.size(); ++i )
                        {
                            auto item = eval_polynom<T>(*list.items[i], types, subs);
                            if ( list.items[i]->kind == Expr::Expand )
                            {
                                item = poly<T>(make_symbol(str(item) + Lexer::str(fold.op)));
                            }
                            if constexpr( std::is_same_v<T,bool_t> )
                            {
                                if ( fold.op == Operator::And )
                                {
                                    value &= item;
                                }
                                else
                                {
                                    value |= item;
                                }
                            }
                            else
                            {
                                if ( fold.op == Operator::Plus )
                                {
                                    value += item;
                                }
                                else
                                {
                                    value *= item;
                                }
                            }
                        }
                        return value;
                    }
                    else
                    {
                        auto item = eval_polynom<T>(*fold.pack, types, subs);
                        return poly<T>(make_symbol(str(item) + Lexer::str(fold.op)));
                    }
                }
                case Expr::Index:
                {
                    auto& subscript = as_index(expr);
                    auto pack = eval_polynom<T>(*subscript.array, types, subs);
                    auto index = eval_polynom<int_t>(*subscript.index, types, subs);
                    return poly<T>(make_symbol(str(pack) + '[' + str(index) + ']'));
                }
                case Expr::Substitute:
                {
                    auto& substitute = as_substitute(expr);
                    auto pack = eval_polynom<T>(*substitute.pack, types, subs);
                    auto index = eval_polynom<int_t>(*substitute.index, types, subs);
                    auto value = eval_polynom<T>(*substitute.value, types, subs);
                    return poly<T>(make_symbol(str(pack) + '[' + str(index) + ']' + "<-" + str(value)));
                }
                case Expr::Identity:
                {
                    if constexpr( std::is_same_v<T,bool_t> )
                    {
                        auto& identity = as_identity(expr);
                        return eval_identity(*identity.left, *identity.right, types, subs);
                    }
                    break;
                }
                case Expr::Contain:
                {
                    if constexpr( std::is_same_v<T,bool_t> )
                    {
                        auto& contain = as_contain(expr);
                        return eval_identity(*contain.item, *contain.pack, types, subs);
                    }
                    break;
                }
                case Expr::Cast:
                {
                    auto& cast = as_cast(expr);
                    if ( !cast.arg )
                    {
                        return poly<T>();
                    }
                    const Typename type = eval_type(*cast.arg, types);
                    switch ( type )
                    {
                        case Typename::Int:
                        {
                            auto arg = eval_polynom<int_t>(*cast.arg, types, subs);
                            if ( arg.is_constant() )
                            {
                                return poly<T>((T)arg.constant_value());
                            }
                            return poly<T>(make_symbol(cast.type + '(' + str(arg) + ')'));
                        }
                        case Typename::Real:
                        {
                            auto arg = eval_polynom<real_t>(*cast.arg, types, subs);
                            if ( arg.is_constant() )
                            {
                                return poly<T>((T)arg.constant_value());
                            }
                            return poly<T>(make_symbol(cast.type + '(' + str(arg) + ')'));
                        }
                        case Typename::Bool:
                        {
                            auto arg = eval_polynom<bool_t>(*cast.arg, types, subs);
                            if ( arg.is_constant() )
                            {
                                return poly<T>((T)arg.constant_value());
                            }
                            return poly<T>(make_symbol(cast.type + '(' + str(arg) + ')'));
                        }
                        default:
                        {
                            assert(false);
                            return poly<T>();
                        }
                    }
                }
                case Expr::Builtin:
                {
                    auto& builtin = as_builtin(expr);
                    auto arg = eval_polynom<T>(*builtin.arg, types, subs);
                    return poly<T>(make_symbol(builtin.func + '(' + str(arg) + ')'));
                }
                default:
                {
                    break;
                }
            }
            assert(false);
            return poly<T>();
        }

    private:
        
        template<typename T>
        poly<bool_t> eval_equality( const poly<T>& lhs, const poly<T>& rhs )
        {
            if ( lhs == rhs )
            {
                return poly<bool_t>(true);
            }
            auto expr = lhs - rhs;
            
            auto leading_value = expr.constant_value() ? expr.constant_value() : expr.begin()->second;
            if ( leading_value < 0 )
            {
                expr.negate();
            }
            return poly<bool_t>(make_symbol(str(expr) + "=="));
        }
        
        poly<bool_t> eval_equality( const poly<bool_t>& lhs, const poly<bool_t>& rhs )
        {
            if ( lhs == rhs )
            {
                return poly<bool_t>(true);
            }
            auto expr = lhs ^ rhs;
            return poly<bool_t>(make_symbol(str(expr) + "=="));
        }
        
        template<typename T>
        poly<bool_t> eval_inequality( const Operator op, const poly<T>& lhs, const poly<T>& rhs )
        {
            bool negate = (op == Operator::LessEqual || op == Operator::GreaterEqual);
            auto expr = (op == Operator::Less || op == Operator::Greater) ? lhs - rhs : rhs - lhs;
            
            poly<bool_t> result;
            if ( expr.is_constant() )
            {
                result = poly<bool_t>(expr.constant_value() < 0);
            }
            else
            {
                result = poly<bool_t>(make_symbol(str(expr) + '<'));
            }
            
            return negate ? result : result.negate();
        }
        
        poly<bool_t> eval_equality( const Expr& left, const Expr& right, const SymbolTypes& types, const Dict<Shared<Expr>>& subs )
        {
            const Typename type = eval_type(left, types);
            switch ( type )
            {
                case Typename::Int:
                {
                    auto lhs = eval_polynom<int_t>(left, types, subs);
                    auto rhs = eval_polynom<int_t>(right, types, subs);
                    return eval_equality(lhs, rhs);
                }
                case Typename::Num:
                case Typename::Real:
                {
                    auto lhs = eval_polynom<real_t>(left, types, subs);
                    auto rhs = eval_polynom<real_t>(right, types, subs);
                    return eval_equality(lhs, rhs);
                }
                case Typename::Bool:
                {
                    auto lhs = eval_polynom<real_t>(left, types, subs);
                    auto rhs = eval_polynom<real_t>(right, types, subs);
                    return eval_equality(lhs, rhs);
                }
                case Typename::Str:
                {
                    auto lhs = str(left);
                    auto rhs = str(right);
                    if ( lhs == rhs )
                    {
                        return poly<bool_t>(true);
                    }
                    return poly<bool_t>(make_symbol(lhs + "==" + rhs));
                }
                default:
                {
                    assert(false);
                    return poly<bool_t>();
                }
            }
        }
        
        poly<bool_t> eval_inequality( const Operator op, const Expr& left, const Expr& right, const SymbolTypes& types, const Dict<Shared<Expr>>& subs )
        {
            const Typename type = eval_type(left, types);
            switch ( type )
            {
                case Typename::Int:
                {
                    auto lhs = eval_polynom<int_t>(left, types, subs);
                    auto rhs = eval_polynom<int_t>(right, types, subs);
                    return eval_inequality(op, lhs, rhs);
                }
                case Typename::Num:
                case Typename::Real:
                {
                    auto lhs = eval_polynom<real_t>(left, types, subs);
                    auto rhs = eval_polynom<real_t>(right, types, subs);
                    return eval_inequality(op, lhs, rhs);
                }
                default:
                {
                    assert(false);
                    return poly<bool_t>();
                }
            }
        }
        
        poly<bool_t> eval_identity( const Expr& left, const Expr& right, const SymbolTypes& types, const Dict<Shared<Expr>>& subs )
        {
            const Typename type = eval_type(left, types);
            switch ( type )
            {
                case Typename::Int:
                {
                    auto lhs = eval_polynom<int_t>(left, types, subs);
                    auto rhs = eval_polynom<int_t>(right, types, subs);
                    return poly<bool_t>(make_symbol(str(lhs) + " is " + str(rhs)));
                }
                case Typename::Real:
                {
                    auto lhs = eval_polynom<real_t>(left, types, subs);
                    auto rhs = eval_polynom<real_t>(right, types, subs);
                    return poly<bool_t>(make_symbol(str(lhs) + " is " + str(rhs)));
                }
                case Typename::Bool:
                {
                    auto lhs = eval_polynom<real_t>(left, types, subs);
                    auto rhs = eval_polynom<real_t>(right, types, subs);
                    return poly<bool_t>(make_symbol(str(lhs) + " is " + str(rhs)));
                }
                case Typename::Str:
                {
                    return poly<bool_t>(make_symbol(str(left) + " is " + str(right)));
                }
                default:
                {
                    assert(false);
                    return poly<bool_t>();
                }
            }
        }
        
        poly<bool_t> eval_contain( const Expr& left, const Expr& right, const SymbolTypes& types, const Dict<Shared<Expr>>& subs )
        {
            const Expr& items = right.kind == Expr::Identifier ? *subs.at(as_identifier(right).name) : right;
            assert(items.kind == Expr::List);
            auto& list = as_list(items);
            
            const Typename type = eval_type(left, types);
            switch ( type )
            {
                case Typename::Int:
                {
                    poly<bool_t> result;
                    auto lhs = eval_polynom<int_t>(left, types, subs);
                    for ( auto& item : list.items )
                    {
                        auto rhs = eval_polynom<int_t>(*item, types, subs);
                        result |= eval_equality(lhs, rhs);
                    }
                    return result;
                }
                case Typename::Real:
                {
                    poly<bool_t> result;
                    auto lhs = eval_polynom<real_t>(left, types, subs);
                    for ( auto& item : list.items )
                    {
                        auto rhs = eval_polynom<real_t>(*item, types, subs);
                        result |= eval_equality(lhs, rhs);
                    }
                    return result;
                }
                case Typename::Bool:
                {
                    poly<bool_t> result;
                    auto lhs = eval_polynom<bool_t>(left, types, subs);
                    for ( auto& item : list.items )
                    {
                        auto rhs = eval_polynom<bool_t>(*item, types, subs);
                        result |= eval_equality(lhs, rhs);
                    }
                    return result;
                }
                case Typename::Str:
                {
                    poly<bool_t> result;
                    auto lhs = str(left);
                    for ( auto& item : list.items )
                    {
                        auto rhs = str(*item);
                        if ( lhs == rhs )
                        {
                            return poly<bool_t>(true);
                        }
                        result |= poly<bool_t>(make_symbol(lhs + "==" + rhs));
                    }
                    return result;
                }
                default:
                {
                    assert(false);
                    return poly<bool_t>();
                }
            }
        }
        
        static Typename eval_type( const Expr& expr, const SymbolTypes& types )
        {
            switch ( expr.kind )
            {
                case Expr::Literal:
                {
                    auto& literal = as_literal(expr);
                    return literal.type.name;
                }
                case Expr::Identifier:
                {
                    auto& name = as_identifier(expr).name;
                    return types(name);
                }
                case Expr::List:
                {
                    auto& list = as_list(expr);
                    for ( auto& item : list.items )
                    {
                        auto type = eval_type(*item, types);
                        if ( type != Typename::Type )
                        {
                            return type;
                        }
                    }
                    return Typename::Type;
                }
                case Expr::Expand:
                {
                    auto& expand = as_expand(expr);
                    return eval_type(*expand.item, types);
                }
                case Expr::Index:
                {
                    auto& index = as_index(expr);
                    return eval_type(*index.array, types);
                }
                case Expr::Access:
                {
                    auto& access = as_access(expr);
                    return eval_type(*access.tensor, types);
                }
                case Expr::Range:
                {
                    return Typename::Int;
                }
                case Expr::Zip:
                {
                    auto& zip = as_zip(expr);
                    for ( auto& item : zip.items )
                    {
                        auto type = eval_type(*item, types);
                        if ( type != Typename::Type )
                        {
                            return type;
                        }
                    }
                    return Typename::Type;
                }
                case Expr::Unary:
                {
                    auto& unary = as_unary(expr);
                    bool logical = unary.op == Lexer::Operator::Not || unary.op == Lexer::Operator::Question;
                    return logical ? Typename::Bool : eval_type(*unary.arg, types);
                }
                case Expr::Binary:
                {
                    auto& binary = as_binary(expr);
                    if ( Lexer::is_comparison(binary.op) )
                    {
                        return Typename::Bool;
                    }
                    auto type = eval_type(*binary.left, types);
                    return type != Typename::Type ? type : eval_type(*binary.right, types);
                }
                case Expr::Select:
                {
                    auto& select = as_select(expr);
                    auto type = eval_type(*select.left, types);
                    return type != Typename::Type ? type : select.right ? eval_type(*select.right, types) : Typename::Type;
                }
                case Expr::Coalesce:
                {
                    auto& coalesce = as_coalesce(expr);
                    auto type = eval_type(*coalesce.condition, types);
                    return type != Typename::Type ? type : eval_type(*coalesce.alternate, types);
                }
                case Expr::Identity:
                case Expr::Contain:
                {
                    return Typename::Bool;
                }
                case Expr::Fold:
                {
                    auto& fold = as_fold(expr);
                    return eval_type(*fold.pack, types);
                }
                case Expr::Cast:
                {
                    auto& cast = as_cast(expr);
                    return cast.base;
                }
                case Expr::Builtin:
                {
                    return Typename::Real;
                }
                case Expr::Format:
                {
                    return Typename::Str;
                }
                case Expr::Substitute:
                {
                    auto& substitute = as_substitute(expr);
                    return eval_type(*substitute.pack, types);
                }
                case Expr::Bounded:
                {
                    assert(false);
                    return Typename::Type;
                }
            }

            assert(false);
            return Typename::Type;
        }
        
    private:
        
        symbol_type make_symbol( const std::string& expr )
        {
            auto result = _symbols.emplace(expr, _symbols.size() + 1);
            return result.first->second;
        }
        
    private:
        
        Dict<symbol_type> _symbols;
    };
    
}   // namespace sknd


#endif
