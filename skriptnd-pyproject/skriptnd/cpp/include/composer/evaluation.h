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

#ifndef _SKND_EVALUATION_H_
#define _SKND_EVALUATION_H_

#include "astexpr.h"
#include "result.h"
#include "either.h"
#include "packable.h"
#include "model.h"
#include "valuexpr.h"
#include "simplification.h"
#include <cassert>
#include <numeric>
#include <algorithm>
#include <variant>
#include <cmath>

#define DECL_BOUNDS(var1, var2, expr) auto var1##var2 = (expr); if ( !var1##var2 ) return std::nullopt; auto[var1, var2] = *var1##var2;


inline bool endswith( const std::string& str, const std::string& suffix )
{
    return str.length() > suffix.length() && str.substr(str.length() - suffix.length()) == suffix;
}


namespace sknd
{

    class Evaluation : public Simplification
    {
    protected:
        
        template<typename T>
        using Dict = std::map<std::string,T>;
        
        typedef std::unordered_map<const Expr*, ValueExpr> Cache;
        
        typedef std::vector<ValueExpr> Shape;
        
        typedef std::vector<Tensor*> Tensors;
        typedef std::function<TensorRef(const ValueExpr&, const Typename&)> AsTensor;
        typedef std::function<TensorRef(const Tensors&, const Typename&, const Shape&)> AsTensorPack;
        
        using Simplification::resolve;
        using Simplification::resolved;
        using Simplification::canonify;
        using Simplification::canonical;
        using Simplification::simplify;
        using Simplification::simplified;
        
    private:
        
        static TensorRef as_null_tensor( const ValueExpr& value, const Typename& type )
        {
            return nullptr;
        }
        
        static TensorRef as_null_tensor_pack( const Tensors& tensors, const Typename& dtype, const Shape& shape, const ValueExpr& size )
        {
            return nullptr;
        }
        
    protected:
        
        struct Valueless {};
        
        struct Symbol : public std::variant<ValueExpr,TensorRef,Valueless>
        {
            enum Kind
            {
                Gtype, Attrib, Input, Output, Variable, Constant, Using, Result, Extent, Index, Carried, Scan, Local,
            };
            
            typedef std::variant<ValueExpr,TensorRef,Valueless> variant_type;
            
            Typename type;
            ValueExpr size;
            Kind kind;
            
            Symbol( const ValueExpr& value, const Typename type, const Kind kind )
                : variant_type(value), type(type), size(value.size()), kind(kind) {}
            Symbol( const ValueExpr& value, const Typename type, const ValueExpr& size, const Kind kind )
                : variant_type(value), type(type), size(size), kind(kind) {}
            Symbol( const TensorRef& tensor, const Typename type, const Kind kind )
                : variant_type(tensor), type(type), size(tensor.size()), kind(kind) {}
            Symbol( const Typename type, const ValueExpr& size, const Kind kind )
                : variant_type(Valueless()), type(type), size(size), kind(kind) {}
            
            template<typename T>
            bool is() const
            {
                return std::holds_alternative<T>(*this);
            }
            
            template<typename T>
            const T& as() const
            {
                return std::get<T>(*this);
            }
            
            template<typename T>
            T& as()
            {
                return std::get<T>(*this);
            }
            
            bool packed() const
            {
                return size != nullptr;
            }
            
            bool is_null() const
            {
                return is<ValueExpr>() ? as<ValueExpr>() == nullptr : is<TensorRef>() ? as<TensorRef>() == nullptr : false;
            }
            
            bool is_runtime() const
            {
                return is<Valueless>() && (kind == Index || kind == Local);
            }
        };
        
    public:
        
        static Result<ValueExpr> eval( const Expr& expr, const Dict<Symbol>& symbols )
        {
            TRY_DECL(rank, eval_rank(expr, symbols))
            return rank != nullptr ? eval_pack(expr, symbols, rank) : eval_item(expr, symbols);
        }
        
        static Result<TensorRef> eval( const Expr& expr, const Dict<Symbol>& symbols, const AsTensor& astensor, const AsTensorPack aspack,
                                       const std::optional<size_t> idx = std::nullopt )
        {
            TRY_DECL(is_null, eval_null(expr, symbols))
            if ( is_null )
            {
                return TensorRef(nullptr);
            }
            else if ( !is_tensor_expr(expr, symbols) )
            {
                TRY_DECL(value, eval_item(expr, symbols, idx, nullptr))
                auto type = eval_type(expr, symbols);
                if ( !is_literal(value) )
                {
                    return Error(expr.position, "dynamic expression not allowed in this context; found '%s'", str(value).c_str());
                }
                return astensor(value, type);
            }
            
            TRY_DECL(size, eval_rank(expr, symbols))
            if ( size != nullptr && !idx )
            {
                return eval_tensor_pack(expr, symbols, size, aspack);
            }
            else
            {
                TRY_DECL(tensor, eval_item<Tensor*>(expr, symbols, idx, nullptr))
                return TensorRef(tensor);
            }
        }
        
        static Result<ValueExpr> eval_optional( const Expr& expr, const Dict<Symbol>& symbols )
        {
            TRY_DECL(is_null, eval_null(expr, symbols))
            return is_null ? ValueExpr(nullptr) : eval(expr, symbols);
        }
        
        template<typename T = ValueExpr>
        static Result<std::vector<T>> eval_items( const Expr& expr, const Dict<Symbol>& symbols, const size_t rank )
        {
            switch ( expr.kind )
            {
                case Expr::Expand:
                {
                    auto& expand = as_expand(expr);
                    return eval_items<T>(*expand.item, symbols, rank);
                }
                case Expr::List:
                {
                    auto& list = as_list(expr);
                    
                    std::vector<T> values(rank);
                    
                    size_t k = 0;
                    for ( auto& item : list.items )
                    {
                        if ( item->kind == Expr::Expand || item->kind == Expr::Range )
                        {
                            TRY_DECL(count, eval_rank(*item, symbols))
                            auto size = (size_t)count.as_int();
                            TRY_DECL(items, eval_items<T>(*item, symbols, size));
                            for ( size_t i = 0; i < size; ++i )
                            {
                                values[k++] = std::move(items[i]);
                            }
                        }
                        else
                        {
                            TRY_DECL(item, eval_item<T>(*item, symbols))
                            values[k++] = std::move(item);
                        }
                    }
                    return values;
                }
                case Expr::Contain:
                {
                    if constexpr( std::is_same_v<T,ValueExpr> )
                    {
                        auto& contain = as_contain(expr);
                        
                        TRY_DECL(length, eval_rank_static(*contain.pack, symbols))
                        if ( is_tensor_expr(*contain.item, symbols) )
                        {
                            TRY_DECL(pack, eval_items<Tensor*>(*contain.pack, symbols, length.as_int()))
                            TRY_DECL(items, eval_items<Tensor*>(*contain.item, symbols, rank))
                            
                            std::vector<ValueExpr> values(rank);
                            for ( size_t i = 0; i < rank; ++i )
                            {
                                values[i] = ValueExpr(std::find(pack.begin(), pack.end(), items[i]) != pack.end());
                            }
                            return values;
                        }
                        else
                        {
                            TRY_DECL(pack, eval_items(*contain.pack, symbols, length.as_int()))
                            TRY_DECL(values, eval_items(*contain.item, symbols, rank))
                            
                            for ( size_t i = 0; i < rank; ++i )
                            {
                                values[i] = ValueExpr(std::find(pack.begin(), pack.end(), values[i]) != pack.end());
                            }
                            return values;
                        }
                    }
                    break;
                }
                case Expr::Index:
                {
                    auto& indexer = as_index(expr);
                    auto index_type = eval_type(*indexer.index, symbols);
                    if ( index_type != Typename::Bool )
                    {
                        break;
                    }
                    
                    TRY_DECL(length, eval_rank_static(*indexer.index, symbols))
                    TRY_DECL(mask, eval_items(*indexer.index, symbols, length.as_int()))
                    if ( !is_literal(mask) )
                    {
                        return Error(expr.position, "index mask must not depend on dynamic shapes");
                    }
                    
                    const size_t count = std::count_if(mask.begin(), mask.end(), []( const ValueExpr& x ){ return x.as_bool();});
                    std::vector<T> values(count);
                    
                    TRY_DECL(items, eval_items<T>(*indexer.array, symbols, mask.size()))
                    
                    size_t k = 0;
                    for ( size_t i = 0; i < mask.size(); ++i )
                    {
                        if ( mask[i].as_bool() )
                        {
                            values[k++] = std::move(items[i]);
                        }
                    }
                    return values;
                }
                case Expr::Fold:
                {
                    auto& fold = as_fold(expr);
                    if constexpr( std::is_same_v<T,ValueExpr> )
                    {
                        TRY_DECL(items, eval_items(*fold.pack, symbols, rank))
                        if ( is_literal(items) )
                        {
                            auto type = eval_type(expr, symbols);
                            eval_accumulate(items, fold.op, type);
                            return items;
                        }
                    }
                    break;
                }
                case Expr::Substitute:
                {
                    auto& substitute = as_substitute(expr);
                    
                    TRY_DECL(index, eval(*substitute.index, symbols))
                    assert(index != nullptr);
                    if ( !is_literal(index) )
                    {
                        return Error(expr.position, "index must not depend on dynamic shapes in substitution expression");
                    }
                    
                    TRY_DECL(value, eval_items<T>(*substitute.pack, symbols, rank))
                    auto length = value.size();
                    if ( index.packed() )
                    {
                        auto size = (size_t)index.size().as_int();
                        TRY_DECL(items, eval_items<T>(*substitute.value, symbols, size))
                        for ( size_t i = 0; i < size; ++i )
                        {
                            auto idx = index[i].as_int();
                            TRY_CALL(check_index_range(idx, length, substitute.index->position))
                            value[idx] = std::move(items[i]);
                        }
                    }
                    else
                    {
                        TRY_DECL(item, eval_item<T>(*substitute.value, symbols))
                        auto idx = index.as_int();
                        TRY_CALL(check_index_range(idx, length, substitute.index->position))
                        value[idx] = std::move(item);
                    }
                    return value;
                }
                default:
                {
                    break;
                }
            }
            
            Cache cache;
            std::vector<T> values(rank);
            for ( size_t i = 0; i < rank; ++i )
            {
                TRY_DECL(item, eval_item<T>(expr, symbols, i, &cache));
                values[i] = item;
            }
            
            return values;
        }
        
    private:
        
        static Result<ValueExpr> eval_pack( const Expr& expr, const Dict<Symbol>& symbols, const ValueExpr& rank )
        {
            bool elementwise = expr.kind == Expr::Unary || expr.kind == Expr::Binary || expr.kind == Expr::Select || expr.kind == Expr::Cast;
            bool always_itemwise = expr.kind == Expr::Identity || expr.kind == Expr::Contain || expr.kind == Expr::Zip ||
                                   expr.kind == Expr::Substitute || expr.kind == Expr::Coalesce || expr.kind == Expr::Bounded;
            if ( rank.is_literal() && (elementwise || always_itemwise) )
            {
                TRY_DECL(items, eval_items(expr, symbols, rank.as_int()))
                if ( always_itemwise || is_literal(items) )
                {
                    auto type = eval_type(expr, symbols);
                    return ValueExpr::list(std::move(items), type);
                }
            }
            
            switch ( expr.kind )
            {
                case Expr::Identifier:
                {
                    auto& iden = as_identifier(expr);
                    auto& symbol = symbols.at(iden.name);
                    if ( symbol.is<ValueExpr>() )
                    {
                        return symbol.as<ValueExpr>();
                    }
                    break;
                }
                case Expr::Access:
                {
                    auto& access = as_access(expr);
                    TRY_DECL(tensor, eval(*access.tensor, symbols, nullptr, nullptr))
                    
                    size_t indices_rank = 0;
                    for ( auto& index : access.indices )
                    {
                        TRY_DECL(index_rank, shape_item_rank(*index, symbols))
                        indices_rank += index_rank;
                    }
                    
                    if ( indices_rank != tensor.rank() )
                    {
                        return Error(expr.position, "tensor index rank (%d) does not match shape rank (%d)",
                                     (int)indices_rank, (int)tensor.rank());
                    }
                    
                    bool packed = tensor.packed();
                    
                    size_t k = 0;
                    std::vector<ValueExpr> indices;
                    for ( auto& index : access.indices )
                    {
                        TRY_DECL(rank, shape_item_rank(*index, symbols))
                        if ( index->kind == Expr::Expand )
                        {
                            const Expr& item = *as_expand(*index).item;
                            TRY_DECL(ix, eval_items(item, symbols, rank))
                            
                            for ( size_t i = 0; i < rank; ++i )
                            {
                                indices.push_back(std::move(ix[i]));
                            }
                        }
                        else
                        {
                            TRY_DECL(ix, eval(*index, symbols))
                            indices.push_back(std::move(ix));
                            packed |= ix.packed();
                        }
                        k += rank;
                    }
                    
                    return ValueExpr(TensorAccess{ tensor, std::move(indices) }, tensor.dtype(), packed);
                }
                case Expr::Fold:
                {
                    auto& fold = as_fold(expr);
                    if ( !rank.is_literal() && !allows_dynamic_fold(fold.op) )
                    {
                        return Error(expr.position, "fold expression with operator '%s' must not be of dynamic length");
                    }
                    
                    TRY_DECL(pack, eval(*fold.pack, symbols))
                    if ( pack.is_list() )
                    {
                        auto& items = pack.as_list();
                        if ( is_literal(items) )
                        {
                            auto type = eval_type(expr, symbols);
                            eval_accumulate(items, fold.op, type);
                            return ValueExpr::list(std::move(items), type);
                        }
                    }
                    return ValueExpr::fold(Lexer::str(fold.op), pack, fold.cumulative);
                }
                case Expr::Expand:
                {
                    auto& expand = as_expand(expr);
                    if ( expand.count )
                    {
                        if ( !rank.is_literal() )
                        {
                            TRY_DECL(item, eval(*expand.item, symbols))
                            return ValueExpr::uniform(item, rank);
                        }
                    }
                    else
                    {
                        return eval_pack(*expand.item, symbols, rank);
                    }
                    break;
                }
                case Expr::Cast:
                {
                    auto& cast = as_cast(expr);
                    TRY_DECL(arg, eval(*cast.arg, symbols))
                    const Typename type = is_abstract(cast.base) ? symbols.at(cast.type).type : cast.base;
                    bool packed = arg.packed();
                    return ValueExpr(ValueExpr::CastExpr{ type, std::move(arg) }, type, packed);
                }
                case Expr::Unary:
                {
                    auto& unary = as_unary(expr);
                    TRY_DECL(arg, eval(*unary.arg, symbols))
                    auto type = arg.dtype();
                    bool packed = arg.packed();
                    return ValueExpr::unary(Lexer::str(unary.op), std::move(arg), type, packed);
                }
                case Expr::Binary:
                {
                    auto& binary = as_binary(expr);
                    TRY_DECL(left, eval(*binary.left, symbols))
                    TRY_DECL(right, eval(*binary.right, symbols))
                    auto type = left.dtype();
                    auto packed = left.packed() || right.packed();
                    return ValueExpr::binary(Lexer::str(binary.op), std::move(left), std::move(right), type, packed);
                }
                case Expr::Select:
                {
                    auto& select = as_select(expr);
                    TRY_DECL(cond, eval(*select.cond, symbols))
                    if ( is_literal(cond) )
                    {
                        if ( cond.is_list() )
                        {
                            Cache cache;
                            std::vector<ValueExpr> items(cond.as_list().size());
                            for ( size_t i = 0; i < items.size(); ++i )
                            {
                                TRY_DECL(item, eval_item(cond[i].as_bool() ? *select.left : *select.right, symbols, i, &cache))
                                items[i] = std::move(item);
                            }
                            auto type = eval_type(expr, symbols);
                            return ValueExpr::list(std::move(items), type);
                        }
                        else if ( cond.is_uniform() )
                        {
                            return eval(cond.as_uniform().value.as_bool() ? *select.left : *select.right, symbols);
                        }
                        else
                        {
                            return eval(cond.as_bool() ? *select.left : *select.right, symbols);
                        }
                    }
                    TRY_DECL(left, eval(*select.left, symbols))
                    TRY_DECL(right, eval(*select.right, symbols))
                    auto packed = cond.packed() || left.packed() || right.packed();
                    return ValueExpr::select(std::move(cond), std::move(left), std::move(right), packed);
                }
                case Expr::List:
                {
                    auto& list = as_list(expr);
                    auto type = eval_type(expr, symbols);
                    
                    if ( list.items.empty() )
                    {
                        return ValueExpr::list({}, type);
                    }
                    else if ( list.items.size() == 1 && list.items.front()->kind == Expr::Expand )
                    {
                        return eval(*list.items.front(), symbols);
                    }
                    
                    std::vector<ValueExpr> items(list.items.size());
                    for ( size_t i = 0; i < items.size(); ++i )
                    {
                        TRY_DECL(item, eval(*list.items[i], symbols))
                        items[i] = std::move(item);
                    }
                    
                    auto can_unroll = []( const ValueExpr& x ){ return !x.packed() || !x.has_dynamic_size(); };
                    if ( std::all_of(items.begin(), items.end(), can_unroll) )
                    {
                        size_t size = 0;
                        for ( auto& item : items )
                        {
                            size += item.packed() ? item.size().as_int() : 1;
                        }
                        
                        std::vector<ValueExpr> values(size);
                        size_t k = 0;
                        for ( auto& item : items )
                        {
                            if ( !item.packed() )
                            {
                                values[k++] = std::move(item);
                            }
                            else if ( item.is_list() )
                            {
                                auto& list = item.as_list();
                                for ( size_t i = 0; i < list.size(); ++i )
                                {
                                    values[k++] = std::move(list[i]);
                                }
                            }
                            else
                            {
                                auto size = (size_t)item.size().as_int();
                                for ( size_t i = 0; i < size; ++i )
                                {
                                    values[k++] = item.at(i);
                                }
                            }
                        }
                        return ValueExpr::list(std::move(values), type);
                    }
                    else
                    {
                        return ValueExpr(ValueExpr::ConcatExpr{ std::move(items) }, type, true);
                    }
                    break;
                }
                case Expr::Index:
                {
                    auto& indexer = as_index(expr);
                    if ( indexer.index->kind == Expr::Range )
                    {
                        auto& range = as_range(*indexer.index);
                        TRY_DECL(stride, range.stride ? eval(*range.stride, symbols) : ValueExpr(1))
                        if ( !stride.is_literal() && (!range.first || !range.last) )
                        {
                            return Error(expr.position, "range begin and end must be explicitly supplied if stride depends on dynamic shapes");
                        }
                        if ( stride == 0 )
                        {
                            return Error(expr.position, "zero stride in range");
                        }
                        
                        ValueExpr first;
                        if ( range.first )
                        {
                            TRY_MOVE(first, eval(*range.first, symbols))
                        }
                        else if ( stride.as_int() > 0 )
                        {
                            first = (int_t)0;
                        }
                        else if ( stride.as_int() < 0 )
                        {
                            TRY_DECL(length, eval_rank(*indexer.array, symbols))
                            first = length - 1;
                        }
                        else
                        {
                            return Error(expr.position, "zero stride in range");
                        }
                        
                        ValueExpr last;
                        if ( range.last )
                        {
                            TRY_MOVE(last, eval(*range.last, symbols))
                        }
                        else if ( stride.as_int() > 0 )
                        {
                            TRY_MOVE(last, eval_rank(*indexer.array, symbols))
                        }
                        else if ( stride.as_int() < 0 )
                        {
                            last = (int_t)-1;
                        }
                        else
                        {
                            return Error(expr.position, "zero stride in range");
                        }
                        
                        if ( first.is_literal() && last.is_literal() && stride.is_literal() )
                        {
                            break;
                        }
                        
                        TRY_DECL(pack, eval(*indexer.array, symbols))
                        auto type = pack.dtype();
                        return ValueExpr(ValueExpr::SliceExpr{ std::move(pack), std::move(first), std::move(last), std::move(stride) }, type, true);
                    }
                    break;
                }
                case Expr::Range:
                {
                    auto& range = as_range(expr);
                    TRY_DECL(first, range.first ? eval(*range.first, symbols) : ValueExpr(0))
                    TRY_DECL(last, eval(*range.last, symbols))
                    TRY_DECL(stride, range.stride ? eval(*range.stride, symbols) : ValueExpr(1))
                    if ( stride == 0 )
                    {
                        return Error(expr.position, "zero stride in range");
                    }
                    if ( first.is_literal() && last.is_literal() && stride.is_literal() )
                    {
                        break;
                    }
                    return ValueExpr(ValueExpr::RangeExpr{ std::move(first), std::move(last), std::move(stride) }, Typename::Int, true);
                }
                case Expr::Zip:
                {
                    return Error(expr.position, "zip expression is not allowed to have length that depends on dynamic shapes");
                }
                case Expr::Substitute:
                {
                    return Error(expr.position, "substitution expression is not allowed to have length that depends on dynamic shapes");
                }
                case Expr::Contain:
                {
                    return Error(expr.position, "'in' expression is not allowed to have length that depends on dynamic shapes");
                }
                default:
                {
                    assert(false);
                }
            }
            
            assert(rank.is_literal());
            TRY_DECL(items, eval_items(expr, symbols, rank.as_int()))
            auto type = eval_type(expr, symbols);
            return ValueExpr::list(std::move(items), type);
        }
        
        static Result<TensorRef> eval_tensor_pack( const Expr& expr, const Dict<Symbol>& symbols, const ValueExpr& rank, const AsTensorPack aspack )
        {
            switch ( expr.kind )
            {
                case Expr::Identifier:
                {
                    auto& iden = as_identifier(expr);
                    auto& symbol = symbols.at(iden.name);
                    assert( symbol.is<TensorRef>());
                    return symbol.as<TensorRef>();
                }
                case Expr::Select:
                {
                    auto& select = as_select(expr);
                    TRY_DECL(cond, eval(*select.cond, symbols))
                    assert(is_literal(cond));
                    assert(!cond.packed());
                    return eval_tensor_pack(cond.as_bool() ? *select.left : *select.right, symbols, rank, aspack);
                }
                case Expr::List:
                {
                    auto& list = as_list(expr);
                    TRY_DECL(tensors, eval_items<Tensor*>(expr, symbols, list.items.size()))
                    TRY_DECL(shape, eval_shape_from_expr(expr, symbols))
                    auto type = eval_type(expr, symbols);
                    return aspack(tensors, type, shape);
                }
                default:
                {
                    return Error(expr.position, "invalid tensor pack expr");
                }
            }
        }
        
        static bool can_eval_itemwise( const Expr& expr, const Dict<Symbol>& symbols )
        {
            switch ( expr.kind )
            {
                case Expr::Identifier:
                {
                    auto& iden = as_identifier(expr);
                    auto& symbol = symbols.at(iden.name);
                    if ( symbol.is<ValueExpr>() )
                    {
                        auto& value = symbol.as<ValueExpr>();
                        return value.is_list() || value.is_shape_access();
                    }
                    return true;
                }
                case Expr::Fold:
                {
                    auto& fold = as_fold(expr);
                    return !fold.cumulative;
                }
                case Expr::List:
                {
                    auto& list = as_list(expr);
                    return std::all_of(list.items.begin(), list.items.end(), []( const Shared<Expr> item )
                    {
                        return item->kind != Expr::Expand && item->kind != Expr::Range;
                    });
                }
                default:
                {
                    return true;
                }
            }
        }
        
        template<typename T = ValueExpr>
        static Result<T> eval_item( const Expr& expr, const Dict<Symbol>& symbols )
        {
            return eval_item<T>(expr, symbols, std::nullopt, nullptr);
        }
        
        template<typename T = ValueExpr, typename = std::enable_if_t<std::is_same_v<T,ValueExpr>>>
        static Result<ValueExpr> eval_item( const Expr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            if constexpr( std::is_same_v<T,ValueExpr> )
            {
                if ( idx && !can_eval_itemwise(expr, symbols) )
                {
                    auto it = cache->find(&expr);
                    if ( it == cache->end() )
                    {
                        TRY_DECL(value, eval(expr, symbols))
                        it = cache->emplace(&expr, std::move(value)).first;
                    }
                    return it->second.at(*idx);
                }
            }
            
            switch ( expr.kind )
            {
                case Expr::Literal:
                {
                    return eval_item(as_literal(expr));
                }
                case Expr::Identifier:
                {
                    return eval_item(as_identifier(expr), symbols, idx, cache);
                }
                case Expr::List:
                {
                    return eval_item(as_list(expr), symbols, idx, cache);
                }
                case Expr::Expand:
                {
                    return eval_item(as_expand(expr), symbols, idx, cache);
                }
                case Expr::Index:
                {
                    return eval_item(as_index(expr), symbols, idx, cache);
                }
                case Expr::Access:
                {
                    return eval_item(as_access(expr), symbols, idx, cache);
                }
                case Expr::Range:
                {
                    return eval_item(as_range(expr), symbols, idx, cache);
                }
                case Expr::Zip:
                {
                    return eval_item(as_zip(expr), symbols, idx, cache);
                }
                case Expr::Unary:
                {
                    return eval_item(as_unary(expr), symbols, idx, cache);
                }
                case Expr::Binary:
                {
                    return eval_item(as_binary(expr), symbols, idx, cache);
                }
                case Expr::Select:
                {
                    return eval_item(as_select(expr), symbols, idx, cache);
                }
                case Expr::Coalesce:
                {
                    return eval_item(as_coalesce(expr), symbols, idx, cache);
                }
                case Expr::Identity:
                {
                    return eval_item(as_identity(expr), symbols, idx, cache);
                }
                case Expr::Contain:
                {
                    return eval_item(as_contain(expr), symbols, idx, cache);
                }
                case Expr::Fold:
                {
                    return eval_item(as_fold(expr), symbols, idx, cache);
                }
                case Expr::Cast:
                {
                    return eval_item(as_cast(expr), symbols, idx, cache);
                }
                case Expr::Builtin:
                {
                    return eval_item(as_builtin(expr), symbols, idx, cache);
                }
                case Expr::Format:
                {
                    return eval_item(as_format(expr), symbols, idx, cache);
                }
                case Expr::Bounded:
                {
                    return eval_item(as_bounded(expr), symbols, idx, cache);
                }
                case Expr::Substitute:
                {
                    return eval_item(as_substitute(expr), symbols, idx, cache);
                }
            }
            return ValueExpr::null();
        }
        
        template<typename T, typename = std::enable_if_t<std::is_same_v<T,Tensor*>>>
        static Result<Tensor*> eval_item( const Expr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            switch ( expr.kind )
            {
                case Expr::Identifier:
                {
                    return eval_item<Tensor*>(as_identifier(expr), symbols, idx, cache);
                }
                case Expr::List:
                {
                    return eval_item<Tensor*>(as_list(expr), symbols, idx, cache);
                }
                case Expr::Expand:
                {
                    return eval_item<Tensor*>(as_expand(expr), symbols, idx, cache);
                }
                case Expr::Index:
                {
                    return eval_item<Tensor*>(as_index(expr), symbols, idx, cache);
                }
                case Expr::Zip:
                {
                    return eval_item<Tensor*>(as_zip(expr), symbols, idx, cache);
                }
                case Expr::Select:
                {
                    return eval_item<Tensor*>(as_select(expr), symbols, idx, cache);
                }
                case Expr::Coalesce:
                {
                    return eval_item<Tensor*>(as_coalesce(expr), symbols, idx, cache);
                }
                case Expr::Substitute:
                {
                    return eval_item<Tensor*>(as_substitute(expr), symbols, idx, cache);
                }
                default:
                {
                    return Error(expr.position, "invalid expr eval");
                }
            }
        }
        
        static Result<ValueExpr> eval_item( const LiteralExpr& expr )
        {
            switch ( expr.type.name )
            {
                case Typename::Int:
                {
                    return ValueExpr(static_cast<const IntExpr&>(expr).value);
                }
                case Typename::Real:
                {
                    return ValueExpr(static_cast<const RealExpr&>(expr).value);
                }
                case Typename::Bool:
                {
                    return ValueExpr(static_cast<const BoolExpr&>(expr).value);
                }
                case Typename::Str:
                {
                    return ValueExpr(static_cast<const StrExpr&>(expr).value);
                }
                default:
                {
                    return Error(expr.position, "invalid literal eval");
                }
            }
        }
        
        template<typename T = ValueExpr, typename = std::enable_if_t<std::is_same_v<T,ValueExpr>>>
        static Result<ValueExpr> eval_item( const IdentifierExpr& iden, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            auto& symbol = symbols.at(iden.name);
            if ( symbol.is_runtime() )
            {
                auto kind = symbol.kind == Symbol::Index ? ValueExpr::IdentifierKind::LoopIndex : ValueExpr::IdentifierKind::LoopLocal;
                return ValueExpr::identifier(symbol.packed() ? iden.name + "_" + std::to_string(*idx + 1) : iden.name, kind, symbol.type);
            }
            auto& value = symbol.as<ValueExpr>();
            if ( value.packed() )
            {
                assert(idx);
                return value.at(*idx);
            }
            else
            {
                return value;
            }
        }
        
        template<typename T, typename = std::enable_if_t<std::is_same_v<T,Tensor*>>>
        static Result<Tensor*> eval_item( const IdentifierExpr& iden, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            auto& symbol = symbols.at(iden.name);
            auto& values = symbol.as<TensorRef>();
            return values.packed() ? values[*idx] : (Tensor*)&*values;
        }
        
        template<typename T, typename = std::enable_if_t<std::is_same_v<T,TensorRef>>>
        static Result<TensorRef> eval( const IdentifierExpr& iden, const Dict<Symbol>& symbols, const std::optional<size_t> idx = std::nullopt )
        {
            auto& symbol = symbols.at(iden.name);
            auto& values = symbol.as<TensorRef>();
            return idx && values.packed() ? TensorRef(values[*idx]) : values;
        }
        
        template<typename T = ValueExpr>
        static Result<T> eval_item( const ListExpr& list, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            size_t k = 0;
            for ( auto& item : list.items )
            {
                if ( item->kind == Expr::Expand || item->kind == Expr::Range )
                {
                    TRY_DECL(rank, eval_rank(*item, symbols))
                    assert(rank.is_literal());
                    if ( idx < k + rank.as_int() )
                    {
                        return eval_item<T>(*item, symbols, *idx - k, cache);
                    }
                    k += rank.as_int();
                }
                else
                {
                    if ( idx == k++ )
                    {
                        return eval_item<T>(*item, symbols);
                    }
                }
            }
            return T();
        }
        
        template<typename T = ValueExpr>
        static Result<T> eval_item( const ExpandExpr& expand, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            if ( expand.count )
            {
                TRY_DECL(rank, eval_rank(*expand.item, symbols))
                return rank != nullptr ? eval_item<T>(*expand.item, symbols, idx, cache) : eval_item<T>(*expand.item, symbols);
            }
            else
            {
                return eval_item<T>(*expand.item, symbols, idx, cache);
            }
        }
        
        template<typename T = ValueExpr>
        static Result<T> eval_item( const IndexExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            if constexpr( std::is_same_v<T,ValueExpr> )
            {
                auto array_type = eval_type(*expr.array, symbols);
                if ( array_type == Typename::Str )
                {
                    TRY_DECL(array, eval_item(*expr.array, symbols, idx, cache))
                    auto& str = array.as_str();
                    auto length = (int_t)str.length();
                    
                    if ( expr.index->kind == Expr::Range )
                    {
                        auto& range = as_range(*expr.index);
                        
                        TRY_DECL(stride, range.stride ? eval_item(*range.stride, symbols) : ValueExpr(1))
                        if ( !stride.is_literal() )
                        {
                            return Error(expr.index->position, "index stride must not depend on dynamic shapes");
                        }
                        
                        auto stride_value = stride.as_int();
                        
                        TRY_DECL(first, range.first ? eval_item(*range.first, symbols) : 
                                 ValueExpr(stride_value < 0 ? length - 1 : 0))
                        TRY_DECL(last, range.last ? eval_item(*range.last, symbols) :
                                 ValueExpr(stride_value < 0 ? -1 : length))
                        
                        if ( !first.is_literal() || !last.is_literal() )
                        {
                            return Error(expr.index->position, "index range must not depend on dynamic shapes");
                        }
                        
                        auto first_index = first.as_int();
                        auto last_index = last.as_int();
                        
                        if ( range.first )
                        {
                            TRY_CALL(check_index_range(first_index, length, expr.index->position))
                        }
                        if ( range.last )
                        {
                            TRY_CALL(check_index_range(last_index, length, expr.index->position))
                        }
                        
                        if ( stride_value == 1 )
                        {
                            return ValueExpr((str_t)str.substr(first_index, last_index - first_index));
                        }
                        
                        std::string result;
                        const size_t count = (last_index - first_index) / stride_value;
                        for ( size_t k = 0, i = first_index; k < count; ++k, i += stride_value )
                        {
                            result += str[i];
                        }
                        return ValueExpr((str_t)result);
                    }
                    else
                    {
                        TRY_DECL(rank, eval_rank_static(*expr.index, symbols))
                        if ( rank != nullptr )
                        {
                            auto size = (size_t)rank.as_int();
                            TRY_DECL(index_value, eval_items(*expr.index, symbols, size))
                            
                            std::string result;
                            for ( size_t i = 0; i < size; ++i )
                            {
                                if ( !index_value[i].is_literal() )
                                {
                                    return Error(expr.index->position, "index must not depend on dynamic shapes");
                                }
                                
                                auto index = index_value[i].as_int();
                                TRY_CALL(check_index_range(index, length, expr.index->position))
                                
                                result += str[index];
                            }
                            return ValueExpr((str_t)result);
                        }
                        else
                        {
                            TRY_DECL(index_value, eval_item(*expr.index, symbols))
                            if ( !index_value.is_literal() )
                            {
                                return Error(expr.index->position, "index must not depend on dynamic shapes");
                            }
                            
                            auto index = index_value.as_int();
                            TRY_CALL(check_index_range(index, length, expr.index->position))
                            
                            return ValueExpr((str_t)str.substr(index, 1));
                        }
                    }
                }
            }
            
            auto index_type = eval_type(*expr.index, symbols);
            if ( index_type == Typename::Bool )
            {
                TRY_DECL(mask, eval(*expr.index, symbols))
                if ( !is_literal(mask) )
                {
                    return Error(expr.position, "index mask must not depend on dynamic shapes");
                }
                
                size_t k = 0;
                auto size = (size_t)mask.size().as_int();
                for ( size_t i = 0; i < size; ++i )
                {
                    if ( mask[i].as_bool() )
                    {
                        if ( k++ == idx )
                        {
                            return eval_item<T>(*expr.array, symbols, i, cache);
                        }
                    }
                }
                return T();
            }
            else
            {
                TRY_DECL(rank, eval_rank(*expr.array, symbols))
                
                ValueExpr index_value;
                if ( expr.index->kind == Expr::Range )
                {
                    TRY_MOVE(index_value, eval_item(as_range(*expr.index), symbols, idx, cache, rank))
                }
                else
                {
                    TRY_MOVE(index_value, eval_item(*expr.index, symbols, idx, cache))
                }
                
                if ( !index_value.is_literal() )
                {
                    if constexpr( std::is_same_v<T,ValueExpr> )
                    {
                        TRY_DECL(value, eval(*expr.array, symbols))
                        return value.at(index_value);
                    }
                    else
                    {
                        return Error(expr.position, "index to tensor pack must not depend on dynamic shapes");
                    }
                }
                else if ( !rank.is_literal() )
                {
                    return Error(expr.position, "indexing a dynamic pack with static index is not allowed");
                }
                
                auto index = index_value.as_int();
                TRY_CALL(check_index_range(index, rank.as_int(), expr.index->position))
                
                return eval_item<T>(*expr.array, symbols, index, cache);
            }
        }
        
        static Result<ValueExpr> eval_item( const AccessExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            ValueExpr item;
            TensorRef tensor;
            if ( expr.tensor->kind == Expr::Index )
            {
                auto& indexer = as_index(*expr.tensor);
                TRY_DECL(subscript, eval_item(*indexer.index, symbols, idx, cache))
                if ( subscript.is_literal() )
                {
                    TRY_MOVE(tensor, eval<TensorRef>(as_identifier(*indexer.array), symbols, subscript.as_int()))
                }
                else
                {
                    TRY_MOVE(tensor, eval<TensorRef>(as_identifier(*indexer.array), symbols))
                    item = std::move(subscript);
                }
            }
            else
            {
                TRY_MOVE(tensor, eval<TensorRef>(as_identifier(*expr.tensor), symbols, idx))
            }
            
            size_t indices_rank = 0;
            for ( auto& index : expr.indices )
            {
                TRY_DECL(index_rank, shape_item_rank(*index, symbols))
                indices_rank += index_rank;
            }
            
            if ( indices_rank != tensor.rank() )
            {
                return Error(expr.position, "tensor index rank (%d) does not match shape rank (%d)",
                             (int)indices_rank, (int)tensor.rank());
            }
            
            std::vector<ValueExpr> indices;
            for ( auto& index : expr.indices )
            {
                if ( index->kind == Expr::Expand )
                {
                    const Expr& item = *as_expand(*index).item;
                    TRY_DECL(repeats, shape_item_rank(*index, symbols))
                    TRY_DECL(ix, eval_items(item, symbols, repeats))
                    
                    for ( size_t i = 0; i < repeats; ++i )
                    {
                        indices.push_back(std::move(ix[i]));
                    }
                }
                else
                {
                    TRY_DECL(ix, eval_item(*index, symbols, idx, cache))
                    indices.push_back(std::move(ix));
                }
            }
            
            return ValueExpr(TensorAccess{ tensor, std::move(indices), std::move(item) }, tensor.dtype());
        }
        
        static Result<ValueExpr> eval_item( const RangeExpr& range, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache,
                                           const ValueExpr& length = nullptr )
        {
            TRY_DECL(stride, range.stride ? eval_item(*range.stride, symbols) : ValueExpr(1))
            if ( stride == 0 )
            {
                return Error(range.position, "zero stride in range");
            }
            assert(stride.is_literal());
            assert(range.first || stride.as_int() >= 0 || length != nullptr);
            TRY_DECL(first, range.first ? eval_item(*range.first, symbols) : stride.as_int() < 0 ? length - 1 : ValueExpr(0))
            return first + (int_t)*idx * stride.as_int();
        }
        
        template<typename T = ValueExpr>
        static Result<T> eval_item( const ZipExpr& zip, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            auto k = *idx % zip.items.size();
            auto i = *idx / zip.items.size();
            return eval_item<T>(*zip.items[k], symbols, i, cache);
        }
        
        static Result<ValueExpr> eval_item( const UnaryExpr& unary, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            if ( unary.op == Lexer::Operator::Question )
            {
                auto& symbol = symbols.at(as_identifier(*unary.arg).name);
                return ValueExpr(!symbol.is_null());
            }
            
            TRY_DECL(arg, eval_item(*unary.arg, symbols, idx, cache))
            assert(arg != nullptr);
            
            if ( unary.op == Lexer::Operator::Plus )
            {
                return arg;
            }
            if ( arg.is_literal() )
            {
                return fold_constants_unary(unary.op, arg);
            }
            else
            {
                return ValueExpr::unary(Lexer::str(unary.op), std::move(arg));
            }
        }
        
        static Result<ValueExpr> eval_item( const BinaryExpr& binary, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            TRY_DECL(left, eval_item(*binary.left, symbols, idx, cache))
            assert(left != nullptr);
            
            if ( left.is_literal() )    // shortcuts without evaluating right
            {
                if ( (binary.op == Lexer::Operator::And && left.as_bool() == false) || (binary.op == Lexer::Operator::Or && left.as_bool() == true) )
                {
                    return left;
                }
                if ( binary.op == Lexer::Operator::Imply && left.as_bool() == false )
                {
                    return ValueExpr(true);
                }
            }
            
            TRY_DECL(right, eval_item(*binary.right, symbols, idx, cache))
            assert(right != nullptr);
            
            if ( right.is_literal() )   // error checks
            {
                if ( binary.op == Lexer::Operator::Divide || binary.op == Lexer::Operator::CeilDivide || binary.op == Lexer::Operator::Modulo )
                {
                    bool zero_div = right.dtype() == Typename::Real ? right.as_real() == 0 : right.as_int() == 0;
                    if ( zero_div )
                    {
                        return Error(binary.position, "division by zero");
                    }
                }
            }
            
            if ( left.is_literal() && right.is_literal() )
            {
                return fold_constants_binary(binary.op, left, right);
            }
            
            bool altered = false;
            auto simplified = simplify_binary(binary.op, left, right, altered);
            if ( simplified != nullptr )
            {
                return simplified;
            }
            else
            {
                auto type = Lexer::is_comparison(binary.op) ? Typename::Bool : left.dtype();
                return ValueExpr::binary(Lexer::str(binary.op), std::move(left), std::move(right), type);
            }
        }
        
        template<typename T = ValueExpr>
        static Result<T> eval_item( const SelectExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            TRY_DECL(cond, eval_item(*expr.cond, symbols, idx, cache))
            assert(cond != nullptr);
            
            if ( cond.is_literal() )
            {
                return cond.as_bool() ? eval_item<T>(*expr.left, symbols, idx, cache) : expr.right ? eval_item<T>(*expr.right, symbols, idx, cache) : T();
            }
            else if constexpr ( std::is_same_v<T,ValueExpr> )
            {
                TRY_DECL(left, eval_item(*expr.left, symbols, idx, cache))
                TRY_DECL(right, eval_item(*expr.right, symbols, idx, cache))
                return ValueExpr::select(std::move(cond), std::move(left), std::move(right));
            }
            else
            {
                return Error(expr.position, "condition in tensor expression must not depend on dynamic tensor shapes");
            }
        }
        
        template<typename T = ValueExpr>
        static Result<T> eval_item( const CoalesceExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            TRY_DECL(is_null, eval_null(*expr.condition, symbols))
            return !is_null ? eval_item<T>(*expr.condition, symbols, idx, cache) : eval_item<T>(*expr.alternate, symbols, idx, cache);
        }
        
        static Result<ValueExpr> eval_item( const IdentityExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            if ( is_tensor_expr(*expr.left, symbols) )
            {
                TRY_DECL(left, eval_item<Tensor*>(*expr.left, symbols, idx, cache))
                TRY_DECL(right, eval_item<Tensor*>(*expr.right, symbols, idx, cache))
                
                return ValueExpr(left == right);
            }
            else
            {
                TRY_DECL(left, eval_item(*expr.left, symbols, idx, cache))
                TRY_DECL(right, eval_item(*expr.right, symbols, idx, cache))
                
                return ValueExpr(canonical(left) == canonical(right));
            }
        }
        
        static Result<ValueExpr> eval_item( const ContainExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            TRY_DECL(length, eval_rank_static(*expr.pack, symbols))
            
            if ( is_tensor_expr(*expr.item, symbols) )
            {
                TRY_DECL(lookup, eval_item<Tensor*>(*expr.item, symbols, idx, cache))
                TRY_DECL(items, eval_items<Tensor*>(*expr.pack, symbols, length.as_int()))
                
                return ValueExpr(std::find(items.begin(), items.end(), lookup) != items.end());
            }
            else
            {
                TRY_DECL(lookup, eval_item(*expr.item, symbols, idx, cache))
                TRY_DECL(items, eval_items(*expr.pack, symbols, length.as_int()))
                
                return ValueExpr(std::find(items.begin(), items.end(), lookup) != items.end());
            }
        }
        
        static Result<ValueExpr> eval_item( const FoldExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            TRY_DECL(rank, eval_rank(*expr.pack, symbols))
            if ( !rank.is_literal() && allows_dynamic_fold(expr.op) )
            {
                TRY_DECL(pack, eval(*expr.pack, symbols))
                return ValueExpr::fold(Lexer::str(expr.op), std::move(pack));
            }
            
            const Typename type = eval_type(expr, symbols);
            switch ( expr.op )
            {
                case Lexer::Operator::Plus:
                {
                    auto count = expr.cumulative ? *idx + 1 : rank.as_int();
                    return type == Typename::Real ? eval_fold<std::plus>(expr, symbols, count, cache, (real_t)0) :
                                                    eval_fold<std::plus>(expr, symbols, count, cache, (int_t)0);
                }
                case Lexer::Operator::Multiply:
                {
                    auto count = expr.cumulative ? *idx + 1 : rank.as_int();
                    return type == Typename::Real ? eval_fold<std::multiplies>(expr, symbols, count, cache, (real_t)1) :
                                                    eval_fold<std::multiplies>(expr, symbols, count, cache, (int_t)1);
                }
                case Lexer::Operator::Min:
                {
                    auto count = expr.cumulative ? *idx + 1 : rank.as_int();
                    if ( count == 0 )
                    {
                        return Error(expr.position, "invalid min fold over empty pack");
                    }
                    return type == Typename::Real ? eval_fold<minimize,real_t>(expr, symbols, count, cache) :
                                                    eval_fold<minimize,int_t>(expr, symbols, count, cache);
                }
                case Lexer::Operator::Max:
                {
                    auto count = expr.cumulative ? *idx + 1 : rank.as_int();
                    if ( count == 0 )
                    {
                        return Error(expr.position, "invalid max fold over empty pack");
                    }
                    return type == Typename::Real ? eval_fold<maximize,real_t>(expr, symbols, count, cache) :
                                                    eval_fold<maximize,int_t>(expr, symbols, count, cache);
                }
                case Lexer::Operator::And:
                {
                    auto count = expr.cumulative ? *idx + 1 : rank.as_int();
                    return eval_fold<std::logical_and>(expr, symbols, count, cache, (bool_t)true);
                }
                case Lexer::Operator::Or:
                {
                    auto count = expr.cumulative ? *idx + 1 : rank.as_int();
                    return eval_fold<std::logical_or>(expr, symbols, count, cache, (bool_t)false);
                }
                case Lexer::Operator::Less:
                {
                    return eval_is_sorted<std::less>(*expr.pack, symbols);
                }
                case Lexer::Operator::Greater:
                {
                    return eval_is_sorted<std::greater>(*expr.pack, symbols);
                }
                case Lexer::Operator::LessEqual:
                {
                    return eval_is_sorted<std::less_equal>(*expr.pack, symbols);
                }
                case Lexer::Operator::GreaterEqual:
                {
                    return eval_is_sorted<std::greater_equal>(*expr.pack, symbols);
                }
                case Lexer::Operator::Equal:
                {
                    return eval_is_uniform(*expr.pack, symbols);
                }
                case Lexer::Operator::NotEqual:
                {
                    return eval_is_unique(*expr.pack, symbols);
                }
                case Lexer::Operator::MakeEqual:
                {
                    return eval_uniform_value(*expr.pack, symbols);
                }
                default:
                {
                    break;
                }
            }
            return Error(expr.position, "invalid fold eval");
        }
        
        static Result<ValueExpr> eval_item( const CastExpr& cast, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            const Typename type = is_abstract(cast.base) ? symbols.at(cast.type).type : cast.base;
            
            if ( !cast.arg )
            {
                switch ( type )
                {
                    case Typename::Int:
                    {
                        return ValueExpr((int_t)0);
                    }
                    case Typename::Real:
                    {
                        return ValueExpr((real_t)0.0);
                    }
                    case Typename::Bool:
                    {
                        return ValueExpr((bool_t)false);
                    }
                    case Typename::Str:
                    {
                        return ValueExpr((str_t)"");
                    }
                    default:
                    {
                        return Error(cast.position, "invalid default value eval");
                    }
                }
            }
            
            TRY_DECL(arg, eval_item(*cast.arg, symbols, idx, cache))
            assert(arg != nullptr);
            
            if ( arg.dtype() == type )
            {
                return arg;
            }
            
            if ( type == Typename::Int )
            {
                if ( arg == inf() || arg == -inf() )
                {
                    return Error(cast.position, "invalid casting of infinity to int");
                }
            }
            
            if ( !arg.is_literal() )
            {
                return ValueExpr(ValueExpr::CastExpr{ type, std::move(arg) }, type);
            }
            
            switch ( type )
            {
                case Typename::Int:
                {
                    return ValueExpr(arg.dtype() == Typename::Real ? (int_t)arg.as_real() : (int_t)arg.as_bool());
                }
                case Typename::Real:
                {
                    return ValueExpr(arg.dtype() == Typename::Int ? (real_t)arg.as_int() : (real_t)arg.as_bool());
                }
                case Typename::Bool:
                {
                    return ValueExpr(arg.dtype() == Typename::Int ? (bool_t)arg.as_int() : (bool_t)arg.as_real());
                }
                default:
                {
                    break;
                }
            }
            return Error(cast.position, "invalid cast eval");
        }
        
        static Result<ValueExpr> eval_item( const BuiltinExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            TRY_DECL(arg, eval_item(*expr.arg, symbols, idx, cache))
            assert(arg != nullptr);
            
            if ( !arg.is_literal() )
            {
                return ValueExpr::unary(expr.func, std::move(arg));
            }
            
            if ( arg.dtype() == Typename::Int )
            {
                return eval_unary_func(expr.func, arg.as_int());
            }
            else if ( arg.dtype() == Typename::Real )
            {
                return eval_unary_func(expr.func, arg.as_real());
            }
            else
            {
                return Error(expr.position, "invalid builtin eval");
            }
        }
        
        static Result<ValueExpr> eval_item( const FormatExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            std::string result;
            size_t offset = 0;
            for ( auto& sub : expr.subs )
            {
                result += expr.str.substr(offset, sub.first - offset);
                TRY_DECL(value, eval_item(*sub.second, symbols, idx, cache))
                assert(value != nullptr);
                if ( !value.is_literal() )
                {
                    return Error(expr.position, "string formatting must not depend on dynamic shapes");
                }
                result += str(value);
                offset = sub.first;
            }
            result += expr.str.substr(offset);
            return ValueExpr(result);
        }
        
        static Result<ValueExpr> eval_item( const BoundedExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            TRY_DECL(arg, eval_item(*expr.index, symbols, idx, cache))
            ValueExpr lower, upper;
            if ( expr.lower_value )
            {
                TRY_MOVE(lower, eval_item(*expr.lower_value, symbols, idx, cache))
            }
            if ( expr.upper_value )
            {
                TRY_MOVE(upper, eval_item(*expr.upper_value, symbols, idx, cache))
            }
            return ValueExpr::bounded(std::move(arg), std::move(lower), std::move(upper));
        }
        
        template<typename T = ValueExpr>
        static Result<T> eval_item( const SubstituteExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx, Cache* cache )
        {
            TRY_DECL(index, eval(*expr.index, symbols))
            assert(index != nullptr);
            if ( !is_literal(index) )
            {
                return Error(expr.position, "index must not depend on dynamic shapes in substitution expression");
            }
            
            TRY_DECL(rank, eval_rank_static(*expr.pack, symbols))
            auto length = (size_t)rank.as_int();
            
            if ( index.packed() )
            {
                auto size = (size_t)index.size().as_int();
                for ( size_t k = 0; k < size; ++k )
                {
                    auto i = index[k].as_int();
                    TRY_CALL(check_index_range(i, length, expr.index->position))
                    
                    if ( i == idx )
                    {
                        return eval_item<T>(*expr.value, symbols, k, cache);
                    }
                }
            }
            else
            {
                auto i = index.as_int();
                TRY_CALL(check_index_range(i, length, expr.index->position))
                
                if ( i == idx )
                {
                    return eval_item<T>(*expr.value, symbols);
                }
            }
            return eval_item<T>(*expr.pack, symbols, idx, cache);
        }
        
    private:
        
        template<template<typename> class F, typename T>
        static Result<ValueExpr> eval_fold( const FoldExpr& fold, const Dict<Symbol>& symbols, const size_t count, Cache* cache,
                                           const std::optional<T> init = std::nullopt )
        {
            const F<T> func;
            auto literal_value = init;
            ValueExpr expr_value;
            TRY_DECL(items, eval_items(*fold.pack, symbols, count))
            
            size_t terms = 0;
            for ( auto& item : items )
            {
                if ( item.is_literal() )
                {
                    if ( is_shortcut(fold.op, (T)item) )
                    {
                        return item;
                    }
                    if ( !literal_value )
                    {
                        ++terms;
                    }
                    literal_value = !literal_value ? (T)item : func(*literal_value, (T)item);
                }
                else
                {
                    expr_value = expr_value == nullptr ? item : ValueExpr::binary(Lexer::str(fold.op), std::move(expr_value), item);
                    ++terms;
                }
                
                if ( terms > 2 )
                {
                    return ValueExpr::fold(Lexer::str(fold.op), ValueExpr::list(std::move(items), typename_of<T>::value));
                }
            }
            
            if ( expr_value != nullptr && literal_value && !is_nop(fold.op, *literal_value) )
            {
                return ValueExpr::binary(Lexer::str(fold.op), std::move(expr_value), ValueExpr(*literal_value));
            }
            else
            {
                return expr_value != nullptr ? expr_value : literal_value ? ValueExpr(*literal_value) : ValueExpr::null();
            }
        }
        
        template<template<typename> class F, typename T>
        static Result<ValueExpr> eval_fold( const FoldExpr& fold, const Dict<Symbol>& symbols, const size_t count, Cache* cache, const T init )
        {
            return eval_fold<F>(fold, symbols, count, cache, std::optional<T>(init));
        }
        
        template<template<typename> class F, typename T>
        static Result<ValueExpr> eval_arg_fold( const FoldExpr& fold, const Dict<Symbol>& symbols )
        {
            const F<T> func;
            TRY_DECL(rank, eval_rank(*fold.pack, symbols))
            if ( rank == 0 )
            {
                return ValueExpr(nullptr);
            }
            
            auto size = (size_t)rank.as_int();
            
            TRY_DECL(items, eval_items(*fold.pack, symbols, size))
            
            T value = (T)items.front();
            size_t idx = 0;
            for ( size_t i = 1; i < size; ++i )
            {
                auto new_value = func(value, (T)items[i]);
                if ( new_value != value )
                {
                    idx = i;
                    value = new_value;
                }
            }
            return ValueExpr((int_t)idx);
        }
        
        template<template<typename> class F, typename T>
        static void eval_accumulate( std::vector<ValueExpr>& items )
        {
            const F<T> func;
            for ( size_t i = 1; i < items.size(); ++i )
            {
                items[i] = func((T)items[i-1], (T)items[i]);
            }
        }
        
        static void eval_accumulate( std::vector<ValueExpr>& items, const Lexer::Operator op, const Typename type )
        {
            switch ( op )
            {
                case Lexer::Operator::Plus:
                {
                    if ( type == Typename::Real )
                    {
                        eval_accumulate<std::plus,real_t>(items);
                    }
                    else
                    {
                        eval_accumulate<std::plus,int_t>(items);
                    }
                    break;
                }
                case Lexer::Operator::Multiply:
                {
                    if ( type == Typename::Real )
                    {
                        eval_accumulate<std::multiplies,real_t>(items);
                    }
                    else
                    {
                        eval_accumulate<std::multiplies,int_t>(items);
                    }
                    break;
                }
                case Lexer::Operator::Min:
                {
                    if ( type == Typename::Real )
                    {
                        eval_accumulate<minimize,real_t>(items);
                    }
                    else
                    {
                        eval_accumulate<minimize,int_t>(items);
                    }
                    break;
                }
                case Lexer::Operator::Max:
                {
                    if ( type == Typename::Real )
                    {
                        eval_accumulate<maximize,real_t>(items);
                    }
                    else
                    {
                        eval_accumulate<maximize,int_t>(items);
                    }
                    break;
                }
                case Lexer::Operator::And:
                {
                    eval_accumulate<std::logical_and,bool_t>(items);
                    break;
                }
                case Lexer::Operator::Or:
                {
                    eval_accumulate<std::logical_or,bool_t>(items);
                    break;
                }
                default:
                {
                    assert(false);
                }
            }
        }
        
        template<template<typename> class C>
        static Result<ValueExpr> eval_is_sorted( const Expr& expr, const Dict<Symbol>& symbols )
        {
            TRY_DECL(rank, eval_rank(expr, symbols))
            TRY_DECL(items, eval_items(expr, symbols, rank.as_int()))
            auto type = eval_type(expr, symbols);
            switch ( type )
            {
                case Typename::Int:
                {
                    return ValueExpr(is_sorted<C,int_t>(items));
                }
                case Typename::Real:
                {
                    return ValueExpr(is_sorted<C,real_t>(items));
                }
                case Typename::Bool:
                {
                    return ValueExpr(is_sorted<C,bool_t>(items));
                }
                case Typename::Str:
                {
                    return ValueExpr(is_sorted<C,str_t>(items));
                }
                default:
                {
                    return Error(expr.position, "invalid sorted eval");
                }
            }
        }
        
        static Result<ValueExpr> eval_is_unique( const Expr& expr, const Dict<Symbol>& symbols )
        {
            TRY_DECL(rank, eval_rank(expr, symbols))
            TRY_DECL(items, eval_items(expr, symbols, rank.as_int()))
            auto type = eval_type(expr, symbols);
            switch ( type )
            {
                case Typename::Int:
                {
                    return ValueExpr(is_unique<int_t>(items));
                }
                case Typename::Real:
                {
                    return ValueExpr(is_unique<real_t>(items));
                }
                case Typename::Bool:
                {
                    return ValueExpr(is_unique<bool_t>(items));
                }
                case Typename::Str:
                {
                    return ValueExpr(is_unique<str_t>(items));
                }
                default:
                {
                    return Error(expr.position, "invalid unique eval");
                }
            }
        }
        
        static Result<ValueExpr> eval_is_uniform( const Expr& expr, const Dict<Symbol>& symbols )
        {
            TRY_DECL(value, eval_uniform_value(expr, symbols))
            return ValueExpr(value != nullptr);
        }
        
        static Result<ValueExpr> eval_uniform_value( const Expr& expr, const Dict<Symbol>& symbols )
        {
            TRY_DECL(rank, eval_rank(expr, symbols))
            if ( rank == nullptr )
            {
                return eval_item(expr, symbols);
            }
            else if ( rank == 0 )
            {
                return eval_uniform_value_empty(expr, symbols);
            }
            else
            {
                TRY_DECL(pack, eval_pack(expr, symbols, rank))
                return uniform_value(pack);
            }
        }
        
        static Result<ValueExpr> eval_uniform_value_empty( const Expr& expr, const Dict<Symbol>& symbols )
        {
            switch ( expr.kind )
            {
                case Expr::Zip:
                {
                    auto& zip = as_zip(expr);
                    TRY_DECL(value,  eval_uniform_value(*zip.items.front(), symbols))
                    for ( size_t i = 1; i < zip.items.size(); ++i )
                    {
                        TRY_DECL(item,  eval_uniform_value(*zip.items[i], symbols))
                        if ( item != value )
                        {
                            return ValueExpr(nullptr);
                        }
                    }
                    return value;
                }
                case Expr::Substitute:
                {
                    auto& substitute = as_substitute(expr);
                    TRY_DECL(pack,  eval_uniform_value(*substitute.pack, symbols))
                    TRY_DECL(value,  eval_uniform_value(*substitute.value, symbols))
                    return value == pack ? value : ValueExpr(nullptr);
                }
                case Expr::Contain:
                {
                    auto& contain = as_contain(expr);
                    TRY_DECL(item,  eval_uniform_value(*contain.item, symbols))
                    TRY_DECL(pack, eval(*contain.pack, symbols))
                    if ( pack.is_list() )
                    {
                        auto& items = pack.as_list();
                        return ValueExpr(std::find(items.begin(), items.end(), item) != items.end());
                    }
                    else
                    {
                        auto value = uniform_value(pack);
                        return value != nullptr ? ValueExpr(value == item) : ValueExpr(nullptr);
                    }
                }
                default:
                {
                    TRY_DECL(pack, eval_pack(expr, symbols, 0))
                    return uniform_value(pack);
                }
            }
        }
        
        static ValueExpr uniform_value( const ValueExpr& expr )
        {
            switch ( expr.kind() )
            {
                case ValueExpr::Null:
                case ValueExpr::Literal:
                case ValueExpr::Identifier:
                case ValueExpr::Placeholder:
                case ValueExpr::SizeAccess:
                {
                    return expr;
                }
                case ValueExpr::TensorAccess:
                {
                    return expr.packed() ? nullptr : expr;
                }
                case ValueExpr::Reference:
                {
                    auto& reference = expr.as_reference();
                    return uniform_value(*reference.target);
                }
                case ValueExpr::ShapeAccess:
                {
                    auto& access = expr.as_shape_access();
                    if ( !access.tensor.packed() )
                    {
                        return expr.packed() ? nullptr : expr;
                    }
                    if ( access.tensor.itemized() && access.tensor.item_count() == 0 )
                    {
                        return nullptr;
                    }
                    if ( !access.dim.is_literal() )
                    {
                        return nullptr;
                    }
                    auto dim = access.dim.as_int();
                    auto& extent = access.tensor.shape()[dim];
                    if ( extent.is_placeholder() )
                    {
                        return access;
                    }
                    return uniform_value(extent);
                }
                case ValueExpr::Cast:
                {
                    auto& cast = expr.as_cast();
                    auto arg = uniform_value(cast.arg);
                    return arg == nullptr ? nullptr : ValueExpr(ValueExpr::CastExpr{ cast.dtype, std::move(arg) }, cast.dtype);
                }
                case ValueExpr::Unary:
                {
                    auto& unary = expr.as_unary();
                    auto arg = uniform_value(unary.arg);
                    return arg == nullptr ? nullptr : ValueExpr::unary(unary.op, std::move(arg));
                }
                case ValueExpr::Binary:
                {
                    auto& binary = expr.as_binary();
                    auto left = uniform_value(binary.left);
                    auto right = uniform_value(binary.right);
                    return left == nullptr || right == nullptr ? nullptr : ValueExpr::binary(binary.op, left, right);
                }
                case ValueExpr::Select:
                {
                    auto& select = expr.as_select();
                    auto cond = uniform_value(select.cond);
                    if ( cond != nullptr )
                    {
                        return cond.as_bool() ? uniform_value(select.left) : uniform_value(select.right);
                    }
                    else
                    {
                        auto left = uniform_value(select.left);
                        auto right = uniform_value(select.right);
                        return left == right ? left : nullptr;
                    }
                }
                case ValueExpr::Fold:
                {
                    auto& fold = expr.as_fold();
                    if ( !fold.cumulative )
                    {
                        return expr;
                    }
                    auto arg = uniform_value(fold.pack);
                    switch ( Lexer::operator_value(fold.op) )
                    {
                        case Lexer::Operator::Plus:
                        {
                            return arg == (int_t)0 || arg == (real_t)0 ? arg : nullptr;
                        }
                        case Lexer::Operator::Multiply:
                        {
                            return arg == (int_t)1 || arg == (real_t)1 ? arg : nullptr;
                        }
                        case Lexer::Operator::Min:
                        case Lexer::Operator::Max:
                        case Lexer::Operator::And:
                        case Lexer::Operator::Or:
                        {
                            return arg;
                        }
                        default:
                        {
                            return nullptr;
                        }
                    }
                }
                case ValueExpr::List:
                {
                    auto& items = expr.as_list();
                    if ( items.empty() )
                    {
                        return nullptr;
                    }
                    auto first = canonical(items.front());
                    bool is_uniform = std::all_of(items.begin() + 1, items.end(), [&first]( const ValueExpr& item )
                    {
                        return canonical(item) == first;
                    });
                    return is_uniform ? items.front() : nullptr;
                }
                case ValueExpr::Concat:
                {
                    auto& concat = expr.as_concat();
                    ValueExpr value = nullptr;
                    for ( auto& item : concat.items )
                    {
                        auto item_value = uniform_value(item);
                        if ( item_value == nullptr )
                        {
                            return item_value;
                        }
                        else if ( value == nullptr )
                        {
                            value = item_value;
                        }
                        else if ( value != item_value )
                        {
                            return nullptr;
                        }
                    }
                    return value;
                }
                case ValueExpr::Slice:
                {
                    auto& slice = expr.as_slice();
                    return uniform_value(slice.pack);
                }
                case ValueExpr::Subscript:
                {
                    auto& subscript = expr.as_subscript();
                    return uniform_value(subscript.pack);
                }
                case ValueExpr::Uniform:
                {
                    auto& uniform = expr.as_uniform();
                    return uniform.value;
                }
                case ValueExpr::Range:
                {
                    auto& range = expr.as_range();
                    return range.stride == 0 ? range.first : nullptr;
                }
                case ValueExpr::Bounded:
                {
                    return nullptr;
                }
            }
        }
        
    public:
        
        template<bool Optional = false>
        static Result<ValueExpr> eval_rank_static( const Expr& expr, const Dict<Symbol>& symbols )
        {
            TRY_DECL(rank, eval_rank<Optional>(expr, symbols))
            if ( rank != nullptr && !rank.is_literal() )
            {
                return Error(expr.position, "length of packed expression must not be dynamic in this context");
            }
            return rank;
        }
        
        template<bool Optional = false>
        static Result<ValueExpr> eval_rank( const Expr& expr, const Dict<Symbol>& symbols )
        {
            if constexpr( Optional )
            {
                TRY_DECL(is_null, eval_null(expr, symbols))
                if ( is_null )
                {
                    return ValueExpr(nullptr);
                }
            }
            switch ( expr.kind )
            {
                case Expr::Identifier:
                {
                    auto& iden = as_identifier(expr);
                    auto& symbol = symbols.at(iden.name);
                    return symbol.size;
                }
                case Expr::List:
                {
                    auto& list = as_list(expr);
                    int_t literal_rank = 0;
                    ValueExpr dynamic_rank;
                    for ( auto& item : list.items )
                    {
                        if ( item->kind == Expr::Expand || item->kind == Expr::Range )
                        {
                            TRY_DECL(len, eval_rank(*item, symbols))
                            if ( len.is_literal() )
                            {
                                literal_rank += len.as_int();
                            }
                            else
                            {
                                dynamic_rank = dynamic_rank == nullptr ? len : dynamic_rank + len;
                            }
                        }
                        else
                        {
                            literal_rank += 1;
                        }
                    }
                    return dynamic_rank != nullptr ? dynamic_rank + literal_rank : ValueExpr(literal_rank);
                }
                case Expr::Expand:
                {
                    auto& expand = as_expand(expr);
                    if ( expand.count )
                    {
                        TRY_DECL(count, eval_item(*expand.count, symbols))
                        if ( count.dtype() == Typename::Bool )
                        {
                            if ( count.is_literal() )
                            {
                                return ValueExpr(count.as_bool() ? 1 : 0);
                            }
                            else
                            {
                                return ValueExpr::select(count, 1, 0);
                            }
                        }
                        return count;
                    }
                    else
                    {
                        return eval_rank(*expand.item, symbols);
                    }
                }
                case Expr::Index:
                {
                    auto& index = as_index(expr);
                    auto array_type = eval_type(*index.array, symbols);
                    auto index_type = eval_type(*index.index, symbols);
                    if ( array_type == Typename::Str )
                    {
                        return ValueExpr(nullptr);
                    }
                    else if ( index_type == Typename::Bool )
                    {
                        TRY_DECL(array_length, eval_rank_static(*index.array, symbols))
                        TRY_DECL(index_length, eval_rank_static(*index.index, symbols))
                        if ( index_length != array_length )
                        {
                            return Error(expr.position, "incompatible mask length and pack length (%d vs %d)",
                                         (int)index_length.as_int(), (int)array_length.as_int());
                        }
                        
                        TRY_DECL(mask, eval_items(*index.index, symbols, index_length.as_int()))
                        if ( !is_literal(mask) )
                        {
                            return Error(expr.position, "index mask must not depend on dynamic shapes");
                        }
                        return ValueExpr((int_t)std::count_if(mask.begin(), mask.end(), []( const ValueExpr& item ){ return item.as_bool(); }));
                    }
                    else if ( index.index->kind == Expr::Range )
                    {
                        auto& range = as_range(*index.index);
                        TRY_DECL(length, eval_rank(*index.array, symbols))
                        
                        TRY_DECL(stride, range.stride ? eval_item(*range.stride, symbols) : ValueExpr(1))
                        
                        TRY_DECL(first, range.first ? eval_item(*range.first, symbols) :
                                        ValueExpr(stride.as_int() < 0 ? length - 1 : 0))
                        TRY_DECL(last, range.last ? eval_item(*range.last, symbols) :
                                        ValueExpr(stride.as_int() < 0 ? -1 : length))
                        if ( range.first && first.is_int() && first.as_int() < 0 )
                        {
                            first = first + length;
                        }
                        if ( range.last && last.is_int() && last.as_int() < 0 )
                        {
                            last = last + length;
                        }
                        
                        return ceil_div(last - first, stride);
                    }
                    else
                    {
                        return eval_rank(*index.index, symbols);
                    }
                }
                case Expr::Access:
                {
                    auto& access = as_access(expr);
                    TRY_DECL(tensor_rank, eval_rank(*access.tensor, symbols))
                    if ( tensor_rank != nullptr )
                    {
                        return tensor_rank;
                    }
                    
                    size_t k = 0;
                    for ( auto& index : access.indices )
                    {
                        if ( index->kind != Expr::Expand )
                        {
                            TRY_DECL(index_rank, eval_rank(*index, symbols))
                            if ( index_rank != nullptr )
                            {
                                return index_rank;
                            }
                        }
                        TRY_DECL(rank, shape_item_rank(*index, symbols))
                        k += rank;
                    }
                    return ValueExpr::null();
                }
                case Expr::Range:
                {
                    auto& range = as_range(expr);
                    TRY_DECL(first, range.first ? eval_item(*range.first, symbols) : ValueExpr(0))
                    TRY_DECL(last, eval_item(*range.last, symbols))
                    TRY_DECL(stride, range.stride ? eval_item(*range.stride, symbols) : ValueExpr(1))
                    return ceil_div(last - first, stride);
                }
                case Expr::Zip:
                {
                    auto& zip = as_zip(expr);
                    ValueExpr rank;
                    for ( size_t i = 0; i < zip.items.size(); ++i )
                    {
                        TRY_DECL(item_rank, eval_rank(*zip.items[i], symbols))
                        if ( item_rank != nullptr )
                        {
                            if ( rank == nullptr )
                            {
                                rank = item_rank;
                            }
                            else if ( item_rank != rank )
                            {
                                return Error(expr.position, "incompatible pack lengths in zip expression ('%s' vs '%s')",
                                             str(rank).c_str(), str(item_rank).c_str());
                            }
                        }
                    }
                    return rank == nullptr ? rank : ValueExpr(rank.as_int() * (int_t)zip.items.size());
                }
                case Expr::Unary:
                {
                    auto& unary = as_unary(expr);
                    if ( unary.op == Lexer::Operator::Question )
                    {
                        return ValueExpr::null();
                    }
                    return eval_rank(*unary.arg, symbols);
                }
                case Expr::Binary:
                {
                    auto& binary = as_binary(expr);
                    TRY_DECL(left, eval_rank(*binary.left, symbols))
                    TRY_DECL(right, eval_rank(*binary.right, symbols))
                    if ( left != nullptr && right != nullptr && left != right )
                    {
                        return Error(expr.position, "incompatible pack lengths in binary expression '%s' ('%s' vs '%s')",
                                     Lexer::str(binary.op), str(left).c_str(), str(right).c_str());
                    }
                    return left != nullptr ? left : right;
                }
                case Expr::Select:
                {
                    auto& select = as_select(expr);
                    TRY_DECL(cond, eval_rank(*select.cond, symbols))
                    TRY_DECL(left, eval_rank(*select.left, symbols))
                    TRY_DECL(right, select.right ? eval_rank(*select.right, symbols) : ValueExpr::null())
                    if ( cond != nullptr )
                    {
                        if ( left != nullptr && right != nullptr && left != right )
                        {
                            return Error(expr.position, "incompatible pack lengths in select expression "
                                         "('%s' vs '%s' for left and right results)", str(left).c_str(), str(right).c_str());
                        }
                        if ( left != nullptr && left != cond )
                        {
                            return Error(expr.position, "incompatible pack lengths in select expression"
                                         "('%s' vs '%s' for left result and condition)", str(left).c_str(), str(cond).c_str());
                        }
                        if ( right != nullptr && right != cond )
                        {
                            return Error(expr.position, "incompatible pack lengths in select expression "
                                         "('%s' vs '%s' for right result and condition)", str(right).c_str(), str(cond).c_str());
                        }
                        return cond;
                    }
                    else if ( left == nullptr )
                    {
                        return right;
                    }
                    else if ( right == nullptr )
                    {
                        return left;
                    }
                    else
                    {
                        TRY_DECL(val, eval(*select.cond, symbols))
                        if ( !val.is_literal() )
                        {
                            return Error(expr.position, "expression rank must not depend on dynamic shape expression");
                        }
                        return val.as_bool() ? left : right;
                    }
                }
                case Expr::Coalesce:
                {
                    auto& coalesce = as_coalesce(expr);
                    TRY_DECL(is_null, eval_null(*coalesce.condition, symbols))
                    if ( !is_null )
                    {
                        return eval_rank(*coalesce.condition, symbols);
                    }
                    else
                    {
                        return eval_rank(*coalesce.alternate, symbols);
                    }
                }
                case Expr::Identity:
                {
                    auto& identity = as_identity(expr);
                    TRY_DECL(left, eval_rank(*identity.left, symbols))
                    TRY_DECL(right, eval_rank(*identity.right, symbols))
                    if ( left != nullptr && right != nullptr && left != right )
                    {
                        return Error(expr.position, "incompatible pack lengths in 'is' expression ('%s' vs '%s')",
                                     str(left).c_str(), str(right).c_str());
                    }
                    return left != nullptr ? left : right;
                }
                case Expr::Contain:
                {
                    auto& contain = as_contain(expr);
                    return eval_rank(*contain.item, symbols);
                }
                case Expr::Fold:
                {
                    auto& fold = as_fold(expr);
                    return fold.cumulative ? eval_rank(*fold.pack, symbols) : ValueExpr(nullptr);
                }
                case Expr::Cast:
                {
                    auto& cast = as_cast(expr);
                    return cast.arg ? eval_rank(*cast.arg, symbols) : ValueExpr(nullptr);
                }
                case Expr::Builtin:
                {
                    auto& builtin = as_builtin(expr);
                    return eval_rank(*builtin.arg, symbols);
                }
                case Expr::Bounded:
                {
                    auto& bounded = as_bounded(expr);
                    TRY_DECL(index_rank, eval_rank(*bounded.index, symbols))
                    
                    if ( bounded.lower_value )
                    {
                        TRY_DECL(lower_rank, eval_rank(*bounded.lower_value, symbols))
                        if ( index_rank != nullptr && lower_rank != nullptr && lower_rank != index_rank )
                        {
                            return Error(expr.position, "incompatible pack lengths in bounded expression for index and lower value ('%s' vs '%s')",
                                         str(index_rank).c_str(), str(lower_rank).c_str());
                        }
                    }
                    if ( bounded.upper_value )
                    {
                        TRY_DECL(upper_rank, eval_rank(*bounded.upper_value, symbols))
                        if ( index_rank != nullptr && upper_rank != nullptr && upper_rank != index_rank )
                        {
                            return Error(expr.position, "incompatible pack lengths in bounded expression for index and upper value ('%s' vs '%s')",
                                         str(index_rank).c_str(), str(upper_rank).c_str());
                        }
                    }
                    return index_rank;
                }
                case Expr::Substitute:
                {
                    auto& substitute = as_substitute(expr);
                    TRY_DECL(pack_rank, eval_rank(*substitute.pack, symbols))
                    TRY_DECL(index_rank, eval_rank(*substitute.index, symbols))
                    TRY_DECL(value_rank, eval_rank(*substitute.value, symbols))
                    if ( index_rank != nullptr && value_rank != nullptr && index_rank != value_rank )
                    {
                        return Error(expr.position, "incompatible pack lengths in substitution expression for index and value ('%s' vs '%s')",
                                     str(index_rank).c_str(), str(value_rank).c_str());
                    }
                    return pack_rank;
                }
                default:
                {
                    return ValueExpr::null();
                }
            }
        }
        
        static Typename eval_type( const Expr& expr, const Dict<Symbol>& symbols )
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
                    return symbols.at(name).type;
                }
                case Expr::List:
                {
                    auto& list = as_list(expr);
                    for ( auto& item : list.items )
                    {
                        auto type = eval_type(*item, symbols);
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
                    return eval_type(*expand.item, symbols);
                }
                case Expr::Index:
                {
                    auto& index = as_index(expr);
                    return eval_type(*index.array, symbols);
                }
                case Expr::Access:
                {
                    auto& access = as_access(expr);
                    return eval_type(*access.tensor, symbols);
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
                        auto type = eval_type(*item, symbols);
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
                    return logical ? Typename::Bool : eval_type(*unary.arg, symbols);
                }
                case Expr::Binary:
                {
                    auto& binary = as_binary(expr);
                    if ( Lexer::is_comparison(binary.op) )
                    {
                        return Typename::Bool;
                    }
                    auto type = eval_type(*binary.left, symbols);
                    return type != Typename::Type ? type : eval_type(*binary.right, symbols);
                }
                case Expr::Select:
                {
                    auto& select = as_select(expr);
                    auto type = eval_type(*select.left, symbols);
                    return type != Typename::Type ? type : select.right ? eval_type(*select.right, symbols) : Typename::Type;
                }
                case Expr::Coalesce:
                {
                    auto& coalesce = as_coalesce(expr);
                    auto type = eval_type(*coalesce.condition, symbols);
                    return type != Typename::Type ? type : eval_type(*coalesce.alternate, symbols);
                }
                case Expr::Identity:
                case Expr::Contain:
                {
                    return Typename::Bool;
                }
                case Expr::Fold:
                {
                    auto& fold = as_fold(expr);
                    return eval_type(*fold.pack, symbols);
                }
                case Expr::Cast:
                {
                    auto& cast = as_cast(expr);
                    return is_abstract(cast.base) ? symbols.at(cast.type).type : cast.base;
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
                    return eval_type(*substitute.pack, symbols);
                }
                case Expr::Bounded:
                {
                    auto& bounded = as_bounded(expr);
                    return eval_type(*bounded.index, symbols);
                }
            }

            assert(false);
            return Typename::Type;
        }
        
        static Result<bool> eval_null( const Expr& expr, const Dict<Symbol>& symbols )
        {
            if ( expr.kind == Expr::Identifier )
            {
                auto& name = as_identifier(expr).name;
                auto it = symbols.find(name);
                if ( it == symbols.end() )
                {
                    return false;   // this is a loop index/local, never null
                }
                return it->second.is_null();
            }
            else if ( expr.kind == Expr::Select )
            {
                auto& select = as_select(expr);
                TRY_DECL(cond, eval_optional(*select.cond, symbols))
                if ( cond == nullptr )
                {
                    return true;
                }
                if ( cond.is_literal() )
                {
                    return cond.as_bool() ? eval_null(*select.left, symbols) : select.right ? eval_null(*select.right, symbols) : true;
                }
                else
                {
                    if ( !select.right )
                    {
                        return Error(expr.position, "select condition must be compile-time expression when right hand side is omitted");
                    }
                    TRY_DECL(left_null, eval_null(*select.left, symbols))
                    TRY_DECL(right_null, eval_null(*select.right, symbols))
                    return left_null || right_null;
                }
            }
            else if ( expr.kind == Expr::Fold )
            {
                auto& fold = as_fold(expr);
                if ( fold.op == Lexer::Operator::Min || fold.op == Lexer::Operator::Max )
                {
                    TRY_DECL(is_null, eval_null(*fold.pack, symbols))
                    if ( is_null )
                    {
                        return true;
                    }
                    TRY_DECL(rank, eval_rank(*fold.pack, symbols))
                    return rank == 0;
                }
            }
            else if ( expr.kind == Expr::Coalesce )
            {
                return false;
            }
            else if ( expr.kind == Expr::Unary && as_unary(expr).op == Lexer::Operator::Question )
            {
                return false;
            }
            return any_recurse_result(expr, [&]( const Expr& e ){ return eval_null(e, symbols); });
        }
        
        static Result<size_t> shape_rank( const Shapedef& shape, const Dict<Symbol>& symbols )
        {
            size_t sum = 0;
            for ( auto& extent : shape.extents )
            {
                if ( extent.value )
                {
                    TRY_DECL(rank, shape_item_rank(*extent.value, symbols))
                    sum += rank;
                }
                else
                {
                    sum += 1;
                }
            }
            return sum;
        }
        
        static Result<size_t> shape_item_rank( const Expr& item, const Dict<Symbol>& symbols )
        {
            if ( item.kind == Expr::Expand )
            {
                auto& expand = as_expand(item);
                if ( expand.count )
                {
                    TRY_DECL(value, eval(*expand.count, symbols))
                    return value.is_bool() ? value.as_bool() ? 1 : 0 : value.as_int();
                }
                else
                {
                    TRY_DECL(rank, eval_rank_static(item, symbols))
                    return rank.as_int();
                }
            }
            else
            {
                return 1;
            }
        }
        
        Result<ValueExpr> eval_shape_expr( const Expr& expr, const Dict<Symbol>& symbols )
        {
            TRY_DECL(rank, eval_rank(expr, symbols))
            if ( rank == nullptr )
            {
                TRY_DECL(value, eval_item(expr, symbols))
                simplify(value);
                TRY_CALL(check_shape_expr(value, expr.position))
                return value;
            }
            else if ( rank.is_literal() )
            {
                TRY_DECL(items, eval_items(expr, symbols, rank.as_int()))
                for ( auto& item : items )
                {
                    simplify(item);
                    TRY_CALL(check_shape_expr(item, expr.position))
                }
                auto type = eval_type(expr, symbols);
                auto value = ValueExpr::list(std::move(items), type);
                return value;
            }
            else    // spread
            {
                TRY_DECL(value, eval_pack(expr, symbols, rank))
                if ( value.is_list() )
                {
                    for ( auto& item : value.as_list() )
                    {
                        simplify(item);
                        TRY_CALL(check_shape_expr(item, expr.position))
                    }
                }
                else if ( value.is_uniform() )
                {
                    auto& item = value.as_uniform().value;
                    simplify(item);
                    TRY_CALL(check_shape_expr(item, expr.position))
                }
                return value;
            }
        }
        
        Result<void> check_shape_expr( const ValueExpr& value, const Position& position )
        {
            if ( value.is_int() && value.as_int() < 0 )
            {
                return Error(position, "extent must be non-negative; found %d", (int)value.as_int());
            }
            return {};
        }
        
        static bool is_tensor_expr( const Expr& expr, const Dict<Symbol>& symbols )
        {
            switch ( expr.kind )
            {
                case Expr::Identifier:
                {
                    auto it = symbols.find(as_identifier(expr).name);
                    return it != symbols.end() && (it->second.is<TensorRef>() || it->second.is_runtime());
                }
                case Expr::Unary:
                {
                    auto& unary = as_unary(expr);
                    if ( unary.op == Lexer::Operator::Question )
                    {
                        return false;
                    }
                    break;
                }
                case Expr::Identity:
                case Expr::Contain:
                case Expr::Access:
                {
                    return false;
                }
                default:
                {
                    break;
                }
            }
            return any_recurse(expr, [&]( const Expr& e ){ return is_tensor_expr(e, symbols); });
        }
        
        static bool is_literal( const std::vector<ValueExpr>& exprs )
        {
            return std::all_of(exprs.begin(), exprs.end(), []( const ValueExpr& x ){ return x.is_literal(); });
        }
        
        static bool is_shape_access( const std::vector<ValueExpr>& exprs )
        {
            return std::all_of(exprs.begin(), exprs.end(), []( const ValueExpr& x ){ return x.is_shape_access(); });
        }
        
        static bool is_literal( const ValueExpr& expr )
        {
            if ( expr.is_list() )
            {
                auto& items = expr.as_list();
                return is_literal(items);
            }
            else if ( expr.is_uniform() )
            {
                auto& uniform = expr.as_uniform();
                return uniform.value.is_literal() && uniform.size.is_literal();
            }
            return expr.is_literal();
        }
        
        static bool is_shape_access( const ValueExpr& expr )
        {
            if ( expr.is_list() )
            {
                auto& items = expr.as_list();
                return is_shape_access(items);
            }
            else if ( expr.is_uniform() )
            {
                auto& uniform = expr.as_uniform();
                return uniform.value.is_shape_access() && uniform.size.is_literal();
            }
            return expr.is_shape_access();
        }
        
        static bool is_implicit_constant( const TensorRef& tensor )
        {
            return tensor.is_constant() && tensor.name().front() == '.';
        }
        
        static bool allows_dynamic_fold( const Lexer::Operator op )
        {
            return op == Lexer::Operator::Plus || op == Lexer::Operator::Multiply ||
                   op == Lexer::Operator::Min || op == Lexer::Operator::Max ||
                   op == Lexer::Operator::And || op == Lexer::Operator::Or;
        }
        
    private:
        
        template<template<typename> class C, typename T>
        static bool is_sorted( const std::vector<ValueExpr>& items )
        {
            const C<T> cmp;
            return std::is_sorted(items.begin(), items.end(), [&cmp]( const ValueExpr& x, const ValueExpr& y )
            {
                return cmp((T)x, (T)y);
            });
        }
        
        template<typename T>
        static bool is_unique( std::vector<ValueExpr>& items )
        {
            auto cmp = []( const ValueExpr& x, const ValueExpr& y )
            {
                return (T)x < (T)y;
            };
            std::sort(items.begin(), items.end(), cmp);
            return std::is_sorted(items.begin(), items.end(), cmp);
        }
        
        template<typename T>
        static bool is_shortcut( const Lexer::Operator op, const T value )
        {
            switch ( op )
            {
                case Lexer::Operator::And:
                {
                    return value == false;
                }
                case Lexer::Operator::Or:
                {
                    return value == true;
                }
                case Lexer::Operator::Multiply:
                {
                    return value == (T)0;
                }
                default:
                {
                    return false;
                }
            }
        }
        
        template<typename T>
        static bool is_nop( const Lexer::Operator op, const T arg )
        {
            switch ( op )
            {
                case Lexer::Operator::Plus:
                {
                    return arg == (T)0;
                }
                case Lexer::Operator::Multiply:
                {
                    return arg == (T)1;
                }
                case Lexer::Operator::And:
                {
                    return arg == true;
                }
                case Lexer::Operator::Or:
                {
                    return arg == false;
                }
                default:
                {
                    return false;
                }
            }
        }
        
    protected:
        
        static Result<Shape> eval_shape_from_expr( const Expr& expr, const Dict<Symbol>& symbols )
        {
            if ( is_empty_list(expr) )
            {
                return Error(expr.position, "could not deduce shape information from empty list");
            }
            if ( !is_tensor_expr(expr, symbols) )
            {
                return Shape();
            }
            
            switch ( expr.kind )
            {
                case Expr::Identifier:
                {
                    auto& iden = as_identifier(expr);
                    auto& symbol = symbols.at(iden.name);
                    return symbol.is<TensorRef>() ? symbol.as<TensorRef>().shape() : Shape();
                }
                case Expr::List:
                {
                    auto& list = as_list(expr);
                    return eval_shape_from_items(list.items, symbols, expr.position);
                }
                case Expr::Expand:
                {
                    auto& expand = as_expand(expr);
                    return eval_shape_from_expr(*expand.item, symbols);
                }
                case Expr::Zip:
                {
                    auto& zip = as_zip(expr);
                    return eval_shape_from_items(zip.items, symbols, expr.position);
                }
                case Expr::Index:
                {
                    auto& index = as_index(expr);
                    return eval_shape_from_expr(*index.array, symbols);
                }
                case Expr::Select:
                {
                    auto& select = as_select(expr);
                    TRY_DECL(value, eval(*select.cond, symbols))
                    return value.as_bool() ? eval_shape_from_expr(*select.left, symbols) : eval_shape_from_expr(*select.right, symbols);
                }
                case Expr::Coalesce:
                {
                    auto& coalesce = as_coalesce(expr);
                    TRY_DECL(is_null, eval_null(*coalesce.condition, symbols))
                    return !is_null ? eval_shape_from_expr(*coalesce.condition, symbols) : eval_shape_from_expr(*coalesce.alternate, symbols);
                }
                default:
                {
                    return Shape();
                }
            }
        }
        
        static Result<Shape> eval_shape_from_items( const std::vector<Shared<Expr>>& items, const Dict<Symbol>& symbols, const Position& position )
        {
            Shape shape;
            Shape canonic_shape;
            for ( size_t i = 0; i < items.size(); ++i )
            {
                TRY_DECL(item_shape, eval_shape_from_expr(*items[i], symbols))
                if ( i == 0 )
                {
                    shape = item_shape;
                    canonic_shape = canonical(item_shape);
                }
                else
                {
                    if ( item_shape.size() != shape.size() )
                    {
                        return Error(position, "could not deduce shape information from items of non-uniform rank");
                    }
                    for ( size_t k = 0; k < shape.size(); ++k )
                    {
                        if ( shape[k].is_list() )
                        {
                            shape[k].as_list()[i] = item_shape[k];
                        }
                        else if ( canonic_shape[k] != canonical(item_shape[k]) )
                        {
                            shape[k] = ValueExpr::list(shape[k], items.size());
                            shape[k].as_list()[i] = item_shape[k];
                        }
                    }
                }
            }
            return shape;
        }
        
        static void resolve( Shape& shape, const unsigned flags = ResolveAll )
        {
            for ( auto& item : shape )
            {
                resolve(item, flags);
            }
        }
        
        static Shape resolved( const Shape& shape, const unsigned flags = ResolveAll )
        {
            Shape resolved = shape;
            resolve(resolved, flags);
            return resolved;
        }
        
        static void simplify( Shape& shape )
        {
            for ( auto& item : shape )
            {
                simplify(item);
            }
        }
        
        static Shape simplified( const Shape& shape )
        {
            Shape simple = shape;
            simplify(simple);
            return simple;
        }
        
        static void canonify( Shape& shape )
        {
            for ( auto& item : shape )
            {
                canonify(item);
            }
        }
        
        static Shape canonical( const Shape& shape )
        {
            Shape canonic = shape;
            canonify(canonic);
            return canonic;
        }
        
        static bool equivalent( const ValueExpr& x, const ValueExpr& y )
        {
            return x == y || canonical(x) == canonical(y);
        }
        
        static bool equivalent( const Shape& x, const Shape& y )
        {
            if ( x.size() != y.size() )
            {
                return false;
            }
            for ( size_t i = 0; i < x.size(); ++i )
            {
                if ( !equivalent(x[i], y[i]) )
                {
                    return false;
                }
            }
            return true;
        }
        
        static bool all_equal( const std::vector<ValueExpr>& exprs )
        {
            for ( size_t i = 1; i < exprs.size(); ++i )
            {
                if ( exprs[i] != exprs.front() )
                {
                    return false;
                }
            }
            return true;
        }
        
        static bool has_undefined_symbols( const Shapedef& shape, const Dict<Symbol>& symbols )
        {
            for ( auto& extent : shape.extents )
            {
                if ( extent.value && has_undefined_symbols(*extent.value, symbols) )
                {
                    return true;
                }
            }
            return false;
        }
        
        static bool has_undefined_symbols( const Expr& expr, const Dict<Symbol>& symbols )
        {
            return any_of(expr, [&]( const Expr& e )
            {
                return e.kind == Expr::Identifier && !symbols.count(as_identifier(e).name);
            });
        }
        
    private:
        
        static Result<void> check_index_range( int_t& index, const size_t length, const Position& position )
        {
            if ( index < 0 )
            {
                index += length;
            }
            if ( index < 0 || index >= length )
            {
                return Error(position, "index %d is out of bounds [0,%d)", (int)index, (int)length);
            }
            return {};
        }
    };

}   // namespace sknd


#undef DECL_BOUNDS

#endif
