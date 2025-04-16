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
        
        typedef std::vector<ValueExpr> Shape;
        
        typedef std::vector<Tensor*> Tensors;
        typedef std::function<TensorRef(const ValueExpr&, const Typename&)> AsTensor;
        typedef std::function<TensorRef(const Tensors&, const Typename&, const Shape&, const std::vector<int_t>&, const ValueExpr&)> AsTensorPack;
        
    protected:
        
        struct LoopIndex {};
        struct LoopLocal {};
        
        struct Symbol : public std::variant<ValueExpr,TensorRef,LoopIndex,LoopLocal>
        {
            typedef std::variant<ValueExpr,TensorRef,LoopIndex,LoopLocal> variant_type;
            
            Typename type;
            std::optional<size_t> rank;
            ValueExpr size;
            
            Symbol( const ValueExpr& value, const Typename type )
                : variant_type(value), type(type), rank(value.max_size_or_null()), 
                  size(value.packed() ? ValueExpr((int_t)value.max_size()) : ValueExpr(nullptr)) {}
            Symbol( const ValueExpr& value, const Typename type, std::optional<size_t> rank, const ValueExpr& size )
                : variant_type(value), type(type), rank(rank), size(size) {}
            Symbol( const TensorRef& tensor, const Typename type )
                : variant_type(tensor), type(type), rank(tensor.max_size_or_null()), size(tensor.size_or_null()) {}
            Symbol( const LoopIndex& index, std::optional<size_t> rank )
                : variant_type(index), type(Typename::Int), rank(rank), size(rank ? (int_t)*rank : ValueExpr(nullptr)) {}
            Symbol( const LoopLocal& local, const Typename type, std::optional<size_t> rank = std::nullopt )
                : variant_type(local), type(type), rank(rank), size(rank ? ValueExpr((int_t)*rank) : nullptr) {}
            
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
                return rank != std::nullopt;
            }
            
            bool is_null() const
            {
                return is<ValueExpr>() ? as<ValueExpr>() == nullptr : is<TensorRef>() ? as<TensorRef>() == nullptr : false;
            }
        };
        
    public:
        
        static Result<ValueExpr> eval( const Expr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx = std::nullopt )
        {
            TRY_DECL(rank, eval_dynamic_rank(expr, symbols))
            if ( rank == nullptr )
            {
                return eval_item(expr, symbols, idx);
            }
            else if ( rank.is_literal() )
            {
                if ( idx )
                {
                    return eval_item(expr, symbols, idx);
                }
                else
                {
                    TRY_DECL(items, eval_pack(expr, symbols, (size_t)rank.as_int()))
                    if ( items.empty() && rank.as_int() != 0 )
                    {
                        return eval_dynamic_pack(expr, symbols, rank);
                    }
                    auto type = eval_type(expr, symbols);
                    return ValueExpr::list(std::move(items), type);
                }
            }
            else
            {
                TRY_DECL(value, eval_dynamic_pack(expr, symbols, rank))
                return idx ? value.at(*idx) : value;
            }
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
                TRY_DECL(value, eval(expr, symbols))
                auto type = eval_type(expr, symbols);
                return astensor(value, type);
            }
            else if ( expr.kind == Expr::Identifier && !idx )    // shortcut to avoid duplicating an existing pack
            {
                auto& iden = as_identifier(expr);
                auto& symbol = symbols.at(iden.name);
                return symbol.is<ValueExpr>() ? astensor(symbol.as<ValueExpr>(), symbol.type) : symbol.as<TensorRef>();
            }
            
            TRY_DECL(rank, eval_max_rank(expr, symbols))
            if ( rank && !idx )
            {
                TRY_DECL(tensors, eval_pack<Tensor*>(expr, symbols, *rank))
                auto type = eval_type(expr, symbols);
                TRY_DECL(shape, eval_shape_from_expr(expr, symbols))
                auto max_shape = eval_shape_max(shape);
                TRY_DECL(size, eval_dynamic_rank(expr, symbols))
                return aspack(tensors, type, shape, max_shape, size);
            }
            else
            {
                TRY_DECL(tensor, eval_item<Tensor*>(expr, symbols, idx))
                return TensorRef(tensor);
            }
        }
        
        static Result<ValueExpr> eval_optional( const Expr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx = std::nullopt )
        {
            TRY_DECL(is_null, eval_null(expr, symbols))
            return is_null ? ValueExpr(nullptr) : eval(expr, symbols, idx);
        }
        
    private:
        
        template<typename T = ValueExpr>
        static Result<std::vector<T>> eval_pack( const Expr& expr, const Dict<Symbol>& symbols, const size_t rank )
        {
            switch ( expr.kind )
            {
                case Expr::Identifier:
                {
                    auto& iden = as_identifier(expr);
                    auto& symbol = symbols.at(iden.name);
                    if ( symbol.is<ValueExpr>() )
                    {
                        auto& expr = symbol.as<ValueExpr>();
                        if ( !expr.is_literal() && !expr.is_list() )
                        {
                            return {};  // eval as dynamic pack
                        }
                    }
                    break;
                }
                case Expr::Expand:
                {
                    auto& expand = as_expand(expr);
                    return eval_pack<T>(*expand.item, symbols, rank);
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
                            TRY_DECL(rank, eval_max_rank(*item, symbols))
                            assert(rank);
                            for ( size_t i = 0; i < rank; ++i )
                            {
                                TRY_DECL(value, eval_item<T>(*item, symbols, i));
                                values[k++] = value;
                            }
                        }
                        else
                        {
                            TRY_DECL(value, eval_item<T>(*item, symbols))
                            values[k++] = value;
                        }
                    }
                    return values;
                }
                case Expr::Contain:
                {
                    if constexpr( std::is_same_v<T,ValueExpr> )
                    {
                        auto& contain = as_contain(expr);
                        
                        TRY_DECL(length, eval_static_rank(*contain.pack, symbols))
                        if ( is_tensor_expr(*contain.item, symbols) )
                        {
                            TRY_DECL(pack, eval_pack<Tensor*>(*contain.pack, symbols, *length))
                            
                            std::vector<ValueExpr> values(rank);
                            for ( size_t i = 0; i < rank; ++i )
                            {
                                TRY_DECL(item, eval_item<Tensor*>(*contain.item, symbols, i))
                                values[i] = ValueExpr(std::find(pack.begin(), pack.end(), item) != pack.end());
                            }
                            return values;
                        }
                        else
                        {
                            TRY_DECL(pack, eval_pack(*contain.pack, symbols, *length))
                            
                            std::vector<ValueExpr> values(rank);
                            for ( size_t i = 0; i < rank; ++i )
                            {
                                TRY_DECL(item, eval_item(*contain.item, symbols, i))
                                values[i] = ValueExpr(std::find(pack.begin(), pack.end(), item) != pack.end());
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
                    
                    TRY_DECL(length, eval_static_rank(*indexer.index, symbols))
                    TRY_DECL(mask, eval_pack(*indexer.index, symbols, *length))
                    if ( !is_literal(mask) )
                    {
                        return Error(expr.position, "index mask must not depend on dynamic shapes");
                    }
                    
                    const size_t count = std::count_if(mask.begin(), mask.end(), []( const ValueExpr& x ){ return x.as_bool();});
                    std::vector<T> values(count);
                    
                    size_t k = 0;
                    for ( size_t i = 0; i < mask.size(); ++i )
                    {
                        if ( mask[i].as_bool() )
                        {
                            TRY_DECL(item, eval_item<T>(*indexer.array, symbols, i))
                            values[k++] = std::move(item);
                        }
                    }
                    return values;
                }
                case Expr::Access:
                {
                    auto& access = as_access(expr);
                    TRY_DECL(rank, eval_dynamic_rank(*access.tensor, symbols))
                    if ( rank != nullptr )
                    {
                        return {};
                    }
                    break;
                }
                case Expr::Fold:
                {
                    auto& fold = as_fold(expr);
                    if constexpr( std::is_same_v<T,ValueExpr> )
                    {
                        auto type = eval_type(expr, symbols);
                        switch ( fold.op )
                        {
                            case Lexer::Operator::Plus:
                            {
                                return type == Typename::Real ? eval_accumulate<std::plus,real_t>(fold, symbols, rank) :
                                                                eval_accumulate<std::plus,int_t>(fold, symbols, rank);
                            }
                            case Lexer::Operator::Multiply:
                            {
                                return type == Typename::Real ? eval_accumulate<std::multiplies,real_t>(fold, symbols, rank) :
                                                                eval_accumulate<std::multiplies,int_t>(fold, symbols, rank);
                            }
                            case Lexer::Operator::Min:
                            {
                                return type == Typename::Real ? eval_accumulate<minimize,real_t>(fold, symbols, rank) :
                                                                eval_accumulate<minimize,int_t>(fold, symbols, rank);
                            }
                            case Lexer::Operator::Max:
                            {
                                return type == Typename::Real ? eval_accumulate<maximize,real_t>(fold, symbols, rank) :
                                                                eval_accumulate<maximize,int_t>(fold, symbols, rank);
                            }
                            case Lexer::Operator::And:
                            {
                                return eval_accumulate<std::logical_and,bool_t>(fold, symbols, rank);
                            }
                            case Lexer::Operator::Or:
                            {
                                return eval_accumulate<std::logical_or,bool_t>(fold, symbols, rank);
                            }
                            default:
                            {
                                return Error(expr.position, "invalid fold eval");
                            }
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
                    
                    TRY_DECL(value, eval_pack<T>(*substitute.pack, symbols, rank))
                    if ( index.packed() )
                    {
                        for ( size_t i = 0; i < index.max_size(); ++i )
                        {
                            TRY_DECL(item, eval_item<T>(*substitute.value, symbols, i))
                            value[index[i].as_int()] = item;
                        }
                    }
                    else
                    {
                        TRY_DECL(item, eval_item<T>(*substitute.value, symbols))
                        value[index.as_int()] = item;
                    }
                    return value;
                }
                default:
                {
                    break;
                }
            }
            
            std::vector<T> values(rank);
            for ( size_t i = 0; i < rank; ++i )
            {
                TRY_DECL(item, eval_item<T>(expr, symbols, i));
                values[i] = item;
            }
            
            if constexpr( std::is_same_v<T,ValueExpr> )
            {
                bool non_literals = std::all_of(values.begin(), values.end(), []( const ValueExpr& x ){ return !x.is_literal(); });
                if ( non_literals && values.size() > 1 && (expr.kind == Expr::Unary || expr.kind == Expr::Binary || expr.kind == Expr::Select) )
                {
                    return {};  // turn into dynamic expr
                }
            }
            
            return values;
        }
        
        static Result<ValueExpr> eval_dynamic_pack( const Expr& expr, const Dict<Symbol>& symbols )
        {
            TRY_DECL(rank, eval_dynamic_rank(expr, symbols))
            return eval_dynamic_pack(expr, symbols, rank);
        }
        
        static Result<ValueExpr> eval_dynamic_pack( const Expr& expr, const Dict<Symbol>& symbols, const ValueExpr& rank )
        {
            auto size = eval_shape_expr_max(rank);
            switch ( expr.kind )
            {
                case Expr::Identifier:
                {
                    auto& iden = as_identifier(expr);
                    auto& value = symbols.at(iden.name).as<ValueExpr>();
                    if ( value.is_shape_access() || value.is_uniform() )
                    {
                        return value;
                    }
                    return ValueExpr(ValueExpr::ReferenceExpr{ iden.name, &value }, value.dtype(), value.max_size());
                }
                case Expr::Expand:
                {
                    auto& expand = as_expand(expr);
                    if ( expand.count )
                    {
                        TRY_DECL(item, eval(*expand.item, symbols))
                        TRY_DECL(count, eval(*expand.count, symbols))
                        return ValueExpr::uniform(item, count, size);
                    }
                    else
                    {
                        return eval_dynamic_pack(*expand.item, symbols, rank);
                    }
                }
                case Expr::Cast:
                {
                    auto& cast = as_cast(expr);
                    TRY_DECL(arg, eval(*cast.arg, symbols))
                    const Typename type = is_abstract(cast.base) ? symbols.at(cast.type).type : cast.base;
                    return ValueExpr(ValueExpr::CastExpr{ type, std::move(arg) }, type, size);
                }
                case Expr::Unary:
                {
                    auto& unary = as_unary(expr);
                    TRY_DECL(arg, eval(*unary.arg, symbols))
                    auto type = arg.dtype();
                    return ValueExpr::unary(Lexer::str(unary.op), std::move(arg), type, size);
                }
                case Expr::Binary:
                {
                    auto& binary = as_binary(expr);
                    TRY_DECL(left, eval(*binary.left, symbols))
                    TRY_DECL(right, eval(*binary.right, symbols))
                    auto type = left.dtype();
                    return ValueExpr::binary(Lexer::str(binary.op), std::move(left), std::move(right), type, size);
                }
                case Expr::Select:
                {
                    auto& select = as_select(expr);
                    TRY_DECL(cond, eval(*select.cond, symbols))
                    TRY_DECL(left, eval(*select.left, symbols))
                    TRY_DECL(right, eval(*select.right, symbols))
                    return ValueExpr::select(std::move(cond), std::move(left), std::move(right), size);
                }
                case Expr::Fold:
                {
                    auto& fold = as_fold(expr);
                    if ( !allows_dynamic_fold(fold.op) )
                    {
                        return Error(expr.position, "fold expression with operator '%s' must not be of dynamic length");
                    }
                    TRY_DECL(pack, eval(*fold.pack, symbols))
                    return ValueExpr::fold(Lexer::str(fold.op), pack, size);
                }
                case Expr::List:
                {
                    auto& list = as_list(expr);
                    auto type = eval_type(expr, symbols);
                    if ( list.items.empty() )
                    {
                        return ValueExpr::list({}, type);
                    }
                    else if ( list.items.size() == 1 )
                    {
                        return eval(*list.items.front(), symbols);
                    }
                    else
                    {
                        std::vector<ValueExpr> items(list.items.size());
                        for ( size_t i = 0; i < items.size(); ++i )
                        {
                            TRY_DECL(item, eval(*list.items[i], symbols))
                            items[i] = std::move(item);
                        }
                        return ValueExpr(ValueExpr::ConcatExpr{ std::move(items) }, type, size);
                    }
                }
                case Expr::Index:
                {
                    auto& indexer = as_index(expr);
                    
                    TRY_DECL(pack, eval(*indexer.array, symbols))
                    auto type = pack.dtype();
                    
                    if ( indexer.index->kind == Expr::Range )
                    {
                        auto& range = as_range(*indexer.index);
                        TRY_DECL(stride, eval(*range.stride, symbols))
                        if ( !stride.is_literal() && (!range.first || !range.last) )
                        {
                            return Error(expr.position, "range begin and end must be explicitly supplied if stride depends on dynamic shapes");
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
                            TRY_DECL(length, eval_dynamic_rank(*indexer.array, symbols))
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
                            TRY_MOVE(last, eval_dynamic_rank(*indexer.array, symbols))
                        }
                        else if ( stride.as_int() < 0 )
                        {
                            last = (int_t)-1;
                        }
                        else
                        {
                            return Error(expr.position, "zero stride in range");
                        }
                        
                        return ValueExpr(ValueExpr::SliceExpr{ std::move(pack), std::move(first), std::move(last), std::move(stride) }, type, size);
                    }
                    else
                    {
                        TRY_DECL(index, eval(*indexer.index, symbols))
                        return ValueExpr(ValueExpr::SubscriptExpr{ std::move(pack), std::move(index) }, type, size);
                    }
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
                    
                    std::vector<ValueExpr> indices;
                    for ( auto& index : access.indices )
                    {
                        if ( index->kind == Expr::Expand )
                        {
                            const Expr& item = *as_expand(*index).item;
                            TRY_DECL(repeats, shape_item_rank(*index, symbols))
                            for ( size_t i = 0; i < repeats; ++i )
                            {
                                TRY_DECL(ix, eval_item(item, symbols, i))
                                indices.push_back(ix);
                            }
                        }
                        else
                        {
                            TRY_DECL(ix, eval_item(*index, symbols))
                            indices.push_back(ix);
                        }
                    }
                    
                    return ValueExpr(TensorAccess{ tensor, std::move(indices) }, tensor.dtype(), tensor.max_size_or_null());
                }
                case Expr::Range:
                {
                    auto& range = as_range(expr);
                    TRY_DECL(first, eval(*range.first, symbols))
                    TRY_DECL(last, eval(*range.last, symbols))
                    TRY_DECL(stride, range.stride ? eval(*range.stride, symbols) : ValueExpr(1))
                    if ( stride == (int_t)0 || stride == (real_t)0 )
                    {
                        return Error(expr.position, "zero stride in range");
                    }
                    return ValueExpr(ValueExpr::RangeExpr{ std::move(first), std::move(last), std::move(stride) }, Typename::Int, size);
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
                    return ValueExpr(nullptr);
                }
            }
        }
        
        template<typename T = ValueExpr, typename = std::enable_if_t<std::is_same_v<T,ValueExpr>>>
        static Result<ValueExpr> eval_item( const Expr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx = std::nullopt )
        {
            switch ( expr.kind )
            {
                case Expr::Literal:
                {
                    return eval_item(as_literal(expr));
                }
                case Expr::Identifier:
                {
                    return eval_item(as_identifier(expr), symbols, idx);
                }
                case Expr::List:
                {
                    return eval_item(as_list(expr), symbols, idx);
                }
                case Expr::Expand:
                {
                    return eval_item(as_expand(expr), symbols, idx);
                }
                case Expr::Index:
                {
                    return eval_item(as_index(expr), symbols, idx);
                }
                case Expr::Access:
                {
                    return eval_item(as_access(expr), symbols, idx);
                }
                case Expr::Range:
                {
                    return eval_item(as_range(expr), symbols, idx);
                }
                case Expr::Zip:
                {
                    return eval_item(as_zip(expr), symbols, idx);
                }
                case Expr::Unary:
                {
                    return eval_item(as_unary(expr), symbols, idx);
                }
                case Expr::Binary:
                {
                    return eval_item(as_binary(expr), symbols, idx);
                }
                case Expr::Select:
                {
                    return eval_item(as_select(expr), symbols, idx);
                }
                case Expr::Coalesce:
                {
                    return eval_item(as_coalesce(expr), symbols, idx);
                }
                case Expr::Identity:
                {
                    return eval_item(as_identity(expr), symbols, idx);
                }
                case Expr::Contain:
                {
                    return eval_item(as_contain(expr), symbols, idx);
                }
                case Expr::Fold:
                {
                    return eval_item(as_fold(expr), symbols, idx);
                }
                case Expr::Cast:
                {
                    return eval_item(as_cast(expr), symbols, idx);
                }
                case Expr::Builtin:
                {
                    return eval_item(as_builtin(expr), symbols, idx);
                }
                case Expr::Format:
                {
                    return eval_item(as_format(expr), symbols, idx);
                }
                case Expr::Bounded:
                {
                    return eval_item(as_bounded(expr), symbols, idx);
                }
                case Expr::Substitute:
                {
                    return eval_item(as_substitute(expr), symbols, idx);
                }
            }
            return ValueExpr::null();
        }
        
        template<typename T, typename = std::enable_if_t<std::is_same_v<T,Tensor*>>>
        static Result<Tensor*> eval_item( const Expr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx = std::nullopt )
        {
            switch ( expr.kind )
            {
                case Expr::Identifier:
                {
                    return eval_item<Tensor*>(as_identifier(expr), symbols, idx);
                }
                case Expr::List:
                {
                    return eval_item<Tensor*>(as_list(expr), symbols, idx);
                }
                case Expr::Expand:
                {
                    return eval_item<Tensor*>(as_expand(expr), symbols, idx);
                }
                case Expr::Index:
                {
                    return eval_item<Tensor*>(as_index(expr), symbols, idx);
                }
                case Expr::Zip:
                {
                    return eval_item<Tensor*>(as_zip(expr), symbols, idx);
                }
                case Expr::Select:
                {
                    return eval_item<Tensor*>(as_select(expr), symbols, idx);
                }
                case Expr::Coalesce:
                {
                    return eval_item<Tensor*>(as_coalesce(expr), symbols, idx);
                }
                case Expr::Substitute:
                {
                    return eval_item<Tensor*>(as_substitute(expr), symbols, idx);
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
        
        static Result<ValueExpr> eval_item( const LiteralExpr& literal, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            switch ( literal.type.name )
            {
                case Typename::Int:
                {
                    return ValueExpr(static_cast<const IntExpr&>(literal).value);
                }
                case Typename::Real:
                {
                    return ValueExpr(static_cast<const RealExpr&>(literal).value);
                }
                case Typename::Bool:
                {
                    return ValueExpr(static_cast<const BoolExpr&>(literal).value);
                }
                case Typename::Str:
                {
                    return ValueExpr(static_cast<const StrExpr&>(literal).value);
                }
                case Typename::Type:
                case Typename::Arith:
                case Typename::Num:
                {
                    break;
                }
            }
            return Error(literal.position, "invalid eval");
        }
        
        template<typename T = ValueExpr, typename = std::enable_if_t<std::is_same_v<T,ValueExpr>>>
        static Result<ValueExpr> eval_item( const IdenfitierExpr& iden, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            auto& symbol = symbols.at(iden.name);
            if ( symbol.is<LoopIndex>() || symbol.is<LoopLocal>() )
            {
                return ValueExpr::identifier(symbol.packed() ? iden.name + "_" + std::to_string(*idx + 1) : iden.name, 
                                             symbol.is<LoopIndex>() ? ValueExpr::IdentifierKind::LoopIndex : ValueExpr::IdentifierKind::LoopLocal,
                                             symbol.type);
            }
            auto& value = symbol.as<ValueExpr>();
            if ( value.packed() )
            {
                if ( idx )
                {
                    if ( endswith(iden.name, ".shape") && !value[*idx].is_literal() )
                    {
                        auto& tensor = symbols.at(iden.name.substr(0, iden.name.length() - 6)).as<TensorRef>();
                        return ValueExpr(ShapeAccess{ tensor, (int_t)*idx });
                    }
                    return value.at(*idx);
                }
                else
                {
                    auto reference = ValueExpr(ValueExpr::ReferenceExpr{ iden.name, &value }, symbol.type);
                    return ValueExpr(ValueExpr::SubscriptExpr{ std::move(reference), ValueExpr((int_t)*idx) }, symbol.type);
                }
            }
            else
            {
                return value;
            }
        }
        
        template<typename T, typename = std::enable_if_t<std::is_same_v<T,Tensor*>>>
        static Result<Tensor*> eval_item( const IdenfitierExpr& iden, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            auto& symbol = symbols.at(iden.name);
            auto& values = symbol.as<TensorRef>();
            return values.packed() ? (Tensor*)&values[*idx] : (Tensor*)&*values;
        }
        
        template<typename T, typename = std::enable_if_t<std::is_same_v<T,TensorRef>>>
        static Result<TensorRef> eval( const IdenfitierExpr& iden, const Dict<Symbol>& symbols, const std::optional<size_t> idx = std::nullopt )
        {
            auto& symbol = symbols.at(iden.name);
            auto& values = symbol.as<TensorRef>();
            return idx && values.packed() ? TensorRef((Tensor*)&values[*idx]) : values;
        }
        
        template<typename T = ValueExpr>
        static Result<T> eval_item( const ListExpr& list, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            size_t k = 0;
            for ( auto& item : list.items )
            {
                if ( item->kind == Expr::Expand || item->kind == Expr::Range )
                {
                    TRY_DECL(rank, eval_max_rank(*item, symbols))
                    assert(rank);
                    if ( idx < k + *rank )
                    {
                        return eval_item<T>(*item, symbols, *idx - k);
                    }
                    k += *rank;
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
        static Result<T> eval_item( const ExpandExpr& expand, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            if ( expand.count )
            {
                TRY_DECL(rank, eval_max_rank(*expand.item, symbols))
                return rank ? eval_item<T>(*expand.item, symbols, idx) : eval_item<T>(*expand.item, symbols);
            }
            else
            {
                return eval_item<T>(*expand.item, symbols, idx);
            }
        }
        
        template<typename T = ValueExpr>
        static Result<T> eval_item( const IndexExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            auto index_type = eval_type(*expr.index, symbols);
            if ( index_type == Typename::Bool )
            {
                TRY_DECL(mask, eval(*expr.index, symbols))
                if ( !is_literal(mask) )
                {
                    return Error(expr.position, "index mask must not depend on dynamic shapes");
                }
                
                size_t k = 0;
                for ( size_t i = 0; i < mask.max_size(); ++i )
                {
                    if ( mask[i].as_bool() )
                    {
                        if ( k++ == idx )
                        {
                            return eval_item<T>(*expr.array, symbols, i);
                        }
                    }
                }
                return T();
            }
            else
            {
                TRY_DECL(rank, eval_dynamic_rank(*expr.array, symbols))
                auto length = eval_shape_expr_max(rank);
                
                ValueExpr index_value;
                if ( expr.index->kind == Expr::Range )
                {
                    TRY_MOVE(index_value, eval_item(as_range(*expr.index), symbols, idx, length))
                }
                else
                {
                    TRY_MOVE(index_value, eval_item(*expr.index, symbols, idx))
                }
                
                if ( !index_value.is_literal() )
                {
                    if constexpr( std::is_same_v<T,ValueExpr> )
                    {
                        if ( expr.array->kind == Expr::Identifier )
                        {
                            auto& iden = as_identifier(*expr.array);
                            if ( endswith(iden.name, ".shape") )
                            {
                                auto& tensor = symbols.at(iden.name.substr(0, iden.name.length() - 6)).as<TensorRef>();
                                return ValueExpr(ShapeAccess{ tensor, index_value });
                            }
                        }
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
                if ( index < 0 )
                {
                    index += length;
                }
                if ( index < 0 || index >= length )
                {
                    return Error(expr.position, "index %d is out of bounds [0,%d)", (int)index, (int)length);
                }
                return eval_item<T>(*expr.array, symbols, index);
            }
        }
        
        static Result<ValueExpr> eval_item( const AccessExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            ValueExpr item;
            TensorRef tensor;
            if ( expr.tensor->kind == Expr::Index )
            {
                auto& indexer = as_index(*expr.tensor);
                TRY_DECL(subscript, eval_item(*indexer.index, symbols, idx))
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
                    for ( size_t i = 0; i < repeats; ++i )
                    {
                        TRY_DECL(ix, eval_item(item, symbols, i))
                        indices.push_back(ix);
                    }
                }
                else
                {
                    TRY_DECL(ix, eval_item(*index, symbols, idx))
                    indices.push_back(ix);
                }
            }
            
            return ValueExpr(TensorAccess{ tensor, std::move(indices), std::move(item) }, tensor.dtype());
        }
        
        static Result<ValueExpr> eval_item( const RangeExpr& range, const Dict<Symbol>& symbols, const std::optional<size_t> idx,
                                           const std::optional<size_t> length = std::nullopt )
        {
            TRY_DECL(stride, range.stride ? eval_item(*range.stride, symbols) : ValueExpr(1))
            if ( stride == 0 )
            {
                return Error(range.position, "zero stride in range");
            }
            assert(stride.is_literal());
            assert(range.first || stride.as_int() >= 0 || length);
            TRY_DECL(first, range.first ? eval_item(*range.first, symbols) : ValueExpr(stride.as_int() < 0 ? (int_t)*length - 1 : 0))
            return ValueExpr(first.as_int() + (int_t)*idx * stride.as_int());
        }
        
        template<typename T = ValueExpr>
        static Result<T> eval_item( const ZipExpr& zip, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            auto k = *idx % zip.items.size();
            auto i = *idx / zip.items.size();
            return eval_item<T>(*zip.items[k], symbols, i);
        }
        
        static Result<ValueExpr> eval_item( const UnaryExpr& unary, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            if ( unary.op == Lexer::Operator::Question )
            {
                auto& symbol = symbols.at(as_identifier(*unary.arg).name);
                return ValueExpr(!symbol.is_null());
            }
            
            TRY_DECL(arg, eval_item(*unary.arg, symbols, idx))
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
        
        static Result<ValueExpr> eval_item( const BinaryExpr& binary, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            TRY_DECL(left, eval_item(*binary.left, symbols, idx))
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
            
            TRY_DECL(right, eval_item(*binary.right, symbols, idx))
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
            
            auto simplified = simplify_binary(binary.op, left, right);
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
        static Result<T> eval_item( const SelectExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            TRY_DECL(cond, eval_item(*expr.cond, symbols, idx))
            assert(cond != nullptr);
            
            if ( cond.is_literal() )
            {
                return cond.as_bool() ? eval_item<T>(*expr.left, symbols, idx) : expr.right ? eval_item<T>(*expr.right, symbols, idx) : T();
            }
            else if constexpr ( std::is_same_v<T,ValueExpr> )
            {
                TRY_DECL(left, eval_item(*expr.left, symbols, idx))
                TRY_DECL(right, eval_item(*expr.right, symbols, idx))
                return ValueExpr::select(std::move(cond), std::move(left), std::move(right));
            }
            else
            {
                return Error(expr.position, "condition in tensor expression must not depend on dynamic tensor shapes");
            }
        }
        
        template<typename T = ValueExpr>
        static Result<T> eval_item( const CoalesceExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            TRY_DECL(is_null, eval_null(*expr.condition, symbols))
            return !is_null ? eval_item<T>(*expr.condition, symbols, idx) : eval_item<T>(*expr.alternate, symbols, idx);
        }
        
        static Result<ValueExpr> eval_item( const IdentityExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            if ( is_tensor_expr(*expr.left, symbols) )
            {
                TRY_DECL(left, eval_item<Tensor*>(*expr.left, symbols, idx))
                TRY_DECL(right, eval_item<Tensor*>(*expr.right, symbols, idx))
                
                return ValueExpr(left == right);
            }
            else
            {
                TRY_DECL(left, eval_item(*expr.left, symbols, idx))
                TRY_DECL(right, eval_item(*expr.right, symbols, idx))
                
                return ValueExpr(left == right);
            }
        }
        
        static Result<ValueExpr> eval_item( const ContainExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            TRY_DECL(length, eval_static_rank(*expr.pack, symbols))
            
            if ( is_tensor_expr(*expr.item, symbols) )
            {
                TRY_DECL(lookup, eval_item<Tensor*>(*expr.item, symbols, idx))
                for ( size_t i = 0; i < *length; ++i )
                {
                    TRY_DECL(item, eval_item<Tensor*>(*expr.pack, symbols, i))
                    if ( item == lookup )
                    {
                        return ValueExpr(true);
                    }
                }
            }
            else
            {
                TRY_DECL(lookup, eval_item(*expr.item, symbols, idx))
                for ( size_t i = 0; i < *length; ++i )
                {
                    TRY_DECL(item, eval_item(*expr.pack, symbols, i))
                    if ( item == lookup )
                    {
                        return ValueExpr(true);
                    }
                }
            }
            return ValueExpr(false);
        }
        
        static Result<ValueExpr> eval_item( const FoldExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            const Typename type = eval_type(expr, symbols);
            switch ( expr.op )
            {
                case Lexer::Operator::Plus:
                {
                    return type == Typename::Real ? eval_fold<std::plus>(expr, symbols, idx, (real_t)0) :
                                                    eval_fold<std::plus>(expr, symbols, idx, (int_t)0);
                }
                case Lexer::Operator::Multiply:
                {
                    return type == Typename::Real ? eval_fold<std::multiplies>(expr, symbols, idx, (real_t)1) :
                                                    eval_fold<std::multiplies>(expr, symbols, idx, (int_t)1);
                }
                case Lexer::Operator::Min:
                {
                    return type == Typename::Real ? eval_fold<minimize,real_t>(expr, symbols, idx) :
                                                    eval_fold<minimize,int_t>(expr, symbols, idx);
                }
                case Lexer::Operator::Max:
                {
                    return type == Typename::Real ? eval_fold<maximize,real_t>(expr, symbols, idx) :
                                                    eval_fold<maximize,int_t>(expr, symbols, idx);
                }
                case Lexer::Operator::ArgMin:
                {
                    return type == Typename::Real ? eval_arg_fold<minimize,real_t>(expr, symbols) :
                                                    eval_arg_fold<minimize,int_t>(expr, symbols);
                }
                case Lexer::Operator::ArgMax:
                {
                    return type == Typename::Real ? eval_arg_fold<maximize,real_t>(expr, symbols) :
                                                    eval_arg_fold<maximize,int_t>(expr, symbols);
                }
                case Lexer::Operator::And:
                {
                    return eval_fold<std::logical_and>(expr, symbols, idx, (bool_t)true);
                }
                case Lexer::Operator::Or:
                {
                    return eval_fold<std::logical_or>(expr, symbols, idx, (bool_t)false);
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
        
        static Result<ValueExpr> eval_item( const CastExpr& cast, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
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
            
            TRY_DECL(arg, eval_item(*cast.arg, symbols, idx))
            assert(arg != nullptr);
            
            if ( arg.dtype() == type )
            {
                return arg;
            }
            
            auto inf = std::numeric_limits<real_t>::infinity();
            if ( arg == inf && type == Typename::Int )
            {
                return ValueExpr::positive_infinity<int_t>();
            }
            else if ( arg == -inf && type == Typename::Int )
            {
                return ValueExpr::negative_infinity<int_t>();
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
        
        static Result<ValueExpr> eval_item( const BuiltinExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            TRY_DECL(arg, eval_item(*expr.arg, symbols, idx))
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
        
        static Result<ValueExpr> eval_item( const FormatExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            std::string result;
            size_t offset = 0;
            for ( auto& sub : expr.subs )
            {
                result += expr.str.substr(offset, sub.first - offset);
                TRY_DECL(value, eval_item(*sub.second, symbols, idx))
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
        
        static Result<ValueExpr> eval_item( const BoundedExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            TRY_DECL(arg, eval_item(*expr.index, symbols, idx))
            ValueExpr lower, upper;
            if ( expr.lower_value )
            {
                TRY_MOVE(lower, eval_item(*expr.lower_value, symbols, idx))
            }
            if ( expr.upper_value )
            {
                TRY_MOVE(upper, eval_item(*expr.upper_value, symbols, idx))
            }
            return ValueExpr::bounded(std::move(arg), std::move(lower), std::move(upper));
        }
        
        template<typename T = ValueExpr>
        static Result<T> eval_item( const SubstituteExpr& expr, const Dict<Symbol>& symbols, const std::optional<size_t> idx )
        {
            TRY_DECL(index, eval(*expr.index, symbols))
            assert(index != nullptr);
            if ( !is_literal(index) )
            {
                return Error(expr.position, "index must not depend on dynamic shapes in substitution expression");
            }
            
            if ( index.packed() )
            {
                for ( size_t i = 0; i < index.max_size(); ++i )
                {
                    if ( index[i].as_int() == idx )
                    {
                        return eval_item<T>(*expr.value, symbols, i);
                    }
                }
            }
            else
            {
                if ( idx == index.as_int() )
                {
                    return eval_item<T>(*expr.value, symbols);
                }
            }
            return eval_item<T>(*expr.pack, symbols, idx);
        }
        
    private:
        
        template<template<typename> class F, typename T>
        static Result<ValueExpr> eval_fold( const FoldExpr& fold, const Dict<Symbol>& symbols, const std::optional<size_t> idx,
                                           const std::optional<T> init = std::nullopt )
        {
            TRY_DECL(dynamic_rank, eval_dynamic_rank(*fold.pack, symbols))
            assert(dynamic_rank != nullptr);
            if ( !dynamic_rank.is_literal() )
            {
                TRY_DECL(pack, eval(*fold.pack, symbols))
                return ValueExpr::fold(Lexer::str(fold.op), std::move(pack));
            }
            
            const F<T> func;
            auto literal_value = init;
            ValueExpr expr_value;
            TRY_DECL(rank, fold.cumulative ? std::optional<size_t>(*idx + 1) : eval_max_rank(*fold.pack, symbols))
            
            size_t terms = 0;
            for ( size_t i = 0; i < rank; ++i )
            {
                TRY_DECL(item, eval_item(*fold.pack, symbols, i))
                assert(item != nullptr);
                
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
                    expr_value = expr_value == nullptr ? item : ValueExpr::binary(Lexer::str(fold.op), std::move(expr_value), std::move(item));
                    ++terms;
                }
                
                if ( terms > 4 )
                {
                    TRY_DECL(pack, eval(*fold.pack, symbols))
                    return ValueExpr::fold(Lexer::str(fold.op), std::move(pack));
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
        static Result<ValueExpr> eval_fold( const FoldExpr& fold, const Dict<Symbol>& symbols, const std::optional<size_t> idx, const T init )
        {
            return eval_fold<F>(fold, symbols, idx, std::optional<T>(init));
        }
        
        template<template<typename> class F, typename T>
        static Result<ValueExpr> eval_arg_fold( const FoldExpr& fold, const Dict<Symbol>& symbols )
        {
            const F<T> func;
            TRY_DECL(rank, eval_max_rank(*fold.pack, symbols))
            if ( rank == 0 )
            {
                return ValueExpr::null();
            }
            TRY_DECL(init, eval_item(*fold.pack, symbols, 0))
            assert(init != nullptr);
            T value = (T)init;
            size_t idx = 0;
            for ( size_t i = 1; i < rank; ++i )
            {
                TRY_DECL(item, eval_item(*fold.pack, symbols, i))
                assert(item != nullptr);
                auto new_value = func(value, (T)item);
                if ( new_value != value )
                {
                    idx = i;
                    value = new_value;
                }
            }
            return ValueExpr((int_t)idx);
        }
        
        template<template<typename> class F, typename T>
        static Result<std::vector<ValueExpr>> eval_accumulate( const FoldExpr& fold, const Dict<Symbol>& symbols, const size_t rank )
        {
            const F<T> func;
            std::vector<ValueExpr> values(rank);
            for ( size_t i = 0; i < rank; ++i )
            {
                TRY_DECL(item, eval_item(*fold.pack, symbols, i))
                assert(item != nullptr);
                
                if ( !item.is_literal() && allows_dynamic_fold(fold.op) )
                {
                    return {};
                }
                
                if ( i == 0 )
                {
                    values[i] = item;
                }
                else
                {
                    values[i] = func((T)values[i-1], (T)item);
                }
            }
            return values;
        }
        
        template<template<typename> class C>
        static Result<ValueExpr> eval_is_sorted( const Expr& expr, const Dict<Symbol>& symbols )
        {
            TRY_DECL(rank, eval_max_rank(expr, symbols))
            TRY_DECL(items, eval_pack(expr, symbols, *rank))
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
            TRY_DECL(rank, eval_max_rank(expr, symbols))
            TRY_DECL(items, eval_pack(expr, symbols, *rank))
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
            TRY_DECL(rank, eval_dynamic_rank(expr, symbols))
            if ( rank == nullptr )
            {
                return eval(expr, symbols);
            }
            else if ( rank == 0 )
            {
                return eval_uniform_value_empty(expr, symbols);
            }
            else if ( rank.is_literal() )
            {
                TRY_DECL(pack, eval_pack(expr, symbols, (size_t)rank.as_int()))
                return is_uniform(pack) ? pack.front() : ValueExpr(nullptr);
            }
            else
            {
                TRY_DECL(pack, eval_dynamic_pack(expr, symbols, rank))
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
                    TRY_DECL(pack, eval_dynamic_pack(expr, symbols, 0))
                    return uniform_value(pack);
                }
            }
        }
        
        static ValueExpr uniform_value( const ValueExpr& expr )
        {
            switch ( expr.kind() )
            {
                case ValueExpr::Literal:
                case ValueExpr::Identifier:
                case ValueExpr::SizeAccess:
                case ValueExpr::TensorAccess:
                {
                    return expr;
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
                        return expr;
                    }
                    if ( access.tensor.max_size() == 0 || !access.dim.is_literal() )
                    {
                        return nullptr;
                    }
                    auto value = access.tensor[0].shape[access.dim.as_int()];
                    for ( size_t i = 1; i < access.tensor.max_size(); ++i )
                    {
                        if ( access.tensor[i].shape[access.dim.as_int()] != value )
                        {
                            return nullptr;
                        }
                    }
                    return value;
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
                    if ( !fold.accumulate )
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
                    return is_uniform(items) ? items.front() : ValueExpr(nullptr);
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
                    return expr.as_uniform().value;
                }
                default:
                {
                    return nullptr;
                }
            }
        }
        
    public:
        
        template<bool Optional = false>
        static Result<std::optional<size_t>> eval_static_rank( const Expr& expr, const Dict<Symbol>& symbols )
        {
            TRY_DECL(rank, eval_dynamic_rank<Optional>(expr, symbols))
            if ( rank == nullptr )
            {
                return (std::optional<size_t>)std::nullopt;
            }
            if ( !rank.is_literal() )
            {
                return Error(expr.position, "length of packed expression must not be dynamic in this context");
            }
            return (std::optional<size_t>)rank.as_int();
        }
        
        template<bool Optional = false>
        static Result<std::optional<size_t>> eval_max_rank( const Expr& expr, const Dict<Symbol>& symbols )
        {
            TRY_DECL(rank, eval_dynamic_rank<Optional>(expr, symbols))
            return rank == nullptr ? (std::optional<size_t>)std::nullopt : (std::optional<size_t>)eval_shape_expr_max(rank);
        }
        
        template<bool Optional = false>
        static Result<ValueExpr> eval_dynamic_rank( const Expr& expr, const Dict<Symbol>& symbols )
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
                            TRY_DECL(len, eval_dynamic_rank(*item, symbols))
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
                        return eval_dynamic_rank(*expand.item, symbols);
                    }
                }
                case Expr::Index:
                {
                    auto& index = as_index(expr);
                    auto index_type = eval_type(*index.index, symbols);
                    if ( index_type == Typename::Bool )
                    {
                        TRY_DECL(array_length, eval_static_rank(*index.array, symbols))
                        TRY_DECL(index_length, eval_static_rank(*index.index, symbols))
                        if ( index_length != array_length )
                        {
                            return Error(expr.position, "incompatible mask length and pack length (%d vs %d)",
                                         (int)*index_length, (int)*array_length);
                        }
                        
                        TRY_DECL(mask, eval_pack(*index.index, symbols, *index_length))
                        if ( !is_literal(mask) )
                        {
                            return Error(expr.position, "index mask must not depend on dynamic shapes");
                        }
                        return ValueExpr((int_t)std::count_if(mask.begin(), mask.end(), []( const ValueExpr& item ){ return item.as_bool(); }));
                    }
                    else if ( index.index->kind == Expr::Range )
                    {
                        auto& range = as_range(*index.index);
                        TRY_DECL(length, eval_dynamic_rank(*index.array, symbols))
                        
                        TRY_DECL(stride, range.stride ? eval_item(*range.stride, symbols) : ValueExpr(1))
                        if ( !stride.is_literal() && (!range.first || !range.last) )
                        {
                            return Error(expr.position, "range begin and end must be explicitly supplied if stride depends on dynamic shapes");
                        }
                        
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
                        
                        return stride.is_literal() && stride.as_int() < 0 ? ceil_div(first - last, -stride) : ceil_div(last - first, stride);
                    }
                    else
                    {
                        return eval_dynamic_rank(*index.index, symbols);
                    }
                }
                case Expr::Access:
                {
                    auto& access = as_access(expr);
                    TRY_DECL(tensor_rank, eval_dynamic_rank(*access.tensor, symbols))
                    if ( tensor_rank != nullptr )
                    {
                        return tensor_rank;
                    }
                    
                    for ( auto& index : access.indices )
                    {
                        if ( index->kind != Expr::Expand )
                        {
                            TRY_DECL(index_rank, eval_dynamic_rank(*index, symbols))
                            if ( index_rank != nullptr )
                            {
                                return index_rank;
                            }
                        }
                    }
                    return ValueExpr::null();
                }
                case Expr::Range:
                {
                    auto& range = as_range(expr);
                    TRY_DECL(first, eval_item(*range.first, symbols))
                    TRY_DECL(last, eval_item(*range.last, symbols))
                    TRY_DECL(stride, range.stride ? eval_item(*range.stride, symbols) : ValueExpr(1))
                    return stride.is_literal() && stride.as_int() < 0 ? ceil_div(first - last, -stride) : ceil_div(last - first, stride);
                }
                case Expr::Zip:
                {
                    auto& zip = as_zip(expr);
                    ValueExpr rank;
                    for ( size_t i = 0; i < zip.items.size(); ++i )
                    {
                        TRY_DECL(item_rank, eval_dynamic_rank(*zip.items[i], symbols))
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
                    return eval_dynamic_rank(*unary.arg, symbols);
                }
                case Expr::Binary:
                {
                    auto& binary = as_binary(expr);
                    TRY_DECL(left, eval_dynamic_rank(*binary.left, symbols))
                    TRY_DECL(right, eval_dynamic_rank(*binary.right, symbols))
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
                    TRY_DECL(cond, eval_dynamic_rank(*select.cond, symbols))
                    TRY_DECL(left, eval_dynamic_rank(*select.left, symbols))
                    TRY_DECL(right, select.right ? eval_dynamic_rank(*select.right, symbols) : ValueExpr::null())
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
                        return eval_dynamic_rank(*coalesce.condition, symbols);
                    }
                    else
                    {
                        return eval_dynamic_rank(*coalesce.alternate, symbols);
                    }
                }
                case Expr::Identity:
                {
                    auto& identity = as_identity(expr);
                    TRY_DECL(left, eval_dynamic_rank(*identity.left, symbols))
                    TRY_DECL(right, eval_dynamic_rank(*identity.right, symbols))
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
                    return eval_dynamic_rank(*contain.item, symbols);
                }
                case Expr::Fold:
                {
                    auto& fold = as_fold(expr);
                    return fold.cumulative ? eval_dynamic_rank(*fold.pack, symbols) : ValueExpr(nullptr);
                }
                case Expr::Cast:
                {
                    auto& cast = as_cast(expr);
                    return cast.arg ? eval_dynamic_rank(*cast.arg, symbols) : ValueExpr(nullptr);
                }
                case Expr::Builtin:
                {
                    auto& builtin = as_builtin(expr);
                    return eval_dynamic_rank(*builtin.arg, symbols);
                }
                case Expr::Bounded:
                {
                    auto& bounded = as_bounded(expr);
                    TRY_DECL(index_rank, eval_dynamic_rank(*bounded.index, symbols))
                    
                    if ( bounded.lower_value )
                    {
                        TRY_DECL(lower_rank, eval_dynamic_rank(*bounded.lower_value, symbols))
                        if ( index_rank != nullptr && lower_rank != nullptr && lower_rank != index_rank )
                        {
                            return Error(expr.position, "incompatible pack lengths in bounded expression for index and lower value ('%s' vs '%s')",
                                         str(index_rank).c_str(), str(lower_rank).c_str());
                        }
                    }
                    if ( bounded.upper_value )
                    {
                        TRY_DECL(upper_rank, eval_dynamic_rank(*bounded.upper_value, symbols))
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
                    TRY_DECL(pack_rank, eval_dynamic_rank(*substitute.pack, symbols))
                    TRY_DECL(index_rank, eval_dynamic_rank(*substitute.index, symbols))
                    TRY_DECL(value_rank, eval_dynamic_rank(*substitute.value, symbols))
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
                    assert(false);
                    return Typename::Type;
                }
            }
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
                    TRY_DECL(rank, eval_max_rank(*fold.pack, symbols))
                    return *rank == 0;
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
            for ( auto& item : shape.extents )
            {
                if ( item )
                {
                    TRY_DECL(rank, shape_item_rank(*item, symbols))
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
                    TRY_DECL(rank, eval_max_rank(item, symbols))
                    return *rank;
                }
            }
            else if ( item.kind == Expr::Range )
            {
                TRY_DECL(rank, eval_max_rank(item, symbols))
                return *rank;
            }
            else
            {
                return 1;
            }
        }
        
        static ValueExpr eval_dynamic_rank( const ValueExpr& expr )
        {
            if ( !expr.packed() )
            {
                return ValueExpr(nullptr);
            }
            switch ( expr.kind() )
            {
                case ValueExpr::ShapeAccess:
                {
                    auto& access = expr.as_shape_access();
                    return access.tensor.size();
                }
                case ValueExpr::Reference:
                {
                    auto& ref = expr.as_reference();
                    return eval_dynamic_rank(*ref.target);
                }
                case ValueExpr::List:
                {
                    auto& list = expr.as_list();
                    return ValueExpr((int_t)list.size());
                }
                case ValueExpr::Unary:
                {
                    auto& unary = expr.as_unary();
                    return eval_dynamic_rank(unary.arg);
                }
                case ValueExpr::Binary:
                {
                    auto& binary = expr.as_binary();
                    return eval_dynamic_rank(binary.left.packed() ? binary.left : binary.right);
                }
                case ValueExpr::Select:
                {
                    auto& select = expr.as_select();
                    return eval_dynamic_rank(select.cond.packed() ? select.cond : select.left.packed() ? select.left : select.right);
                }
                case ValueExpr::Fold:
                {
                    auto& fold = expr.as_fold();
                    return eval_dynamic_rank(fold.pack);
                }
                case ValueExpr::Concat:
                {
                    auto& concat = expr.as_concat();
                    ValueExpr const_rank;
                    ValueExpr expr_rank;
                    for ( auto& item : concat.items )
                    {
                        auto item_rank = eval_dynamic_rank(item);
                        if ( item_rank.is_literal() )
                        {
                            const_rank = const_rank == nullptr ? item_rank.as_int() : const_rank.as_int() + item_rank.as_int();
                        }
                        else
                        {
                            expr_rank = expr_rank == nullptr ? item_rank : expr_rank + item_rank;
                        }
                    }
                    return const_rank != nullptr ? expr_rank + const_rank : expr_rank;
                }
                case ValueExpr::Slice:
                {
                    auto& slice = expr.as_slice();
                    return slice.last - slice.first;
                }
                case ValueExpr::Uniform:
                {
                    return expr.as_uniform().size;
                }
                default:
                {
                    return ValueExpr(nullptr);
                }
            }
        }
        
        template<typename T>
        static std::pair<T,T> duplicate( const T& value )
        {
            return std::make_pair(value, value);
        }
        
        template<typename T = int_t>
        static std::pair<T,T> eval_shape_expr_bounds( const ValueExpr& expr, const std::optional<size_t>& idx = std::nullopt )
        {
            if ( expr.packed() && !idx )
            {
                T min = 0, max = 0;
                for ( size_t i = 0; i < expr.max_size(); ++i )
                {
                    auto [item_min, item_max] = eval_shape_expr_bounds(expr, i);
                    if ( i == 0 || item_min < min )
                    {
                        min = item_min;
                    }
                    if ( i == 0 || item_max > max )
                    {
                        max = item_max;
                    }
                }
                return std::make_pair(min, max);
            }
            
            switch ( expr.kind() )
            {
                case ValueExpr::Literal:
                {
                    return duplicate(expr.as<T>());
                }
                case ValueExpr::Placeholder:
                {
                    auto& placeholder = expr.as_placeholder();
                    auto [min, max] = eval_shape_expr_bounds<T>(placeholder.max_value, idx);
                    return std::make_pair((T)0, max);
                }
                case ValueExpr::Reference:
                {
                    auto& ref = expr.as_reference();
                    return eval_shape_expr_bounds<T>(*ref.target, idx);
                }
                case ValueExpr::SizeAccess:
                {
                    auto& access = expr.as_size_access();
                    return std::make_pair((T)0, (T)access.pack.max_size());
                }
                case ValueExpr::ShapeAccess:
                {
                    auto& access = expr.as_shape_access();
                    auto& max_shape = idx ? access.tensor[*idx].max_shape : access.tensor.max_shape();
                    auto max = access.dim.is_literal() ? max_shape[access.dim.as_int()] : *std::max_element(max_shape.begin(), max_shape.end());
                    return std::make_pair((T)0, (T)max);
                }
                case ValueExpr::Cast:
                {
                    auto& cast = expr.as_cast();
                    if ( cast.arg.dtype() == cast.dtype )
                    {
                        return eval_shape_expr_bounds<T>(cast.arg, idx);
                    }
                    if ( cast.arg.dtype() == Typename::Bool )
                    {
                        return duplicate((T)1);
                    }
                    if ( cast.dtype == Typename::Bool )
                    {
                        auto [min, max] = eval_shape_expr_bounds<T>(cast.arg, idx);
                        return duplicate(min != (T)0 || max != (T)0);
                    }
                    if ( cast.arg == std::numeric_limits<real_t>::infinity() )
                    {
                        return duplicate(std::numeric_limits<T>::max());
                    }
                    if ( cast.arg == -std::numeric_limits<real_t>::infinity() )
                    {
                        return duplicate(std::numeric_limits<T>::min());
                    }
                    if ( cast.dtype == Typename::Int )
                    {
                        assert(cast.arg.dtype() == Typename::Real);
                        auto [min, max] = eval_shape_expr_bounds<real_t>(cast.arg, idx);
                        return std::make_pair((int_t)min, (int_t)max);
                    }
                    else if ( cast.dtype == Typename::Real )
                    {
                        assert(cast.arg.dtype() == Typename::Int);
                        auto [min, max] = eval_shape_expr_bounds<int_t>(cast.arg, idx);
                        return std::make_pair((real_t)min, (real_t)max);
                    }
                    else
                    {
                        assert(false);
                        return duplicate((T)0);
                    }
                }
                case ValueExpr::Unary:
                {
                    auto& unary = expr.as_unary();
                    auto [min, max] = eval_shape_expr_bounds<T>(unary.arg, idx);
                    switch ( Lexer::operator_value(unary.op) )
                    {
                        case Lexer::Operator::Plus:
                        {
                            return std::make_pair(min, max);
                        }
                        case Lexer::Operator::Minus:
                        {
                            return std::make_pair(-max, -min);
                        }
                        default:
                        {
                            return std::make_pair((T)eval_unary_func(unary.op, min), (T)eval_unary_func(unary.op, max));
                        }
                    }
                }
                case ValueExpr::Binary:
                {
                    auto& binary = expr.as_binary();
                    auto [left_min, left_max] = eval_shape_expr_bounds<T>(binary.left, idx);
                    auto [right_min, right_max] = eval_shape_expr_bounds<T>(binary.right, idx);
                    switch ( Lexer::operator_value(binary.op) )
                    {
                        case Lexer::Operator::Plus:
                        {
                            return std::make_pair(left_min + right_min, left_max + right_max);
                        }
                        case Lexer::Operator::Minus:
                        {
                            return std::make_pair(left_min - right_max, left_max - right_min);
                        }
                        case Lexer::Operator::Multiply:
                        {
                            return std::make_pair(left_min * right_min, left_max * right_max);
                        }
                        case Lexer::Operator::Divide:
                        {
                            return std::make_pair(left_min / right_max, right_min == 0 ? left_max : left_max / right_min);
                        }
                        case Lexer::Operator::CeilDivide:
                        {
                            return std::make_pair(ceil_div(left_min, right_max), right_min == 0 ? left_max : ceil_div(left_max, right_min));
                        }
                        case Lexer::Operator::Modulo:
                        {
                            return std::make_pair((T)0, std::abs(right_max) - (T)1);
                        }
                        case Lexer::Operator::Power:
                        {
                            return std::make_pair((T)pow(left_min, right_min), (T)pow(left_max, right_max));
                        }
                        case Lexer::Operator::Min:
                        {
                            return std::make_pair(std::min(left_min, right_min), std::min(left_max, right_max));
                        }
                        case Lexer::Operator::Max:
                        {
                            return std::make_pair(std::max(left_min, right_min), std::max(left_max, right_max));
                        }
                        default:
                        {
                            assert(false);
                            return duplicate((T)0);
                        }
                    }
                }
                case ValueExpr::Select:
                {
                    auto& select = expr.as_select();
                    auto [left_min, left_max] = eval_shape_expr_bounds<T>(select.left, idx);
                    auto [right_min, right_max] = eval_shape_expr_bounds<T>(select.right, idx);
                    return std::make_pair(std::min(left_min, right_min), std::max(left_max, right_max));
                }
                case ValueExpr::List:
                {
                    assert(idx);
                    auto& list = expr.as_list();
                    return eval_shape_expr_bounds<T>(list[*idx]);
                }
                case ValueExpr::Fold:
                {
                    auto& fold = expr.as_fold();
                    auto& pack = fold.pack.is_reference() ? *fold.pack.as_reference().target : fold.pack;
                    bool dynamic_size = pack.has_dynamic_size();
                    switch ( Lexer::operator_value(fold.op) )
                    {
                        case Lexer::Operator::Plus:
                        {
                            T sum_min = 0;
                            T sum_max = 0;
                            T max = 0;
                            for ( size_t i = 0; i < pack.max_size(); ++i )
                            {
                                auto [item_min, item_max] = eval_shape_expr_bounds<T>(pack, i);
                                sum_min += item_min;
                                sum_max += item_max;
                                if ( sum_max > max )
                                {
                                    max = sum_max;
                                }
                            }
                            return dynamic_size ? std::make_pair((T)0, max) : std::make_pair(sum_min, sum_max);
                        }
                        case Lexer::Operator::Multiply:
                        {
                            T prod_min = 1;
                            T prod_max = 1;
                            T max = 1;
                            for ( size_t i = 0; i < pack.max_size(); ++i )
                            {
                                auto [item_min, item_max] = eval_shape_expr_bounds<T>(pack, i);
                                prod_min *= item_min;
                                prod_max *= item_max;
                                if ( prod_max > max )
                                {
                                    max = prod_max;
                                }
                            }
                            return dynamic_size ? std::make_pair((T)1, max) : std::make_pair(prod_min, prod_max);
                        }
                        case Lexer::Operator::Min:
                        {
                            auto [min, max] = eval_shape_expr_bounds<T>(pack.at(0));
                            for ( size_t i = 1; i < pack.max_size(); ++i )
                            {
                                auto [item_min, item_max] = eval_shape_expr_bounds<T>(pack, i);
                                min = std::min(min, item_min);
                                if ( !dynamic_size )
                                {
                                    max = std::min(max, item_max);
                                }
                            }
                            return std::make_pair(min, max);
                        }
                        case Lexer::Operator::Max:
                        {
                            auto [min, max] = eval_shape_expr_bounds<T>(pack.at(0));
                            for ( size_t i = 1; i < pack.max_size(); ++i )
                            {
                                auto [item_min, item_max] = eval_shape_expr_bounds<T>(pack, i);
                                max = std::max(max, item_max);
                                if ( !dynamic_size )
                                {
                                    min = std::max(min, item_min);
                                }
                            }
                            return std::make_pair(min, max);
                        }
                        case Lexer::Operator::MakeEqual:
                        {
                            return eval_shape_expr_bounds<T>(pack.at(0));
                        }
                        default:
                        {
                            assert(false);
                            return duplicate((T)0);
                        }
                    }
                }
                case ValueExpr::Subscript:
                {
                    auto& subscript = expr.as_subscript();
                    auto& pack = subscript.pack.is_reference() ? *subscript.pack.as_reference().target : subscript.pack;
                    auto [idx_min, idx_max] = eval_shape_expr_bounds<int_t>(subscript.index, idx);
                    auto [min, max] = eval_shape_expr_bounds<T>(pack.at(idx_min));
                    for ( size_t i = idx_min + 1; i <= idx_max; ++i )
                    {
                        auto [item_min, item_max] = eval_shape_expr_bounds<T>(pack, i);
                        max = std::max(max, item_max);
                        min = std::min(min, item_min);
                    }
                    return std::make_pair(min, max);
                }
                case ValueExpr::Uniform:
                {
                    auto& uniform = expr.as_uniform();
                    return eval_shape_expr_bounds<T>(uniform.value);
                }
                default:
                {
                    assert(false);
                    return duplicate((T)0);
                }
            }
        }
        
        static int_t eval_shape_expr_max( const ValueExpr& expr, const std::optional<size_t>& idx = std::nullopt )
        {
            auto [min, max] = eval_shape_expr_bounds<int_t>(expr, idx);
            return max;
        }
        
        static std::vector<int_t> eval_shape_max( const Shape& shape )
        {
            std::vector<int_t> max_shape(shape.size());
            for ( size_t i = 0; i < shape.size(); ++i )
            {
                auto& extent = shape[i];
                if ( extent.packed() )
                {
                    if ( extent.max_size() == 0 )
                    {
                        max_shape[i] = -1;
                        continue;
                    }
                    max_shape[i] = eval_shape_expr_max(extent, 0);
                    for ( size_t k = 1; k < extent.max_size(); ++k )
                    {
                        auto item_max = eval_shape_expr_max(extent, k);
                        if ( max_shape[i] != item_max )
                        {
                            max_shape[i] = -1;
                            break;
                        }
                    }
                }
                else
                {
                    max_shape[i] = eval_shape_expr_max(extent);
                }
            }
            return max_shape;
        }
        
        static bool is_tensor_expr( const Expr& expr, const Dict<Symbol>& symbols )
        {
            switch ( expr.kind )
            {
                case Expr::Identifier:
                {
                    auto it = symbols.find(as_identifier(expr).name);
                    return it != symbols.end() && (it->second.is<TensorRef>() || it->second.is<LoopLocal>());
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
        
        static bool is_uniform( const std::vector<ValueExpr>& items )
        {
            if ( items.empty() )
            {
                return false;
            }
            auto& first = items.front();
            return std::all_of(items.begin() + 1, items.end(), [&first]( const ValueExpr& item ){ return item == first; });
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
                    auto& symbol = symbols.at(iden.name + ".shape");
                    auto& items = symbol.as<ValueExpr>().as_list();
                    return Shape(items.begin(), items.end());
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
            for ( size_t i = 0; i < items.size(); ++i )
            {
                TRY_DECL(item_shape, eval_shape_from_expr(*items[i], symbols))
                if ( i == 0 )
                {
                    shape = item_shape;
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
                        else if ( shape[k] != item_shape[k] )
                        {
                            shape[k] = ValueExpr::list(shape[k], items.size());
                            shape[k].as_list()[i] = item_shape[k];
                        }
                    }
                }
            }
            return shape;
        }
        
        static ValueExpr common_shape_expr( const std::vector<ValueExpr>& exprs )
        {
            auto canonical = canonical_shape_expr(exprs[0]);
            for ( size_t i = 1; i < exprs.size(); ++i )
            {
                auto item = canonical_shape_expr(exprs[i]);
                if ( item != canonical )
                {
                    return nullptr;
                }
            }
            return canonical;
        }
        
        static const ValueExpr& follow_shape_accesses( const ValueExpr& expr )
        {
            const ValueExpr* ptr = &expr;
            while ( ptr->is_size_access() || ptr->is_shape_access() )
            {
                if ( ptr->is_size_access() )
                {
                    auto& access = ptr->as_size_access();
                    ptr = &access.pack.size();
                }
                else if ( ptr->is_shape_access() )
                {
                    auto& access = ptr->as_shape_access();
                    if ( !access.dim.is_literal() )
                    {
                        break;
                    }
                    ptr = &access.tensor.shape()[access.dim.as_int()];
                }
            }
            return *ptr;
        }
        
        static ValueExpr canonical_shape_expr( const ValueExpr& expr )
        {
            ValueExpr canonical = expr;
            preorder_traverse(canonical, []( ValueExpr& x )
            {
                x = follow_shape_accesses(x);
            });
            simplify(canonical);
            return canonical;
        }
        
        static std::vector<ValueExpr> canonical_shape( const std::vector<ValueExpr>& shape )
        {
            std::vector<ValueExpr> canonical(shape.size());
            for ( size_t i = 0; i < shape.size(); ++i )
            {
                canonical[i] = canonical_shape_expr(shape[i]);
            }
            return canonical;
        }
        
        static bool canonical_shape_expr_equals( const ValueExpr& x, const ValueExpr& y )
        {
            return x == y || canonical_shape_expr(x) == canonical_shape_expr(y);
        }
        
        static bool has_undefined_symbols( const Shapedef& shape, const Dict<Symbol>& symbols )
        {
            for ( auto& item : shape.extents )
            {
                if ( item && has_undefined_symbols(*item, symbols) )
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
        
        static bool has_loop_index_symbols( const Expr& expr, const Dict<Symbol>& symbols )
        {
            return any_of(expr, [&]( const Expr& e )
            {
                return e.kind == Expr::Identifier && symbols.at(as_identifier(e).name).is<LoopIndex>();
            });
        }
    };

}   // namespace sknd


#endif
