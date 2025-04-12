#ifndef _TS_EXPR_H_
#define _TS_EXPR_H_

#include "position.h"
#include "function.h"
#include "types.h"
#include "lexer.h"
#include <memory>
#include <sstream>
#include <string>
#include <vector>


namespace sknd
{
    
    template<typename T>
    using Shared = std::shared_ptr<const T>;
    
    
    struct Expr
    {
        enum Kind : char { Literal, Identifier, List, Index, Access, Range, Zip, Expand, Unary, Binary, Select, Coalesce,
            Identity, Contain, Fold, Cast, Builtin, Format, Bounded, Substitute };

		Expr( const Position& position, const Kind kind )
        : position(position), kind(kind)
        {
        }

        virtual ~Expr() {}

        virtual void print( std::ostream& os ) const = 0;

        const Position position;
        const Kind kind;
    };
    
    
    inline std::ostream& operator<<( std::ostream& os, const Expr& expr )
    {
        expr.print(os);
        return os;
    }
    
    inline std::string str( const Expr& expr )
    {
        std::stringstream ss;
        ss << expr;
        return ss.str();
    }
    
    
    struct LiteralExpr : public Expr
    {
        LiteralExpr( const Position& position, const Typename type, const bool optional = false )
        : Expr(position, Literal), type{ type, optional, false, false }
        {
        }
        
        const Type type;
    };


    struct RealExpr : public LiteralExpr
    {
        RealExpr( const Position& position, const real_t value )
        : LiteralExpr(position, Typename::Real), value(value)
        {
        }
        
        void print( std::ostream& os ) const
        {
            os << value;
            if ( value == (int_t)value )
            {
                os << ".0";
            }
        }

        const real_t value;
    };


    struct IntExpr : public LiteralExpr
    {
        IntExpr( const Position& position, const int_t value )
        : LiteralExpr(position, Typename::Int), value(value)
        {
        }
        
        void print( std::ostream& os ) const
        {
            os << value;
        }

        const int_t value;
    };


    struct BoolExpr : public LiteralExpr
    {
        BoolExpr( const Position& position, const bool_t value )
        : LiteralExpr(position, Typename::Bool), value(value)
        {
        }
        
        void print( std::ostream& os ) const
        {
            os << std::boolalpha << value;
        }

        const bool_t value;
    };
    
    
    struct StrExpr : public LiteralExpr
    {
        StrExpr( const Position& position, const str_t value )
        : LiteralExpr(position, Typename::Str), value(value)
        {
        }
        
        void print( std::ostream& os ) const
        {
            os << '\'' << value << '\'';
        }

        const str_t value;
    };


    struct IdenfitierExpr : public Expr
    {
        IdenfitierExpr( const Position& position, const std::string& name )
        : Expr(position, Identifier), name(name)
        {
        }
        
        void print( std::ostream& os ) const
        {
            os << name;
        }

        const std::string name;
    };
    
    
    struct ListExpr : public Expr
    {
        ListExpr( const Position& position )
        : Expr(position, List)
        {
        }
        
		ListExpr( const Position& position, std::vector<Shared<Expr>>&& items )
        : Expr(position, List), items(std::move(items))
        {
        }
        
        void print( std::ostream& os ) const
        {
            os << '[';
            for ( size_t i = 0; i < items.size(); ++i )
            {
                if ( i )
                {
                    os << ',';
                }
                os << *items[i];
            }
            os << ']';
        }
        
        const std::vector<Shared<Expr>> items;
    };
    
    
    struct IndexExpr : public Expr
    {
        IndexExpr( const Position& position, const Shared<Expr> array, const Shared<Expr> index )
        : Expr(position, Index), array(array), index(index)
        {
        }
        
        void print( std::ostream& os ) const
        {
            os << *array;
            if ( index->kind == Expr::List )
            {
                os << *index;
            }
            else
            {
                os << '[' << *index << ']';
            }
        }
        
        const Shared<Expr> array;
        const Shared<Expr> index;
    };


    struct AccessExpr : public Expr
    {
        AccessExpr( const Position& position, const Shared<Expr> tensor, std::vector<Shared<Expr>>&& indices )
        : Expr(position, Access), tensor(tensor), indices(std::move(indices))
        {
        }
        
        void print( std::ostream& os ) const
        {
            os << *tensor;
            os << '[';
            for ( size_t i = 0; i < indices.size(); ++i )
            {
                if ( i )
                {
                    os << ',';
                }
                os << *indices[i];
            }
            os << ']';
        }
        
        const Shared<Expr> tensor;
        const std::vector<Shared<Expr>> indices;
    };
    
    
    struct RangeExpr : public Expr
    {
    public:
        
        RangeExpr( const Position& position, const Shared<Expr> first, const Shared<Expr> last, const Shared<Expr> stride )
        : Expr(position, Range), first(first), last(last), stride(stride)
        {
        }
        
        void print( std::ostream& os ) const
        {
            if ( first )
            {
                os << *first;
            }
            os << ':';
            if ( last )
            {
                os << *last;
            }
            if ( stride )
            {
                os << ':';
                os << *stride;
            }
        }
        
        const Shared<Expr> first;
        const Shared<Expr> last;
        const Shared<Expr> stride;
    };
    
    
    struct ZipExpr : public Expr
    {
        ZipExpr( const Position& position, std::vector<Shared<Expr>>&& items )
        : Expr(position, Expr::Zip), items(std::move(items))
        {
        }
        
        void print( std::ostream& os ) const
        {
            os << '(';
            for ( size_t i = 0; i < items.size(); ++i )
            {
                if ( i )
                {
                    os << ',';
                }
                os << *items[i];
            }
            os << ')';
        }
        
        const std::vector<Shared<Expr>> items;
    };
    
    
    struct ExpandExpr : public Expr
    {
        ExpandExpr( const Position& position, const Shared<Expr> item, const Shared<Expr> count )
		: Expr(position, Expand), item(item), count(count)
        {
        }
        
        void print( std::ostream& os ) const
        {
            const bool parenthesize = item->kind == Expr::Unary || item->kind == Expr::Binary || item->kind == Expr::Select || item->kind == Expr::Fold;
            if ( parenthesize )
            {
                os << '(';
            }
            os << *item;
            if ( parenthesize )
            {
                os << ')';
            }
            if ( item->kind == Expr::Literal )
            {
                os << ' ';
            }
            os << "..";
            if ( count )
            {
                os << '(' << *count << ')';
            }
        }
        
        const Shared<Expr> item;
        const Shared<Expr> count;
    };


    struct UnaryExpr : public Expr
    {
		UnaryExpr( const Position& position, const Shared<Expr> arg, const Lexer::Operator op )
        : Expr(position, Unary), arg(arg), op(op)
        {
        }

        void print( std::ostream& os ) const
        {
            os << Lexer::str(op) << *arg;
        }

        const Shared<Expr> arg;
        const Lexer::Operator op;
    };
    
    
    struct BinaryExpr : public Expr
    {
        BinaryExpr( const Position& position, const Shared<Expr> left, const Shared<Expr> right,
                   const Lexer::Operator op )
        : Expr(position, Binary), left(left), right(right), op(op)
        {
        }

        void print( std::ostream& os ) const
        {
            const bool paren_left = (left->kind == Binary && static_cast<const BinaryExpr&>(*left).op != op) ||
                                     left->kind == Fold || left->kind == Select;
            if ( paren_left )
            {
                os << '(';
            }
            os << *left;
            if ( paren_left )
            {
                os << ')';
            }
            
            os << ' ' << Lexer::str(op) << ' ';
            
            const bool paren_right = (right->kind == Binary && static_cast<const BinaryExpr&>(*right).op != op) ||
                                      right->kind == Fold || right->kind == Select;
            if ( paren_right )
            {
                os << '(';
            }
            os << *right;
            if ( paren_right )
            {
                os << ')';
            }
        }

        const Shared<Expr> left;
        const Shared<Expr> right;
        const Lexer::Operator op;
    };


    struct SelectExpr : public Expr
    {
        SelectExpr( const Position& position, const Shared<Expr> cond, const Shared<Expr> left, const Shared<Expr> right )
        : Expr(position, Select), cond(cond), left(left), right(right)
        {
        }

        void print( std::ostream& os ) const
        {
            os << *cond << " ? " << *left;
            if ( right )
            {
                os << " : " << *right;
            }
        }

        const Shared<Expr> cond;
        const Shared<Expr> left;
        const Shared<Expr> right;
    };
    
    
    struct CoalesceExpr : public Expr
    {
        CoalesceExpr( const Position& position, const Shared<Expr> condition, const Shared<Expr> alternate )
        : Expr(position, Coalesce), condition(condition), alternate(alternate)
        {
        }

        void print( std::ostream& os ) const
        {
            os << *condition << " ?? " << *alternate;
        }

        const Shared<Expr> condition;
        const Shared<Expr> alternate;
    };


    struct IdentityExpr : public Expr
    {
        IdentityExpr( const Position& position, const Shared<Expr> left, const Shared<Expr> right )
        : Expr(position, Identity), left(left), right(right)
        {
        }
        
        void print( std::ostream& os ) const
        {
            os << *left << " is " << *right;
        }
        
        const Shared<Expr> left;
        const Shared<Expr> right;
    };
    
    
    struct ContainExpr : public Expr
    {
        ContainExpr( const Position& position, const Shared<Expr> item, const Shared<Expr> pack )
        : Expr(position, Contain), item(item), pack(pack)
        {
        }
        
        void print( std::ostream& os ) const
        {
            os << *item << " in " << *pack;
        }
        
        const Shared<Expr> item;
        const Shared<Expr> pack;
    };


    struct FoldExpr : public Expr
    {
        FoldExpr( const Position& position, const Shared<Expr> pack, const Lexer::Operator op, const bool cumulative = false )
        : Expr(position, Fold), pack(pack), op(op), cumulative(cumulative)
        {
        }

        void print( std::ostream& os ) const
        {
            const bool paren = pack->kind == Binary || pack->kind == Select;
            
            if ( paren )
            {
                os << '(';
            }
            os << *pack;
            if ( paren )
            {
                os << ')';
            }
            
            os << ' ' << Lexer::str(op) << (cumulative ? " ..." : " ..");
        }

        const Shared<Expr> pack;
        const Lexer::Operator op;
        const bool cumulative;
    };
    
    
    struct CastExpr : public Expr
    {
        CastExpr( const Position& position, const std::string& type, const Typename base, const Shared<Expr> arg )
        : Expr(position, Cast), type(type), base(base), arg(arg)
        {
        }
        
        CastExpr( const Position& position, const Typename type, const Shared<Expr> arg )
        : Expr(position, Cast), type(str(type)), base(type), arg(arg)
        {
        }
        
        void print( std::ostream& os ) const
        {
            os << type << '(';
            if ( arg )
            {
                os << *arg;
            }
            os << ')';
        }
        
        const std::string type;
        const Typename base;
        const Shared<Expr> arg;
    };
    
    
    struct BuiltinExpr : public Expr
    {
        BuiltinExpr( const Position& position, const std::string& func, const Shared<Expr> arg )
        : Expr(position, Builtin), func(func), arg(arg)
        {
        }

        void print( std::ostream& os ) const
        {
            os << '`' << func << '`' << '(' << *arg << ')';
        }

        const std::string func;
        const Shared<Expr> arg;
    };
    
    
    struct FormatExpr : public Expr
    {
        FormatExpr( const Position& position, const std::string str, std::map<size_t,Shared<Expr>>&& subs )
        : Expr(position, Format), str(str), subs(subs)
        {
        }
        
        void print( std::ostream& os ) const
        {
            size_t offset = 0;
            os << '"';
            for ( auto& sub : subs )
            {
                os << str.substr(offset, sub.first - offset) << '{' << sknd::str(*sub.second) << '}';
                offset = sub.first;
            }
            os << str.substr(offset) << '"';
        }
        
        const std::string str;
        const std::map<size_t,Shared<Expr>> subs;
    };
    
    
    struct BoundedExpr : public Expr
    {
        BoundedExpr( const Position& position, const Shared<Expr> index, const Shared<Expr> lower_value, const Shared<Expr> upper_value )
        : Expr(position, Bounded), index(index), lower_value(lower_value), upper_value(upper_value)
        {
        }
        
        void print( std::ostream& os ) const
        {
            os << '|' << *index;
            if ( lower_value && upper_value )
            {
                os << " <> " << *lower_value << " : " << *upper_value;
            }
            os << '|';
        }
        
        const Shared<Expr> index;
        const Shared<Expr> lower_value;
        const Shared<Expr> upper_value;
    };
    
    
    struct SubstituteExpr : public Expr
    {
        SubstituteExpr( const Position& position, const Shared<Expr> pack, const Shared<Expr> index, const Shared<Expr> value )
        : Expr(position, Substitute), pack(pack), index(index), value(value)
        {
        }
        
        void print( std::ostream& os ) const
        {
            os << *pack << "[" << *index << "] <- " << *value;
        }
        
        const Shared<Expr> pack;
        const Shared<Expr> index;
        const Shared<Expr> value;
    };
    
    
    inline const LiteralExpr& as_literal( const Expr& expr )
    {
        return static_cast<const LiteralExpr&>(expr);
    }
    
    inline const IntExpr& as_int( const Expr& expr )
    {
        return static_cast<const IntExpr&>(expr);
    }
    
    inline const RealExpr& as_real( const Expr& expr )
    {
        return static_cast<const RealExpr&>(expr);
    }
    
    inline const BoolExpr& as_bool( const Expr& expr )
    {
        return static_cast<const BoolExpr&>(expr);
    }

    inline const StrExpr& as_str( const Expr& expr )
    {
        return static_cast<const StrExpr&>(expr);
    }
    
    inline const IdenfitierExpr& as_identifier( const Expr& expr )
    {
        return static_cast<const IdenfitierExpr&>(expr);
    }
    
    inline const ListExpr& as_list( const Expr& expr )
    {
        return static_cast<const ListExpr&>(expr);
    }
    
    inline const ExpandExpr& as_expand( const Expr& expr )
    {
        return static_cast<const ExpandExpr&>(expr);
    }
    
    inline const UnaryExpr& as_unary( const Expr& expr )
    {
        return static_cast<const UnaryExpr&>(expr);
    }
    
    inline const BinaryExpr& as_binary( const Expr& expr )
    {
        return static_cast<const BinaryExpr&>(expr);
    }
    
    inline const FoldExpr& as_fold( const Expr& expr )
    {
        return static_cast<const FoldExpr&>(expr);
    }
    
    inline const SelectExpr& as_select( const Expr& expr )
    {
        return static_cast<const SelectExpr&>(expr);
    }
    
    inline const CoalesceExpr& as_coalesce( const Expr& expr )
    {
        return static_cast<const CoalesceExpr&>(expr);
    }
    
    inline const IdentityExpr& as_identity( const Expr& expr )
    {
        return static_cast<const IdentityExpr&>(expr);
    }

    inline const ContainExpr& as_contain( const Expr& expr )
    {
        return static_cast<const ContainExpr&>(expr);
    }
    
    inline const IndexExpr& as_index( const Expr& expr )
    {
        return static_cast<const IndexExpr&>(expr);
    }

    inline const AccessExpr& as_access( const Expr& expr )
    {
        return static_cast<const AccessExpr&>(expr);
    }
    
    inline const RangeExpr& as_range( const Expr& expr )
    {
        return static_cast<const RangeExpr&>(expr);
    }
    
    inline const ZipExpr& as_zip( const Expr& expr )
    {
        return static_cast<const ZipExpr&>(expr);
    }
    
    inline const CastExpr& as_cast( const Expr& expr )
    {
        return static_cast<const CastExpr&>(expr);
    }
    
    inline const BuiltinExpr& as_builtin( const Expr& expr )
    {
        return static_cast<const BuiltinExpr&>(expr);
    }
    
    inline const FormatExpr& as_format( const Expr& expr )
    {
        return static_cast<const FormatExpr&>(expr);
    }
    
    inline const BoundedExpr& as_bounded( const Expr& expr )
    {
        return static_cast<const BoundedExpr&>(expr);
    }
    
    inline const SubstituteExpr& as_substitute( const Expr& expr )
    {
        return static_cast<const SubstituteExpr&>(expr);
    }
    
    inline const Expr& expanded( const Expr& expr )
    {
        return expr.kind == Expr::Expand ? *as_expand(expr).item : expr;
    }

    inline const Expr& unwrapped( const Expr& expr )
    {
        return expr.kind == Expr::Expand ? *as_expand(expr).item : expr;
    }

    inline Shared<Expr> expanded( const Shared<Expr>& expr )
    {
        return expr->kind == Expr::Expand ? as_expand(*expr).item : expr;
    }
    
    inline Shared<Expr> unwrapped( const Shared<Expr>& expr )
    {
        return expr->kind == Expr::Expand ? as_expand(*expr).item : expr;
    }


    template<typename T>
    T recurse( const Expr& expr, function_view<T( const Expr& )> callback, function_view<void( T&, const T& )> update, T value )
    {
        switch ( expr.kind )
        {
            case Expr::Literal:
            case Expr::Identifier:
            {
                break;
            }
            case Expr::List:
            {
                auto& list = as_list(expr);
                for ( auto& item : list.items )
                {
                    update(value, callback(*item));
                }
                break;
            }
            case Expr::Expand:
            {
                auto& expand = as_expand(expr);
                update(value, callback(*expand.item));
                if ( expand.count )
                {
                    update(value, callback(*expand.count));
                }
                break;
            }
            case Expr::Index:
            {
                auto& index = as_index(expr);
                update(value, callback(*index.array));
                update(value, callback(*index.index));
                break;
            }
            case Expr::Access:
            {
                auto& access = as_access(expr);
                update(value, callback(*access.tensor));
                for ( auto& item : access.indices )
                {
                    update(value, callback(*item));
                }
                break;
            }
            case Expr::Range:
            {
                auto& range = as_range(expr);
                if ( range.first )
                {
                    update(value, callback(*range.first));
                }
                if ( range.last )
                {
                    update(value, callback(*range.last));
                }
                if ( range.stride )
                {
                    update(value, callback(*range.stride));
                }
                break;
            }
            case Expr::Zip:
            {
                auto& zip = as_zip(expr);
                for ( auto& item : zip.items )
                {
                    update(value, callback(*item));
                }
                break;
            }
            case Expr::Unary:
            {
                auto& unary = as_unary(expr);
                update(value, callback(*unary.arg));
                break;
            }
            case Expr::Binary:
            {
                auto& binary = as_binary(expr);
                update(value, callback(*binary.left));
                update(value, callback(*binary.right));
                break;
            }
            case Expr::Select:
            {
                auto& select = as_select(expr);
                update(value, callback(*select.cond));
                update(value, callback(*select.left));
                if ( select.right )
                {
                    update(value, callback(*select.right));
                }
                break;
            }
            case Expr::Coalesce:
            {
                auto& coalesce = as_coalesce(expr);
                update(value, callback(*coalesce.condition));
                update(value, callback(*coalesce.alternate));
                break;
            }
            case Expr::Identity:
            {
                auto& iden = as_identity(expr);
                update(value, callback(*iden.left));
                update(value, callback(*iden.right));
                break;
            }
            case Expr::Contain:
            {
                auto& contain = as_contain(expr);
                update(value, callback(*contain.item));
                update(value, callback(*contain.pack));
                break;
            }
            case Expr::Fold:
            {
                auto& fold = as_fold(expr);
                update(value, callback(*fold.pack));
                break;
            }
            case Expr::Cast:
            {
                auto& cast = as_cast(expr);
                if ( cast.arg )
                {
                    update(value, callback(*cast.arg));
                }
                break;
            }
            case Expr::Builtin:
            {
                auto& builtin = as_builtin(expr);
                update(value, callback(*builtin.arg));
                break;
            }
            case Expr::Format:
            {
                auto& format = as_format(expr);
                for ( auto& item : format.subs )
                {
                    update(value, callback(*item.second));
                }
                break;
            }
            case Expr::Substitute:
            {
                auto& substitute = as_substitute(expr);
                update(value, callback(*substitute.pack));
                update(value, callback(*substitute.index));
                update(value, callback(*substitute.value));
                break;
            }
            case Expr::Bounded:
            {
                auto& bounded = as_bounded(expr);
                update(value, callback(*bounded.index));
                if ( bounded.lower_value )
                {
                    update(value, callback(*bounded.lower_value));
                }
                if ( bounded.upper_value )
                {
                    update(value, callback(*bounded.upper_value));
                }
                break;
            }
        }
        return value;
    }

    inline void recurse( const Expr& expr, function_view<void( const Expr& )> callback )
    {
        recurse<int>(expr, [&]( const Expr& e ){ callback(e); return 0; }, []( int&, const int ){}, 0);
    }

    inline Result<void> recurse_result( const Expr& expr, function_view<Result<void>( const Expr& )> callback )
    {
        return recurse<Result<void>>(expr, callback, []( Result<void>& x, const Result<void>& y ){ if ( x && !y ) x = y; }, Result<void>());
    }

    inline bool any_recurse( const Expr& expr, function_view<bool( const Expr& )> callback, const bool init = false )
    {
        return recurse<bool>(expr, callback, []( bool& x, const bool y ){ x |= y; }, init);
    }

    inline bool all_recurse( const Expr& expr, function_view<bool( const Expr& )> callback, const bool init = true )
    {
        return recurse<bool>(expr, callback, []( bool& x, const bool y ){ x &= y; }, init);
    }

    inline Result<bool> any_recurse_result( const Expr& expr, function_view<Result<bool>( const Expr& )> callback, 
                                           const bool init = false )
    {
        return recurse<Result<bool>>(expr, callback, []( Result<bool>& x, const Result<bool>& y ){ *x |= *y; }, init);
    }

    inline Result<bool> all_recurse_result( const Expr& expr, function_view<Result<bool>( const Expr& )> callback, 
                                           const bool init = true )
    {
        return recurse<Result<bool>>(expr, callback, []( Result<bool>& x, const Result<bool>& y ){ *x &= *y; }, init);
    }

    inline void preorder_traverse( const Expr& expr, function_view<void( const Expr& )> callback )
    {
        callback(expr);
        recurse(expr, [&]( const Expr& e ){ preorder_traverse(e, callback); });
    }

    inline void postorder_traverse( const Expr& expr, function_view<void( const Expr& )> callback )
    {
        recurse(expr, [&]( const Expr& e ){ postorder_traverse(e, callback); });
        callback(expr);
    }

    inline bool any_of( const Expr& expr, function_view<bool( const Expr& )> callback )
    {
        return any_recurse(expr, [&]( const Expr& e ){ return any_of(e, callback); }, callback(expr));
    }

    inline bool all_of( const Expr& expr, function_view<bool( const Expr& )> callback )
    {
        return all_recurse(expr, [&]( const Expr& e ){ return all_of(e, callback); }, callback(expr));
    }


    inline bool is_empty_list( const Expr& expr )
    {
        return expr.kind == Expr::List && as_list(expr).items.empty();
    }

    inline bool is_const_expr( const Expr& expr )
    {
        return all_of(expr, [&]( const Expr& e ){ return e.kind != Expr::Identifier; });
    }

    inline bool is_identifier( const Expr& expr, const std::string& iden )
    {
        return expr.kind == Expr::Identifier && as_identifier(expr).name == iden;
    }

    inline bool is_unary( const Expr& expr, const Lexer::Operator op )
    {
        return expr.kind == Expr::Unary && as_unary(expr).op == op;
    }

    inline bool is_binary( const Expr& expr, const Lexer::Operator op )
    {
        return expr.kind == Expr::Binary && as_binary(expr).op == op;
    }

    inline bool is_fold( const Expr& expr, const Lexer::Operator op )
    {
        return expr.kind == Expr::Fold && as_fold(expr).op == op;
    }
    
}   // namespace sknd


#endif
