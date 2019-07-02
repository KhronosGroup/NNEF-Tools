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

#ifndef _NNEF_EXPRESSION_H_
#define _NNEF_EXPRESSION_H_

#include "../common/dictionary.h"
#include "../common/typespec.h"
#include "../common/lexer.h"
#include <functional>
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <string>


namespace nnef
{

    template<typename T>
    using Shared = std::shared_ptr<T>;
    


    class Expr
    {
    public:

        typedef Error::Position Position;

        enum Kind { Literal, Identifier, Array, Tuple, Subscript, Comprehension, Unary, Binary, Select, Invocation, Builtin };

    public:

        Expr( const Position& position )
        : _position(position)
        {
        }

        const Position& position() const
        {
            return _position;
        }

        virtual ~Expr() {}

        virtual Kind kind() const = 0;
        virtual const Type* type() const = 0;
        virtual void print( std::ostream& os ) const = 0;

    private:

        Position _position;
    };


    inline std::ostream& operator<<( std::ostream& os, const Expr& expr )
    {
        expr.print(os);
        return os;
    }


    template<typename V>
    class LiteralExpr : public Expr
    {
    public:

        typedef V value_type;

    public:

        LiteralExpr( const Position& position, const value_type& value, const Type* type )
        : Expr(position), _value(value), _type(type)
        {
        }

        const value_type& value() const
        {
            return _value;
        }

        virtual Kind kind() const
        {
            return Literal;
        }

        virtual const Type* type() const
        {
            return _type;
        }

        virtual void print( std::ostream& os ) const
        {
            print(os, _value);
        }

    private:

        template<typename S>
        static void print( std::ostream& os, const S& value )
        {
            os << std::boolalpha << value;
        }

        static void print( std::ostream& os, const std::string& value )
        {
            os << '\'' << value << '\'';
        }

    protected:

        value_type _value;
        const Type* _type;
    };


    typedef LiteralExpr<float> ScalarExpr;
    typedef LiteralExpr<int> IntegerExpr;
    typedef LiteralExpr<bool> LogicalExpr;
    typedef LiteralExpr<std::string> StringExpr;


    class IdentifierExpr : public Expr
    {
    public:

        IdentifierExpr( const Position& position, const std::string& name, const Type* type )
        : Expr(position), _name(name), _type(type)
        {
        }

        const std::string& name() const
        {
            return _name;
        }

        virtual Kind kind() const
        {
            return Identifier;
        }

        virtual const Type* type() const
        {
            return _type;
        }

        virtual void print( std::ostream& os ) const
        {
            os << _name;
        }

    private:

        std::string _name;
        const Type* _type;
    };


    class SubscriptExpr : public Expr
    {
    public:

        SubscriptExpr( const Position& position, const Shared<Expr>& sequence, const Shared<Expr>& begin, const Shared<Expr>& end, const Type* type )
        : Expr(position), _sequence(sequence), _begin(begin), _end(end), _type(type)
        {
        }

        virtual bool isRange() const
        {
            return _begin != _end || !_begin;
        }

        const Expr& sequence() const
        {
            return *_sequence;
        }

        const Expr* begin() const
        {
            return _begin.get();
        }

        const Expr* end() const
        {
            return _end.get();
        }

        virtual Kind kind() const
        {
            return Subscript;
        }

        virtual const Type* type() const
        {
            return _type;
        }

        virtual void print( std::ostream& os ) const
        {
            _sequence->print(os);
            os << '[';
            if ( _begin )
            {
                _begin->print(os);
            }
            if ( isRange() )
            {
                os << ':';
            }
            if ( _end )
            {
                _begin->print(os);
            }
            os << ']';
        }

    private:

        const Shared<Expr> _sequence;
        const Shared<Expr> _begin;
        const Shared<Expr> _end;
        const Type* _type;
    };


    class ItemExpr : public Expr
    {
    public:

        ItemExpr( const Position& position, const Type* type )
        : Expr(position), _type(type)
        {
        }

        ItemExpr( const Position& position, std::vector<Shared<Expr>>& items, const Type* type )
        : Expr(position), _items(std::move(items)), _type(type)
        {
        }

        size_t size() const
        {
            return _items.size();
        }

        const Expr& item( const size_t i ) const
        {
            return *_items[i];
        }

        virtual const Type* type() const
        {
            return _type;
        }

    protected:
        
        std::vector<Shared<Expr>> _items;
        const Type* _type;
    };


    class ArrayExpr : public ItemExpr
    {
    public:

        ArrayExpr( const Position& position, const Type* type )
        : ItemExpr(position, type)
        {
        }

        ArrayExpr( const Position& position, std::vector<Shared<Expr>>& items, const Type* type )
        : ItemExpr(position, items, type)
        {
        }

        virtual Kind kind() const
        {
            return Array;
        }

        virtual void print( std::ostream& os ) const
        {
            os << '[';
            for ( size_t i = 0; i < _items.size(); ++i )
            {
                if ( i )
                {
                    os << ',';
                }
                _items[i]->print(os);
            }
            os << ']';
        }
    };


    class TupleExpr : public ItemExpr
    {
    public:
        
        TupleExpr( const Position& position, const Type* type )
        : ItemExpr(position, type)
        {
        }

        TupleExpr( const Position& position, std::vector<Shared<Expr>>& items, const Type* type )
        : ItemExpr(position, items, type)
        {
        }

        virtual Kind kind() const
        {
            return Tuple;
        }

        virtual void print( std::ostream& os ) const
        {
            os << '(';
            for ( size_t i = 0; i < _items.size(); ++i )
            {
                if ( i )
                {
                    os << ',';
                }
                _items[i]->print(os);
            }
            os << ')';
        }
    };


    class ComprehensionExpr : public Expr
    {
    public:

        ComprehensionExpr( const Position& position, std::vector<Shared<Expr>>& iterators, std::vector<Shared<Expr>>& iterables,
                          const Shared<Expr>& condition, const Shared<Expr>& item, const Type* type )
        : Expr(position), _iterators(std::move(iterators)), _iterables(std::move(iterables)), _condition(condition), _item(item), _type(type)
        {
        }
        
        size_t iteratorCount() const
        {
            return _iterators.size();
        }

        const Expr& iterator( const size_t i ) const
        {
            return *_iterators[i];
        }

        const Expr& iterable( const size_t i ) const
        {
            return *_iterables[i];
        }

        const Expr* condition() const
        {
            return _condition.get();
        }
        
        const Expr& item() const
        {
            return *_item;
        }

        virtual Kind kind() const
        {
            return Comprehension;
        }

        virtual const Type* type() const
        {
            return _type;
        }

        virtual void print( std::ostream& os ) const
        {
            os << '[';
            os << "for ";
            for ( size_t i = 0; i < _iterators.size(); ++i )
            {
                if ( i )
                {
                    os << ", ";
                }
                _iterators[i]->print(os);
                os << " in ";
                _iterables[i]->print(os);
            }
            if ( _condition )
            {
                os << " if ";
                _condition->print(os);
            }
            os << " yield ";
            _item->print(os);
            os << ']';
        }

    private:

        const std::vector<Shared<Expr>> _iterators;
        const std::vector<Shared<Expr>> _iterables;
        const Shared<Expr> _condition;
        const Shared<Expr> _item;
        const Type* _type;
    };


    class UnaryExpr : public Expr
    {
    public:

        UnaryExpr( const Position& position, const Shared<Expr>& right, int op, const Type* type )
        : Expr(position), _right(right), _type(type), _op(op)
        {
        }

        const Expr& right() const
        {
            return *_right;
        }

        int op() const
        {
            return _op;
        }

        virtual Kind kind() const
        {
            return Unary;
        }

        virtual const Type* type() const
        {
            return _type;
        }

        virtual void print( std::ostream& os ) const
        {
            const std::string str = Lexer::tokenString(_op);

            os << str;
            if ( str.length() > 1 )
            {
                os << '(';
            }
            _right->print(os);
            if ( str.length() > 1 )
            {
                os << ')';
            }
        }

    private:

        const Shared<Expr> _right;
        const Type* _type;
        int _op;
    };


    class BinaryExpr : public Expr
    {
    public:

        BinaryExpr( const Position& position, const Shared<Expr>& left, const Shared<Expr>& right, int op, const Type* type )
        : Expr(position), _left(left), _right(right), _type(type), _op(op)
        {
        }

        const Expr& left() const
        {
            return *_left;
        }

        const Expr& right() const
        {
            return *_right;
        }

        int op() const
        {
            return _op;
        }

        virtual Kind kind() const
        {
            return Binary;
        }

        virtual const Type* type() const
        {
            return _type;
        }

        virtual void print( std::ostream& os ) const
        {
            if ( _left->kind() == Binary )
            {
                os << '(';
            }
            _left->print(os);
            if ( _left->kind() == Binary )
            {
                os << ')';
            }
            os << ' ' << Lexer::tokenString(_op) << ' ';
            if ( _right->kind() == Binary )
            {
                os << '(';
            }
            _right->print(os);
            if ( _right->kind() == Binary )
            {
                os << ')';
            }
        }

    private:

        const Shared<Expr> _left;
        const Shared<Expr> _right;
        const Type* _type;
        int _op;
    };


    class BuiltinExpr : public Expr
    {
    public:

        BuiltinExpr( const Position& position, const Shared<Expr>& arg, int op, const Type* type )
        : Expr(position), _arg(arg), _type(type), _op(op)
        {
        }

        const Expr& arg() const
        {
            return *_arg;
        }

        int op() const
        {
            return _op;
        }

        virtual Kind kind() const
        {
            return Builtin;
        }

        virtual const Type* type() const
        {
            return _type;
        }

        virtual void print( std::ostream& os ) const
        {
            os << Lexer::tokenString(_op) << '(';
            _arg->print(os);
            os << ')';
        }

    private:

        const Shared<Expr> _arg;
        const Type* _type;
        int _op;
    };


    class SelectExpr : public Expr
    {
    public:

        SelectExpr( const Position& position, const Shared<Expr>& condition, const Shared<Expr>& trueValue, const Shared<Expr>& falseValue, const Type* type )
        : Expr(position), _cond(condition), _true(trueValue), _false(falseValue), _type(type)
        {
        }

        const Expr& condition() const
        {
            return *_cond;
        }

        const Expr& trueValue() const
        {
            return *_true;
        }

        const Expr& falseValue() const
        {
            return *_false;
        }

        virtual Kind kind() const
        {
            return Select;
        }

        virtual const Type* type() const
        {
            return _type;
        }

        virtual void print( std::ostream& os ) const
        {
            _true->print(os);
            os << " if ";
            _cond->print(os);
            os << " else ";
            _false->print(os);
        }

    private:

        const Shared<Expr> _cond;
        const Shared<Expr> _true;
        const Shared<Expr> _false;
        const Type* _type;
    };


    class InvocationExpr : public Expr
    {
	public:
		
		typedef Dictionary<Shared<Expr>>::const_iterator const_iterator;
		
    public:

        InvocationExpr( const Position& position, const std::string& target, Dictionary<Shared<Expr>>& args, const Type* type,
                       const PrimitiveType* dataType = nullptr )
        : Expr(position), _target(target), _dataType(dataType), _args(std::move(args)), _type(type)
        {
        }

        const std::string& target() const
        {
            return _target;
        }
        
        const PrimitiveType* dataType() const
        {
            return _dataType;
        }

        const Expr* arg( const std::string& name ) const
        {
            auto it = _args.find(name);
            return it != _args.end() ? it->second.get() : nullptr;
        }
		
		const_iterator begin() const
		{
			return _args.begin();
		}
		
		const_iterator end() const
		{
			return _args.end();
		}

        virtual Kind kind() const
        {
            return Kind::Invocation;
        }

        virtual const Type* type() const
        {
            return _type;
        }

        virtual void print( std::ostream& os ) const
        {
            os << _target;
            if ( _dataType )
            {
                os << '<' << _dataType->toString() << '>';
            }
            os << '(';
            for ( auto it = _args.begin(); it != _args.end(); ++it )
            {
                if ( it != _args.begin() )
                {
                    os << ", ";
                }
                os << it->first << " = " << *it->second;
            }
            os << ')';
        }

    private:

        std::string _target;
        const PrimitiveType* _dataType;
        Dictionary<Shared<Expr>> _args;
        const Type* _type;
    };

}   // namespace nnef


#endif
