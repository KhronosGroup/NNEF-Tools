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

#ifndef _NNEF_VALUE_H_
#define _NNEF_VALUE_H_

#include <string>
#include <array>


namespace nnef
{
    
    class Value
    {
    public:
        
        typedef int integer_t;
        typedef float scalar_t;
        typedef bool logical_t;
        typedef std::string string_t;
        typedef std::vector<Value> items_t;
        typedef struct { std::string id; } tensor_t;
        
        enum Kind { None, Integer, Scalar, Logical, String, Tensor, Array, Tuple };
        
    private:
        
        Value( const Kind kind, const integer_t& value )
        : _kind(kind), _integer(value)
        {
        }

        Value( const Kind kind, const scalar_t& value )
        : _kind(kind), _scalar(value)
        {
        }

        Value( const Kind kind, const logical_t& value )
        : _kind(kind), _logical(value)
        {
        }

        Value( const Kind kind, const string_t& value )
        : _kind(kind), _string(value)
        {
        }

        Value( const Kind kind, const tensor_t& value )
        : _kind(kind), _tensor(value)
        {
        }

        Value( const Kind kind, const items_t& value )
        : _kind(kind), _items(value)
        {
        }
        
        Value( const Kind kind, items_t&& items )
        : _kind(kind), _items(std::forward<items_t>(items))
        {
        }
        
    public:
        
        static Value integer( const integer_t& value )
        {
            return Value(Integer, value);
        }
        
        static Value scalar( const scalar_t& value )
        {
            return Value(Scalar, value);
        }
        
        static Value logical( const logical_t& value )
        {
            return Value(Logical, value);
        }
        
        static Value string( const string_t& value )
        {
            return Value(String, value);
        }
        
        static Value tensor( const tensor_t& value )
        {
            return Value(Tensor, value);
        }

        static Value array( const items_t& value )
        {
            return Value(Array, value);
        }

        static Value tuple( const items_t& value )
        {
            return Value(Tuple, value);
        }
        
        static Value array( items_t&& items )
        {
            return Value(Array, std::forward<items_t>(items));
        }
        
        static Value tuple( items_t&& items )
        {
            return Value(Tuple, std::forward<items_t>(items));
        }

        static Value none()
        {
            return Value();
        }

        static Value make( const integer_t& value )
        {
            return Value(Integer, value);
        }

        static Value make( const scalar_t& value )
        {
            return Value(Scalar, value);
        }

        static Value make( const logical_t& value )
        {
            return Value(Logical, value);
        }

        static Value make( const string_t& value )
        {
            return Value(String, value);
        }

        static Value make( const tensor_t& value )
        {
            return Value(Tensor, value);
        }
        
    public:
        
        Value()
        : _kind(None)
        {
        }
        
        Value( const Value& other )
        {
            if ( &other != this )
            {
                construct(other);
            }
        }
        
        Value( Value&& other )
        {
            if ( &other != this )
            {
                move(other);
            }
        }
        
        ~Value()
        {
            destroy();
        }
        
        Value& operator=( const Value& other )
        {
            if ( &other != this )
            {
                destroy();
                construct(other);
            }
            return *this;
        }
        
        Value& operator=( Value&& other )
        {
            if ( &other != this )
            {
                destroy();
                move(other);
            }
            return *this;
        }
        
        operator bool() const
        {
            return _kind != None;
        }
        
        Kind kind() const
        {
            return _kind;
        }
        
        const integer_t& integer() const
        {
            checkKind(Integer);
            return _integer;
        }
        
        const scalar_t& scalar() const
        {
            checkKind(Scalar);
            return _scalar;
        }
        
        const logical_t& logical() const
        {
            checkKind(Logical);
            return _logical;
        }
        
        const string_t& string() const
        {
            checkKind(String);
            return _string;
        }
        
        const tensor_t& tensor() const
        {
            checkKind(Tensor);
            return _tensor;
        }

        const items_t& array() const
        {
            checkKind(Array);
            return _items;
        }

        const items_t& tuple() const
        {
            checkKind(Tuple);
            return _items;
        }

        const items_t& items() const
        {
            checkNested();
            return _items;
        }

        size_t size() const
        {
            checkNested();
            return _items.size();
        }

        const Value& operator[]( const size_t i ) const
        {
            checkNested();
            return _items[i];
        }

        bool operator==( const Value& other ) const
        {
            return equals(other);
        }

        bool operator!=( const Value& other ) const
        {
            return !equals(other);
        }
        
    private:
        
        void checkKind( const Kind kind ) const
        {
            if ( _kind != kind )
            {
                throw std::runtime_error("argument kind mismatch");
            }
        }
        
        void checkNested() const
        {
            if ( _kind != Array && _kind != Tuple )
            {
                throw std::runtime_error("argument kind mismatch");
            }
        }
        
        void move( Value& other )
        {
            _kind = other._kind;
            switch ( _kind )
            {
                case Array:
                case Tuple:
                {
                    new(&_items) items_t(std::move(other._items));
                    break;
                }
                case String:
                {
                    new(&_string) string_t(std::move(other._string));
                    break;
                }
                case Tensor:
                {
                    new(&_tensor) tensor_t(std::move(other._tensor));
                    break;
                }
                case Integer:
                {
                    _integer = other._integer;
                    break;
                }
                case Scalar:
                {
                    _scalar = other._scalar;
                    break;
                }
                case Logical:
                {
                    _logical = other._logical;
                    break;
                }
                case None:
                {
                    break;
                }
            }
        }
        
        void construct( const Value& other )
        {
            _kind = other._kind;
            switch ( _kind )
            {
                case Array:
                case Tuple:
                {
                    new(&_items) items_t(other._items);
                    break;
                }
                case String:
                {
                    new(&_string) string_t(other._string);
                    break;
                }
                case Tensor:
                {
                    new(&_tensor) tensor_t(other._tensor);
                    break;
                }
                case Integer:
                {
                    _integer = other._integer;
                    break;
                }
                case Scalar:
                {
                    _scalar = other._scalar;
                    break;
                }
                case Logical:
                {
                    _logical = other._logical;
                    break;
                }
                case None:
                {
                    break;
                }
            }
        }
        
        void destroy()
        {
            switch ( _kind )
            {
                case Array:
                case Tuple:
                {
                    _items.~items_t();
                    break;
                }
                case String:
                {
                    _string.~string_t();
                    break;
                }
                case Tensor:
                {
                    _tensor.~tensor_t();
                    break;
                }
                default:
                {
                    break;
                }
            }
        }

        bool equals( const Value& other ) const
        {
            if ( _kind != other._kind )
            {
                return false;
            }
            switch ( _kind )
            {
                case Array:
                case Tuple:
                {
                    return _items == other._items;
                }
                case String:
                {
                    return _string == other._string;
                }
                case Tensor:
                {
                    return _tensor.id == other._tensor.id;
                }
                case Integer:
                {
                    return _integer == other._integer;
                }
                case Scalar:
                {
                    return _scalar == other._scalar;
                }
                case Logical:
                {
                    return _logical == other._logical;
                }
                case None:
                {
                    return true;
                }
            }
        }
        
    private:
        
        Kind _kind;
        union
        {
            integer_t _integer;
            scalar_t _scalar;
            logical_t _logical;
            string_t _string;
            tensor_t _tensor;
            items_t _items;
        };
    };
    

    inline std::ostream& operator<<( std::ostream& os, const Value& arg )
    {
        switch ( arg.kind() )
        {
            case Value::None:
            {
                os << "none";
                break;
            }
            case Value::Integer:
            {
                os << arg.integer();
                break;
            }
            case Value::Scalar:
            {
                os << arg.scalar();
                if ( (Value::integer_t)arg.scalar() == arg.scalar() )
                {
                    os << ".0";
                }
                break;
            }
            case Value::Logical:
            {
                os << std::boolalpha << arg.logical();
                break;
            }
            case Value::String:
            {
                os << '\'' << arg.string() << '\'';
                break;
            }
            case Value::Tensor:
            {
                os << arg.tensor().id;
                break;
            }
            case Value::Array:
            {
                os << '[';
                for ( size_t i = 0; i < arg.size(); ++i )
                {
                    if ( i )
                    {
                        os << ',';
                    }
                    os << arg[i];
                }
                os << ']';
                break;
            }
            case Value::Tuple:
            {
                os << '(';
                for ( size_t i = 0; i < arg.size(); ++i )
                {
                    if ( i )
                    {
                        os << ',';
                    }
                    os << arg[i];
                }
                os << ')';
                break;
            }
        }
        return os;
    }
    
}   // namespace nnef


#endif
