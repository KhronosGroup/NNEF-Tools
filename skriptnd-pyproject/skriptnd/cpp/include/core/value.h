#ifndef _SKND_VALUE_H_
#define _SKND_VALUE_H_

#include "packable.h"
#include "types.h"
#include <iostream>
#include <sstream>


namespace sknd
{

    class Value;
    
    std::ostream& operator<<( std::ostream& os, const Value& value );
    
    
    class Value
    {
        typedef std::nullptr_t none_t;
        
    private:
        
        Value( const void* )
        : _type(Typename::Type), _none(nullptr, 0)
        {
        }
        
    public:
        
        static const Value& null()
        {
            static const Value _null = Value();
            return _null;
        }
        
        static const Value& empty()
        {
            static const Value _empty = Value(nullptr);
            return _empty;
        }
        
        template<typename T>
        static const Value empty()
        {
            return Value((T*)nullptr, 0);
        }
        
    public:
        
        Value()
        : _type(Typename::Type), _none(nullptr)
        {
        }
        
        explicit Value( const int_t& value )
        : _type(Typename::Int), _int(value)
        {
        }
        
        explicit Value( const real_t& value )
        : _type(Typename::Real), _real(value)
        {
        }
        
        explicit Value( const bool_t& value )
        : _type(Typename::Bool), _bool(value)
        {
        }
        
        explicit Value( const str_t& value )
        : _type(Typename::Str), _str(value)
        {
        }
        
        Value( const int_t& value, const size_t size )
        : _type(Typename::Int), _int(value, size)
        {
        }
        
        Value( const real_t& value, const size_t size )
        : _type(Typename::Real), _real(value, size)
        {
        }
        
        Value( const bool_t& value, const size_t size )
        : _type(Typename::Bool), _bool(value, size)
        {
        }
        
        Value( const str_t& value, const size_t size )
        : _type(Typename::Str), _str(value, size)
        {
        }
        
        Value( const int_t* values, const size_t count, const size_t stride = 1 )
        : _type(Typename::Int), _int(values, count, stride)
        {
        }
        
        Value( const real_t* values, const size_t count, const size_t stride = 1 )
        : _type(Typename::Real), _real(values, count, stride)
        {
        }
        
        Value( const bool_t* values, const size_t count, const size_t stride = 1 )
        : _type(Typename::Bool), _bool(values, count, stride)
        {
        }
        
        Value( const str_t* values, const size_t count, const size_t stride = 1 )
        : _type(Typename::Str), _str(values, count, stride)
        {
        }
        
        Value( const Value& other )
        {
            construct(other);
        }
        
        ~Value()
        {
            destruct();
        }
        
        Value& operator=( const Value& other )
        {
            destruct();
            construct(other);
            return *this;
        }
        
        Value& operator=( const int_t& value )
        {
            destruct();
            _type = Typename::Int;
            new (&_int) Packable<int_t>(value);
            return *this;
        }
        
        Value& operator=( const real_t& value )
        {
            destruct();
            _type = Typename::Real;
            new (&_real) Packable<real_t>(value);
            return *this;
        }
        
        Value& operator=( const bool_t& value )
        {
            destruct();
            _type = Typename::Bool;
            new (&_bool) Packable<bool_t>(value);
            return *this;
        }
        
        Value& operator=( const str_t& value )
        {
            destruct();
            _type = Typename::Str;
            new (&_str) Packable<str_t>(value);
            return *this;
        }
        
        Typename type() const
        {
            return _type;
        }
        
        bool packed() const
        {
            return _none.packed();
        }
        
        size_t size() const
        {
            assert(packed());
            return _none.size();
        }
        
        Value operator[]( const size_t idx ) const
        {
            assert(packed());
            return at(idx);
        }
        
        Value operator()( const size_t first, const size_t last ) const
        {
            assert(packed());
            return range(first, last - first);
        }
        
        Value operator()( const size_t first, const size_t count, const size_t stride ) const
        {
            assert(packed());
            return range(first, count, stride);
        }
        
        Value repeat( const size_t count ) const
        {
            assert(!packed());
            return repeated(count);
        }
        
        void assign( const size_t idx, const Value& value )
        {
            assert(packed());
            assert(!value.packed());
            set(idx, value);
        }
        
        explicit operator const int_t&() const
        {
            assert(type() == Typename::Int);
            return (const int_t&)_int;
        }
        
        explicit operator const real_t&() const
        {
            assert(type() == Typename::Real);
            return (const real_t&)_real;
        }
        
        explicit operator const bool_t&() const
        {
            assert(type() == Typename::Bool);
            return (const bool_t&)_bool;
        }
        
        explicit operator const str_t&() const
        {
            assert(type() == Typename::Str);
            return (const str_t&)_str;
        }
        
        explicit operator const int_t*() const
        {
            assert(type() == Typename::Int);
            return (const int_t*)_int;
        }
        
        explicit operator const real_t*() const
        {
            assert(type() == Typename::Real);
            return (const real_t*)_real;
        }
        
        explicit operator const bool_t*() const
        {
            assert(type() == Typename::Bool);
            return (const bool_t*)_bool;
        }
        
        explicit operator const str_t*() const
        {
            assert(type() == Typename::Str);
            return (const str_t*)_str;
        }
        
        explicit operator int_t&()
        {
            assert(type() == Typename::Int);
            return (int_t&)_int;
        }
        
        explicit operator real_t&()
        {
            assert(type() == Typename::Real);
            return (real_t&)_real;
        }
        
        explicit operator bool_t&()
        {
            assert(type() == Typename::Bool);
            return (bool_t&)_bool;
        }
        
        explicit operator str_t&()
        {
            assert(type() == Typename::Str);
            return (str_t&)_str;
        }
        
        explicit operator int_t*()
        {
            assert(type() == Typename::Int);
            return (int_t*)_int;
        }
        
        explicit operator real_t*()
        {
            assert(type() == Typename::Real);
            return (real_t*)_real;
        }
        
        explicit operator bool_t*()
        {
            assert(type() == Typename::Bool);
            return (bool_t*)_bool;
        }
        
        explicit operator str_t*()
        {
            assert(type() == Typename::Str);
            return (str_t*)_str;
        }
        
        template<typename T>
        const T* begin() const
        {
            return (const T*)(*this);
        }
        
        template<typename T>
        const T* end() const
        {
            return (const T*)(*this) + size();
        }
        
        template<typename T>
        T* begin()
        {
            return (T*)(*this);
        }
        
        template<typename T>
        T* end()
        {
            return (T*)(*this) + size();
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
        
        void construct( const Value& other )
        {
            _type = other._type;
            switch ( _type )
            {
                case Typename::Type:
                {
                    new (&_none) Packable<none_t>(other._none);
                    break;
                }
                case Typename::Int:
                {
                    new (&_int) Packable<int_t>(other._int);
                    break;
                }
                case Typename::Real:
                {
                    new (&_real) Packable<real_t>(other._real);
                    break;
                }
                case Typename::Bool:
                {
                    new (&_bool) Packable<bool_t>(other._bool);
                    break;
                }
                case Typename::Str:
                {
                    new (&_str) Packable<str_t>(other._str);
                    break;
                }
                default:
                {
                    assert(false);
                    break;
                }
            }
        }
        
        void destruct()
        {
            switch ( _type )
            {
                case Typename::Type:
                {
                    _none.~Packable<none_t>();
                    break;
                }
                case Typename::Int:
                {
                    _int.~Packable<int_t>();
                    break;
                }
                case Typename::Real:
                {
                    _real.~Packable<real_t>();
                    break;
                }
                case Typename::Bool:
                {
                    _bool.~Packable<bool_t>();
                    break;
                }
                case Typename::Str:
                {
                    _str.~Packable<str_t>();
                    break;
                }
                default:
                {
                    assert(false);
                    break;
                }
            }
        }
        
        Value at( const size_t idx ) const
        {
            switch ( _type )
            {
                case Typename::Int:
                {
                    return Value(_int[idx]);
                }
                case Typename::Real:
                {
                    return Value(_real[idx]);
                }
                case Typename::Bool:
                {
                    return Value(_bool[idx]);
                }
                case Typename::Str:
                {
                    return Value(_str[idx]);
                }
                default:
                {
                    assert(false);
                    return Value();
                }
            }
        }
        
        void set( const size_t idx, const Value& value )
        {
            switch ( _type )
            {
                case Typename::Int:
                {
                    _int[idx] = (const int_t)value._int;
                }
                case Typename::Real:
                {
                    _real[idx] = (const real_t)value._real;
                }
                case Typename::Bool:
                {
                    _bool[idx] = (const bool_t)value._bool;
                }
                case Typename::Str:
                {
                    _str[idx] = (const str_t)value._str;
                }
                default:
                {
                    assert(false);
                }
            }
        }
        
        Value range( const size_t offset, const size_t count, const size_t stride = 1 ) const
        {
            switch ( _type )
            {
                case Typename::Int:
                {
                    return Value((const int_t*)_int + offset, count, stride);
                }
                case Typename::Real:
                {
                    return Value((const real_t*)_real + offset, count, stride);
                }
                case Typename::Bool:
                {
                    return Value((const bool_t*)_bool + offset, count, stride);
                }
                case Typename::Str:
                {
                    return Value((const str_t*)_str + offset, count, stride);
                }
                default:
                {
                    assert(false);
                    return Value();
                }
            }
        }
        
        Value repeated( const size_t count ) const
        {
            switch ( _type )
            {
                case Typename::Int:
                {
                    return Value((const int_t)_int, count);
                }
                case Typename::Real:
                {
                    return Value((const real_t)_real, count);
                }
                case Typename::Bool:
                {
                    return Value((const bool_t)_bool, count);
                }
                case Typename::Str:
                {
                    return Value((const str_t)_str, count);
                }
                default:
                {
                    assert(false);
                    return Value();
                }
            }
        }
        
        bool equals( const Value& other ) const
        {
            if ( _type != other._type )
            {
                return false;
            }
            switch ( _type )
            {
                case Typename::Type:
                {
                    return _none == other._none;
                }
                case Typename::Int:
                {
                    return _int == other._int;
                }
                case Typename::Real:
                {
                    return _real == other._real;
                }
                case Typename::Bool:
                {
                    return _bool == other._bool;
                }
                case Typename::Str:
                {
                    return _str == other._str;
                }
                default:
                {
                    assert(false);
                    return false;
                }
            }
        }
        
    private:
        
        Typename _type;
        union
        {
            Packable<none_t> _none;
            Packable<int_t> _int;
            Packable<real_t> _real;
            Packable<bool_t> _bool;
            Packable<str_t> _str;
        };
    };
    
    
    inline std::ostream& operator<<( std::ostream& os, const Value& value )
    {
        if ( value.packed() )
        {
            if ( value.size() == 0 && value.type() != Typename::Type )
            {
                os << str(value.type()) << "([])";
            }
            else
            {
                os << '[';
                for ( size_t i = 0; i < value.size(); ++i )
                {
                    if ( i != 0 )
                    {
                        os << ", ";
                    }
                    os << value[i];
                }
                os << ']';
            }
        }
        else
        {
            switch ( value.type() )
            {
                case Typename::Int:
                {
                    os << (int_t)value;
                    break;
                }
                case Typename::Real:
                {
                    auto v = (real_t)value;
                    if ( (int_t)v == v )
                    {
                        os << (int_t)v << ".0";
                    }
                    else
                    {
                        os << v;
                    }
                    break;
                }
                case Typename::Bool:
                {
                    os << std::boolalpha << (bool_t)value;
                    break;
                }
                case Typename::Str:
                {
                    os << '\"' << (str_t)value << '\"';
                    break;
                }
                case Typename::Type:
                {
                    os << "null";
                    break;
                }
                default:
                {
                    assert(false);
                    break;
                }
            }
        }
        return os;
    }
    
}   // namespace sknd


namespace std
{
    
    inline string to_string( const sknd::Value& value )
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }
    
}   // namespace std


#endif
