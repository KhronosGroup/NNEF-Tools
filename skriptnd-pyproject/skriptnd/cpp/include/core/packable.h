#ifndef _TS_PACKABLE_H_
#define _TS_PACKABLE_H_

#include <algorithm>
#include <type_traits>
#include <cassert>
#include <iostream>
#include <sstream>


namespace nd
{

    template<typename T, typename S = size_t>
    class Packable
    {
        static const S UnpackedSize = -1;
        
    public:
        
        typedef T value_type;
        typedef S size_type;
        typedef value_type* iterator;
        typedef const value_type* const_iterator;
        typedef value_type& reference;
        typedef const value_type& const_reference;
        
    public:
        
        static const Packable& empty()
        {
            static const Packable _empty(nullptr, 0);
            return _empty;
        }
        
    public:
        
        Packable( const value_type& value = value_type() )
        : _size(UnpackedSize), _value(value)
        {
        }
        
        template<typename U = value_type, typename = std::enable_if_t<std::is_move_constructible_v<U>>>
        Packable( value_type&& value )
        : _size(UnpackedSize), _value(std::forward<value_type>(value))
        {
        }
        
        Packable( const value_type& value, const size_type size )
        : _size(size), _values(allocate(size))
        {
            std::uninitialized_fill_n(_values, size, value);
        }
        
        Packable( const value_type* values, const size_type size )
        : _size(size), _values(allocate(size))
        {
            if ( values )
            {
                std::uninitialized_copy_n(values, size, _values);
            }
            else if ( size )
            {
                throw std::invalid_argument("size must be zero if values is null");
            }
        }
        
        Packable( const value_type* values, const size_type size, const size_type stride )
        : _size(size), _values(allocate(size))
        {
            if ( values )
            {
                auto ptr = _values;
                for ( size_t i = 0; i < size; ++i, ++ptr, values += stride )
                {
                    new(ptr) value_type(*values);
                }
            }
            else if ( size )
            {
                throw std::invalid_argument("size must be zero if values is null");
            }
        }
        
        Packable( std::initializer_list<value_type> items )
        : _size(items.size()), _values(allocate(items.size()))
        {
            std::uninitialized_copy_n(items.begin(), items.size(), _values);
        }
        
        Packable( const Packable& other )
        {
            construct(other);
        }
        
        template<typename U = value_type, typename = std::enable_if_t<std::is_move_constructible_v<U>>>
        Packable( Packable&& other )
        {
            construct(std::forward<Packable>(other));
        }
        
        ~Packable()
        {
            destruct();
        }
        
        Packable& operator=( const Packable& other )
        {
            destruct();
            construct(other);
            return *this;
        }
        
        template<typename U = value_type, typename = std::enable_if_t<std::is_move_constructible_v<U>>>
        Packable& operator=( Packable&& other )
        {
            destruct();
            construct(std::forward<Packable>(other));
            return *this;
        }
        
        Packable& operator=( const value_type& value )
        {
            destruct();
            new(&_value) value_type(value);
            _size = UnpackedSize;
            return *this;
        }
        
        template<typename U = value_type, typename = std::enable_if_t<std::is_move_constructible_v<U>>>
        Packable& operator=( value_type&& value )
        {
            destruct();
            new(&_value) value_type(std::forward<value_type>(value));
            _size = UnpackedSize;
            return *this;
        }
        
        bool packed() const
        {
            return _size != UnpackedSize;
        }
        
        size_type size() const
        {
            assert(packed());
            return _size;
        }
        
        const_reference operator*() const
        {
            assert(!packed());
            return _value;
        }
        
        reference operator*()
        {
            assert(!packed());
            return _value;
        }
        
        const value_type* operator->() const
        {
            assert(!packed());
            return &_value;
        }
        
        value_type* operator->()
        {
            assert(!packed());
            return &_value;
        }
        
        explicit operator const_reference() const
        {
            assert(!packed());
            return _value;
        }
        
        explicit operator reference()
        {
            assert(!packed());
            return _value;
        }
        
        explicit operator const value_type*() const
        {
            assert(packed());
            return _values;
        }
        
        explicit operator value_type*()
        {
            assert(packed());
            return _values;
        }
        
        reference operator[]( size_type idx )
        {
            assert(packed());
            assert(idx < size());
            return _values[idx];
        }
        
        const_reference operator[]( size_type idx ) const
        {
            assert(packed());
            assert(idx < size());
            return _values[idx];
        }
        
        Packable operator()( const size_t first, const size_t last ) const
        {
            assert(packed());
            assert(first <= last && last <= size());
            return Packable(_values + first, last - first);
        }
        
        Packable operator()( const size_t first, const size_t count, const size_t stride ) const
        {
            assert(packed());
            assert(first + (count - 1) * stride < size());
            return Packable(_values + first, count, stride);
        }
        
        iterator begin()
        {
            assert(packed());
            return _values;
        }
        
        iterator end()
        {
            assert(packed());
            return _values + _size;
        }
        
        const_iterator begin() const
        {
            assert(packed());
            return _values;
        }
        
        const_iterator end() const
        {
            assert(packed());
            return _values + _size;
        }
        
        bool operator==( const Packable& other ) const
        {
            return equals(other);
        }
        
        bool operator!=( const Packable& other ) const
        {
            return !equals(other);
        }
        
        bool operator==( const value_type& value ) const
        {
            return !packed() && _value == value;
        }
        
        bool operator!=( const value_type& value ) const
        {
            return !(*this == value);
        }
        
    private:
        
        static value_type* allocate( const size_t size )
        {
            return size ? (value_type*)new char[size * sizeof(value_type)] : nullptr;
        }
        
        void deallocate()
        {
            delete[] (char*)_values;
        }
        
        void construct( const Packable& other )
        {
            _size = other._size;
            if ( other.packed() )
            {
                _values = allocate(_size);
                std::uninitialized_copy_n(other._values, _size, _values);
            }
            else
            {
                new(&_value) value_type(other._value);
            }
        }
        
        void construct( Packable&& other )
        {
            _size = other._size;
            if ( other.packed() )
            {
                _values = other._values;
                other._values = nullptr;
                other._size = 0;
            }
            else
            {
                new(&_value) value_type(std::move(other._value));
            }
        }
        
        void destruct()
        {
            if ( packed() )
            {
                value_type* ptr = _values;
                for ( size_t i = 0; i < _size; ++i, ++ptr )
                {
                    ptr->~value_type();
                }
                deallocate();
            }
            else
            {
                _value.~value_type();
            }
        }
        
        bool equals( const Packable& other ) const
        {
            if ( _size != other._size )
            {
                return false;
            }
            if ( packed() )
            {
                return std::equal(begin(), end(), other.begin());
            }
            else
            {
                return _value == other._value;
            }
        }
        
    private:
        
        size_type _size;
        union
        {
            value_type _value;
            value_type* _values;
        };
    };
    
    
    template<typename T>
    std::ostream& operator<<( std::ostream& os, const Packable<T>& packable )
    {
        if ( packable.packed() )
        {
            os << '[';
            for ( size_t i = 0; i < packable.size(); ++i )
            {
                if ( i )
                {
                    os << ", ";
                }
                os << packable[i];
            }
            os << ']';
        }
        else
        {
            os << *packable;
        }
        return os;
    }
    
    template<typename T>
    std::string str( const Packable<T>& packable )
    {
        std::stringstream ss;
        ss << packable;
        return ss.str();
    }
    
}   // namespace nd


template<typename T>
struct std::hash<nd::Packable<T>>
{
    std::size_t operator()( const nd::Packable<T>& p ) const noexcept
    {
        std::hash<T> hasher;
        if ( p.packed() )
        {
            size_t hash = 0;
            for ( auto& v : p )
            {
                hash ^= hasher(v) + 0x9e3779b9 + (hash<<6) + (hash>>2);
            }
            return hash;
        }
        else
        {
            return hasher(*p);
        }
    }
};


namespace std
{

    template<typename T>
    std::string to_string( const nd::Packable<T>& packable )
    {
        std::stringstream ss;
        ss << packable;
        return ss.str();
    }

}   // namespace std


#endif
