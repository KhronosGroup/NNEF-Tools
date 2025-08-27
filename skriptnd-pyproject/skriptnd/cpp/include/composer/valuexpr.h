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

#ifndef _SKND_VALUE_EXPR_H_
#define _SKND_VALUE_EXPR_H_

#include "types.h"
#include "variant.h"
#include "function.h"
#include "tensorref.h"
#include <cstdint>
#include <optional>
#include <sstream>
#include <cassert>
#include <vector>
#include <limits>
#include <cmath>
#include <unordered_map>
#include <exception>


namespace sknd
{

    class ValueExpr;

    inline ValueExpr operator-( const ValueExpr& arg );
    inline ValueExpr operator+( const ValueExpr& left, const ValueExpr& right );
    inline ValueExpr operator-( const ValueExpr& left, const ValueExpr& right );
    inline ValueExpr operator*( const ValueExpr& left, const ValueExpr& right );
    inline ValueExpr operator/( const ValueExpr& left, const ValueExpr& right );
    inline ValueExpr ceil_div( const ValueExpr& left, const ValueExpr& right );
    
    
    class ValueExpr
    {
    public:
        
        struct PlaceholderExpr;
        struct IdentifierExpr;
        struct ReferenceExpr;
        struct SizeAccessExpr;
        struct ShapeAccessExpr;
        struct TensorAccessExpr;
        struct UnaryExpr;
        struct BinaryExpr;
        struct SelectExpr;
        struct FoldExpr;
        struct ListExpr;
        struct CastExpr;
        struct BoundedExpr;
        struct ConcatExpr;
        struct SliceExpr;
        struct SubscriptExpr;
        struct UniformExpr;
        struct RangeExpr;
        
        enum class IdentifierKind { LoopIndex, LoopLocal };
        
    private:
        
        typedef VariantData<int_t,real_t,bool_t,str_t,
                            PlaceholderExpr,IdentifierExpr,ReferenceExpr,
                            SizeAccessExpr,ShapeAccessExpr,TensorAccessExpr,
                            UnaryExpr,BinaryExpr,SelectExpr,FoldExpr,ListExpr,
                            CastExpr,BoundedExpr,ConcatExpr,SliceExpr,SubscriptExpr,
                            UniformExpr,RangeExpr> variant_data;
        
        typedef uint8_t index_type;
        typedef uint32_t size_type;
        
        static constexpr index_type NullIndex = -1;
        static constexpr size_type UnpackedSize = -1;
        
    public:
        
        enum Kind : char 
        {
            Null, Literal, Placeholder, Identifier, Reference, SizeAccess, ShapeAccess, TensorAccess,
            Unary, Binary, Select, Fold, List, Cast, Bounded, Concat, Slice, Subscript, Uniform, Range,
        };
        
        static const ValueExpr& null()
        {
            static const ValueExpr null = nullptr;
            return null;
        }
        
        static const ValueExpr& empty()
        {
            static const ValueExpr empty(nullptr, 0, Typename::Type);
            return empty;
        }
        
        template<typename T = real_t, typename = std::enable_if_t<std::is_same_v<T,real_t> || std::is_same_v<T,int_t>>>
        static const ValueExpr& positive_infinity()
        {
            static const ValueExpr inf = std::numeric_limits<real_t>::infinity();
            return inf;
        }
        
        template<typename T = real_t, typename = std::enable_if_t<std::is_same_v<T,real_t> || std::is_same_v<T,int_t>>>
        static const ValueExpr& negative_infinity()
        {
            static const ValueExpr inf = -std::numeric_limits<real_t>::infinity();
            return inf;
        }
        
        template<typename T>
        static Kind kind_of()
        {
            auto index = variant_data::index_of<T>();
            return index < 4 ? Literal : (Kind)(index - 2);
        }
        
        static ValueExpr placeholder( const std::string& id, const ValueExpr& max_value );
        static ValueExpr identifier( const std::string& name, const IdentifierKind& kind,
                                    const Typename type, const std::optional<size_t> size = std::nullopt );
        
        static ValueExpr unary( const std::string& op, const ValueExpr& arg, const Typename type,
                               const std::optional<size_t> size = std::nullopt );
        static ValueExpr binary( const std::string& op, const ValueExpr& left, const ValueExpr& right, const Typename type,
                                const std::optional<size_t> size = std::nullopt );
        static ValueExpr select( const ValueExpr& cond, const ValueExpr& left, const ValueExpr& right,
                                const std::optional<size_t> size = std::nullopt );
        static ValueExpr fold( const std::string& op, const ValueExpr& pack,
                              const std::optional<size_t> size = std::nullopt );
        static ValueExpr list( const std::vector<ValueExpr>& items, const Typename type );
        static ValueExpr list( const ValueExpr& item, const size_t count );
        static ValueExpr bounded( const ValueExpr& arg, const ValueExpr& lower, const ValueExpr& upper );
        static ValueExpr uniform( const ValueExpr& value, const ValueExpr& size, const size_t max_size );
        
        static ValueExpr unary( const std::string& op, ValueExpr&& arg, const Typename type,
                               const std::optional<size_t> size = std::nullopt );
        static ValueExpr binary( const std::string& op, ValueExpr&& left, ValueExpr&& right, const Typename type,
                                const std::optional<size_t> size = std::nullopt );
        static ValueExpr select( ValueExpr&& cond, ValueExpr&& left, ValueExpr&& right,
                                const std::optional<size_t> size = std::nullopt );
        static ValueExpr fold( const std::string& op, ValueExpr&& pack,
                              const std::optional<size_t> size = std::nullopt );
        static ValueExpr list( std::vector<ValueExpr>&& items, const Typename type );
        static ValueExpr bounded( ValueExpr&& arg, ValueExpr&& lower, ValueExpr&& upper );
        static ValueExpr uniform( ValueExpr&& value, ValueExpr&& size, const size_t max_size );
        
        static ValueExpr unary( const std::string& op, const ValueExpr& arg )
        {
            return unary(op, arg, arg.dtype(), arg.max_size_or_null());
        }
        
        static ValueExpr binary( const std::string& op, const ValueExpr& left, const ValueExpr& right )
        {
            auto type = left.dtype();
            auto size = left.packed() ? left.max_size() : right.packed() ? right.max_size() : (std::optional<size_t>)std::nullopt;
            return binary(op, left, right, type, size);
        }
        
        static ValueExpr unary( const std::string& op, ValueExpr&& arg )
        {
            auto type = arg.dtype();
            auto size = arg.max_size_or_null();
            return unary(op, std::forward<ValueExpr>(arg), type, size);
        }
        
        static ValueExpr binary( const std::string& op, ValueExpr&& left, ValueExpr&& right )
        {
            auto type = left.dtype();
            auto size = left.packed() ? left.max_size() : right.packed() ? right.max_size() : (std::optional<size_t>)std::nullopt;
            return binary(op, std::forward<ValueExpr>(left), std::forward<ValueExpr>(right), type, size);
        }
        
    public:
        
        ValueExpr( std::nullptr_t = nullptr )
        : _kind(Null), _dtype(Typename::Type), _index(NullIndex), _size(UnpackedSize)
        {
        }
        
        ValueExpr( const int& value )
        : _kind(Literal), _dtype(Typename::Int), _index(variant_data::index_of<int_t>()),
          _size(UnpackedSize), _data((int_t)value)
        {
        }
        
        ValueExpr( const bool& value )
        : _kind(Literal), _dtype(Typename::Bool), _index(variant_data::index_of<bool_t>()),
          _size(UnpackedSize), _data((bool_t)value)
        {
        }
        
        ValueExpr( const float& value )
        : _kind(Literal), _dtype(Typename::Real), _index(variant_data::index_of<real_t>()),
          _size(UnpackedSize), _data((real_t)value)
        {
        }
        
        ValueExpr( const double& value )
        : _kind(Literal), _dtype(Typename::Real), _index(variant_data::index_of<real_t>()),
          _size(UnpackedSize), _data((real_t)value)
        {
        }
        
        ValueExpr( const char* value )
        : _kind(Literal), _dtype(Typename::Str), _index(variant_data::index_of<str_t>()),
          _size(UnpackedSize), _data((str_t)value)
        {
        }
        
        template<typename T, typename = std::enable_if_t<is_typename<T>>>
        ValueExpr( const T& value )
        : _kind(Literal), _dtype(typename_of<T>::value), _index(variant_data::index_of<T>()),
          _size(UnpackedSize), _data(value)
        {
        }
        
        template<typename T, typename = std::enable_if_t<variant_data::contains<T>() && !is_typename<T>>>
        ValueExpr( const T& value, const Typename type, const std::optional<size_t> size = std::nullopt )
        : _kind(kind_of<T>()), _dtype(type), _index(variant_data::index_of<T>()), 
          _size(size ? (size_type)*size : UnpackedSize), _data(value)
        {
        }
        
        template<typename T, typename = std::enable_if_t<variant_data::contains<T>() && !is_typename<T>>>
        ValueExpr( T&& value, const Typename type, const std::optional<size_t> size = std::nullopt )
        : _kind(kind_of<T>()), _dtype(type), _index(variant_data::index_of<T>()), 
          _size(size ? (size_type)*size : UnpackedSize), _data(std::forward<T>(value))
        {
        }
        
        ValueExpr( const SizeAccessExpr& access );
        ValueExpr( const ShapeAccessExpr& access );
        ValueExpr( const TensorAccessExpr& access );
        
        ValueExpr( const ValueExpr& other )
        : _kind(other._kind), _dtype(other._dtype), _index(other._index), _size(other._size)
        {
            if ( other != nullptr )
            {
                _data.construct(other._data, _index);
            }
        }
        
        ValueExpr( ValueExpr&& other )
        : ValueExpr()
        {
            swap(other);
        }
        
        ValueExpr( const ValueExpr* values, const size_t size, const Typename type );
        ValueExpr( const ValueExpr* values, const size_t size, const size_t stride, const Typename type );
        
        ValueExpr& operator=( const ValueExpr& other )
        {
            if ( &other != this )
            {
                ValueExpr copy = other;
                swap(copy);
            }
            return *this;
        }
        
        ValueExpr& operator=( ValueExpr&& other )
        {
            if ( &other != this )
            {
                swap(other);
            }
            return *this;
        }
        
        ~ValueExpr()
        {
            if ( *this != nullptr )
            {
                _data.destruct(_index);
            }
        }
        
        Kind kind() const
        {
            return _kind;
        }
        
        Typename dtype() const
        {
            return _dtype;
        }
        
        bool packed() const
        {
            return _size != UnpackedSize;
        }
        
        size_t max_size() const
        {
            return _size;
        }
        
        std::optional<size_t> max_size_or_null() const
        {
            return packed() ? _size : (std::optional<size_t>)std::nullopt;
        }
        
        template<typename T, typename = std::enable_if_t<variant_data::contains<T>()>>
        bool is() const
        {
            return _index == _data.index_of<T>();
        }
        
        bool is_literal() const
        {
            return _kind == Literal;
        }
        
        bool is_int() const
        {
            return is<int_t>();
        }
        
        bool is_real() const
        {
            return is<real_t>();
        }
        
        bool is_bool() const
        {
            return is<bool_t>();
        }
        
        bool is_str() const
        {
            return is<str_t>();
        }
        
        bool is_placeholder() const
        {
            return _kind == Placeholder;
        }
        
        bool is_identifier() const
        {
            return _kind == Identifier;
        }
        
        bool is_reference() const
        {
            return _kind == Reference;
        }
        
        bool is_size_access() const
        {
            return _kind == SizeAccess;
        }
        
        bool is_shape_access() const
        {
            return _kind == ShapeAccess;
        }
        
        bool is_tensor_access() const
        {
            return _kind == TensorAccess;
        }
        
        bool is_unary() const
        {
            return _kind == Unary;
        }
        
        bool is_binary() const
        {
            return _kind == Binary;
        }
        
        bool is_select() const
        {
            return _kind == Select;
        }
        
        bool is_fold() const
        {
            return _kind == Fold;
        }
        
        bool is_list() const
        {
            return _kind == List;
        }
        
        bool is_cast() const
        {
            return _kind == Cast;
        }
        
        bool is_bounded() const
        {
            return _kind == Bounded;
        }
        
        bool is_concat() const
        {
            return _kind == Concat;
        }
        
        bool is_slice() const
        {
            return _kind == Slice;
        }
        
        bool is_subscript() const
        {
            return _kind == Subscript;
        }
        
        bool is_uniform() const
        {
            return _kind == Uniform;
        }
        
        bool is_range() const
        {
            return _kind == Range;
        }
        
        template<typename T, typename = std::enable_if_t<variant_data::contains<T>()>>
        const T& as() const
        {
            if ( !is<T>() )
            {
                throw std::bad_cast();
            }
            return _data.as<T>();
        }
        
        template<typename T, typename = std::enable_if_t<variant_data::contains<T>()>>
        T& as()
        {
            if ( !is<T>() )
            {
                throw std::bad_cast();
            }
            return _data.as<T>();
        }
        
        const int_t& as_int() const
        {
            return as<int_t>();
        }
        
        int_t& as_int()
        {
            return as<int_t>();
        }
        
        const real_t& as_real() const
        {
            return as<real_t>();
        }
        
        real_t& as_real()
        {
            return as<real_t>();
        }
        
        const bool_t& as_bool() const
        {
            return as<bool_t>();
        }
        
        bool_t& as_bool()
        {
            return as<bool_t>();
        }
        
        const str_t& as_str() const
        {
            return as<str_t>();
        }
        
        str_t& as_str()
        {
            return as<str_t>();
        }
        
        const PlaceholderExpr& as_placeholder() const
        {
            return as<PlaceholderExpr>();
        }
        
        PlaceholderExpr& as_placeholder()
        {
            return as<PlaceholderExpr>();
        }
        
        const IdentifierExpr& as_identifier() const
        {
            return as<IdentifierExpr>();
        }
        
        IdentifierExpr& as_identifier()
        {
            return as<IdentifierExpr>();
        }
        
        const ReferenceExpr& as_reference() const
        {
            return as<ReferenceExpr>();
        }
        
        ReferenceExpr& as_reference()
        {
            return as<ReferenceExpr>();
        }
        
        const SizeAccessExpr& as_size_access() const
        {
            return as<SizeAccessExpr>();
        }
        
        SizeAccessExpr& as_size_access()
        {
            return as<SizeAccessExpr>();
        }
        
        const ShapeAccessExpr& as_shape_access() const
        {
            return as<ShapeAccessExpr>();
        }
        
        ShapeAccessExpr& as_shape_access()
        {
            return as<ShapeAccessExpr>();
        }
        
        const TensorAccessExpr& as_tensor_access() const
        {
            return as<TensorAccessExpr>();
        }
        
        TensorAccessExpr& as_tensor_access()
        {
            return as<TensorAccessExpr>();
        }
        
        const UnaryExpr& as_unary() const
        {
            return as<UnaryExpr>();
        }
        
        UnaryExpr& as_unary()
        {
            return as<UnaryExpr>();
        }
        
        const BinaryExpr& as_binary() const
        {
            return as<BinaryExpr>();
        }
        
        BinaryExpr& as_binary()
        {
            return as<BinaryExpr>();
        }
        
        const SelectExpr& as_select() const
        {
            return as<SelectExpr>();
        }
        
        SelectExpr& as_select()
        {
            return as<SelectExpr>();
        }
        
        const FoldExpr& as_fold() const
        {
            return as<FoldExpr>();
        }
        
        FoldExpr& as_fold()
        {
            return as<FoldExpr>();
        }
        
        const ListExpr& as_list() const
        {
            return as<ListExpr>();
        }
        
        ListExpr& as_list()
        {
            return as<ListExpr>();
        }
        
        const CastExpr& as_cast() const
        {
            return as<CastExpr>();
        }
        
        CastExpr& as_cast()
        {
            return as<CastExpr>();
        }
        
        const BoundedExpr& as_bounded() const
        {
            return as<BoundedExpr>();
        }
        
        BoundedExpr& as_bounded()
        {
            return as<BoundedExpr>();
        }
        
        const ConcatExpr& as_concat() const
        {
            return as<ConcatExpr>();
        }
        
        ConcatExpr& as_concat()
        {
            return as<ConcatExpr>();
        }
        
        const SliceExpr& as_slice() const
        {
            return as<SliceExpr>();
        }
        
        SliceExpr& as_slice()
        {
            return as<SliceExpr>();
        }
        
        const SubscriptExpr& as_subscript() const
        {
            return as<SubscriptExpr>();
        }
        
        SubscriptExpr& as_subscript()
        {
            return as<SubscriptExpr>();
        }
        
        const UniformExpr& as_uniform() const
        {
            return as<UniformExpr>();
        }
        
        UniformExpr& as_uniform()
        {
            return as<UniformExpr>();
        }
        
        const RangeExpr& as_range() const
        {
            return as<RangeExpr>();
        }
        
        RangeExpr& as_range()
        {
            return as<RangeExpr>();
        }
        
        explicit operator const int_t&() const
        {
            return as_int();
        }
        
        explicit operator const real_t&() const
        {
            return as_real();
        }
        
        explicit operator const bool_t&() const
        {
            return as_bool();
        }
        
        explicit operator const str_t&() const
        {
            return as_str();
        }
        
        explicit operator int_t&()
        {
            return as_int();
        }
        
        explicit operator real_t&()
        {
            return as_real();
        }
        
        explicit operator bool_t&()
        {
            return as_bool();
        }
        
        explicit operator str_t&()
        {
            return as_str();
        }
        
        bool operator==( std::nullptr_t ) const
        {
            return _kind == Null;
        }
        
        bool operator!=( std::nullptr_t ) const
        {
            return _kind != Null;
        }
        
        bool operator==( const ValueExpr& other ) const
        {
            return _index == other._index && (_index == NullIndex || _data.equals(other._data, _index));
        }
        
        bool operator!=( const ValueExpr& other ) const
        {
            return !(*this == other);
        }
        
        template<typename T, typename = std::enable_if_t<is_typename<T>>>
        bool operator==( const T& value ) const
        {
            return equal_literal(value);
        }
        
        template<typename T, typename = std::enable_if_t<is_typename<T>>>
        bool operator!=( const T& value ) const
        {
            return !equal_literal(value);
        }
        
        template<typename T, typename = std::enable_if_t<is_arithmetic<T>>>
        bool operator<( const T& value ) const
        {
            return less_literal(value);
        }
        
        template<typename T, typename = std::enable_if_t<is_arithmetic<T>>>
        bool operator>( const T& value ) const
        {
            return !less_equal_literal(value);
        }
        
        template<typename T, typename = std::enable_if_t<is_arithmetic<T>>>
        bool operator<=( const T& value ) const
        {
            return less_equal_literal(value);
        }
        
        template<typename T, typename = std::enable_if_t<is_arithmetic<T>>>
        bool operator>=( const T& value ) const
        {
            return !less_literal(value);
        }
        
        bool operator==( const int& value ) const
        {
            return equal_literal((int_t)value);
        }
        
        bool operator!=( const int& value ) const
        {
            return !equal_literal((int_t)value);
        }
        
        bool operator==( const bool& value ) const
        {
            return equal_literal((bool_t)value);
        }
        
        bool operator!=( const bool& value ) const
        {
            return !equal_literal((bool_t)value);
        }
        
        bool operator==( const float& value ) const
        {
            return equal_literal((real_t)value);
        }
        
        bool operator!=( const float& value ) const
        {
            return !equal_literal((real_t)value);
        }
        
        bool operator==( const double& value ) const
        {
            return equal_literal((real_t)value);
        }
        
        bool operator!=( const double& value ) const
        {
            return !equal_literal((real_t)value);
        }
        
        bool operator==( const char* value ) const
        {
            return equal_literal((str_t)value);
        }
        
        bool operator!=( const char* value ) const
        {
            return !equal_literal((str_t)value);
        }
        
        bool operator<( const ValueExpr& other ) const
        {
            if ( other.is_int() )
            {
                return *this < other.as_int();
            }
            else if ( other.is_real() )
            {
                return *this < other.as_real();
            }
            else
            {
                throw std::bad_cast();
            }
        }
        
        bool operator>( const ValueExpr& other ) const
        {
            if ( other.is_int() )
            {
                return *this > other.as_int();
            }
            else if ( other.is_real() )
            {
                return *this > other.as_real();
            }
            else
            {
                throw std::bad_cast();
            }
        }
        
        bool operator<=( const ValueExpr& other ) const
        {
            if ( other.is_int() )
            {
                return *this <= other.as_int();
            }
            else if ( other.is_real() )
            {
                return *this <= other.as_real();
            }
            else
            {
                throw std::bad_cast();
            }
        }
        
        bool operator>=( const ValueExpr& other ) const
        {
            if ( other.is_int() )
            {
                return *this >= other.as_int();
            }
            else if ( other.is_real() )
            {
                return *this >= other.as_real();
            }
            else
            {
                throw std::bad_cast();
            }
        }
        
        void swap( ValueExpr& other )
        {
            std::swap(_kind, other._kind);
            std::swap(_dtype, other._dtype);
            std::swap(_index, other._index);
            std::swap(_size, other._size);
            _data.swap(other._data);
        }
        
        ValueExpr detach()
        {
            ValueExpr x;
            swap(x);
            return x;
        }
        
        template<typename C>
        void visit( const C& callback ) const
        {
            _data.visit(callback, _index);
        }
        
        const ValueExpr& operator[]( const size_t i ) const;
        
        ValueExpr at( const size_t idx ) const;
        ValueExpr at( const ValueExpr& idx ) const;
        
        template<typename T = real_t, typename = std::enable_if_t<std::is_same_v<T,real_t> || std::is_same_v<T,int_t>>>
        bool is_infinity() const
        {
            return *this == positive_infinity<T>() || *this == negative_infinity<T>();
        }
        
        bool is_positive_infinity() const;
        bool is_negative_infinity() const;
        
        bool is_dynamic() const;
        bool has_dynamic_size() const;
        ValueExpr size() const;
        
        bool is_unary( const std::string& op ) const;
        bool is_binary( const std::string& op ) const;
        bool is_fold( const std::string& op ) const;
        
    private:
        
        template<typename T, typename = std::enable_if_t<is_typename<T>>>
        bool equal_literal( const T& value ) const
        {
            return _index == _data.index_of<T>() && _data.as<T>() == value;
        }
        
        template<typename T, typename = std::enable_if_t<is_arithmetic<T>>>
        bool less_literal( const T& value ) const
        {
            if ( _index != _data.index_of<T>() )
            {
                throw std::bad_cast();
            }
            return _data.as<T>() < value;
        }
        
        template<typename T, typename = std::enable_if_t<is_arithmetic<T>>>
        bool less_equal_literal( const T& value ) const
        {
            if ( _index != _data.index_of<T>() )
            {
                throw std::bad_cast();
            }
            return _data.as<T>() <= value;
        }
        
    private:
        
        Kind _kind;
        Typename _dtype;
        index_type _index;
        size_type _size;
        variant_data _data;
    };
    

    struct ValueExpr::PlaceholderExpr
    {
        std::string id;
        ValueExpr max_value;
        
        bool operator==( const PlaceholderExpr& x ) const { return id == x.id; };
        bool operator!=( const PlaceholderExpr& x ) const { return !(*this == x); };
    };
    
    struct ValueExpr::IdentifierExpr
    {
        std::string name;
        IdentifierKind kind;
        
        bool operator==( const IdentifierExpr& x ) const { return name == x.name && kind == x.kind; };
        bool operator!=( const IdentifierExpr& x ) const { return !(*this == x); };
    };

    struct ValueExpr::ReferenceExpr
    {
        std::string name;
        const ValueExpr* target;
        
        bool operator==( const ReferenceExpr& x ) const { return name == x.name; };
        bool operator!=( const ReferenceExpr& x ) const { return !(*this == x); };
    };

    struct ValueExpr::SizeAccessExpr
    {
        TensorRef pack;
        
        bool operator==( const SizeAccessExpr& x ) const { return pack == x.pack; };
        bool operator!=( const SizeAccessExpr& x ) const { return !(*this == x); };
    };

    struct ValueExpr::ShapeAccessExpr
    {
        TensorRef tensor;
        ValueExpr dim;
        ValueExpr item;
        
        bool operator==( const ShapeAccessExpr& x ) const { return tensor == x.tensor && dim == x.dim && item == x.item; };
        bool operator!=( const ShapeAccessExpr& x ) const { return !(*this == x); };
    };

    struct ValueExpr::TensorAccessExpr
    {
        TensorRef tensor;
        std::vector<ValueExpr> indices;
        ValueExpr item;
        
        bool operator==( const TensorAccessExpr& x ) const { return tensor == x.tensor && indices == x.indices && item == x.item; };
        bool operator!=( const TensorAccessExpr& x ) const { return !(*this == x); };
    };
    
    struct ValueExpr::UnaryExpr
    {
        std::string op;
        ValueExpr arg;
        
        bool operator==( const UnaryExpr& x ) const { return op == x.op && arg == x.arg; };
        bool operator!=( const UnaryExpr& x ) const { return !(*this == x); };
    };
    
    struct ValueExpr::BinaryExpr
    {
        std::string op;
        ValueExpr left;
        ValueExpr right;
        
        bool operator==( const BinaryExpr& x ) const { return op == x.op && left == x.left && right == x.right; };
        bool operator!=( const BinaryExpr& x ) const { return !(*this == x); };
    };
    
    struct ValueExpr::SelectExpr
    {
        ValueExpr cond;
        ValueExpr left;
        ValueExpr right;
        
        bool operator==( const SelectExpr& x ) const { return cond == x.cond && left == x.left && right == x.right; };
        bool operator!=( const SelectExpr& x ) const { return !(*this == x); };
    };
    
    struct ValueExpr::FoldExpr
    {
        std::string op;
        ValueExpr pack;
        bool accumulate;
        
        bool operator==( const FoldExpr& x ) const { return op == x.op && pack == x.pack && accumulate == x.accumulate; };
        bool operator!=( const FoldExpr& x ) const { return !(*this == x); };
    };

    struct ValueExpr::ListExpr : std::vector<ValueExpr>
    {
        bool operator==( const ListExpr& x ) const { return (std::vector<ValueExpr>)(*this) == (std::vector<ValueExpr>)x; }
        bool operator!=( const ListExpr& x ) const { return !(*this == x); };
    };

    struct ValueExpr::CastExpr
    {
        Typename dtype;
        ValueExpr arg;
        
        bool operator==( const CastExpr& x ) const { return dtype == x.dtype && arg == x.arg; };
        bool operator!=( const CastExpr& x ) const { return !(*this == x); };
    };

    struct ValueExpr::BoundedExpr
    {
        ValueExpr arg;
        ValueExpr lower;
        ValueExpr upper;
        
        bool operator==( const BoundedExpr& x ) const { return arg == x.arg && lower == x.lower && upper == x.upper; };
        bool operator!=( const BoundedExpr& x ) const { return !(*this == x); };
    };

    struct ValueExpr::ConcatExpr
    {
        std::vector<ValueExpr> items;
        
        bool operator==( const ConcatExpr& x ) const { return items == x.items; };
        bool operator!=( const ConcatExpr& x ) const { return !(*this == x); };
    };

    struct ValueExpr::SliceExpr
    {
        ValueExpr pack;
        ValueExpr first;
        ValueExpr last;
        ValueExpr stride = 1;
        
        bool operator==( const SliceExpr& x ) const { return pack == x.pack && first == x.first && last == x.last && stride == x.stride; };
        bool operator!=( const SliceExpr& x ) const { return !(*this == x); };
    };

    struct ValueExpr::SubscriptExpr
    {
        ValueExpr pack;
        ValueExpr index;
        
        bool operator==( const SubscriptExpr& x ) const { return pack == x.pack && index == x.index; };
        bool operator!=( const SubscriptExpr& x ) const { return !(*this == x); };
    };

    struct ValueExpr::UniformExpr
    {
        ValueExpr value;
        ValueExpr size;
        
        bool operator==( const UniformExpr& x ) const { return value == x.value && size == x.size; };
        bool operator!=( const UniformExpr& x ) const { return !(*this == x); };
    };

    struct ValueExpr::RangeExpr
    {
        ValueExpr first;
        ValueExpr last;
        ValueExpr stride = 1;
        
        bool operator==( const RangeExpr& x ) const { return first == x.first && last == x.last && stride == x.stride; };
        bool operator!=( const RangeExpr& x ) const { return !(*this == x); };
    };


    inline ValueExpr::ValueExpr( const ValueExpr* values, const size_t size, const Typename type )
    : _kind(List), _dtype(type), _index(variant_data::index_of<ListExpr>()), _size((size_type)size)
    {
        _data = ListExpr{ std::vector<ValueExpr>(values, values + size) };
    }

    inline ValueExpr::ValueExpr( const ValueExpr* values, const size_t size, const size_t stride, const Typename type )
    : _kind(List), _dtype(type), _index(variant_data::index_of<ListExpr>()), _size((size_type)size)
    {
        std::vector<ValueExpr> items(size);
        for ( size_t i = 0; i < size; ++i, values += stride )
        {
            items[i] = *values;
        }
        _data = ListExpr{ std::move(items) };
    }

    inline ValueExpr::ValueExpr( const SizeAccessExpr& access )
    : _kind(SizeAccess), _dtype(Typename::Int), _index(variant_data::index_of<SizeAccessExpr>()),
      _size(UnpackedSize), _data(access)
    {
    }

    inline ValueExpr::ValueExpr( const ShapeAccessExpr& access )
    : _kind(ShapeAccess), _dtype(Typename::Int), _index(variant_data::index_of<ShapeAccessExpr>()),
      _size(UnpackedSize), _data(access)
    {
    }
    
    inline ValueExpr::ValueExpr( const TensorAccessExpr& access )
    : _kind(TensorAccess), _dtype(access.tensor.dtype()), _index(variant_data::index_of<TensorAccessExpr>()), 
      _size(UnpackedSize), _data(access)
    {
    }


    inline ValueExpr ValueExpr::placeholder( const std::string& id, const ValueExpr& max_value )
    {
        return ValueExpr(PlaceholderExpr{ id, max_value }, Typename::Int);
    }

    inline ValueExpr ValueExpr::identifier( const std::string& name, const IdentifierKind& kind,
                                           const Typename type, const std::optional<size_t> size )
    {
        return ValueExpr(IdentifierExpr{ name, kind }, type, size);
    }

    inline ValueExpr ValueExpr::unary( const std::string& op, const ValueExpr& arg,  const Typename type, const std::optional<size_t> size )
    {
        return ValueExpr(UnaryExpr{ op, arg }, type, size);
    }

    inline ValueExpr ValueExpr::binary( const std::string& op, const ValueExpr& left, const ValueExpr& right,
                                       const Typename type, const std::optional<size_t> size )
    {
        return ValueExpr(BinaryExpr{ op, left, right }, type, size);
    }

    inline ValueExpr ValueExpr::select( const ValueExpr& cond, const ValueExpr& left, const ValueExpr& right, 
                                       const std::optional<size_t> size )
    {
        return ValueExpr(SelectExpr{ cond, left, right }, left.dtype(), size);
    }

    inline ValueExpr ValueExpr::fold( const std::string& op, const ValueExpr& pack,
                                     const std::optional<size_t> size )
    {
        return ValueExpr(FoldExpr{ op, pack, (bool)size }, pack.dtype(), size);
    }

    inline ValueExpr ValueExpr::list( const std::vector<ValueExpr>& items, const Typename type )
    {
        return ValueExpr(ListExpr{ items }, type, items.size());
    }

    inline ValueExpr ValueExpr::list( const ValueExpr& item, const size_t count )
    {
        return ValueExpr(ListExpr{ std::vector<ValueExpr>(count, item) }, item.dtype(), count);
    }

    inline ValueExpr ValueExpr::bounded( const ValueExpr& arg, const ValueExpr& lower, const ValueExpr& upper )
    {
        return ValueExpr(BoundedExpr{ arg, lower, upper }, arg.dtype());
    }

    inline ValueExpr ValueExpr::uniform( const ValueExpr& value, const ValueExpr& size, const size_t max_size )
    {
        return ValueExpr(UniformExpr{ value, size }, value.dtype(), max_size);
    }


    inline ValueExpr ValueExpr::unary( const std::string& op, ValueExpr&& arg,
                                      const Typename type, const std::optional<size_t> size )
    {
        return ValueExpr(UnaryExpr{ op, std::forward<ValueExpr>(arg) }, type, size);
    }

    inline ValueExpr ValueExpr::binary( const std::string& op, ValueExpr&& left, ValueExpr&& right,
                                       const Typename type, const std::optional<size_t> size )
    {
        return ValueExpr(BinaryExpr{ op, std::forward<ValueExpr>(left), std::forward<ValueExpr>(right) }, type, size);
    }

    inline ValueExpr ValueExpr::select( ValueExpr&& cond, ValueExpr&& left, ValueExpr&& right, 
                                       const std::optional<size_t> size )
    {
        auto type = left.dtype();
        return ValueExpr(SelectExpr{ std::forward<ValueExpr>(cond), std::forward<ValueExpr>(left), std::forward<ValueExpr>(right) }, type, size);
    }

    inline ValueExpr ValueExpr::fold( const std::string& op, ValueExpr&& pack,
                                     const std::optional<size_t> size )
    {
        auto type = pack.dtype();
        return ValueExpr(FoldExpr{ op, std::forward<ValueExpr>(pack), (bool)size }, type, size);
    }

    inline ValueExpr ValueExpr::list( std::vector<ValueExpr>&& items, const Typename type )
    {
        auto size = items.size();
        return ValueExpr(ListExpr{ std::forward<std::vector<ValueExpr>>(items) }, type, size);
    }

    inline ValueExpr ValueExpr::bounded( ValueExpr&& arg, ValueExpr&& lower, ValueExpr&& upper )
    {
        auto type = arg.dtype();
        return ValueExpr(BoundedExpr{ std::forward<ValueExpr>(arg), std::forward<ValueExpr>(lower), std::forward<ValueExpr>(upper) }, type);
    }

    inline ValueExpr ValueExpr::uniform( ValueExpr&& value, ValueExpr&& size, const size_t max_size )
    {
        auto type = value.dtype();
        return ValueExpr(UniformExpr{ std::forward<ValueExpr>(value), std::forward<ValueExpr>(size) }, type, max_size);
    }

    template<>
    inline const ValueExpr& ValueExpr::positive_infinity<int_t>()
    {
        static const ValueExpr inf(UnaryExpr{"int", positive_infinity()}, Typename::Int);
        return inf;
    }

    template<>
    inline const ValueExpr& ValueExpr::negative_infinity<int_t>()
    {
        static const ValueExpr inf(UnaryExpr{"int", negative_infinity()}, Typename::Int);
        return inf;
    }

    inline bool ValueExpr::is_positive_infinity() const
    {
        return *this == positive_infinity<real_t>() || *this == positive_infinity<int_t>();
    }

    inline bool ValueExpr::is_negative_infinity() const
    {
        return *this == negative_infinity<real_t>() || *this == negative_infinity<int_t>();
    }

    inline bool ValueExpr::is_unary( const std::string& op ) const
    {
        return is_unary() && as_unary().op == op;
    }

    inline bool ValueExpr::is_binary( const std::string& op ) const
    {
        return is_binary() && as_binary().op == op;
    }
        
    inline bool ValueExpr::is_fold( const std::string& op ) const
    {
        return is_fold() && as_fold().op == op;
    }


    namespace detail
    {
    
        template<typename T>
        struct RangeCache
        {
            const ValueExpr& operator()( const T& value )
            {
                auto it = _cached.find(value);
                if ( it == _cached.end() )
                {
                    it = _cached.emplace(value, ValueExpr(value)).first;
                }
                return it->second;
            }
            
            std::unordered_map<T, ValueExpr> _cached;
        };
    
        template<>
        struct RangeCache<int_t>
        {
            const ValueExpr& operator()( const int_t& idx )
            {
                if ( _cached.size() <= idx )
                {
                    const size_t size = _cached.size();
                    _cached.resize(idx + 1);
                    for ( size_t i = size; i < _cached.size(); ++i )
                    {
                        _cached[i] = std::make_unique<ValueExpr>((int_t)i);
                    }
                }
                return *_cached[idx];
            }
            
            std::vector<std::unique_ptr<ValueExpr>> _cached;
        };
    
    }   // namespace detail


    inline bool ValueExpr::is_dynamic() const
    {
        switch ( kind() )
        {
            case Null:
            case Literal:
            {
                return false;
            }
            case List:
            {
                auto& items = as_list();
                return std::any_of(items.begin(), items.end(), []( const ValueExpr& x ){ return !x.is_literal(); });
            }
            case Uniform:
            {
                auto& uniform = as_uniform();
                return !uniform.size.is_literal() || !uniform.value.is_literal();
            }
            case Range:
            {
                auto& range = as_range();
                return !range.first.is_literal() || !range.last.is_literal() || !range.stride.is_literal();
            }
            default:
            {
                return true;
            }
        }
    }

    inline bool ValueExpr::has_dynamic_size() const
    {
        switch ( kind() )
        {
            case Null:
            case Literal:
            case Placeholder:
            case Identifier:
            case SizeAccess:
            case Bounded:
            case List:
            {
                return false;
            }
            case ShapeAccess:
            {
                auto& access = as_shape_access();
                return access.tensor.packed() && !access.tensor.size().is_literal();
            }
            case TensorAccess:
            {
                auto& access = as_tensor_access();
                return access.tensor.packed() && access.item == nullptr && !access.tensor.size().is_literal();
            }
            case Unary:
            {
                auto& unary = as_unary();
                return unary.arg.has_dynamic_size();
            }
            case Binary:
            {
                auto& binary = as_binary();
                return binary.left.has_dynamic_size() || binary.right.has_dynamic_size();
            }
            case Select:
            {
                auto& select = as_select();
                return select.cond.has_dynamic_size() || select.left.has_dynamic_size() || select.right.has_dynamic_size();
            }
            case Fold:
            {
                auto& fold = as_fold();
                return fold.accumulate && fold.pack.has_dynamic_size();
            }
            case Cast:
            {
                auto& cast = as_cast();
                return cast.arg.has_dynamic_size();
            }
            case Reference:
            {
                auto& ref = as_reference();
                return ref.target->has_dynamic_size();
            }
            case Concat:
            {
                auto& concat = as_concat();
                return std::any_of(concat.items.begin(), concat.items.end(), []( const ValueExpr& x ){ return x.has_dynamic_size(); });
            }
            case Slice:
            {
                auto& slice = as_slice();
                return !slice.first.is_literal() || !slice.last.is_literal() || !slice.stride.is_literal();
            }
            case Subscript:
            {
                auto& subscript = as_subscript();
                return subscript.index.has_dynamic_size();
            }
            case Uniform:
            {
                auto uniform = as_uniform();
                return !uniform.size.is_literal();
            }
            case Range:
            {
                auto range = as_range();
                return !range.first.is_literal() || !range.last.is_literal() || !range.stride.is_literal();
            }
        }

        assert(false);
        return false;
    }

    inline ValueExpr ValueExpr::size() const
    {
        switch ( kind() )
        {
            case Null:
            case Literal:
            case Placeholder:
            case Identifier:
            case SizeAccess:
            case Bounded:
            {
                return nullptr;
            }
            case List:
            {
                return ValueExpr((int_t)as_list().size());
            }
            case ShapeAccess:
            {
                auto& access = as_shape_access();
                return access.tensor.packed() ? access.tensor.size() : nullptr;
            }
            case TensorAccess:
            {
                auto& access = as_tensor_access();
                return access.tensor.packed() && access.item == nullptr ? access.tensor.size() : nullptr;
            }
            case Unary:
            {
                auto& unary = as_unary();
                return unary.arg.size();
            }
            case Binary:
            {
                auto& binary = as_binary();
                auto left_size = binary.left.size();
                return left_size != nullptr ? left_size : binary.right.size();
            }
            case Select:
            {
                auto& select = as_select();
                auto cond_size = select.cond.size();
                if ( cond_size != nullptr )
                {
                    return cond_size;
                }
                auto left_size = select.left.size();
                return left_size != nullptr ? left_size : select.right.size();
            }
            case Fold:
            {
                auto& fold = as_fold();
                return fold.accumulate ? fold.pack.size() : nullptr;
            }
            case Cast:
            {
                auto& cast = as_cast();
                return cast.arg.size();
            }
            case Reference:
            {
                auto& ref = as_reference();
                return ref.target->size();
            }
            case Concat:
            {
                auto& concat = as_concat();
                ValueExpr expr;
                int_t constant = 0;
                for ( auto& item : concat.items )
                {
                    if ( item.packed() )
                    {
                        auto size = item.size();
                        if ( size.is_literal() )
                        {
                            constant += size.as_int();
                        }
                        else
                        {
                            expr = expr == nullptr ? size : expr + size;
                        }
                    }
                    else
                    {
                        constant += 1;
                    }
                }
                return constant ? expr + constant : expr;
            }
            case Slice:
            {
                auto& slice = as_slice();
                return ceil_div(slice.last - slice.first, slice.stride);
            }
            case Subscript:
            {
                auto& subscript = as_subscript();
                return subscript.index.size();
            }
            case Uniform:
            {
                auto uniform = as_uniform();
                return uniform.size;
            }
            case Range:
            {
                auto range = as_range();
                return ceil_div(range.last - range.first, range.stride);
            }
        }
        
        assert(false);
        return nullptr;
    }

    inline ValueExpr ValueExpr::at( const size_t idx ) const
    {
        switch ( kind() )
        {
            case Null:
            case Literal:
            case Placeholder:
            case Identifier:
            case SizeAccess:
            case TensorAccess:
            case Bounded:
            {
                return *this;
            }
            case List:
            {
                return as_list()[idx];
            }
            case ShapeAccess:
            {
                auto& access = as_shape_access();
                if ( access.tensor.packed() )
                {
                    return ValueExpr(ShapeAccessExpr{ (Tensor*)&access.tensor[idx], access.dim }, dtype());
                }
                else
                {
                    return *this;
                }
            }
            case Unary:
            {
                auto& unary = as_unary();
                return ValueExpr(UnaryExpr{ unary.op, unary.arg.at(idx) }, dtype());
            }
            case Binary:
            {
                auto& binary = as_binary();
                return ValueExpr(BinaryExpr{ binary.op, binary.left.at(idx), binary.right.at(idx) }, dtype());
            }
            case Select:
            {
                auto& select = as_select();
                return ValueExpr(SelectExpr{ select.cond.at(idx), select.left.at(idx), select.right.at(idx) }, dtype());
            }
            case Fold:
            {
                auto& fold = as_fold();
                if ( fold.accumulate )
                {
                    auto pack = ValueExpr(SliceExpr{ fold.pack, ValueExpr((int_t)0), ValueExpr((int_t)idx+1) }, dtype(), idx + 1);
                    return ValueExpr(FoldExpr{ fold.op, pack, false }, dtype());
                }
                else
                {
                    return *this;
                }
            }
            case Cast:
            {
                auto& cast = as_cast();
                return ValueExpr(CastExpr{ cast.dtype, cast.arg.at(idx) }, dtype());
            }
            case Reference:
            case Concat:
            {
                return ValueExpr(SubscriptExpr{ *this, ValueExpr((int_t)idx) }, dtype());
            }
            case Slice:
            {
                auto& slice = as_slice();
                return slice.pack.at(slice.first + (int_t)idx * slice.stride);
            }
            case Subscript:
            {
                auto& subscript = as_subscript();
                return subscript.pack.at(subscript.index.at(idx));
            }
            case Uniform:
            {
                auto uniform = as_uniform();
                return uniform.value;
            }
            case Range:
            {
                auto range = as_range();
                auto i = dtype() == Typename::Real ? ValueExpr((real_t)idx) : ValueExpr((int_t)idx);
                return range.first + i * range.stride;
            }
        }

        assert(false);
        return *this;
    }

    inline ValueExpr ValueExpr::at( const ValueExpr& idx ) const
    {
        if ( idx.is_literal() )
        {
            return at(idx.as_int());
        }
        switch ( kind() )
        {
            case Null:
            case Literal:
            case Placeholder:
            case Identifier:
            case SizeAccess:
            case TensorAccess:
            case Bounded:
            {
                return *this;
            }
            case ShapeAccess:
            {
                auto& access = as_shape_access();
                if ( access.tensor.packed() )
                {
                    return ValueExpr(ShapeAccessExpr{ access.tensor, access.dim, idx }, dtype());
                }
                else
                {
                    return *this;
                }
            }
            case Unary:
            {
                auto& unary = as_unary();
                return ValueExpr(UnaryExpr{ unary.op, unary.arg.at(idx) }, dtype());
            }
            case Binary:
            {
                auto& binary = as_binary();
                return ValueExpr(BinaryExpr{ binary.op, binary.left.at(idx), binary.right.at(idx) }, dtype());
            }
            case Select:
            {
                auto& select = as_select();
                return ValueExpr(SelectExpr{ select.cond.at(idx), select.left.at(idx), select.right.at(idx) }, dtype());
            }
            case Fold:
            {
                auto& fold = as_fold();
                if ( fold.accumulate )
                {
                    auto first = ValueExpr((int_t)0);
                    auto last = idx.is_literal() ? ValueExpr(idx.as_int() + (int_t)1) : ValueExpr(BinaryExpr{ "+", idx, (int_t)1 }, Typename::Int);
                    auto pack = ValueExpr(SliceExpr{ fold.pack, first, last }, dtype(), fold.pack.max_size_or_null());
                    return ValueExpr(FoldExpr{ fold.op, pack, false }, dtype());
                }
                else
                {
                    return *this;
                }
            }
            case Cast:
            {
                auto& cast = as_cast();
                return ValueExpr(CastExpr{ cast.dtype, cast.arg.at(idx) }, dtype());
            }
            case Reference:
            case List:
            case Concat:
            {
                return ValueExpr(SubscriptExpr{ *this, idx }, dtype());
            }
            case Slice:
            {
                auto& slice = as_slice();
                return slice.pack.at(slice.first + idx * slice.stride);
            }
            case Subscript:
            {
                auto& subscript = as_subscript();
                return subscript.pack.at(subscript.index.at(idx));
            }
            case Uniform:
            {
                auto uniform = as_uniform();
                return uniform.value;
            }
            case Range:
            {
                auto range = as_range();
                return range.first + idx * range.stride;
            }
        }

        assert(false);
        return *this;
    }

    inline const ValueExpr& ValueExpr::operator[]( const size_t i ) const
    {
        switch ( kind() )
        {
            case List:
            {
                auto& items = as_list();
                return items[i];
            }
            case Uniform:
            {
                auto& uniform = as_uniform();
                return uniform.value;
            }
            case Range:
            {
                static detail::RangeCache<int_t> _int_cache;
                static detail::RangeCache<int_t> _real_cache;
                
                auto& range = as_range();
                if ( dtype() == Typename::Real )
                {
                    return _real_cache(range.first.as_real() + (real_t)i * range.stride.as_real());
                }
                else
                {
                    return _int_cache(range.first.as_int() + (int_t)i * range.stride.as_int());
                }
            }
            default:
            {
                throw std::invalid_argument("operator[] called on invalid ValueExpr");
            }
        }
    }
    
    
    inline std::ostream& operator<<( std::ostream& os, const ValueExpr& expr )
    {
        if ( expr == nullptr )
        {
            os << "null";
        }
        else
        {
            expr.visit([&]( auto& x )
            {
                using T = std::decay_t<decltype(x)>;
                if constexpr( std::is_same_v<T,bool_t> )
                {
                    os << std::boolalpha << x;
                }
                else if constexpr( std::is_same_v<T,real_t> )
                {
                    os << x;
                    if ( (int_t)x == x )
                    {
                        os << ".0";
                    }
                }
                else if constexpr( std::is_same_v<T,str_t> )
                {
                    os << "'" << x << "'";
                }
                else
                {
                    os << x;
                }
            });
        }
        return os;
    }
    
    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::PlaceholderExpr& placeholder )
    {
        os << placeholder.id << '|' << placeholder.max_value;
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::IdentifierExpr& iden )
    {
        os << iden.name;
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::ReferenceExpr& ref )
    {
        os << ref.name;
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::SizeAccessExpr& access )
    {
        os << access.pack.name() << ".size";
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::ShapeAccessExpr& access )
    {
        os << access.tensor.name();
        if ( access.item != nullptr )
        {
            os << '[' << access.item << ']';
        }
        os << ".shape";
        if ( access.dim != nullptr )
        {
            os << '[' << access.dim << ']';
        }
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::TensorAccessExpr& access )
    {
        os << access.tensor.name();
        if ( access.item != nullptr )
        {
            os << '[' << access.item << ']';
        }
        os << '[';
        size_t dim = 0;
        for ( auto& index : access.indices )
        {
            if ( dim++ )
            {
                os << ", ";
            }
            os << index;
        }
        os << ']';
        return os;
    }
    
    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::UnaryExpr& unary )
    {
        os << unary.op << '(' << unary.arg << ')';
        return os;
    }
    
    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::BinaryExpr& binary )
    {
        os << '(' << binary.left << ' ' << binary.op << ' ' << binary.right << ')';
        return os;
    }
    
    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::SelectExpr& select )
    {
        os << '(' << select.cond << " ? " << select.left << " : " << select.right << ')';
        return os;
    }
    
    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::FoldExpr& fold )
    {
        os << '(' << fold.pack << ' ' << fold.op << (fold.accumulate ? " ..." : " ..") << ')';
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::CastExpr& cast )
    {
        os << str(cast.dtype) << '(' << cast.arg << ')';
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::ListExpr& list )
    {
        os << '[';
        for ( size_t i = 0; i < list.size(); ++i )
        {
            if ( i )
            {
                os << ", ";
            }
            os << list[i];
        }
        os << ']';
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::BoundedExpr& bounded )
    {
        os << '|';
        os << bounded.arg;
        if ( bounded.lower != nullptr && bounded.upper != nullptr )
        {
            os << " <> " << bounded.lower << " : " << bounded.upper;
        }
        os << '|';
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::ConcatExpr& concat )
    {
        os << '[';
        for ( size_t i = 0; i < concat.items.size(); ++i )
        {
            if ( i )
            {
                os << ", ";
            }
            os << concat.items[i];
            if ( concat.items[i].packed() )
            {
                os << "..";
            }
        }
        os << ']';
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::SliceExpr& slice )
    {
        os << slice.pack << '[';
        os << slice.first << ':' << slice.last;
        if ( slice.stride != 1 )
        {
            os << ':' << slice.stride;
        }
        os << ']';
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::SubscriptExpr& subscript )
    {
        os << subscript.pack << '[' << subscript.index << ']';
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::UniformExpr& uniform )
    {
        os << '[' << uniform.value << (uniform.value.is_literal() ? " .." : "..") << '(' << uniform.size << ')' << ']';
        return os;
    }

    inline std::ostream& operator<<( std::ostream& os, const ValueExpr::RangeExpr& range )
    {
        os << '(';
        os << range.first << ':' << range.last;
        if ( range.stride != 1 )
        {
            os << ':' << range.stride;
        }
        os << ')';
        return os;
    }


    typedef ValueExpr::SizeAccessExpr SizeAccess;
    typedef ValueExpr::ShapeAccessExpr ShapeAccess;
    typedef ValueExpr::TensorAccessExpr TensorAccess;


    inline std::string str( const ValueExpr& expr )
    {
        std::stringstream ss;
        ss << expr;
        return ss.str();
    }
    
    inline void recurse( ValueExpr& expr, function_view<void(ValueExpr&)> callback )
    {
        switch ( expr.kind() )
        {
            case ValueExpr::Kind::Null:
            case ValueExpr::Kind::Literal:
            case ValueExpr::Kind::Identifier:
            case ValueExpr::Kind::SizeAccess:
            case ValueExpr::Kind::Reference:
            {
                break;
            }
            case ValueExpr::Kind::Placeholder:
            {
                auto& placeholder = expr.as_placeholder();
                callback(placeholder.max_value);
                break;
            }
            case ValueExpr::Kind::Unary:
            {
                auto& unary = expr.as_unary();
                callback(unary.arg);
                break;
            }
            case ValueExpr::Kind::Binary:
            {
                auto& binary = expr.as_binary();
                callback(binary.left);
                callback(binary.right);
                break;
            }
            case ValueExpr::Kind::Select:
            {
                auto& select = expr.as_select();
                callback(select.cond);
                callback(select.left);
                callback(select.right);
                break;
            }
            case ValueExpr::Kind::Fold:
            {
                auto& fold = expr.as_fold();
                callback(fold.pack);
                break;
            }
            case ValueExpr::Kind::List:
            {
                auto& list = expr.as_list();
                for ( auto& item : list )
                {
                    callback(item);
                }
                break;
            }
            case ValueExpr::Kind::Cast:
            {
                auto& cast = expr.as_cast();
                callback(cast.arg);
                break;
            }
            case ValueExpr::Kind::Bounded:
            {
                auto& bounded = expr.as_bounded();
                callback(bounded.arg);
                callback(bounded.lower);
                callback(bounded.upper);
                break;
            }
            case ValueExpr::Kind::ShapeAccess:
            {
                auto& access = expr.as_shape_access();
                callback(access.item);
                break;
            }
            case ValueExpr::Kind::TensorAccess:
            {
                auto& access = expr.as_tensor_access();
                callback(access.item);
                for ( auto& index : access.indices )
                {
                    callback(index);
                }
                break;
            }
            case ValueExpr::Kind::Concat:
            {
                auto& concat = expr.as_concat();
                for ( auto& item : concat.items )
                {
                    callback(item);
                }
                break;
            }
            case ValueExpr::Kind::Slice:
            {
                auto& slice = expr.as_slice();
                callback(slice.pack);
                callback(slice.first);
                callback(slice.last);
                callback(slice.stride);
                break;
            }
            case ValueExpr::Kind::Subscript:
            {
                auto& subscript = expr.as_subscript();
                callback(subscript.pack);
                callback(subscript.index);
                break;
            }
            case ValueExpr::Kind::Uniform:
            {
                auto& uniform = expr.as_uniform();
                callback(uniform.value);
                callback(uniform.size);
                break;
            }
            case ValueExpr::Kind::Range:
            {
                auto& range = expr.as_range();
                callback(range.first);
                callback(range.last);
                callback(range.stride);
                break;
            }
        }
    }

    inline void recurse( const ValueExpr& expr, function_view<void(const ValueExpr&)> callback, bool follow_references = false )
    {
        switch ( expr.kind() )
        {
            case ValueExpr::Kind::Null:
            case ValueExpr::Kind::Literal:
            case ValueExpr::Kind::Identifier:
            case ValueExpr::Kind::SizeAccess:
            {
                break;
            }
            case ValueExpr::Kind::Placeholder:
            {
                auto& placeholder = expr.as_placeholder();
                callback(placeholder.max_value);
                break;
            }
            case ValueExpr::Kind::Reference:
            {
                auto& ref = expr.as_reference();
                if ( follow_references )
                {
                    callback(*ref.target);
                }
                break;
            }
            case ValueExpr::Kind::Unary:
            {
                auto& unary = expr.as_unary();
                callback(unary.arg);
                break;
            }
            case ValueExpr::Kind::Binary:
            {
                auto& binary = expr.as_binary();
                callback(binary.left);
                callback(binary.right);
                break;
            }
            case ValueExpr::Kind::Select:
            {
                auto& select = expr.as_select();
                callback(select.cond);
                callback(select.left);
                callback(select.right);
                break;
            }
            case ValueExpr::Kind::Fold:
            {
                auto& fold = expr.as_fold();
                callback(fold.pack);
                break;
            }
            case ValueExpr::Kind::List:
            {
                auto& list = expr.as_list();
                for ( auto& item : list )
                {
                    callback(item);
                }
                break;
            }
            case ValueExpr::Kind::Cast:
            {
                auto& cast = expr.as_cast();
                callback(cast.arg);
                break;
            }
            case ValueExpr::Kind::Bounded:
            {
                auto& bounded = expr.as_bounded();
                callback(bounded.arg);
                callback(bounded.lower);
                callback(bounded.upper);
                break;
            }
            case ValueExpr::Kind::ShapeAccess:
            {
                auto& access = expr.as_shape_access();
                callback(access.item);
                break;
            }
            case ValueExpr::Kind::TensorAccess:
            {
                auto& access = expr.as_tensor_access();
                callback(access.item);
                for ( auto& index : access.indices )
                {
                    callback(index);
                }
                break;
            }
            case ValueExpr::Kind::Concat:
            {
                auto& concat = expr.as_concat();
                for ( auto& item : concat.items )
                {
                    callback(item);
                }
                break;
            }
            case ValueExpr::Kind::Slice:
            {
                auto& slice = expr.as_slice();
                callback(slice.pack);
                callback(slice.first);
                callback(slice.last);
                callback(slice.stride);
                break;
            }
            case ValueExpr::Kind::Subscript:
            {
                auto& subscript = expr.as_subscript();
                callback(subscript.pack);
                callback(subscript.index);
                break;
            }
            case ValueExpr::Kind::Uniform:
            {
                auto& uniform = expr.as_uniform();
                callback(uniform.value);
                callback(uniform.size);
                break;
            }
            case ValueExpr::Kind::Range:
            {
                auto& range = expr.as_range();
                callback(range.first);
                callback(range.last);
                callback(range.stride);
                break;
            }
        }
    }

    inline void preorder_traverse( ValueExpr& expr, function_view<void(ValueExpr&)> callback )
    {
        callback(expr);
        recurse(expr, [&callback]( ValueExpr& x ){ preorder_traverse(x, callback); });
    }

    inline void postorder_traverse( ValueExpr& expr, function_view<void(ValueExpr&)> callback )
    {
        recurse(expr, [&callback]( ValueExpr& x ){ preorder_traverse(x, callback); });
        callback(expr);
    }

    inline void preorder_traverse( const ValueExpr& expr, function_view<void(const ValueExpr&)> callback, bool follow_references = false )
    {
        callback(expr);
        recurse(expr, [&callback]( const ValueExpr& x ){ preorder_traverse(x, callback); }, follow_references);
    }

    inline void postorder_traverse( const ValueExpr& expr, function_view<void(const ValueExpr&)> callback, bool follow_references = false )
    {
        recurse(expr, [&callback]( const ValueExpr& x ){ preorder_traverse(x, callback); }, follow_references);
        callback(expr);
    }

    inline ValueExpr operator-( const ValueExpr& arg )
    {
        if ( arg.is_literal() )
        {
            return arg.dtype() == sknd::Typename::Real ? ValueExpr(-arg.as_real()) : ValueExpr(-arg.as_int());
        }
        else
        {
            return ValueExpr::unary("-", arg);
        }
    }

    inline ValueExpr operator-( ValueExpr&& arg )
    {
        return ValueExpr::unary("-", std::forward<ValueExpr>(arg));
    }

    inline ValueExpr operator+( const ValueExpr& left, const ValueExpr& right )
    {
        if ( left.is_literal() && right.is_literal() )
        {
            return left.dtype() == sknd::Typename::Real ? ValueExpr(left.as_real() + right.as_real()) : ValueExpr(left.as_int() + right.as_int());
        }
        else if ( left == (real_t)0 || left == (int_t)0 )
        {
            return right;
        }
        else if ( right == (real_t)0 || right == (int_t)0 )
        {
            return left;
        }
        else
        {
            return ValueExpr::binary("+", left, right);
        }
    }

    inline ValueExpr operator-( const ValueExpr& left, const ValueExpr& right )
    {
        if ( left.is_literal() && right.is_literal() )
        {
            return left.dtype() == sknd::Typename::Real ? ValueExpr(left.as_real() - right.as_real()) : ValueExpr(left.as_int() - right.as_int());
        }
        else if ( left == (real_t)0 || left == (int_t)0 )
        {
            return -right;
        }
        else if ( right == (real_t)0 || right == (int_t)0 )
        {
            return left;
        }
        else
        {
            return ValueExpr::binary("-", left, right);
        }
    }

    inline ValueExpr operator*( const ValueExpr& left, const ValueExpr& right )
    {
        if ( left.is_literal() && right.is_literal() )
        {
            return left.dtype() == sknd::Typename::Real ? ValueExpr(left.as_real() * right.as_real()) : ValueExpr(left.as_int() * right.as_int());
        }
        else if ( left == (real_t)1 || left == (int_t)1 )
        {
            return right;
        }
        else if ( right == (real_t)1 || right == (int_t)1 )
        {
            return left;
        }
        else if ( left == (real_t)-1 || left == (int_t)-1 )
        {
            return -right;
        }
        else if ( right == (real_t)-1 || right == (int_t)-1 )
        {
            return -left;
        }
        else
        {
            return ValueExpr::binary("*", left, right);
        }
    }

    inline ValueExpr operator/( const ValueExpr& left, const ValueExpr& right )
    {
        if ( left.is_literal() && right.is_literal() )
        {
            return left.dtype() == sknd::Typename::Real ? ValueExpr(left.as_real() / right.as_real()) : ValueExpr(left.as_int() / right.as_int());
        }
        else if ( right == (real_t)1 || right == (int_t)1 )
        {
            return left;
        }
        else if ( right == (real_t)-1 || right == (int_t)-1 )
        {
            return -left;
        }
        else
        {
            return ValueExpr::binary("/", left, right);
        }
    }

    inline ValueExpr operator%( const ValueExpr& left, const ValueExpr& right )
    {
        if ( left.is_literal() && right.is_literal() )
        {
            return left.dtype() == sknd::Typename::Real ? ValueExpr(std::fmod(left.as_real(), right.as_real())) : ValueExpr(left.as_int() % right.as_int());
        }
        else if ( right == (real_t)1 )
        {
            return (real_t)0;
        }
        else if ( right == (int_t)1 )
        {
            return (int_t)0;
        }
        else
        {
            return ValueExpr::binary("%", left, right);
        }
    }

    inline ValueExpr ceil_div( const ValueExpr& left, const ValueExpr& right )
    {
        if ( left.is_literal() && right.is_literal() )
        {
            return left.dtype() == sknd::Typename::Real ? ValueExpr(ceil_div(left.as_real(), right.as_real())) :
                                                        ValueExpr(ceil_div(left.as_int(), right.as_int()));
        }
        else if ( right == (real_t)1 || right == (int_t)1 )
        {
            return left;
        }
        else if ( right == (real_t)-1 || right == (int_t)-1 )
        {
            return -left;
        }
        else
        {
            return ValueExpr::binary("\\", left, right);
        }
    }

    inline ValueExpr pow( const ValueExpr& left, const ValueExpr& right )
    {
        if ( left.is_literal() && right.is_literal() )
        {
            return left.dtype() == sknd::Typename::Real ? ValueExpr(std::pow(left.as_real(), right.as_real())) :
                                                       ValueExpr((int_t)std::pow(left.as_int(), right.as_int()));
        }
        else if ( left == (real_t)1 || left == (int_t)1 )
        {
            return left;
        }
        else if ( right == (real_t)1 || right == (int_t)1 )
        {
            return left;
        }
        else
        {
            return ValueExpr::binary("**", left, right);
        }
    }

    inline ValueExpr operator!( const ValueExpr& arg )
    {
        if ( arg.is_literal() )
        {
            return ValueExpr(!arg.as_bool());
        }
        else
        {
            return ValueExpr::unary("!", arg);
        }
    }

    inline ValueExpr operator!( ValueExpr&& arg )
    {
        return ValueExpr::unary("!", std::forward<ValueExpr>(arg));
    }

    inline ValueExpr operator&&( const ValueExpr& left, const ValueExpr& right )
    {
        if ( left.is_literal() && right.is_literal() )
        {
            return ValueExpr(left.as_bool() && right.as_bool());
        }
        else if ( left == true )
        {
            return right;
        }
        else if ( left == false )
        {
            return left;
        }
        else if ( right == true )
        {
            return left;
        }
        else if ( right == false )
        {
            return right;
        }
        else
        {
            return ValueExpr::binary("&&", left, right);
        }
    }

    inline ValueExpr operator||( const ValueExpr& left, const ValueExpr& right )
    {
        if ( left.is_literal() && right.is_literal() )
        {
            return ValueExpr(left.as_bool() || right.as_bool());
        }
        else if ( left == true )
        {
            return left;
        }
        else if ( left == false )
        {
            return right;
        }
        else if ( right == true )
        {
            return right;
        }
        else if ( right == false )
        {
            return left;
        }
        else
        {
            return ValueExpr::binary("||", left, right);
        }
    }

}   // namespace sknd


namespace std
{

    inline string to_string( const sknd::ValueExpr& expr )
    {
        std::stringstream ss;
        ss << expr;
        return ss.str();
    }

    inline void swap( sknd::ValueExpr& a, sknd::ValueExpr& b )
    {
        a.swap(b);
    }

}   // namespace std


#endif
