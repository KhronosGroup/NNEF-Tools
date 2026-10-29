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

#ifndef _SKND_RUNTIME_H_
#define _SKND_RUNTIME_H_

#include <array>
#include <vector>
#include <string>
#include <limits>
#include "either.h"
#include "print.h"


namespace sknd
{

    template<typename ... Args>
    std::string string_format( const std::string& format, Args... args )
    {
        int size = std::snprintf( nullptr, 0, format.c_str(), args...);
        std::unique_ptr<char[]> buf( new char[size + 1] );
        std::snprintf(buf.get(), size + 1, format.c_str(), std::to_string(args).c_str()...);
        return std::string( buf.get(), buf.get() + size);
    }


    namespace rt
    {
    
        enum class Dtype { Real, Int, Bool };
    
        typedef int int_t;
        typedef bool bool_t;
        typedef float real_t;
        
        template<typename T> struct dtype_of {};
        template<> struct dtype_of<real_t> { static const Dtype value = Dtype::Real; };
        template<> struct dtype_of<bool_t> { static const Dtype value = Dtype::Bool; };
        template<> struct dtype_of<int_t> { static const Dtype value = Dtype::Int; };
    
        template<typename T>
        inline constexpr bool is_typename = std::is_same_v<T,int_t> || std::is_same_v<T,real_t> || std::is_same_v<T,bool_t>;
    
    
        inline int_t ceil_div( const int_t x, const int_t y )
        {
            auto z = std::abs(y) - 1;
            return x < 0 ? (x - z) / y : (x + z) / y;
        }

        inline real_t ceil_div( const real_t x, const real_t y )
        {
            return std::ceil(x / y);
        }

        inline int_t sign( const int_t x )
        {
            return x < 0 ? -1 : x > 0 ? 1 : 0;
        }

        inline real_t sign( const real_t x )
        {
            return x < 0.f ? -1.f : x > 0.f ? 1.f : 0.f;
        }

        inline real_t frac( const real_t x )
        {
            real_t integral;
            return std::modf(x, &integral);
        }
    
        
        template<typename T>
        struct ceil_divides
        {
            T operator()( const T& a, const T& b ) const
            {
                return ceil_div(a, b);
            }
        };
    
        template<typename T>
        struct minimize
        {
            T operator()( const T& a, const T& b ) const
            {
                return std::min(a, b);
            }
        };
        
        template<typename T>
        struct maximize
        {
            T operator()( const T& a, const T& b ) const
            {
                return std::max(a, b);
            }
        };
    
        template<typename R, typename T>
        struct cast
        {
            R operator()( const T& x ) const
            {
                return (R)x;
            }
        };
    
        template<typename T> using to_int = cast<int_t,T>;
        template<typename T> using to_real = cast<real_t,T>;
        template<typename T> using to_bool = cast<bool_t,T>;
    
    
        template<typename P> class ShapePack;
        template<typename T> class ValuePack;
        
        
        struct TensorView
        {
            virtual Dtype dtype() const = 0;
            virtual size_t rank() const = 0;
            virtual const int* shape() const = 0;
            
            virtual void reshape( const size_t rank, const long* shape ) = 0;
            
            virtual const char* bytes() const = 0;
            virtual char* bytes() = 0;
        };
        
    
        struct TensorPackView
        {
            virtual Dtype dtype() const = 0;
            virtual size_t rank() const = 0;
            
            virtual int size() const = 0;
            virtual const TensorView& operator[]( const size_t i ) const = 0;
            virtual TensorView& operator[]( const size_t i ) = 0;
            
            virtual void resize( const int size ) = 0;
        };
    
    
        typedef Either<TensorView*,TensorPackView*> TensorRef;
    
        
        template<size_t N, typename T>
        class Tensor final : public TensorView
        {
            static const size_t MaxTensorSize = 1 << 30;
            
            typedef std::conditional_t<std::is_same_v<T,bool>, char, T> storage_type;
            
        public:
            
            typedef T value_type;
            typedef int size_type;
            
        public:
            
            Tensor()
            {
                _max_shape.fill(-1);
                _shape.fill(-1);
            }
            
            template<typename ...Ts, typename = std::enable_if_t<sizeof...(Ts) == N && (std::is_same_v<Ts, size_type> && ...)>>
            Tensor( Ts... shape )
            : _max_shape({ shape... }), _shape(_max_shape)
            {
                bool preallocate = std::all_of(_max_shape.begin(), _max_shape.end(), []( const size_type& x ){ return x != -1; });
                if ( preallocate )
                {
                    const size_t n = max_volume();
                    if ( n > MaxTensorSize )
                    {
                        throw std::invalid_argument("trying to allocate tensor larger than 1GB");
                    }
                    _data.resize(n);
                }
            }
            
            Tensor( const Tensor& other )
            : _max_shape(other._max_shape), _shape(other._shape), _data(other._data)
            {
            }
            
            Tensor& operator=( const Tensor& other )
            {
                if ( _shape != other._shape )
                {
                    throw std::invalid_argument("invalid tensor assignment with shape " + str(other._shape) + " while tensor shape is " + str(_shape));
                }
                _max_shape = other._max_shape;
                _data = other._data;
                return *this;
            }
            
            template<typename ...Ts, typename = std::enable_if_t<sizeof...(Ts) + 1 == N && (std::is_same_v<Ts, size_type> && ...)>>
            T& operator()( size_type idx, Ts ...indices )
            {
                if constexpr(sizeof...(indices) == 0)
                {
                    return (T&)_data[idx];
                }
                else
                {
                    return (T&)_data[offset(idx, indices...)];
                }
            }
            
            template<typename ...Ts, typename = std::enable_if_t<sizeof...(Ts) + 1 == N && (std::is_same_v<Ts, size_type> && ...)>>
            const T& operator()( size_type idx, Ts ...indices ) const
            {
                if constexpr(sizeof...(indices) == 0)
                {
                    return (const T&)_data[idx];
                }
                else
                {
                    return (const T&)_data[offset(idx, indices...)];
                }
            }
            
            Dtype dtype() const
            {
                return dtype_of<T>::value;
            }
            
            size_t rank() const
            {
                return _shape.size();
            }
            
            size_t volume() const
            {
                size_t prod = 1;
                for ( size_t i = 0; i < _shape.size(); ++i )
                {
                    prod *= _shape[i];
                }
                return prod;
            }
            
            size_t max_volume() const
            {
                size_t prod = 1;
                for ( size_t i = 0; i < _max_shape.size(); ++i )
                {
                    prod *= _max_shape[i];
                }
                return prod;
            }
            
            const size_type* max_shape() const
            {
                return _max_shape.data();
            }
            
            const size_type* shape() const
            {
                return _shape.data();
            }
            
            const size_type max_shape( const size_t i ) const
            {
                return _max_shape[i];
            }
            
            const size_type shape( const size_t i ) const
            {
                return _shape[i];
            }
            
            const T* data() const
            {
                return (const T*)_data.data();
            }
            
            T* data()
            {
                return (T*)_data.data();
            }
            
            const char* bytes() const
            {
                return (const char*)_data.data();
            }
            
            char* bytes()
            {
                return (char*)_data.data();
            }
            
            Tensor& operator=( const T& value )
            {
                std::fill_n((T*)_data.data(), volume(), value);
                return *this;
            }
            
            Tensor& operator=( std::initializer_list<T> values )
            {
                std::copy(values.begin(), values.end(), (T*)_data.data());
                return *this;
            }
            
            Tensor& operator=( const T* values )
            {
                std::copy_n(values, volume(), (T*)_data.data());
                return *this;
            }
            
            void swap( Tensor& other )
            {
                std::swap(_max_shape, other._max_shape);
                std::swap(_shape, other._shape);
                std::swap(_data, other._data);
            }
            
            template<typename ...Ts, typename = std::enable_if_t<sizeof...(Ts) == N && (std::is_same_v<Ts, size_type> && ...)>>
            void reshape( Ts... shape )
            {
                _shape = { shape... };
                if ( !check_reshape() )
                {
                    throw std::invalid_argument("invalid tensor reshaping to shape " + str(_shape) + " while allocated max shape is " + str(_max_shape));
                }
                _data.resize(volume());
            }
            
            template<typename S>
            void reshape( const size_t rank, const S* shape )
            {
                if ( rank != _shape.size() )
                {
                    throw std::invalid_argument("invalid tensor reshaping to rank " + std::to_string(rank) + " while allocated rank is " + std::to_string(_shape.size()));
                }
                std::copy_n(shape, rank, _shape.data());
                if ( !check_reshape() )
                {
                    throw std::invalid_argument("invalid tensor reshaping to shape " + str(_shape) + " while allocated max shape is " + str(_max_shape));
                }
            }
            
            void reshape( const size_t rank, const long* shape )
            {
                reshape<long>(rank, shape);
            }
            
        private:
            
            template<typename ...Ts>
            size_t offset( size_t init, size_type idx, Ts ...indices ) const
            {
                size_t offs = init * _shape[N - 1 - sizeof...(indices)] + idx;
                if constexpr(sizeof...(indices) > 0 )
                {
                    return offset(offs, indices...);
                }
                else
                {
                    return offs;
                }
            }
            
            bool check_reshape() const
            {
                for ( size_t i = 0; i < _shape.size(); ++i )
                {
                    if ( _max_shape[i] != -1 )
                    {
                        if ( _shape[i] < 0 || _shape[i] > _max_shape[i] )
                        {
                            return false;
                        }
                    }
                }
                return true;
            }
            
        private:
            
            std::array<size_type,N> _max_shape;
            std::array<size_type,N> _shape;
            std::vector<storage_type> _data;
        };
        
        
        template<typename T>
        class Tensor<0,T> final : public TensorView
        {
        public:
            
            typedef T value_type;
            typedef int size_type;
            
        public:
            
            Tensor( const T& value = T() )
            : _data(value)
            {
            }
            
            Tensor( const Tensor& other )
            : _data(other._data)
            {
            }
            
            Tensor& operator=( const Tensor& other )
            {
                _data = other._data;
                return *this;
            }
            
            T& operator()()
            {
                return _data;
            }
            
            const T& operator()() const
            {
                return _data;
            }
            
            Dtype dtype() const
            {
                return dtype_of<T>::value;
            }
            
            size_t rank() const
            {
                return 0;
            }
            
            size_t volume() const
            {
                return 1;
            }
            
            size_t max_volume() const
            {
                return 1;
            }
            
            const size_type* max_shape() const
            {
                return nullptr;
            }
            
            const size_type* shape() const
            {
                return nullptr;
            }
            
            const size_type max_shape( const size_t i ) const
            {
                return 0;
            }
            
            const size_type shape( const size_t i ) const
            {
                return 0;
            }
            
            const T* data() const
            {
                return &_data;
            }
            
            T* data()
            {
                return &_data;
            }
            
            const char* bytes() const
            {
                return (const char*)&_data;
            }
            
            char* bytes()
            {
                return (char*)&_data;
            }
            
            Tensor& operator=( const T& value )
            {
                _data = value;
                return *this;
            }
            
            Tensor& operator=( std::initializer_list<T> values )
            {
                _data = *values.begin();
                return *this;
            }
            
            Tensor& operator=( const T* values )
            {
                _data = *values;
                return *this;
            }
            
            void swap( Tensor& other )
            {
                std::swap(_data, other._data);
            }
            
            void reshape()
            {
            }
            
            void reshape( const size_t rank, const long* shape )
            {
                if ( rank != 0 )
                {
                    throw std::invalid_argument("invalid tensor reshaping to rank " + std::to_string(rank) + " while allocated rank is 0");
                }
            }
            
        private:
            
            T _data;
        };
        
        
        template<size_t N, typename T>
        class TensorPack final : public TensorPackView
        {
        public:
            
            typedef T value_type;
            typedef Tensor<N,T> tensor_type;
            typedef typename tensor_type::size_type size_type;
            typedef ValuePack<size_type> size_pack_type;
            typedef ShapePack<TensorPack<N,T>> shape_pack_type;
            
            template<typename S>
            static constexpr bool is_size_arg = std::is_same_v<S,size_type> || std::is_same_v<S,size_pack_type>;
            
        public:
            
            TensorPack( const size_type max_size )
            : _max_size(max_size)
            {
                _max_shape.fill(-1);
                if ( _max_size != -1 )
                {
                    _items.resize(_max_size);
                }
            }
            
            template<typename ...Ts, typename = std::enable_if_t<sizeof...(Ts) == N && (std::is_same_v<Ts, size_type> && ...)>>
            TensorPack( const size_type max_size, Ts... shape )
            : _max_shape({ shape... }), _max_size(max_size)
            {
                if ( _max_size != -1 )
                {
                    _items.resize(_max_size, tensor_type(shape...));
                }
            }
            
            TensorPack& operator=( const T& value )
            {
                for ( auto& item : _items )
                {
                    *item = value;
                }
                return *this;
            }
            
            TensorPack& operator=( std::initializer_list<T> values )
            {
                auto itt = values.begin();
                for ( auto it = _items.begin(); it != _items.end(); ++it )
                {
                    *it = *itt++;
                }
                return *this;
            }
            
            TensorPack& operator=( const T* values )
            {
                for ( size_t i = 0; i < _items.size(); ++i )
                {
                    _items[i] = values[i];
                }
                return *this;
            }
            
            Dtype dtype() const
            {
                return dtype_of<T>::value;
            }
            
            size_t rank() const
            {
                return N;
            }
            
            size_type size() const
            {
                return (size_type)_items.size();
            }
            
            size_type max_size() const
            {
                return _max_size;
            }
            
            tensor_type& operator[]( const size_t i )
            {
                return _items[i];
            }
            
            const tensor_type& operator[]( const size_t i ) const
            {
                return _items[i];
            }
            
            template<typename ...Ts, typename = std::enable_if_t<sizeof...(Ts) == N && (is_size_arg<Ts> && ...)>>
            void reshape( Ts... shape )
            {
                for ( size_t i = 0; i < _items.size(); ++i )
                {
                    _items[i].reshape(get_shape_item(shape, i)...);
                }
            }
            
            void resize( const size_type size )
            {
                if ( max_size() != -1 && size > max_size() )
                {
                    throw std::invalid_argument("invalid tensor pack resizing to size " + std::to_string(size) + " while allocated size is " + std::to_string(max_size()));
                }
                _items.resize(size, std::make_from_tuple<tensor_type>(_max_shape));
            }
            
            void append( const tensor_type& item )
            {
                _items.push_back(item);
            }
            
            void append( tensor_type&& item )
            {
                _items.push_back(std::forward<tensor_type>(item));
            }
            
            shape_pack_type shapes( const size_t i ) const
            {
                return shape_pack_type(*this, i);
            }
            
            size_type shape( const size_t i ) const
            {
                return largest_shape(i);
            }
            
        private:
            
            static size_type get_shape_item( const size_type& value, const size_t idx )
            {
                return value;
            }
            
            static size_type get_shape_item( const size_pack_type& value, const size_t idx )
            {
                return value[idx];
            }
            
            size_type largest_shape( const size_t dim ) const
            {
                size_type largest = _items.front().shape(dim);
                for ( size_t i = 1; i < _items.size(); ++i )
                {
                    auto shape = _items[i].shape(dim);
                    if ( shape > largest )
                    {
                        largest = shape;
                    }
                }
                return largest;
            }
            
        private:
            
            std::vector<tensor_type> _items;
            std::array<size_type,N> _max_shape;
            size_type _max_size;
        };
    
    
        template<size_t N, typename T>
        class TensorList final : public TensorPackView
        {
        public:
            
            typedef T value_type;
            typedef Tensor<N,T> tensor_type;
            typedef typename tensor_type::size_type size_type;
            typedef ValuePack<size_type> size_pack_type;
            typedef ShapePack<TensorList<N,T>> shape_pack_type;
            
            template<typename S>
            static constexpr bool is_size_arg = std::is_same_v<S,size_type> || std::is_same_v<S,size_pack_type>;
            
        public:
            
            TensorList( size_t size ) : _items(size)
            {
            }
            
            template<typename ...Items>
            TensorList( Items& ...items ) : _items({ items... })
            {
            }
            
            template<typename ...Items>
            void populate( Items& ...items )
            {
                if ( sizeof...(items) != _items.size() )
                {
                    throw std::invalid_argument("invalid tensor list population to size " + std::to_string(sizeof...(items)) + " while fixed size is " + std::to_string(_items.size()));
                }
                _items = { &items... };
            }
            
            Dtype dtype() const
            {
                return dtype_of<T>::value;
            }
            
            size_t rank() const
            {
                return N;
            }
            
            size_type size() const
            {
                return (size_type)_items.size();
            }
            
            tensor_type& operator[]( const size_t i )
            {
                return *_items[i];
            }
            
            const tensor_type& operator[]( const size_t i ) const
            {
                return *_items[i];
            }
            
            template<typename ...Ts, typename = std::enable_if_t<sizeof...(Ts) == N && (is_size_arg<Ts> && ...)>>
            void reshape( Ts... shape )
            {
                for ( size_t i = 0; i < _items.size(); ++i )
                {
                    _items[i]->reshape(get_shape_item(shape, i)...);
                }
            }
            
            void resize( const size_type size )
            {
                if ( size != (size_type)_items.size() )
                {
                    throw std::invalid_argument("invalid tensor list resizing to size " + std::to_string(size) + " while fixed size is " + std::to_string(_items.size()));
                }
            }
            
            shape_pack_type shapes( const size_t i ) const
            {
                return shape_pack_type(*this, i);
            }
            
            size_type shape( const size_t i ) const
            {
                return largest_shape(i);
            }
            
        private:
            
            static size_type get_shape_item( const size_type& value, const size_t idx )
            {
                return value;
            }
            
            static size_type get_shape_item( const size_pack_type& value, const size_t idx )
            {
                return value[idx];
            }
            
            size_type largest_shape( const size_t dim ) const
            {
                size_type largest = _items.front().shape(dim);
                for ( size_t i = 1; i < _items.size(); ++i )
                {
                    auto shape = _items[i].shape(dim);
                    if ( shape > largest )
                    {
                        largest = shape;
                    }
                }
                return largest;
            }
            
        private:
            
            std::vector<tensor_type*> _items;
        };
    
    
        template<typename P>
        class ShapePack
        {
            typedef P tensor_pack_type;
            
        public:
            
            typedef typename tensor_pack_type::size_type value_type;
            
        public:
            
            ShapePack( const tensor_pack_type& tensors, const size_t dim )
            : _tensors(tensors), _dim(dim)
            {
            }
            
            size_t size() const
            {
                return _tensors.size();
            }
            
            const value_type operator[]( const size_t i ) const
            {
                return _tensors[i].shape(_dim);
            }
            
        private:
            
            const tensor_pack_type& _tensors;
            size_t _dim;
        };
    
    
        template<typename T>
        class ValuePack
        {
            typedef std::vector<T> container_type;
            
        public:
            
            typedef T value_type;
            
        public:
            
            ValuePack( const size_t size = 0, const value_type value = value_type() )
            : _data(size, value)
            {
            }
            
            ValuePack( std::initializer_list<T> il )
            : _data(il)
            {
            }
            
            template<typename X>
            ValuePack( const X& x )
            : _data(x.size())
            {
                for ( size_t i = 0; i < x.size(); ++i )
                {
                    _data[i] = x[i];
                }
            }
            
            template<typename X>
            ValuePack& operator=( const X& x )
            {
                _data.resize(x.size());
                for ( size_t i = 0; i < x.size(); ++i )
                {
                    _data[i] = x[i];
                }
                return *this;
            }
            
            size_t size() const
            {
                return _data.size();
            }
            
            value_type& operator[]( const size_t i )
            {
                return _data[i];
            }
            
            const value_type& operator[]( const size_t i ) const
            {
                return _data[i];
            }
            
            void append( const value_type& v )
            {
                _data.push_back(v);
            }
            
        private:
            
            container_type _data;
        };
    
    
        template<typename T>
        class UniformExpr
        {
        public:
            
            typedef T value_type;
            
        public:
            
            UniformExpr( const value_type& value, const size_t size )
            : _value(value), _size(size)
            {
            }
            
            size_t size() const
            {
                return _size;
            }
            
            const value_type& operator[]( const size_t i ) const
            {
                return _value;
            }
            
        private:
            
            value_type _value;
            size_t _size;
        };
    
    
        template<template<typename> class Op, typename Arg>
        class UnaryExpr
        {
            typedef Arg arg_type;
            
        public:
            
            typedef Op<typename arg_type::value_type> op_type;
            typedef decltype(op_type()(typename arg_type::value_type())) value_type;
            
        public:
            
            UnaryExpr( const arg_type& arg )
            : _arg(arg)
            {
            }
            
            size_t size() const
            {
                return _arg.size();
            }
            
            value_type operator[]( const size_t i ) const
            {
                static op_type op;
                return op(_arg[i]);
            }
            
        private:
            
            const arg_type& _arg;
        };
    
    
        template<template<typename> class Op, typename Left, typename Right>
        class BinaryExpr
        {
            typedef Left left_type;
            typedef Right right_type;
            
        public:
            
            static_assert(std::is_same_v<typename left_type::value_type, typename right_type::value_type>);
            
            typedef Op<typename left_type::value_type> op_type;
            typedef decltype(op_type()(typename left_type::value_type(),typename left_type::value_type())) value_type;
            
        public:
            
            BinaryExpr( const left_type& left, const right_type& right )
            : _left(left), _right(right), _size(std::max(left.size(), right.size()))
            {
            }
            
            size_t size() const
            {
                return _size;
            }
            
            value_type operator[]( const size_t i ) const
            {
                static op_type op;
                return op(_left[i], _right[i]);
            }
            
        private:
            
            const left_type& _left;
            const right_type& _right;
            const size_t _size;
        };
    
    
        template<typename Cond, typename Left, typename Right>
        class SelectExpr
        {
            typedef Cond cond_type;
            typedef Left left_type;
            typedef Right right_type;
            
        public:
            
            static_assert(std::is_same_v<typename cond_type::value_type, bool_t>);
            static_assert(std::is_same_v<typename left_type::value_type, typename right_type::value_type>);
            
            typedef typename left_type::value_type value_type;
            
        public:
            
            SelectExpr( const cond_type& cond, const left_type& left, const right_type& right )
            : _cond(cond), _left(left), _right(right), _size(std::max(cond.size(), std::max(left.size(), right.size())))
            {
            }
            
            size_t size() const
            {
                return _size;
            }
            
            value_type operator[]( const size_t i ) const
            {
                return _cond[i] ? _left[i] : _right[i];
            }
            
        private:
            
            const cond_type& _cond;
            const left_type& _left;
            const right_type& _right;
            const size_t _size;
        };
    
    
        template<typename Pack>
        class SliceExpr
        {
            typedef Pack pack_type;
            
        public:
            
            static constexpr size_t MaxSize = pack_type::MaxSize;
            
            typedef typename pack_type::value_type value_type;
            typedef int size_type;
            
        public:
            
            SliceExpr( const pack_type& pack, const size_type first, const size_type last, const size_type stride )
            : _pack(pack), _first(first), _stride(stride), _size((last - first) / stride)
            {
            }
            
            size_t size() const
            {
                return _size;
            }
            
            value_type operator[]( const size_t i ) const
            {
                return _pack[_first + i * _stride];
            }
            
        private:
            
            const pack_type& _pack;
            const size_type _first;
            const size_type _stride;
            const size_t _size;
        };
    
    
        class RangeExpr
        {
        public:
            
            typedef int_t value_type;
            typedef int size_type;
            
        public:
            
            RangeExpr( const size_type first, const size_type last, const size_type stride )
            : _first(first), _stride(stride), _size((last - first) / stride)
            {
            }
            
            size_t size() const
            {
                return _size;
            }
            
            value_type operator[]( const size_t i ) const
            {
                return (value_type)(_first + i * _stride);
            }
            
        private:
            
            const size_type _first;
            const size_type _stride;
            const size_t _size;
        };
    
    
        namespace detail
        {
        
            template<typename Arg, typename... Args>
            struct _first_type
            {
                typedef Arg type;
            };
        
            template<typename... Args>
            using first_type = typename _first_type<Args...>::type;
        
            template<typename Arg, typename... Args>
            struct _value_type
            {
                typedef typename Arg::value_type type;
            };
            
            template<typename... Args> struct _value_type<int_t,Args...> { typedef int_t type; };
            template<typename... Args> struct _value_type<real_t,Args...> { typedef real_t type; };
            template<typename... Args> struct _value_type<bool_t,Args...> { typedef bool_t type; };
            
            template<typename... Args>
            using value_type = typename _value_type<Args...>::type;
        
        }   // namespace detail
        
        
        template<typename... Args, typename = std::enable_if_t<(is_typename<Args> && ...)>>
        ValuePack<detail::first_type<Args...>> list( const Args&... args )
        {
            return ValuePack<detail::first_type<Args...>>{ args... };
        }
    
        template<typename T, typename = std::enable_if_t<is_typename<T>>>
        UniformExpr<T> uniform( const T& value, const size_t size )
        {
            return UniformExpr<T>(value, size);
        }
    
        template<template<typename> class Op, typename Arg>
        inline UnaryExpr<Op,Arg> unary( const Arg& arg )
        {
            return UnaryExpr<Op,Arg>(arg);
        }
        
        template<template<typename> class Op, typename Left, typename Right>
        inline BinaryExpr<Op,Left,Right> binary( const Left& left, const Right& right )
        {
            return BinaryExpr<Op,Left,Right>(left, right);
        }
    
        template<typename Cond, typename Left, typename Right>
        inline SelectExpr<Cond,Left,Right> select( const Cond& cond, const Left& left, const Right& right )
        {
            return SelectExpr<Cond,Left,Right>(cond, left, right);
        }
    
        namespace detail
        {
        
            template<template<typename> class Op, typename T> struct reduce_init {};
        
            template<typename T> struct reduce_init<std::plus, T>
            {
                static T value() { return (T)0; }
            };
        
            template<typename T> struct reduce_init<std::multiplies, T>
            {
                static T value() { return (T)1; }
            };
        
            template<> struct reduce_init<minimize, real_t>
            {
                static real_t value() { return std::numeric_limits<real_t>::infinity(); }
            };
        
            template<> struct reduce_init<maximize, real_t>
            {
                static real_t value() { return -std::numeric_limits<real_t>::infinity(); }
            };
        
            template<> struct reduce_init<minimize, int_t>
            {
                static int_t value() { return std::numeric_limits<int_t>::max(); }
            };
        
            template<> struct reduce_init<maximize, int_t>
            {
                static int_t value() { return std::numeric_limits<int_t>::min(); }
            };
        
            template<> struct reduce_init<std::logical_or, bool_t>
            {
                static bool_t value() { return false; }
            };
        
            template<> struct reduce_init<std::logical_and, bool_t>
            {
                static bool_t value() { return true; }
            };
        
        
            template<template<typename> class Op, typename T> struct reduce_check
            {
                static void check( const size_t size ) {}
            };
        
            template<typename T> struct reduce_check<minimize, T>
            {
                static void check( const size_t size ) { if ( size == 0 ) throw std::invalid_argument("zero size in min reduction"); }
            };
        
            template<typename T> struct reduce_check<maximize, T>
            {
                static void check( const size_t size ) { if ( size == 0 ) throw std::invalid_argument("zero size in max reduction"); }
            };
        
        }   // namespace detail
    
        template<template<typename> class Op, typename Arg>
        inline typename Arg::value_type reduce( const Arg& arg )
        {
            detail::reduce_check<Op, typename Arg::value_type>::check(arg.size());
            static Op<typename Arg::value_type> op;
            typename Arg::value_type res = detail::reduce_init<Op, typename Arg::value_type>::value();
            for ( size_t i = 0; i < arg.size(); ++i )
            {
                res = op(res, arg[i]);
            }
            return res;
        }
    
        template<template<typename> class Op, typename Arg>
        inline ValuePack<typename Arg::value_type> accum( const Arg& arg )
        {
            static Op<typename Arg::value_type> op;
            ValuePack<typename Arg::value_type> res(arg.size());
            if ( res.size() )
            {
                res[0] = arg[0];
            }
            for ( size_t i = 1; i < arg.size(); ++i )
            {
                res[i] = op(res[i-1], arg[i]);
            }
            return res;
        }
    
        namespace detail
        {
        
            template<typename T, typename Arg, typename... Args>
            inline void concat( ValuePack<T>& result, const Arg& first, const Args& ...rest )
            {
                if constexpr( is_typename<Arg> )
                {
                    result.append(first);
                }
                else
                {
                    for ( size_t i = 0; i < first.size(); ++i )
                    {
                        result.append(first[i]);
                    }
                }
                if constexpr( sizeof...(rest) > 0 )
                {
                    concat(result, rest...);
                }
            }
        
        }   // namespace detail
    
        template<typename... Args>
        inline ValuePack<detail::value_type<Args...>> concat( const Args& ...args )
        {
            ValuePack<detail::value_type<Args...>> result;
            detail::concat(result, args...);
            return result;
        }
    
        template<typename P>
        inline SliceExpr<P> slice( const P& pack, const int first, const int last, const int stride = 1 )
        {
            return SliceExpr<P>(pack, first, last, stride);
        }
    
        inline RangeExpr range( const int first, const int last, const int stride = 1 )
        {
            return RangeExpr(first, last, stride);
        }
        
        
        template<typename T>
        inline std::string str( const T* items, const size_t count )
        {
            std::string str;
            str += "(";
            for ( size_t i = 0; i < count; ++i )
            {
                if ( i )
                {
                    str += ", ";
                }
                str += std::to_string(items[i]);
            }
            str += ")";
            return str;
        }
        
        
        template<size_t N>
        inline Tensor<N,bool_t>& condition_result()
        {
            static auto result = Tensor<N,bool_t>::singular();
            return result;
        }
    
        
    }   // namespace rt
}   // namespace sknd


namespace std
{
    
    template<size_t N, typename T>
    void swap( sknd::rt::Tensor<N,T>& left, sknd::rt::Tensor<N,T>& right )
    {
        left.swap(right);
    }
}


template<typename T>
std::ostream& operator<<( std::ostream& os, const sknd::rt::ValuePack<T>& pack )
{
    os << '[';
    for ( size_t i = 0; i < pack.size(); ++i )
    {
        if ( i )
        {
            os << ", ";
        }
        os << pack[i];
    }
    os << ']';
    return os;
}


#endif
