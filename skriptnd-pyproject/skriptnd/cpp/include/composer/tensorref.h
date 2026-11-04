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

#ifndef _SKND_TENSORREF_H_
#define _SKND_TENSORREF_H_

#include "variant.h"
#include "types.h"
#include <vector>
#include <string>
#include <optional>


namespace sknd
{

    struct Tensor;
    struct TensorPack;
    class ValueExpr;

    
    namespace impl
    {
    
        template<typename T, typename P, typename V>
        class TensorRef
        {
            typedef Variant<nullptr_t,T*,P*> variant_type;
            
        public:
            
            TensorRef( nullptr_t = nullptr ) : _data(nullptr) {}
            TensorRef( T* tensor ) : _data(tensor) {}
            TensorRef( P* pack ) : _data(pack) {}
            
            TensorRef& operator=( T* tensor )
            {
                _data = tensor;
                return *this;
            }
            
            TensorRef& operator=( P* pack )
            {
                _data = pack;
                return *this;
            }
            
            TensorRef& operator=( const TensorRef& other )
            {
                _data = other._data;
                return *this;
            }
            
            bool packed() const
            {
                return _data.index() == 2;
            }
            
            bool itemized() const
            {
                return packed() && as_pack().items != nullptr;
            }
            
            T& as_tensor()
            {
                assert(_data.template is<T*>());
                return *_data.template as<T*>();
            }
            
            const T& as_tensor() const
            {
                assert(_data.template is<T*>());
                return *_data.template as<T*>();
            }
            
            P& as_pack()
            {
                assert(_data.template is<P*>());
                return *_data.template as<P*>();
            }
            
            const P& as_pack() const
            {
                assert(_data.template is<P*>());
                return *_data.template as<P*>();
            }
            
        public:
            
            const std::string& name() const
            {
                return packed() ? as_pack().name : as_tensor().name;
            }
            
            const Typename& dtype() const
            {
                return packed() ? as_pack().dtype : as_tensor().dtype;
            }
            
            const std::vector<ValueExpr>& shape() const
            {
                return packed() ? as_pack().shape : as_tensor().shape;
            }
            
            std::vector<ValueExpr>& shape()
            {
                return packed() ? as_pack().shape : as_tensor().shape;
            }
            
            size_t rank() const
            {
                return shape().size();
            }
            
            const ValueExpr& size() const
            {
                return packed() ? as_pack().size : V::null();
            }
            
            size_t item_count() const
            {
                return as_pack().size.as_int();
            }
            
            T* operator[]( const size_t idx ) const
            {
                return as_pack().items[idx];
            }
            
            auto begin() const
            {
                return as_pack().items.get();
            }
            
            auto end() const
            {
                return as_pack().items.get() + as_pack().size.as_int();
            }
            
            bool is_constant() const
            {
                if ( itemized() )
                {
                    for ( auto& item : *this )
                    {
                        if ( item->value == nullptr )
                        {
                            return false;
                        }
                    }
                    return true;
                }
                else if ( !packed() )
                {
                    return as_tensor().value != nullptr;
                }
                return false;
            }
            
            T* operator->() { return &as_tensor(); }
            T& operator*() { return as_tensor(); }
            
            const T* operator->() const { return &as_tensor(); }
            const T& operator*() const { return as_tensor(); }
            
            bool operator==( const TensorRef& other ) const { return _data == other._data; }
            bool operator!=( const TensorRef& other ) const { return _data != other._data; }
            bool operator==( const nullptr_t& ) const { return _data == nullptr; }
            
        private:
            
            variant_type _data;
        };
    
    }   // namespace impl
    

    using TensorRef = impl::TensorRef<Tensor,TensorPack,ValueExpr>;

}   // namespace sknd


#endif
