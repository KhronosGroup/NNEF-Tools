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
        struct TensorRef
        {
            typedef Variant<nullptr_t,T*,P*> variant_type;
            
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
            
            T& as_tensor()
            {
                return *_data.template as<T*>();
            }
            
            const T& as_tensor() const
            {
                return *_data.template as<T*>();
            }
            
            P& as_pack()
            {
                return *_data.template as<P*>();
            }
            
            const P& as_pack() const
            {
                return *_data.template as<P*>();
            }
            
            const std::string& name() const { return packed() ? as_pack().name : as_tensor().name; }
            std::string& name() { return packed() ? as_pack().name : as_tensor().name; }
            const Typename& dtype() const { return packed() ? as_pack().dtype : as_tensor().dtype; }
            Typename& dtype() { return packed() ? as_pack().dtype : as_tensor().dtype; }
            const std::vector<ValueExpr>& shape() const { return packed() ? as_pack().shape : as_tensor().shape; }
            std::vector<ValueExpr>& shape() { return packed() ? as_pack().shape : as_tensor().shape; }
            const std::vector<int_t>& max_shape() const { return packed() ? as_pack().max_shape : as_tensor().max_shape; }
            std::vector<int_t>& max_shape() { return packed() ? as_pack().max_shape : as_tensor().max_shape; }
            size_t rank() const { return packed() ? as_pack().shape.size() : as_tensor().shape.size(); }
            bool packed() const { return _data.index() == 2; }
            bool listed() const { return as_pack().items != nullptr; }
            int_t max_size() const { return as_pack().max_size; }
            std::optional<int_t> max_size_or_null() const { return packed() ? max_size() : (std::optional<int_t>)std::nullopt; }
            ValueExpr& size() { return as_pack().size; }
            const ValueExpr& size() const { return as_pack().size; }
            const ValueExpr& size_or_null() const { return packed() ? size() : V::null(); }
            
            bool is_constant() const
            {
                if ( packed() )
                {
                    auto& pack = as_pack();
                    if ( !pack.items )
                    {
                        return false;
                    }
                    for ( size_t i = 0; i < pack.max_size; ++i )
                    {
                        if ( pack.items[i]->value == nullptr )
                        {
                            return false;
                        }
                    }
                    return true;
                }
                else
                {
                    return as_tensor().value != nullptr;
                }
            }
            
            T* operator->() { return &as_tensor(); }
            T& operator*() { return as_tensor(); }
            T& operator[]( const size_t i ) { return *as_pack().items[i]; }
            
            const T* operator->() const { return &as_tensor(); }
            const T& operator*() const { return as_tensor(); }
            const T& operator[]( const size_t i ) const { return *as_pack().items[i]; }
            
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
