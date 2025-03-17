#ifndef _TS_TENSORREF_H_
#define _TS_TENSORREF_H_

#include "either.h"
#include "types.h"
#include <vector>
#include <string>


namespace nd
{

    struct Tensor;
    struct TensorPack;
    class ValueExpr;

    
    namespace impl
    {
    
        template<typename T, typename P, typename V>
        struct TensorRef : public Either<T*,P*>
        {
            typedef Either<T*,P*> either_type;
            
            TensorRef( std::nullptr_t = nullptr ) : either_type((T*)nullptr) {}
            TensorRef( T* tensor ) : either_type(tensor) {}
            TensorRef( P* pack ) : either_type(pack) {}
            
            TensorRef& operator=( T* tensor ) { (either_type&)(*this) = tensor; return *this; }
            TensorRef& operator=( P* pack ) { (either_type&)(*this) = pack; return *this; }
            TensorRef& operator=( const TensorRef& ref ) { (either_type&)(*this) = ref; return *this; }
            
            template<typename X> bool is() const { return either_type::template is<X>(); }
            template<typename X> X& as() { return either_type::template as<X>(); }
            template<typename X> const X& as() const { return either_type::template as<X>(); }
            
            const std::string& name() const { return packed() ? as<P*>()->name : as<T*>()->name; }
            std::string& name() { return packed() ? as<P*>()->name : as<T*>()->name; }
            const Typename& dtype() const { return packed() ? as<P*>()->dtype : as<T*>()->dtype; }
            Typename& dtype() { return packed() ? as<P*>()->dtype : as<T*>()->dtype; }
            const std::vector<ValueExpr>& shape() const { return packed() ? as<P*>()->shape : as<T*>()->shape; }
            std::vector<ValueExpr>& shape() { return packed() ? as<P*>()->shape : as<T*>()->shape; }
            const std::vector<int_t>& max_shape() const { return packed() ? as<P*>()->max_shape : as<T*>()->max_shape; }
            std::vector<int_t>& max_shape() { return packed() ? as<P*>()->max_shape : as<T*>()->max_shape; }
            size_t rank() const { return packed() ? as<P*>()->shape.size() : as<T*>()->shape.size(); }
            bool packed() const { return is<P*>(); }
            size_t max_size() const { return as<P*>()->items.size(); }
            std::optional<size_t> max_size_or_null() const { return packed() ? max_size() : (std::optional<size_t>)std::nullopt; }
            const ValueExpr& size() const { return as<P*>()->size; }
            const ValueExpr& size_or_null() const { return packed() ? size() : V::null(); }
            bool is_constant() const { return !packed() ? as<T*>()->value != nullptr :
                std::all_of(as<P*>()->items.begin(), as<P*>()->items.end(), []( const T* item ){ return item->value != nullptr; }); }
            
            T* operator->() { return as<T*>(); }
            T& operator*() { return *as<T*>(); }
            T& operator[]( size_t i ) { return *as<P*>()->items[i]; }
            explicit operator T&() { return *as<T*>(); }
            explicit operator P&() { return *as<P*>(); }
            explicit operator T*() { return as<T*>(); }
            explicit operator P*() { return as<P*>(); }
            
            const T* operator->() const { return as<T*>(); }
            const T& operator*() const { return *as<T*>(); }
            const T& operator[]( size_t i ) const { return *as<P*>()->items[i]; }
            explicit operator const T&() const { return *as<T*>(); }
            explicit operator const P&() const { return *as<P*>(); }
            explicit operator const T*() const { return as<T*>(); }
            explicit operator const P*() const { return as<P*>(); }
            
            bool operator==( const TensorRef& other ) const { return (either_type)(*this) == (either_type)other; }
            bool operator!=( const TensorRef& other ) const { return (either_type)(*this) != (either_type)other; }
            bool operator==( const std::nullptr_t& ) const { return packed() ? as<P*>() == nullptr : as<T*>() == nullptr; }
        };
    
    }   // namespace impl
    

    using TensorRef = impl::TensorRef<Tensor,TensorPack,ValueExpr>;

}   // namespace nd


#endif
