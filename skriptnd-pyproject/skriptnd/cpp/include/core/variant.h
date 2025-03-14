#ifndef _TS_VARIANT_H_
#define _TS_VARIANT_H_

#include <algorithm>
#include <type_traits>


namespace ts
{

    namespace detail
    {
    
        template<size_t Idx, typename T, typename T1, typename... Ts>
        struct index_of_t
        {
            enum { value = std::is_same<T,T1>::value ? Idx : index_of_t<Idx+1,T,Ts...>::value };
        };
    
        template<size_t Idx, typename T, typename T1>
        struct index_of_t<Idx,T,T1>
        {
            enum { value = std::is_same<T,T1>::value ? Idx : -1 };
        };
    
        template<typename T, typename... Ts>
        constexpr size_t index_of()
        {
            return index_of_t<0,T,Ts...>::value;
        }
    
        template<typename T1, typename... Ts>
        struct is_unique_t
        {
            enum { value = (!std::is_same<T1,Ts>::value && ...) };
        };
    
        template<typename T>
        struct is_unique_t<T>
        {
            enum { value = 1 };
        };
    
        template<typename... Ts>
        constexpr bool is_unique()
        {
            return is_unique_t<Ts...>::value;
        }
    
        template<typename T, typename = void>
        struct variant_type_traits
        {
            static size_t construct( const T& value )
            {
                return reinterpret_cast<size_t>(new T(value));
            }
            
            static size_t construct( T&& value )
            {
                return reinterpret_cast<size_t>(new T(std::forward<T>(value)));
            }
            
            static void destruct( size_t data )
            {
                delete reinterpret_cast<T*>(data);
            }
            
            static T& get( size_t& data )
            {
                return *reinterpret_cast<T*>(data);
            }
            
            static const T& get( const size_t& data )
            {
                return *reinterpret_cast<const T*>(data);
            }
        };
    
        template<typename T>
        struct variant_type_traits<T,std::enable_if_t<sizeof(T) <= sizeof(size_t)>>
        {
            static size_t construct( const T& value )
            {
                size_t data;
                *reinterpret_cast<T*>(&data) = value;
                return data;
            }
            
            static void destruct( size_t data )
            {
            }
            
            static T& get( size_t& data )
            {
                return *reinterpret_cast<T*>(&data);
            }
            
            static const T& get( const size_t& data )
            {
                return *reinterpret_cast<const T*>(&data);
            }
        };
    
    }   // namespace detail


    template<typename... Ts>
    class Variant
    {
        static_assert(sizeof...(Ts) > 0, "variant types must not be empty");
        static_assert(detail::is_unique<Ts...>(), "variant types must be unique");
        
    public:
        
        template<typename T>
        static constexpr size_t contains()
        {
            return (std::is_same_v<T,Ts> || ...);
        }
        
        template<typename T>
        static constexpr size_t index_of()
        {
            return detail::index_of<T,Ts...>();
        }
        
        static const size_t invalid_index = -1;
        
    public:
        
        Variant( std::nullptr_t = nullptr )
        : _index(invalid_index)
        {
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        Variant( const T& value )
        {
            construct(value);
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        Variant( T&& value )
        {
            construct(std::forward<T>(value));
        }
        
        Variant( const Variant& other )
        {
            copy(other);
        }
        
        Variant( Variant&& other )
        : Variant()
        {
            swap(other);
        }
        
        Variant& operator=( const Variant& other )
        {
            destruct();
            copy(other);
            return *this;
        }
        
        Variant& operator=( Variant&& other )
        {
            swap(other);
            return *this;
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        Variant& operator=( const T& value )
        {
            destruct();
            construct(value);
            return *this;
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        Variant& operator=( T&& value )
        {
            destruct();
            construct(std::forward<T>(value));
            return *this;
        }
        
        Variant& operator=( std::nullptr_t )
        {
            destruct();
            _index = invalid_index;
            return *this;
        }
        
        ~Variant()
        {
            destruct();
        }
        
        bool valid() const
        {
            return _index != invalid_index;
        }
        
        size_t index() const
        {
            return _index;
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        bool is() const
        {
            return index() == index_of<T>();
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        const T& as() const
        {
            return detail::variant_type_traits<T>::get(_data);
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        T& as()
        {
            return detail::variant_type_traits<T>::get(_data);
        }
        
        void swap( Variant& other )
        {
            std::swap(_index, other._index);
            std::swap(_data, other._data);
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        bool operator==( const T& value )
        {
            return _index == index_of<T>() && as<T>() == value;
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        bool operator!=( const T& value )
        {
            return !(*this == value);
        }
        
        bool operator==( const Variant& other ) const
        {
            return equals(other);
        }
        
        bool operator!=( const Variant& other ) const
        {
            return !(*this == other);
        }
        
        bool operator==( std::nullptr_t ) const
        {
            return _index == invalid_index;
        }
        
        bool operator!=( std::nullptr_t ) const
        {
            return _index != invalid_index;
        }
        
        template<typename C>
        void visit( const C& callback ) const
        {
            using Func = void(const C&, const Variant&);
            static Func* const Visitors[sizeof...(Ts)] = { []( const C& c, const Variant& v ){ c(v.as<Ts>()); }... };
            
            Visitors[_index](callback, *this);
        }
        
    private:
        
        template<typename T>
        void construct( const T& value )
        {
            _index = index_of<T>();
            _data = detail::variant_type_traits<T>::construct(value);
        }
        
        template<typename T>
        void construct( T&& value )
        {
            _index = index_of<T>();
            _data = detail::variant_type_traits<T>::construct(std::forward<T>(value));
        }
        
        void destruct()
        {
            using Func = void(Variant&);
            static Func* const Destructors[sizeof...(Ts)] = { []( Variant& v ){ detail::variant_type_traits<Ts>::destruct(v._data); }... };
            
            if ( this->valid() )
            {
                Destructors[index()](*this);
            }
        }
        
        void copy( const Variant& other )
        {
            using Func = size_t(const Variant&);
            static Func* const Constructors[sizeof...(Ts)] = { []( const Variant& v ){ return detail::variant_type_traits<Ts>::construct(v.as<Ts>()); }... };
            
            _index = other.index();
            if ( other.valid() )
            {
                _data = Constructors[other.index()](other);
            }
        }
        
        bool equals( const Variant& other ) const
        {
            using Func = bool(const Variant&,const Variant&);
            static Func* const Comparators[sizeof...(Ts)] = { []( const Variant& a, const Variant& b ){ return a.as<Ts>() == b.as<Ts>(); }... };
            
            return _index == other._index && (_index == invalid_index || Comparators[_index](*this, other));
        }
        
    private:
        
        size_t _index;
        size_t _data;
    };


    template<typename... Ts>
    class VariantData
    {
        static_assert(sizeof...(Ts) > 0, "variant types must not be empty");
        static_assert(detail::is_unique<Ts...>(), "variant types must be unique");
        
    public:
        
        template<typename T>
        static constexpr size_t contains()
        {
            return (std::is_same_v<T,Ts> || ...);
        }
        
        template<typename T>
        static constexpr size_t index_of()
        {
            return detail::index_of<T,Ts...>();
        }
        
    public:
        
        VariantData()
        : _data(0)
        {
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        VariantData( const T& value )
        {
            construct(value);
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        VariantData( T&& value )
        {
            construct(std::forward<T>(value));
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        const T& as() const
        {
            return detail::variant_type_traits<T>::get(_data);
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        T& as()
        {
            return detail::variant_type_traits<T>::get(_data);
        }
        
        void swap( VariantData& other )
        {
            std::swap(_data, other._data);
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        bool operator==( const T& value )
        {
            return as<T>() == value;
        }
        
        template<typename T, typename = std::enable_if_t<contains<T>()>>
        bool operator!=( const T& value )
        {
            return !(*this == value);
        }
        
        template<typename T>
        void construct( const T& value )
        {
            _data = detail::variant_type_traits<T>::construct(value);
        }
        
        template<typename T>
        void construct( T&& value )
        {
            _data = detail::variant_type_traits<T>::construct(std::forward<T>(value));
        }
        
        void construct( const VariantData& other, const size_t index )
        {
            using Func = size_t(const VariantData&);
            static Func* const Constructors[sizeof...(Ts)] = { []( const VariantData& v ){ return detail::variant_type_traits<Ts>::construct(v.as<Ts>()); }... };
            
            _data = Constructors[index](other);
        }
        
        void destruct( const size_t index )
        {
            using Func = void(VariantData&);
            static Func* const Destructors[sizeof...(Ts)] = { []( VariantData& v ){ detail::variant_type_traits<Ts>::destruct(v._data); }... };
            
            Destructors[index](*this);
        }
        
        template<typename T>
        void destruct()
        {
            destruct(index_of<T>());
        }
        
        bool equals( const VariantData& other, const size_t index ) const
        {
            using Func = bool(const VariantData&,const VariantData&);
            static Func* const Comparators[sizeof...(Ts)] = { []( const VariantData& a, const VariantData& b ){ return a.as<Ts>() == b.as<Ts>(); }... };
            
            return Comparators[index](*this, other);
        }
        
        template<typename C>
        void visit( const C& callback, const size_t index ) const
        {
            using Func = void(const C&, const VariantData&);
            static Func* const Visitors[sizeof...(Ts)] = { []( const C& c, const VariantData& v ){ c(v.as<Ts>()); }... };
            
            Visitors[index](callback, *this);
        }
        
    private:
        
        size_t _data;
    };

}   // namespace ts


#endif
