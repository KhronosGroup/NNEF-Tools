#ifndef _TS_TYPES_H_
#define _TS_TYPES_H_

#include <vector>
#include <string>
#include <cmath>
#include <map>


namespace nd
{
    
    typedef float real_t;
    typedef int int_t;
    typedef bool bool_t;
    typedef std::string str_t;
    
    enum class Typename : char { Type = 0, Arith = 1, Num = 2, Int = 3, Real = 4, Bool = 5, Str = 6 };


    template<typename T>
    struct typename_of {};

    template<> struct typename_of<void> { static const Typename value = Typename::Type; };
    template<> struct typename_of<int_t> { static const Typename value = Typename::Int; };
    template<> struct typename_of<real_t> { static const Typename value = Typename::Real; };
    template<> struct typename_of<bool_t> { static const Typename value = Typename::Bool; };
    template<> struct typename_of<str_t> { static const Typename value = Typename::Str; };


    template<typename T>
    inline constexpr bool is_typename = std::is_same_v<T,int_t> || std::is_same_v<T,real_t> || std::is_same_v<T,bool_t> || std::is_same_v<T,str_t>;
    
    
    struct Type
    {
        Typename name = Typename::Type;
        bool optional = false;
        bool tensor = false;
        bool packed = false;
    };
    
    
    inline bool operator==( const Type& lhs, const Type& rhs )
    {
        return lhs.name == rhs.name && lhs.optional == rhs.optional && lhs.tensor == rhs.tensor && lhs.packed == rhs.packed;
    }
    
    inline bool operator!=( const Type& lhs, const Type& rhs )
    {
        return !(lhs == rhs);
    }
    
    inline Type make_type( const Typename& name )
    {
        return Type{ name, false, false, false };
    }
    
    inline Type make_type( const Typename& name, const bool optional, const bool tensor, const bool packed )
    {
        return Type{ name, optional, tensor, packed };
    }
    
    inline Type as_optional( const Type& type )
    {
        return Type{ type.name, true, type.tensor, type.packed };
    }
    
    inline Type as_tensor( const Type& type )
    {
        return Type{ type.name, type.optional, true, type.packed };
    }
    
    inline Type as_packed( const Type& type )
    {
        return Type{ type.name, type.optional, type.tensor, true };
    }
    
    inline Type as_non_optional( const Type& type )
    {
        return Type{ type.name, false, type.tensor, type.packed };
    }
    
    inline Type as_non_tensor( const Type& type )
    {
        return Type{ type.name, type.optional, false, type.packed };
    }
    
    inline Type as_non_packed( const Type& type )
    {
        return Type{ type.name, type.optional, type.tensor, false };
    }
    
    inline bool is_abstract( const Typename& type )
    {
        return type == Typename::Type || type == Typename::Arith || type == Typename::Num;
    }
    
    inline bool is_compatible( const Typename& param, const Typename arg )
    {
        if ( param == arg || param == Typename::Type )
        {
            return true;
        }
        else if ( param == Typename::Arith )
        {
            return arg == Typename::Int || arg == Typename::Real || arg == Typename::Bool || arg == Typename::Num;
        }
        else if ( param == Typename::Num )
        {
            return arg == Typename::Int || arg == Typename::Real;
        }
        else
        {
            return false;
        }
    }

    inline bool is_empty_pack( const Type& type )
    {
        return type.packed && type.name == Typename::Type;
    }
    
    inline const std::string& str( const Typename& type )
    {
        static const std::string typenames[] = { "type", "arith", "num", "int", "real", "bool", "str", };
        return typenames[(size_t)type];
    }
    
    inline std::string str( const Type& type )
    {
        std::string str;
        if ( type.optional )
        {
            str += "optional ";
        }
        str += nd::str(type.name);
        if ( type.tensor )
        {
            str += "[]";
        }
        if ( type.packed )
        {
            str += "..";
        }
        return str;
    }


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
    
}   // namespace nd


#endif
