#ifndef _TS_PRINT_H_
#define _TS_PRINT_H_

#include <string>
#include <vector>
#include <map>


namespace ts
{
    
    template<typename T>
    inline std::string str( const std::vector<T>& list, const char* sep = ", " )
    {
        std::string str;
        str += '[';
        for ( auto it = list.begin(); it != list.end(); ++it )
        {
            if ( it != list.begin() )
            {
                str += sep;
            }
            str += std::to_string(*it);
        }
        str += ']';
        return str;
    }

    template<typename T, size_t N>
    inline std::string str( const std::array<T,N>& list, const char* sep = ", " )
    {
        std::string str;
        str += '[';
        for ( auto it = list.begin(); it != list.end(); ++it )
        {
            if ( it != list.begin() )
            {
                str += sep;
            }
            str += std::to_string(*it);
        }
        str += ']';
        return str;
    }
    
    template<typename T>
    inline std::string str( const std::map<std::string,T>& dict, const char* item_sep = ", ", const char* entry_sep = ": " )
    {
        std::string str;
        
        str += '{';
        for ( auto it = dict.begin(); it != dict.end(); ++it )
        {
            if ( it != dict.begin() )
            {
                str += item_sep;
            }
            str += it->first;
            str += entry_sep;
            str += std::to_string(it->second);
        }
        str += '}';
        
        return str;
    }

    inline std::string str()
    {
        return {};
    }

    template<typename Arg, typename... Args>
    inline std::string str( const Arg& arg, const Args&... args )
    {
        return std::to_string(arg) + ((", " + std::to_string(args)) + ... + "");
    }
    
}   // namespace ts


#endif
