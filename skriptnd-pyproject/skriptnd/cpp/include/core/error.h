#ifndef _TS_ERROR_H_
#define _TS_ERROR_H_

#include "position.h"
#include <functional>
#include <string>
#include <list>
#include <cstdarg>
#include <type_traits>


namespace nd
{
    
    typedef std::pair<std::string,Position> StackTraceItem;
    typedef std::list<StackTraceItem> StackTrace;
    
    typedef std::function<void( const Position&, const std::string&, const StackTrace&, bool warning )> ErrorCallback;
    
    
    struct Error
    {
        Position position;
        std::string message;
        StackTrace trace;
        
        Error()
        {
        }
        
        template<typename... Args>
        Error( const Position& position, const char* format, Args&&... args )
        : position(position), message(format_string(format, std::forward<Args>(args)...))
        {
        }
        
        explicit operator bool() const
        {
            return !message.empty();
        }
        
        static std::string format_string( const char* fmt, ... )
        {
            va_list args;
            
            va_start(args, fmt);
            auto length = vsnprintf(nullptr, 0, fmt, args);
            va_end(args);
            
            if ( length < 0 )
            {
                throw std::runtime_error("string formatting error");
            }
            
            std::string str(length, '\0');
            
            va_start(args, fmt);
            vsnprintf((char*)str.data(), length + 1, fmt, args);
            va_end(args);
            
            return str;
        }
    };
    
}   // namespace nd


#endif
