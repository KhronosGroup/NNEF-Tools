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

#ifndef _SKND_ERROR_H_
#define _SKND_ERROR_H_

#include "position.h"
#include <functional>
#include <string>
#include <list>
#include <cstdarg>
#include <type_traits>
#include <stdexcept>


namespace sknd
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
        Error( const char* format, Args&&... args )
        : message(format_string(format, std::forward<Args>(args)...))
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
    
}   // namespace sknd


#endif
