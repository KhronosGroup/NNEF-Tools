/*
 * Copyright (c) 2017 The Khronos Group Inc.
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

#ifndef _NNEF_ERROR_H_
#define _NNEF_ERROR_H_

#include <exception>
#include <cstdarg>
#include <string>


namespace nnef
{
    
    class Error : public std::exception
    {
    public:
        
        struct Position
        {
            unsigned line;
            unsigned column;
            const char* filename;
            const Position* origin;
        };
        
    public:
        
        template<class... Args>
        Error( const Position& position, const char* format, Args&&... args )
        : _position(position), _message(formatString(format, std::forward<Args>(args)...))
        {
        }
        
        template<class... Args>
        Error( const char* format, Args&&... args )
        : _position({0,0,nullptr,nullptr}), _message(formatString(format, std::forward<Args>(args)...))
        {
        }
        
        virtual const char* what() const noexcept
        {
            return _message.c_str();
        }
        
        const Position& position() const
        {
            return _position;
        }
        
    public:
        
        static std::string formatString( const char* fmt, ... )
        {
            va_list args;
            
            va_start(args, fmt);
            auto length = vsnprintf(nullptr, 0, fmt, args);
            va_end(args);
            
            if ( length < 0 )
            {
                throw std::logic_error("string formatting error");
            }
            
            std::string str(length, '\0');
            
            va_start(args, fmt);
            vsnprintf((char*)str.data(), length + 1, fmt, args);
            va_end(args);
            
            return str;
        }
        
    private:
        
        Position _position;
        std::string _message;
    };

}   // namespace nnef


#endif
