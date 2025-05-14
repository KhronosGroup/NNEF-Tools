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

#ifndef _SKND_MAYBE_H_
#define _SKND_MAYBE_H_


namespace sknd
{

    template<typename T>
    class Maybe
    {
    public:
        
        typedef T value_type;
        
    public:
        
        Maybe( std::nullptr_t = nullptr )
        : _value(nullptr)
        {
        }
        
        Maybe( const value_type& value )
        : _value(new value_type(value))
        {
        }
        
        Maybe( value_type&& value )
        : _value(new value_type(std::forward<value_type>(value)))
        {
        }
        
        Maybe( const Maybe& other )
        : _value(other._value ? new value_type(*other._value) : nullptr)
        {
        }
        
        Maybe( Maybe&& other )
        : _value(nullptr)
        {
            swap(other);
        }
        
        ~Maybe()
        {
            destruct();
        }
        
        void reset() const
        {
            destruct();
            _value = nullptr;
        }
        
        operator bool() const
        {
            return (bool)_value;
        }
        
        const value_type& operator*() const
        {
            return *_value;
        }
        
        value_type& operator*()
        {
            return *_value;
        }
        
        const value_type* operator->() const
        {
            return _value;
        }
        
        value_type* operator->()
        {
            return _value;
        }
        
        Maybe& operator=( const Maybe& other )
        {
            destruct();
            _value = other._value ? new value_type(*other._value) : nullptr;
            return *this;
        }
        
        Maybe& operator=( Maybe&& other )
        {
            swap(other);
            return *this;
        }
        
        bool operator==( const Maybe& other ) const
        {
            return _value == other._value;
        }
        
        bool operator!=( const Maybe& other ) const
        {
            return _value != other._value;
        }
        
        bool operator==( std::nullptr_t ) const
        {
            return _value == nullptr;
        }
        
        bool operator!=( std::nullptr_t ) const
        {
            return _value != nullptr;
        }
        
        void swap( Maybe& other )
        {
            std::swap(_value, other._value);
        }
        
    private:
        
        void destruct()
        {
            if ( _value )
            {
                delete _value;
            }
        }
        
    private:
        
        value_type* _value;
    };

}   // namespace sknd


#endif
