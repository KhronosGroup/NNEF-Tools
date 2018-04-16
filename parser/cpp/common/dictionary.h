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

#ifndef _NNEF_DICTIONARY_H_
#define _NNEF_DICTIONARY_H_

#include <initializer_list>
#include <string>
#include <iostream>
#include <memory>
#include <map>


namespace nnef
{

    template<typename T>
    struct Dictionary : public std::map<std::string,T>
    {
        typedef std::map<std::string,T> map_type;
        typedef typename map_type::value_type value_type;

        using map_type::operator[];

        Dictionary()
        {
        }

        Dictionary( std::initializer_list<value_type> items )
        : map_type(items)
        {
        }

        const T& operator[]( const std::string& key ) const
        {
            static const T defaultValue = T();
            auto it = map_type::find(key);
            return it != map_type::end() ? it->second : defaultValue;
        }

        bool contains( const std::string& key ) const
        {
            return map_type::find(key) != map_type::end();
        }
    };


    template<typename T>
    std::ostream& operator<<( std::ostream& os, const Dictionary<T>& dict )
    {
        for ( auto it = dict.begin(); it != dict.end(); ++it )
        {
            if ( it != dict.begin() )
            {
                os << ", ";
            }
            os << it->first << " = " << *it->second;
        }
        return os;
    }

}   // namespace nnef


#endif
