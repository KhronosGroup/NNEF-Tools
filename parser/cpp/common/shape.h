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

#ifndef _NNEF_SHAPE_H_
#define _NNEF_SHAPE_H_

#include <array>
#include <numeric>
#include <functional>


namespace nnef
{

    class Shape
    {
    public:
        
        enum { MaxRank = 8 };

        typedef int extent_type;
        
    public:
        
        Shape( const extent_type filler = 1 )
        {
            _items.fill(filler);
        }

        extent_type* extents()
        {
            return _items.data();
        }

        const extent_type* extents() const
        {
            return _items.data();
        }

        extent_type& operator[]( const size_t i )
        {
            return _items[i];
        }

        extent_type operator[]( const size_t i ) const
        {
            return _items[i];
        }

        bool operator==( const Shape& other ) const
        {
            return _items == other._items;
        }

        bool operator!=( const Shape& other ) const
        {
            return _items != other._items;
        }

        std::string toString() const
        {
            std::string str;

            auto rank = std::max(Shape::rank(), (size_t)2);

            str += '[';
            for ( size_t i = 0; i < rank; ++i )
            {
                if ( i )
                {
                    str += ',';
                }
                str += std::to_string(_items[i]);
            }
            str += ']';

            return str;
        }

        size_t volume() const
        {
            return std::accumulate(_items.begin(), _items.end(), (size_t)1, std::multiplies<size_t>());
        }

        size_t rank() const
        {
            size_t rank = MaxRank;
            while ( rank && _items[rank-1] == 1 )
            {
                --rank;
            }
            return rank;
        }

    public:

        static const Shape& undefined()
        {
            static const Shape shape(0);
            return shape;
        }
        
        static const Shape& singleton()
        {
            static const Shape shape(1);
            return shape;
        }

    private:

        std::array<extent_type,MaxRank> _items;
    };


    inline std::ostream& operator<<( std::ostream& os, const Shape& shape )
    {
        auto rank = std::max(shape.rank(), (size_t)2);

        os << '[';
        for ( size_t i = 0; i < rank; ++i )
        {
            if ( i )
            {
                os << ',';
            }
            os << shape[i];
        }
        os << ']';

        return os;
    }
    
}   // namespace nnef


#endif
