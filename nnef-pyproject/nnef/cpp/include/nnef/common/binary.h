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

#ifndef _NNEF_BINARY_H_
#define _NNEF_BINARY_H_

#include "error.h"
#include <functional>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <string>


namespace nnef
{

    struct TensorHeader
    {
        enum { MaxRank = 8 };

        enum ItemType { Float, Uint, Quint, Qint, Int, Bool };

        uint8_t magic[2];
        uint8_t version[2];
        uint32_t data_length;
        uint32_t rank;
        uint32_t extents[MaxRank];
        uint32_t bits_per_item;
        uint32_t item_type;
        uint32_t reserved[19];
    };


    template<typename In, typename Out>
    void copy_and_cast_n( In* input, size_t n, Out* output )
    {
        for ( size_t i = 0; i < n; ++i )
        {
            *output++ = (Out)*input++;
        }
    }


    template<typename T>
    inline void fill_tensor_header( TensorHeader& header, const size_t version[2], const size_t rank, const T* extents,
                                   const size_t bits_per_item, const TensorHeader::ItemType item_type )
    {
        const char* magic = "N\xEF";

        std::fill_n((uint8_t*)&header, sizeof(header), (uint8_t)0);

        header.magic[0] = (uint8_t)magic[0];
        header.magic[1] = (uint8_t)magic[1];

        header.version[0] = (uint8_t)version[0];
        header.version[1] = (uint8_t)version[1];

        if ( rank > TensorHeader::MaxRank )
        {
            throw Error("tensor rank %d exceeds maximum possible value (%d)", (int)rank, (int)TensorHeader::MaxRank);
        }

        const uint32_t item_count = std::accumulate(extents, extents + rank, (uint32_t)1, std::multiplies<uint32_t>());
        header.data_length = (uint32_t)((item_count * bits_per_item + 7) / 8);
        header.bits_per_item = (uint32_t)bits_per_item;
        header.rank = (uint32_t)rank;
        header.item_type = item_type;

        std::copy_n(extents, rank, header.extents);
    }

    inline void validate_tensor_header( const TensorHeader& header )
    {
        if ( header.magic[0] != 'N' || header.magic[1] != 0xEF )
        {
            throw Error("invliad magic number in tensor binary");
        }
        if ( header.version[0] != 1 || header.version[1] != 0 )
        {
            throw Error("unknown version number %d.%d", (int)header.version[0], (int)header.version[1]);
        }
        if ( header.rank > TensorHeader::MaxRank )
        {
            throw Error("tensor rank %d exceeds maximum allowed rank (%d)", (int)header.rank, (int)TensorHeader::MaxRank);
        }

        const size_t item_count = std::accumulate(header.extents, header.extents + header.rank, (size_t)1, std::multiplies<size_t>());
        if ( (size_t)header.data_length != (item_count * header.bits_per_item + 7) / 8 )
        {
            throw Error("data length is not compatible with extents and bits per item");
        }

        if ( (header.item_type & 0xffff0000) == 0 )     // Khronos-defined item type
        {
            const uint32_t code = (header.item_type & 0x0000ffff);

            switch ( code )
            {
                case TensorHeader::Float:
                {
                    if ( header.bits_per_item != 16 && header.bits_per_item != 32 && header.bits_per_item != 64 )
                    {
                        throw Error("invalid bits per item for float item type: %d", (int)header.bits_per_item);
                    }
                    break;
                }
                case TensorHeader::Int:
                case TensorHeader::Uint:
                case TensorHeader::Quint:
                case TensorHeader::Qint:
                {
                    if ( header.bits_per_item > 64 )
                    {
                        throw Error("invalid bits per item for integer item type: %d", (int)header.bits_per_item);
                    }
                    break;
                }
                case TensorHeader::Bool:
                {
                    if ( header.bits_per_item != 1 && header.bits_per_item != 8 )
                    {
                        throw Error("invalid bits per item for bool item type: %d", (int)header.bits_per_item);
                    }
                    break;
                }
                default:
                {
                    throw Error("unkown Khronos-defined item type code: %x", (int)code);
                }
            }
        }
    }

    inline void pack_bits( const size_t n, const bool* data, char* bytes )
    {
        for ( size_t i = 0; i < n; ++i )
        {
            bytes[i / 8] |= (data[i] << (7 - (i % 8)));
        }
    }

    inline void unpack_bits( const size_t n, const char* bytes, bool* data )
    {
        for ( size_t i = 0; i < n; ++i )
        {
            data[i] = (bytes[i / 8] >> (7 - (i % 8))) & 0x01;
        }
    }
    
    inline void from_bytes( const char* bytes, const size_t count, const size_t bits_per_item, float* data )
    {
        if ( bits_per_item == 32 )
        {
            copy_and_cast_n((const float*)bytes, count, data);
        }
        else if ( bits_per_item == 64 )
        {
            copy_and_cast_n((const double*)bytes, count, data);
        }
        else
        {
            throw std::runtime_error("cannot load float data of " + std::to_string(bits_per_item) + " bits per item");
        }
    }

    inline void from_bytes( const char* bytes, const size_t count, const size_t bits_per_item, int* data, const bool is_signed )
    {
        if ( bits_per_item == 8 )
        {
            if ( is_signed )
            {
                copy_and_cast_n((const int8_t*)bytes, count, data);
            }
            else
            {
                copy_and_cast_n((const uint8_t*)bytes, count, data);
            }
        }
        else if ( bits_per_item == 16 )
        {
            if ( is_signed )
            {
                copy_and_cast_n((const int16_t*)bytes, count, data);
            }
            else
            {
                copy_and_cast_n((const uint16_t*)bytes, count, data);
            }
        }
        else if ( bits_per_item == 32 )
        {
            if ( is_signed )
            {
                copy_and_cast_n((const int32_t*)bytes, count, data);
            }
            else
            {
                copy_and_cast_n((const uint32_t*)bytes, count, data);
            }
        }
        else if ( bits_per_item == 64 )
        {
            if ( is_signed )
            {
                copy_and_cast_n((const int64_t*)bytes, count, data);
            }
            else
            {
                copy_and_cast_n((const uint64_t*)bytes, count, data);
            }
        }
        else
        {
            throw std::runtime_error("cannot load int data of " + std::to_string(bits_per_item) + " bits per item");
        }
    }

    inline void from_bytes( const char* bytes, const size_t count, const size_t bits_per_item, bool* data )
    {
        if ( bits_per_item == 1 )
        {
            unpack_bits(count, bytes, data);
        }
        else if ( bits_per_item == 8 )
        {
            copy_and_cast_n((const int8_t*)bytes, count, data);
        }
        else
        {
            throw std::runtime_error("cannot load bool data of " + std::to_string(bits_per_item) + " bits per item");
        }
    }

    inline void to_bytes( const float* data, const size_t count, char* bytes )
    {
        copy_and_cast_n(data, count, (float*)bytes);
    }

    inline void to_bytes( const int* data, const size_t count, char* bytes, const bool as_signed )
    {
        if ( as_signed )
        {
            copy_and_cast_n(data, count, (int32_t*)bytes);
        }
        else
        {
            copy_and_cast_n(data, count, (uint32_t*)bytes);
        }
    }

    inline void to_bytes( const bool* data, const size_t count, char* bytes )
    {
        pack_bits(count, data, bytes);
    }

}   // namespace nnef


#endif
