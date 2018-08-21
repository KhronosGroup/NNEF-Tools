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
#include <iostream>
#include <string>


namespace nnef
{

    struct TensorHeader
    {
        enum { MaxRank = 8 };

        enum QuantCode { Float = 0x00, Integer = 0x01, Linear = 0x10, Logarithmic = 0x11 };

        uint8_t magic[2];
        uint8_t version[2];
        uint32_t data_length;
        uint32_t rank;
        uint32_t extents[MaxRank];
        uint32_t bits_per_item;
        uint32_t quant_code;
        uint8_t  quant_params[76];
    };


    template<typename T, typename U = float>
    inline void fill_tensor_header( TensorHeader& header, const size_t version[2], const size_t rank, const T* extents, const size_t bits_per_item,
                                   const TensorHeader::QuantCode quant_code = TensorHeader::Float, const std::vector<U>& quant_params = {} )
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
        header.quant_code = quant_code;

        std::copy_n(extents, rank, header.extents);

        if ( sizeof(U) * quant_params.size() > 32 )
        {
            throw Error("quantization parameters exceed maximum possible length of 32 bytes (found %d btyes)", (int)(sizeof(U) * quant_params.size()));
        }

        std::copy(quant_params.begin(), quant_params.end(), (U*)header.quant_params);
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
            throw Error("tensor rank %d exceeds maximum possible value (%d)", (int)header.rank, (int)TensorHeader::MaxRank);
        }

        const uint32_t item_count = std::accumulate(header.extents, header.extents + header.rank, (uint32_t)1, std::multiplies<uint32_t>());
        if ( header.data_length != (item_count * header.bits_per_item + 7) / 8 )
        {
            throw Error("data length is not compatible with extents and bits per item");
        }

        if ( (header.quant_code & 0xffff0000) == 0 )     // Khronos-defined item type
        {
            const uint32_t code = (header.quant_code & 0x0000ffff);

            switch ( code )
            {
                case TensorHeader::Float:
                {
                    if ( header.bits_per_item != 16 && header.bits_per_item != 32 && header.bits_per_item != 64 )
                    {
                        throw Error("invalid bits per item for float item type: %d", (int)header.bits_per_item);
                    }
                }
                case TensorHeader::Integer:
                case TensorHeader::Linear:
                case TensorHeader::Logarithmic:
                {
                    if ( header.bits_per_item > 64 )
                    {
                        throw Error("invalid bits per item for integer item type: %d", (int)header.bits_per_item);
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

}   // namespace nnef


#endif
