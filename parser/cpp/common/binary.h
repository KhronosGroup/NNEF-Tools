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

#include "shape.h"
#include <iostream>
#include <string>


namespace nnef
{

    enum class TensorDtype : uint8_t { Float, Quantized, Signed, Unsigned, DtypeCount };


    inline void write_tensor_version( std::ostream& os, const size_t major, const size_t minor )
    {
        const char* magic = "N\xEF";
        os.write((char*)magic, 2);

        uint8_t major8 = (uint8_t)major;
        uint8_t minor8 = (uint8_t)minor;

        os.write((char*)&major8, sizeof(major8));
        os.write((char*)&minor8, sizeof(minor8));
    }

    inline bool read_tensor_version( std::istream& is, size_t& major, size_t& minor )
    {
        unsigned char magic[2];
        is.read((char*)magic, 2);

        uint8_t major8, minor8;

        is.read((char*)&major8, sizeof(major8));
        is.read((char*)&minor8, sizeof(minor8));

        major = major8;
        minor = minor8;

        return magic[0] == 'N' && magic[1] == 0xEF;
    }


    template<typename E>
    size_t read_tensor_extents( std::istream& is, E extents[], size_t max_rank )
    {
        uint32_t rank, extent;
        is.read((char*)&rank, sizeof(rank));

        if ( rank > max_rank )
        {
            return 0;
        }

        for ( size_t i = 0; i < rank; ++i )
        {
            is.read((char*)&extent, sizeof(extent));
            extents[i] = (E)extent;
        }

        return rank;
    }

    template<typename E>
    void write_tensor_extents( std::ostream& os, const size_t rank, const E extents[] )
    {
        uint32_t dims = (uint32_t)rank;
        os.write((char*)&dims, sizeof(dims));

        for ( size_t i = 0; i < rank; ++i )
        {
            uint32_t extent = (uint32_t)extents[i];
            os.write((char*)&extent, sizeof(extent));
        }
    }
    
    inline void write_tensor_dtype( std::ostream& os, const size_t bits, const std::string& quantization )
    {
        uint8_t dtype = quantization.empty() ? (uint8_t)TensorDtype::Float : (uint8_t)TensorDtype::Quantized;
        uint8_t bits8 = (uint8_t)bits;
        uint16_t qlen = (uint16_t)quantization.length();

        os.write((char*)&dtype, sizeof(dtype));
        os.write((char*)&bits8, sizeof(bits8));
        os.write((char*)&qlen, sizeof(qlen));
        os.write((char*)quantization.data(), quantization.length());
    }

    inline TensorDtype read_tensor_dtype( std::istream& is, size_t& bits, std::string& quantization )
    {
        uint8_t dtype, bits8;
        is.read((char*)&dtype, sizeof(dtype));
        is.read((char*)&bits8, sizeof(bits8));

        bits = bits8;

        uint16_t qlen;
        is.read((char*)&qlen, sizeof(qlen));

        quantization.resize(qlen);
        is.read((char*)quantization.data(), qlen);

        return (TensorDtype)dtype;
    }

    template<typename E>
    size_t tensor_data_bytes( const size_t rank, E extents[], const size_t bits )
    {
        size_t count = bits;
        for ( size_t i = 0; i < rank; ++i )
        {
            count *= extents[i];
        }
        return (count + 7) / 8;
    }

    inline size_t tensor_header_length( const size_t rank, const size_t qlen )
    {
        return 4 + 4 + (rank + 1) * 4 + 4 + qlen;
    }

    inline void write_header_length( std::ostream& os, size_t length )
    {
        uint32_t length32 = (uint32_t)length;
        os.write((char*)&length32, sizeof(length32));
    }

    inline size_t read_header_length( std::istream& is )
    {
        uint32_t length;
        is.read((char*)&length, sizeof(length));
        return (size_t)length;
    }


    struct TensorHeader
    {
        struct Version
        {
            size_t major, minor;
        };

        Version version;
        size_t length;
        Shape shape;
        TensorDtype dtype;
        size_t bits;
        std::string quantization;
    };


    inline void write_tensor_header( std::ostream& os, const TensorHeader& header )
    {
        write_tensor_version(os, header.version.major, header.version.minor);
        write_header_length(os, tensor_header_length(header.shape.rank(), header.quantization.length()));
        write_tensor_extents(os, header.shape.rank(), header.shape.extents());
        write_tensor_dtype(os, header.bits, header.quantization);
    }

    inline bool read_tensor_header( std::istream& is, TensorHeader& header )
    {
        if ( !read_tensor_version(is, header.version.major, header.version.minor) )
        {
            return false;
        }
        header.length = read_header_length(is);

        auto rank = read_tensor_extents(is, header.shape.extents(), Shape::MaxRank);
        if ( !rank )
        {
            return false;
        }
        for ( size_t i = rank; i < Shape::MaxRank; ++i )
        {
            header.shape[i] = 1;
        }
        header.dtype = read_tensor_dtype(is, header.bits, header.quantization);
        if ( !(header.dtype < TensorDtype::DtypeCount) )
        {
            return false;
        }

        if ( header.length != tensor_header_length(rank, header.quantization.length()) )
        {
            return false;
        }

        return (bool)is;
    }

}   // namespace nnef


#endif
