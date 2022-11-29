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

#ifndef _NNEF_SHAPES_H_
#define _NNEF_SHAPES_H_

#include <vector>
#include <string>
#include <numeric>
#include <iostream>
#include <functional>
#include <algorithm>
#include <limits>
#include "value.h"
#include "error.h"


namespace nnef
{

    typedef std::vector<int> Shape;


    inline std::string to_string( const Shape& shape )
    {
        std::string str;
        
        str += '[';
        for ( size_t i = 0; i < shape.size(); ++i )
        {
            if ( i )
            {
                str += ',';
            }
            str += std::to_string(shape[i]);
        }
        str += ']';
        
        return str;
    }

    inline Shape make_shape( const Value& arg, const size_t offset = 0 )
    {
        Shape shape = Shape(offset + arg.size(), 1);
        for ( size_t i = 0; i < arg.size(); ++i )
        {
            shape[i + offset] = arg[i].integer();
        }
        return shape;
    }

    inline Shape make_padding_shape( const Value& arg, const size_t offset = 0 )
    {
        Shape padding(offset + arg.size(), 0);
        for ( size_t i = 0; i < arg.size(); ++i )
        {
            padding[i + offset] = arg[i][0].integer() + arg[i][1].integer();
        }
        return padding;
    }

    inline size_t volume_of( const Shape& shape )
    {
        return std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
    }
    
    inline size_t volume_of( const Shape& shape, const size_t offset, const size_t length )
    {
        return std::accumulate(shape.begin() + offset, shape.begin() + offset + length, (size_t)1, std::multiplies<size_t>());
    }
    
    inline bool broadcastable( const Shape& xShape, const Shape& yShape, const size_t n )
    {
        for ( size_t i = 0; i < n; ++i )
        {
            auto xi = i < xShape.size() ? xShape[i] : 1;
            auto yi = i < yShape.size() ? yShape[i] : 1;
            if ( !(xi == yi || xi == 1) )
            {
                return false;
            }
        }
        return true;
    }
    
    inline bool broadcastable( const Shape& xShape, const Shape& yShape )
    {
        const size_t rank = std::max(xShape.size(), yShape.size());
        return broadcastable(xShape, yShape, rank);
    }

    inline bool broadcast_compatible( const Shape& xShape, const Shape& yShape, const size_t n )
    {
        for ( size_t i = 0; i < n; ++i )
        {
            auto xi = i < xShape.size() ? xShape[i] : 1;
            auto yi = i < yShape.size() ? yShape[i] : 1;
            if ( !(xi == yi || xi == 1 || yi == 1) )
            {
                return false;
            }
        }
        return true;
    }

    inline bool broadcast_compatible( const Shape& xShape, const Shape& yShape )
    {
        const size_t rank = std::max(xShape.size(), yShape.size());
        return broadcast_compatible(xShape, yShape, rank);
    }

    inline bool axes_compatible_with_rank( const Value& axes, const size_t rank )
    {
        for ( size_t i = 0; i < axes.size(); ++i )
        {
            auto axis = axes[i].integer();
            if ( axis < 0 || axis >= (Value::integer_t)rank )
            {
                return false;
            }
        }
        return true;
    }

    inline bool contains_axis( const Value& axes, const size_t axis )
    {
        for ( size_t i = 0; i < axes.size(); ++i )
        {
            if ( axes[i].integer() == (Value::integer_t)axis )
            {
                return true;
            }
        }
        return false;
    }
    
    template <typename T>
    inline int sign( T val )
    {
        return (T(0) < val) - (val < T(0));
    }

    inline int ceil_div( int x, int y )
    {
        return y > 0 ? (x + y - 1) / y : (x + y + 1) / y;
    }
    
    template<typename T>
    inline T downsize( const T input, const T size, const T padding, const T stride, const T dilation )
    {
        const T window = 1 + (size - 1) * dilation;
        return sign(input) * ((std::abs(input) + padding - window) / stride + 1);
    }
    
    template<typename T>
    inline T downsize( const T input, const T stride )
    {
        return sign(input) * ((std::abs(input) + stride - 1) / stride);
    }
    
    template<typename T>
    inline T upsize( const T input, const T size, const T padding, const T stride, const T dilation )
    {
        const T window = 1 + (size - 1) * dilation;
        return sign(input) * ((std::abs(input) - 1) * stride + window - padding);
    }
    
    template<typename T>
    inline T upsize( const T input, const T stride )
    {
        return input * stride;
    }
    
    
    template<typename... Args>
    inline void check( bool condition, const char* message, Args&&... args )
    {
        if ( !condition )
        {
            throw std::logic_error(Error::formatString(message, std::forward<Args>(args)...));
        }
    }

    inline void check_axis_compatible_with_rank( const Value& axis, const size_t rank )
    {
        check(axis.integer() >= 0 && axis.integer() < (Value::integer_t)rank,
                "axis must be in range [0,%d); found %d", (int)rank, (int)axis.integer());
    }

    inline void check_axes_compatible_with_rank( const Value& axes, const size_t rank )
    {
        check(axes_compatible_with_rank(axes, rank),
                "axes must be in range [0,%d); found %s", (int)rank, axes.toString().c_str());
    }

    inline void check_range( const char* name, const Value& value, const Value::integer_t min )
    {
        if ( value.kind() == Value::Array || value.kind() == Value::Tuple )
        {
            for ( size_t i = 0; i < value.size(); ++i )
            {
                check_range(name, value[i], min);
            }
        }
        else if ( value.kind() == Value::Integer )
        {
            check(value.integer() >= min, "'%s' must be >= %d (found %d)", name, min, (int)value.integer());
        }
    }

    inline void check_rank( const char* name, const Value& value, const size_t rank )
    {
        check(value.size() == rank, "length of array '%s' must be %d to match rank of operation (found %d)",
                name, (int)rank, (int)value.size());
    }

    
    inline Shape broadcast_shape( const Shape& xShape, const Shape& yShape, const size_t n )
    {
        const size_t rank = std::max(xShape.size(), yShape.size());
        Shape zShape(rank);
        
        for ( size_t i = 0; i < n; ++i )
        {
            auto xi = i < xShape.size() ? xShape[i] : 1;
            auto yi = i < yShape.size() ? yShape[i] : 1;
            zShape[i] = std::max(xi, yi);
        }
        return zShape;
    }
    
    inline Shape broadcast_shape( const Shape& xShape, const Shape& yShape )
    {
        const size_t rank = std::max(xShape.size(), yShape.size());
        return broadcast_shape(xShape, yShape, rank);
    }

    inline Shape nullary_shape( const Value& shape )
    {
        return make_shape(shape);
    }

    inline Shape constant_shape( const Value& shape, const Value& value )
    {
        auto result = nullary_shape(shape);
        check(value.size() == volume_of(result) || value.size() == 1,
                "shape volume (%d) does not match number of values (%d)", (int)volume_of(result), (int)value.size());
        
        return result;
    }

    inline Shape unary_shape( const Shape& shape )
    {
        return shape;
    }

    inline Shape binary_shape( const Shape& shape1, const Shape& shape2 )
    {
        check(broadcast_compatible(shape1, shape2),
              "incompatible tensor shapes for broadcasting (%s vs %s)",
              to_string(shape1).c_str(), to_string(shape2).c_str());
        
        return broadcast_shape(shape1, shape2);
    }
    
    inline Shape asymmetric_binary_shape( const Shape& shape1, const Shape& shape2 )
    {
        check(broadcastable(shape2, shape1),
              "cannot broadcast second argument shape (%s) to first argument shape (%s)",
              to_string(shape2).c_str(), to_string(shape1).c_str());
        
        return shape1;
    }

    inline Shape ternary_shape( const Shape& shape1, const Shape& shape2, const Shape& shape3 )
    {
        return binary_shape(binary_shape(shape1, shape2), shape3);
    }

    inline Shape reduce_shape( const Shape& input, const Value& axes )
    {
        check_axes_compatible_with_rank(axes, input.size());
        
        Shape output = input;
        for ( size_t i = 0; i < axes.size(); ++i )
        {
            auto axis = axes[i].integer();
            output[axis] = 1;
        }
        
        return output;
    }
    
    inline Shape downsample_shape( const Shape& input, const Value& factor )
    {
        for ( size_t i = 0; i < factor.size(); ++i )
        {
            auto scale = factor[i].integer();
            check(input[i+2] % scale == 0, "input extent (%d) must be divisible by factor (%d)", (int)input[i+2], (int)scale);
        }
        
        Shape output = input;
        for ( size_t i = 0; i < factor.size(); ++i )
        {
            output[i+2] /= factor[i].integer();
        }
        return output;
    }
    
    inline Shape upsample_shape( const Shape& input, const Value& factor )
    {
        check_rank("factor", factor, input.size() - 2);
        
        Shape output = input;
        for ( size_t i = 0; i < factor.size(); ++i )
        {
            output[i+2] *= factor[i].integer();
        }
        return output;
    }
    
    inline Shape downsize_shape( const Shape& input, const Shape& kernel, const Shape& padding, const Shape& stride, const Shape& dilation,
                                const size_t offset )
    {
        Shape output(input.size());
        for ( size_t i = offset; i < output.size(); ++i )
        {
            output[i] = padding.size() ? downsize(input[i], kernel[i], padding[i], stride[i], dilation[i]) : downsize(input[i], stride[i]);
        }
        return output;
    }
    
    inline Shape upsize_shape( const Shape& input, const Shape& kernel, const Shape& padding, const Shape& stride, const Shape& dilation,
                              const size_t offset )
    {
        Shape output(input.size());
        for ( size_t i = offset; i < output.size(); ++i )
        {
            output[i] = padding.size() ? upsize(input[i], kernel[i], padding[i], stride[i], dilation[i]) : upsize(input[i], stride[i]);
        }
        return output;
    }

    inline Shape conv_like_shape( const Shape& input, const Shape& filter, const Shape& bias,
                                 const Value& /*border*/, const Value& padding, const Value& stride, const Value& dilation,
                                 const Value& groups, const Value& output_shape, const bool transposed )
    {
        auto rank = input.size();
        
        if ( padding.size() )
        {
            check_rank("padding", padding, rank - 2);
        }
        if ( stride.size() )
        {
            check_rank("stride", stride, rank - 2);
        }
        if ( dilation.size() )
        {
            check_rank("dilation", dilation, rank - 2);
        }
        
        check_range("stride", stride, 1);
        check_range("dilation", dilation, 1);
        check_range("groups", groups, 0);
        
        auto groupCount = groups.integer() != 0 ? groups.integer() : transposed && output_shape && output_shape.size() ? output_shape[1].integer() : input[1];
        
        if ( transposed )
        {
            check(input[1] == filter[0], "filter batch (%d) does not match input channels (%d)",
                    (int)filter[0], (int)input[1]);
        }
        else
        {
            check(input[1] == filter[1] * groupCount, "filter channels (%d) does not match input channels (%d) times groups (%d)",
                    (int)filter[1], (int)input[1], (int)groupCount);
        }
        
        check(filter[0] % groupCount == 0, "filter batch (%d) must be divisible by groups (%d)", (int)filter[0], (int)groupCount);
        check(bias.size() <= 2, "bias shape must be of rank at most 2, found %d", (int)bias.size());
        
        if ( bias.size() == 2 )
        {
            check(bias[0] == 1, "bias shape must be singular for the batch dimension");
        }
        if ( bias.size() > 0 )
        {
            auto channels = transposed ? filter[1] * groupCount : filter[0];
            check(bias.back() == channels || bias.back() == 1, "bias channels (%d) does not match output channels (%d)",
                    (int)bias.back(), (int)channels);
        }
        
        const Shape strideShape = make_shape(stride, stride.size() ? 2 : rank);
        const Shape dilationShape = make_shape(dilation, dilation.size() ? 2 : rank);
        const Shape paddingShape = padding.size() ? make_padding_shape(padding, 2) : Shape();
        
        if ( output_shape && output_shape.size() )
        {
            const Shape outputShape = make_shape(output_shape);
            
            check_rank("output_shape", output_shape, rank);
            check_range("output_shape", output_shape, 1);
            
            check(outputShape[0] == input[0], "output batch (%d) does not match input batch (%d)", (int)outputShape[0], (int)input[0]);
            check(outputShape[1] == filter[1] * groupCount, "output channels (%d) does not match filter channels (%d) times groups (%d)",
                    (int)outputShape[1], (int)filter[1], (int)groupCount);
            
            Shape expected = downsize_shape(outputShape, filter, paddingShape, strideShape, dilationShape, 2);
            std::copy_n(input.begin(), 2, expected.begin());
            
            check(input == expected, "expected input shape %s derived from output shape is incompatible with actual input shape %s",
                    to_string(expected).c_str(), to_string(input).c_str());
            
            return outputShape;
        }
        
        if ( transposed )
        {
            auto output = upsize_shape(input, filter, paddingShape, strideShape, dilationShape, 2);
            output[0] = input[0];
            output[1] = filter[1] * groupCount;
            return output;
        }
        else
        {
            auto output = downsize_shape(input, filter, paddingShape, strideShape, dilationShape, 2);
            output[0] = input[0];
            output[1] = filter[0];
            return output;
        }
    }
    
    inline Shape separable_conv_like_shape( const Shape& input, const Shape& plane_filter, const Shape& point_filter, const Shape& bias,
                                           const Value& border, const Value& padding, const Value& stride, const Value& dilation,
                                           const Value& groups, const Value& output_shape, const bool transposed )
    {
        for ( size_t i = 2; i < point_filter.size(); ++i )
        {
            check(point_filter[i] == 1, "point filter must have singular extents in spatial dimensions");
        }
        check(point_filter[1] == plane_filter[0], "channel dimension of point filter must equal batch dimension of plane filter");
        check(plane_filter[1] == 1, "channel dimension of plane filter must be singular");
        
        Shape filter = plane_filter;
        filter[0] = point_filter[0];
        filter[1] = transposed ? point_filter[1] : input[1];
        
        return conv_like_shape(input, filter, bias, border, padding, stride, dilation, groups, output_shape, transposed);
    }
    
    inline Shape conv_shape( const Shape& input, const Shape& filter, const Shape& bias,
                            const Value& border, const Value& padding, const Value& stride, const Value& dilation,
                            const Value& groups )
    {
        return conv_like_shape(input, filter, bias, border, padding, stride, dilation, groups, Value::none(), false);
    }

    inline Shape deconv_shape( const Shape& input, const Shape& filter, const Shape& bias,
                              const Value& border, const Value& padding, const Value& stride, const Value& dilation,
                              const Value& output_shape, const Value& groups )
    {
        return conv_like_shape(input, filter, bias, border, padding, stride, dilation, groups, output_shape, true);
    }
    
    inline Shape separable_conv_shape( const Shape& input, const Shape& plane_filter, const Shape& point_filter, const Shape& bias,
                                      const Value& border, const Value& padding, const Value& stride, const Value& dilation,
                                      const Value& groups )
    {
        return separable_conv_like_shape(input, plane_filter, point_filter, bias, border, padding, stride, dilation, groups, Value::none(), false);
    }
    
    inline Shape separable_deconv_shape( const Shape& input, const Shape& plane_filter, const Shape& point_filter, const Shape& bias,
                                        const Value& border, const Value& padding, const Value& stride, const Value& dilation,
                                        const Value& output_shape, const Value& groups )
    {
        return separable_conv_like_shape(input, plane_filter, point_filter, bias, border, padding, stride, dilation, groups, output_shape, true);
    }
    
    inline Shape pool_like_shape( const Shape& input, const Value& size, const Value& /*border*/, const Value& padding,
                                 const Value& stride, const Value& dilation, const Value& output_shape, const bool transposed )
    {
        auto rank = input.size();
        
        check_rank("size", size, rank);
        if ( padding.size() )
        {
            check_rank("padding", padding, rank);
        }
        if ( stride.size() )
        {
            check_rank("stride", stride, rank);
        }
        if ( dilation.size() )
        {
            check_rank("dilation", dilation, rank);
        }
        
        check_range("size", size, 1);
        check_range("stride", stride, 1);
        check_range("dilation", dilation, 1);
        
        auto kernelShape = make_shape(size);
        auto strideShape = make_shape(stride, stride.size() ? 0 : rank);
        auto dilationShape = make_shape(dilation, dilation.size() ? 0 : rank);
        auto paddingShape = padding.size() ? make_padding_shape(padding) : Shape();
        
        if ( output_shape && output_shape.size() )
        {
            const Shape outputShape = make_shape(output_shape);
            
            check_rank("output_shape", output_shape, rank);
            check_range("output_shape", output_shape, 1);
            
            const Shape expected = downsize_shape(outputShape, kernelShape, paddingShape, strideShape, dilationShape, 0);
            check(input == expected, "expected input shape %s derived from output shape is incompatible with actual input shape %s",
                    to_string(expected).c_str(), to_string(input).c_str());
            
            return outputShape;
        }
        
        if ( transposed )
        {
            return upsize_shape(input, kernelShape, paddingShape, strideShape, dilationShape, 0);
        }
        else
        {
            return downsize_shape(input, kernelShape, paddingShape, strideShape, dilationShape, 0);
        }
    }
    
    inline Shape sample_like_shape( const Shape& input, const Shape& index, const Value& size, const Value& border, const Value& padding,
                                   const Value& stride, const Value& dilation, const Value& output_shape, const bool transposed )
    {
        check(index == input, "index shape incompatible with input shape (%s vs %s)",
                to_string(index).c_str(), to_string(input).c_str());
        return pool_like_shape(input, size, border, padding, stride, dilation, output_shape, transposed);
    }

    inline Shape pool_shape( const Shape& input, const Value& size, const Value& border, const Value& padding,
                            const Value& stride, const Value& dilation )
    {
        return pool_like_shape(input, size, border, padding, stride, dilation, Value::none(), false);
    }

    inline Shape unpool_shape( const Shape& input, const Value& size, const Value& border, const Value& padding,
                              const Value& stride, const Value& dilation, const Value& output_shape )
    {
        return pool_like_shape(input, size, border, padding, stride, dilation, output_shape, true);
    }
    
    inline Shape sample_shape( const Shape& input, const Shape& index, const Value& size, const Value& border, const Value& padding,
                              const Value& stride, const Value& dilation )
    {
        return sample_like_shape(input, index, size, border, padding, stride, dilation, Value::none(), false);
    }
    
    inline Shape desample_shape( const Shape& input, const Shape& index, const Value& size, const Value& border, const Value& padding,
                                const Value& stride, const Value& dilation, const Value& output_shape )
    {
        return sample_like_shape(input, index, size, border, padding, stride, dilation, output_shape, true);
    }

    inline Shape normalize_shape_axes( const Shape& input, const Value& axes )
    {
        check_axes_compatible_with_rank(axes, input.size());
        
        return input;
    }

    inline Shape normalize_shape_size( const Shape& input, const Value& size )
    {
        check_rank("size", size, input.size());
        check_range("size", size, 1);
        
        return input;
    }

    inline Shape batchnorm_shape( const Shape& input, const Shape& mean, const Shape& variance, const Shape& offset, const Shape& scale, const Value& /*epsilon*/ )
    {
        check(broadcastable(mean, input), "cannot broadcast 'mean' shape (%s) to 'input' shape (%s)",
              to_string(mean).c_str(), to_string(input).c_str());
        check(broadcastable(variance, input), "cannot broadcast 'variance' shape (%s) to 'input' shape (%s)",
              to_string(variance).c_str(), to_string(input).c_str());
        check(broadcastable(offset, input), "cannot broadcast 'offset' shape (%s) to 'input' shape (%s)",
              to_string(offset).c_str(), to_string(input).c_str());
        check(broadcastable(scale, input), "cannot broadcast 'scale' shape (%s) to 'input' shape (%s)",
              to_string(scale).c_str(), to_string(input).c_str());
        
        return input;
    }

    inline Shape roi_shape( const Shape& input, const Shape& rois, const Shape& index, const Value& size )
    {
        check_rank("output_size", size, input.size() - 2);
        check_range("output_size", size, 1);
        
        check(rois.size() == 2, "'rois' must be a rank-2 tensor");
        check(index.size() == 1, "'batch_index' must be a rank-1 tensor");
        check(rois[1] == 4, "rois must be of extent 4 along dimension 1 (found %d)", (int)rois[1]);
        check(index[0] == rois[0], "'batch_index' must be of same length as dimension 0 of rois; found (%d vs %d)", (int)index[0], (int)rois[0]);
        
        Shape output(input.size());
        output[0] = rois[0];
        output[1] = input[1];
        for ( size_t i = 0; i < size.size(); ++i )
        {
            output[i+2] = (Shape::value_type)size[i].integer();
        }
        return output;
    }

    inline Shape roi_shape_resample( const Shape& input, const Shape& rois, const Shape& index, const Value& size, const Value& rate )
    {
        check_rank("sampling_rate", rate, input.size() - 2);
        check_range("sampling_rate", rate, 1);
        
        return roi_shape(input, rois, index, size);
    }

    inline Shape reshape_shape( const Shape& input, const Value& shape, const Value& axis_start, const Value& axis_count )
    {
        check_axis_compatible_with_rank(axis_start, input.size() + 1);
        check_range("axis_count", axis_start, -1);
        
        const size_t offset = axis_start.integer();
        const size_t length = axis_count.integer() == -1 ? input.size() - axis_start.integer() : axis_count.integer();
        
        check(offset + length <= input.size(), "'axis_start' + 'axis_count' must be in range [0,%d], found %d",
              (int)input.size(), (int)(offset + length));
        
        Shape output(input.begin(), input.begin() + offset);
        
        size_t autoAxis = std::numeric_limits<size_t>::max();
        for ( size_t i = 0; i < shape.size(); ++i )
        {
            auto s = shape[i].integer();
            if ( s == 0 )
            {
                s = input[i + offset];
            }
            else if ( s == -1 )
            {
                check(autoAxis == std::numeric_limits<size_t>::max(), "shape may only contain at most one -1 value");
                
                s = 1;
                autoAxis = i + offset;
            }
            output.push_back(s);
        }
        
        output.insert(output.end(), input.begin() + offset + length, input.end());
        
        auto inputVolume = volume_of(input, offset, length);
        auto outputVolume = volume_of(output, offset, shape.size());
        
        if ( autoAxis != std::numeric_limits<size_t>::max() )
        {
            check(inputVolume % outputVolume == 0, "automatic output shape (%s) incompatible with input shape (%s)", (int)outputVolume, (int)inputVolume);
            
            output[autoAxis] = (Shape::value_type)(inputVolume / outputVolume);
        }
        else
        {
            check(inputVolume == outputVolume, "input volume (%d) does not equal output volume (%d)", (int)inputVolume, (int)outputVolume);
        }
        
        return output;
    }

    inline Shape transpose_shape( const Shape& input, const Value& axes )
    {
        std::vector<size_t> perm(axes.size());
        for ( size_t i = 0; i < axes.size(); ++i )
        {
            perm[i] = axes[i].integer();
        }
        
        std::sort(perm.begin(), perm.end());
        for ( size_t i = 0; i < perm.size(); ++i )
        {
            check(perm[i] == i, "'axes' array must contain a permutation of dimensions from 0 to %d-1", (int)perm.size());
        }
        
        Shape output = input;
        for ( size_t i = 0; i < axes.size(); ++i )
        {
            auto j = axes[i].integer();
            output[i] = input[j];
        }
        return output;
    }

    inline std::vector<Shape> split_shape( const Shape& value, const Value& axis, const Value& ratios )
    {
        check_axis_compatible_with_rank(axis, value.size());
        check_range("ratios", ratios, 1);
        
        auto idx = axis.integer();
        
        Value::integer_t total = 0;
        for ( size_t i = 0; i < ratios.size(); ++i )
        {
            total += ratios[i].integer();
        }
        
        check(value[idx] % total == 0, "sum of split ratios (%d) does not divide whole extent (%d)", (int)total, (int)value[idx]);
        
        const Value::integer_t unit = value[idx] / total;
        
        std::vector<Shape> values(ratios.size());
        for ( size_t i = 0; i < values.size(); ++i )
        {
            Shape item = value;
            item[idx] = unit * ratios[i].integer();
            
            values[i] = item;
        }
        return values;
    }

    inline Shape concat_shape( const std::vector<Shape>& valuesShape, const Value& axis )
    {
        check(valuesShape.size() != 0, "input array must be non-empty");
        
        Shape outputShape = valuesShape[0];
        
        check_axis_compatible_with_rank(axis, outputShape.size());
        
        const size_t idx = axis.integer();
        
        bool compatibleShape = true;
        for ( size_t i = 1; i < valuesShape.size(); ++i )
        {
            auto& partShape = valuesShape[i];
            if ( partShape.size() != outputShape.size() )
            {
                compatibleShape = false;
                break;
            }
            
            for ( size_t i = 0; i < outputShape.size(); ++i )
            {
                if ( i == idx )
                {
                    outputShape[i] += partShape[i];
                }
                else
                {
                    compatibleShape &= outputShape[i] == partShape[i];
                }
            }
        }
        
        check(compatibleShape, "incompatible tensor shapes in input array");
        
        return outputShape;
    }

    inline Shape slice_shape( const Shape& input, const Value& axes, const Value& begin, const Value& end, const Value& stride )
    {
        check(begin.size() == axes.size() && end.size() == axes.size(), "'axes', 'begin' and 'end' arrays must have the same length");
        check(stride.size() == 0 || stride.size() == axes.size(), "'stride' must have the same length as 'axes'");
        
        check_axes_compatible_with_rank(axes, input.size());
        
        Shape output = input;
        for ( size_t i = 0; i < axes.size(); ++i )
        {
            auto axis = axes[i].integer();
            auto extent = input[axis];
            auto str = stride.size() ? stride[i].integer() : 1;
            
            auto first = begin[i].integer();
            if ( first < 0 )
            {
                first += extent;
            }
            
            auto last = end[i].integer();
            if ( last < 0 )
            {
                last += extent;
            }
            else if ( last == 0 && str == 1 )
            {
                last = extent;
            }
            
            if ( first < 0 )
            {
                first = -1;
            }
            if ( first > extent )
            {
                first = extent;
            }
            if ( last < 0 )
            {
                last = -1;
            }
            if ( last > extent )
            {
                last = extent;
            }
            
            check(str != 0, "'stride' must be non-zero");
            
            if ( str > 0 )
            {
                check(first >= 0 && last >= first, "slice range (%d:%d:%d) is invalid for axis %d",
                      (int)first, (int)last, (int)str, (int)axis);
            }
            else
            {
                check(first < extent && last <= first, "slice range (%d:%d:%d) is invalid for axis %d",
                      (int)first, (int)last, (int)str, (int)axis);
            }
            
            output[axis] = ceil_div(last - first, str);
        }
        return output;
    }

    inline Shape stack_shape( const std::vector<Shape>& inputs, const Value& axis )
    {
        auto& input = inputs[0];
        
        bool compatibleShapes = std::all_of(inputs.begin() + 1, inputs.end(), [&]( const Shape& shape ){ return shape == input; });
        check(compatibleShapes, "incompatible tensor shapes in input array");
        
        Shape output(input.size() + 1);
        
        check_axis_compatible_with_rank(axis, output.size());
        
        const size_t idx = axis.integer();
        for ( size_t i = 0; i < idx; ++i )
        {
            output[i] = input[i];
        }
        output[idx] = (Shape::value_type)inputs.size();
        for ( size_t i = idx + 1; i < output.size(); ++i )
        {
            output[i] = input[i-1];
        }
        return output;
    }

    inline std::vector<Shape> unstack_shape( const Shape& input, const Value& axis )
    {
        check_axis_compatible_with_rank(axis, input.size());
        
        const size_t idx = axis.integer();
        
        Shape output(input.size() - 1);
        for ( size_t i = 0; i < idx; ++i )
        {
            output[i] = input[i];
        }
        for ( size_t i = idx; i < output.size(); ++i )
        {
            output[i] = input[i+1];
        }
        
        return std::vector<Shape>(input[idx], output);
    }

    inline Shape squeeze_shape( const Shape& input, const Value& axes )
    {
        check_axes_compatible_with_rank(axes, input.size());
        
        for ( size_t i = 0; i < axes.size(); ++i )
        {
            auto axis = axes[i].integer();
            check(input[axis] == 1, "squeezed dimension is not singleton (has extent %d)", (int)input[axis]);
        }
        
        Shape output(input.size() - axes.size());
        for ( size_t i = 0, k = 0; i < input.size(); ++i )
        {
            if ( !contains_axis(axes, i) )
            {
                output[k++] = input[i];
            }
        }
        return output;
    }

    inline Shape unsqueeze_shape( const Shape& input, const Value& axes )
    {
        Shape output(input.size() + axes.size());
        
        check_axes_compatible_with_rank(axes, output.size());
        
        for ( size_t i = 0, k = 0; i < output.size(); ++i )
        {
            output[i] = contains_axis(axes, i) ? (Shape::value_type)1 : input[k++];
        }
        return output;
    }
    
    inline Shape tile_shape( const Shape& input, const Value& repeats )
    {
        check_rank("repeats", repeats, input.size());
        check_range("repeats", repeats, 1);
        
        Shape output(input.size());
        for ( size_t i = 0; i < output.size(); ++i )
        {
            output[i] = input[i] * repeats[i].integer();
        }
        return output;
    }
    
    inline Shape pad_shape( const Shape& input, const Value& padding )
    {
        check_rank("padding", padding, input.size());
        
        Shape output(input.size());
        for ( size_t i = 0; i < output.size(); ++i )
        {
            output[i] = padding[i][0].integer() + input[i] + padding[i][1].integer();
        }
        return output;
    }
    
    inline Shape gather_shape( const Shape& input, const Shape& indices, const Value& axis )
    {
        check_axis_compatible_with_rank(axis, input.size());
        
        const size_t idx = axis.integer();
        
        Shape output(input.size() + indices.size() - 1);
        std::copy_n(input.begin(), idx, output.begin());
        std::copy_n(indices.begin(), indices.size(), output.begin() + idx);
        std::copy(input.begin() + idx + 1, input.end(), output.begin() + idx + indices.size());
        
        return output;
    }

    inline Shape matmul_shape( const Shape& A, const Shape& B, const Value& trA, const Value& trB )
    {
        check(A.size() == B.size(), "rank mismatch for A and B (%d vs %d)", (int)A.size(), (int)B.size());
        
        auto rank = A.size();
        check(rank >= 2, "rank of A and B must be at least 2, found %d", (int)rank);
        
        auto batch_dims = rank - 2;
        check(broadcast_compatible(A, B, batch_dims),
              "incompatible tensor shapes for broadcasting first %d dimensions (%s vs %s)",
              (int)batch_dims, to_string(A).c_str(), to_string(B).c_str());
        
        auto i0 = batch_dims + 0;
        auto i1 = batch_dims + 1;
        
        auto m = trA.logical() ? A[i1] : A[i0];
        auto n = trB.logical() ? B[i0] : B[i1];
        auto kA = trA.logical() ? A[i0] : A[i1];
        auto kB = trB.logical() ? B[i1] : B[i0];
        
        check(kA == kB, "inner dimensions must agree (%d vs %d)", (int)kA, (int)kB);
        
        Shape C = broadcast_shape(A, B, batch_dims);
        C[i0] = m;
        C[i1] = n;
        return C;
    }

    inline Shape linear_shape( const Shape& input, const Shape& filter, const Shape& bias )
    {
        check(input.size() == 2, "input shape must be of rank 2 (found %d)", (int)input.size());
        check(filter.size() == 2, "filter shape must be of rank 2 (found %d)", (int)filter.size());
        check(input[1] == filter[1], "inner dimensions must agree (%d vs %d)", (int)input[1], (int)filter[1]);
        if ( bias.size() )
        {
            check(bias[1] == filter[0], "bias channels (%d) does not match filter count (%d)", (int)bias[1], (int)filter[0]);
        }
		
        return Shape({ input[0], filter[0] });
    }

    inline Shape update_shape( const Shape& variable, const Shape& value )
    {
        check(value == variable, "updated shape %s does not equal variable shape %s", to_string(value).c_str(), to_string(variable).c_str());
        return variable;
    }

    inline Shape softmax_shape( const Shape& inputShape, const Value& axes )
    {
        check_axes_compatible_with_rank(axes, inputShape.size());
        return inputShape;
    }

    inline std::vector<Shape> copy_n_shape( const Shape& shape, const Value& times )
    {
        check_range("times", times, 1);
        return std::vector<Shape>(times.integer(), shape);
    }

    inline Shape add_n_shape( const std::vector<Shape>& inputs )
    {
        check(inputs.size() != 0, "input array must be non-empty");
        
        auto& shape = inputs[0];
        for ( size_t i = 1; i < inputs.size(); ++i )
        {
            check(inputs[i] == shape, "incompatible item shapes in array (%s vs %s)", to_string(shape).c_str(), to_string(inputs[i]).c_str());
        }
        return shape;
    }
	
	inline Shape quantize_shape( const Shape& input, const Shape& min, const Shape& max, const Value& bits )
	{
		check(broadcastable(min, input), "cannot broadcast 'min' shape (%s) to 'input' shape (%s)",
			  to_string(min).c_str(), to_string(input).c_str());
		check(broadcastable(max, input), "cannot broadcast 'max' shape (%s) to 'input' shape (%s)",
			  to_string(max).c_str(), to_string(input).c_str());
		
		check_range("bits", bits, 0);
		
		return input;
	}
	
	inline Shape linear_quantize_shape( const Shape& input, const Shape& min, const Shape& max, const Value& bits )
	{
		return quantize_shape(input, min, max, bits);
	}
	
	inline Shape logarithmic_quantize_shape( const Shape& input, const Shape& max, const Value& bits )
	{
		return quantize_shape(input, Shape(), max, bits);
	}
    
	inline Shape zero_point_linear_quantize_shape( const Shape& input, const Shape& zero_point, const Shape& scale, const Value& bits )
	{
        check(broadcastable(zero_point, input), "cannot broadcast 'zero_point' shape (%s) to 'input' shape (%s)",
              to_string(zero_point).c_str(), to_string(input).c_str());
        check(broadcastable(scale, input), "cannot broadcast 'scale' shape (%s) to 'input' shape (%s)",
              to_string(scale).c_str(), to_string(input).c_str());
        
        check_range("bits", bits, 0);
        
        return input;
	}

}   // namespace nnef


#endif
