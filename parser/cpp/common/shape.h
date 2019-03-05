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

#include <vector>
#include <string>
#include <numeric>
#include <iostream>
#include <functional>
#include <algorithm>
#include "value.h"
#include "error.h"


namespace nnef
{
    
    typedef std::vector<int> Shape;
    
    
    inline std::string toString( const Shape& shape )
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
    
    inline const Shape& getShape( const Value& value, const std::map<std::string,Shape>& shapes )
    {
        static const Shape singleton;
        return value.kind() == Value::Identifier ? shapes.at(value.identifier()) : singleton;
    }
    
    inline Shape makeShape( const Value& arg, const size_t offset = 0 )
    {
        Shape shape = Shape(offset + arg.size(), 1);
        for ( size_t i = 0; i < arg.size(); ++i )
        {
            shape[i + offset] = arg[i].integer();
        }
        return shape;
    }
    
    inline Shape makePaddingShape( const Value& arg, const size_t offset = 0 )
    {
        Shape padding(offset + arg.size(), 0);
        for ( size_t i = 0; i < arg.size(); ++i )
        {
            padding[i + offset] = arg[i][0].integer() + arg[i][1].integer();
        }
        return padding;
    }
    
    inline Shape makeSeparableFilterShape( const Shape& planeShape, const Shape& pointShape )
    {
        for ( size_t i = 2; i < pointShape.size(); ++i )
        {
            if ( pointShape[i] != 1 )
            {
                throw Error("point filter must have singular extents in spatial dimensions");
            }
        }
        if ( pointShape[1] != planeShape[0] )
        {
            throw Error("channel dimension of point filter must equal batch dimension of plane filter");
        }
        if ( planeShape[1] != 1 )
        {
            throw Error("channel dimension of plane filter must be singular");
        }
        
        Shape shape = planeShape;
        shape[0] = pointShape[0];
        shape[1] = pointShape[1];
        return shape;
    }
    
    inline size_t volumeOf( const Shape& shape )
    {
        return std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
    }
    
    inline bool isBroadcastCompatible( const Shape& xShape, const Shape& yShape, const size_t n, bool bidirectional = true )
    {
        for ( size_t i = 0; i < n; ++i )
        {
            auto xi = i < xShape.size() ? xShape[i] : 1;
            auto yi = i < yShape.size() ? yShape[i] : 1;
            if ( !(xi == yi || (xi == 1 && bidirectional) || yi == 1) )
            {
                return false;
            }
        }
        return true;
    }
    
    inline bool isBroadcastCompatible( const Shape& xShape, const Shape& yShape )
    {
        const size_t rank = std::max(xShape.size(), yShape.size());
        return isBroadcastCompatible(xShape, yShape, rank);
    }
    
    inline void checkBroadcastCompatible( const Shape& shape1, const Shape& shape2, const size_t n, bool bidirectional = true )
    {
        if ( !isBroadcastCompatible(shape1, shape2, n) )
        {
            throw Error("incompatible tensor shapes for broadcasting first %d dimensions (%s vs %s)",
                        (int)n, toString(shape1).c_str(), toString(shape2).c_str());
        }
    }
    
    inline void checkBroadcastCompatible( const Shape& shape1, const Shape& shape2, bool bidirectional = true )
    {
        if ( !isBroadcastCompatible(shape1, shape2) )
        {
            throw Error("incompatible tensor shapes for broadcasting (%s vs %s)",
                        toString(shape1).c_str(), toString(shape2).c_str());
        }
    }
    
    
    inline Shape broadcastShape( const Shape& xShape, const Shape& yShape, const size_t n )
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
    
    inline Shape broadcastShape( const Shape& xShape, const Shape& yShape )
    {
        const size_t rank = std::max(xShape.size(), yShape.size());
        return broadcastShape(xShape, yShape, rank);
    }
    
    
    inline bool isAxesCompatibleWithRank( const Value& axes, const size_t rank )
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
    
    inline bool containsAxis( const Value& axes, const size_t axis )
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
    
    inline void checkAxisCompatibleWithRank( const Value& axis, const size_t rank )
    {
        if ( axis.integer() < 0 || axis.integer() >= (Value::integer_t)rank )
        {
            throw Error("axis must be in range [0,%d); found %d", (int)rank, (int)axis.integer());
        }
    }
    
    inline void checkAxesCompatibleWithRank( const Value& axes, const size_t rank )
    {
        if ( !isAxesCompatibleWithRank(axes, rank) )
        {
            throw Error("axes must be in range [0,%d); found %s", (int)rank, axes.toString().c_str());
        }
    }
    
    inline void checkRange( const char* name, const Value& value, const Value::integer_t min )
    {
        if ( value.kind() == Value::Array || value.kind() == Value::Tuple )
        {
            for ( size_t i = 0; i < value.size(); ++i )
            {
                checkRange(name, value[i], min);
            }
        }
        else if ( value.kind() == Value::Integer )
        {
            if ( value.integer() < min )
            {
                throw Error("'%s' must be >= %d (found %d)", name, min, (int)value.integer());
            }
        }
    }
    
    inline void checkRank( const char* name, const Value& value, const size_t rank )
    {
        if ( value.size() != rank )
        {
            throw Error("length of array '%s' must be %d to match rank of operation (found %d)", name, (int)rank, (int)value.size());
        }
    }
    
    inline void checkBias( const Shape& biasShape, const Shape::value_type channels )
    {
        if ( biasShape.size() == 2 )
        {
            if ( biasShape[0] != 1 )
            {
                throw Error("bias shape must be singular for the batch dimension");
            }
            if ( biasShape[1] != channels && biasShape[1] != 1 )
            {
                throw Error("bias channels (%d) does not match output channels (%d)", (int)biasShape[1], (int)channels);
            }
        }
        else if ( biasShape.size() != 0 )
        {
            throw Error("bias must be of rank 0 or 2");
        }
    }
    
    template<typename T>
    inline T downSize( const T input, const T size, const T padding, const T stride, const T dilation )
    {
        const T window = 1 + (size - 1) * dilation;
        return (input + padding - window) / stride + 1;
    }
    
    template<typename T>
    inline T downSize( const T input, const T stride )
    {
        return (input + stride - 1) / stride;
    }
    
    template<typename T>
    inline T upSize( const T input, const T size, const T padding, const T stride, const T dilation )
    {
        const T window = 1 + (size - 1) * dilation;
        return (input - 1) * stride + window - padding;
    }
    
    template<typename T>
    inline T upSize( const T input, const T stride )
    {
        return input * stride;
    }
    
    
    inline Shape binaryShape( const Shape& shape1, const Shape& shape2 )
    {
        checkBroadcastCompatible(shape1, shape2);
        return broadcastShape(shape1, shape2);
    }
    
    inline Shape ternaryShape( const Shape& shape1, const Shape& shape2, const Shape& shape3 )
    {
        checkBroadcastCompatible(shape1, shape2);
        const Shape shape12 = broadcastShape(shape1, shape2);
        
        checkBroadcastCompatible(shape12, shape3);
        return broadcastShape(shape12, shape3);
    }
    
    inline Shape reduceShape( const Shape& shape, const Value& axes )
    {
        checkAxesCompatibleWithRank(axes, shape.size());
        
        Shape outputShape = shape;
        for ( size_t i = 0; i < axes.size(); ++i )
        {
            auto axis = axes[i].integer();
            outputShape[axis] = 1;
        }
        
        return outputShape;
    }
    
    inline Shape poolShape( const Shape& inputShape, const Shape& kernelShape,
                           const Shape& strideShape, const Shape& dilationShape,
                           const Shape& paddingShape, const size_t offset = 0 )
    {
        Shape outputShape(inputShape.size());
        for ( size_t i = offset; i < outputShape.size(); ++i )
        {
            outputShape[i] = paddingShape.size() ? downSize(inputShape[i], kernelShape[i], paddingShape[i], strideShape[i], dilationShape[i]) :
                                                    downSize(inputShape[i], strideShape[i]);
        }
        return outputShape;
    }
    
    inline Shape unpoolShape( const Shape& inputShape, const Shape& kernelShape,
                             const Shape& strideShape, const Shape& dilationShape,
                             const Shape& paddingShape, const Shape& outputShape,
                             const size_t offset = 0 )
    {
        if ( outputShape.size() )
        {
            Shape expectedShape(outputShape.size());
            std::copy_n(inputShape.begin(), offset, expectedShape.begin());
            for ( size_t i = offset; i < expectedShape.size(); ++i )
            {
                expectedShape[i] = paddingShape.size() ? downSize(outputShape[i], kernelShape[i], paddingShape[i], strideShape[i], dilationShape[i])
                                                        : downSize(outputShape[i], strideShape[i]);
            }
            
            if ( expectedShape != inputShape )
            {
                throw Error("expected input shape %s derived from output shape is incompatible with actual input shape %s",
                            toString(expectedShape).c_str(), toString(inputShape).c_str());
            }
            return outputShape;
        }
        else
        {
            Shape resultShape(inputShape.size());
            for ( size_t i = offset; i < resultShape.size(); ++i )
            {
                resultShape[i] = paddingShape.size() ? upSize(inputShape[i], kernelShape[i], paddingShape[i], strideShape[i], dilationShape[i])
                                                     : upSize(inputShape[i], strideShape[i]);
            }
            return resultShape;
        }
    }
    
    inline Shape convShape( const Shape& inputShape, const Shape& filterShape,
                           const Shape& strideShape, const Shape& dilationShape,
                           const Shape& paddingShape, const Shape::value_type groups )
    {
        if ( inputShape[1] != filterShape[1] * groups )
        {
            throw Error("filter channels (%d) does not match input channels (%d) times groups (%d)",
                        (int)filterShape[1], (int)inputShape[1], (int)groups);
        }
        if ( filterShape[0] % groups )
        {
            throw Error("filter batch (%d) must be divisible by groups (%d)", (int)filterShape[0], (int)groups);
        }
        
        Shape outputShape = poolShape(inputShape, filterShape, strideShape, dilationShape, paddingShape, 2);
        outputShape[0] = inputShape[0];
        outputShape[1] = filterShape[0];
        return outputShape;
    }
    
    inline Shape deconvShape( const Shape& inputShape, const Shape& filterShape,
                             const Shape& strideShape, const Shape& dilationShape,
                             const Shape& paddingShape, const Shape& outputShape,
                             const Shape::value_type groups )
    {
        if ( inputShape[1] != filterShape[0] )
        {
            throw Error("filter batch (%d) does not match input channels (%d)", (int)filterShape[0], (int)inputShape[1]);
        }
        if ( filterShape[0] % groups )
        {
            throw Error("filter batch (%d) must be divisible by groups (%d)", (int)filterShape[0], (int)groups);
        }
        if ( outputShape.size() )
        {
            if ( outputShape[0] != inputShape[0] )
            {
                throw Error("output batch (%d) does not match input batch (%d)", (int)outputShape[0], (int)inputShape[0]);
            }
            if ( outputShape[1] != filterShape[1] * groups )
            {
                throw Error("output channels (%d) does not match filter channels (%d) times groups (%d)",
                            (int)outputShape[1], (int)filterShape[1], (int)groups);
            }
        }
        
        Shape resultShape = unpoolShape(inputShape, filterShape, strideShape, dilationShape, paddingShape, outputShape, 2);
        resultShape[0] = inputShape[0];
        resultShape[1] = filterShape[1] * groups;
        return resultShape;
    }
    
    inline Shape downsampleShape( const Shape& inputShape, const Value& factor )
    {
        Shape outputShape = inputShape;
        for ( size_t i = 0; i < factor.size(); ++i )
        {
            outputShape[i+2] /= factor[i].integer();
        }
        
        return outputShape;
    }
    
    inline Shape upsampleShape( const Shape& inputShape, const Value& factor )
    {
        Shape outputShape = inputShape;
        for ( size_t i = 0; i < factor.size(); ++i )
        {
            outputShape[i+2] *= factor[i].integer();
        }
        
        return outputShape;
    }
    
    
    
    inline void inferShapeTransitive( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& output = args.at("output");
        
        shapes[output.identifier()] = getShape(input, shapes);
    }
    
    inline void inferShapeNullary( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& shape = args.at("shape");
        auto& output = args.at("output");
        
        auto outputShape = makeShape(shape);;
        
        if ( args.count("value") )
        {
            auto& value = args.at("value");
            if ( value.size() != volumeOf(outputShape) && value.size() != 1 )
            {
                throw Error("shape volume (%d) does not match number of values (%d)", (int)volumeOf(outputShape), (int)value.size());
            }
        }
        
        shapes[output.identifier()] = outputShape;
    }
    
    inline void inferShapeUnary( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& x = args.at("x");
        auto& y = args.at("y");
        
        shapes[y.identifier()] = getShape(x, shapes);
    }
    
    inline void inferShapeBinary( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& x = args.at("x");
        auto& y = args.at("y");
        auto& z = args.at("z");
        
        shapes[z.identifier()] = binaryShape(getShape(x, shapes), getShape(y, shapes));
    }
    
    inline Shape getFilterShape( const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        if ( args.count("filter") )
        {
            return getShape(args.at("filter"), shapes);
        }
        else
        {
            auto planeShape = getShape(args.at("plane_filter"), shapes);
            auto pointShape = getShape(args.at("point_filter"), shapes);
            return makeSeparableFilterShape(planeShape, pointShape);
        }
    }
    
    inline void inferShapeConv( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& bias = args.at("bias");
        auto& stride = args.at("stride");
        auto& dilation = args.at("dilation");
        auto& padding = args.at("padding");
        auto& output = args.at("output");
        auto& groups = args.at("groups");
        
        auto& inputShape = getShape(input, shapes);
        auto rank = inputShape.size();
        
        auto filterShape = getFilterShape(args, shapes);
        auto& biasShape = getShape(bias, shapes);
        auto strideShape = makeShape(stride, stride.size() ? 2 : rank);
        auto dilationShape = makeShape(dilation, dilation.size() ? 2 : rank);
        auto paddingShape = padding.size() ? makePaddingShape(padding, 2) : Shape();
        
        if ( padding.size() )
        {
            checkRank("padding", padding, rank - 2);
        }
        if ( stride.size() )
        {
            checkRank("stride", stride, rank - 2);
        }
        if ( dilation.size() )
        {
            checkRank("dilation", dilation, rank - 2);
        }
        
        checkRange("stride", stride, 1);
        checkRange("dilation", dilation, 1);
        checkRange("groups", groups, 0);
        
        checkBias(biasShape, filterShape[0]);
        
        auto groupCount = groups.integer() != 0 ? groups.integer() : inputShape[1];
        
        shapes[output.identifier()] = convShape(inputShape, filterShape, strideShape, dilationShape, paddingShape, groupCount);
    }
    
    inline void inferShapeDeconv( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& bias = args.at("bias");
        auto& stride = args.at("stride");
        auto& dilation = args.at("dilation");
        auto& padding = args.at("padding");
        auto& output = args.at("output");
        auto& groups = args.at("groups");
        auto& output_shape = args.at("output_shape");
        
        auto& inputShape = getShape(input, shapes);
        auto filterShape = getFilterShape(args, shapes);
        auto& biasShape = getShape(bias, shapes);
        auto rank = inputShape.size();
        
        auto strideShape = makeShape(stride, stride.size() ? 2 : inputShape.size());
        auto dilationShape = makeShape(dilation, dilation.size() ? 2 : inputShape.size());
        auto paddingShape = padding.size() ? makePaddingShape(padding, 2) : Shape();
        auto outputShape = output_shape.size() ? makeShape(output_shape) : Shape();
        
        if ( padding.size() )
        {
            checkRank("padding", padding, rank - 2);
        }
        if ( stride.size() )
        {
            checkRank("stride", stride, rank - 2);
        }
        if ( dilation.size() )
        {
            checkRank("dilation", dilation, rank - 2);
        }
        if ( output_shape.size() )
        {
            checkRank("output_shape", output_shape, rank);
        }
        
        checkRange("stride", stride, 1);
        checkRange("dilation", dilation, 1);
        checkRange("output_shape", output_shape, 1);
        checkRange("groups", groups, 0);
        
        auto groupCount = groups.integer() != 0 ? groups.integer() : inputShape[1];
        
        checkBias(biasShape, filterShape[1] * groupCount);
        
        shapes[output.identifier()] = deconvShape(inputShape, filterShape, strideShape, dilationShape, paddingShape, outputShape, groupCount);
    }
    
    inline void inferShapePool( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& size = args.at("size");
        auto& stride = args.at("stride");
        auto& dilation = args.at("dilation");
        auto& padding = args.at("padding");
        
        auto& inputShape = getShape(input, shapes);
        auto rank = inputShape.size();
        
        auto kernelShape = makeShape(size);
        auto strideShape = makeShape(stride, stride.size() ? 0 : rank);
        auto dilationShape = makeShape(dilation, dilation.size() ? 0 : rank);
        auto paddingShape = padding.size() ? makePaddingShape(padding) : Shape();
        
        checkRank("size", size, rank);
        if ( padding.size() )
        {
            checkRank("padding", padding, rank);
        }
        if ( stride.size() )
        {
            checkRank("stride", stride, rank);
        }
        if ( dilation.size() )
        {
            checkRank("dilation", dilation, rank);
        }
        
        checkRange("size", size, 1);
        checkRange("stride", stride, 1);
        checkRange("dilation", dilation, 1);
        
        auto shape = poolShape(inputShape, kernelShape, strideShape, dilationShape, paddingShape);
        
        if ( args.count("output") )
        {
            shapes[args.at("output").identifier()] = shape;
        }
        if ( args.count("index") )
        {
            shapes[args.at("index").identifier()] = shape;
        }
    }
    
    inline void inferShapeUnpool( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& size = args.at("size");
        auto& stride = args.at("stride");
        auto& dilation = args.at("dilation");
        auto& padding = args.at("padding");
        auto& output = args.at("output");
        auto& output_shape = args.at("output_shape");
        
        auto& inputShape = getShape(input, shapes);
        auto rank = inputShape.size();
        auto kernelShape = makeShape(size);
        auto strideShape = makeShape(stride, stride.size() ? 0 : rank);
        auto dilationShape = makeShape(dilation, dilation.size() ? 0 : rank);
        auto paddingShape = padding.size() ? makePaddingShape(padding) : Shape();
        auto outputShape = output_shape.size() ? makeShape(output_shape) : Shape();
        
        checkRank("size", size, rank);
        if ( padding.size() )
        {
            checkRank("padding", padding, rank);
        }
        if ( stride.size() )
        {
            checkRank("stride", stride, rank);
        }
        if ( dilation.size() )
        {
            checkRank("dilation", dilation, rank);
        }
        if ( output_shape.size() )
        {
            checkRank("output_shape", output_shape, rank);
        }
        
        checkRange("size", size, 1);
        checkRange("stride", stride, 1);
        checkRange("dilation", dilation, 1);
        checkRange("output_shape", output_shape, 1);
        
        shapes[output.identifier()] = unpoolShape(inputShape, kernelShape, strideShape, dilationShape, paddingShape, outputShape);
        
        if ( args.count("index") )
        {
            auto& index = args.at("index");
            auto& indexShape = getShape(index, shapes);
            if ( indexShape != inputShape )
            {
                throw Error("index shape incompatible with input shape (%s vs %s)",
                            toString(indexShape).c_str(), toString(inputShape).c_str());
            }
        }
    }
    
    inline void inferShapeReduce( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& output = args.at("output");
        auto& axes = args.at("axes");
        
        shapes[output.identifier()] = reduceShape(getShape(input, shapes), axes);
    }
    
    inline void inferShapeMoments( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& axes = args.at("axes");
        auto& mean = args.at("mean");
        auto& variance = args.at("variance");
        
        auto shape = reduceShape(getShape(input, shapes), axes);
        
        shapes[mean.identifier()] = shape;
        shapes[variance.identifier()] = shape;
    }
    
    inline void inferShapeNormalize( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& output = args.at("output");
        
        auto& inputShape = getShape(input, shapes);
        
        if ( args.count("axes") )
        {
            checkAxesCompatibleWithRank(args.at("axes"), inputShape.size());
        }
        if ( args.count("size") )
        {
            auto& size = args.at("size");
            checkRank("size", size, inputShape.size());
            checkRange("size", size, 1);
        }
        
        shapes[output.identifier()] = inputShape;
    }
    
    inline void inferShapeBatchNorm( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& output = args.at("output");
        auto& mean = args.at("mean");
        auto& variance = args.at("variance");
        auto& offset = args.at("offset");
        auto& scale = args.at("scale");
        
        auto& inputShape = getShape(input, shapes);
        
        checkBroadcastCompatible(inputShape, getShape(mean, shapes));
        checkBroadcastCompatible(inputShape, getShape(variance, shapes));
        checkBroadcastCompatible(inputShape, getShape(offset, shapes));
        checkBroadcastCompatible(inputShape, getShape(scale, shapes));
        
        shapes[output.identifier()] = inputShape;
    }
    
    inline void inferShapeDownsample( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& output = args.at("output");
        auto& factor = args.at("factor");
        
        auto& inputShape = getShape(input, shapes);
        
        checkRank("factor", factor, inputShape.size() - 2);
        
        for ( size_t i = 0; i < factor.size(); ++i )
        {
            auto scale = factor[i].integer();
            if ( inputShape[i+2] % scale )
            {
                throw Error("input extent (%d) must be divisible by factor (%d)", (int)inputShape[i+2], (int)scale);
            }
        }
        
        shapes[output.identifier()] = downsampleShape(inputShape, factor);
    }
    
    inline void inferShapeUpsample( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& output = args.at("output");
        auto& factor = args.at("factor");
        
        auto& inputShape = getShape(input, shapes);
        
        checkRank("factor", factor, inputShape.size() - 2);
        
        shapes[output.identifier()] = upsampleShape(inputShape, factor);
    }
    
    inline void inferShapeRoi( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& output = args.at("output");
        auto& rois = args.at("rois");
        auto& index = args.at("batch_index");
        auto& size = args.at("output_size");
        
        auto& inputShape = getShape(input, shapes);
        auto& roisShape = getShape(rois, shapes);
        auto& indexShape = getShape(index, shapes);
        
        checkRank("pooled_size", size, inputShape.size() - 2);
        checkRange("pooled_size", size, 1);
        
        if ( args.count("sampling_rate") )
        {
            auto& rate = args.at("sampling_rate");
            checkRank("sampling_rate", rate, inputShape.size() - 2);
            checkRange("sampling_rate", rate, 1);
        }
        
        if ( roisShape.size() != 2 )
        {
            throw Error("'rois' must be a rank-2 tensor");
        }
        if ( indexShape.size() != 1 )
        {
            throw Error("'batch_index' must be a rank-1 tensor");
        }
        
        if ( roisShape[1] != 4 )
        {
            throw Error("rois must be of extent 4 along dimension 1 (found %d)", (int)roisShape[1]);
        }
        if ( indexShape[0] != roisShape[0] )
        {
            throw Error("'batch_index' must be of same length as dimension 0 of rois; found (%d vs %d)", (int)indexShape[0], (int)roisShape[0]);
        }
        
        Shape outputShape(inputShape.size());
        outputShape[0] = roisShape[0];
        outputShape[1] = inputShape[1];
        for ( size_t i = 0; i < size.size(); ++i )
        {
            outputShape[i+2] = (Shape::value_type)size[i].integer();
        }
        
        shapes[output.identifier()] = outputShape;
    }
    
    inline void inferShapeReshape( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& output = args.at("output");
        auto& shape = args.at("shape");
        
        auto& inputShape = getShape(input, shapes);
        Shape outputShape = makeShape(shape);
        
        size_t autoAxis = std::numeric_limits<size_t>::max();
        for ( size_t i = 0; i < outputShape.size(); ++i )
        {
            if ( outputShape[i] == 0 )
            {
                outputShape[i] = inputShape[i];
            }
            else if ( outputShape[i] == -1 )
            {
                if ( autoAxis != std::numeric_limits<size_t>::max() )
                {
                    throw Error("shape may only contain at most one -1 value");
                }
                outputShape[i] = 1;
                autoAxis = i;
            }
        }
        
        auto inputVolume = volumeOf(inputShape);
        auto outputVolume = volumeOf(outputShape);
        
        if ( autoAxis != std::numeric_limits<size_t>::max() )
        {
            if ( inputVolume % outputVolume )
            {
                throw Error("automatic output shape (%s) incompatible with input shape (%s)", (int)outputVolume, (int)inputVolume);
            }
            outputShape[autoAxis] = (Shape::value_type)(inputVolume / outputVolume);
        }
        else if ( inputVolume != outputVolume )
        {
            throw Error("input volume (%d) does not equal output volume (%d)", (int)inputVolume, (int)outputVolume);
        }
        
        shapes[output.identifier()] = outputShape;
    }
    
    inline void inferShapeTranspose( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& output = args.at("output");
        auto& axes = args.at("axes");
        
        auto& inputShape = getShape(input, shapes);
        Shape outputShape = inputShape;
        
        std::vector<size_t> perm(axes.size());
        for ( size_t i = 0; i < axes.size(); ++i )
        {
            perm[i] = axes[i].integer();
        }
        
        std::sort(perm.begin(), perm.end());
        for ( size_t i = 0; i < perm.size(); ++i )
        {
            if ( perm[i] != i )
            {
                throw Error("'axes' array must contain a permutation of dimensions from 0 to %d", (int)perm.size());
            }
        }
        
        for ( size_t i = 0; i < axes.size(); ++i )
        {
            auto j = axes[i].integer();
            outputShape[i] = inputShape[j];
        }
        
        shapes[output.identifier()] = outputShape;
    }
    
    inline void inferShapeSplit( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& value = args.at("value");
        auto& values = args.at("values");
        auto& axis = args.at("axis");
        auto& ratios = args.at("ratios");
        
        auto& wholeShape = getShape(value, shapes);
        
        checkAxisCompatibleWithRank(axis, wholeShape.size());
        checkRange("ratios", ratios, 1);
        
        auto idx = axis.integer();
        
        Value::integer_t sumRatios = 0;
        for ( size_t i = 0; i < ratios.size(); ++i )
        {
            sumRatios += ratios[i].integer();
        }
        
        if ( wholeShape[idx] % sumRatios != 0 )
        {
            throw Error("sum of split ratios (%d) does not divide whole extent (%d)", (int)sumRatios, (int)wholeShape[idx]);
        }
        
        const Value::integer_t unit = wholeShape[idx] / sumRatios;
        
        if ( ratios.size() != values.size() )
        {
            throw Error("length of split ratios (%d) does not match length of values (%d)", (int)ratios.size(), (int)values.size());
        }
        
        for ( size_t i = 0; i < ratios.size(); ++i )
        {
            Shape itemShape = wholeShape;
            itemShape[idx] = unit * ratios[i].integer();
            
            shapes[values[i].identifier()] = itemShape;
        }
    }
    
    inline void inferShapeConcat( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& values = args.at("values");
        auto& value = args.at("value");
        auto& axis = args.at("axis");
        
        Shape outputShape = getShape(values[0], shapes);
        
        checkAxisCompatibleWithRank(axis, outputShape.size());
        
        const size_t idx = axis.integer();
        
        bool compatibleShape = true;
        for ( size_t i = 1; i < values.size(); ++i )
        {
            auto& partShape = getShape(values[i], shapes);
            
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
        
        if ( !compatibleShape )
        {
            throw Error("incompatible tensor shapes in input array");
        }
        
        shapes[value.identifier()] = outputShape;
    }
    
    inline void inferShapeSlice( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& output = args.at("output");
        auto& axes = args.at("axes");
        auto& begin = args.at("begin");
        auto& end = args.at("end");
        
        if ( begin.size() != axes.size() || end.size() != axes.size() )
        {
            throw Error("'axes', 'begin' and 'end' arrays must have the same length");
        }
        
        Shape outputShape = getShape(input, shapes);
        
        checkAxesCompatibleWithRank(axes, outputShape.size());
        
        for ( size_t i = 0; i < axes.size(); ++i )
        {
            auto axis = axes[i].integer();
            auto extent = outputShape[axis];
            
            auto first = begin[i].integer();
            if ( first < 0 )
            {
                first += extent;
            }
            
            auto last = end[i].integer();
            if ( last <= 0 )
            {
                last += extent;
            }
            
            if ( last <= first )
            {
                throw Error("slice range (%d,%d) is empty for axis %d", (int)first, (int)last, (int)axis);
            }
            
            if ( first < 0 || last > extent )
            {
                throw Error("slice range (%d,%d) is out of tensor shape for axis %d", (int)first, (int)last, (int)axis);
            }
            
            outputShape[axis] = last - first;
        }
        
        shapes[output.identifier()] = outputShape;
    }
    
    inline void inferShapeStack( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& values = args.at("values");
        auto& value = args.at("value");
        auto& axis = args.at("axis");
        
        auto& inputShape = getShape(values[0], shapes);
        
        bool compatibleShape = true;
        for ( size_t i = 1; i < values.size(); ++i )
        {
            auto& partShape = getShape(values[i], shapes);
            
            if ( partShape.size() != inputShape.size() )
            {
                compatibleShape = false;
                break;
            }
            
            for ( size_t i = 0; i < inputShape.size(); ++i )
            {
                compatibleShape &= inputShape[i] == partShape[i];
            }
        }
        
        if ( !compatibleShape )
        {
            throw Error("incompatible tensor shapes in input array");
        }
        
        Shape outputShape(inputShape.size() + 1);
        
        checkAxisCompatibleWithRank(axis, outputShape.size());
        
        const size_t idx = axis.integer();
        for ( size_t i = 0; i < idx; ++i )
        {
            outputShape[i] = inputShape[i];
        }
        outputShape[idx] = (Shape::value_type)values.size();
        for ( size_t i = idx + 1; i < outputShape.size(); ++i )
        {
            outputShape[i] = inputShape[i-1];
        }
        
        shapes[value.identifier()] = outputShape;
    }
    
    inline void inferShapeUnstack( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& values = args.at("values");
        auto& value = args.at("value");
        auto& axis = args.at("axis");
        
        Shape inputShape = getShape(value, shapes);
        
        checkAxisCompatibleWithRank(axis, inputShape.size());
        
        const size_t idx = axis.integer();
        
        Shape outputShape(inputShape.size() - 1);
        for ( size_t i = 0; i < idx; ++i )
        {
            outputShape[i] = inputShape[i];
        }
        for ( size_t i = idx; i < outputShape.size(); ++i )
        {
            outputShape[i] = inputShape[i+1];
        }
        
        const size_t count = inputShape[idx];
        if ( values.size() != count )
        {
            throw Error("length of values (%d) does not match shape of value along axis (%d)", (int)values.size(), (int)count);
        }
        for ( size_t i = 0; i < count; ++i )
        {
            shapes[values[i].identifier()] = outputShape;
        }
    }
    
    inline void inferShapeSqueeze( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& output = args.at("output");
        auto& axes = args.at("axes");
        
        auto& inputShape = getShape(input, shapes);
        
        checkAxesCompatibleWithRank(axes, inputShape.size());
        
        for ( size_t i = 0; i < axes.size(); ++i )
        {
            auto axis = axes[i].integer();
            if ( inputShape[axis] != 1 )
            {
                throw Error("squeezed dimension is not singleton (has extent %d)", (int)inputShape[axis]);
            }
        }
        
        Shape outputShape(inputShape.size() - axes.size());
        for ( size_t i = 0, k = 0; i < inputShape.size(); ++i )
        {
            if ( !containsAxis(axes, i) )
            {
                outputShape[k++] = inputShape[i];
            }
        }
        
        shapes[output.identifier()] = outputShape;
    }
    
    inline void inferShapeUnsqueeze( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& output = args.at("output");
        auto& axes = args.at("axes");
        
        auto& inputShape = getShape(input, shapes);
        
        Shape outputShape(inputShape.size() + axes.size());
        
        checkAxesCompatibleWithRank(axes, outputShape.size());
        
        for ( size_t i = 0, k = 0; i < outputShape.size(); ++i )
        {
            outputShape[i] = containsAxis(axes, i) ? (Shape::value_type)1 : inputShape[k++];
        }
        
        shapes[output.identifier()] = outputShape;
    }
    
    inline void inferShapeMatmul( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& A = args.at("A");
        auto& B = args.at("B");
        auto& C = args.at("C");
        auto& trA = args.at("transposeA");
        auto& trB = args.at("transposeB");
        
        auto& aShape = getShape(A, shapes);
        auto& bShape = getShape(B, shapes);
        
        if ( aShape.size() != bShape.size() )
        {
            throw Error("rank mismatch for A and B (%d vs %d)", (int)aShape.size(), (int)bShape.size());
        }
        
        auto rank = aShape.size();
        if ( rank < 2 )
        {
            throw Error("rank of A and B must be at least 2");
        }
        
        auto batch_dims = rank - 2;
        
        if ( !isBroadcastCompatible(aShape, bShape, batch_dims) )
        {
            throw Error("shape of A and B must be broadcast compatible for batch dimensions");
        }
        
        auto i0 = batch_dims + 0;
        auto i1 = batch_dims + 1;
        
        auto m = trA.logical() ? aShape[i1] : aShape[i0];
        auto n = trB.logical() ? bShape[i0] : bShape[i1];
        auto kA = trA.logical() ? aShape[i0] : aShape[i1];
        auto kB = trB.logical() ? bShape[i1] : bShape[i0];
        
        if ( kA != kB )
        {
            throw Error("inner dimensions must agree (%d vs %d)", (int)kA, (int)kB);
        }
        
        Shape cShape = broadcastShape(aShape, bShape, batch_dims);
        cShape[i0] = m;
        cShape[i1] = n;
        
        shapes[C.identifier()] = cShape;
    }
    
    inline void inferShapeLinear( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& output = args.at("output");
        auto& filter = args.at("filter");
        
        auto& inputShape = getShape(input, shapes);
        auto& filterShape = getShape(filter, shapes);
        
        if ( inputShape.size() != 2 )
        {
            throw Error("input shape must be of rank 2 (found %d)", (int)inputShape.size());
        }
        if ( filterShape.size() != 2 )
        {
            throw Error("filter shape must be of rank 2 (found %d)", (int)filterShape.size());
        }
        
        if ( inputShape[1] != filterShape[1] )
        {
            throw Error("inner dimensions must agree (%d vs %d)", (int)inputShape[1], (int)filterShape[1]);
        }
        
        Shape outputShape(2);
        outputShape[0] = inputShape[0];
        outputShape[1] = filterShape[0];
        
        shapes[output.identifier()] = outputShape;
    }
    
    inline void inferShapeUpdate( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& variable = args.at("variable");
        auto& result = args.at("result");
        auto& value = args.at("value");
        
        auto& varShape = getShape(variable, shapes);
        auto& valShape = getShape(value, shapes);
        
        if ( valShape != varShape )
        {
            throw Error("updated shape %s does not equal variable shape %s", toString(valShape).c_str(), toString(varShape).c_str());
        }
        
        shapes[result.identifier()] = varShape;
    }
    
    inline void inferShapeSoftmax( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& input = args.at("input");
        auto& output = args.at("output");
        auto& inputShape = shapes.at(input.identifier());
        
        shapes[output.identifier()] = inputShape;
        
        auto& axes = args.at("axes");
        checkAxesCompatibleWithRank(axes, inputShape.size());
    }
    
    inline void inferShapeCopyN( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& times = args.at("times");
        checkRange("times", times, 1);
        
        auto& x = args.at("x");
        auto& y = args.at("y");
        auto& shape = getShape(x, shapes);
        
        if ( (size_t)times.integer() != y.size() )
        {
            throw Error("argument times (%d) does not equal length of y", (int)times.integer(), (int)y.size());
        }
        
        for ( size_t i = 0; i < y.size(); ++i )
        {
            shapes[y[i].identifier()] = shape;
        }
    }
    
    inline void inferShapeAddN( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& x = args.at("x");
        auto& y = args.at("y");
        
        if ( x.size() == 0 )
        {
            throw Error("array 'x' must be non-empty");
        }
        
        auto& yShape = getShape(x[0], shapes);
        for ( size_t i = 1; i < x.size(); ++i )
        {
            auto& shape = getShape(x[i], shapes);
            if ( !isBroadcastCompatible(yShape, shape) )
            {
                throw Error("incompatible item shapes in array");
            }
        }
        
        shapes[y.identifier()] = yShape;
    }
    
    inline void inferShapeSelect( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& condition = args.at("condition");
        auto& true_value = args.at("true_value");
        auto& false_value = args.at("false_value");
        auto& result = args.at("output");
        
        shapes[result.identifier()] = ternaryShape(getShape(condition, shapes), getShape(true_value, shapes), getShape(false_value, shapes));
    }

    inline void inferShapeClamp( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes )
    {
        auto& x = args.at("x");
        auto& a = args.at("a");
        auto& b = args.at("b");
        auto& y = args.at("y");
        
        shapes[y.identifier()] = ternaryShape(getShape(x, shapes), getShape(a, shapes), getShape(b, shapes));
    }

    
    typedef void (*shape_func)( const std::string& op, const std::map<std::string,Value>& args, std::map<std::string,Shape>& shapes );
    typedef std::map<std::string,shape_func> ShapeFuncs;
    
    
    inline const ShapeFuncs& standardShapeFuncs()
    {
        static const ShapeFuncs funcs =
        {
            { "external", inferShapeNullary },
            { "constant", inferShapeNullary },
            { "variable", inferShapeNullary },
            
            { "copy", inferShapeUnary },
            { "neg", inferShapeUnary },
            { "rcp", inferShapeUnary },
            { "exp", inferShapeUnary },
            { "log", inferShapeUnary },
            { "abs", inferShapeUnary },
            { "sign", inferShapeUnary },
            { "floor", inferShapeUnary },
            { "ceil", inferShapeUnary },
            { "round", inferShapeUnary },
            { "sqr", inferShapeUnary },
            { "sqrt", inferShapeUnary },
            { "rsqr", inferShapeUnary },
            { "rsqrt", inferShapeUnary },
            { "not", inferShapeUnary },
            { "log2", inferShapeUnary },
            
            { "relu", inferShapeUnary },
            { "sigmoid", inferShapeUnary },
            { "tanh", inferShapeUnary },
            { "elu", inferShapeUnary },
            { "softabs", inferShapeUnary },
            { "softmax", inferShapeUnary },
            { "softplus", inferShapeUnary },
            { "leaky_relu", inferShapeUnary },
            { "prelu", inferShapeUnary },
            
            { "linear_quantize", inferShapeUnary },
            { "logarithmic_quantize", inferShapeUnary },
            { "binary_quantize", inferShapeUnary },
            { "ternary_quantize", inferShapeUnary },
            
            { "add", inferShapeBinary },
            { "sub", inferShapeBinary },
            { "mul", inferShapeBinary },
            { "div", inferShapeBinary },
            { "min", inferShapeBinary },
            { "max", inferShapeBinary },
            { "pow", inferShapeBinary },
            { "lt",  inferShapeBinary },
            { "le",  inferShapeBinary },
            { "gt",  inferShapeBinary },
            { "ge",  inferShapeBinary },
            { "eq",  inferShapeBinary },
            { "ne",  inferShapeBinary },
            { "and", inferShapeBinary },
            { "or",  inferShapeBinary },
            { "min", inferShapeBinary },
            { "max", inferShapeBinary },
            
            { "conv", inferShapeConv },
            { "separable_conv", inferShapeConv },
            { "deconv", inferShapeDeconv },
            { "separable_deconv", inferShapeDeconv },
            
            { "box", inferShapePool },
            { "sample", inferShapePool },
            { "max_pool", inferShapePool },
            { "argmax_pool", inferShapePool },
            { "max_pool_with_index", inferShapePool },
            { "avg_pool", inferShapePool },
            { "rms_pool", inferShapePool },
            { "debox", inferShapeUnpool },
            { "desample", inferShapeUnpool },
            
            { "sum_reduce", inferShapeReduce },
            { "min_reduce", inferShapeReduce },
            { "max_reduce", inferShapeReduce },
            { "mean_reduce", inferShapeReduce },
            { "argmax_reduce", inferShapeReduce },
            { "argmin_reduce", inferShapeReduce },
            { "moments", inferShapeMoments },
            
            { "nearest_downsample", inferShapeDownsample },
            { "area_downsample", inferShapeDownsample },
            { "nearest_upsample", inferShapeUpsample },
            { "multilinear_upsample", inferShapeUpsample },
            
            { "local_response_normalization", inferShapeNormalize },
            { "local_mean_normalization", inferShapeNormalize },
            { "local_variance_normalization", inferShapeNormalize },
            { "local_contrast_normalization", inferShapeNormalize },
            { "l1_normalization", inferShapeNormalize },
            { "l2_normalization", inferShapeNormalize },
            { "batch_normalization", inferShapeBatchNorm },
            
            { "avg_roi_pool", inferShapeRoi },
            { "max_roi_pool", inferShapeRoi },
            { "avg_roi_align", inferShapeRoi },
            { "max_roi_align", inferShapeRoi },
            { "roi_resample", inferShapeRoi },
            
            { "reshape", inferShapeReshape },
            { "transpose", inferShapeTranspose },
            { "split", inferShapeSplit },
            { "concat", inferShapeConcat },
            { "slice", inferShapeSlice },
            { "stack", inferShapeStack },
            { "unstack", inferShapeUnstack },
            { "squeeze", inferShapeSqueeze },
            { "unsqueeze", inferShapeUnsqueeze },
            { "matmul", inferShapeMatmul },
            { "linear", inferShapeLinear },
            { "update", inferShapeUpdate },
            { "softmax", inferShapeSoftmax },
            { "copy_n", inferShapeCopyN },
            { "add_n", inferShapeAddN },
            { "select", inferShapeSelect },
            { "clamp", inferShapeClamp },
        };
        return funcs;
    }
    
}   // namespace nnef


#endif
