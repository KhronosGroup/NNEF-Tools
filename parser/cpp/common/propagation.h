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

#ifndef _NNEF_PROPAGATION_H_
#define _NNEF_PROPAGATION_H_

#include "shape.h"
#include "error.h"
#include "prototype.h"
#include "dictionary.h"
#include <algorithm>
#include <cassert>


namespace nnef
{

    enum class PropagationGroup
    {
        Unknown, Intro, Unary, Binary, Reduce, Conv, Deconv, Pool, UpSample, DownSample, Normalize, Unique
    };


    inline PropagationGroup getPropagationGroup( const std::string& name )
    {
        const std::map<std::string,PropagationGroup> operationGroups =
        {
            std::make_pair("external", PropagationGroup::Intro),
            std::make_pair("variable", PropagationGroup::Intro),
            std::make_pair("constant", PropagationGroup::Intro),

            std::make_pair("add", PropagationGroup::Binary),
            std::make_pair("sub", PropagationGroup::Binary),
            std::make_pair("mul", PropagationGroup::Binary),
            std::make_pair("div", PropagationGroup::Binary),
            std::make_pair("min", PropagationGroup::Binary),
            std::make_pair("max", PropagationGroup::Binary),
            std::make_pair("pow", PropagationGroup::Binary),
            std::make_pair("lt",  PropagationGroup::Binary),
            std::make_pair("le",  PropagationGroup::Binary),
            std::make_pair("gt",  PropagationGroup::Binary),
            std::make_pair("ge",  PropagationGroup::Binary),
            std::make_pair("eq",  PropagationGroup::Binary),
            std::make_pair("ne",  PropagationGroup::Binary),
            std::make_pair("and", PropagationGroup::Binary),
            std::make_pair("or",  PropagationGroup::Binary),
            std::make_pair("min",  PropagationGroup::Binary),
            std::make_pair("max",  PropagationGroup::Binary),

            std::make_pair("idn", PropagationGroup::Unary),
            std::make_pair("neg", PropagationGroup::Unary),
            std::make_pair("rcp", PropagationGroup::Unary),
            std::make_pair("exp", PropagationGroup::Unary),
            std::make_pair("log", PropagationGroup::Unary),
            std::make_pair("abs", PropagationGroup::Unary),
            std::make_pair("sign", PropagationGroup::Unary),
            std::make_pair("floor", PropagationGroup::Unary),
            std::make_pair("ceil", PropagationGroup::Unary),
            std::make_pair("round", PropagationGroup::Unary),
            std::make_pair("sqr", PropagationGroup::Unary),
            std::make_pair("sqrt", PropagationGroup::Unary),
            std::make_pair("rsqr", PropagationGroup::Unary),
            std::make_pair("rsqrt", PropagationGroup::Unary),
            std::make_pair("not", PropagationGroup::Unary),
            std::make_pair("log2", PropagationGroup::Unary),

            std::make_pair("relu", PropagationGroup::Unary),
            std::make_pair("sigmoid", PropagationGroup::Unary),
            std::make_pair("tanh", PropagationGroup::Unary),
            std::make_pair("elu", PropagationGroup::Unary),
            std::make_pair("softabs", PropagationGroup::Unary),
            std::make_pair("softmax", PropagationGroup::Unary),
            std::make_pair("softplus", PropagationGroup::Unary),
            std::make_pair("leaky_relu", PropagationGroup::Unary),

            std::make_pair("linear_quantize", PropagationGroup::Unary),
            std::make_pair("logarithmic_quantize", PropagationGroup::Unary),
            std::make_pair("binary_quantize", PropagationGroup::Unary),
            std::make_pair("ternary_quantize", PropagationGroup::Unary),

            std::make_pair("conv", PropagationGroup::Conv),
            std::make_pair("box", PropagationGroup::Conv),
            std::make_pair("sample", PropagationGroup::Conv),
            std::make_pair("separable_conv", PropagationGroup::Conv),
            std::make_pair("planewise_conv", PropagationGroup::Conv),

            std::make_pair("deconv", PropagationGroup::Deconv),
            std::make_pair("debox", PropagationGroup::Deconv),
            std::make_pair("desample", PropagationGroup::Deconv),
            std::make_pair("separable_deconv", PropagationGroup::Deconv),
            std::make_pair("planewise_deconv", PropagationGroup::Deconv),

            std::make_pair("max_pool", PropagationGroup::Pool),
            std::make_pair("argmax_pool", PropagationGroup::Pool),
            std::make_pair("max_pool_with_index", PropagationGroup::Pool),
            std::make_pair("avg_pool", PropagationGroup::Pool),
            std::make_pair("rms_pool", PropagationGroup::Pool),

            std::make_pair("sum_reduce", PropagationGroup::Reduce),
            std::make_pair("min_reduce", PropagationGroup::Reduce),
            std::make_pair("max_reduce", PropagationGroup::Reduce),
            std::make_pair("mean_reduce", PropagationGroup::Reduce),
            std::make_pair("moments", PropagationGroup::Reduce),

            std::make_pair("nearest_downsample", PropagationGroup::DownSample),
            std::make_pair("area_downsample", PropagationGroup::DownSample),
            std::make_pair("nearest_upsample", PropagationGroup::UpSample),
            std::make_pair("multilinear_upsample", PropagationGroup::UpSample),

            std::make_pair("local_response_normalization", PropagationGroup::Normalize),
            std::make_pair("local_mean_normalization", PropagationGroup::Normalize),
            std::make_pair("local_variance_normalization", PropagationGroup::Normalize),
            std::make_pair("local_contrast_normalization", PropagationGroup::Normalize),
            std::make_pair("l1_normalization", PropagationGroup::Normalize),
            std::make_pair("l2_normalization", PropagationGroup::Normalize),
            std::make_pair("layer_normalization", PropagationGroup::Normalize),
            std::make_pair("divisive_normalization", PropagationGroup::Normalize),
            std::make_pair("batch_normalization", PropagationGroup::Normalize),

            std::make_pair("reshape", PropagationGroup::Unique),
            std::make_pair("transpose", PropagationGroup::Unique),
            std::make_pair("split", PropagationGroup::Unique),
            std::make_pair("concat", PropagationGroup::Unique),
            std::make_pair("select", PropagationGroup::Unique),
            std::make_pair("matmul", PropagationGroup::Unique),
            std::make_pair("linear", PropagationGroup::Unique),
            std::make_pair("update", PropagationGroup::Unique),
            std::make_pair("softmax", PropagationGroup::Unique),
            std::make_pair("copy_n", PropagationGroup::Unique),
            std::make_pair("add_n", PropagationGroup::Unique),
        };

        auto it = operationGroups.find(name);
        return it != operationGroups.end() ? it->second : PropagationGroup::Unknown;
    }


    class Propagation
    {
    public:

        enum { MaxRank = Shape::MaxRank };

    public:

        bool propagateShapes( const Prototype& proto, const Dictionary<Value> args, Dictionary<Shape>& shapes )
        {
            const std::string& op = proto.name();
            const PropagationGroup group = getPropagationGroup(op);
            switch ( group )
            {
                case PropagationGroup::Intro:
                {
                    propagateShapesIntro(proto, args, shapes);
                    if ( op == "variable" )
                    {
                        checkSharedVariableShapes(args);
                    }
                    break;
                }
                case PropagationGroup::Unary:
                {
                    propagateShapesUnary(proto, args, shapes);
                    break;
                }
                case PropagationGroup::Binary:
                {
                    propagateShapesBinary(proto, args, shapes);
                    break;
                }
                case PropagationGroup::Reduce:
                {
                    propagateShapesReduce(proto, args, shapes);
                    break;
                }
                case PropagationGroup::Conv:
                case PropagationGroup::Deconv:
                case PropagationGroup::Pool:
                {
                    propagateShapesSliding(proto, args, shapes, group);
                    break;
                }
                case PropagationGroup::DownSample:
                {
                    propagateShapesDownsample(proto, args, shapes);
                    break;
                }
                case PropagationGroup::UpSample:
                {
                    propagateShapesUpsample(proto, args, shapes);
                    break;
                }
                case PropagationGroup::Normalize:
                {
                    propagateShapesNormalize(proto, args, shapes);
                    break;
                }
                case PropagationGroup::Unique:
                {
                    if ( op == "reshape" )
                    {
                        propagateShapesReshape(proto, args, shapes);
                    }
                    else if ( op == "transpose" )
                    {
                        propagateShapesTranspose(proto, args, shapes);
                    }
                    else if ( op == "split" )
                    {
                        propagateShapesSplit(proto, args, shapes);
                    }
                    else if ( op == "concat" )
                    {
                        propagateShapesConcat(proto, args, shapes);
                    }
                    else if ( op == "select" )
                    {
                        propagateShapesSelect(proto, args, shapes);
                    }
                    else if ( op == "matmul" )
                    {
                        propagateShapesMatmul(proto, args, shapes);
                    }
                    else if ( op == "linear" )
                    {
                        propagateShapesLinear(proto, args, shapes);
                    }
                    else if ( op == "update" )
                    {
                        propagateShapesUpdate(proto, args, shapes);
                    }
                    else if ( op == "softmax" )
                    {
                        propagateShapesSoftmax(proto, args, shapes);
                    }
                    else if ( op == "copy_n" )
                    {
                        propagateShapesCopyN(proto, args, shapes);
                    }
                    else if ( op == "add_n" )
                    {
                        propagateShapesAddN(proto, args, shapes);
                    }
                    else
                    {
                        assert(false);
                    }
                    break;
                }
                case PropagationGroup::Unknown:
                {
                    return false;
                }
            }
            return true;
        }
        
        static size_t resultArrayLength( const Prototype& proto, const Dictionary<Value>& args, const size_t idx )
        {
            if ( proto.name() == "split" && idx == 0 )
            {
                return args["ratios"].size();
            }
            else if ( proto.name() == "copy_n" && idx == 0 )
            {
                return args["times"].integer();
            }
            return 0;
        }

    private:

        static void propagateShapesIntro( const Prototype& proto, const Dictionary<Value> args, Dictionary<Shape>& shapes )
        {
            auto& shape = args["shape"];
            auto& output = args["output"];

            checkRange("shape", shape, proto.name() == "external" ? 0 : 1);

            auto outputShape = makeShape(shape);

            auto& value = args["value"];
            if ( value && value.size() != outputShape.volume() && value.size() != 1 )
            {
                throw Error("shape volume (%d) does not match number of values (%d)", (int)outputShape.volume(), (int)value.size());
            }

            setShape(output, shapes, outputShape);
        }

        static void propagateShapesUnary( const Prototype& proto, const Dictionary<Value> args, Dictionary<Shape>& shapes )
        {
            auto& x = args["x"];
            auto& y = args["y"];

            setShape(y, shapes, getShape(x, shapes));
        }

        static void propagateShapesBinary( const Prototype& proto, const Dictionary<Value> args, Dictionary<Shape>& shapes )
        {
            auto& x = args["x"];
            auto& y = args["y"];
            auto& z = args["z"];

            auto& xShape = getShape(x, shapes);
            auto& yShape = getShape(y, shapes);

            if ( !isBroadcastCompatible(xShape, yShape) )
            {
                throw Error("incompatible tensor shapes in binary operation (%s vs %s)",
                                 xShape.toString().c_str(), yShape.toString().c_str());
            }

            setShape(z, shapes, broadcastShape(xShape, yShape));
        }

        static void propagateShapesReduce( const Prototype& proto, const Dictionary<Value> args, Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& axes = args["axes"];

            checkAxes("axes", axes);

            Shape outputShape = getShape(input, shapes);

            for ( size_t i = 0; i < axes.size(); ++i )
            {
                auto axis = axes[i].integer();
                outputShape[axis] = 1;
            }

            if ( output )
            {
                setShape(output, shapes, outputShape);
            }
            else
            {
                setShape(args["mean"], shapes, outputShape);
                setShape(args["variance"], shapes, outputShape);
            }
        }

        static void propagateShapesSliding( const Prototype& proto, const Dictionary<Value> args, Dictionary<Shape>& shapes,
                                           const PropagationGroup group )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& size = args["size"];
            auto& filter = args["filter"];
            auto& planeFilter = args["plane_filter"];
            auto& pointFilter = args["point_filter"];
            auto& padding = args["padding"];
            auto& stride = args["stride"];
            auto& dilation = args["dilation"];

            const std::string& op = proto.name();
            bool convolutional = op.length() >= 4 && op.substr(op.length() - 4) == "conv";
            bool separable = pointFilter && planeFilter;
            const size_t offset = convolutional ? 2 : 0;

            if ( size )
            {
                checkRank("size", size, MaxRank - offset);
                checkRange("size", size, 0);
            }

            checkRank("padding", padding, MaxRank - offset);
            checkRank("stride", stride, MaxRank - offset);
            checkRank("dilation", dilation, MaxRank - offset);

            checkRange("padding", stride, 0);
            checkRange("stride", stride, 1);
            checkRange("dilation", dilation, 1);

            auto& groups = args["groups"];
            if ( groups )
            {
                checkRange("groups", groups, 0);
            }

            auto& inputShape = getShape(input, shapes);
            auto strideShape = makeShape(stride, offset);
            auto dilationShape = makeShape(dilation, offset);
            auto paddingShapes = makePadding(padding, offset);

            Shape kernelShape;
            if ( size )
            {
                kernelShape = makeShape(size, offset);
            }
            else if ( filter )
            {
                kernelShape = getShape(filter, shapes);
            }
            else if ( separable )
            {
                auto& planeShape = getShape(planeFilter, shapes);
                auto& pointShape = getShape(pointFilter, shapes);
                kernelShape = makeSeparableFilterShape(planeShape, pointShape, inputShape[1]);
            }
            else
            {
                throw Error("kernel shape is not defined for operation '%s'", op.c_str());
            }

            Shape outputShape;

            if ( group == PropagationGroup::Conv || group == PropagationGroup::Pool )
            {
                makeDownscalePadding(inputShape, kernelShape, strideShape, dilationShape, offset + padding.size(), paddingShapes);

                if ( offset )
                {
                    if ( inputShape[1] )
                    {
                        const auto group_count = groups && groups.integer() != 0 ? groups.integer() : inputShape[1];
                        if ( inputShape[1] != kernelShape[1] * group_count )
                        {
                            throw Error("filter channels (%d) does not match input channels (%d)", (int)kernelShape[1], (int)inputShape[1]);
                        }
                        if ( kernelShape[0] % group_count )
                        {
                            throw Error("filter batch (%d) must be divisible by groups (%d)", (int)kernelShape[0], (int)group_count);
                        }
                    }
                    outputShape[0] = inputShape[0];
                    outputShape[1] = kernelShape[0];
                }

                for ( size_t i = offset; i < MaxRank; ++i )
                {
                    if ( inputShape[i] )
                    {
                        outputShape[i] = convExtent(inputShape[i], kernelShape[i], paddingShapes.first[i], paddingShapes.second[i],
                                                    strideShape[i], dilationShape[i]);
                    }
                }
            }
            else if ( group == PropagationGroup::Deconv )
            {
                makeUpscalePadding(inputShape, kernelShape, strideShape, dilationShape, offset + padding.size(), paddingShapes);

                if ( offset )
                {
                    const auto group_count = groups && groups.integer() != 0 ? groups.integer() : inputShape[1];
                    if ( inputShape[1] )
                    {
                        if ( inputShape[1] != kernelShape[0] )
                        {
                            throw Error("filter batch (%d) does not match input channels (%d)", (int)kernelShape[0], (int)inputShape[1]);
                        }
                        if ( kernelShape[0] % group_count )
                        {
                            throw Error("filter batch (%d) must be divisible by groups (%d)", (int)kernelShape[0], (int)group_count);
                        }
                    }
                    outputShape[0] = inputShape[0];
                    outputShape[1] = kernelShape[1] * group_count;
                }

                for ( size_t i = offset; i < MaxRank; ++i )
                {
                    if ( inputShape[i] )
                    {
                        outputShape[i] = deconvExtent(inputShape[i], kernelShape[i], paddingShapes.first[i], paddingShapes.second[i],
                                                      strideShape[i], dilationShape[i]);
                    }
                }
            }

            auto& bias = args["bias"];
            if ( bias )
            {
                const Shape& biasShape = getShape(bias, shapes);
                for ( size_t i = 0; i < MaxRank; ++i )
                {
                    if ( i != 1 && biasShape[i] && biasShape[i] != 1 )
                    {
                        throw Error("bias shape must be singular for non-channel dimensions");
                    }
                }
                if ( biasShape[1] != outputShape[1] && biasShape[1] != 1 )
                {
                    throw Error("bias channels (%d) does not match output channels (%d)", (int)biasShape[1], (int)outputShape[1]);
                }
            }

            auto& index = args["index"];
            if ( index )
            {
                if ( op == "argmax_pool" || op == "max_pool_with_index" )
                {
                    setShape(index, shapes, outputShape);
                }
                else
                {
                    const Shape& indexShape = getShape(index, shapes);

                    if ( op == "sample" && indexShape != outputShape )
                    {
                        throw Error("index shape %s does not match output shape %s",
                                    indexShape.toString().c_str(), outputShape.toString().c_str());
                    }
                    else if ( op == "desample" && indexShape != inputShape )
                    {
                        throw Error("index shape %s does not match input shape %s",
                                    indexShape.toString().c_str(), inputShape.toString().c_str());
                    }
                }
            }

            if ( output )
            {
                setShape(output, shapes, outputShape);
            }
        }

        static void propagateShapesReshape( const Prototype& proto, const Dictionary<Value> args,
                                           Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& shape = args["shape"];

            auto& inputShape = getShape(input, shapes);
            Shape outputShape = makeShape(shape);

            checkRange("shape", shape, -1);

            size_t autoAxis = MaxRank;
            for ( size_t i = 0; i < MaxRank; ++i )
            {
                if ( outputShape[i] == 0 )
                {
                    outputShape[i] = inputShape[i];
                }
                else if ( outputShape[i] == -1 )
                {
                    if ( autoAxis != MaxRank )
                    {
                        throw Error("shape may only contain at most one -1 value");
                    }
                    outputShape[i] = 1;
                    autoAxis = i;
                }
            }

            auto inputVolume = inputShape.volume();
            auto outputVolume = outputShape.volume();

            if ( autoAxis != MaxRank )
            {
                if ( inputVolume % outputVolume )
                {
                    throw Error("automatic output shape (%s) incompatible with input shape (%s)", (int)outputVolume, (int)inputVolume);
                }
                outputShape[autoAxis] = (Shape::extent_type)(inputVolume / outputVolume);
            }
            else if ( inputVolume != outputVolume )
            {
                throw Error("input volume (%d) does not equal output volume (%d)", (int)inputVolume, (int)outputVolume);
            }

            setShape(output, shapes, outputShape);
        }

        static void propagateShapesTranspose( const Prototype& proto, const Dictionary<Value> args,
                                             Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& perm = args["perm"];

            auto& inputShape = getShape(input, shapes);
            Shape outputShape = inputShape;

            auto rank = perm.size();

            size_t axes[MaxRank];
            for ( size_t i = 0; i < rank; ++i )
            {
                axes[i] = perm[i].integer();
            }

            std::sort(axes, axes + rank);
            for ( size_t i = 0; i < rank; ++i )
            {
                if ( axes[i] != i )
                {
                    throw Error("'perm' array must contain a permutation of dimensions from 0 to %d", (int)rank);
                }
            }

            for ( size_t i = 0; i < rank; ++i )
            {
                auto j = perm[i].integer();
                outputShape[i] = inputShape[j];
            }

            setShape(output, shapes, outputShape);
        }

        static void propagateShapesSplit( const Prototype& proto, const Dictionary<Value> args,
                                         Dictionary<Shape>& shapes )
        {
            auto& value = args["value"];
            auto& values = args["values"];
            auto& axis = args["axis"];
            auto& ratios = args["ratios"];

            checkAxis("axis", axis);
            checkRange("ratios", ratios, 1);

            auto& wholeShape = getShape(value, shapes);

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

                setShape(values[i], shapes, itemShape);
            }
        }

        static void propagateShapesConcat( const Prototype& proto, const Dictionary<Value> args,
                                          Dictionary<Shape>& shapes )
        {
            auto& values = args["values"];
            auto& value = args["value"];
            auto& axis = args["axis"];

            checkAxis("axis", axis);

            const size_t idx = axis.integer();

            Shape outputShape = getShape(values[0], shapes);

            bool compatibleShape = true;
            for ( size_t i = 1; i < values.size(); ++i )
            {
                auto& partShape = getShape(values[i], shapes);

                for ( size_t i = 0; i < MaxRank; ++i )
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

            setShape(value, shapes, outputShape);
        }

        static void propagateShapesUpsample( const Prototype& proto, const Dictionary<Value> args,
                                            Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& factor = args["factor"];

            Shape outputShape = getShape(input, shapes);
            for ( size_t i = 0; i < factor.size(); ++i )
            {
                auto scale = factor[i].integer();
                outputShape[i+2] *= scale;
            }

            setShape(output, shapes, outputShape);
        }

        static void propagateShapesDownsample( const Prototype& proto, const Dictionary<Value> args,
                                              Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& factor = args["factor"];

            Shape outputShape = getShape(input, shapes);
            for ( size_t i = 0; i < factor.size(); ++i )
            {
                auto scale = factor[i].integer();
                if ( outputShape[i+2] % scale )
                {
                    throw Error("input extent (%d) must be divisible by factor (%d)", (int)outputShape[i+2], (int)scale);
                }

                outputShape[i+2] /= scale;
            }

            setShape(output, shapes, outputShape);
        }

        static void propagateShapesNormalize( const Prototype& proto, const Dictionary<Value> args,
                                             Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];

            auto& axes = args["axes"];
            if ( axes )
            {
                checkAxes("name", axes);
            }

            auto& size = args["size"];
            if ( size )
            {
                checkRank("size", size);
                checkRange("size", size, 1);
            }

            setShape(output, shapes, getShape(input, shapes));
        }

        static void propagateShapesSelect( const Prototype& proto, const Dictionary<Value> args,
                                          Dictionary<Shape>& shapes )
        {
            auto& condition = args["condition"];
            auto& trueValue = args["true_value"];
            auto& falseValue = args["false_value"];
            auto& output = args["output"];

            Shape valueShape;
            if ( trueValue.kind() == Value::Tensor && falseValue.kind() == Value::Tensor )
            {
                auto& trueShape = getShape(trueValue, shapes);
                auto& falseShape = getShape(falseValue, shapes);

                for ( size_t i = 0; i < MaxRank; ++i )
                {
                    if ( trueShape[i] && falseShape[i] )
                    {
                        if ( trueShape[i] != falseShape[i] )
                        {
                            throw Error("incompatible result shapes in select operation (%s vs %s)",
                                  trueShape.toString().c_str(), falseShape.toString().c_str());
                        }
                        else
                        {
                            valueShape[i] = trueShape[i];
                        }
                    }
                }
            }
            else if ( trueValue.kind() == Value::Tensor && falseValue.kind() != Value::Tensor )
            {
                valueShape = getShape(trueValue, shapes);
            }
            else if ( trueValue.kind() != Value::Tensor && falseValue.kind() == Value::Tensor )
            {
                valueShape = getShape(falseValue, shapes);
            }
            else
            {
                valueShape = Shape::singleton();
            }

            auto& conditionShape = getShape(condition, shapes);

            Shape outputShape;

            bool compatible = true;
            for ( size_t i = 0; i < MaxRank; ++i )
            {
                if ( conditionShape[i] && valueShape[i] )
                {
                    compatible &= (conditionShape[i] == valueShape[i] || conditionShape[i] == 1 || valueShape[i] == 1);
                    outputShape[i] = std::max(conditionShape[i], valueShape[i]);
                }
            }
            if ( !compatible )
            {
                throw Error("condition shape incompatible with result shape in select operation");
            }

            setShape(output, shapes, outputShape);
        }

        static void propagateShapesMatmul( const Prototype& proto, const Dictionary<Value> args,
                                          Dictionary<Shape>& shapes )
        {
            auto& A = args["A"];
            auto& B = args["B"];
            auto& C = args["C"];
            auto& trA = args["trA"];
            auto& trB = args["trB"];

            auto& aShape = getShape(A, shapes);
            auto& bShape = getShape(B, shapes);

            auto m = trA.logical() ? aShape[1] : aShape[0];
            auto n = trB.logical() ? bShape[0] : bShape[1];
            auto kA = trA.logical() ? aShape[0] : aShape[1];
            auto kB = trB.logical() ? bShape[1] : bShape[0];

            if ( kA != kB )
            {
                throw Error("inner dimensions must agree (%d vs %d)", (int)kA, (int)kB);
            }

            for ( size_t i = 2; i < MaxRank; ++i )
            {
                if ( aShape[i] != 1 || bShape[i] != 1 )
                {
                    throw Error("argument shapes must be singleton for dimension > 2");
                }
            }

            Shape cShape = Shape::singleton();
            cShape[0] = m;
            cShape[1] = n;

            setShape(C, shapes, cShape);
        }

        static void propagateShapesLinear( const Prototype& proto, const Dictionary<Value> args,
                                          Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& filter = args["filter"];

            auto& inputShape = getShape(input, shapes);
            auto& filterShape = getShape(filter, shapes);

            if ( inputShape[1] != filterShape[1] )
            {
                throw Error("inner dimensions must agree (%d vs %d)", (int)inputShape[1], (int)filterShape[1]);
            }

            Shape outputShape = Shape::singleton();
            outputShape[0] = inputShape[0];
            outputShape[1] = filterShape[0];

            setShape(output, shapes, outputShape);
        }

        static void propagateShapesUpdate( const Prototype& proto, const Dictionary<Value> args,
                                          Dictionary<Shape>& shapes )
        {
            auto& variable = args["variable"];
            auto& result = args["result"];
            auto& value = args["value"];

            auto& varShape = getShape(variable, shapes);
            auto& valShape = getShape(value, shapes);

            if ( valShape != varShape )
            {
                throw Error("updated shape %s does not equal variable shape %s", valShape.toString().c_str(), varShape.toString().c_str());
            }

            setShape(result, shapes, varShape);
        }

        static void propagateShapesSoftmax( const Prototype& proto, const Dictionary<Value> args,
                                           Dictionary<Shape>& shapes )
        {
            propagateShapesUnary(proto, args, shapes);
            
            auto& axes = args["axes"];
            checkAxes("axes", axes);
        }

        static void propagateShapesCopyN( const Prototype& proto, const Dictionary<Value> args,
                                         Dictionary<Shape>& shapes )
        {
            auto& times = args["times"];
            checkRange("times", times, 1);

            auto& x = args["x"];
            auto& y = args["y"];
            auto& shape = getShape(x, shapes);

            if ( (size_t)times.integer() != y.size() )
            {
                throw Error("argument times (%d) does not equal length of y", (int)times.integer(), (int)y.size());
            }

            setShape(y, shapes, shape);
        }

        static void propagateShapesAddN( const Prototype& proto, const Dictionary<Value> args,
                                        Dictionary<Shape>& shapes )
        {
            auto& x = args["x"];
            auto& y = args["y"];

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

            setShape(y, shapes, yShape);
        }

    private:

        static const Shape& getShape( const Value& arg, const Dictionary<Shape>& shapes )
        {
            return arg.kind() == Value::Tensor ? shapes[arg.tensor()] : Shape::singleton();
        }

        static void setShape( const Value& arg, Dictionary<Shape>& shapes, const Shape& shape )
        {
            if ( arg.kind() == Value::Tuple || arg.kind() == Value::Array )
            {
                for ( size_t i = 0; i < arg.size(); ++i )
                {
                    setShape(arg[i], shapes, shape);
                }
            }
            else
            {
                shapes[arg.tensor()] = shape;
            }
        }

        static Shape makeShape( const Value& arg, const size_t offset = 0 )
        {
            Shape shape = Shape(1);

            for ( size_t i = 0; i < arg.size(); ++i )
            {
                shape[i + offset] = arg[i].integer();
            }
            return shape;
        }

        static std::pair<Shape,Shape> makePadding( const Value& arg, const size_t offset = 0 )
        {
            std::pair<Shape,Shape> padding = std::make_pair(Shape(0), Shape(0));

            for ( size_t i = 0; i < arg.size(); ++i )
            {
                padding.first[i + offset] = arg[i][0].integer();
                padding.second[i + offset] = arg[i][1].integer();
            }

            return padding;
        }

        static void makeDownscalePadding( const Shape& inputShape, const Shape& kernelShape,
                                         const Shape& strideShape, const Shape& dilationShape,
                                         const size_t offset, std::pair<Shape,Shape>& paddingShape )
        {
            for ( size_t i = offset; i < MaxRank; ++i )
            {
                auto outputSize = ceildiv(inputShape[i], strideShape[i]);
                auto window = (kernelShape[i] - 1) * dilationShape[i] + 1;
                auto total = (outputSize - 1) * strideShape[i] + window - inputShape[i];

                paddingShape.first[i] = total / 2;
                paddingShape.second[i] = total - total / 2;
            }
        }

        static void makeUpscalePadding( const Shape& inputShape, const Shape& kernelShape,
                                       const Shape& strideShape, const Shape& dilationShape,
                                       const size_t offset, std::pair<Shape,Shape>& paddingShape )
        {
            for ( size_t i = offset; i < MaxRank; ++i )
            {
                auto window = (kernelShape[i] - 1) * dilationShape[i] + 1;
                auto total = window - strideShape[i];

                paddingShape.first[i] = total / 2;
                paddingShape.second[i] = total - total / 2;
            }
        }

        static Shape makeSeparableFilterShape( const Shape& planeShape, const Shape& pointShape, const int channels )
        {
            for ( size_t i = 2; i < MaxRank; ++i )
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
            shape[1] = channels;
            return shape;
        }

    private:

        static bool isBroadcastCompatible( const Shape& xShape, const Shape& yShape )
        {
            for ( size_t i = 0; i < MaxRank; ++i )
            {
                if ( !(xShape[i] == yShape[i] || xShape[i] == 1 || yShape[i] == 1) )
                {
                    return false;
                }
            }
            return true;
        }

        static Shape broadcastShape( const Shape& xShape, const Shape& yShape )
        {
            Shape zShape;
            for ( size_t i = 0; i < MaxRank; ++i )
            {
                zShape[i] = xShape[i] && yShape[i] ? std::max(xShape[i], yShape[i]) : 0;
            }
            return zShape;
        }

        static void checkAxis( const char* name, const Value& value )
        {
            auto axis = value.integer();
            if ( axis < 0 || axis >= MaxRank )
            {
                throw Error("'%s' must be in range [0,%d)", name, (int)MaxRank);
            }
        }

        static void checkAxes( const char* name, const Value& value )
        {
            bool seen[MaxRank];
            std::fill_n(seen, (size_t)MaxRank, false);

            for ( size_t i = 0; i < value.size(); ++i )
            {
                auto& item = value[i];
                checkAxis(name, item);

                auto axis = item.integer();
                if ( seen[axis] )
                {
                    throw Error("duplicate item '%d' in array '%s'", (int)axis, name);
                }
                seen[axis] = true;
            }
        }

        static void checkRange( const char* name, const Value& value, const int min )
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

        static void checkRank( const char* name, const Value& value, const size_t maxRank = MaxRank )
        {
            if ( value.size() > maxRank )
            {
                throw Error("length of array '%s' must be <= maximum supported tensor dimensions (%d)", name, (int)maxRank);
            }
        }

        template<typename T>
        static T convExtent( const T input, const T size, const T front_padding, const T back_padding, const T stride, const T dilation )
        {
            const T window = 1 + (size - 1) * dilation;
            return (input + front_padding + back_padding - window) / stride + 1;
        }

        template<typename T>
        static T deconvExtent( const T output, const T size, const T front_padding, const T back_padding, const T stride, const T dilation )
        {
            const T window = 1 + (size - 1) * dilation;
            return (output - 1) * stride + window - (front_padding + back_padding);
        }

        template<typename T>
        static T ceildiv( const T x, const T y )
        {
            return (x + y - 1) / y;
        }

    private:

        void checkSharedVariableShapes( const Dictionary<Value> args )
        {
            auto& label = args["label"];
            auto& shape = args["shape"];

            const Shape variableShape = makeShape(shape);

            auto it = _variableShapes.find(label.string());
            if ( it != _variableShapes.end() )
            {
                if ( it->second != variableShape )
                {
                    throw Error("variable shape %s does not match previously defined shape %s for label '%s'",
                                variableShape.toString().c_str(), it->second.toString().c_str(), label.string().c_str());
                }
            }
            else
            {
                _variableShapes.emplace(label.string(), variableShape);
            }
        }

    public:

        const Dictionary<Shape>& variableShapes() const
        {
            return _variableShapes;
        }

    private:

        Dictionary<Shape> _variableShapes;
    };

}   // namespace nnef


#endif
