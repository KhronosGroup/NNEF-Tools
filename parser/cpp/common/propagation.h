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
#include <string>
#include <cctype>


namespace nnef
{

    enum class PropagationGroup
    {
        Unknown, Intro, Unary, Binary, Reduce, Conv, Deconv, Pool, UpSample, DownSample, Normalize, Roi, Unique
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

            std::make_pair("copy", PropagationGroup::Unary),
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
            std::make_pair("prelu", PropagationGroup::Unary),

            std::make_pair("linear_quantize", PropagationGroup::Unary),
            std::make_pair("logarithmic_quantize", PropagationGroup::Unary),
            std::make_pair("binary_quantize", PropagationGroup::Unary),
            std::make_pair("ternary_quantize", PropagationGroup::Unary),

            std::make_pair("conv", PropagationGroup::Conv),
            std::make_pair("box", PropagationGroup::Conv),
            std::make_pair("sample", PropagationGroup::Conv),
            std::make_pair("separable_conv", PropagationGroup::Conv),

            std::make_pair("deconv", PropagationGroup::Deconv),
            std::make_pair("debox", PropagationGroup::Deconv),
            std::make_pair("desample", PropagationGroup::Deconv),
            std::make_pair("separable_deconv", PropagationGroup::Deconv),

            std::make_pair("max_pool", PropagationGroup::Pool),
            std::make_pair("argmax_pool", PropagationGroup::Pool),
            std::make_pair("max_pool_with_index", PropagationGroup::Pool),
            std::make_pair("avg_pool", PropagationGroup::Pool),
            std::make_pair("rms_pool", PropagationGroup::Pool),

            std::make_pair("sum_reduce", PropagationGroup::Reduce),
            std::make_pair("min_reduce", PropagationGroup::Reduce),
            std::make_pair("max_reduce", PropagationGroup::Reduce),
            std::make_pair("mean_reduce", PropagationGroup::Reduce),
            std::make_pair("argmax_reduce", PropagationGroup::Reduce),
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
            std::make_pair("batch_normalization", PropagationGroup::Normalize),

            std::make_pair("avg_roi_pool", PropagationGroup::Roi),
            std::make_pair("max_roi_pool", PropagationGroup::Roi),
            std::make_pair("avg_roi_align", PropagationGroup::Roi),
            std::make_pair("max_roi_align", PropagationGroup::Roi),
            std::make_pair("roi_resample", PropagationGroup::Roi),

            std::make_pair("reshape", PropagationGroup::Unique),
            std::make_pair("transpose", PropagationGroup::Unique),
            std::make_pair("split", PropagationGroup::Unique),
            std::make_pair("concat", PropagationGroup::Unique),
            std::make_pair("slice", PropagationGroup::Unique),
            std::make_pair("stack", PropagationGroup::Unique),
            std::make_pair("unstack", PropagationGroup::Unique),
            std::make_pair("squeeze", PropagationGroup::Unique),
            std::make_pair("unsqueeze", PropagationGroup::Unique),
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

        Propagation( const size_t maxRank )
        : _maxRank(maxRank)
        {
        }

        void reset()
        {
            _variableShapes.clear();
        }

        virtual void propagateShapes( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            const std::string& op = proto.name();
            const PropagationGroup group = getPropagationGroup(op);
            switch ( group )
            {
                case PropagationGroup::Intro:
                {
                    propagateShapesIntro(proto, args, shapes);
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
                case PropagationGroup::Roi:
                {
                    propagateShapesRoi(proto, args, shapes);
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
                    else if ( op == "slice" )
                    {
                        propagateShapesSlice(proto, args, shapes);
                    }
                    else if ( op == "stack" )
                    {
                        propagateShapesStack(proto, args, shapes);
                    }
                    else if ( op == "unstack" )
                    {
                        propagateShapesUnstack(proto, args, shapes);
                    }
                    else if ( op == "squeeze" )
                    {
                        propagateShapesSqueeze(proto, args, shapes);
                    }
                    else if ( op == "unsqueeze" )
                    {
                        propagateShapesUnsqueeze(proto, args, shapes);
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
                    throw Error("shape propagation not defined for operation '%s'", proto.name().c_str());
                }
            }
        }
        
        virtual size_t resultArrayLength( const Prototype& proto, const std::string& result, const Dictionary<Value>& args, const Dictionary<Shape>& shapes )
        {
            if ( proto.name() == "split" && result == "values" )
            {
                return args["ratios"].size();
            }
            else if ( proto.name() == "unstack" && result == "values" )
            {
                auto& shape = getShape(args["input"], shapes);
                return shape[args["axis"].integer()];
            }
            else if ( proto.name() == "copy_n" && result == "y" )
            {
                return args["times"].integer();
            }
            return 0;
        }

        virtual bool shouldDeferShapeOf( const Prototype& proto, const std::string& param ) const
        {
            return false;
        }

    public:

        void propagateShapesIntro( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& shape = args["shape"];
            auto& output = args["output"];

            if ( shape.kind() != Value::ShapeOf )
            {
                checkRange("shape", shape, 1);
            }

            auto outputShape = shape.kind() == Value::ShapeOf ? getShape(shape, shapes) : makeShape(shape);

            checkMaxRank("output", outputShape);

            auto& value = args["value"];
            if ( value && value.size() != outputShape.volume() && value.size() != 1 )
            {
                throw Error("shape volume (%d) does not match number of values (%d)", (int)outputShape.volume(), (int)value.size());
            }

            auto& label = args["label"];
            if ( label )
            {
                auto& str = label.string();
                auto cit = std::find_if_not(str.begin(), str.end(), isLabelChar);
                if ( cit != str.end() )
                {
                    throw Error("labels must only contain alphanumeric ascii characters, and the characters '/\\-_.'; found '%c'", *cit);
                }

                checkSharedVariableShapes(str, outputShape);
            }

            setShape(output, shapes, outputShape);
        }

        void propagateShapesUnary( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& x = args["x"];
            auto& y = args["y"];

            setShape(y, shapes, getShape(x, shapes));
        }

        void propagateShapesBinary( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
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

        void propagateShapesReduce( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& axes = args["axes"];

            Shape outputShape = getShape(input, shapes);

            checkAxes("axes", axes, outputShape.rank());

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

        void propagateShapesSliding( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes, const PropagationGroup group )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& size = args["size"];
            auto& filter = args["filter"];
            auto& plane_filter = args["plane_filter"];
            auto& point_filter = args["point_filter"];
            auto& padding = args["padding"];
            auto& stride = args["stride"];
            auto& dilation = args["dilation"];
            auto& output_shape = args["output_shape"];

            const std::string& op = proto.name();
            bool convolutional = op.length() >= 4 && op.substr(op.length() - 4) == "conv";
            bool separable = point_filter && plane_filter;
            const size_t offset = convolutional ? 2 : 0;

            auto& inputShape = getShape(input, shapes);
            const size_t rank = inputShape.rank();

            if ( size )
            {
                checkRank("size", size, rank - offset);
                checkRange("size", size, 0);
            }

            if ( padding.size() )
            {
                checkRank("padding", padding, rank - offset);
            }
            if ( stride.size() )
            {
                checkRank("stride", stride, rank - offset);
            }
            if ( dilation.size() )
            {
                checkRank("dilation", dilation, rank - offset);
            }

            checkRange("stride", stride, 1);
            checkRange("dilation", dilation, 1);

            auto& groups = args["groups"];
            if ( groups )
            {
                checkRange("groups", groups, 0);
            }
            const auto groupCount = groups && groups.integer() != 0 ? groups.integer() : inputShape[1];

            auto strideShape = makeShape(stride, stride.size() ? offset : rank);
            auto dilationShape = makeShape(dilation, dilation.size() ? offset : rank);
            auto paddingShapes = makePadding(padding, padding.size() ? offset : rank);

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
                auto& planeShape = getShape(plane_filter, shapes);
                auto& pointShape = getShape(point_filter, shapes);
                kernelShape = makeSeparableFilterShape(planeShape, pointShape, inputShape[1]);
            }
            else
            {
                throw Error("kernel shape is not defined for operation '%s'", op.c_str());
            }

            if ( kernelShape.rank() != inputShape.rank() )
            {
                throw Error("kernel rank incompatible with input rank (%d vs %d)", (int)kernelShape.rank(), (int)inputShape.rank());
            }

            Shape outputShape(inputShape.rank());
            bool outputShapeSupplied = false;

            if ( output_shape )
            {
                outputShapeSupplied = true;

                if ( output_shape.kind() == Value::ShapeOf )
                {
                    outputShape = getShape(output_shape, shapes);

                    if ( outputShape.rank() != inputShape.rank() )
                    {
                        throw Error("rank of supplied output shape (%d) does not match that of input shape (%d)",
                                    (int)outputShape.rank(), (int)inputShape.rank());
                    }
                }
                else if ( output_shape.size() )
                {
                    checkRank("output_shape", output_shape, rank);
                    checkRange("output_shape", output_shape, 1);

                    outputShape = makeShape(output_shape);
                }
                else
                {
                    outputShapeSupplied = false;
                }
            }

            if ( group == PropagationGroup::Conv || group == PropagationGroup::Pool )
            {
                if ( !padding.size() )
                {
                    makeDownscalePadding(inputShape, kernelShape, strideShape, dilationShape, offset, paddingShapes);
                }
                if ( offset )
                {
                    if ( inputShape[1] != kernelShape[1] * groupCount )
                    {
                        throw Error("filter channels (%d) does not match input channels (%d)", (int)kernelShape[1], (int)inputShape[1]);
                    }
                    if ( kernelShape[0] % groupCount )
                    {
                        throw Error("filter batch (%d) must be divisible by groups (%d)", (int)kernelShape[0], (int)groupCount);
                    }

                    outputShape[0] = inputShape[0];
                    outputShape[1] = kernelShape[0];
                }

                for ( size_t i = offset; i < rank; ++i )
                {
                    outputShape[i] = convExtent(inputShape[i], kernelShape[i], paddingShapes.first[i], paddingShapes.second[i],
                                                strideShape[i], dilationShape[i]);
                }
            }
            else if ( group == PropagationGroup::Deconv )
            {
                if ( !padding.size() )
                {
                    makeUpscalePadding(inputShape, kernelShape, strideShape, dilationShape, offset, paddingShapes);
                }
                if ( offset )
                {
                    if ( inputShape[1] != kernelShape[0] )
                    {
                        throw Error("filter batch (%d) does not match input channels (%d)", (int)kernelShape[0], (int)inputShape[1]);
                    }
                    if ( kernelShape[0] % groupCount )
                    {
                        throw Error("filter batch (%d) must be divisible by groups (%d)", (int)kernelShape[0], (int)groupCount);
                    }
                }

                if ( outputShapeSupplied )
                {
                    Shape expectedInputShape(outputShape.rank());
                    if ( offset )
                    {
                        expectedInputShape[0] = outputShape[0];
                        expectedInputShape[1] = kernelShape[0];
                    }
                    for ( size_t i = offset; i < rank; ++i )
                    {
                        expectedInputShape[i] = convExtent(outputShape[i], kernelShape[i], paddingShapes.first[i], paddingShapes.second[i],
                                                            strideShape[i], dilationShape[i]);
                    }

                    if ( expectedInputShape != inputShape )
                    {
                        throw Error("expected input shape %s derived from output shape is incompatible with actual input shape %s",
                                    expectedInputShape.toString().c_str(), inputShape.toString().c_str());
                    }
                }
                else
                {
                    if ( offset )
                    {
                        outputShape[0] = inputShape[0];
                        outputShape[1] = kernelShape[1] * groupCount;
                    }
                    for ( size_t i = offset; i < rank; ++i )
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
                if ( biasShape.rank() != 0 && biasShape.rank() != 2 )
                {
                    throw Error("bias must be of rank 0 or 2");
                }
                if ( biasShape.rank() == 2 )
                {
                    if ( biasShape[0] != 1 )
                    {
                        throw Error("bias shape must be singular for the batch dimension");
                    }
                    if ( biasShape[1] != outputShape[1] && biasShape[1] != 1 )
                    {
                        throw Error("bias channels (%d) does not match output channels (%d)", (int)biasShape[1], (int)outputShape[1]);
                    }
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

        void propagateShapesConv( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            propagateShapesSliding(proto, args, shapes, PropagationGroup::Conv);
        }

        void propagateShapesDeconv( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            propagateShapesSliding(proto, args, shapes, PropagationGroup::Deconv);
        }

        void propagateShapesPool( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            propagateShapesSliding(proto, args, shapes, PropagationGroup::Pool);
        }

        void propagateShapesReshape( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& shape = args["shape"];

            if ( shape.kind() != Value::ShapeOf )
            {
                checkRange("shape", shape, -1);
            }

            auto& inputShape = getShape(input, shapes);
            Shape outputShape = shape.kind() == Value::ShapeOf ? getShape(shape, shapes) : makeShape(shape);

            checkMaxRank("output", outputShape);

            size_t autoAxis = std::numeric_limits<size_t>::max();
            for ( size_t i = 0; i < outputShape.rank(); ++i )
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

            auto inputVolume = inputShape.volume();
            auto outputVolume = outputShape.volume();

            if ( autoAxis != std::numeric_limits<size_t>::max() )
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

        void propagateShapesTranspose( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& axes = args["axes"];

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

            setShape(output, shapes, outputShape);
        }

        void propagateShapesSplit( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& value = args["value"];
            auto& values = args["values"];
            auto& axis = args["axis"];
            auto& ratios = args["ratios"];

            auto& wholeShape = getShape(value, shapes);

            checkAxis("axis", axis, wholeShape.rank());
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

                setShape(values[i], shapes, itemShape);
            }
        }

        void propagateShapesConcat( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& values = args["values"];
            auto& value = args["value"];
            auto& axis = args["axis"];

            Shape outputShape = getShape(values[0], shapes);

            checkAxis("axis", axis, outputShape.rank());

            const size_t idx = axis.integer();

            bool compatibleShape = true;
            for ( size_t i = 1; i < values.size(); ++i )
            {
                auto& partShape = getShape(values[i], shapes);

                if ( partShape.rank() != outputShape.rank() )
                {
                    compatibleShape = false;
                    break;
                }

                for ( size_t i = 0; i < outputShape.rank(); ++i )
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

        void propagateShapesSlice( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& axes = args["axes"];
            auto& begin = args["begin"];
            auto& end = args["end"];

            if ( begin.size() != axes.size() || end.size() != axes.size() )
            {
                throw Error("'axes', 'begin' and 'end' arrays must have the same length");
            }

            Shape outputShape = getShape(input, shapes);

            checkAxes("axes", axes, outputShape.rank());

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

            setShape(output, shapes, outputShape);
        }

        void propagateShapesStack( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& values = args["values"];
            auto& value = args["value"];
            auto& axis = args["axis"];

            auto& inputShape = getShape(values[0], shapes);

            bool compatibleShape = true;
            for ( size_t i = 1; i < values.size(); ++i )
            {
                auto& partShape = getShape(values[i], shapes);

                if ( partShape.rank() != inputShape.rank() )
                {
                    compatibleShape = false;
                    break;
                }

                for ( size_t i = 0; i < inputShape.rank(); ++i )
                {
                    compatibleShape &= inputShape[i] == partShape[i];
                }
            }

            if ( !compatibleShape )
            {
                throw Error("incompatible tensor shapes in input array");
            }

            Shape outputShape(inputShape.rank() + 1);

            checkAxis("axis", axis, outputShape.rank());

            const size_t idx = axis.integer();
            for ( size_t i = 0; i < idx; ++i )
            {
                outputShape[i] = inputShape[i];
            }
            outputShape[idx] = (Shape::extent_type)values.size();
            for ( size_t i = idx + 1; i < outputShape.rank(); ++i )
            {
                outputShape[i] = inputShape[i-1];
            }

            checkMaxRank("value", outputShape);

            setShape(value, shapes, outputShape);
        }

        void propagateShapesUnstack( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& values = args["values"];
            auto& value = args["value"];
            auto& axis = args["axis"];

            Shape inputShape = getShape(value, shapes);

            checkAxis("axis", axis, inputShape.rank());

            const size_t idx = axis.integer();

            Shape outputShape(inputShape.rank() - 1);
            for ( size_t i = 0; i < idx; ++i )
            {
                outputShape[i] = inputShape[i];
            }
            for ( size_t i = idx; i < outputShape.rank(); ++i )
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
                setShape(values[i], shapes, outputShape);
            }
        }

        void propagateShapesSqueeze( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& axes = args["axes"];

            auto& inputShape = getShape(input, shapes);

            checkAxes("axes", axes, inputShape.rank());

            for ( size_t i = 0; i < axes.size(); ++i )
            {
                auto axis = axes[i].integer();
                if ( inputShape[axis] != 1 )
                {
                    throw Error("squeezed dimension is not singleton (has extent %d)", (int)inputShape[axis]);
                }
            }

            Shape outputShape(inputShape.rank() - axes.size());
            for ( size_t i = 0, k = 0; i < inputShape.rank(); ++i )
            {
                if ( !containsAxis(axes, i) )
                {
                    outputShape[k++] = inputShape[i];
                }
            }

            setShape(output, shapes, outputShape);
        }

        void propagateShapesUnsqueeze( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& axes = args["axes"];

            auto& inputShape = getShape(input, shapes);

            Shape outputShape(inputShape.rank() + axes.size());

            checkAxes("axes", axes, outputShape.rank());

            for ( size_t i = 0, k = 0; i < outputShape.rank(); ++i )
            {
                outputShape[i] = containsAxis(axes, i) ? (Shape::extent_type)1 : inputShape[k++];
            }

            checkMaxRank("output", outputShape);

            setShape(output, shapes, outputShape);
        }

        void propagateShapesUpsample( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
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

        void propagateShapesDownsample( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
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

        void propagateShapesNormalize( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];

            auto& inputShape = getShape(input, shapes);

            auto& axes = args["axes"];
            if ( axes )
            {
                checkAxes("name", axes, inputShape.rank());
            }

            auto& size = args["size"];
            if ( size )
            {
                checkRank("size", size, inputShape.rank());
                checkRange("size", size, 1);
            }

            setShape(output, shapes, inputShape);
        }

        void propagateShapesRoi( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& rois = args["rois"];
            auto& index = args["batch_index"];
            auto& size = args["output_size"];
            auto& rate = args["sampling_rate"];

            auto& inputShape = getShape(input, shapes);
            auto& roisShape = getShape(rois, shapes);
            auto& indexShape = getShape(index, shapes);

            checkRank("pooled_size", size, inputShape.rank() - 2);
            checkRange("pooled_size", size, 1);

            if ( rate )
            {
                checkRank("sampling_rate", rate, inputShape.rank() - 2);
                checkRange("sampling_rate", rate, 1);
            }

            if ( roisShape.rank() != 2 )
            {
                throw Error("'rois' must be a rank-2 tensor");
            }
            if ( indexShape.rank() != 1 )
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

            Shape outputShape(inputShape.rank());
            outputShape[0] = roisShape[0];
            outputShape[1] = inputShape[1];
            for ( size_t i = 0; i < size.size(); ++i )
            {
                outputShape[i+2] = (Shape::extent_type)size[i].integer();
            }
            setShape(output, shapes, outputShape);
        }

        void propagateShapesSelect( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& condition = args["condition"];
            auto& trueValue = args["true_value"];
            auto& falseValue = args["false_value"];
            auto& output = args["output"];

            auto& conditionShape = getShape(condition, shapes);
            auto& trueShape = getShape(trueValue, shapes);
            auto& falseShape = getShape(falseValue, shapes);

            if ( !isBroadcastCompatible(trueShape, falseShape) )
            {
                throw Error("incompatible tensor shapes in select operation (%s vs %s)",
                            trueShape.toString().c_str(), falseShape.toString().c_str());
            }

            Shape valueShape = broadcastShape(trueShape, falseShape);

            if ( !isBroadcastCompatible(conditionShape, valueShape) )
            {
                throw Error("condition shape incompatible with result shape in select operation (%s vs %s)",
                            conditionShape.toString().c_str(), valueShape.toString().c_str());
            }

            Shape outputShape = broadcastShape(conditionShape, valueShape);

            setShape(output, shapes, outputShape);
        }

        void propagateShapesMatmul( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& A = args["A"];
            auto& B = args["B"];
            auto& C = args["C"];
            auto& trA = args["transposeA"];
            auto& trB = args["transposeB"];

            auto& aShape = getShape(A, shapes);
            auto& bShape = getShape(B, shapes);

            if ( aShape.rank() != bShape.rank() )
            {
                throw Error("rank mismatch for A and B (%d vs %d)", (int)aShape.rank(), (int)bShape.rank());
            }

            auto rank = aShape.rank();
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

            setShape(C, shapes, cShape);
        }

        void propagateShapesLinear( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            auto& input = args["input"];
            auto& output = args["output"];
            auto& filter = args["filter"];

            auto& inputShape = getShape(input, shapes);
            auto& filterShape = getShape(filter, shapes);

            if ( inputShape.rank() != 2 )
            {
                throw Error("input shape must be of rank 2 (found %d)", (int)inputShape.rank());
            }
            if ( filterShape.rank() != 2 )
            {
                throw Error("filter shape must be of rank 2 (found %d)", (int)filterShape.rank());
            }

            if ( inputShape[1] != filterShape[1] )
            {
                throw Error("inner dimensions must agree (%d vs %d)", (int)inputShape[1], (int)filterShape[1]);
            }

            Shape outputShape(2);
            outputShape[0] = inputShape[0];
            outputShape[1] = filterShape[0];

            setShape(output, shapes, outputShape);
        }

        void propagateShapesUpdate( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
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

        void propagateShapesSoftmax( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
        {
            propagateShapesUnary(proto, args, shapes);

            auto& input = args["input"];
            auto& inputShape = shapes[input.identifier()];
            
            auto& axes = args["axes"];
            checkAxes("axes", axes, inputShape.rank());
        }

        void propagateShapesCopyN( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
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

        void propagateShapesAddN( const Prototype& proto, const Dictionary<Value>& args, Dictionary<Shape>& shapes )
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
            if ( arg.kind() == Value::Identifier )
            {
                assert(shapes.contains(arg.identifier()));
                return shapes[arg.identifier()];
            }
            else if ( arg.kind() == Value::ShapeOf )
            {
                assert(shapes.contains(arg.shape_of().id));
                return shapes[arg.shape_of().id];
            }
            else
            {
                static Shape singleton;
                return singleton;
            }
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
                shapes[arg.identifier()] = shape;
            }
        }

        static Shape makeShape( const Value& arg, const size_t offset = 0 )
        {
            Shape shape = Shape(offset + arg.size());

            for ( size_t i = 0; i < offset; ++i )
            {
                shape[i] = 1;
            }
            for ( size_t i = 0; i < arg.size(); ++i )
            {
                shape[i + offset] = arg[i].integer();
            }
            return shape;
        }

        static std::pair<Shape,Shape> makePadding( const Value& arg, const size_t offset = 0 )
        {
            const size_t rank = offset + arg.size();
            std::pair<Shape,Shape> padding = std::make_pair(Shape(rank), Shape(rank));

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
            for ( size_t i = offset; i < inputShape.rank(); ++i )
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
            for ( size_t i = offset; i < inputShape.rank(); ++i )
            {
                auto window = (kernelShape[i] - 1) * dilationShape[i] + 1;
                auto total = window - strideShape[i];

                paddingShape.first[i] = total / 2;
                paddingShape.second[i] = total - total / 2;
            }
        }

        static Shape makeSeparableFilterShape( const Shape& planeShape, const Shape& pointShape, const int channels )
        {
            for ( size_t i = 2; i < pointShape.rank(); ++i )
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
            const size_t rank = std::max(xShape.rank(), yShape.rank());
            return isBroadcastCompatible(xShape, yShape, rank);
        }

        static bool isBroadcastCompatible( const Shape& xShape, const Shape& yShape, const size_t count )
        {
            for ( size_t i = 0; i < count; ++i )
            {
                auto xi = i < xShape.rank() ? xShape[i] : 1;
                auto yi = i < yShape.rank() ? yShape[i] : 1;
                if ( !(xi == yi || xi == 1 || yi == 1) )
                {
                    return false;
                }
            }
            return true;
        }

        static Shape broadcastShape( const Shape& xShape, const Shape& yShape )
        {
            const size_t rank = std::max(xShape.rank(), yShape.rank());
            return broadcastShape(xShape, yShape, rank);
        }

        static Shape broadcastShape( const Shape& xShape, const Shape& yShape, const size_t count )
        {
            const size_t rank = std::max(xShape.rank(), yShape.rank());
            Shape zShape(rank);

            for ( size_t i = 0; i < count; ++i )
            {
                auto xi = i < xShape.rank() ? xShape[i] : 1;
                auto yi = i < yShape.rank() ? yShape[i] : 1;
                zShape[i] = std::max(xi, yi);
            }
            return zShape;
        }

        void checkAxis( const char* name, const Value& value, const size_t rank )
        {
            auto axis = value.integer();
            if ( axis < 0 )
            {
                throw Error("'%s' must be non-negative", name);
            }
            if ( axis >= (Value::integer_t)rank )
            {
                throw Error("'%s' must be less than the tensor rank (%d)", name, (int)rank);
            }
        }

        void checkAxes( const char* name, const Value& value, const size_t rank )
        {
            auto& items = value.items();

            for ( size_t i = 0; i < value.size(); ++i )
            {
                auto& item = value[i];
                checkAxis(name, item, rank);

                auto last = items.begin() + i;
                if ( std::find(items.begin(), last, item) != last )
                {
                    throw Error("duplicate item '%d' in array '%s'", (int)item.integer(), name);
                }
            }
        }

        static bool containsAxis( const Value& axes, const size_t axis )
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

        static void checkRank( const char* name, const Value& value, const size_t rank )
        {
            if ( value.size() != rank )
            {
                throw Error("length of array '%s' must be %d to match rank of operation (found %d)", name, (int)rank, (int)value.size());
            }
        }

        void checkMaxRank( const char* name, const Shape& shape )
        {
            if ( shape.rank() > _maxRank )
            {
                throw Error("rank %d of '%s' exceeds maximum supported dimensions (%d)", (int)shape.rank(), name, (int)_maxRank);
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

        void checkSharedVariableShapes( const std::string& label, const Shape& shape )
        {
            std::string lowercase = label;
            std::transform(label.begin(), label.end(), lowercase.begin(), []( unsigned char ch ){ return std::tolower(ch); });

            auto it = _variableShapes.find(lowercase);
            if ( it != _variableShapes.end() )
            {
                if ( it->second != shape )
                {
                    throw Error("variable shape %s does not match previously defined shape %s for label '%s'",
                                shape.toString().c_str(), it->second.toString().c_str(), label.c_str());
                }
            }
            else
            {
                _variableShapes.emplace(lowercase, shape);
            }
        }

        static bool isLabelChar( unsigned char ch )
        {
            return std::isalnum(ch) || ch == '/' || ch == '\\' || ch == '-' || ch == '_' || ch == '.';
        }

    public:

        const Dictionary<Shape>& variableShapes() const
        {
            return _variableShapes;
        }

    private:

        Dictionary<Shape> _variableShapes;
        const size_t _maxRank;
    };

}   // namespace nnef


#endif
