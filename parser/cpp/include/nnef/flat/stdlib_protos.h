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

#ifndef _NNEF_STDLIB_PROTOS_H_
#define _NNEF_STDLIB_PROTOS_H_

#include "../common/value.h"
#include "../common/typespec.h"
#include "../common/prototype.h"
#include "../common/dictionary.h"


namespace nnef
{

    static std::vector<Prototype> stdlibPrototypes()
    {
        static const PrimitiveType* Scalar = primitiveType(Typename::Scalar);
        static const PrimitiveType* Integer = primitiveType(Typename::Integer);
        static const PrimitiveType* Logical = primitiveType(Typename::Logical);
        static const PrimitiveType* String = primitiveType(Typename::String);
        static const PrimitiveType* Generic = primitiveType(Typename::Generic);
        
        static const Type* ScalarTensor = tensorType(Typename::Scalar);
        static const Type* IntegerTensor = tensorType(Typename::Integer);
        static const Type* LogicalTensor = tensorType(Typename::Logical);
        static const Type* GenericTensor = tensorType(Typename::Generic);
        static const Type* TypelessTensor = tensorType();

        static const Type* Integers = arrayType(Integer);
        static const Type* Generics = arrayType(Generic);
        static const Type* Tensors = arrayType(ScalarTensor);
        static const Type* GenericTensors = arrayType(GenericTensor);
        static const Type* IntegerPair = tupleType({ Integer, Integer });
        static const Type* IntegerPairs = arrayType(IntegerPair);

        static const Value ScalarZero = Value::scalar(0.0);
        static const Value ScalarOne = Value::scalar(1.0);
        static const Value ScalarHalf = Value::scalar(0.5);

        static const Value IntegerMinusOne = Value::integer(-1);
        static const Value IntegerZero = Value::integer(0);
        static const Value IntegerOne = Value::integer(1);

        static const Value LogicalFalse = Value::logical(false);
        static const Value LogicalTrue = Value::logical(true);

        static const Value StringConstant = Value::string("constant");
        static const Value StringSymmetric = Value::string("symmetric");
        static const Value StringReplicate = Value::string("replicate");

        static const Value EmptyArray = Value::array({});
        static const Value IntegersOne = Value::array({ IntegerOne });

        
        static const std::vector<Prototype> prototypes =
        {
            Prototype("external", {
                Param("shape", Integers),
            }, { Result("output", GenericTensor) }, Scalar),
            
            Prototype("constant", {
                Param("shape", Integers),
                Param("value", Generics),
            }, { Result("output", GenericTensor) }, Scalar),

            Prototype("variable", {
                Param("shape", Integers),
                Param("label", String),
            }, { Result("output", GenericTensor) }, Scalar),

            Prototype("update", {
                Param("variable", GenericTensor),
                Param("value", GenericTensor),
            }, { Result("result", GenericTensor) }),


            Prototype("reshape", {
                Param("input", GenericTensor),
                Param("shape", Integers),
                Param("axis_start", Integer, IntegerZero),
                Param("axis_count", Integer, IntegerMinusOne),
            }, { Result("output", GenericTensor) }),

            Prototype("transpose", {
                Param("input", GenericTensor),
                Param("axes", Integers),
            }, { Result("output", GenericTensor) }),

            Prototype("concat", {
                Param("values", GenericTensors),
                Param("axis", Integer),
            }, { Result("value", GenericTensor) }),

            Prototype("split", {
                Param("value", GenericTensor),
                Param("axis", Integer),
                Param("ratios", Integers),
            }, { Result("values", GenericTensors) }),

            Prototype("slice", {
                Param("input", GenericTensor),
                Param("axes", Integers),
                Param("begin", Integers),
                Param("end", Integers),
                Param("stride", Integers, EmptyArray),
            }, { Result("output", GenericTensor) }),

            Prototype("stack", {
                Param("values", GenericTensors),
                Param("axis", Integer),
            }, { Result("value", GenericTensor) }),

            Prototype("unstack", {
                Param("value", GenericTensor),
                Param("axis", Integer),
            }, { Result("values", GenericTensors) }),

            Prototype("squeeze", {
                Param("input", GenericTensor),
                Param("axes", Integers),
            }, { Result("output", GenericTensor) }),

            Prototype("unsqueeze", {
                Param("input", GenericTensor),
                Param("axes", Integers),
            }, { Result("output", GenericTensor) }),

            Prototype("pad", {
                Param("input", ScalarTensor),
                Param("padding", IntegerPairs),
                Param("border", String, StringConstant),
                Param("value", Scalar, ScalarZero),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("tile", {
                Param("input", GenericTensor),
                Param("repeats", Integers),
            }, { Result("output", GenericTensor) }),
            
            Prototype("gather", {
                Param("input", GenericTensor),
                Param("indices", IntegerTensor),
                Param("axis", Integer, IntegerZero),
            }, { Result("output", GenericTensor) }),
            
            Prototype("cast", {
                Param("input", TypelessTensor),
            }, { Result("output", GenericTensor) }),

            Prototype("add", {
                Param("x", ScalarTensor),
                Param("y", ScalarTensor)
            }, { Result("z", ScalarTensor) }),

            Prototype("sub", {
                Param("x", ScalarTensor),
                Param("y", ScalarTensor)
            }, { Result("z", ScalarTensor) }),

            Prototype("mul", {
                Param("x", ScalarTensor),
                Param("y", ScalarTensor)
            }, { Result("z", ScalarTensor) }),

            Prototype("div", {
                Param("x", ScalarTensor),
                Param("y", ScalarTensor)
            }, { Result("z", ScalarTensor) }),

            Prototype("pow", {
                Param("x", ScalarTensor),
                Param("y", ScalarTensor)
            }, { Result("z", ScalarTensor) }),
            
            Prototype("min", {
                Param("x", ScalarTensor),
                Param("y", ScalarTensor)
            }, { Result("z", ScalarTensor) }),
            
            Prototype("max", {
                Param("x", ScalarTensor),
                Param("y", ScalarTensor)
            }, { Result("z", ScalarTensor) }),
            
            Prototype("lt", {
                Param("x", ScalarTensor),
                Param("y", ScalarTensor)
            }, { Result("z", LogicalTensor) }),
            
            Prototype("le", {
                Param("x", ScalarTensor),
                Param("y", ScalarTensor)
            }, { Result("z", LogicalTensor) }),
            
            Prototype("gt", {
                Param("x", ScalarTensor),
                Param("y", ScalarTensor)
            }, { Result("z", LogicalTensor) }),
            
            Prototype("ge", {
                Param("x", ScalarTensor),
                Param("y", ScalarTensor)
            }, { Result("z", LogicalTensor) }),
            
            Prototype("eq", {
                Param("x", ScalarTensor),
                Param("y", ScalarTensor)
            }, { Result("z", LogicalTensor) }),
            
            Prototype("ne", {
                Param("x", ScalarTensor),
                Param("y", ScalarTensor)
            }, { Result("z", LogicalTensor) }),
            
            Prototype("and", {
                Param("x", LogicalTensor),
                Param("y", LogicalTensor)
            }, { Result("z", LogicalTensor) }),
            
            Prototype("or", {
                Param("x", LogicalTensor),
                Param("y", LogicalTensor)
            }, { Result("z", LogicalTensor) }),
            
            
            Prototype("select", {
                Param("condition", LogicalTensor),
                Param("true_value", GenericTensor),
                Param("false_value", GenericTensor),
            }, { Result("output", GenericTensor) }),
            
            
            Prototype("clamp", {
                Param("x", ScalarTensor),
                Param("a", ScalarTensor),
                Param("b", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            
            Prototype("copy", {
                Param("x", GenericTensor),
            }, { Result("y", GenericTensor) }),
            
            Prototype("neg", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("rcp", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("exp", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("log", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("sin", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("cos", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("tan", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("asin", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("acos", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("atan", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("sinh", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("cosh", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("tanh", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("asinh", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("acosh", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("atanh", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("abs", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("sign", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("floor", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("ceil", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("round", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("sqr", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("sqrt", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("rsqr", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("rsqrt", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("log2", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("not", {
                Param("x", LogicalTensor),
            }, { Result("y", LogicalTensor) }),
            
            
            Prototype("relu", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("sigmoid", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("elu", {
                Param("x", ScalarTensor),
                Param("alpha", ScalarTensor, ScalarOne),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("selu", {
                Param("x", ScalarTensor),
                Param("alpha", ScalarTensor, Value::scalar(1.67326319)),
                Param("lambda", ScalarTensor, Value::scalar(1.05070102)),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("gelu", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("silu", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("prelu", {
                Param("x", ScalarTensor),
                Param("alpha", ScalarTensor),
            }, { Result("y", ScalarTensor) }),

            Prototype("leaky_relu", {
                Param("x", ScalarTensor),
                Param("alpha", Scalar),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("softabs", {
                Param("x", ScalarTensor),
                Param("epsilon", Scalar),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("softplus", {
                Param("x", ScalarTensor),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("softmax", {
                Param("x", ScalarTensor),
                Param("axes", Integers, IntegersOne),
            }, { Result("y", ScalarTensor) }),


            Prototype("conv", {
                Param("input", ScalarTensor),
                Param("filter", ScalarTensor),
                Param("bias", ScalarTensor, ScalarZero),
                Param("border", String, StringConstant),
                Param("padding", IntegerPairs, EmptyArray),
                Param("stride", Integers, EmptyArray),
                Param("dilation", Integers, EmptyArray),
                Param("groups", Integer, IntegerOne),
            }, { Result("output", ScalarTensor) }),

            Prototype("deconv", {
                Param("input", ScalarTensor),
                Param("filter", ScalarTensor),
                Param("bias", ScalarTensor, ScalarZero),
                Param("border", String, StringConstant),
                Param("padding", IntegerPairs, EmptyArray),
                Param("stride", Integers, EmptyArray),
                Param("dilation", Integers, EmptyArray),
                Param("output_shape", Integers, EmptyArray),
                Param("groups", Integer, IntegerOne),
            }, { Result("output", ScalarTensor) }),

            Prototype("box", {
                Param("input", ScalarTensor),
                Param("size", Integers),
                Param("border", String, StringConstant),
                Param("padding", IntegerPairs, EmptyArray),
                Param("stride", Integers, EmptyArray),
                Param("dilation", Integers, EmptyArray),
                Param("normalize", Logical, LogicalFalse),
            }, { Result("output", ScalarTensor) }),

            Prototype("debox", {
                Param("input", ScalarTensor),
                Param("size", Integers),
                Param("border", String, StringConstant),
                Param("padding", IntegerPairs, EmptyArray),
                Param("stride", Integers, EmptyArray),
                Param("dilation", Integers, EmptyArray),
                Param("output_shape", Integers, EmptyArray),
                Param("normalize", Logical, LogicalFalse),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("sample", {
                Param("input", ScalarTensor),
                Param("index", IntegerTensor),
                Param("size", Integers),
                Param("border", String, StringConstant),
                Param("padding", IntegerPairs, EmptyArray),
                Param("stride", Integers, EmptyArray),
                Param("dilation", Integers, EmptyArray),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("desample", {
                Param("input", ScalarTensor),
                Param("index", IntegerTensor),
                Param("size", Integers),
                Param("border", String, StringConstant),
                Param("padding", IntegerPairs, EmptyArray),
                Param("stride", Integers, EmptyArray),
                Param("dilation", Integers, EmptyArray),
                Param("output_shape", Integers, EmptyArray),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("max_pool", {
                Param("input", ScalarTensor),
                Param("size", Integers),
                Param("border", String, StringConstant),
                Param("padding", IntegerPairs, EmptyArray),
                Param("stride", Integers, EmptyArray),
                Param("dilation", Integers, EmptyArray),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("argmax_pool", {
                Param("input", ScalarTensor),
                Param("size", Integers),
                Param("border", String, StringConstant),
                Param("padding", IntegerPairs, EmptyArray),
                Param("stride", Integers, EmptyArray),
                Param("dilation", Integers, EmptyArray),
            }, { Result("index", IntegerTensor) }),
            
            Prototype("max_pool_with_index", {
                Param("input", ScalarTensor),
                Param("size", Integers),
                Param("border", String, StringConstant),
                Param("padding", IntegerPairs, EmptyArray),
                Param("stride", Integers, EmptyArray),
                Param("dilation", Integers, EmptyArray),
            }, { Result("output", ScalarTensor), Result("index", IntegerTensor) }),
            
            Prototype("avg_pool", {
                Param("input", ScalarTensor),
                Param("size", Integers),
                Param("border", String, StringConstant),
                Param("padding", IntegerPairs, EmptyArray),
                Param("stride", Integers, EmptyArray),
                Param("dilation", Integers, EmptyArray),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("rms_pool", {
                Param("input", ScalarTensor),
                Param("size", Integers),
                Param("border", String, StringConstant),
                Param("padding", IntegerPairs, EmptyArray),
                Param("stride", Integers, EmptyArray),
                Param("dilation", Integers, EmptyArray),
            }, { Result("output", ScalarTensor) }),

            
            Prototype("separable_conv", {
                Param("input", ScalarTensor),
                Param("plane_filter", ScalarTensor),
                Param("point_filter", ScalarTensor),
                Param("bias", ScalarTensor, ScalarZero),
                Param("border", String, StringConstant),
                Param("padding", IntegerPairs, EmptyArray),
                Param("stride", Integers, EmptyArray),
                Param("dilation", Integers, EmptyArray),
                Param("groups", Integer, IntegerOne),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("separable_deconv", {
                Param("input", ScalarTensor),
                Param("plane_filter", ScalarTensor),
                Param("point_filter", ScalarTensor),
                Param("bias", ScalarTensor, ScalarZero),
                Param("border", String, StringConstant),
                Param("padding", IntegerPairs, EmptyArray),
                Param("stride", Integers, EmptyArray),
                Param("dilation", Integers, EmptyArray),
                Param("output_shape", Integers, EmptyArray),
                Param("groups", Integer, IntegerOne),
            }, { Result("output", ScalarTensor) }),
            
            
            Prototype("nearest_downsample", {
                Param("input", ScalarTensor),
                Param("factor", Integers),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("nearest_upsample", {
                Param("input", ScalarTensor),
                Param("factor", Integers),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("area_downsample", {
                Param("input", ScalarTensor),
                Param("factor", Integers),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("multilinear_upsample", {
                Param("input", ScalarTensor),
                Param("factor", Integers),
                Param("method", String, StringSymmetric),
                Param("border", String, StringReplicate),
            }, { Result("output", ScalarTensor) }),
            
            
            Prototype("local_response_normalization", {
                Param("input", ScalarTensor),
                Param("size", Integers),
                Param("alpha", Scalar, ScalarOne),
                Param("beta", Scalar, ScalarHalf),
                Param("bias", Scalar, ScalarOne),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("local_mean_normalization", {
                Param("input", ScalarTensor),
                Param("size", Integers),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("local_variance_normalization", {
                Param("input", ScalarTensor),
                Param("size", Integers),
                Param("bias", Scalar, ScalarZero),
                Param("epsilon", Scalar, ScalarZero),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("local_contrast_normalization", {
                Param("input", ScalarTensor),
                Param("size", Integers),
                Param("bias", Scalar, ScalarZero),
                Param("epsilon", Scalar, ScalarZero),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("l1_normalization", {
                Param("input", ScalarTensor),
                Param("axes", Integers),
                Param("bias", Scalar, ScalarZero),
                Param("epsilon", Scalar, ScalarZero),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("l2_normalization", {
                Param("input", ScalarTensor),
                Param("axes", Integers),
                Param("bias", Scalar, ScalarZero),
                Param("epsilon", Scalar, ScalarZero),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("batch_normalization", {
                Param("input", ScalarTensor),
                Param("mean", ScalarTensor),
                Param("variance", ScalarTensor),
                Param("offset", ScalarTensor, ScalarZero),
                Param("scale", ScalarTensor, ScalarOne),
                Param("epsilon", Scalar, ScalarZero),
            }, { Result("output", ScalarTensor) }),
            
            
            Prototype("sum_reduce", {
                Param("input", ScalarTensor),
                Param("axes", Integers),
                Param("normalize", Logical, LogicalFalse),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("min_reduce", {
                Param("input", ScalarTensor),
                Param("axes", Integers),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("max_reduce", {
                Param("input", ScalarTensor),
                Param("axes", Integers),
            }, { Result("output", ScalarTensor) }),
            
            Prototype("mean_reduce", {
                Param("input", ScalarTensor),
                Param("axes", Integers),
            }, { Result("output", ScalarTensor) }),

            Prototype("argmax_reduce", {
                Param("input", ScalarTensor),
                Param("axes", Integers),
            }, { Result("output", IntegerTensor) }),

            Prototype("argmin_reduce", {
                Param("input", ScalarTensor),
                Param("axes", Integers),
            }, { Result("output", IntegerTensor) }),
            
            Prototype("any_reduce", {
                Param("input", LogicalTensor),
                Param("axes", Integers),
            }, { Result("output", LogicalTensor) }),
            
            Prototype("all_reduce", {
                Param("input", LogicalTensor),
                Param("axes", Integers),
            }, { Result("output", LogicalTensor) }),
            
            Prototype("moments", {
                Param("input", ScalarTensor),
                Param("axes", Integers),
            }, { Result("mean", ScalarTensor), Result("variance", ScalarTensor) }),


            Prototype("max_roi_pool", {
                Param("input", ScalarTensor),
                Param("rois", ScalarTensor),
                Param("batch_index", IntegerTensor),
                Param("output_size", Integers),
            }, { Result("output", ScalarTensor) }),

            Prototype("avg_roi_pool", {
                Param("input", ScalarTensor),
                Param("rois", ScalarTensor),
                Param("batch_index", IntegerTensor),
                Param("output_size", Integers),
            }, { Result("output", ScalarTensor) }),

            Prototype("roi_resample", {
                Param("input", ScalarTensor),
                Param("rois", ScalarTensor),
                Param("batch_index", IntegerTensor),
                Param("output_size", Integers),
                Param("method", String, StringSymmetric),
            }, { Result("output", ScalarTensor) }),

            Prototype("max_roi_align", {
                Param("input", ScalarTensor),
                Param("rois", ScalarTensor),
                Param("batch_index", IntegerTensor),
                Param("output_size", Integers),
                Param("sampling_rate", Integers),
                Param("resize_method", String, StringSymmetric),
            }, { Result("output", ScalarTensor) }),

            Prototype("avg_roi_align", {
                Param("input", ScalarTensor),
                Param("rois", ScalarTensor),
                Param("batch_index", IntegerTensor),
                Param("output_size", Integers),
                Param("sampling_rate", Integers),
                Param("resize_method", String, StringSymmetric),
            }, { Result("output", ScalarTensor) }),
            
            
            Prototype("matmul", {
                Param("A", ScalarTensor),
                Param("B", ScalarTensor),
                Param("transposeA", Logical, LogicalFalse),
                Param("transposeB", Logical, LogicalFalse),
            }, { Result("C", ScalarTensor) }),
            
            Prototype("linear", {
                Param("input", ScalarTensor),
                Param("filter", ScalarTensor),
                Param("bias", ScalarTensor, ScalarZero),
            }, { Result("output", ScalarTensor) }),
            
            
            Prototype("add_n", {
                Param("x", Tensors),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("copy_n", {
                Param("x", GenericTensor),
                Param("times", Integer),
            }, { Result("y", GenericTensors) }),
            
            
            Prototype("min_max_linear_quantize", {
                Param("x", ScalarTensor),
                Param("min", ScalarTensor),
                Param("max", ScalarTensor),
                Param("bits", Integer),
                Param("signed", Logical, LogicalTrue),
                Param("symmetric", Logical, LogicalFalse),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("zero_point_linear_quantize", {
                Param("x", ScalarTensor),
                Param("zero_point", IntegerTensor),
                Param("scale", ScalarTensor),
                Param("bits", Integer),
                Param("signed", Logical),
                Param("symmetric", Logical),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("linear_quantize", {
                Param("x", ScalarTensor),
                Param("min", ScalarTensor),
                Param("max", ScalarTensor),
                Param("bits", Integer),
            }, { Result("y", ScalarTensor) }),
            
            Prototype("logarithmic_quantize", {
                Param("x", ScalarTensor),
                Param("max", ScalarTensor),
                Param("bits", Integer),
            }, { Result("y", ScalarTensor) }),
        };

        return prototypes;
    }

}   // namespace nnef


#endif
