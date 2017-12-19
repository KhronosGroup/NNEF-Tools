/*
 * Copyright (c) 2012-2017 The Khronos Group Inc.
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
        static const Type* Scalar = new PrimitiveType(Typename::Scalar, false);
        static const Type* Extent = new PrimitiveType(Typename::Extent, false);
        static const Type* Logical = new PrimitiveType(Typename::Logical, false);
        static const Type* String = new PrimitiveType(Typename::String, false);
        static const Type* Tensor = new PrimitiveType(Typename::Scalar, true);
        static const Type* ExtentTensor = new PrimitiveType(Typename::Extent, true);
        static const Type* LogicalTensor = new PrimitiveType(Typename::Logical, true);
        
        static const Type* Scalars = new ArrayType(Scalar);
        static const Type* Extents = new ArrayType(Extent);
        static const Type* Tensors = new ArrayType(Tensor);
        static const Type* ExtentPair = new TupleType({ Extent, Extent });
        static const Type* ExtentPairs = new ArrayType(ExtentPair);

        static const Value ScalarZero = Value::scalar(0.0);
        static const Value ScalarOne = Value::scalar(1.0);
        static const Value ScalarHalf = Value::scalar(0.5);

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
                Param("shape", Extents),
            }, { Result("output", Tensor) }),

            Prototype("constant", {
                Param("shape", Extents),
                Param("value", Scalars),
            }, { Result("output", Tensor) }),

            Prototype("variable", {
                Param("shape", Extents),
                Param("label", String),
            }, { Result("output", Tensor) }),

            Prototype("update", {
                Param("variable", Tensor),
                Param("value", Tensor),
            }, { Result("result", Tensor) }),


            Prototype("reshape", {
                Param("input", Tensor),
                Param("shape", Extents),
            }, { Result("output", Tensor) }),

            Prototype("transpose", {
                Param("input", Tensor),
                Param("perm", Extents),
            }, { Result("output", Tensor) }),

            Prototype("concat", {
                Param("values", Tensors),
                Param("axis", Extent),
            }, { Result("value", Tensor) }),

            Prototype("split", {
                Param("value", Tensor),
                Param("axis", Extent),
                Param("ratios", Extents),
            }, { Result("values", Tensors) }),


            Prototype("add", {
                Param("x", Tensor),
                Param("y", Tensor)
            }, { Result("z", Tensor) }),

            Prototype("sub", {
                Param("x", Tensor),
                Param("y", Tensor)
            }, { Result("z", Tensor) }),

            Prototype("mul", {
                Param("x", Tensor),
                Param("y", Tensor)
            }, { Result("z", Tensor) }),

            Prototype("div", {
                Param("x", Tensor),
                Param("y", Tensor)
            }, { Result("z", Tensor) }),

            Prototype("pow", {
                Param("x", Tensor),
                Param("y", Tensor)
            }, { Result("z", Tensor) }),
            
            Prototype("min", {
                Param("x", Tensor),
                Param("y", Tensor)
            }, { Result("z", Tensor) }),
            
            Prototype("max", {
                Param("x", Tensor),
                Param("y", Tensor)
            }, { Result("z", Tensor) }),
            
            Prototype("lt", {
                Param("x", Tensor),
                Param("y", Tensor)
            }, { Result("z", LogicalTensor) }),
            
            Prototype("le", {
                Param("x", Tensor),
                Param("y", Tensor)
            }, { Result("z", LogicalTensor) }),
            
            Prototype("gt", {
                Param("x", Tensor),
                Param("y", Tensor)
            }, { Result("z", LogicalTensor) }),
            
            Prototype("ge", {
                Param("x", Tensor),
                Param("y", Tensor)
            }, { Result("z", LogicalTensor) }),
            
            Prototype("eq", {
                Param("x", Tensor),
                Param("y", Tensor)
            }, { Result("z", LogicalTensor) }),
            
            Prototype("ne", {
                Param("x", Tensor),
                Param("y", Tensor)
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
                Param("true_value", Tensor),
                Param("false_value", Tensor),
            }, { Result("output", Tensors) }),
            
            
            Prototype("clamp", {
                Param("x", Tensor),
                Param("a", Tensor),
                Param("b", Tensor),
            }, { Result("y", Tensors) }),
            
            
            Prototype("idn", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("neg", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("rcp", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("exp", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("log", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("abs", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("sign", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("floor", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("ceil", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("round", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("sqr", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("sqrt", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("rsqr", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("rsqrt", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("log2", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("not", {
                Param("x", LogicalTensor),
            }, { Result("y", LogicalTensor) }),
            
            
            Prototype("relu", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("sigmoid", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("tanh", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("elu", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("leaky_relu", {
                Param("x", Tensor),
                Param("alpha", Scalar),
            }, { Result("y", Tensor) }),
            
            Prototype("softabs", {
                Param("x", Tensor),
                Param("epsilon", Scalar),
            }, { Result("y", Tensor) }),
            
            Prototype("softplus", {
                Param("x", Tensor),
            }, { Result("y", Tensor) }),
            
            Prototype("softmax", {
                Param("x", Tensor),
                Param("axes", Extents, IntegersOne),
            }, { Result("y", Tensor) }),


            Prototype("conv", {
                Param("input", Tensor),
                Param("filter", Tensor),
                Param("bias", Tensor, ScalarOne),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
                Param("groups", Extent, IntegerOne),
            }, { Result("output", Tensor) }),

            Prototype("deconv", {
                Param("input", Tensor),
                Param("filter", Tensor),
                Param("bias", Tensor, ScalarOne),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
                Param("groups", Extent, IntegerOne),
            }, { Result("output", Tensor) }),

            Prototype("box", {
                Param("input", Tensor),
                Param("size", Extents),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
                Param("normalize", Logical, LogicalFalse),
            }, { Result("output", Tensor) }),

            Prototype("debox", {
                Param("input", Tensor),
                Param("size", Extents),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
                Param("normalize", Logical, LogicalFalse),
            }, { Result("output", Tensor) }),
            
            Prototype("sample", {
                Param("input", Tensor),
                Param("index", ExtentTensor),
                Param("size", Extents),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
            }, { Result("output", Tensor) }),
            
            Prototype("desample", {
                Param("input", Tensor),
                Param("index", ExtentTensor),
                Param("size", Extents),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
            }, { Result("output", Tensor) }),
            
            Prototype("max_pool", {
                Param("input", Tensor),
                Param("size", Extents),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
            }, { Result("output", Tensor) }),
            
            Prototype("argmax_pool", {
                Param("input", Tensor),
                Param("size", Extents),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
            }, { Result("index", ExtentTensor) }),
            
            Prototype("max_pool_with_index", {
                Param("input", Tensor),
                Param("size", Extents),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
            }, { Result("output", Tensor), Result("index", ExtentTensor) }),
            
            Prototype("avg_pool", {
                Param("input", Tensor),
                Param("size", Extents),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
            }, { Result("output", Tensor) }),
            
            Prototype("rms_pool", {
                Param("input", Tensor),
                Param("size", Extents),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
            }, { Result("output", Tensor) }),
            
            
            Prototype("planewise_conv", {
                Param("input", Tensor),
                Param("filter", Tensor),
                Param("bias", Tensor, ScalarZero),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
            }, { Result("output", Tensor) }),
            
            Prototype("planewise_deconv", {
                Param("input", Tensor),
                Param("filter", Tensor),
                Param("bias", Tensor, ScalarZero),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
            }, { Result("output", Tensor) }),
            
            Prototype("separable_conv", {
                Param("input", Tensor),
                Param("plane_filter", Tensor),
                Param("point_filter", Tensor),
                Param("bias", Tensor, ScalarZero),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
                Param("groups", Extent, IntegerOne),
            }, { Result("output", Tensor) }),
            
            Prototype("separable_deconv", {
                Param("input", Tensor),
                Param("plane_filter", Tensor),
                Param("point_filter", Tensor),
                Param("bias", Tensor, ScalarZero),
                Param("border", String, StringConstant),
                Param("padding", ExtentPairs, EmptyArray),
                Param("stride", Extents, EmptyArray),
                Param("dilation", Extents, EmptyArray),
                Param("groups", Extent, IntegerOne),
            }, { Result("output", Tensor) }),
            
            
            Prototype("nearest_downsample", {
                Param("input", Tensor),
                Param("factor", Extents),
            }, { Result("output", Tensor) }),
            
            Prototype("nearest_upsample", {
                Param("input", Tensor),
                Param("factor", Extents),
            }, { Result("output", Tensor) }),
            
            Prototype("area_downsample", {
                Param("input", Tensor),
                Param("factor", Extents),
            }, { Result("output", Tensor) }),
            
            Prototype("multilinear_upsample", {
                Param("input", Tensor),
                Param("factor", Extents),
                Param("method", String, StringSymmetric),
                Param("border", String, StringReplicate),
            }, { Result("output", Tensor) }),
            
            
            Prototype("local_response_normalization", {
                Param("input", Tensor),
                Param("size", Extents),
                Param("alpha", Scalar, ScalarOne),
                Param("beta", Scalar, ScalarHalf),
                Param("bias", Scalar, ScalarOne),
            }, { Result("output", Tensor) }),
            
            Prototype("local_mean_normalization", {
                Param("input", Tensor),
                Param("size", Extents),
            }, { Result("output", Tensor) }),
            
            Prototype("local_variance_normalization", {
                Param("input", Tensor),
                Param("size", Extents),
                Param("bias", Scalar, ScalarZero),
            }, { Result("output", Tensor) }),
            
            Prototype("local_contrast_normalization", {
                Param("input", Tensor),
                Param("size", Extents),
                Param("bias", Scalar, ScalarZero),
            }, { Result("output", Tensor) }),
            
            Prototype("l1_normalization", {
                Param("input", Tensor),
                Param("axes", Extents),
                Param("bias", Scalar, ScalarZero),
            }, { Result("output", Tensor) }),
            
            Prototype("l2_normalization", {
                Param("input", Tensor),
                Param("axes", Extents),
                Param("bias", Scalar, ScalarZero),
            }, { Result("output", Tensor) }),
            
            Prototype("layer_normalization", {
                Param("input", Tensor),
                Param("axes", Extents),
                Param("bias", Scalar, ScalarZero),
            }, { Result("output", Tensor) }),
            
            Prototype("divisive_normalization", {
                Param("input", Tensor),
                Param("axes", Extents),
                Param("bias", Scalar, ScalarZero),
            }, { Result("output", Tensor) }),
            
            Prototype("batch_normalization", {
                Param("input", Tensor),
                Param("mean", Tensor),
                Param("variance", Tensor),
                Param("offset", Tensor, ScalarZero),
                Param("scale", Tensor, ScalarOne),
                Param("epsilon", Scalar, ScalarZero),
            }, { Result("output", Tensor) }),
            
            
            Prototype("sum_reduce", {
                Param("input", Tensor),
                Param("axes", Extents),
                Param("normalize", Logical, LogicalFalse),
            }, { Result("output", Tensor) }),
            
            Prototype("min_reduce", {
                Param("input", Tensor),
                Param("axes", Extents),
            }, { Result("output", Tensor) }),
            
            Prototype("max_reduce", {
                Param("input", Tensor),
                Param("axes", Extents),
            }, { Result("output", Tensor) }),
            
            Prototype("mean_reduce", {
                Param("input", Tensor),
                Param("axes", Extents),
            }, { Result("output", Tensor) }),
            
            Prototype("moments", {
                Param("input", Tensor),
                Param("axes", Extents),
            }, { Result("mean", Tensor), Result("variance", Tensor) }),
            
            
            Prototype("matmul", {
                Param("A", Tensor),
                Param("B", Tensor),
                Param("trA", Logical, LogicalFalse),
                Param("trB", Logical, LogicalFalse),
            }, { Result("C", Tensor) }),
            
            Prototype("linear", {
                Param("input", Tensor),
                Param("filter", Tensor),
                Param("bias", Tensor, ScalarZero),
            }, { Result("output", Tensor) }),
            
            
            Prototype("add_n", {
                Param("x", Tensors),
            }, { Result("y", Tensor) }),
            
            Prototype("copy_n", {
                Param("x", Tensor),
                Param("times", Extent),
            }, { Result("y", Tensors) }),
            
            
            Prototype("linear_quantize", {
                Param("x", Tensor),
                Param("min", Tensor),
                Param("max", Tensor),
                Param("bits", Extent),
            }, { Result("y", Tensor) }),
            
            Prototype("logarithmic_quantize", {
                Param("x", Tensor),
                Param("max", Tensor),
                Param("bits", Extent),
            }, { Result("y", Tensor) }),
        };

        return prototypes;
    }

}   // namespace nnef


#endif
