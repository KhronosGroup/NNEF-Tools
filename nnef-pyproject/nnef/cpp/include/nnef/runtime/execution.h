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

#ifndef _NNEF_RUNTIME_EXECUTION_H_
#define _NNEF_RUNTIME_EXECUTION_H_

#include "nnef.h"
#include "operations.h"
#include <cassert>


#define DISPATCH_BY_DTYPE(name) \
    inline void execute_##name( const Operation& op, TensorDict& tensors ) \
    { \
        if ( op.dtype == "scalar" ) _execute_##name<float>(op, tensors); \
        else if ( op.dtype == "integer" ) _execute_##name<int>(op, tensors); \
        else if ( op.dtype == "logical" ) _execute_##name<bool>(op, tensors); \
        else throw std::runtime_error("operation not implemented: " + std::string(#name) + "<string>"); \
    } \


namespace nnef { namespace rt
{

    inline Tensor _make_tensor( const size_t rank, const int shape[], const size_t item_bytes )
    {
        Tensor tensor;
        tensor.shape.assign(shape, shape + rank);
        tensor.data.resize(volume_of(tensor.shape) * item_bytes);
        return tensor;
    }

    typedef std::map<std::string,Tensor> TensorDict;
    typedef std::function<void( const Operation& op, TensorDict& tensors )> Executor;


    template<typename T>
    tensor_view<T> _tensor_view( const Tensor& tensor )
    {
        return tensor_view<T>{ tensor.shape.size(), volume_of(tensor.shape), tensor.shape.data(), (T*)tensor.data.data() };
    }

    template<typename T>
    tensor_view<T> _tensor_view( const T& value )
    {
        return tensor_view<T>{ 0, 1, nullptr, (T*)&value };
    }

    template<typename T>
    tensor_view<T> _tensor_view( const Value& value, const TensorDict& tensors )
    {
        return value.kind() == Value::Identifier ? _tensor_view<T>(tensors.at(value.identifier())) : _tensor_view<T>(value.get<T>());
    }

    const std::string& _literal_dtype( const Value& value )
    {
        static const std::string dtypes[] = { "", "integer", "scalar", "logical", "string" };
        return dtypes[(size_t)value.kind()];
    }


    inline void check_supported_rank( const std::string& op, const size_t rank, const size_t max )
    {
        if ( rank > max )
        {
            throw std::runtime_error("operation not implemented: " + op + " with rank = " + std::to_string(rank));
        }
    }


    inline void execute_external( const Operation& op, TensorDict& tensors )
    {
    }

    inline void execute_variable( const Operation& op, TensorDict& tensors )
    {
    }

    template<typename T>
    inline void _execute_constant( const Operation& op, TensorDict& tensors )
    {
        auto& output = op.outputs.get("output");
        auto& value = op.attribs.get("value");
        
        auto& tensor = tensors.at(output.identifier());
        const size_t n = volume_of(tensor.shape);
        auto data = (T*)tensor.data.data();
        
        if ( value.kind() == Value::Array )
        {
            if ( value.size() == n )
            {
                for ( size_t i = 0; i < n; ++i )
                {
                    data[i] = value[i].get<T>();
                }
            }
            else
            {
                std::fill_n(data, n, value[0].scalar());
            }
        }
        else
        {
            std::fill_n(data, n, value.scalar());
        }
    }

    DISPATCH_BY_DTYPE(constant)
    

    template<typename T, typename F>
    Executor make_unary_executor( const F func )
    {
        return [=]( const Operation& op, TensorDict& tensors )
        {
            auto& x = op.inputs.get("x");
            auto& y = op.outputs.get("y");
            
            unary(_tensor_view<const T>(x, tensors), _tensor_view<T>(y, tensors), func);
        };
    }
    
    template<typename T, typename F, typename... S>
    Executor make_unary_executor_ext( const F func, const S ...attrib )
    {
        return [=]( const Operation& op, TensorDict& tensors )
        {
            auto& x = op.inputs.get("x");
            auto& y = op.outputs.get("y");
            
            unary(_tensor_view<const T>(x, tensors), _tensor_view<T>(y, tensors), [&]( const T x )
            {
                return func(x, op.attribs.get(attrib).scalar()...);
            });
        };
    }
    
    template<typename T, typename R = T, typename F>
    Executor make_binary_executor( const F func )
    {
        return [=]( const Operation& op, TensorDict& tensors )
        {
            auto& x = op.inputs.get("x");
            auto& y = op.inputs.get("y");
            auto& z = op.outputs.get("z");
            
            binary(_tensor_view<const T>(x, tensors), _tensor_view<const T>(y, tensors), _tensor_view<R>(z, tensors), func);
        };
    }

    template<typename T, typename F>
    Executor make_reduce_executor( const F func, const T init )
    {
        return [=]( const Operation& op, TensorDict& tensors )
        {
            auto& input = op.inputs.get("input");
            auto& output = op.outputs.get("output");
            
            auto input_view = _tensor_view<const T>(input, tensors);
            auto output_view = _tensor_view<T>(output, tensors);
            
            reduce(input_view, output_view, func, init);
            
            if ( op.name == "mean_reduce" || (op.name == "sum_reduce" && op.attribs.get("normalize").logical()) )
            {
                const T volume = (T)(input_view.volume / output_view.volume);
                binary((tensor_view<const T>)output_view, _tensor_view<const T>(volume), output_view, std::divides<T>());
            }
        };
    }
    
    template<typename T>
    void _execute_select( const Operation& op, TensorDict& tensors )
    {
        auto& c = op.inputs.get("condition");
        auto& x = op.inputs.get("true_value");
        auto& y = op.inputs.get("false_value");
        auto& z = op.outputs.get("output");
        
        select(_tensor_view<const bool>(c, tensors),
               _tensor_view<const T>(x, tensors),
               _tensor_view<const T>(y, tensors),
               _tensor_view<T>(z, tensors));
    }

    DISPATCH_BY_DTYPE(select)

    inline Shape _extract_items( const Value& value )
    {
        Shape items(value.size());
        for ( size_t i = 0; i < value.size(); ++i )
        {
            auto& v = value[i];
            items[i] = v.kind() == Value::Tuple ? v[0].integer() : v.integer();
        }
        return items;
    }

    inline Shape _make_padding( const size_t rank, const int input[], const int output[], const int filter[],
                              const int stride[], const int dilation[] )
    {
        Shape padding(rank);
        for ( size_t i = 0; i < rank; ++i )
        {
            padding[i] = std::max((output[i] - 1) * stride[i] + (filter[i] - 1) * dilation[i] + 1 - input[i], 0) / 2;
        }
        return padding;
    }

    template<bool Transposed, typename T>
    void execute_conv( const Operation& op, TensorDict& tensors )
    {
        auto& input = op.inputs.get("input");
        auto& filter = op.inputs.get("filter");
        auto& bias = op.inputs.get("bias");
        auto& output = op.outputs.get("output");
        
        auto& padding = op.attribs.get("padding");
        auto& stride = op.attribs.get("stride");
        auto& dilation = op.attribs.get("dilation");
        auto& groups = op.attribs.get("groups").integer();
        auto& border = op.attribs.get("border").string();
        
        if ( border != "constant" )
        {
            throw std::runtime_error("operation not implemented: " + op.name + " with border = '" + border + "'");
        }
        
        auto input_view = _tensor_view<T>(Transposed ? output : input, tensors);
        auto output_view = _tensor_view<T>(Transposed ? input : output, tensors);
        auto filter_view = _tensor_view<const T>(filter, tensors);
        auto bias_view = _tensor_view<const T>(bias, tensors);
        
        const size_t d = input_view.rank - 2;
        check_supported_rank(op.name, d, 3);
        
        const Shape strideShape = stride.size() ? _extract_items(stride) : Shape(d, 1);
        const Shape dilationShape = dilation.size() ? _extract_items(dilation) : Shape(d, 1);
        const Shape paddingShape = padding.size() ? _extract_items(padding) : _make_padding(d, input_view.shape + 2,
                                                                                            output_view.shape + 2, filter_view.shape + 2,
                                                                                            strideShape.data(), dilationShape.data());
        if ( groups == 1 )
        {
            conv<Transposed>(filter_view, bias_view, input_view, output_view,
                             paddingShape.data(), strideShape.data(), dilationShape.data());
        }
        else if ( groups == 0 || groups == input_view.shape[1] )
        {
            depthwise_conv<Transposed>(filter_view, bias_view, input_view, output_view,
                                       paddingShape.data(), strideShape.data(), dilationShape.data());
        }
        else
        {
            grouped_conv<Transposed>(filter_view, bias_view, input_view, output_view,
                                     paddingShape.data(), strideShape.data(), dilationShape.data(), groups);
        }
    }

    template<bool Transposed, typename T, typename F>
    void _execute_pool( const Operation& op, TensorDict& tensors, const F func, const T init )
    {
        auto& input = op.inputs.get("input");
        auto& output = op.outputs.get("output");
        
        auto& size = op.attribs.get("size");
        auto& padding = op.attribs.get("padding");
        auto& stride = op.attribs.get("stride");
        auto& dilation = op.attribs.get("dilation");
        auto& border = op.attribs.get("border").string();
        
        if ( border != "constant" && border != "ignore" )
        {
            throw std::runtime_error("operation not implemented: " + op.name + " with border = '" + border + "'");
        }
        
        auto input_view = _tensor_view<T>(Transposed ? output : input, tensors);
        auto output_view = _tensor_view<T>(Transposed ? input : output, tensors);
        
        const size_t d = input_view.rank;
        check_supported_rank(op.name, d, 5);
        
        const Shape sizeShape = _extract_items(size);
        const Shape strideShape = stride.size() ? _extract_items(stride) : Shape(d, 1);
        const Shape dilationShape = dilation.size() ? _extract_items(dilation) : Shape(d, 1);
        const Shape paddingShape = padding.size() ? _extract_items(padding) : _make_padding(d, input_view.shape, output_view.shape,
                                                                                            sizeShape.data(), strideShape.data(),
                                                                                            dilationShape.data());
        
        pool<Transposed>(input_view, output_view, sizeShape.data(), paddingShape.data(), strideShape.data(), dilationShape.data(),
                         func, init, border != "ignore");
        
        if ( op.name == "avg_pool" || op.name == "avg_unpool" || ((op.name == "box" || op.name == "debox") &&
                                                                  op.attribs.get("normalize").logical()) )
        {
            if ( border == "constant" )
            {
                const T volume = (T)volume_of(sizeShape);
                binary((tensor_view<const T>)output_view, _tensor_view<const T>(volume), output_view, std::divides<T>());
            }
            else if ( border == "ignore" )
            {
                Tensor tensor = _make_tensor(d, output_view.shape, sizeof(T));
                
                pool_area<Transposed>(_tensor_view<T>(tensor), input_view.shape, output_view.shape, sizeShape.data(),
                                      paddingShape.data(), strideShape.data(), dilationShape.data());
                
                binary((tensor_view<const T>)output_view, _tensor_view<const T>(tensor), output_view, std::divides<T>());
            }
        }
    }

    template<bool Transposed, typename T, typename F>
    Executor make_pool_executor( const F func, const T init )
    {
        return [=]( const Operation& op, TensorDict& tensors )
        {
            _execute_pool<Transposed>(op, tensors, func, init);
        };
    }
    
    template<typename T>
    void _execute_reshape( const Operation& op, TensorDict& tensors )
    {
        auto& input = op.inputs.get("input");
        auto& output = op.outputs.get("output");
        
        auto input_view = _tensor_view<const T>(input, tensors);
        auto output_view = _tensor_view<T>(output, tensors);
        
        std::copy_n(input_view.data, input_view.volume, output_view.data);
    }

    DISPATCH_BY_DTYPE(reshape)
    
    template<typename T>
    void _execute_transpose( const Operation& op, TensorDict& tensors )
    {
        auto& input = op.inputs.get("input");
        auto& output = op.outputs.get("output");
        auto& axes = op.attribs.get("axes");
        
        const size_t rank = tensors.at(input.identifier()).shape.size();
        check_supported_rank(op.name, rank, 5);
        
        std::vector<size_t> perm(rank);
        for ( size_t i = 0; i < axes.size(); ++i )
        {
            perm[i] = axes[i].integer();
        }
        std::iota(perm.begin() + axes.size(), perm.end(), axes.size());
        
        transpose(_tensor_view<const T>(input, tensors), _tensor_view<T>(output, tensors), perm.data());
    }

    DISPATCH_BY_DTYPE(transpose)

    template<typename T>
    void _execute_concat( const Operation& op, TensorDict& tensors )
    {
        auto& values = op.inputs.get("values");
        auto& value = op.outputs.get("value");
        auto& axis = op.attribs.get("axis").integer();
        
        std::vector<tensor_view<const T>> v;
        for ( size_t i = 0; i < values.size(); ++i )
        {
            v.emplace_back(_tensor_view<const T>(values[i], tensors));
        }
        
        if ( op.name == "stack" )
        {
            concat<true>(v.size(), v.data(), _tensor_view<T>(value, tensors), axis);
        }
        else
        {
            concat<false>(v.size(), v.data(), _tensor_view<T>(value, tensors), axis);
        }
    }

    DISPATCH_BY_DTYPE(concat)

    template<typename T>
    void _execute_split( const Operation& op, TensorDict& tensors )
    {
        auto& value = op.inputs.get("value");
        auto& values = op.outputs.get("values");
        auto& axis = op.attribs.get("axis").integer();
        
        std::vector<tensor_view<T>> v;
        for ( size_t i = 0; i < values.size(); ++i )
        {
            v.emplace_back(_tensor_view<T>(values[i], tensors));
        }
        
        if ( op.name == "unstack" )
        {
            split<true>(v.size(), _tensor_view<const T>(value, tensors), v.data(), axis);
        }
        else
        {
            split<false>(v.size(), _tensor_view<const T>(value, tensors), v.data(), axis);
        }
    }

    DISPATCH_BY_DTYPE(split)

    template<typename T>
    void execute_pad( const Operation& op, TensorDict& tensors )
    {
        auto& input = op.inputs.get("input");
        auto& output = op.outputs.get("output");
        auto& padding = op.attribs.get("padding");
        auto& border = op.attribs.get("border").string();
        auto& value = op.attribs.get("value");
        
        auto input_view = _tensor_view<T>(input, tensors);
        auto output_view = _tensor_view<T>(output, tensors);
        
        auto paddingShape = _extract_items(padding);
        
        const size_t d = input_view.rank;
        check_supported_rank(op.name, d, 5);
        
        if ( border == "constant" )
        {
            pad_constant<T>(input_view, output_view, paddingShape.data(), value.get<T>());
        }
        else if ( border == "replicate" )
        {
            pad_replicate<T>(input_view, output_view, paddingShape.data());
        }
        else if ( border == "reflect" )
        {
            pad_reflect<T>(input_view, output_view, paddingShape.data());
        }
        else if ( border == "reflect-even" )
        {
            pad_reflect_even<T>(input_view, output_view, paddingShape.data());
        }
        else
        {
            throw std::runtime_error("operation not implemented: pad with border == '" + border + "'");
        }
    }

    template<typename T>
    void _execute_tile( const Operation& op, TensorDict& tensors )
    {
        auto& input = op.inputs.get("input");
        auto& output = op.outputs.get("output");
        
        auto input_view = _tensor_view<T>(input, tensors);
        auto output_view = _tensor_view<T>(output, tensors);
        
        const size_t d = input_view.rank;
        check_supported_rank(op.name, d, 5);
        
        tile<T>(input_view, output_view);
    }

    DISPATCH_BY_DTYPE(tile)

    template<typename T>
    void _execute_slice( const Operation& op, TensorDict& tensors )
    {
        auto& input = op.inputs.get("input");
        auto& output = op.outputs.get("output");
        auto& axes = op.attribs.get("axes");
        auto& begin = op.attribs.get("begin");
        auto& stride = op.attribs.get("stride");
        
        auto input_view = _tensor_view<T>(input, tensors);
        auto output_view = _tensor_view<T>(output, tensors);
        
        const size_t d = input_view.rank;
        check_supported_rank(op.name, d, 5);
        
        std::vector<int> offset(d, 0);
        std::vector<int> step(d, 1);
        for ( size_t i = 0; i < axes.size(); ++i )
        {
            auto axis = axes[i].integer();
            auto offs = begin[i].integer();
            if ( offs < 0 )
            {
                offs += input_view.shape[axis];
            }
            if ( offs < 0 )
            {
                offs = -1;
            }
            if ( offs > input_view.shape[axis] )
            {
                offs = input_view.shape[axis];
            }
            
            offset[axis] = offs;
            step[axis] = stride.size() ? stride[i].integer() : 1;
        }
        
        slice<T>(input_view, output_view, offset.data(), step.data());
    }

    DISPATCH_BY_DTYPE(slice)
    
    template<typename T>
    void _execute_gather( const Operation& op, TensorDict& tensors )
    {
        auto& input = op.inputs.get("input");
        auto& indices = op.inputs.get("indices");
        auto& output = op.outputs.get("output");
        auto& axis = op.attribs.get("axis").integer();
        
        auto input_view = _tensor_view<T>(input, tensors);
        auto indices_view = _tensor_view<const int>(indices, tensors);
        auto output_view = _tensor_view<T>(output, tensors);
        
        gather<T>(input_view, indices_view, output_view, axis);
    }
    
    DISPATCH_BY_DTYPE(gather)

    template<typename T>
    void _execute_cast( const Operation& op, TensorDict& tensors )
    {
        auto& input = op.inputs.get("input");
        auto& output = op.outputs.get("output");
        
        auto& input_dtype = input.kind() == Value::Identifier ? tensors.at(input.identifier()).dtype : _literal_dtype(input);
        auto output_view = _tensor_view<T>(output, tensors);
        
        if ( input_dtype == "scalar" )
        {
            auto input_view = _tensor_view<const Value::scalar_t>(input, tensors);
            std::copy_n(input_view.data, input_view.volume, output_view.data);
        }
        else if ( input_dtype == "integer" )
        {
            auto input_view = _tensor_view<const Value::integer_t>(input, tensors);
            std::copy_n(input_view.data, input_view.volume, output_view.data);
        }
        else if ( input_dtype == "logical" )
        {
            auto input_view = _tensor_view<const Value::logical_t>(input, tensors);
            std::copy_n(input_view.data, input_view.volume, output_view.data);
        }
        else
        {
            throw std::runtime_error("operation 'cast' from dtype 'string' is not implemented");
        }
    }

    DISPATCH_BY_DTYPE(cast)

    template<typename T>
    void execute_matmul( const Operation& op, TensorDict& tensors )
    {
        auto& A = op.inputs.get("A");
        auto& B = op.inputs.get("B");
        auto& C = op.outputs.get("C");
        
        bool trA = op.attribs.get("transposeA").logical();
        bool trB = op.attribs.get("transposeB").logical();
        
        matmul(trA, trB, _tensor_view<const T>(A, tensors), _tensor_view<const T>(B, tensors), _tensor_view<T>(C, tensors));
    }

    template<typename T>
    void execute_linear( const Operation& op, TensorDict& tensors )
    {
        auto& input = op.inputs.get("input");
        auto& filter = op.inputs.get("filter");
        auto& bias = op.inputs.get("bias");
        auto& output = op.outputs.get("output");
        
        linear(_tensor_view<const T>(filter, tensors), _tensor_view<const T>(bias, tensors),
               _tensor_view<const T>(input, tensors), _tensor_view<T>(output, tensors));
    }
    
    template<typename T>
    void execute_softmax( const Operation& op, TensorDict& tensors )
    {
        auto& input = op.inputs.get("x");
        auto& output = op.outputs.get("y");
        auto& axes = op.attribs.get("axes");
        
        auto input_view = _tensor_view<const T>(input, tensors);
        auto output_view = _tensor_view<T>(output, tensors);
        
        if ( axes.size() != 1 )
        {
            throw std::runtime_error("operation not implemented: softmax with multiple axes");
        }
        
        softmax(input_view, output_view, axes[0].integer());
    }

    template<typename T, typename I, typename F>
    Executor make_arg_reduce_executor( const F func )
    {
        return [=]( const Operation& op, TensorDict& tensors )
        {
            auto& input = op.inputs.get("input");
            auto& output = op.outputs.get("output");
            auto& axes = op.attribs.get("axes");
            
            auto input_view = _tensor_view<const T>(input, tensors);
            auto output_view = _tensor_view<I>(output, tensors);
            
            if ( axes.size() != 1 )
            {
                throw std::runtime_error("operation not implemented: argmax_reduce with multiple axes");
            }
            
            arg_reduce(input_view, output_view, axes[0].integer(), func);
        };
    }

    template<typename T>
    void execute_multilinear_upsample( const Operation& op, TensorDict& tensors )
    {
        auto& input = op.inputs.get("input");
        auto& output = op.outputs.get("output");
        auto& factor = op.attribs.get("factor");
        auto& border = op.attribs.get("border").string();
        auto& method = op.attribs.get("method").string();
        
        auto input_view = _tensor_view<T>(input, tensors);
        auto output_view = _tensor_view<T>(output, tensors);
        
        const size_t d = input_view.rank - 2;
        check_supported_rank(op.name, d, 2);
        
        for ( size_t i = 0; i < factor.size(); ++i )
        {
            if ( factor[i].integer() != 2 )
            {
                throw std::runtime_error("operation not implemented: multilinear_upsample with factor != 2");
            }
        }
        
        if ( method == "aligned" )
        {
            throw std::runtime_error("operation not implemented: multilinear_upsample with method == 'aligned'");
        }
        
        if ( border == "constant" )
        {
            if ( method == "symmetric" )
            {
                multilinear_upsample2x_symmetric(input_view, output_view);
            }
            else if ( method == "asymmetric" )
            {
                multilinear_upsample2x_asymmetric(input_view, output_view);
            }
        }
        else if ( border == "replicate" )
        {
            Shape input_padding(input_view.rank, 0);
            for ( size_t i = 2; i < input_view.rank; ++i )
            {
                input_padding[i] = 1;
            }
            
            Shape output_padding(output_view.rank, 0);
            for ( size_t i = 2; i < output_view.rank; ++i )
            {
                output_padding[i] = factor[i-2].integer();
            }
            
            Shape padded_input_shape(input_view.shape, input_view.shape + input_view.rank);
            for ( size_t i = 2; i < padded_input_shape.size(); ++i )
            {
                padded_input_shape[i] += 1 + 1;
            }
            
            Shape padded_output_shape(output_view.shape, output_view.shape + output_view.rank);
            for ( size_t i = 2; i < padded_output_shape.size(); ++i )
            {
                padded_output_shape[i] += 2 * factor[i-2].integer();
            }
            
            Tensor padded_input = _make_tensor(padded_input_shape.size(), padded_input_shape.data(), sizeof(T));
            Tensor padded_output = _make_tensor(padded_output_shape.size(), padded_output_shape.data(), sizeof(T));
            
            pad_replicate((tensor_view<const T>)input_view, _tensor_view<T>(padded_input), input_padding.data());
            
            if ( method == "symmetric" )
            {
                multilinear_upsample2x_symmetric(_tensor_view<T>(padded_input), _tensor_view<T>(padded_output));
            }
            else if ( method == "asymmetric" )
            {
                multilinear_upsample2x_asymmetric(_tensor_view<T>(padded_input), _tensor_view<T>(padded_output));
            }
            
            const Shape stride(input_view.rank, 1);
            
            slice(_tensor_view<const T>(padded_output), output_view, output_padding.data(), stride.data());
        }
        else
        {
            throw std::runtime_error("operation not implemented: multilinear_upsample with border == '" + border + "'");
        }
    }

    template<typename T>
    void _execute_update( const Operation& op, TensorDict& tensors )
    {
        auto& value = op.inputs.get("value");
        auto& result = op.outputs.get("result");
        
        auto input_view = _tensor_view<const T>(value, tensors);
        auto output_view = _tensor_view<T>(result, tensors);
        
        std::copy_n(input_view.data, input_view.volume, output_view.data);
    }

    DISPATCH_BY_DTYPE(update)

    
    static const std::map<std::string,Executor> Executors =
    {
        { "external", execute_external },
        { "constant", execute_constant },
        { "variable", execute_variable },
        
        { "neg", make_unary_executor<float>(std::negate<float>()) },
        { "not", make_unary_executor<bool>(std::logical_not<bool>()) },
        { "abs", make_unary_executor<float>([]( float x ){ return std::abs(x); }) },
        { "sign", make_unary_executor<float>([]( float x ){ return x > 0.f ? 1.f : x < 0.f ? -1.f : 0.f; }) },
        { "exp", make_unary_executor<float>([]( float x ){ return std::exp(x); }) },
        { "log", make_unary_executor<float>([]( float x ){ return std::log(x); }) },
        { "log2", make_unary_executor<float>([]( float x ){ return std::log(x) / std::log(2.f); }) },
        { "sin", make_unary_executor<float>([]( float x ){ return std::sin(x); }) },
        { "cos", make_unary_executor<float>([]( float x ){ return std::cos(x); }) },
        { "tan", make_unary_executor<float>([]( float x ){ return std::tan(x); }) },
        { "asin", make_unary_executor<float>([]( float x ){ return std::asin(x); }) },
        { "acos", make_unary_executor<float>([]( float x ){ return std::acos(x); }) },
        { "atan", make_unary_executor<float>([]( float x ){ return std::atan(x); }) },
        { "sinh", make_unary_executor<float>([]( float x ){ return std::sinh(x); }) },
        { "cosh", make_unary_executor<float>([]( float x ){ return std::cosh(x); }) },
        { "tanh", make_unary_executor<float>([]( float x ){ return std::tanh(x); }) },
        { "asinh", make_unary_executor<float>([]( float x ){ return std::asinh(x); }) },
        { "acosh", make_unary_executor<float>([]( float x ){ return std::acosh(x); }) },
        { "atanh", make_unary_executor<float>([]( float x ){ return std::atanh(x); }) },
        { "round", make_unary_executor<float>([]( float x ){ return std::round(x); }) },
        { "floor", make_unary_executor<float>([]( float x ){ return std::floor(x); }) },
        { "ceil", make_unary_executor<float>([]( float x ){ return std::ceil(x); }) },
        { "sqrt", make_unary_executor<float>([]( float x ){ return std::sqrt(x); }) },
        { "sqr", make_unary_executor<float>([]( float x ){ return x * x; }) },
        { "rsqrt", make_unary_executor<float>([]( float x ){ return 1.f / std::sqrt(x); }) },
        { "rsqr", make_unary_executor<float>([]( float x ){ return 1.f / (x * x); }) },
        { "rcp", make_unary_executor<float>([]( float x ){ return 1.f / x; }) },
        { "copy", make_unary_executor<float>([]( float x ){ return x; }) },
        
        { "sigmoid", make_unary_executor<float>([]( float x ){ return 1.f / (1.f + std::exp(-x)); }) },
        { "tanh", make_unary_executor<float>([]( float x ){ return std::tanh(x); }) },
        { "relu", make_unary_executor<float>([]( float x ){ return std::max(x, 0.f); }) },
        { "leaky_relu", make_unary_executor_ext<float>([]( float x, float alpha )
            { return x < 0.f ? alpha * x : x; }, "alpha") },
        { "elu", make_unary_executor_ext<float>([]( float x, float alpha )
            { return x < 0.f ? alpha * (std::exp(x) - 1.f) : x; }, "alpha") },
        { "selu", make_unary_executor_ext<float>([]( float x, float alpha, float lambda )
            { return lambda * (x < 0.f ? alpha * (std::exp(x) - 1.f) : x); }, "alpha", "lambda") },
        { "gelu", make_unary_executor<float>([]( float x ){ return x / (1.f + std::exp(-1.702f * x)); }) },
        { "silu", make_unary_executor<float>([]( float x ){ return x / (1.f + std::exp(-x)); }) },
        { "softplus", make_unary_executor<float>([]( float x ){ return std::log(std::exp(x) + 1.f); }) },
        
        { "add", make_binary_executor<float>(std::plus<float>()) },
        { "sub", make_binary_executor<float>(std::minus<float>()) },
        { "mul", make_binary_executor<float>(std::multiplies<float>()) },
        { "div", make_binary_executor<float>(std::divides<float>()) },
        { "pow", make_binary_executor<float>([]( float x, float y ){ return std::pow(x,y); }) },
        { "min", make_binary_executor<float>([]( float x, float y ){ return std::min(x,y); }) },
        { "max", make_binary_executor<float>([]( float x, float y ){ return std::max(x,y); }) },
        { "and", make_binary_executor<bool>(std::logical_and<bool>()) },
        { "or", make_binary_executor<bool>(std::logical_or<bool>()) },
        { "lt", make_binary_executor<float,bool>(std::less<float>()) },
        { "gt", make_binary_executor<float,bool>(std::greater<float>()) },
        { "le", make_binary_executor<float,bool>(std::less_equal<float>()) },
        { "ge", make_binary_executor<float,bool>(std::greater_equal<float>()) },
        { "eq", make_binary_executor<float,bool>(std::equal_to<float>()) },
        { "ne", make_binary_executor<float,bool>(std::not_equal_to<float>()) },
        
        { "select", execute_select },
        
        { "sum_reduce", make_reduce_executor(std::plus<float>(), 0.f) },
        { "mean_reduce", make_reduce_executor(std::plus<float>(), 0.f) },
        { "min_reduce", make_reduce_executor([]( float x, float y ){ return std::min(x,y); }, std::numeric_limits<float>::infinity()) },
        { "max_reduce", make_reduce_executor([]( float x, float y ){ return std::max(x,y); }, -std::numeric_limits<float>::infinity()) },
        { "any_reduce", make_reduce_executor(std::logical_or<bool>(), false) },
        { "all_reduce", make_reduce_executor(std::logical_and<bool>(), true) },
        
        { "conv", execute_conv<false,float> },
        { "deconv", execute_conv<true,float> },
        
        { "box", make_pool_executor<false>(std::plus<float>(), 0.f) },
        { "debox", make_pool_executor<true>(std::plus<float>(), 0.f) },
        { "sum_pool", make_pool_executor<false>(std::plus<float>(), 0.f) },
        { "sum_unpool", make_pool_executor<true>(std::plus<float>(), 0.f) },
        { "avg_pool", make_pool_executor<false>(std::plus<float>(), 0.f) },
        { "avg_unpool", make_pool_executor<true>(std::plus<float>(), 0.f) },
        { "min_pool", make_pool_executor<false>([]( float x, float y ){ return std::min(x,y); }, std::numeric_limits<float>::infinity()) },
        { "max_pool", make_pool_executor<false>([]( float x, float y ){ return std::max(x,y); }, -std::numeric_limits<float>::infinity()) },
        
        { "reshape", execute_reshape },
        { "squeeze", execute_reshape },
        { "unsqueeze", execute_reshape },
        { "transpose", execute_transpose },
        
        { "concat", execute_concat },
        { "split", execute_split },
        { "stack", execute_concat },
        { "unstack", execute_split },
        { "pad", execute_pad<float> },
        { "tile", execute_tile },
        { "slice", execute_slice },
        { "gather", execute_gather },
        { "cast", execute_cast },
        
        { "matmul", execute_matmul<float> },
        { "linear", execute_linear<float> },
        
        { "softmax", execute_softmax<float> },
        { "argmin_reduce", make_arg_reduce_executor<float,int>(std::less<float>()) },
        { "argmax_reduce", make_arg_reduce_executor<float,int>(std::greater<float>()) },
        
        { "multilinear_upsample", execute_multilinear_upsample<float> },
        
        { "update", execute_update },
    };

}}   // namespace nnef::rt


#endif
