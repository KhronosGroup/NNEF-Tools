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

#include <fstream>
#include <iterator>

#include "nnef.h"
#include "nnef/comp/comp_parser.h"
#include "nnef/flat/quant_parser.h"
#include "nnef/common/binary.h"
#include "nnef/common/shapes.h"
#include "nnef/runtime/execution.h"


namespace nnef
{
    
    struct ParseCallback : public Parser::Callback
    {
        Graph& graph;
        
        std::istream& qis;
        const std::string& qfn;
        Dictionary<Dictionary<Value>> quantizations;
        
        ParseCallback( Graph& graph, std::istream& qis, const std::string& qfn )
        : graph(graph), qis(qis), qfn(qfn)
        {
        }
        
        virtual void beginGraph( const Prototype& proto, const Dictionary<Prototype>& fragments )
        {
            graph.name = proto.name();
            graph.operations.clear();
            graph.tensors.clear();
            
            graph.inputs.resize(proto.paramCount());
            for ( size_t i = 0; i < proto.paramCount(); ++i )
            {
                graph.inputs[i] = proto.param(i).name();
            }
            
            graph.outputs.resize(proto.resultCount());
            for ( size_t i = 0; i < proto.resultCount(); ++i )
            {
                graph.outputs[i] = proto.result(i).name();
            }
            
            if ( qis )
            {
                quantizations = nnef::QuantParser::parse(qis, qfn.c_str(), fragments);
            }
        }
        
        virtual void endGraph( const Prototype& proto, const Dictionary<Typename>& dtypes )
        {
            for ( auto& it : dtypes )
            {
                Tensor tensor;
                tensor.name = it.first;
                tensor.dtype = toString(it.second);
                if ( quantizations.count(it.first) )
                {
                    for ( auto& item : quantizations.at(it.first) )
                    {
                        tensor.quantization.push_back(item);
                    }
                }
                
                graph.tensors.emplace(it.first, std::move(tensor));
            }
        }
        
        virtual void operation( const Prototype& proto, const Dictionary<Value>& args, const Dictionary<Typename>& dtypes )
        {
            Operation operation;
            operation.name = proto.name();
            operation.dtype = args.count("?") ? args.at("?").string() : std::string();
            
            for ( size_t i = 0; i < proto.paramCount(); ++i )
            {
                auto& param = proto.param(i);
                auto& value = args.at(param.name());
                if ( param.type()->isAttribute() )
                {
                    operation.attribs.emplace_back(param.name(), value);
                }
                else
                {
                    operation.inputs.emplace_back(param.name(), value);
                }
            }
            for ( size_t i = 0; i < proto.resultCount(); ++i )
            {
                auto& result = proto.result(i);
                auto& value = args.at(result.name());
                operation.outputs.emplace_back(result.name(), value);
            }
            
            graph.operations.push_back(std::move(operation));
        }
    };
    
    std::string format_error_position( const Error::Position& pos )
    {
        return "'" + std::string(pos.filename) + "' [" + std::to_string(pos.line) + ":" + std::to_string(pos.column) + "]";
    }
    
    bool parse( std::istream& graph_is, const std::string& graph_fn, std::istream& quant_is, const std::string& quant_fn,
               Graph& graph, std::string& error, const std::string& stdlib, const std::set<std::string>& lowered ) noexcept
    {
        ParseCallback callback(graph, quant_is, quant_fn);
        CompParser parser(stdlib, lowered);
        
        try
        {
            parser.parse(graph_is, graph_fn.c_str(), callback);
            return true;
        }
        catch ( const nnef::Error& e )
        {
            error = "Parse error in file " + format_error_position(e.position()) + " " + e.what();
            
            auto origin = e.position().origin;
            while ( origin )
            {
                error += "\n... evaluated from file " + format_error_position(e.position());
                origin = origin->origin;
            }
            return false;
        }
    }
    
    bool parse_file( const std::string& graph_fn, const std::string& quant_fn, Graph& graph, std::string& error,
                     const std::string& stdlib, const std::set<std::string>& lowered ) noexcept
    {
        std::ifstream graph_is(graph_fn);
        if ( !graph_is )
        {
            error = "Could not open graph file: " + std::string(graph_fn);
            return false;
        }
        
        std::ifstream quant_is;
        if ( !quant_fn.empty() )
        {
            quant_is.open(quant_fn);
            if ( !quant_is )
            {
                error = "Could not open quantization file: " + std::string(quant_fn);
                return false;
            }
        }
        
        return parse(graph_is, graph_fn, quant_is, quant_fn, graph, error, stdlib, lowered);
    }
    
    bool parse_string( const std::string& graph_str, const std::string& quant_str, Graph& graph, std::string& error,
                      const std::string& stdlib, const std::set<std::string>& lowered ) noexcept
    {
        std::stringstream graph_is(graph_str);
        std::stringstream quant_is;
        if ( !quant_str.empty() )
        {
            quant_is.str(quant_str);
        }
        return parse(graph_is, "input", quant_is, "quantization", graph, error, stdlib, lowered);
    }

    size_t item_bytes( const std::string& dtype )
    {
        return dtype == "scalar" ? sizeof(float) : dtype == "integer" ? sizeof(int) : dtype == "logical" ? sizeof(bool) : 0;
    }
    
    size_t item_bits( const std::string& dtype )
    {
        return dtype == "scalar" ? 32 : dtype == "integer" ? sizeof(int) * 8 : dtype == "logical" ? 1 : 0;
    }

    bool read_tensor( std::istream& is, Tensor& tensor, std::string& error ) noexcept
    {
        TensorHeader header;
        is.read((char*)&header, sizeof(header));
        if ( header.item_type == TensorHeader::Uint && header.reserved[0] != 0 )
        {
            header.item_type = TensorHeader::Int;
        }
        
        try
        {
            validate_tensor_header(header);
            tensor.shape.assign(header.extents, header.extents + header.rank);
        }
        catch ( const nnef::Error& e )
        {
            error = "Invalid tensor header: " + std::string(e.what());
            return false;
        }
        
        std::vector<char> bytes(header.data_length);
        is.read(bytes.data(), bytes.size());
        
        if ( !is )
        {
            error = "Failed to read tensor data";
            return false;
        }

        const size_t count = volume_of(tensor.shape);
        
        if ( header.item_type == TensorHeader::Float )
        {
            tensor.dtype = "scalar";
            tensor.data.resize(count * sizeof(float));
            from_bytes(bytes.data(), count, header.bits_per_item, (float*)tensor.data.data());
        }
        else if ( header.item_type == TensorHeader::Bool )
        {
            tensor.dtype = "logical";
            tensor.data.resize(count * sizeof(bool));
            from_bytes(bytes.data(), count, header.bits_per_item, (bool*)tensor.data.data());
        }
        else if ( header.item_type == TensorHeader::Int || header.item_type == TensorHeader::Uint )
        {
            tensor.dtype = "integer";
            tensor.data.resize(count * sizeof(int));
            from_bytes(bytes.data(), count, header.bits_per_item, (int*)tensor.data.data(), header.item_type == TensorHeader::Int);
        }
        else if ( header.item_type == TensorHeader::Qint || header.item_type == TensorHeader::Quint )
        {
            tensor.dtype = "scalar";
            tensor.data.resize(header.data_length);
            tensor.data = bytes;
            tensor.quantization.emplace_back("signed", Value::logical(header.item_type == TensorHeader::Qint));
        }
        else
        {
            error = "Unsupported tensor item-type '" + std::to_string(header.item_type) + "' and bits per item '" + std::to_string(header.bits_per_item) + "'";
            return false;
        }
        
        return (bool)is;
    }

    bool write_tensor( std::ostream& os, const Tensor& tensor, std::string& error ) noexcept
    {
        if ( tensor.shape.size() > TensorHeader::MaxRank )
        {
            error = "Tensor rank " + std::to_string(tensor.shape.size()) + " exceeds maximum allowed rank (" + std::to_string(TensorHeader::MaxRank) + ")";
            return false;
        }
        
        const bool quantized = !tensor.quantization.empty();
        const bool is_signed = tensor.quantization.get("signed", Value::logical(true)).logical();
        const TensorHeader::ItemType item_type = quantized ? (is_signed ? TensorHeader::Qint : TensorHeader::Quint) :
                                                 tensor.dtype == "scalar" ? TensorHeader::Float :
                                                 tensor.dtype == "integer" ? TensorHeader::Int : TensorHeader::Bool;
        
        TensorHeader header;
        const size_t version[] = { 1, 0 };
        const size_t count = volume_of(tensor.shape);
        const size_t bits_per_item = quantized ? tensor.data.size() * 8 / count : item_bits(tensor.dtype);
        fill_tensor_header(header, version, tensor.shape.size(), tensor.shape.data(), bits_per_item, item_type);
        
        std::vector<char> bytes(header.data_length);
        
        if ( tensor.dtype == "scalar" )
        {
            if ( quantized )
            {
                bytes = tensor.data;
            }
            else
            {
                to_bytes((const float*)tensor.data.data(), count, bytes.data());
            }
        }
        else if ( tensor.dtype == "integer" )
        {
            to_bytes((const int*)tensor.data.data(), count, bytes.data(), true);
        }
        else if ( tensor.dtype == "logical" )
        {
            to_bytes((const bool*)tensor.data.data(), count, bytes.data());
        }
        else
        {
            error = "Invalid tensor data-type: '" + tensor.dtype + "'";
            return false;
        }
        
        os.write((char*)&header, sizeof(header));
        os.write(bytes.data(), bytes.size());
        
        if ( !os )
        {
            error = "Failed to write tensor data";
            return false;
        }
        return true;
    }

    bool read_tensor( const std::string& filename, Tensor& tensor, std::string& error ) noexcept
    {
        std::ifstream is(filename, std::ios::binary);
        if ( !is )
        {
            error = "Could not open tensor file: " + filename;
            return false;
        }
        return read_tensor(is, tensor, error);
    }

    bool write_tensor( const std::string& filename, const Tensor& tensor, std::string& error ) noexcept
    {
        std::ofstream os(filename, std::ios::binary);
        if ( !os )
        {
            error = "Could not open tensor file: " + filename;
            return false;
        }
        return write_tensor(os, tensor, error);
    }
    
    bool load_variables( const std::string& path, Graph& graph, std::string& error ) noexcept
    {
        const std::string sep = path.back() == '/' || path.back() == '\\' ? "" : "/";
        
        for ( auto& op : graph.operations )
        {
            if ( op.name == "variable" )
            {
                auto& label = op.attribs.get("label").string();
                auto& shape = op.attribs.get("shape");
                auto& id = op.outputs.begin()->second.identifier();
                auto& tensor = graph.tensors.at(id);
                
                const std::string filename = path + sep + label + ".dat";
                if ( !read_tensor(filename, tensor, error) )
                {
                    return false;
                }
                
                if ( tensor.dtype != op.dtype )
                {
                    error = "item-type '" + tensor.dtype + "' in variable file '" + filename + "' does not match data-type '" + op.dtype +
                            "' defined in network structure";
                    return false;
                }
                
                Value::items_t items(tensor.shape.size());
                for ( size_t i = 0; i < items.size(); ++i )
                {
                    items[i] = Value::integer(tensor.shape[i]);
                }
                Value tensorShape = Value::array(items);
                
                if ( tensorShape != shape )
                {
                    error = "shape " + tensorShape.toString() + " in variable file '" + filename + "' does not match shape "
                            + shape.toString() + " defined in network structure";
                    return false;
                }
            }
        }
        return true;
    }
    
    bool file_exists( const std::string& path )
    {
        std::ifstream is(path);
        return is.is_open();
    }
    
    bool load_graph( const std::string& path, Graph& graph, std::string& error,
                    const std::string& stdlib, const std::set<std::string>& lowered ) noexcept
    {
        const std::string sep = path.back() == '/' || path.back() == '\\' ? "" : "/";
        const std::string graph_fn = path + sep + "graph.nnef";
        const std::string quant_fn = path + sep + "graph.quant";
        
        if ( !file_exists(graph_fn) )
        {
            return parse_file(path, "", graph, error, stdlib, lowered);
        }
        
        if ( !parse_file(graph_fn, file_exists(quant_fn) ? quant_fn : "", graph, error, stdlib, lowered) )
        {
            return false;
        }
        if ( !load_variables(path, graph, error) )
        {
            return false;
        }
        return true;
    }
    
    
    namespace impl
    {
        
        template<size_t...> struct index_sequence {};
        
        template<std::size_t N, std::size_t... Next>
        struct index_sequence_maker : public index_sequence_maker<N-1U, N-1U, Next...> {};
        
        template<std::size_t... Next>
        struct index_sequence_maker<0U, Next ... > { using type = index_sequence<Next ... >; };
        
        template<std::size_t N>
        using make_index_sequence = typename index_sequence_maker<N>::type;
        
        
        template<typename T, typename... Args>
        struct front_count_of
        {
            enum { value = 0 };
        };
        
        template<typename T, typename... Args>
        struct front_count_of<T,T,Args...>
        {
            enum { value = front_count_of<T,Args...>::value + 1 };
        };
        
        
        const Shape shape_of( const Graph& graph, const Value& value )
        {
            return value.kind() == Value::Identifier ? graph.tensors.at(value.identifier()).shape : nestedArrayShape(value);
        }
        
        Shape& shape_ref( Graph& graph, const Value& value )
        {
            return graph.tensors[value.identifier()].shape;
        }
        
        
        template<typename... Args, size_t... Idxs1, size_t... Idxs2>
        ShapeFunc make_shape_func( Shape(*func)(const Args&...), index_sequence<Idxs1...>, index_sequence<Idxs2...> )
        {
            return [=]( const Operation& op, Graph& graph )
            {
                const Shape shape = func(shape_of(graph, op.inputs[Idxs1].second)..., op.attribs[Idxs2].second...);
                for ( size_t i = 0; i < op.outputs.size(); ++i )
                {
                    shape_ref(graph, op.outputs[i].second) = shape;
                }
            };
        }
        
        template<typename... Args, size_t... Idxs>
        ShapeFunc make_shape_func( std::vector<Shape>(*func)(const Shape&,const Args&...), index_sequence<Idxs...> )
        {
            return [=]( const Operation& op, Graph& graph )
            {
                const std::vector<Shape> shapes = func(shape_of(graph, op.inputs.front().second), op.attribs[Idxs].second...);
                
                const auto& outputs = op.outputs.front().second;
                check(shapes.size() == outputs.size(), "number of shapes (%d) does not match number of outputs (%d)", (int)shapes.size(), (int)outputs.size());
                
                for ( size_t i = 0; i < outputs.size(); ++i )
                {
                    shape_ref(graph, outputs[i]) = shapes[i];
                }
            };
        }
        
        template<typename... Args, size_t... Idxs>
        ShapeFunc make_shape_func( Shape(*func)(const std::vector<Shape>&,const Args&...), index_sequence<Idxs...> )
        {
            return [=]( const Operation& op, Graph& graph )
            {
                const auto& inputs = op.inputs.front().second;
                std::vector<Shape> shapes(inputs.size());
                for ( size_t i = 0; i < shapes.size(); ++i )
                {
                    shapes[i] = shape_of(graph, inputs[i]);
                }
                
                const Shape shape = func(shapes, op.attribs[Idxs].second...);
                for ( size_t i = 0; i < op.outputs.size(); ++i )
                {
                    shape_ref(graph, op.outputs[i].second) = shape;
                }
            };
        }
        
    }   // namespace impl
    
    
    template<typename... Args>
    ShapeFunc make_shape_func( Shape(*func)(const Value&,const Args&...) )
    {
        return impl::make_shape_func(func, impl::make_index_sequence<0>(), impl::make_index_sequence<sizeof...(Args)+1>());
    }
    
    template<typename... Args, size_t N = impl::front_count_of<Shape,Args...>::value>
    ShapeFunc make_shape_func( Shape(*func)(const Shape&,const Args&...) )
    {
        return impl::make_shape_func(func, impl::make_index_sequence<N+1>(), impl::make_index_sequence<sizeof...(Args)-N>());
    }
    
    template<typename... Args>
    ShapeFunc make_shape_func( Shape(*func)(const std::vector<Shape>&,const Args&...) )
    {
        return impl::make_shape_func(func, impl::make_index_sequence<sizeof...(Args)>());
    }
    
    template<typename... Args>
    ShapeFunc make_shape_func( std::vector<Shape>(*func)(const Shape&,const Args&...) )
    {
        return impl::make_shape_func(func, impl::make_index_sequence<sizeof...(Args)>());
    }
    
    
    static const std::map<std::string,ShapeFunc> StandardShapeFuncs =
    {
        { "external", make_shape_func(nullary_shape) },
        { "constant", make_shape_func(constant_shape) },
        { "variable", make_shape_func(nullary_shape) },
        
        { "copy", make_shape_func(unary_shape) },
        { "neg", make_shape_func(unary_shape) },
        { "not", make_shape_func(unary_shape) },
        { "rcp", make_shape_func(unary_shape) },
        { "exp", make_shape_func(unary_shape) },
        { "log", make_shape_func(unary_shape) },
        { "sin", make_shape_func(unary_shape) },
        { "cos", make_shape_func(unary_shape) },
        { "tan", make_shape_func(unary_shape) },
        { "asin", make_shape_func(unary_shape) },
        { "acos", make_shape_func(unary_shape) },
        { "atan", make_shape_func(unary_shape) },
        { "sinh", make_shape_func(unary_shape) },
        { "cosh", make_shape_func(unary_shape) },
        { "tanh", make_shape_func(unary_shape) },
        { "asinh", make_shape_func(unary_shape) },
        { "acosh", make_shape_func(unary_shape) },
        { "atanh", make_shape_func(unary_shape) },
        { "abs", make_shape_func(unary_shape) },
        { "sign", make_shape_func(unary_shape) },
        { "floor", make_shape_func(unary_shape) },
        { "ceil", make_shape_func(unary_shape) },
        { "round", make_shape_func(unary_shape) },
        { "sqr", make_shape_func(unary_shape) },
        { "sqrt", make_shape_func(unary_shape) },
        { "rsqr", make_shape_func(unary_shape) },
        { "rsqrt", make_shape_func(unary_shape) },
        { "log2", make_shape_func(unary_shape) },
        
        { "relu", make_shape_func(unary_shape) },
        { "sigmoid", make_shape_func(unary_shape) },
        { "elu", make_shape_func(unary_shape) },
        { "selu", make_shape_func(unary_shape) },
        { "gelu", make_shape_func(unary_shape) },
        { "silu", make_shape_func(unary_shape) },
        { "softabs", make_shape_func(unary_shape) },
        { "softplus", make_shape_func(unary_shape) },
        { "leaky_relu", make_shape_func(unary_shape) },
        { "prelu", make_shape_func(asymmetric_binary_shape) },
        
        { "linear_quantize", make_shape_func(linear_quantize_shape) },
        { "logarithmic_quantize", make_shape_func(logarithmic_quantize_shape) },
        { "min_max_linear_quantize", make_shape_func(linear_quantize_shape) },
        { "zero_point_linear_quantize", make_shape_func(zero_point_linear_quantize_shape) },
        
        { "add", make_shape_func(binary_shape) },
        { "sub", make_shape_func(binary_shape) },
        { "mul", make_shape_func(binary_shape) },
        { "div", make_shape_func(binary_shape) },
        { "min", make_shape_func(binary_shape) },
        { "max", make_shape_func(binary_shape) },
        { "pow", make_shape_func(binary_shape) },
        { "lt",  make_shape_func(binary_shape) },
        { "le",  make_shape_func(binary_shape) },
        { "gt",  make_shape_func(binary_shape) },
        { "ge",  make_shape_func(binary_shape) },
        { "eq",  make_shape_func(binary_shape) },
        { "ne",  make_shape_func(binary_shape) },
        { "and", make_shape_func(binary_shape) },
        { "or",  make_shape_func(binary_shape) },
        
        { "conv", make_shape_func(conv_shape) },
        { "deconv", make_shape_func(deconv_shape) },
        { "separable_conv", make_shape_func(separable_conv_shape) },
        { "separable_deconv", make_shape_func(separable_deconv_shape) },
        
        { "box", make_shape_func(pool_shape) },
        { "max_pool", make_shape_func(pool_shape) },
        { "argmax_pool", make_shape_func(pool_shape) },
        { "max_pool_with_index", make_shape_func(pool_shape) },
        { "avg_pool", make_shape_func(pool_shape) },
        { "rms_pool", make_shape_func(pool_shape) },
        { "debox", make_shape_func(unpool_shape) },
        { "sample", make_shape_func(sample_shape) },
        { "desample", make_shape_func(desample_shape) },
        
        { "sum_reduce", make_shape_func(reduce_shape) },
        { "min_reduce", make_shape_func(reduce_shape) },
        { "max_reduce", make_shape_func(reduce_shape) },
        { "mean_reduce", make_shape_func(reduce_shape) },
        { "argmax_reduce", make_shape_func(reduce_shape) },
        { "argmin_reduce", make_shape_func(reduce_shape) },
        { "any_reduce", make_shape_func(reduce_shape) },
        { "all_reduce", make_shape_func(reduce_shape) },
        { "moments", make_shape_func(reduce_shape) },
        
        { "nearest_downsample", make_shape_func(downsample_shape) },
        { "area_downsample", make_shape_func(downsample_shape) },
        { "nearest_upsample", make_shape_func(upsample_shape) },
        { "multilinear_upsample", make_shape_func(upsample_shape) },
        
        { "local_response_normalization", make_shape_func(normalize_shape_size) },
        { "local_mean_normalization", make_shape_func(normalize_shape_size) },
        { "local_variance_normalization", make_shape_func(normalize_shape_size) },
        { "local_contrast_normalization", make_shape_func(normalize_shape_size) },
        { "l1_normalization", make_shape_func(normalize_shape_axes) },
        { "l2_normalization", make_shape_func(normalize_shape_axes) },
        { "batch_normalization", make_shape_func(batchnorm_shape) },
        
        { "avg_roi_pool", make_shape_func(roi_shape) },
        { "max_roi_pool", make_shape_func(roi_shape) },
        { "avg_roi_align", make_shape_func(roi_shape) },
        { "max_roi_align", make_shape_func(roi_shape) },
        { "roi_resample", make_shape_func(roi_shape_resample) },
        
        { "reshape", make_shape_func(reshape_shape) },
        { "transpose", make_shape_func(transpose_shape) },
        { "split", make_shape_func(split_shape) },
        { "concat", make_shape_func(concat_shape) },
        { "slice", make_shape_func(slice_shape) },
        { "stack", make_shape_func(stack_shape) },
        { "unstack", make_shape_func(unstack_shape) },
        { "squeeze", make_shape_func(squeeze_shape) },
        { "unsqueeze", make_shape_func(unsqueeze_shape) },
        { "tile", make_shape_func(tile_shape) },
        { "pad", make_shape_func(pad_shape) },
        { "cast", make_shape_func(unary_shape) },
        { "gather", make_shape_func(gather_shape) },
        { "matmul", make_shape_func(matmul_shape) },
        { "linear", make_shape_func(linear_shape) },
        { "update", make_shape_func(update_shape) },
        { "softmax", make_shape_func(softmax_shape) },
        { "copy_n", make_shape_func(copy_n_shape) },
        { "add_n", make_shape_func(add_n_shape) },
        { "select", make_shape_func(ternary_shape) },
        { "clamp", make_shape_func(ternary_shape) },
    };
    
    
    bool infer_shapes( Graph& graph, std::string& error, const std::map<std::string,Shape>& input_shapes,
                      const std::map<std::string,ShapeFunc>& custom_shapes ) noexcept
    {
        for ( auto& op : graph.operations )
        {
            auto it = StandardShapeFuncs.find(op.name);
            if ( it == StandardShapeFuncs.end() )
            {
                it = custom_shapes.find(op.name);
				if ( it == custom_shapes.end() )
				{
					error = "Shape function for operation '" + op.name + "' is not provided";
					return false;
				}
            }
            auto func = it->second;
            
            if ( op.name == "external" )
            {
                auto& id = op.outputs.get("output").identifier();
                auto it = input_shapes.find(id);
                if ( it != input_shapes.end() )
                {
                    auto& original = op.attribs.get("shape");
                    if ( it->second.size() != original.size() )
                    {
                        error = "Overridden external shape rank (" + std::to_string(it->second.size()) +
                                ") does not match original rank (" + std::to_string(original.size()) + ")";
                        return false;
                    }
                    graph.tensors.at(id).shape = it->second;
                    continue;
                }
            }
            
            try
            {
                func(op, graph);
            }
            catch ( const std::exception& e )
            {
                auto& output = op.outputs.front().second;
                auto& id = output.kind() == Value::Identifier ? output.identifier() : output[0].identifier();
                error = "Shape error while inferring shape of tensor '" + id +
                        "' (operation '" + op.name + "'): " + e.what();
                return false;
            }
        }
        return true;
    }
    
    
    bool allocate_buffers( Graph& graph, std::string& error ) noexcept
    {
        for ( auto& item : graph.tensors )
        {
            auto& tensor = item.second;
            tensor.data.resize(volume_of(tensor.shape) * item_bytes(tensor.dtype));
        }
        return true;
    }
    

    bool execute( Graph& graph, std::string& error ) noexcept
    {
        try
        {
            for ( auto& op : graph.operations )
            {
                auto it = rt::Executors.find(op.name);
                if ( it == rt::Executors.end() )
                {
                    throw std::runtime_error("operation not implemented: " + op.name);
                }
                auto& func = it->second;
                func(op, graph.tensors);
            }
            return true;
        }
        catch ( const std::runtime_error& e )
        {
            error = "Runtime error: " + std::string(e.what());
            return false;
        }
    }
    
}   // namespace nnef
