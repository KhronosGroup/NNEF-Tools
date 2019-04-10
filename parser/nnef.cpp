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
#include "nnef.h"
#include "comp/comp_parser.h"
#include "flat/quant_parser.h"
#include "common/binary.h"
#include "common/shapes.h"


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
        catch ( nnef::Error e )
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
        
        std::ifstream quant_is(quant_fn);
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
    
    bool read_tensor( const std::string& filename, Tensor& tensor, std::string& error ) noexcept
    {
        std::ifstream is(filename, std::ios::binary);
        if ( !is )
        {
            error = "Could not open tensor file: " + filename;
            return false;
        }
        
        TensorHeader header;
        is.read((char*)&header, sizeof(header));
        
        try
        {
            validate_tensor_header(header);
            
            tensor.shape.resize(header.rank);
            std::copy_n(header.extents, header.rank, tensor.shape.data());
        }
        catch ( nnef::Error e )
        {
            error = "Failed to read header from tensor file: " + filename + ": " + e.what();
            return false;
        }
        
        tensor.compression.emplace_back("op-code", Value::integer(header.quant_code));
        tensor.compression.emplace_back("bits-per-item", Value::integer(header.bits_per_item));
        
        if ( header.quant_code == TensorHeader::Linear || header.quant_code == TensorHeader::Logarithmic )
        {
            tensor.compression.emplace_back("min", Value::scalar(reinterpret_cast<Value::scalar_t&>(header.quant_params[0])));
            tensor.compression.emplace_back("max", Value::scalar(reinterpret_cast<Value::scalar_t&>(header.quant_params[1])));
        }
        else if ( header.quant_code == TensorHeader::Integer )
        {
            tensor.compression.emplace_back("signed", Value::logical(header.quant_params[0] != 0));
        }
        
        tensor.data.resize(header.data_length);
        is.read((char*)tensor.data.data(), header.data_length);
        
        if ( !is )
        {
            error = "Could not read tensor data from file: " + filename;
            return false;
        }
        
        return true;
    }
    
    bool write_tensor( const std::string& filename, const Tensor& tensor, std::string& error ) noexcept
    {
        std::ofstream os(filename, std::ios::binary);
        if ( !os )
        {
            error = "Could not open tensor file: " + filename;
            return false;
        }
        
        if ( tensor.shape.size() > TensorHeader::MaxRank )
        {
            error = "Tensor rank " + std::to_string(tensor.shape.size()) + " exceeds maximum allowed rank (" + std::to_string(TensorHeader::MaxRank) + ")";
            return false;
        }
        
        auto item_count = std::accumulate(tensor.shape.begin(), tensor.shape.end(), 1, std::multiplies<int>());
        
        TensorHeader header;
        header.magic[0] = 'N';
        header.magic[1] = 0xEF;
        header.version[0] = 1;
        header.version[1] = 0;
        header.rank = (uint32_t)tensor.shape.size();
        std::copy(tensor.shape.begin(), tensor.shape.end(), header.extents);
        header.quant_code = tensor.compression.get("op-code", Value::integer(0)).integer();
        auto bits_per_item = header.quant_code == TensorHeader::Float || header.quant_code == TensorHeader::Integer ? 32 : 8;
        header.bits_per_item = tensor.compression.get("bits-per-item", Value::integer(bits_per_item)).integer();
        header.data_length = (item_count * header.bits_per_item + 7) / 8;
        
        if ( tensor.data.size() != header.data_length )
        {
            error = "Tensor data length (" + std::to_string(tensor.data.size()) + ") does not match item count (" + std::to_string(item_count) + ")  implied by shape";
            return false;
        }
        
        if ( header.quant_code == TensorHeader::Linear || header.quant_code == TensorHeader::Logarithmic )
        {
            if ( !tensor.compression.contains("min") || !tensor.compression.contains("max") )
            {
                error = "Tensor compression dictionary must contain 'min' and 'max' values";
                return false;
            }
            
            header.quant_params[0] = reinterpret_cast<const uint32_t&>(tensor.compression.get("min").scalar());
            header.quant_params[1] = reinterpret_cast<const uint32_t&>(tensor.compression.get("max").scalar());
        }
        else if ( header.quant_code == TensorHeader::Integer )
        {
            header.quant_params[0] = tensor.compression.get("signed", Value::logical(false)).logical() ? 1 : 0;
        }
        
        os.write((char*)&header, sizeof(header));
        os.write((char*)tensor.data.data(), tensor.data.size());
        
        if ( !os )
        {
            error = "Could not write data to file: " + filename;
            return false;
        }
        
        return true;
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
                
                Value::items_t items(tensor.shape.size());
                for ( size_t i = 0; i < items.size(); ++i )
                {
                    items[i] = Value::integer(tensor.shape[i]);
                }
                Value tensorShape = Value::array(items);
                
                if ( tensorShape != shape )
                {
                    error = "shape " + tensorShape.toString() + " in variable file does not match shape "
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
        
        
        const Shape& shape_of( const Graph& graph, const Value& value )
        {
            static const Shape singleton;
            return value.kind() == Value::Identifier ? graph.tensors.at(value.identifier()).shape : singleton;
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
        { "tanh", make_shape_func(unary_shape) },
        { "elu", make_shape_func(unary_shape) },
        { "softabs", make_shape_func(unary_shape) },
        { "softplus", make_shape_func(unary_shape) },
        { "leaky_relu", make_shape_func(unary_shape) },
        { "prelu", make_shape_func(unary_shape) },
        
        { "linear_quantize", make_shape_func(unary_shape) },
        { "logarithmic_quantize", make_shape_func(unary_shape) },
        
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
        { "matmul", make_shape_func(matmul_shape) },
        { "linear", make_shape_func(linear_shape) },
        { "update", make_shape_func(update_shape) },
        { "softmax", make_shape_func(softmax_shape) },
        { "copy_n", make_shape_func(copy_n_shape) },
        { "add_n", make_shape_func(add_n_shape) },
        { "select", make_shape_func(ternary_shape) },
        { "clamp", make_shape_func(ternary_shape) },
    };
    
    
    bool infer_shapes( Graph& graph, std::string& error, const std::map<std::string,ShapeFunc>& custom_shapes ) noexcept
    {
        for ( auto& op : graph.operations )
        {
            auto it = StandardShapeFuncs.find(op.name);
            if ( it == StandardShapeFuncs.end() )
            {
                it = custom_shapes.find(op.name);
            }
            if ( it == custom_shapes.end() )
            {
                error = "Shape function for operation '" + op.name + "' is not provided";
                return false;
            }
            auto func = it->second;
            
            try
            {
                func(op, graph);
            }
            catch ( const std::exception& e )
            {
                error = "Shape error while inferring shape of tensor '" + op.outputs.front().second.identifier() +
                        "' (operation '" + op.name + "'): " + e.what();
                return false;
            }
        }
        return true;
    }
    
}   // namespace nnef
