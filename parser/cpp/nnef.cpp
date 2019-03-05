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
#include "common/shape.h"
#include "common/binary.h"


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
        
        virtual void endGraph( const Prototype& proto, const Dictionary<Typename>& dtypes, const Dictionary<Shape>& shapes )
        {
            for ( auto& it : shapes )
            {
                Tensor tensor;
                tensor.name = it.first;
                tensor.dtype = toString(dtypes.at(it.first));
                tensor.shape = it.second;
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
        
        virtual void operation( const Prototype& proto, const Dictionary<Value>& args, const Dictionary<Typename>& dtypes,
                               const Dictionary<Shape>& shapes )
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
    
    bool parse_graph( const std::string& filename, const std::string& quantization, Graph& graph, std::string& error,
                     const std::string& customs, const ShapeFuncs& shapeFuncs )
    {
        std::ifstream is(filename);
        if ( !is )
        {
            error = "Could not open graph file: " + std::string(filename);
            return false;
        }
        
        std::ifstream qis(quantization);
        if ( !quantization.empty() )
        {
            qis.open(quantization);
            if ( !qis )
            {
                error = "Could not open quantization file: " + std::string(quantization);
                return false;
            }
        }
        
        ParseCallback callback(graph, qis, quantization);
        CompParser parser(shapeFuncs);
        
        try
        {
            parser.import("custom", customs);
            parser.parse(is, filename.c_str(), callback);
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
    
    bool read_tensor( const std::string& filename, Tensor& tensor, std::string& error, bool validate_shape )
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
            if ( validate_shape )
            {
                validate_tensor_shape(header, tensor.shape);
            }
            else
            {
                tensor.shape.resize(header.rank);
                std::copy_n(header.extents, header.rank, tensor.shape.data());
            }
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
            if ( header.quant_params[1] )
            {
                tensor.compression.emplace_back("scale", Value::scalar(header.quant_params[1]));
            }
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
    
    bool write_tensor( const std::string& filename, const Tensor& tensor, std::string& error )
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
        
        auto item_count = std::accumulate(tensor.shape.begin(), tensor.shape.end(), 1, std::multiplies<Shape::value_type>());
        
        TensorHeader header;
        header.magic[0] = 'N';
        header.magic[1] = 0xEF;
        header.version[0] = 1;
        header.version[1] = 0;
        header.rank = (uint32_t)tensor.shape.size();
        std::copy(tensor.shape.begin(), tensor.shape.end(), header.extents);
        header.quant_code = tensor.compression.at("op-code", Value::integer(0)).integer();
        auto bits_per_item = header.quant_code == TensorHeader::Float || header.quant_code == TensorHeader::Integer ? 32 : 8;
        header.bits_per_item = tensor.compression.at("bits-per-item", Value::integer(bits_per_item)).integer();
        header.data_length = (item_count * header.bits_per_item + 7) / 8;
        
        if ( tensor.data.size() != header.data_length )
        {
            error = "Tensor data length (" + std::to_string(tensor.data.size()) + ") does not match item count (" + std::to_string(item_count) + ")  implied by shape " + toString(tensor.shape);
            return false;
        }
        
        if ( header.quant_code == TensorHeader::Linear || header.quant_code == TensorHeader::Logarithmic )
        {
            if ( !tensor.compression.contains("min") || !tensor.compression.contains("max") )
            {
                error = "Tensor compression dictionary must contain 'min' and 'max' values";
                return false;
            }
            
            header.quant_params[0] = reinterpret_cast<const uint32_t&>(tensor.compression.at("min").scalar());
            header.quant_params[1] = reinterpret_cast<const uint32_t&>(tensor.compression.at("max").scalar());
        }
        else if ( header.quant_code == TensorHeader::Integer )
        {
            header.quant_params[0] = tensor.compression.at("signed", Value::logical(false)).logical() ? 1 : 0;
            header.quant_params[1] = tensor.compression.at("scale", Value::scalar(0.0)).scalar();
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
    
    bool load_variables( const std::string& path, Graph& graph, std::string& error )
    {
        const std::string sep = path.back() == '/' || path.back() == '\\' ? "" : "/";
        
        for ( auto& op : graph.operations )
        {
            if ( op.name == "variable" )
            {
                auto& label = op.attribs.at("label").string();
                auto& id = op.outputs.begin()->second.identifier();
                auto& tensor = graph.tensors.at(id);
                
                const std::string filename = path + sep + label + ".dat";
                if ( !read_tensor(filename, tensor, error, true) )
                {
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
    
    bool load_model( const std::string& path, Graph& graph, std::string& error, const std::string& customs, const ShapeFuncs& shapeFuncs )
    {
        const std::string sep = path.back() == '/' || path.back() == '\\' ? "" : "/";
        const std::string filename = path + sep + "graph.nnef";
        const std::string quantization = path + sep + "graph.quant";
        
        if ( !parse_graph(filename, file_exists(quantization) ? quantization : "", graph, error, customs, shapeFuncs) )
        {
            return false;
        }
        if ( !load_variables(path, graph, error) )
        {
            return false;
        }
        return true;
    }
    
}   // namespace nnef
