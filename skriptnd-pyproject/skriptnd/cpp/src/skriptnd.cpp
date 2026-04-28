/*
 * Copyright (c) 2017-2025 The Khronos Group Inc.
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

#include "skriptnd.h"
#include "parser.h"
#include "composer.h"
#include "binary.h"
#include <functional>


namespace sknd
{

    std::string model_name_from_path( const std::string& path )
    {
        size_t offs = path.back() == '\\' || path.back() == '/';
        auto end = path.length() - offs;
        auto beg = path.find_last_of("\\/", end - 1) + 1;
        return path.substr(beg, end - beg);
    }

    std::string module_path_from_name( const std::string& module_name )
    {
        std::string module_path = module_name;
        std::replace(module_path.begin(), module_path.end(), '.', '/');
        module_path += ".sknd";
        return module_path;
    }

    std::unique_ptr<std::istream> try_import_from_path( const std::string& module_name,
                                                        const std::string& import_path )
    {
        const std::string module_path = module_path_from_name(module_name);
        
        std::ifstream is(import_path + module_path);
        if ( is.is_open() )
        {
            return std::make_unique<std::ifstream>(std::move(is));
        }
        return nullptr;
    }

    std::unique_ptr<std::istream> try_import_from_paths( const std::string& module_name,
                                                        const std::vector<std::string>& import_paths )
    {
        const std::string module_path = module_path_from_name(module_name);
        
        std::ifstream is;
        for ( auto& import_path : import_paths )
        {
            is.open(import_path + module_path);
            if ( is.is_open() )
            {
                return std::make_unique<std::ifstream>(std::move(is));
            }
        }
        return nullptr;
    }

    std::optional<Model> read_model( const std::string& path,
                                    const ImportCallback importer,
                                    const ErrorCallback error,
                                    const std::string& entry_point,
                                    const std::map<std::string, sknd::ValueExpr>& attribs,
                                    const unsigned flags ) noexcept
    {
        const std::string folder = path.back() != '\\' && path.back() != '/' ? path + "/" : path;
        
        std::ifstream is(folder + "main.sknd");
        if ( !is )
        {
            error(Position(), "Could not open main.sknd in model folder '" + folder + "'", {}, false);
            return std::nullopt;
        }
        
        auto folder_importer = [&]( const std::string& module_name )
        {
            auto is = try_import_from_path(module_name, folder);
            return is ? std::move(is) : importer(module_name);
        };
        
        auto model = read_model(is, "main", folder_importer, error, entry_point, attribs, flags);
        if ( model )
        {
            model->name = model_name_from_path(path);
            load_variables(path, *model, error);
        }
        return model;
    }
    
    std::optional<Model> read_model( std::istream& is, const std::string& module_name,
                                    const ImportCallback importer,
                                    const ErrorCallback error,
                                    const std::string& entry_point,
                                    const std::map<std::string, sknd::ValueExpr>& attribs,
                                    const unsigned flags ) noexcept
    {
        size_t warning_count = 0;
        size_t error_count = 0;
        auto counted_error = [&]( const Position& position, const std::string& message, const StackTrace& trace, const bool warning )
        {
            error(position, message, trace, warning);
            if ( warning )
            {
                warning_count += 1;
            }
            else
            {
                error_count += 1;
            }
        };
        
        Parser parser(counted_error);
        
        std::vector<Module> modules;
        std::set<std::string> imported;
        
        modules.push_back(parser.parse_module(is, module_name, true));
        
        for ( size_t i = 0; i < modules.size(); ++i )
        {
            for ( size_t j = 0; j < modules[i].imports.size(); ++j )
            {
                auto import = modules[i].imports[j];
                if ( !imported.count(import.name) )
                {
                    auto is = importer(import.name);
                    if ( !is )
                    {
                        error(import.position, Error::format_string("could not import module '%s'", import.name.c_str()), {}, false);
                        continue;
                    }
                    
                    modules.push_back(parser.parse_module(*is, import.name));
                    imported.insert(import.name);
                }
            }
        }
        
        if ( error_count )
        {
            return std::nullopt;
        }
        
        std::map<std::string,const Operator*> operators;
        for ( auto& mod : modules )
        {
            for ( auto& op : mod.operators )
            {
                auto key = mod.name + "." + op.name;
                operators.emplace(key, &op);
            }
        }
        
        Typing typing(counted_error);
        
        auto& main_module = modules.front();
        auto main_op = typing.find_main_op(main_module, entry_point);
        if ( !main_op )
        {
            std::string message = "could not find graph definition";
            if ( !entry_point.empty() )
            {
                message += " '";
                message += entry_point;
                message += "'";
            }
            error(Position{ module_name }, message, {}, false);
            return std::nullopt;
        }
        
        auto main_name = main_module.name + "." + main_op->name;
        
        for ( auto& mod : modules )
        {
            typing.check_module(mod, operators, main_op);
        }
        if ( error_count )
        {
            return std::nullopt;
        }
        
        Composer composer(counted_error, flags);
        auto model = composer(operators, main_name, attribs);
        if ( !model )
        {
            error(model.error().position, model.error().message, model.error().trace, false);
            return std::nullopt;
        }
        
        return error_count ? std::optional<Model>() : std::move(*model);
    }

    void flatten_model( Model& model, const OperationFilter is_atomic ) noexcept
    {
        for ( auto& graph : model.graphs )
        {
            for ( auto it = graph.operations.begin(); it != graph.operations.end(); ++it )
            {
                if ( it->nodes > 1 )
                {
                    if ( is_atomic(*it) )
                    {
                        for ( auto itt = it + 1; itt < it + it->nodes; ++itt )
                        {
                            if ( itt->nodes == 1 )
                            {
                                std::move(itt->contractions.begin(), itt->contractions.end(), std::back_inserter(it->contractions));
                            }
                        }
                        graph.operations.erase(it + 1, it + it->nodes);
                        it->nodes = 1;
                    }
                    else
                    {
                        graph.operations.erase(it--);
                    }
                }
            }
        }
    }

    inline size_t item_bytes( const Typename dtype )
    {
        return dtype == Typename::Real ? sizeof(float) : dtype == Typename::Int ? sizeof(int) : dtype == Typename::Bool ? sizeof(bool) : 0;
    }

    inline size_t item_bits( const Typename dtype )
    {
        return dtype == Typename::Real ? 32 : dtype == Typename::Int ? 32 : dtype == Typename::Bool ? 1 : 0;
    }

    inline size_t volume_of( const std::vector<int_t>& shape )
    {
        return std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
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
            
            tensor.shape.resize(header.rank);
            tensor.canonic_shape.resize(header.rank);
            tensor.max_shape.resize(header.rank);
            for ( size_t i = 0; i < header.rank; ++i )
            {
                tensor.shape[i] = tensor.canonic_shape[i] = tensor.max_shape[i] = (int_t)header.extents[i];
            }
        }
        catch ( const Error& e )
        {
            error = "Invalid tensor header: " + e.message;
            return false;
        }
        
        std::vector<char> bytes(header.data_length);
        is.read(bytes.data(), bytes.size());
        
        if ( !is )
        {
            error = "Failed to read tensor data";
            return false;
        }

        const size_t count = volume_of(tensor.max_shape);
        
        try
        {
            if ( header.item_type == TensorHeader::Float )
            {
                tensor.dtype = Typename::Real;
                tensor.data.resize(count * sizeof(float));
                from_bytes(bytes.data(), count, header.bits_per_item, (float*)tensor.data.data());
            }
            else if ( header.item_type == TensorHeader::Bool )
            {
                tensor.dtype = Typename::Bool;
                tensor.data.resize(count * sizeof(bool));
                from_bytes(bytes.data(), count, header.bits_per_item, (bool*)tensor.data.data());
            }
            else if ( header.item_type == TensorHeader::Int || header.item_type == TensorHeader::Uint )
            {
                tensor.dtype = Typename::Int;
                tensor.data.resize(count * sizeof(int));
                from_bytes(bytes.data(), count, header.bits_per_item, (int*)tensor.data.data(), header.item_type == TensorHeader::Int);
            }
            else if ( header.item_type == TensorHeader::Qint || header.item_type == TensorHeader::Quint )
            {
                tensor.dtype = Typename::Real;
                tensor.data.resize(header.data_length);
                tensor.data = bytes;
                tensor.quant["signed"] = ValueExpr((bool_t)(header.item_type == TensorHeader::Qint));
            }
            else
            {
                error = "Unsupported tensor item-type '" + std::to_string(header.item_type) + "' and bits per item '" + std::to_string(header.bits_per_item) + "'";
                return false;
            }
        }
        catch ( const Error& e )
        {
            error = e.message;
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
        
        const bool quantized = !tensor.quant.empty();
        const bool is_signed = tensor.quant.count("signed") ? tensor.quant.at("signed").as_bool() : true;
        const TensorHeader::ItemType item_type = quantized ? (is_signed ? TensorHeader::Qint : TensorHeader::Quint) :
                                                 tensor.dtype == Typename::Real ? TensorHeader::Float :
                                                 tensor.dtype == Typename::Int ? TensorHeader::Int : TensorHeader::Bool;
        
        TensorHeader header;
        const size_t version[] = { 1, 0 };
        const size_t count = volume_of(tensor.max_shape);
        const size_t bits_per_item = quantized ? tensor.data.size() * 8 / count : item_bits(tensor.dtype);
        fill_tensor_header(header, version, tensor.max_shape.size(), tensor.max_shape.data(), bits_per_item, item_type);
        
        std::vector<char> bytes(header.data_length);
        
        if ( tensor.dtype == Typename::Real )
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
        else if ( tensor.dtype == Typename::Int )
        {
            to_bytes((const int*)tensor.data.data(), count, bytes.data(), true);
        }
        else if ( tensor.dtype == Typename::Bool )
        {
            to_bytes((const bool*)tensor.data.data(), count, bytes.data());
        }
        else
        {
            error = "Invalid tensor data-type: '" + str(tensor.dtype) + "'";
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

    bool load_variables( const std::string& path, Model& model, const ErrorCallback error ) noexcept
    {
        const std::string folder = path.back() != '\\' && path.back() != '/' ? path + "/" : path;
        
        bool success = true;
        for ( auto& graph : model.graphs )
        {
            for ( auto& item : graph.tensors )
            {
                if ( item->variable )
                {
                    auto& variable = *item;
                    const std::string filename = folder + variable.name + ".dat";
                    
                    Tensor tensor;
                    std::string message;
                    if ( !read_tensor(filename, tensor, message) )
                    {
                        error(Position{ filename }, message.c_str(), {}, false);
                        success = false;
                        continue;
                    }
                    if ( tensor.dtype != variable.dtype )
                    {
                        message = "item-type '" + str(tensor.dtype) + "' in variable file '" + filename + "' does not match data-type '" + str(variable.dtype) + "' defined in model structure";
                        error(Position{ filename }, message.c_str(), {}, false);
                        success = false;
                    }
                    if ( tensor.max_shape != variable.max_shape )
                    {
                        message = "shape " + str(tensor.shape) + " in variable file '" + filename + "' does not match shape "
                                + str(variable.shape) + " defined in model structure";
                        error(Position{ filename }, message.c_str(), {}, false);
                        success = false;
                    }
                    
                    if ( message.empty() )
                    {
                        variable.data.swap(tensor.data);
                    }
                }
            }
        }
        return success;
    }
    
}   // namespace sknd
