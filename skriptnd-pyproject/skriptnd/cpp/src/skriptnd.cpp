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
#include <unordered_set>


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
        
        Typing typing(operators, counted_error);
        
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
            typing.check_module(mod, main_op);
        }
        if ( error_count )
        {
            return std::nullopt;
        }
        
        Composer composer(operators, counted_error, flags);
        auto model = composer(main_name, attribs);
        if ( !model )
        {
            error(model.error().position, model.error().message, model.error().trace, false);
            return std::nullopt;
        }
        
        return error_count ? std::optional<Model>() : std::move(*model);
    }

    inline size_t item_bytes( const Typename dtype )
    {
        return dtype == Typename::Real ? sizeof(float) : dtype == Typename::Int ? sizeof(int) : dtype == Typename::Bool ? sizeof(bool) : 0;
    }

    inline size_t item_bits( const Typename dtype )
    {
        return dtype == Typename::Real ? 32 : dtype == Typename::Int ? 32 : dtype == Typename::Bool ? 1 : 0;
    }

    inline size_t volume_of( const std::vector<ValueExpr>& shape )
    {
        size_t count = 1;
        for ( auto& item : shape )
        {
            count *= item.as_int();
        }
        return count;
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
            for ( size_t i = 0; i < header.rank; ++i )
            {
                tensor.shape[i] = (int_t)header.extents[i];
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

        const size_t count = volume_of(tensor.shape);
        
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
        const size_t count = volume_of(tensor.shape);
        const size_t bits_per_item = quantized ? tensor.data.size() * 8 / count : item_bits(tensor.dtype);
        fill_tensor_header(header, version, tensor.shape.size(), count, bits_per_item, item_type);
        
        for ( size_t i = 0; i < tensor.shape.size(); ++i )
        {
            header.extents[i] = (uint32_t)tensor.shape[i].as_int();
        }
        
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
            for ( auto& item : graph->tensors )
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
                    if ( tensor.shape != variable.shape )
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

    void replace_tensor_accesses( ValueExpr& value, const std::unordered_map<TensorRef,TensorRef>& tensor_remap )
    {
        preorder_traverse(value, [&]( ValueExpr& expr )
        {
            if ( expr.is_shape_access() )
            {
                auto& access = expr.as_shape_access();
                auto it = tensor_remap.find(access.tensor);
                if ( it != tensor_remap.end() )
                {
                    access.tensor = it->second;
                }
            }
            else if ( expr.is_size_access() )
            {
                auto& access = expr.as_size_access();
                auto it = tensor_remap.find(access.pack);
                if ( it != tensor_remap.end() )
                {
                    access.pack = it->second;
                }
            }
        });
    }

    void replace_tensor_usage( std::vector<TensorRef>& tensors, const std::unordered_map<TensorRef,TensorRef>& tensor_remap )
    {
        for ( auto& tensor : tensors )
        {
            auto it = tensor_remap.find(tensor);
            if ( it != tensor_remap.end() )
            {
                tensor = it->second;
            }
        }
    }

    void replace_tensor_usage( Graph& graph, const std::unordered_map<TensorRef,TensorRef>& tensor_remap )
    {
        replace_tensor_usage(graph.inputs, tensor_remap);
        replace_tensor_usage(graph.outputs, tensor_remap);
        for ( auto& op : graph.operations )
        {
            replace_tensor_usage(op.inputs, tensor_remap);
            replace_tensor_usage(op.outputs, tensor_remap);
            for ( auto& [key, value] : op.attribs )
            {
                replace_tensor_accesses(value, tensor_remap);
            }
            for ( auto& expr : op.subexprs )
            {
                replace_tensor_accesses(*expr.as_reference().target, tensor_remap);
            }
            for ( auto& subgraph : op.subgraphs )
            {
                replace_tensor_usage(*subgraph, tensor_remap);
            }
        }
        for ( auto& tensor : graph.tensors )
        {
            for ( auto& expr : tensor->shape )
            {
                replace_tensor_accesses(expr, tensor_remap);
            }
        }
        for ( auto& pack : graph.packs )
        {
            for ( auto& expr : pack->shape )
            {
                replace_tensor_accesses(expr, tensor_remap);
            }
            replace_tensor_accesses(pack->size, tensor_remap);
        }
    }

    bool is_compound( const Operation& op )
    {
        return op.subgraphs.size() == 1 && !op.intrinsic;
    }

    void inline_compounds( Model& model, OperationFilter filter ) noexcept
    {
        std::unordered_set<Graph*> removed_subgraphs;
        
        for ( auto it = model.graphs.rbegin(); it != model.graphs.rend(); ++it )
        {
            auto& graph = **it;
            
            std::vector<Operation> ops;
            std::unordered_map<TensorRef,TensorRef> tensor_remap;
            
            for ( auto& op : graph.operations )
            {
                if ( is_compound(op) && filter(op) )
                {
                    auto& body = *op.subgraphs.front();
                    std::move(body.operations.begin(), body.operations.end(), std::back_inserter(ops));
                    std::move(body.tensors.begin(), body.tensors.end(), std::back_inserter(graph.tensors));
                    std::move(body.packs.begin(), body.packs.end(), std::back_inserter(graph.packs));
                    for ( size_t i = 0; i < op.outputs.size(); ++i )
                    {
                        tensor_remap[body.outputs[i]] = op.outputs[i];
                    }
                    removed_subgraphs.insert(&body);
                }
                else
                {
                    ops.push_back(std::move(op));
                }
            }
            
            graph.operations = std::move(ops);
            
            replace_tensor_usage(graph, tensor_remap);
            
            auto last_tensor = std::remove_if(graph.tensors.begin(), graph.tensors.end(),
                                              [&]( const auto& tensor ){ return tensor_remap.count(tensor.get()) != 0; });
            auto last_pack = std::remove_if(graph.packs.begin(), graph.packs.end(),
                                            [&]( const auto& pack ){ return tensor_remap.count(pack.get()) != 0; });
            graph.tensors.erase(last_tensor, graph.tensors.end());
            graph.packs.erase(last_pack, graph.packs.end());
        }
        
        for ( auto& graph : model.graphs )
        {
            if ( graph->parent && removed_subgraphs.count(graph->parent) )
            {
                graph->parent = graph->parent->parent;
            }
        }
        
        auto last = std::remove_if(model.graphs.begin(), model.graphs.end(),
                                   [&]( const auto& graph ){ return removed_subgraphs.count(graph.get()); });
        model.graphs.erase(last, model.graphs.end());
    }

    void atomize_compounds( Model& model, OperationFilter filter ) noexcept
    {
        std::unordered_set<Graph*> removed_subgraphs;
        
        for ( auto& graph : model.graphs )
        {
            for ( auto& op : graph->operations )
            {
                if ( is_compound(op) && filter(op) )
                {
                    removed_subgraphs.insert(op.subgraphs.front());
                    op.subgraphs.clear();
                }
            }
        }
        
        for ( auto& graph : model.graphs )
        {
            if ( graph->parent && removed_subgraphs.count(graph->parent) )
            {
                removed_subgraphs.insert(graph.get());
            }
        }
        
        auto last = std::remove_if(model.graphs.begin(), model.graphs.end(),
                                   [&]( const auto& graph ){ return removed_subgraphs.count(graph.get()); });
        model.graphs.erase(last, model.graphs.end());
    }

    void collect_references( const ValueExpr& expr, std::vector<ValueExpr>& references )
    {
        recurse(expr, [&]( const ValueExpr& x ){ collect_references(x, references); }, true);
        if ( expr.is_reference() )
        {
            if ( std::find(references.begin(), references.end(), expr) == references.end() )
            {
                references.push_back(expr);
            }
        }
    }

    void collect_references( const TensorRef& tensor, std::vector<ValueExpr>& references )
    {
        for ( auto& s : tensor.shape() )
        {
            collect_references(s, references);
        }
        if ( tensor.packed() )
        {
            collect_references(tensor.size(), references);
        }
    }

    void collect_references( const Contraction& contraction, std::vector<ValueExpr>& references )
    {
        for ( auto& [id, expr] : contraction.locals )
        {
            collect_references(expr, references);
        }
        for ( auto& [id, expr] : contraction.bounds )
        {
            collect_references(expr, references);
        }
        collect_references(contraction.left, references);
        collect_references(contraction.right, references);
    }

    void collect_references( const Assertion& assert, std::vector<ValueExpr>& references )
    {
        collect_references(assert.condition, references);
        for ( auto& expr : assert.args )
        {
            collect_references(expr, references);
        }
    }

    void collect_references( const Operation& op, std::vector<ValueExpr>& references )
    {
        for ( auto& [id, expr] : op.attribs )
        {
            collect_references(expr, references);
        }
        for ( auto& tensor : op.inputs )
        {
            collect_references(tensor, references);
        }
        for ( auto& assert : op.asserts )
        {
            collect_references(assert, references);
        }
        for ( auto& tensor : op.internals )
        {
            collect_references(tensor, references);
        }
        for ( auto& tensor : op.outputs )
        {
            collect_references(tensor, references);
        }
        for ( auto& contraction : op.contractions )
        {
            collect_references(contraction, references);
        }
    }


#define DECL_BOUNDS(var1, var2, expr) auto var1##var2 = (expr); if ( !var1##var2 ) return std::nullopt; auto[var1, var2] = *var1##var2;


    template<typename T>
    std::pair<T,T> duplicate( const T& value )
    {
        return std::make_pair(value, value);
    }
    
    template<typename T = int_t>
    std::optional<std::pair<T,T>> shape_bounds( const ValueExpr& expr, const std::optional<size_t>& idx = std::nullopt )
    {
        if ( expr.packed() && !idx )
        {
            if ( expr.is_list() )
            {
                auto& list = expr.as_list();
                T min = 0, max = 0;
                for ( size_t i = 0; i < list.size(); ++i )
                {
                    DECL_BOUNDS(item_min, item_max, shape_bounds(expr[i]))
                    if ( i == 0 || item_min < min )
                    {
                        min = item_min;
                    }
                    if ( i == 0 || item_max > max )
                    {
                        max = item_max;
                    }
                }
                return std::make_pair(min, max);
            }
            else if ( expr.is_uniform() )
            {
                auto& uniform = expr.as_uniform();
                return shape_bounds(uniform.value);
            }
            else
            {
                assert(false);  // needs to be generalized
                return std::nullopt;
            }
        }
        
        switch ( expr.kind() )
        {
            case ValueExpr::Literal:
            {
                return duplicate(expr.as<T>());
            }
            case ValueExpr::Placeholder:
            {
                auto& placeholder = expr.as_placeholder();
                if ( placeholder.max_value == nullptr )
                {
                    return std::nullopt;
                }
                DECL_BOUNDS(min, max, shape_bounds<T>(placeholder.max_value, idx))
                return std::make_pair((T)0, max);
            }
            case ValueExpr::Reference:
            {
                auto& ref = expr.as_reference();
                return shape_bounds<T>(*ref.target, idx);
            }
            case ValueExpr::SizeAccess:
            {
                auto& access = expr.as_size_access();
                return shape_bounds(access.pack.size());
            }
            case ValueExpr::ShapeAccess:
            {
                auto& access = expr.as_shape_access();
                assert(!idx);
                if ( access.dim.is_literal() )
                {
                    return shape_bounds(access.tensor.shape()[access.dim.as_int()]);
                }
                else
                {
                    T min = 0, max = 0;
                    for ( size_t i = 0; i < access.tensor.rank(); ++i )
                    {
                        DECL_BOUNDS(item_min, item_max, shape_bounds(access.tensor.shape()[i]))
                        if ( i == 0 || item_min < min )
                        {
                            min = item_min;
                        }
                        if ( i == 0 || item_max > max )
                        {
                            max = item_max;
                        }
                    }
                    return std::make_pair(min, max);
                }
            }
            case ValueExpr::Cast:
            {
                auto& cast = expr.as_cast();
                if ( cast.arg.dtype() == cast.dtype )
                {
                    return shape_bounds<T>(cast.arg, idx);
                }
                if ( cast.arg.dtype() == Typename::Bool )
                {
                    return duplicate((T)1);
                }
                if ( cast.arg == inf() )
                {
                    return duplicate(std::numeric_limits<T>::max());
                }
                if ( cast.arg == -inf() )
                {
                    return duplicate(std::numeric_limits<T>::min());
                }
                if ( cast.dtype == Typename::Bool )
                {
                    DECL_BOUNDS(min, max, shape_bounds<T>(cast.arg, idx))
                    return duplicate(min != (T)0 || max != (T)0);
                }
                if ( cast.dtype == Typename::Int )
                {
                    assert(cast.arg.dtype() == Typename::Real);
                    DECL_BOUNDS(min, max, shape_bounds<real_t>(cast.arg, idx))
                    return std::make_pair((int_t)min, (int_t)max);
                }
                else if ( cast.dtype == Typename::Real )
                {
                    assert(cast.arg.dtype() == Typename::Int);
                    DECL_BOUNDS(min, max, shape_bounds<int_t>(cast.arg, idx))
                    return std::make_pair((real_t)min, (real_t)max);
                }
                else
                {
                    assert(false);
                    return duplicate((T)0);
                }
            }
            case ValueExpr::Unary:
            {
                auto& unary = expr.as_unary();
                DECL_BOUNDS(min, max, shape_bounds<T>(unary.arg, idx))
                switch ( Lexer::operator_value(unary.op) )
                {
                    case Lexer::Operator::Plus:
                    {
                        return std::make_pair(min, max);
                    }
                    case Lexer::Operator::Minus:
                    {
                        return std::make_pair(-max, -min);
                    }
                    default:
                    {
                        return std::make_pair((T)Evaluation::eval_unary_func(unary.op, min), (T)Evaluation::eval_unary_func(unary.op, max));
                    }
                }
            }
            case ValueExpr::Binary:
            {
                auto& binary = expr.as_binary();
                DECL_BOUNDS(left_min, left_max, shape_bounds<T>(binary.left, idx))
                DECL_BOUNDS(right_min, right_max, shape_bounds<T>(binary.right, idx))
                switch ( Lexer::operator_value(binary.op) )
                {
                    case Lexer::Operator::Plus:
                    {
                        return std::make_pair(left_min + right_min, left_max + right_max);
                    }
                    case Lexer::Operator::Minus:
                    {
                        return std::make_pair(left_min - right_max, left_max - right_min);
                    }
                    case Lexer::Operator::Multiply:
                    {
                        return std::make_pair(left_min * right_min, left_max * right_max);
                    }
                    case Lexer::Operator::Divide:
                    {
                        return std::make_pair(left_min / right_max, right_min == 0 ? left_max : left_max / right_min);
                    }
                    case Lexer::Operator::CeilDivide:
                    {
                        return std::make_pair(ceil_div(left_min, right_max), right_min == 0 ? left_max : ceil_div(left_max, right_min));
                    }
                    case Lexer::Operator::Modulo:
                    {
                        return std::make_pair((T)0, std::abs(right_max) - (T)1);
                    }
                    case Lexer::Operator::Power:
                    {
                        return std::make_pair((T)pow(left_min, right_min), (T)pow(left_max, right_max));
                    }
                    case Lexer::Operator::Min:
                    {
                        return std::make_pair(std::min(left_min, right_min), std::min(left_max, right_max));
                    }
                    case Lexer::Operator::Max:
                    {
                        return std::make_pair(std::max(left_min, right_min), std::max(left_max, right_max));
                    }
                    default:
                    {
                        assert(false);
                        return duplicate((T)0);
                    }
                }
            }
            case ValueExpr::Select:
            {
                auto& select = expr.as_select();
                DECL_BOUNDS(left_min, left_max, shape_bounds<T>(select.left, idx))
                DECL_BOUNDS(right_min, right_max, shape_bounds<T>(select.right, idx))
                return std::make_pair(std::min(left_min, right_min), std::max(left_max, right_max));
            }
            case ValueExpr::List:
            {
                assert(idx);
                auto& list = expr.as_list();
                return shape_bounds<T>(list[*idx]);
            }
            case ValueExpr::Fold:
            {
                auto& fold = expr.as_fold();
                auto& pack = fold.pack.is_reference() ? *fold.pack.as_reference().target : fold.pack;
                assert(pack.is_list()); // needs to be generalized
                bool dynamic_size = pack.has_dynamic_size();
                auto size = pack.as_list().size();
                switch ( Lexer::operator_value(fold.op) )
                {
                    case Lexer::Operator::Plus:
                    {
                        T sum_min = 0;
                        T sum_max = 0;
                        T max = 0;
                        for ( size_t i = 0; i < size; ++i )
                        {
                            DECL_BOUNDS(item_min, item_max, shape_bounds<T>(pack, i))
                            sum_min += item_min;
                            sum_max += item_max;
                            if ( sum_max > max )
                            {
                                max = sum_max;
                            }
                        }
                        return dynamic_size ? std::make_pair((T)0, max) : std::make_pair(sum_min, sum_max);
                    }
                    case Lexer::Operator::Multiply:
                    {
                        T prod_min = 1;
                        T prod_max = 1;
                        T max = 1;
                        for ( size_t i = 0; i < size; ++i )
                        {
                            DECL_BOUNDS(item_min, item_max, shape_bounds<T>(pack, i))
                            prod_min *= item_min;
                            prod_max *= item_max;
                            if ( prod_max > max )
                            {
                                max = prod_max;
                            }
                        }
                        return dynamic_size ? std::make_pair((T)1, max) : std::make_pair(prod_min, prod_max);
                    }
                    case Lexer::Operator::Min:
                    {
                        DECL_BOUNDS(min, max, shape_bounds<T>(pack.at(0)))
                        for ( size_t i = 1; i < size; ++i )
                        {
                            DECL_BOUNDS(item_min, item_max, shape_bounds<T>(pack, i))
                            min = std::min(min, item_min);
                            if ( !dynamic_size )
                            {
                                max = std::min(max, item_max);
                            }
                        }
                        return std::make_pair(min, max);
                    }
                    case Lexer::Operator::Max:
                    {
                        DECL_BOUNDS(min, max, shape_bounds<T>(pack.at(0)))
                        for ( size_t i = 1; i < size; ++i )
                        {
                            DECL_BOUNDS(item_min, item_max, shape_bounds<T>(pack, i))
                            max = std::max(max, item_max);
                            if ( !dynamic_size )
                            {
                                min = std::max(min, item_min);
                            }
                        }
                        return std::make_pair(min, max);
                    }
                    case Lexer::Operator::MakeEqual:
                    {
                        return shape_bounds<T>(pack.at(0));
                    }
                    default:
                    {
                        assert(false);
                        return duplicate((T)0);
                    }
                }
            }
            case ValueExpr::Subscript:
            {
                auto& subscript = expr.as_subscript();
                auto& pack = subscript.pack.is_reference() ? *subscript.pack.as_reference().target : subscript.pack;
                if ( subscript.index.is_literal() )
                {
                    return shape_bounds<T>(pack.at(subscript.index.as_int()));
                }
                else
                {
                    return shape_bounds<T>(pack);
                }
            }
            case ValueExpr::Concat:
            {
                auto& concat = expr.as_concat();
                DECL_BOUNDS(min, max, shape_bounds<T>(concat.items.front()))
                for ( size_t i = 1; i < concat.items.size(); ++i )
                {
                    DECL_BOUNDS(item_min, item_max, shape_bounds(concat.items[i]))
                    if ( item_min < min )
                    {
                        min = item_min;
                    }
                    if ( item_max > max )
                    {
                        max = item_max;
                    }
                }
                return std::make_pair(min, max);
            }
            case ValueExpr::Uniform:
            {
                auto& uniform = expr.as_uniform();
                return shape_bounds<T>(uniform.value);
            }
            default:
            {
                assert(false);
                return duplicate((T)0);
            }
        }
    }
    
    int_t shape_max( const ValueExpr& expr ) noexcept
    {
        auto bounds = shape_bounds<int_t>(expr);
        return bounds ? bounds->second : -1;
    }
    
    std::vector<int_t> shape_max( const Shape& shape ) noexcept
    {
        std::vector<int_t> max_shape(shape.size());
        for ( size_t i = 0; i < shape.size(); ++i )
        {
            auto& extent = shape[i];
            if ( extent.is_list() )
            {
                const size_t rank = extent.as_list().size();
                
                if ( rank == 0 )
                {
                    max_shape[i] = -1;
                    continue;
                }
                auto bounds = shape_bounds(extent, 0);
                for ( size_t k = 1; k < rank; ++k )
                {
                    auto item_bounds = shape_bounds(extent, k);
                    if ( bounds != item_bounds )
                    {
                        bounds = std::nullopt;
                        break;
                    }
                }
                max_shape[i] = bounds ? bounds->second : -1;
            }
            else if ( extent.is_uniform() )
            {
                auto bounds = shape_bounds(extent.as_uniform().value);
                max_shape[i] = bounds ? bounds->second : -1;
            }
            else
            {
                assert(!extent.packed());
                auto bounds = shape_bounds(extent);
                max_shape[i] = bounds ? bounds->second : -1;
            }
        }
        return max_shape;
    }

#undef DECL_BOUNDS

    
}   // namespace sknd
