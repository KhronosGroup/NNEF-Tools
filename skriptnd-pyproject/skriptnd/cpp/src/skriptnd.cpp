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


namespace sknd
{

    std::string model_name_from_path( const std::string& path )
    {
        size_t offs = path.back() == '\\' || path.back() == '/';
        auto end = path.length() - offs;
        auto beg = path.find_last_of("\\/", end - 1) + 1;
        return path.substr(beg, end - beg);
    }
    
    std::vector<std::string> enum_graph_names( std::istream& is )
    {
        std::vector<std::string> names;
        
        Lexer lexer(is, "");
        while ( !lexer.empty() )
        {
            bool graph = lexer.is_token(Lexer::Keyword::Graph);
            if ( !lexer.accept() )
            {
                return {};
            }
            if ( graph )
            {
                const std::string id = lexer.token();
                auto result = lexer.accept_if(Lexer::Category::Identifier);
                if ( !result )
                {
                    return {};
                }
                if ( *result )
                {
                    names.push_back(id);
                }
            }
        }
        
        return names;
    }

    OperationCallback make_operation_callback( std::set<std::string>&& names )
    {
        return [=]( const std::string& name, const std::map<std::string,Typename>&,
                   const std::map<std::string,ValueExpr>&, const std::vector<TensorRef>& )
        {
            return (bool)names.count(name);
        };
    }
    
    std::optional<Model> read_model( const std::string& path, const std::string& graph_name, const std::string& stdlib_path,
                                    const ErrorCallback error, const OperationCallback atomic, const OperationCallback unroll,
                                    const std::map<std::string, ValueExpr>& attribs )
    {
        const std::string folder = path.back() != '\\' && path.back() != '/' ? path + "/" : path;
        
        std::ifstream is(folder + "main.sknd");
        if ( !is )
        {
            error(Position(), "Could not open main.sknd in model folder '" + folder + "'", {}, false);
            return std::nullopt;
        }
        
        auto model = read_model(is, "main", graph_name, stdlib_path, folder, error, atomic, unroll, attribs);
        if ( model )
        {
            model->name = model_name_from_path(path);
        }
        return model;
    }
    
    std::optional<Model> read_model( std::istream& is, const std::string& module, const std::string& main_graph,
                                    const std::string& stdlib_path, const std::string& import_path,
                                    const ErrorCallback error, const OperationCallback atomic, const OperationCallback unroll,
                                    const std::map<std::string, ValueExpr>& attribs )
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
        
        Parser parser(stdlib_path, import_path, counted_error);
        auto [operators, graph_names] = parser(is, module);
        if ( error_count )
        {
            return std::nullopt;
        }
        
        auto graph_name = !main_graph.empty() ? main_graph : graph_names.front();
        auto scoped_graph_name = module + "." + graph_name;
        
        Typing typing(counted_error);
        for ( auto& [key, op] : operators )
        {
            typing.check_operator(op, operators, key == scoped_graph_name);
        }
        if ( error_count )
        {
            return std::nullopt;
        }
        
        if ( graph_names.empty() )
        {
            error(Position(), "could not find graph definition", {}, false);
            return std::nullopt;
        }
        
        auto it = operators.find(scoped_graph_name);
        if ( it == operators.end() )
        {
            error(Position{ module }, "could not find definition of graph '" + graph_name + "'", {}, false);
            return std::nullopt;
        }
        
        Composer composer(counted_error, atomic, unroll);
        auto model = composer(operators, scoped_graph_name, attribs);
        if ( !model )
        {
            error(model.error().position, model.error().message, model.error().trace, false);
            return std::nullopt;
        }
        
        return error_count ? std::optional<Model>() : std::move(*model);
    }
    
}   // namespace sknd
