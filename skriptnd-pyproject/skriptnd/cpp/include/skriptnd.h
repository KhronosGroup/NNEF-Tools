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

#ifndef _SKRIPTND_H_
#define _SKRIPTND_H_

#include <set>
#include <optional>
#include "model.h"
#include "error.h"


namespace sknd
{
    
    std::string model_name_from_path( const std::string& path );
    std::vector<std::string> enum_graph_names( std::istream& is );

    OperationCallback make_operation_callback( const std::set<std::string>& names );
    OperationCallback make_operation_callback( std::set<std::string>&& names );
    
    std::optional<Model> read_model( const std::string& path, const std::string& graph_name, const std::string& stdlib_path,
                                    const ErrorCallback error, const OperationCallback atomic = nullptr, const OperationCallback unroll = nullptr,
                                    const std::map<std::string, sknd::ValueExpr>& attribs = {} ) noexcept;
    std::optional<Model> read_model( std::istream& is, const std::string& module, const std::string& graph_name,
                                    const std::string& stdlib_path, const std::string& import_path,
                                    const ErrorCallback error, const OperationCallback atomic = nullptr, const OperationCallback unroll = nullptr,
                                    const std::map<std::string, sknd::ValueExpr>& attribs = {} ) noexcept;

    bool read_tensor( std::istream& is, Tensor& tensor, std::string& error ) noexcept;
    bool write_tensor( std::ostream& os, const Tensor& tensor, std::string& error ) noexcept;

    bool read_tensor( const std::string& filename, Tensor& tensor, std::string& error ) noexcept;
    bool write_tensor( const std::string& filename, const Tensor& tensor, std::string& error ) noexcept;

    bool load_variables( const std::string& path, Model& model, const ErrorCallback error ) noexcept;
    
}   // namespace sknd


#endif
