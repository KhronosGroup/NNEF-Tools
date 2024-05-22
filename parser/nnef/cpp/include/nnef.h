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

#ifndef _NNEF_H_
#define _NNEF_H_

#include <map>
#include <set>
#include <string>
#include <iostream>
#include <functional>
#include <algorithm>
#include "nnef/common/value.h"


namespace nnef
{
    
    /*
     * Ordered key-value pairs of arbitrary typed parameter values used for operation attributes
     */
    struct ValueDict : public std::vector<std::pair<std::string,Value>>
    {
        typedef std::pair<std::string,Value> item_type;
        
        bool contains( const std::string& key ) const
        {
            return std::find_if(this->begin(), this->end(), [&]( const item_type& item ){ return item.first == key; }) != this->end();
        }
        
        const Value& get( const std::string& key, const Value& defult = Value::none() ) const
        {
            auto it = std::find_if(this->begin(), this->end(), [&]( const item_type& item ){ return item.first == key; });
            return it != this->end() ? it->second : defult;
        }
        
        const Value& get( const size_t idx ) const
        {
            return std::vector<item_type>::at(idx).second;
        }
    };
    

    /*
     * Tensor data-structure used both for activation and variable tensors
     */
    struct Tensor
    {
        std::string name;           // name of the tensor in the graph
        std::string dtype;          // data-type of the tensor (such as "scalar", "integer", "logical")
        std::vector<int> shape;     // shape of the tensor, filled if shape propagation is in effect
        std::vector<char> data;     // byte array of the data of the tensor, filled in if tensor is a variable
        ValueDict quantization;     // quantization algorithm info for both activation and variable tensors
                                    // used keys: "op-name" (string), attribute names depending on op-name
    };

    
    /*
     * Operation data-structure to represent a single operation in the graph
     */
    struct Operation
    {
        std::string name;           // name (kind) of the operation
        std::string dtype;          // data-type in case the operation is generic (such as "scalar", "integer", "logical")
        ValueDict attribs;          // ordered dictionary of non-tensor attributes of the operation (declaration order)
        ValueDict inputs;           // ordered dictionary of tensor inputs of the operation (may also contain constants)
        ValueDict outputs;          // ordered dictionary tensor outputs of the operation
    };

    
    /*
     * Graph data-structure, list of tensors and operations
     */
    struct Graph
    {
        std::string name;                           // name of the graph
        std::map<std::string,Tensor> tensors;       // list of tensors in the graph
        std::vector<Operation> operations;          // list of operations in the graph, in topograpic order
        std::vector<std::string> inputs;            // list of input tensor ids
        std::vector<std::string> outputs;           // list of output tensor ids
    };


    /*
     * Parse the NNEF graph from file
     *
     * @param graph_fn: name of the graph file
     * @param quant_fn: name of the quantization file
     * @param graph: the graph data structure to fill in
     * @param error: the string to store the error message if any
     * @param stdlib: the implementation of standard operations to use
     * @param lowered: a list of operations to be lowered
     *
     * @return true if there were no parsing errors, false otherwise
     */
    bool parse_file( const std::string& graph_fn, const std::string& quant_fn, Graph& graph, std::string& error,
                    const std::string& stdlib = "", const std::set<std::string>& lowered = {} ) noexcept;
    
    /*
     * Parse the NNEF graph from string
     *
     * @param graph_str: the graph string
     * @param quant_str: the quantization string
     * @param graph: the graph data structure to fill in
     * @param error: the string to store the error message if any
     * @param stdlib: the implementation of standard operations to use
     * @param lowered: a list of operations to be lowered
     *
     * @return true if there were no parsing errors, false otherwise
     */
    bool parse_string( const std::string& graph_str, const std::string& quant_str, Graph& graph, std::string& error,
                      const std::string& stdlib = "", const std::set<std::string>& lowered = {} ) noexcept;
    
    /*
     * Read/write a single tensor from/to binary stream
     *
     * @param is/os: the stream to read from/write to
     * @param tensor: the tensor object to fill into/from
     * @param error: the string to store the error message if any
     *
     * @return true if there were no errors, false otherwise
     */
    bool read_tensor( std::istream& is, Tensor& tensor, std::string& error ) noexcept;
    bool write_tensor( std::ostream& os, const Tensor& tensor, std::string& error ) noexcept;

    /*
     * Read/write a single tensor from/to a binary file
     *
     * @param filename: the name of the file to read from/write to
     * @param tensor: the tensor object to fill into/from
     * @param error: the string to store the error message if any
     *
     * @return true if there were no errors, false otherwise
     */
    bool read_tensor( const std::string& filename, Tensor& tensor, std::string& error ) noexcept;
    bool write_tensor( const std::string& filename, const Tensor& tensor, std::string& error ) noexcept;
    
    /*
     * Load variables/whole model from set of files in a folder
     *
     * @param path: the path to the top level NNEF model folder
     * @param graph: the graph object to load tensors into
     * @param error: the string to store the error message if any
     * @param stdlib: the implementation of standard operations to use
     * @param lowered: a list of operations to be lowered
     *
     * @return true if there were no errors, false otherwise
     */
    bool load_variables( const std::string& path, Graph& graph, std::string& error ) noexcept;
    bool load_graph( const std::string& path, Graph& graph, std::string& error,
                    const std::string& stdlib = "", const std::set<std::string>& lowered = {} ) noexcept;
    
    
    /*
     * Shape propagation function type
     */
    typedef std::function<void( const Operation& op, Graph& graph )> ShapeFunc;
    
    /*
     * Perform shape inference on the graph
     *
     * @param graph: the graph object
     * @param error: the string to store the error message if any
     * @param custom_shapes: shape inference functions for custom operations
     *
     * @return true if there were no errors, false otherwise
     */
    bool infer_shapes( Graph& graph, std::string& error, const std::map<std::string,std::vector<int>>& input_shapes = {},
                      const std::map<std::string,ShapeFunc>& custom_shapes = {} ) noexcept;

    /*
     * Allocate tensor buffers in the graph
     *
     * @param graph: the graph object
     * @param error: the string to store the error message if any
     *
     * @return true if there were no errors, false otherwise
     */
    bool allocate_buffers( Graph& graph, std::string& error ) noexcept;

    /*
     * Execute a graph
     *
     * @param graph: the graph object
     * @param error: the string to store the error message if any
     *
     * @return true if there were no errors, false otherwise
     */
    bool execute( Graph& graph, std::string& error ) noexcept;

}   // namespace nnef


#endif
