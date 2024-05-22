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

#include "cnnef.h"
#include "nnef.h"
#include <cstring>


using namespace nnef;


nnef_graph_t nnef_graph_load( const char* path, char *perror )
{
    std::string error;
    Graph *nnef_graph = new Graph();
    bool success = load_graph(path, *nnef_graph, error);
    if ( !success )
    {
        if ( perror != NULL )
        {
            strncpy(perror, error.c_str(), error.length() + 1);
        }

        return NULL;
    }
    return nnef_graph;
}

nnef_graph_t nnef_graph_copy( nnef_graph_t graph )
{
    const Graph *nnef_graph = (const Graph*)graph;
    return new Graph(*nnef_graph);
}

void nnef_graph_release( nnef_graph_t graph )
{
    Graph *nnef_graph = (Graph *)graph;
    if ( nnef_graph )
    {
        delete nnef_graph;
    }
}

int nnef_graph_infer_shapes( nnef_graph_t graph, char *perror )
{
    std::string error;

    Graph* nnef_graph = (Graph*)graph;

    if ( !infer_shapes(*nnef_graph, error) )
    {
        if ( perror != NULL )
        {
            strncpy(perror, error.c_str(), error.length() + 1);
        }
        return 0;
    }
    return 1;
}

int nnef_graph_allocate_buffers( nnef_graph_t graph, char *perror )
{
    std::string error;
    Graph *nnef_graph = (Graph *)graph;

    if ( nnef_graph == NULL )
    {
        return 0;
    }
    if ( !nnef::allocate_buffers(*nnef_graph, error) )
    {
        if ( perror != NULL )
        {
            strncpy(perror, error.c_str(), error.length() + 1);
        }
        return 0;
    }
    return 1;
}

int nnef_graph_execute( nnef_graph_t graph, char *perror )
{
    Graph *nnef_graph = (Graph *)graph;
    if ( nnef_graph == NULL )
    {
        return 0;
    }
    
    std::string error;
    if ( !nnef::execute(*nnef_graph, error) )
    {
        if ( perror != NULL )
        {
            strncpy(perror, error.c_str(), error.length() + 1);
        }
        return 0;
    }
    return 1;
}

size_t nnef_graph_input_names( nnef_graph_t graph, const char** inputs )
{
    const Graph* nnef_graph = (const Graph*)graph;

    if ( inputs != NULL )
    {
        for ( size_t i = 0; i < nnef_graph->inputs.size(); ++i )
        {
            inputs[i] = nnef_graph->inputs[i].c_str();
        }
    }
    return nnef_graph->inputs.size();
}

size_t nnef_graph_output_names( nnef_graph_t graph, const char** outputs )
{
    const Graph* nnef_graph = (const Graph*)graph;

    if ( outputs != NULL )
    {
        for ( size_t i = 0; i < nnef_graph->outputs.size(); ++i )
        {
            outputs[i] = nnef_graph->outputs[i].c_str();
        }
    }
    return nnef_graph->outputs.size();
}

nnef_tensor_t nnef_graph_find_tensor( nnef_graph_t graph, const char* tensor_name )
{
    const Graph *nnef_graph = (const Graph*)graph;
    if ( nnef_graph == NULL )
    {
        return NULL;
    }
    
    std::map<std::string,Tensor>::const_iterator it = nnef_graph->tensors.find(tensor_name);
    return it != nnef_graph->tensors.end() ? (nnef_tensor_t)&it->second : NULL;
}

const char* nnef_graph_name( nnef_graph_t graph )
{
    const Graph *nnef_graph = (const Graph*)graph;
    return nnef_graph->name.c_str();
}



nnef_tensor_t nnef_tensor_create(void)
{
    return new nnef::Tensor();
}

void nnef_tensor_release( nnef_tensor_t tensor )
{
    Tensor* nnef_tensor = (Tensor*)tensor;
    if ( nnef_tensor != NULL )
    {
        delete nnef_tensor;
    }
}

const char* nnef_tensor_name( nnef_tensor_t tensor )
{
    const Tensor *nnef_tensor = (const Tensor*)tensor;
    return nnef_tensor->name.c_str();
}

const char* nnef_tensor_dtype( nnef_tensor_t tensor )
{
    const Tensor *nnef_tensor = (const Tensor*)tensor;
    return nnef_tensor->dtype.c_str();
}

size_t nnef_tensor_rank( nnef_tensor_t tensor )
{
    const Tensor *nnef_tensor = (const Tensor*)tensor;
    return nnef_tensor->shape.size();
}

const int* nnef_tensor_dims( nnef_tensor_t tensor )
{
    const Tensor *nnef_tensor = (const Tensor*)tensor;
    return nnef_tensor->shape.data();
}

void* nnef_tensor_data( nnef_tensor_t tensor )
{
    const Tensor* nnef_tensor = (const Tensor*)tensor;
    return (void*)nnef_tensor->data.data();
}

int nnef_tensor_read( const char* path, nnef_tensor_t tensor, char *perror )
{
    Tensor *nnef_tensor = (Tensor *)tensor;
    if ( nnef_tensor == NULL )
    {
        return 0;
    }
    
    std::string error;
    if ( !read_tensor(path, *nnef_tensor, error) )
    {
        if ( perror != NULL )
        {
            strncpy(perror, error.c_str(), error.length() + 1);
        }
        return 0;
    }
    return 1;
}

int nnef_tensor_write( const char* path, nnef_tensor_t tensor, char *perror )
{
    const Tensor *nnef_tensor = (const Tensor *)tensor;
    if ( nnef_tensor == NULL )
    {
        return 0;
    }
    
    std::string error;
    if ( !write_tensor(path, *nnef_tensor, error) )
    {
        if ( perror != NULL )
        {
            strncpy(perror, error.c_str(), error.length() + 1);
        }
        return 0;
    }
    return 1;
}
