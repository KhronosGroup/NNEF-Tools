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

#ifndef _CNNEF_H_
#define _CNNEF_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef    __cplusplus
#if _WIN32
#define EXPORTDLL extern "C" __declspec(dllexport)
#else
#define EXPORTDLL extern "C"
#endif
#else  // __cplusplus
#if _WIN32
#define EXPORTDLL __declspec(dllexport)
#else
#define EXPORTDLL
#endif
#endif // __cplusplus

    
    typedef void* nnef_graph_t;
    typedef void* nnef_tensor_t;

    /*
     * Load NNEF graph from file
     *
     * @param path: the path to the NNEF model folder
     * @param error: the string to store the error message if any
     *
     * @return NNEF graph
     */
    EXPORTDLL nnef_graph_t nnef_graph_load( const char* path, char *error );

    /*
     * Copy an NNEF graph
     *
     * @param graph: NNEF graph
     *
     * @return the copy of NNEF graph
     */
    EXPORTDLL nnef_graph_t nnef_graph_copy( nnef_graph_t graph );
    
    /*
     * Release NNEF graph
     *
     * @param graph: NNEF graph
     */
    EXPORTDLL void nnef_graph_release( nnef_graph_t graph );
    
    /*
     * Perform shape inference on the graph
     *
     * @param graph: the graph object
     * @param error: the string to store the error message if any
     *
     * @return true if there were no errors, false otherwise
     */
    EXPORTDLL int nnef_graph_infer_shapes( nnef_graph_t graph, char *error );
    
    /*
     * Allocate tensor buffers in the graph
     *
     * @param graph: the graph object
     * @param error: the string to store the error message if any
     *
     * @return true if there were no errors, false otherwise
     */
    EXPORTDLL int nnef_graph_allocate_buffers( nnef_graph_t graph, char *error );
    
    /*
     * Execute a graph
     *
     * @param graph: the graph object
     * @param error: the string to store the error message if any
     *
     * @return true if there were no errors, false otherwise
     */
    EXPORTDLL int nnef_graph_execute( nnef_graph_t graph, char *error );
    
    /*
     * Query input names from NNEF graph
     *
     * @param graph: NNEF graph
     * @param inputs: input names
     *
     * @return input count
     */
    EXPORTDLL size_t nnef_graph_input_names( nnef_graph_t graph, const char** inputs );

    /*
     * Query output names from NNEF graph
     *
     * @param graph: NNEF graph
     * @param inputs: output names
     *
     * @return output count
     */
    EXPORTDLL size_t nnef_graph_output_names( nnef_graph_t graph, const char** outputs );

    /*
     * Find tensor in NNEF graph by name
     *
     * @param graph: NNEF graph
     * @param tensor_name: tensor name
     *
     * @return tensor
     */
    EXPORTDLL nnef_tensor_t nnef_graph_find_tensor( nnef_graph_t graph, const char* tensor_name );
    
    /*
     * Query name of an NNEF graph
     *
     * @param graph: NNEF graph
     *
     * @return graph name
     */
    EXPORTDLL const char* nnef_graph_name( nnef_graph_t graph );

    
    
    /*
     * Create a new tensor
     *
     * @return tensor
     */
    EXPORTDLL nnef_tensor_t nnef_tensor_create(void);

    /*
     * Release a tensor
     */
    EXPORTDLL void nnef_tensor_release( nnef_tensor_t tensor );
    
    /*
     * Query tensor name
     *
     * @param tensor: tensor
     *
     * @return tensor name
     */
    EXPORTDLL const char* nnef_tensor_name( nnef_tensor_t tensor );
    
    /*
     * Query tensor data-type
     *
     * @param tensor: tensor
     *
     * @return data-type name
     */
    EXPORTDLL const char* nnef_tensor_dtype( nnef_tensor_t tensor );
    
    /*
     * Query tensor rank
     *
     * @param tensor: tensor
     *
     * @return tensor rank
     */
    EXPORTDLL size_t nnef_tensor_rank( nnef_tensor_t tensor );

    /*
     * Query tensor dims
     *
     * @param tensor: tensor
     *
     * @return tensor rank
     */
    EXPORTDLL const int* nnef_tensor_dims( nnef_tensor_t tensor );
    
    /*
     * Query tensor data
     *
     * @param tensor: tensor
     *
     * @return tensor data
     */
    EXPORTDLL void* nnef_tensor_data( nnef_tensor_t tensor );

    /*
     * Read tensor from binary file
     *
     * @param url: the name of the file to read from
     * @param tensor: tensor
     * @param error: the string to store the error message if any
     *
     * @return true if there were no errors, false otherwise
     */
    EXPORTDLL int nnef_tensor_read( const char* path, nnef_tensor_t tensor, char *error );

    /*
     * Write tensor to binary file
     *
     * @param url: the name of the file to write to
     * @param tensor: tensor
     * @param error: the string to store the error message if any
     *
     * @return true if there were no errors, false otherwise
     */
    EXPORTDLL int nnef_tensor_write( const char* path, nnef_tensor_t tensor, char *error );

#ifdef __cplusplus
}
#endif

#endif
