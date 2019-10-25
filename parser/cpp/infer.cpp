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

#include "nnef.h"
#include "nnef/runtime/execution.h"

#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif


const std::set<std::string> lowered =
{
    "separable_conv",
    "separable_deconv",
    "rms_pool",
    "local_response_normalization",
    "local_mean_normalization",
    "local_variance_normalization",
    "local_contrast_normalization",
    "l1_normalization",
    "l2_normalization",
    "batch_normalization",
    "area_downsample",
    "nearest_downsample",
    "nearest_upsample",
    "linear_quantize",
    "logarithmic_quantize",
    "leaky_relu",
    "prelu",
    "clamp",
};

std::string read_file( const char* fn )
{
    std::ifstream is(fn);
    if ( !is )
    {
        throw std::runtime_error("file not found: " + std::string(fn));
    }
    
    return std::string((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
}

bool read_inputs_from_cin( nnef::Graph& graph, std::string& error )
{
    for ( auto& input : graph.inputs )
    {
        auto& tensor = graph.tensors.at(input);
        if ( !nnef::read_tensor(std::cin, tensor, error) )
        {
            return false;
        }
    }
    return true;
}

bool read_inputs_from_file( nnef::Graph& graph, const std::vector<std::string>& inputs, std::string& error )
{
    size_t idx = 0;
    for ( auto& input : graph.inputs )
    {
        auto& tensor = graph.tensors.at(input);
        if ( !nnef::read_tensor(inputs[idx++], tensor, error) )
        {
            return false;
        }
    }
    return true;
}

bool write_output_to_cout( const nnef::Graph& graph, std::string& error )
{
    for ( auto& output : graph.outputs )
    {
        auto& tensor = graph.tensors.at(output);
        if ( !nnef::write_tensor(std::cout, tensor, error) )
        {
            return false;
        }
    }
    return true;
}

bool write_output_to_file( const nnef::Graph& graph, const std::vector<std::string>& outputs, std::string& error )
{
    size_t idx = 0;
    for ( auto& output : graph.outputs )
    {
        auto& tensor = graph.tensors.at(output);
        if ( !nnef::write_tensor(outputs[idx++], tensor, error) )
        {
            return false;
        }
    }
    return true;
}

template<typename T>
T sqr( const T x )
{
    return x * x;
}

template<typename T>
T relative_difference( const size_t n, const T* ref, const T* dat )
{
    T diff = 0;
    T range = 0;
    for ( size_t i = 0; i < n; ++i )
    {
        diff += sqr(ref[i] - dat[i]);
        range += sqr(ref[i]);
    }
    return std::sqrt(diff / range);
}


int main( int argc, const char * argv[] )
{
    if ( argc < 2 )
    {
        std::cerr << "Input file name must be provided" << std::endl;
        return -1;
    }
    
    const std::string path = argv[1];
    std::string stdlib;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    bool compare = false;
    
    for ( size_t i = 2; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if ( arg == "--stdlib" )
        {
            if ( ++i == argc )
            {
                std::cerr << "Stdlib file name must be provided after --stdlib; ignoring option" << std::endl;
            }
            try
            {
                stdlib = read_file(argv[i]);
            }
            catch ( std::runtime_error e )
            {
                std::cerr << e.what() << std::endl;
            }
        }
        else if ( arg == "--input" )
        {
            if ( i + 1 == argc )
            {
                std::cerr << "Input file name(s) must be provided after --input; ignoring option" << std::endl;
            }
            while ( i + 1 < argc && *argv[i+1] != '-' )
            {
                inputs.push_back(argv[++i]);
            }
        }
        else if ( arg == "--output" )
        {
            if ( i + 1 == argc )
            {
                std::cerr << "Output file name(s) must be provided after --output; ignoring option" << std::endl;
            }
            while ( i + 1 < argc && *argv[i+1] != '-' )
            {
                outputs.push_back(argv[++i]);
            }
        }
        else if ( arg == "--compare" )
        {
            compare = true;
        }
        else
        {
            std::cerr << "Unrecognized option: " << argv[i] << "; ignoring" << std::endl;
        }
    }
    
    nnef::Graph graph;
    std::string error;
    
    if ( !nnef::load_graph(path, graph, error, stdlib, lowered) )
    {
        std::cerr << error << std::endl;
        return -1;
    }
    
    if ( !inputs.empty() || !isatty(STDIN_FILENO) )
    {
        bool read = !inputs.empty() ? read_inputs_from_file(graph, inputs, error) : read_inputs_from_cin(graph, error);
        if ( !read )
        {
            std::cerr << error << std::endl;
            return -1;
        }
    }
    
    if ( !nnef::infer_shapes(graph, error) )
    {
        std::cerr << error << std::endl;
        return -1;
    }
    
    std::cerr << "Successfully parsed file: " << path << std::endl;
    
    try
    {
        nnef::rt::Execution exec(graph);
        
        for ( auto& input : graph.inputs )
        {
            auto& tensor = graph.tensors.at(input);
            if ( !tensor.data.empty() )
            {
                exec.push_tensor(input, tensor.dtype, tensor.compression.get("bits-per-item").integer(), tensor.data.data());
            }
        }
        
        std::cout << "Running inference... " << std::flush;
        
        auto start = std::chrono::system_clock::now();
        
        exec();
        
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        std::cout << "done; elapsed time: " << elapsed.count() << "s" << std::endl;
        
        for ( auto& output : graph.outputs )
        {
            auto& tensor = graph.tensors.at(output);
            
            const size_t volume = nnef::volume_of(tensor.shape);
            const size_t bits_per_item = tensor.dtype == "scalar" ? sizeof(float) * 8 :
                                         tensor.dtype == "integer" ? sizeof(int) * 8 :
                                         tensor.dtype == "logical" ? 1 : 0;
            
            tensor.data.resize((volume * bits_per_item + 7) / 8);
            exec.pull_tensor(output, tensor.dtype, bits_per_item, tensor.data.data());
        }
    }
    catch ( std::runtime_error e )
    {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return -1;
    }
    
    if ( compare && !outputs.empty() )
    {
        for ( size_t i = 0; i < graph.outputs.size(); ++i )
        {
            nnef::Tensor tensor;
            nnef::read_tensor(outputs[i], tensor, error);
            
            const nnef::Tensor& output = graph.tensors.at(graph.outputs[i]);
            
            if ( output.shape != tensor.shape )
            {
                std::cout << "shape " << nnef::to_string(output.shape) << " of '" << graph.outputs[i] << "' does not match reference shape " << nnef::to_string(tensor.shape) << std::endl;
            }
            else
            {
                auto diff = relative_difference(nnef::volume_of(tensor.shape), (const float*)tensor.data.data(),
                                                (const float*)output.data.data());
                std::cout << "'" << graph.outputs[i] << "' diff = " << diff << std::endl;
            }
        }
    }
    else if ( !outputs.empty() || !isatty(STDOUT_FILENO) )
    {
        bool write = !outputs.empty() ? write_output_to_file(graph, outputs, error) : write_output_to_cout(graph, error);
        if ( !write )
        {
            std::cerr << error << std::endl;
            return -1;
        }
    }
    
    return 0;
}
