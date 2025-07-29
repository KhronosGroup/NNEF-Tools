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

#ifndef _NNEF_RUNTIME_OPERATIONS_H_
#define _NNEF_RUNTIME_OPERATIONS_H_

#include <cmath>
#include <algorithm>
#include "ndrange.h"


namespace nnef { namespace rt
{
    
    template<typename T, typename Op>
    void _unary( const size_t n, const T* x, const size_t dx, T* y, const size_t dy, const Op& op )
    {
        for ( size_t i = 0; i < n; ++ i, x += dx, y += dy )
        {
            *y = op(*x);
        }
    }

    template<typename T, typename Op>
    void unary( tensor_view<const T> x, tensor_view<T> y, const Op& op )
    {
        _unary(y.volume, x.data, 1, y.data, 1, op);
    }


    template<typename T, typename R, typename Op>
    inline void _binary( const size_t n, const T* x, const size_t dx, const T* y, const size_t dy, R* z, const size_t dz, const Op& op )
    {
        for ( size_t i = 0; i < n; ++i, x += dx, y += dy, z += dz )
        {
            *z = op(*x, *y);
        }
    }

    template<typename T, typename R, typename Op>
    inline void binary( tensor_view<const T> x, tensor_view<const T> y, tensor_view<R> z, const Op& op )
    {
        if ( (x.volume == z.volume || x.volume == 1) && (y.volume == z.volume || y.volume == 1) )
        {
            _binary(z.volume, x.data, x.volume == z.volume, y.data, y.volume == z.volume, z.data, 1, op);
        }
        else
        {
            const size_t dx = *x.shape != 1;
            const size_t dy = *y.shape != 1;
            
            for ( size_t xi = 0, yi = 0, zi = 0; zi < *z.shape; ++zi, xi += dx, yi += dy )
            {
                binary(x[xi], y[yi], z[zi], op);
            }
        }
    }

    
    template<typename T>
    void _select( const size_t n, const bool* c, const size_t dc, const T* x, const size_t dx, const T* y, const size_t dy, T* z, const size_t dz )
    {
        for ( size_t i = 0; i < n; ++i, c += dc, x += dx, y += dy, z += dz )
        {
            *z = *c ? *x : *y;
        }
    }
    
    template<typename T>
    void select( tensor_view<const bool> c, tensor_view<const T> x, tensor_view<const T> y, tensor_view<T> z )
    {
        if ( (c.volume == z.volume || c.volume == 1) && (x.volume == z.volume || x.volume == 1) && (y.volume == z.volume || y.volume == 1) )
        {
            _select(z.volume, c.data, c.volume == z.volume, x.data, x.volume == z.volume, y.data, y.volume == z.volume, z.data, 1);
        }
        else
        {
            const size_t dc = *c.shape != 1;
            const size_t dx = *x.shape != 1;
            const size_t dy = *y.shape != 1;
            
            for ( size_t ci = 0, xi = 0, yi = 0, zi = 0; zi < *z.shape; ++zi, ci += dc, xi += dx, yi += dy )
            {
                select(c[ci], x[xi], y[yi], z[zi]);
            }
        }
    }
    

    template<typename T, typename Op>
    void _reduce( const size_t n, const T* x, const size_t dx, T* y, const size_t dy, const Op& op )
    {
        for ( size_t i = 0; i < n; ++i, x += dx, y += dy )
        {
            *y = op(*x, *y);
        }
    }

    template<typename T, typename Op>
    void _reduce( tensor_view<const T> x, tensor_view<T> y, const Op& op )
    {
        if ( y.volume == x.volume || y.volume == 1 )
        {
            _reduce(x.volume, x.data, 1, y.data, y.volume == x.volume, op);
        }
        else
        {
            const size_t dy = *y.shape != 1;
            
            for ( size_t xi = 0, yi = 0; xi < *x.shape; ++xi, yi += dy )
            {
                _reduce(x[xi], y[yi], op);
            }
        }
    }

    template<typename T, typename Op>
    void reduce( tensor_view<const T> x, tensor_view<T> y, const Op& op, const T init )
    {
        std::fill_n(y.data, y.volume, init);
        _reduce(x, y, op);
    }


    template<typename T>
    void _bias( tensor_view<const T> bias, tensor_view<T> tensor )
    {
        if ( bias.volume == 1 )
        {
            std::fill_n(tensor.data, tensor.volume, *bias.data);
        }
        else
        {
            T* data = tensor.data;
            const size_t size = nd_volume(tensor.rank - 2, tensor.shape + 2);
            for ( size_t b = 0; b < tensor.shape[0]; ++b )
            {
                for ( size_t c = 0; c < tensor.shape[1]; ++c, data += size )
                {
                    std::fill_n(data, size, bias.data[c]);
                }
            }
        }
    }

    template<bool Transposed, size_t D, typename T>
    static void _conv_core( tensor_view<const T> filter, tensor_view<T> input, tensor_view<T> output,
                           const int padding[], const int stride[], const int dilation[] )
    {
        int input_index[D];
        nd_loop<D,int>(output.shape, [&]( const int output_index[] )
        {
            nd_loop<D,int>(filter.shape, [&]( const int filter_index[] )
            {
                for_n<D>([&]( const size_t k )
                {
                    input_index[k] = output_index[k] * stride[k] + filter_index[k] * dilation[k] - padding[k];
                });
                
                if ( all_n<D>([&]( const size_t k ){ return input_index[k] >= 0 && input_index[k] < input.shape[k]; }) )
                {
                    if ( Transposed )
                    {
                        at<D>(input, input_index) += at<D>(output, output_index) * at<D>(filter, filter_index);
                    }
                    else
                    {
                        at<D>(output, output_index) += at<D>(input, input_index) * at<D>(filter, filter_index);
                    }
                }
            });
        });
    }

    template<bool Transposed, size_t D, typename T>
    void _conv( tensor_view<const T> filter, tensor_view<const T> bias, tensor_view<T> input, tensor_view<T> output,
               const int padding[], const int stride[], const int dilation[] )
    {
        _bias(bias, Transposed ? input : output);
        
        for ( size_t b = 0; b < output.shape[0]; ++b )
        {
            for ( size_t z = 0; z < output.shape[1]; ++z )
            {
                for ( size_t c = 0; c < input.shape[1]; ++c )
                {
                    _conv_core<Transposed,D>(filter[z][c], input[b][c], output[b][z], padding, stride, dilation);
                }
            }
        }
    }

    template<bool Transposed, typename T>
    void conv( tensor_view<const T> filter, tensor_view<const T> bias, tensor_view<T> input, tensor_view<T> output,
               const int padding[], const int stride[], const int dilation[] )
    {
        static decltype(&_conv<Transposed,1,T>) funcs[] =
        {
            _conv<Transposed,1,T>,
            _conv<Transposed,2,T>,
            _conv<Transposed,3,T>,
        };
        funcs[input.rank - 3](filter, bias, input, output, padding, stride, dilation);
    }

    template<bool Transposed, size_t D, typename T>
    void _depthwise_conv( tensor_view<const T> filter, tensor_view<const T> bias, tensor_view<T> input, tensor_view<T> output,
                         const int padding[], const int stride[], const int dilation[] )
    {
        const size_t multiplier = output.shape[1] / input.shape[1];
        const bool broadcast = filter.shape[0] == 1;
        
        _bias(bias, Transposed ? input : output);
        
        for ( size_t b = 0; b < input.shape[0]; ++b )
        {
            for ( size_t c = 0; c < input.shape[1]; ++c )
            {
                for ( size_t m = 0; m < multiplier; ++m )
                {
                    const size_t z = multiplier * c + m;
                    _conv_core<Transposed,D>(filter[broadcast ? 0 : z][0], input[b][c], output[b][z], padding, stride, dilation);
                }
            }
        }
    }
    
    template<bool Transposed, typename T>
    void depthwise_conv( tensor_view<const T> filter, tensor_view<const T> bias, tensor_view<T> input, tensor_view<T> output,
                        const int padding[], const int stride[], const int dilation[] )
    {
        static decltype(&_depthwise_conv<Transposed,1,T>) funcs[] =
        {
            _depthwise_conv<Transposed,1,T>,
            _depthwise_conv<Transposed,2,T>,
            _depthwise_conv<Transposed,3,T>,
        };
        funcs[input.rank - 3](filter, bias, input, output, padding, stride, dilation);
    }

    template<bool Transposed, size_t D, typename T>
    void _grouped_conv( tensor_view<const T> filter, tensor_view<const T> bias, tensor_view<T> input, tensor_view<T> output,
                       const int padding[], const int stride[], const int dilation[], const size_t groups )
    {
        _bias(bias, Transposed ? input : output);
        
        const size_t input_block = input.shape[1] / groups;
        const size_t output_block = output.shape[1] / groups;
        
        for ( size_t b = 0; b < input.shape[0]; ++b )
        {
            for ( size_t g = 0; g < groups; ++g )
            {
                for ( size_t z = 0; z < output_block; ++z )
                {
                    for ( size_t c = 0; c < input_block; ++c )
                    {
                        _conv_core<Transposed,D>(filter[g * output_block + z][c],
                                                 input[b][g * input_block + c],
                                                 output[b][g * output_block + z],
                                                 padding, stride, dilation);
                    }
                }
            }
        }
    }

    template<bool Transposed, typename T>
    void grouped_conv( tensor_view<const T> filter, tensor_view<const T> bias, tensor_view<T> input, tensor_view<T> output,
                      const int padding[], const int stride[], const int dilation[], const size_t groups )
    {
        static decltype(&_grouped_conv<Transposed,1,T>) funcs[] =
        {
            _grouped_conv<Transposed,1,T>,
            _grouped_conv<Transposed,2,T>,
            _grouped_conv<Transposed,3,T>,
        };
        funcs[input.rank - 3](filter, bias, input, output, padding, stride, dilation, groups);
    }


    template<bool Transposed, size_t D, typename T, typename Op>
    static void _pool_core( tensor_view<T> input, tensor_view<T> output, const int size[], const int padding[],
                           const int stride[], const int dilation[], const Op& op, const bool include_border )
    {
        int input_index[D];
        nd_loop<D,int>(output.shape, [&]( const int output_index[] )
        {
            nd_loop<D,int>(size, [&]( const int kernel_index[] )
            {
                for_n<D>([&]( const size_t k )
                {
                    input_index[k] = output_index[k] * stride[k] + kernel_index[k] * dilation[k] - padding[k];
                });
                
                const bool valid = all_n<D>([&]( const size_t k ){ return input_index[k] >= 0 && input_index[k] < input.shape[k]; });
                
                T& value = Transposed ? at<D>(input, input_index) : at<D>(output, output_index);
                
                if ( valid )
                {
                    value = op(value, Transposed ? at<D>(output, output_index) : at<D>(input, input_index));
                }
                else if ( include_border && !Transposed )
                {
                    value = op(value, (T)0);
                }
            });
        });
    }

    template<bool Transposed, size_t D, typename T, typename Op>
    void _pool( tensor_view<T> input, tensor_view<T> output, const int size[], const int padding[],
               const int stride[], const int dilation[], const Op& op, const T init, const bool include_border )
    {
        std::fill_n(Transposed ? input.data : output.data, Transposed ? input.volume : output.volume, init);
        
        _pool_core<Transposed,D>(input, output, size, padding, stride, dilation, op, include_border);
    }

    template<bool Transposed, typename T, typename Op>
    void pool( tensor_view<T> input, tensor_view<T> output, const int size[], const int padding[],
              const int stride[], const int dilation[], const Op& op, const T init, const bool include_border )
    {
        static decltype(&_pool<Transposed,1,T,Op>) funcs[] =
        {
            _pool<Transposed,1,T,Op>,
            _pool<Transposed,2,T,Op>,
            _pool<Transposed,3,T,Op>,
            _pool<Transposed,4,T,Op>,
            _pool<Transposed,5,T,Op>,
        };
        funcs[input.rank - 1](input, output, size, padding, stride, dilation, op, init, include_border);
    }

    template<bool Transposed, size_t D, typename T>
    static void _pool_area( tensor_view<T> tensor, const int input_shape[], const int output_shape[],
                           const int size[], const int padding[], const int stride[], const int dilation[] )
    {
        std::fill_n(tensor.data, nd_volume<D>(tensor.shape), (T)0);
        
        int input_index[D];
        nd_loop<D,int>(output_shape, [&]( const int output_index[] )
        {
            nd_loop<D,int>(size, [&]( const int kernel_index[] )
            {
                for_n<D>([&]( const size_t k )
                {
                    input_index[k] = output_index[k] * stride[k] + kernel_index[k] * dilation[k] - padding[k];
                });
                
                if ( all_n<D>([&]( const size_t k ){ return input_index[k] >= 0 && input_index[k] < input_shape[k]; }) )
                {
                    ++at<D>(tensor, Transposed ? input_index : output_index);
                }
            });
        });
    }

    template<bool Transposed, typename T>
    static void pool_area( tensor_view<T> tensor, const int input_shape[], const int output_shape[],
                          const int size[], const int padding[], const int stride[], const int dilation[] )
    {
        static decltype(&_pool_area<Transposed,1,T>) funcs[] =
        {
            _pool_area<Transposed,1,T>,
            _pool_area<Transposed,2,T>,
            _pool_area<Transposed,3,T>,
            _pool_area<Transposed,4,T>,
            _pool_area<Transposed,5,T>,
        };
        funcs[tensor.rank - 1](tensor, input_shape, output_shape, size, padding, stride, dilation);
    }


    template<bool trA, bool trB, typename T>
    void _matmul( const size_t m, const size_t n, const size_t k, const T* A, const T* B, T* C )
    {
        for ( size_t i = 0; i < m; ++i )
        {
            for ( size_t j = 0; j < n; ++j, ++C )
            {
                for ( size_t l = 0; l < k; ++l )
                {
                    *C += A[trA ? l * m + i : i * k + l] * B[trB ? j * k + l : l * n + j];
                }
            }
        }
    }

    template<typename T>
    void matmul( const bool trA, const bool trB, tensor_view<const T> A, tensor_view<const T> B, tensor_view<T> C )
    {
        std::fill_n(C.data, nd_volume(C.rank, C.shape), 0.f);
        
        const size_t offset = C.rank - 2;
        const size_t dA = nd_volume<2>(A.shape + offset);
        const size_t dB = nd_volume<2>(B.shape + offset);
        const size_t dC = nd_volume<2>(C.shape + offset);
        const size_t m = C.shape[offset];
        const size_t n = C.shape[offset + 1];
        const size_t k = trA ? A.shape[offset] : A.shape[offset + 1];
        
        const size_t b = nd_volume(offset, C.shape);
        for ( size_t i = 0; i < b; ++i, A.data += dA, B.data += dB, C.data += dC )
        {
            if ( trA && trB )
            {
                _matmul<true,true>(m, n, k, A.data, B.data, C.data);
            }
            else if ( trA )
            {
                _matmul<true,false>(m, n, k, A.data, B.data, C.data);
            }
            else if ( trB )
            {
                _matmul<false,true>(m, n, k, A.data, B.data, C.data);
            }
            else
            {
                _matmul<false,false>(m, n, k, A.data, B.data, C.data);
            }
        }
    }

    template<typename T>
    void linear( tensor_view<const T> filter, tensor_view<const T> bias, tensor_view<const T> input, tensor_view<T> output )
    {
        const size_t m = output.shape[0];
        const size_t n = output.shape[1];
        const size_t k = input.shape[1];
        
        if ( bias.volume == 1 )
        {
            std::fill_n(output.data, m * n, *bias.data);
        }
        else
        {
            T* data = output.data;
            for ( size_t i = 0; i < m; ++i, data += n )
            {
                std::copy_n(bias.data, n, data);
            }
        }
        
        _matmul<false,true>(m, n, k, input.data, filter.data, output.data);
    }
    

    template<typename T>
    void _linear_upsample2x_symmetric( tensor_view<T> input, tensor_view<T> output )
    {
        const T zero = 0;
        const T weights[] =
        {
            0.25, 0.75, 0.75, 0.25
        };
        const int shape[] = { 1, 1, 4 };
        tensor_view<const T> filter = { 3, 4, shape, weights };
        tensor_view<const T> bias = { 0, 1, nullptr, &zero };
        const int padding[] = { 1 };
        const int stride[] = { 2 };
        const int dilation[] = { 1 };
        return _depthwise_conv<true,1>(filter, bias, output, input, padding, stride, dilation);
    }

    template<typename T>
    void _linear_upsample2x_asymmetric( tensor_view<T> input, tensor_view<T> output )
    {
        const T zero = 0;
        const T weights[] =
        {
            0.5, 1.0, 0.5,
        };
        const int shape[] = { 1, 1, 3 };
        tensor_view<const T> filter = { 3, 3, shape, weights };
        tensor_view<const T> bias = { 0, 1, nullptr, &zero };
        const int padding[] = { 1 };
        const int stride[] = { 2 };
        const int dilation[] = { 1 };
        return _depthwise_conv<true,1>(filter, bias, output, input, padding, stride, dilation);
    }

    template<typename T>
    void _bilinear_upsample2x_symmetric( tensor_view<T> input, tensor_view<T> output )
    {
        const T zero = 0;
        const T weights[] =
        {
            0.0625, 0.1875, 0.1875, 0.0625,
            0.1875, 0.5625, 0.5625, 0.1875,
            0.1875, 0.5625, 0.5625, 0.1875,
            0.0625, 0.1875, 0.1875, 0.0625,
        };
        const int shape[] = { 1, 1, 4, 4 };
        tensor_view<const T> filter = { 4, 16, shape, weights };
        tensor_view<const T> bias = { 0, 1, nullptr, &zero };
        const int padding[] = { 1, 1 };
        const int stride[] = { 2, 2 };
        const int dilation[] = { 1, 1 };
        return _depthwise_conv<true,2>(filter, bias, output, input, padding, stride, dilation);
    }

    template<typename T>
    void _bilinear_upsample2x_asymmetric( tensor_view<T> input, tensor_view<T> output )
    {
        const T zero = 0;
        const T weights[] =
        {
            0.25, 0.5, 0.25,
            0.50, 1.0, 0.50,
            0.25, 0.5, 0.25,
        };
        const int shape[] = { 1, 1, 3, 3 };
        tensor_view<const T> filter = { 4, 9, shape, weights };
        tensor_view<const T> bias = { 0, 1, nullptr, &zero };
        const int padding[] = { 1, 1 };
        const int stride[] = { 2, 2 };
        const int dilation[] = { 1, 1 };
        return _depthwise_conv<true,2>(filter, bias, output, input, padding, stride, dilation);
    }

    template<typename T>
    void multilinear_upsample2x_symmetric( tensor_view<T> input, tensor_view<T> output )
    {
        static decltype(&_linear_upsample2x_symmetric<T>) funcs[] =
        {
            _linear_upsample2x_symmetric<T>,
            _bilinear_upsample2x_symmetric<T>,
        };
        return funcs[input.rank - 3](input, output);
    }

    template<typename T>
    void multilinear_upsample2x_asymmetric( tensor_view<T> input, tensor_view<T> output )
    {
        static decltype(&_linear_upsample2x_asymmetric<T>) funcs[] =
        {
            _linear_upsample2x_asymmetric<T>,
            _bilinear_upsample2x_asymmetric<T>,
        };
        return funcs[input.rank - 3](input, output);
    }


    template<typename T, typename Op>
    T _reduce( const size_t n, const T* x, const size_t dx, const Op& op )
    {
        T r = *x;
        x += dx;
        for ( size_t i = 1; i < n; ++i, x += dx )
        {
            r = op(r, *x);
        }
        return r;
    }

    template<typename T>
    void _softmax( const size_t n, const size_t m, const T* x, T* y )
    {
        const T max = _reduce(n, x, m, []( const T x, const T y ){ return std::max(x,y); });
        _unary(n, x, m, y, m, [&]( const T x ){ return std::exp(x - max); });
        const T sum = _reduce(n, y, m, std::plus<T>());
        _binary(n, y, m, &sum, 0, y, m, std::divides<T>());
    }

    template<typename T>
    void softmax( tensor_view<const T> input, tensor_view<T> output, const size_t axis )
    {
        const size_t batch = nd_volume(axis, input.shape);
        const size_t channels = input.shape[axis];
        const size_t size = nd_volume(input.rank - axis - 1, input.shape + axis + 1);
        const size_t volume = channels * size;
        
        for ( size_t i = 0; i < batch; ++i, input.data += volume, output.data += volume )
        {
            for ( size_t j = 0; j < size; ++j )
            {
                _softmax(channels, size, input.data + j, output.data + j);
            }
        }
    }

    template<typename I, typename T, typename Op>
    I _arg_reduce( const size_t n, const T* x, const size_t dx, const Op& op )
    {
        I idx = 0;
        T val = *x;
        x += dx;
        for ( size_t i = 1; i < n; ++i, x += dx )
        {
            if ( op(*x, val) )
            {
                val = *x;
                idx = (I)i;
            }
        }
        return idx;
    }

    template<typename T, typename I, typename Op>
    void arg_reduce( tensor_view<const T> input, tensor_view<I> output, const size_t axis, const Op& op )
    {
        const size_t batch = nd_volume(axis, input.shape);
        const size_t channels = input.shape[axis];
        const size_t size = nd_volume(input.rank - axis - 1, input.shape + axis + 1);
        const size_t volume = channels * size;
        
        for ( size_t i = 0; i < batch; ++i, input.data += volume, output.data += size )
        {
            for ( size_t j = 0; j < size; ++j )
            {
                output.data[j] = _arg_reduce<I>(channels, input.data + j, size, op);
            }
        }
    }


    template<size_t D, typename T>
    void _transpose( tensor_view<const T> x, tensor_view<T> y, const size_t perm[] )
    {
        int yi[D];
        nd_loop<D,int>(x.shape, [&]( const int xi[] )
        {
            for_n<D>([&]( const size_t k )
            {
                yi[k] = xi[perm[k]];
            });
            at<D>(y,yi) = at<D>(x,xi);
        });
    }

    template<typename T>
    void transpose( tensor_view<const T> x, tensor_view<T> y, const size_t perm[] )
    {
        static decltype(&_transpose<1,T>) funcs[] =
        {
            _transpose<1,T>,
            _transpose<2,T>,
            _transpose<3,T>,
            _transpose<4,T>,
            _transpose<5,T>,
        };
        funcs[x.rank - 1](x, y, perm);
    }
    
    template<bool Singular, typename T>
    void concat( const size_t n, tensor_view<const T> x[], tensor_view<T> y, const size_t axis )
    {
        const size_t b = nd_volume(axis, y.shape);
        const size_t m = nd_volume(y.rank - axis - 1, y.shape + axis + 1);
        
        for ( size_t i = 0; i < b; ++i )
        {
            for ( size_t j = 0; j < n; ++j )
            {
                const size_t size = Singular ? m : x[j].shape[axis] * m;
                std::copy_n(x[j].data, size, y.data);
                x[j].data += size;
                y.data += size;
            }
        }
    }

    template<bool Singular, typename T>
    void split( const size_t n, tensor_view<const T> x, tensor_view<T> y[], const size_t axis )
    {
        const size_t b = nd_volume(axis, x.shape);
        const size_t m = nd_volume(x.rank - axis - 1, x.shape + axis + 1);
        
        for ( size_t i = 0; i < b; ++i )
        {
            for ( size_t j = 0; j < n; ++j )
            {
                const size_t size = Singular ? m : y[j].shape[axis] * m;
                std::copy_n(x.data, size, y[j].data);
                x.data += size;
                y[j].data += size;
            }
        }
    }

    template<size_t D, typename T>
    void _tile( tensor_view<const T> input, tensor_view<T> output )
    {
        int input_index[D];
        nd_loop<D,int>(output.shape, [&]( const int output_index[] )
        {
            for_n<D>([&]( const size_t k )
            {
                input_index[k] = output_index[k] % input.shape[k];
            });
            at<D>(output, output_index) = at<D>(input, input_index);
        });
    }

    template<typename T>
    void tile( tensor_view<const T> input, tensor_view<T> output )
    {
        static decltype(&_tile<1,T>) funcs[] =
        {
            _tile<1,T>,
            _tile<2,T>,
            _tile<3,T>,
            _tile<4,T>,
            _tile<5,T>,
        };
        return funcs[input.rank - 1](input, output);
    }

    template<size_t D, typename T>
    void _pad_constant( tensor_view<const T> input, tensor_view<T> output, const int padding[], const T value )
    {
        int input_index[D];
        nd_loop<D,int>(output.shape, [&]( const int output_index[] )
        {
            for_n<D>([&]( const size_t k )
            {
                input_index[k] = output_index[k] - padding[k];
            });
            
            const bool valid = all_n<D>([&]( const size_t k ){ return input_index[k] >= 0 && input_index[k] < input.shape[k]; });
            at<D>(output, output_index) = valid ? at<D>(input, input_index) : (T)value;
        });
    }

    template<size_t D, typename T>
    void _pad_replicate( tensor_view<const T> input, tensor_view<T> output, const int padding[] )
    {
        int input_index[D];
        nd_loop<D,int>(output.shape, [&]( const int output_index[] )
        {
            for_n<D>([&]( const size_t k )
            {
                input_index[k] = std::min(std::max(output_index[k] - padding[k], 0), input.shape[k] - 1);
            });
            
            at<D>(output, output_index) = at<D>(input, input_index);
        });
    }

    template<size_t D, typename T>
    void _pad_reflect( tensor_view<const T> input, tensor_view<T> output, const int padding[] )
    {
        int input_index[D];
        nd_loop<D,int>(output.shape, [&]( const int output_index[] )
        {
            for_n<D>([&]( const size_t k )
            {
                auto index = output_index[k] - padding[k];
                if ( index < 0 )
                {
                    input_index[k] = -index;
                }
                else if ( index >= input.shape[k] )
                {
                    input_index[k] = 2 * (input.shape[k] - 1) - index;
                }
                else
                {
                    input_index[k] = index;
                }
            });
            
            at<D>(output, output_index) = at<D>(input, input_index);
        });
    }

    template<size_t D, typename T>
    void _pad_reflect_even( tensor_view<const T> input, tensor_view<T> output, const int padding[] )
    {
        int input_index[D];
        nd_loop<D,int>(output.shape, [&]( const int output_index[] )
        {
            for_n<D>([&]( const size_t k )
            {
                auto index = output_index[k] - padding[k];
                if ( index < 0 )
                {
                    input_index[k] = -index - 1;
                }
                else if ( index >= input.shape[k] )
                {
                    input_index[k] = 2 * (input.shape[k] - 1) - index + 1;
                }
                else
                {
                    input_index[k] = index;
                }
            });
            
            at<D>(output, output_index) = at<D>(input, input_index);
        });
    }
    
    template<typename T>
    void pad_constant( tensor_view<const T> input, tensor_view<T> output, const int padding[], const T value )
    {
        static decltype(&_pad_constant<1,T>) funcs[] =
        {
            _pad_constant<1,T>,
            _pad_constant<2,T>,
            _pad_constant<3,T>,
            _pad_constant<4,T>,
            _pad_constant<5,T>,
        };
        return funcs[input.rank - 1](input, output, padding, value);
    }

    template<typename T>
    void pad_replicate( tensor_view<const T> input, tensor_view<T> output, const int padding[] )
    {
        static decltype(&_pad_replicate<1,T>) funcs[] =
        {
            _pad_replicate<1,T>,
            _pad_replicate<2,T>,
            _pad_replicate<3,T>,
            _pad_replicate<4,T>,
            _pad_replicate<5,T>,
        };
        return funcs[input.rank - 1](input, output, padding);
    }

    template<typename T>
    void pad_reflect( tensor_view<const T> input, tensor_view<T> output, const int padding[] )
    {
        static decltype(&_pad_reflect<1,T>) funcs[] =
        {
            _pad_reflect<1,T>,
            _pad_reflect<2,T>,
            _pad_reflect<3,T>,
            _pad_reflect<4,T>,
            _pad_reflect<5,T>,
        };
        return funcs[input.rank - 1](input, output, padding);
    }

    template<typename T>
    void pad_reflect_even( tensor_view<const T> input, tensor_view<T> output, const int padding[] )
    {
        static decltype(&_pad_reflect_even<1,T>) funcs[] =
        {
            _pad_reflect_even<1,T>,
            _pad_reflect_even<2,T>,
            _pad_reflect_even<3,T>,
            _pad_reflect_even<4,T>,
            _pad_reflect_even<5,T>,
        };
        return funcs[input.rank - 1](input, output, padding);
    }

    template<size_t D, typename T>
    void _slice( tensor_view<const T> input, tensor_view<T> output, const int offset[], const int stride[] )
    {
        int input_index[D];
        nd_loop<D,int>(output.shape, [&]( const int output_index[] )
        {
            for_n<D>([&]( const size_t k )
            {
                input_index[k] = offset[k] + stride[k] * output_index[k];
            });
            
            at<D>(output, output_index) = at<D>(input, input_index);
        });
    }

    template<typename T>
    void slice( tensor_view<const T> input, tensor_view<T> output, const int offset[], const int stride[] )
    {
        static decltype(&_slice<1,T>) funcs[] =
        {
            _slice<1,T>,
            _slice<2,T>,
            _slice<3,T>,
            _slice<4,T>,
            _slice<5,T>,
        };
        return funcs[input.rank - 1](input, output, offset, stride);
    }
    
    template<typename T, typename I>
    void _gather( const T* input, const I* indices, T* output, const size_t b, const size_t d, const size_t n, const size_t m )
    {
        for ( size_t k = 0; k < b; ++k, input += d * m )
        {
            for ( size_t i = 0; i < n; ++i, output += m )
            {
                std::copy_n(input + indices[i] * m, m, output);
            }
        }
    }
    
    template<typename T, typename I>
    void gather( tensor_view<const T> input, tensor_view<const I> indices, tensor_view<T> output, const size_t axis )
    {
        const size_t b = nd_volume(axis, input.shape);
        const size_t d = input.shape[axis];
        const size_t n = nd_volume(indices.rank, indices.shape);
        const size_t m = nd_volume(input.rank - axis - 1, input.shape + axis + 1);
        
        _gather(input.data, indices.data, output.data, b, d, n, m);
    }

}}   // namespace nnef::rt


#endif
