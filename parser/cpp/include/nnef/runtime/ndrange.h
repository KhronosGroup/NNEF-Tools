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

#ifndef _NNEF_RUNTIME_NDRANGE_H_
#define _NNEF_RUNTIME_NDRANGE_H_


namespace nnef { namespace rt
{

    template<size_t N, size_t K, typename I, typename S, typename Op>
    struct _nd_loop
    {
        static inline void call( const S shape[], I index[], const Op& op )
        {
            for ( index[N-K] = 0; index[N-K] < shape[N-K]; ++index[N-K] )
            {
                _nd_loop<N,K-1,I,S,Op>::call(shape, index, op);
            }
        }
    };

    template<size_t N, typename I, typename S, typename Op>
    struct _nd_loop<N,1,I,S,Op>
    {
        static inline void call( const S shape[], I index[], const Op& op )
        {
            for ( index[N-1] = 0; index[N-1] < shape[N-1]; ++index[N-1] )
            {
                op(index);
            }
        }
    };
    
    template<typename I, typename S, typename Op>
    struct _nd_loop<0,0,I,S,Op>
    {
        static inline void call( const S shape[], I index[], const Op& op )
        {
            op(index);
        }
    };

    template<size_t N, typename I, typename S, typename Op>
    inline void nd_loop( const S shape[], const Op& op )
    {
        I index[N];
        _nd_loop<N,N,S,I,Op>::call(shape, index, op);
    };


    template<size_t N, typename I, typename S>
    struct _nd_offset
    {
        static inline size_t call( const S shape[], const I index[] )
        {
            return _nd_offset<N-1,I,S>::call(shape, index) * shape[N-1] + index[N-1];
        }
    };

    template<typename I, typename S>
    struct _nd_offset<1,I,S>
    {
        static inline size_t call( const S shape[], const I index[] )
        {
            return index[0];
        }
    };

    template<typename I, typename S>
    struct _nd_offset<0,I,S>
    {
        static inline size_t call( const S shape[], const I index[] )
        {
            return 0;
        }
    };

    template<size_t N, typename I, typename S>
    inline size_t nd_offset( const S shape[], const I index[] )
    {
        return _nd_offset<N,I,S>::call(shape, index);
    }

    
    template<size_t N, typename S>
    struct _nd_volume
    {
        static inline size_t call( const S shape[] )
        {
            return _nd_volume<N-1,S>::call(shape) * shape[N-1];
        }
    };

    template<typename S>
    struct _nd_volume<1,S>
    {
        static inline size_t call( const S shape[] )
        {
            return shape[0];
        }
    };

    template<typename S>
    struct _nd_volume<0,S>
    {
        static inline size_t call( const S shape[] )
        {
            return 1;
        }
    };

    template<size_t N, typename S>
    inline size_t nd_volume( const S shape[] )
    {
        return _nd_volume<N,S>::call(shape);
    }

    template<typename S>
    inline size_t nd_volume( const size_t rank, const S shape[] )
    {
        return std::accumulate(shape, shape + rank, (S)1, std::multiplies<S>());
    }

    
    template<size_t N, typename Op>
    struct _for_n
    {
        static inline void call( const Op& op )
        {
            _for_n<N-1,Op>::call(op);
            op(N-1);
        };
    };

    template<typename Op>
    struct _for_n<1,Op>
    {
        static inline void call( const Op& op )
        {
            op(0);
        };
    };

    template<typename Op>
    struct _for_n<0,Op>
    {
        static inline void call( const Op& op )
        {
        };
    };

    template<size_t N, typename Op>
    inline void for_n( const Op& op )
    {
        _for_n<N,Op>::call(op);
    }


    template<size_t N, typename Op>
    struct _all_n
    {
        static inline bool call( const Op& op )
        {
            return _all_n<N-1,Op>::call(op) && op(N-1);
        };
    };

    template<typename Op>
    struct _all_n<1,Op>
    {
        static inline bool call( const Op& op )
        {
            return op(0);
        };
    };

    template<typename Op>
    struct _all_n<0,Op>
    {
        static inline bool call( const Op& op )
        {
            return true;
        };
    };

    template<size_t N, typename Op>
    inline bool all_n( const Op& op )
    {
        return _all_n<N,Op>::call(op);
    };


    template<typename T>
    struct tensor_view
    {
        const size_t rank;
        const size_t volume;
        const int* shape;
        T* data;
        
        tensor_view operator[]( const size_t idx ) const
        {
            const size_t size = volume / *shape;
            return tensor_view{ rank - 1, size, shape + 1, data + size * idx };
        }
        
        operator tensor_view<const T>() const
        {
            return tensor_view<const T>{ rank, volume, shape, data };
        }
    };

    
    template<size_t D, typename T>
    T& at( tensor_view<T>& view, const int idx[] )
    {
        return view.data[nd_offset<D>(view.shape, idx)];
    }

}}  // namespace nnef::rt

#endif
