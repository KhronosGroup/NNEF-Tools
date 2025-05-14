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

#ifndef _SKND_INTRINSICS_H_
#define _SKND_INTRINSICS_H_

#include "runtime.h"
#include <algorithm>


namespace sknd
{
    namespace rt
    {
        namespace detail
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
            
            size_t volume( const int* shape, const size_t rank )
            {
                size_t vol = 1;
                for ( size_t i = 0; i < rank; ++i )
                {
                    vol *= shape[i];
                }
                return vol;
            }
        
            template<typename T>
            size_t nnz( const T* input, const size_t count )
            {
                size_t n = 0;
                for ( size_t i = 0; i < count; ++i, ++input )
                {
                    if ( *input )
                    {
                        ++n;
                    }
                }
                return n;
            }
            
            
            template<typename T, typename I>
            struct StableGreater
            {
                bool operator()( const std::pair<T,I>& p, const std::pair<T,I>& q )
                {
                    return p.first > q.first || (p.first == q.first && p.second < q.second);
                }
            };
            
            template<typename T, typename I>
            struct StableLess
            {
                bool operator()( const std::pair<T,I>& p, const std::pair<T,I>& q )
                {
                    return p.first < q.first || (p.first == q.first && p.second < q.second);
                }
            };
            
            
            template<typename T, typename I>
            void top_k( const T* input, T* values, I* indices, const size_t input_count, const size_t output_count,
                       const size_t stride, const bool largest, const bool sorted, std::pair<T,I>* buffer )
            {
                for ( size_t i = 0; i < input_count; ++i, input += stride )
                {
                    buffer[i].first = *input;
                    buffer[i].second = (I)i;
                }
                
                if ( sorted )
                {
                    if ( largest )
                    {
                        std::partial_sort(buffer, buffer + output_count, buffer + input_count, StableGreater<T,I>());
                    }
                    else
                    {
                        std::partial_sort(buffer, buffer + output_count, buffer + input_count, StableLess<T,I>());
                    }
                }
                else
                {
                    if ( largest )
                    {
                        std::nth_element(buffer, buffer + output_count, buffer + input_count, StableGreater<T,I>());
                    }
                    else
                    {
                        std::nth_element(buffer, buffer + output_count, buffer + input_count, StableLess<T,I>());
                    }
                }
                
                for ( size_t i = 0; i < output_count; ++i, values += stride, indices += stride )
                {
                    *values = buffer[i].first;
                    *indices = buffer[i].second;
                }
            }
        
            
            template<typename T>
            struct BBox
            {
                T y_min, x_min, y_max, x_max;
                
                T area() const { return (x_max - x_min) * (y_max - y_min); }
            };
            
            template<typename T>
            BBox<T> extract_bbox( const T* coords, bool centered )
            {
                if ( centered )
                {
                    const T x = coords[0];
                    const T y = coords[1];
                    const T width = coords[2];
                    const T height = coords[3];
                    return BBox<T>{ y - height / 2, x - width / 2, y + height / 2, x + width / 2 };
                }
                else
                {
                    const T y1 = coords[0];
                    const T x1 = coords[1];
                    const T y2 = coords[2];
                    const T x2 = coords[3];
                    return BBox<T>{ std::min(y1, y2), std::min(x1, x2), std::max(y1, y2), std::max(x1, x2) };
                }
            }
            
            template<typename T>
            T iou( const BBox<T>& a, const BBox<T>& b )
            {
                auto x_min = std::max(a.x_min, b.x_min);
                auto y_min = std::max(a.y_min, b.y_min);
                auto x_max = std::min(a.x_max, b.x_max);
                auto y_max = std::min(a.y_max, b.y_max);
                
                auto area = std::max((T)0, x_max - x_min) * std::max((T)0, y_max - y_min);
                
                return area ? area / (a.area() + b.area() - area) : (T)0;
            }
            
            template<typename T, typename I>
            size_t bbox_nms( const T* boxes, const T* scores, I* indices, const size_t batch, const size_t clazz,
                            const size_t anchors, const size_t max_outputs, const bool box_format_centered,
                            const T iou_threshold, const std::optional<T> score_threshold )
            {
                std::vector<std::pair<T,size_t>> filtered_boxes;
                for ( size_t i = 0; i < anchors; ++i )
                {
                    if ( !score_threshold || scores[i] > score_threshold )
                    {
                        filtered_boxes.push_back(std::make_pair(scores[i], i));
                    }
                }
                
                std::make_heap(filtered_boxes.begin(), filtered_boxes.end());
                
                size_t count = 0;
                while ( !filtered_boxes.empty() && count < max_outputs )
                {
                    std::pop_heap(filtered_boxes.begin(), filtered_boxes.end());
                    const size_t i = filtered_boxes.back().second;
                    filtered_boxes.pop_back();
                    
                    const BBox<T> next_bbox = extract_bbox(boxes + 4 * i, box_format_centered);
                    
                    bool overlaps = false;
                    for ( size_t k = 0; k < count; ++k )
                    {
                        const size_t j = indices[3 * k + 2];
                        const BBox<T> bbox = extract_bbox(boxes + 4 * j, box_format_centered);
                        if ( iou(next_bbox, bbox) > iou_threshold )
                        {
                            overlaps = true;
                            break;
                        }
                    }
                    if ( !overlaps )
                    {
                        size_t offs = 3 * count++;
                        indices[offs + 0] = (I)batch;
                        indices[offs + 1] = (I)clazz;
                        indices[offs + 2] = (I)i;
                    }
                }
                
                return count;
            }
        
        }   // namespace detail
        
        
        template<size_t N, typename T, typename I>
        void nonzero( const Tensor<N,T>& input, Tensor<2,I>& indices )
        {
            auto count = detail::nnz(input.data(), input.volume());
            indices.reshape((int)N, (int)count);
            
            auto input_data = input.data();
            int k = 0;
            
            detail::nd_loop<N,I>(input.shape(), [&]( const I index[N] )
            {
                if ( *input_data++ )
                {
                    for ( int i = 0; i < (int)N; ++i )
                    {
                        indices(i,k) = index[i];
                    }
                    k += 1;
                }
            });
        }
        
        template<size_t N, typename T, typename I>
        void top_k( const Tensor<N,T>& input, Tensor<N,T>& values, Tensor<N,I>& indices,
                   const size_t k, const size_t axis, const bool largest, const bool sorted )
        {
            const size_t batch = detail::volume(input.shape(), axis);
            const size_t stride = detail::volume(input.shape() + axis + 1, input.rank() - axis - 1);
            const size_t input_count = input.shape()[axis];
            const size_t output_count = std::min((size_t)values.shape()[axis], k);
            
            const T* input_data = input.data();
            T* values_data = values.data();
            I* indices_data = indices.data();
            
            std::vector<std::pair<T,I>> pairs(input_count);
            for ( size_t i = 0; i < batch; ++i )
            {
                for ( size_t j = 0; j < stride; ++j, ++input_data, ++values_data, ++indices_data )
                {
                    detail::top_k(input_data, values_data, indices_data, input_count, output_count, stride, largest, sorted, pairs.data());
                }
                
                input_data += input_count * stride - stride;
                values_data += output_count * stride - stride;
                indices_data += output_count * stride - stride;
            }
        }
        
        template<typename T, typename I>
        void bbox_nms( const Tensor<3,T>& boxes, const Tensor<3,T>& scores, Tensor<2,I>& indices,
                      const bool box_format_centered, const size_t max_outputs_per_class,
                      const real_t iou_threshold, const std::optional<real_t> score_threshold = std::nullopt )
        {
            const size_t batch = scores.shape()[0];
            const size_t classes = scores.shape()[1];
            const size_t anchors = scores.shape()[2];
            
            const T* boxes_data = boxes.data();
            const T* scores_data = scores.data();
            
            indices.reshape((int)(batch * classes * max_outputs_per_class), 3);
            
            size_t count = 0;
            for ( size_t b = 0; b < batch; ++b, boxes_data += anchors * 4 )
            {
                for ( size_t c = 0; c < classes; ++c, scores_data += anchors )
                {
                    count += detail::bbox_nms(boxes_data, scores_data, indices.data() + count * 3, b, c, anchors,
                                              max_outputs_per_class, box_format_centered, iou_threshold, score_threshold);
                }
            }
            
            indices.reshape((int)count, 3);
        }
    
    }   // namespace rt
}   // namespace sknd


#endif
