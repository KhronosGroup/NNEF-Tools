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

#ifndef _NNEF_STDLIB_SOURCE_H_
#define _NNEF_STDLIB_SOURCE_H_


namespace nnef {

    template<typename T> struct _stdlib_source { static const char* text; };
    template<typename T> const char* _stdlib_source<T>::text = R"STDLIB(


    # tensor declaration operations

    fragment external<? = scalar>( shape: integer[] ) -> ( output: tensor<?> );
    fragment variable<? = scalar>( shape: integer[], label: string ) -> ( output: tensor<?> );
    fragment constant<? = scalar>( shape: integer[], value: ?[] ) ->  ( output: tensor<?> );

    fragment update<?>( variable: tensor<?>, value: tensor<?> ) -> ( result: tensor<?> );


    # tensor shape operations

    fragment reshape<?>( input: tensor<?>, shape: integer[], axis_start: integer = 0, axis_count: integer = -1 ) -> ( output: tensor<?> );
    fragment transpose<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
    fragment concat<?>( values: tensor<?>[], axis: integer ) -> ( value: tensor<?> );
    fragment split<?>( value: tensor<?>, axis: integer, ratios: integer[] ) -> ( values: tensor<?>[] );
    fragment slice<?>( input: tensor<?>, axes: integer[], begin: integer[], end: integer[] ) -> ( output: tensor<?> );
    fragment squeeze<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
    fragment unsqueeze<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
    fragment stack<?>( values: tensor<?>[], axis: integer ) -> ( value: tensor<?> );
    fragment unstack<?>( value: tensor<?>, axis: integer ) -> ( values: tensor<?>[] );
    fragment tile<?>( input: tensor<?>, repeats: integer[] ) -> ( output: tensor<?> );
    fragment pad( input: tensor<scalar>, padding: (integer, integer)[], border: string = 'constant', value: scalar = 0.0 ) -> ( output: tensor<scalar> );


    # element-wise arithmetic operations

    fragment add( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<scalar> );
    fragment sub( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<scalar> );
    fragment mul( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<scalar> );
    fragment div( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<scalar> );
    fragment pow( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<scalar> );

    fragment exp( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment log( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment sin( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment cos( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment abs( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment sign( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment rcp( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment neg( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment copy<?>( x: tensor<?> ) -> ( y: tensor<?> );

    # element-wise comparison operations

    fragment lt( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<logical> );
    fragment gt( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<logical> );
    fragment le( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<logical> );
    fragment ge( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<logical> );
    fragment eq( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<logical> );
    fragment ne( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<logical> );

    # element-wise logical operations

    fragment and( x: tensor<logical>, y: tensor<logical> ) -> ( z: tensor<logical> );
    fragment or( x: tensor<logical>, y: tensor<logical> ) -> ( z: tensor<logical> );
    fragment not( x: tensor<logical> ) -> ( y: tensor<logical> );

    # element-wise rounding operations

    fragment floor( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment ceil( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment round( x: tensor<scalar> ) -> ( y: tensor<scalar> );

    # element-wise select operation

    fragment select<?>( condition: tensor<logical>, true_value: tensor<?>, false_value: tensor<?> ) -> ( output: tensor<?> );

    # simplifier operations

    fragment sqr( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment sqrt( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment rsqr( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment rsqrt( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment log2( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment min( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<scalar> );
    fragment max( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<scalar> );
    fragment clamp( x: tensor<scalar>, a: tensor<scalar>, b: tensor<scalar> ) -> ( y: tensor<scalar> );

    # matrix multiplication

    fragment matmul( A: tensor<scalar>, B: tensor<scalar>, transposeA: logical = false, transposeB: logical = false ) -> ( C: tensor<scalar> );


    # sliding-window operations

    fragment conv(
        input: tensor<scalar>,
        filter: tensor<scalar>,
        bias: tensor<scalar> = 0.0,
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [],
        groups: integer = 1 )
    -> ( output: tensor<scalar> );

    fragment deconv(
        input: tensor<scalar>,
        filter: tensor<scalar>,
        bias: tensor<scalar> = 0.0,
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [],
        output_shape: integer[] = [],
        groups: integer = 1 )
    -> ( output: tensor<scalar> );


    fragment box(
        input: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [],
        normalize: logical = false )
    -> ( output: tensor<scalar> );

    fragment debox(
        input: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [],
        output_shape: integer[] = [],
        normalize: logical = false )
    -> ( output: tensor<scalar> );


    fragment argmax_pool(
        input: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( index: tensor<integer> );


    fragment sample(
        input: tensor<scalar>,
        index: tensor<integer>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( output: tensor<scalar> );

    fragment desample(
        input: tensor<scalar>,
        index: tensor<integer>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [],
        output_shape: integer[] = [] )
    -> ( output: tensor<scalar> );


    # up/down-sampling operations

    fragment nearest_downsample( input: tensor<scalar>, factor: integer[] ) -> ( output: tensor<scalar> );
    fragment area_downsample( input: tensor<scalar>, factor: integer[] ) -> ( output: tensor<scalar> );
    fragment nearest_upsample( input: tensor<scalar>, factor: integer[] ) -> ( output: tensor<scalar> );
    fragment multilinear_upsample( input: tensor<scalar>, factor: integer[], method: string = 'symmetric', border: string = 'replicate' )
    -> ( output: tensor<scalar> );


    # reduce operations

    fragment sum_reduce( input: tensor<scalar>, axes: integer[], normalize: logical = false ) -> ( output: tensor<scalar> );
    fragment max_reduce( input: tensor<scalar>, axes: integer[] ) -> ( output: tensor<scalar> );
    fragment min_reduce( input: tensor<scalar>, axes: integer[] ) -> ( output: tensor<scalar> );
    fragment argmax_reduce( input: tensor<scalar>, axes: integer[] ) -> ( output: tensor<integer> );
    fragment argmin_reduce( input: tensor<scalar>, axes: integer[] ) -> ( output: tensor<integer> );
    fragment any_reduce( input: tensor<logical>, axes: integer[] ) -> ( output: tensor<logical> );
    fragment all_reduce( input: tensor<logical>, axes: integer[] ) -> ( output: tensor<logical> );

    fragment mean_reduce( input: tensor<scalar>, axes: integer[] ) -> ( output: tensor<scalar> );
    fragment moments( input: tensor<scalar>, axes: integer[] ) -> ( mean: tensor<scalar>, variance: tensor<scalar> );


    # activation functions

    fragment relu( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment sigmoid( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment tanh( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment softabs( x: tensor<scalar>, epsilon: scalar ) -> ( y: tensor<scalar> );
    fragment softmax( x: tensor<scalar>, axes: integer[] = [1] ) -> ( y: tensor<scalar> );
    fragment softplus( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment elu( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment prelu( x: tensor<scalar>, alpha: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment leaky_relu( x: tensor<scalar>, alpha: scalar ) -> ( y: tensor<scalar> );


    # pooling operations

    fragment max_pool_with_index(
        input: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( output: tensor<scalar>, index: tensor<integer> );

    fragment max_pool(
        input: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( output: tensor<scalar> );

    fragment avg_pool(
        input: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( output: tensor<scalar> );

    fragment rms_pool(
        input: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( output: tensor<scalar> );


    # linear operations

    fragment linear(
        input: tensor<scalar>,
        filter: tensor<scalar>,
        bias: tensor<scalar> = 0.0 )
    -> ( output: tensor<scalar> );

    fragment separable_conv(
        input: tensor<scalar>,
        plane_filter: tensor<scalar>,
        point_filter: tensor<scalar>,
        bias: tensor<scalar> = 0.0,
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [],
        groups: integer = 1 )
    -> ( output: tensor<scalar> );

    fragment separable_deconv(
        input: tensor<scalar>,
        plane_filter: tensor<scalar>,
        point_filter: tensor<scalar>,
        bias: tensor<scalar> = 0.0,
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [],
        output_shape: integer[] = [],
        groups: integer = 1 )
    -> ( output: tensor<scalar> );


    # normalization operations

    fragment local_response_normalization(
        input: tensor<scalar>,
        size: integer[],
        alpha: scalar = 1.0,
        beta: scalar = 0.5,
        bias: scalar = 1.0 )
    -> ( output: tensor<scalar> );

    fragment local_mean_normalization(
        input: tensor<scalar>,
        size: integer[] )
    -> ( output: tensor<scalar> );

    fragment local_variance_normalization(
        input: tensor<scalar>,
        size: integer[],
        bias: scalar = 0.0,
        epsilon: scalar = 0.0 )
    -> ( output: tensor<scalar> );

    fragment local_contrast_normalization(
        input: tensor<scalar>,
        size: integer[],
        bias: scalar = 0.0,
        epsilon: scalar = 0.0 )
    -> ( output: tensor<scalar> );

    fragment l1_normalization(
        input: tensor<scalar>,
        axes: integer[],
        bias: scalar = 0.0,
        epsilon: scalar = 0.0 )
    -> ( output: tensor<scalar> );

    fragment l2_normalization(
        input: tensor<scalar>,
        axes: integer[],
        bias: scalar = 0.0,
        epsilon: scalar = 0.0 )
    -> ( output: tensor<scalar> );

    fragment batch_normalization(
        input: tensor<scalar>,
        mean: tensor<scalar>,
        variance: tensor<scalar>,
        offset: tensor<scalar>,
        scale: tensor<scalar>,
        epsilon: scalar )
    -> ( output: tensor<scalar> );


    # roi operations

    fragment avg_roi_pool(
        input: tensor<scalar>,
        rois: tensor<scalar>,
        batch_index: tensor<integer>,
        output_size: integer[] )
    -> ( output: tensor<scalar> );

    fragment max_roi_pool(
        input: tensor<scalar>,
        rois: tensor<scalar>,
        batch_index: tensor<integer>,
        output_size: integer[] )
    -> ( output: tensor<scalar> );

    fragment roi_resample(
        input: tensor<scalar>,
        rois: tensor<scalar>,
        batch_index: tensor<integer>,
        output_size: integer[],
        method: string = 'symmetric' )
    -> ( output: tensor<scalar> );

    fragment avg_roi_align(
        input: tensor<scalar>,
        rois: tensor<scalar>,
        batch_index: tensor<integer>,
        output_size: integer[],
        sampling_rate: integer[],
        resize_method: string = 'symmetric' )
    -> ( output: tensor<scalar> );

    fragment max_roi_align(
        input: tensor<scalar>,
        rois: tensor<scalar>,
        batch_index: tensor<integer>,
        output_size: integer[],
        sampling_rate: integer[],
        resize_method: string = 'symmetric' )
    -> ( output: tensor<scalar> );


    # quantization operations

    fragment linear_quantize( x: tensor<scalar>, min: tensor<scalar>, max: tensor<scalar>, bits: integer ) -> ( y: tensor<scalar> );
    fragment logarithmic_quantize( x: tensor<scalar>, max: tensor<scalar>, bits: integer ) -> ( y: tensor<scalar> );


    # misc operations

    fragment copy_n<?>( x: tensor<?>, times: integer ) -> ( y: tensor<?>[] );
    fragment add_n( x: tensor<scalar>[] ) -> ( y: tensor<scalar> );


    )STDLIB";


    inline const char* stdlib_source()
    {
        return _stdlib_source<void>::text;
    }

}   // namespace nnef

#endif
