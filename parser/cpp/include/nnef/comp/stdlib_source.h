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
    fragment slice<?>( input: tensor<?>, axes: integer[], begin: integer[], end: integer[], stride: integer[] = [] ) -> ( output: tensor<?> );
    fragment squeeze<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
    fragment unsqueeze<?>( input: tensor<?>, axes: integer[] ) -> ( output: tensor<?> );
    fragment stack<?>( values: tensor<?>[], axis: integer ) -> ( value: tensor<?> );
    fragment unstack<?>( value: tensor<?>, axis: integer ) -> ( values: tensor<?>[] );
    fragment tile<?>( input: tensor<?>, repeats: integer[] ) -> ( output: tensor<?> );
    fragment pad( input: tensor<scalar>, padding: (integer, integer)[], border: string = 'constant', value: scalar = 0.0 ) -> ( output: tensor<scalar> );
    fragment gather<?>( input: tensor<?>, indices: tensor<integer>, axis: integer = 0 ) -> ( output: tensor<?> );
    fragment cast<?>( input: tensor<> ) -> ( output: tensor<?> );


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
    fragment tan( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment sinh( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment cosh( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment tanh( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment asin( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment acos( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment atan( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment asinh( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment acosh( x: tensor<scalar> ) -> ( y: tensor<scalar> );
    fragment atanh( x: tensor<scalar> ) -> ( y: tensor<scalar> );
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

    fragment sqr( x: tensor<scalar> ) -> ( y: tensor<scalar> )
    {
        y = x ^ 2.0;
    }

    fragment sqrt( x: tensor<scalar> ) -> ( y: tensor<scalar> )
    {
        y = x ^ 0.5;
    }

    fragment rsqr( x: tensor<scalar> ) -> ( y: tensor<scalar> )
    {
        y = x ^ -2.0;
    }

    fragment rsqrt( x: tensor<scalar> ) -> ( y: tensor<scalar> )
    {
        y = x ^ -0.5;
    }

    fragment log2( x: tensor<scalar> ) -> ( y: tensor<scalar> )
    {
        y = log(x) / log(2.0);
    }

    fragment min( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<scalar> )
    {
        z = select(x < y, x, y);
    }

    fragment max( x: tensor<scalar>, y: tensor<scalar> ) -> ( z: tensor<scalar> )
    {
        z = select(x > y, x, y);
    }

    fragment clamp( x: tensor<scalar>, a: tensor<scalar>, b: tensor<scalar> ) -> ( y: tensor<scalar> )
    {
        y = max(min(x, b), a);
    }


    # matrix multiplication

    fragment matmul( A: tensor<scalar>, B: tensor<scalar>, transposeA: logical = false, transposeB: logical = false ) -> ( C: tensor<scalar> );

    
    )STDLIB" /* break the raw literal because of max length limit */ R"STDLIB(

    
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

    fragment nearest_downsample( input: tensor<scalar>, factor: integer[] ) -> ( output: tensor<scalar> )
    {
        dims = 2 + length_of(factor);
        output = box(input, size = [1] * dims, stride = [1,1] + factor, padding = [(0,0)] * dims);
    }

    fragment area_downsample( input: tensor<scalar>, factor: integer[] ) -> ( output: tensor<scalar> )
    {
        dims = 2 + length_of(factor);
        output = box(input, size = [1,1] + factor, stride = [1,1] + factor, padding = [(0,0)] * dims, normalize = true);
    }

    fragment nearest_upsample( input: tensor<scalar>, factor: integer[] ) -> ( output: tensor<scalar> )
    {
        dims = 2 + length_of(factor);
        output = debox(input, size = [1,1] + factor, stride = [1,1] + factor, padding = [(0,0)] * dims);
    }

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

    fragment mean_reduce( input: tensor<scalar>, axes: integer[] ) -> ( output: tensor<scalar> )
    {
        output = sum_reduce(input, axes = axes, normalize = true);
    }

    fragment moments( input: tensor<scalar>, axes: integer[] ) -> ( mean: tensor<scalar>, variance: tensor<scalar> )
    {
        mean = mean_reduce(input, axes = axes);
        variance = mean_reduce(sqr(input - mean), axes = axes);
    }


    # activation functions

    fragment relu( x: tensor<scalar> ) -> ( y: tensor<scalar> )
    {
        y = max(x, 0.0);
    }

    fragment sigmoid( x: tensor<scalar> ) -> ( y: tensor<scalar> )
    {
        y = 1.0 / (1.0 + exp(-x));
    }

    fragment softabs( x: tensor<scalar>, epsilon: scalar ) -> ( y: tensor<scalar> )
    {
        y = sqrt(sqr(x) + epsilon);
    }

    fragment softmax( x: tensor<scalar>, axes: integer[] = [1] ) -> ( y: tensor<scalar> )
    {
        m = max_reduce(x, axes = axes);
        e = exp(x - m);
        y = e / sum_reduce(e, axes = axes);
    }

    fragment softplus( x: tensor<scalar> ) -> ( y: tensor<scalar> )
    {
        y = log(exp(x) + 1.0);
    }

    fragment elu( x: tensor<scalar>, alpha: scalar = 1.0 ) -> ( y: tensor<scalar> )
    {
        y = select(x < 0.0, alpha * (exp(x) - 1.0), x);
    }
    
    fragment selu( x: tensor<scalar>, alpha: scalar = 1.67326319, lambda: scalar = 1.05070102 ) -> ( y: tensor<scalar> )
    {
        y = lambda * select(x < 0.0, alpha * (exp(x) - 1.0), x);
    }

    fragment gelu( x: tensor<scalar> ) -> ( y: tensor<scalar> )
    {
        # the exact definition of gelu is x * Phi(x) where Phi(x) is the
        # CDF of the standard normal distribution, which can be approximated
        # for example by sigmoid(1.702 * x)

        y = x * sigmoid(1.702 * x);
    }

    fragment silu( x: tensor<scalar> ) -> ( y: tensor<scalar> )
    {
        y = x * sigmoid(x);
    }

    fragment prelu( x: tensor<scalar>, alpha: tensor<scalar> ) -> ( y: tensor<scalar> )
    {
        y = select(x < 0.0, alpha * x, x);
    }

    fragment leaky_relu( x: tensor<scalar>, alpha: scalar ) -> ( y: tensor<scalar> )
    {
        y = prelu(x, alpha = alpha);
    }
    
    
    )STDLIB" /* break the raw literal because of max length limit */ R"STDLIB(


    # pooling operations

    fragment max_pool_with_index(
        input: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( output: tensor<scalar>, index: tensor<integer> )
    {
        index = argmax_pool(input, size = size, border = border, padding = padding, stride = stride, dilation = dilation);
        output = sample(input, index, size = size, border = border, padding = padding, stride = stride, dilation = dilation);
    }

    fragment max_pool(
        input: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( output: tensor<scalar> )
    {
        output, index = max_pool_with_index(input, size = size, border = border, padding = padding, stride = stride, dilation = dilation);
    }

    fragment avg_pool(
        input: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( output: tensor<scalar> )
    {
        output = box(input, size = size, border = border, padding = padding, stride = stride, dilation = dilation, normalize = true);
    }

    fragment rms_pool(
        input: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( output: tensor<scalar> )
    {
        output = sqrt(avg_pool(sqr(input), size = size, border = border, padding = padding, stride = stride, dilation = dilation));
    }


    # linear operations

    fragment linear(
        input: tensor<scalar>,
        filter: tensor<scalar>,
        bias: tensor<scalar> = 0.0 )
    -> ( output: tensor<scalar> )
    {
        output = matmul(input, filter, transposeB = true) + bias;
    }

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
    -> ( output: tensor<scalar> )
    {
        filtered = conv(input, plane_filter, border = border, padding = padding,
                        stride = stride, dilation = dilation, groups = 0);
        output = conv(filtered, point_filter, bias, groups = groups);
    }

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
    -> ( output: tensor<scalar> )
    {
        filtered = deconv(input, point_filter, groups = groups);
        output = deconv(filtered, plane_filter, bias, border = border, padding = padding,
                        stride = stride, dilation = dilation, output_shape = output_shape, groups = 0);
    }


    # normalization operations

    fragment local_response_normalization(
        input: tensor<scalar>,
        size: integer[],
        alpha: scalar = 1.0,
        beta: scalar = 0.5,
        bias: scalar = 1.0 )
    -> ( output: tensor<scalar> )
    {
        sigma = bias + alpha * box(sqr(input), size = size, normalize = true);
        output = input / (sigma ^ beta);
    }

    fragment local_mean_normalization( input: tensor<scalar>, size: integer[] ) -> ( output: tensor<scalar> )
    {
        mean = box(input, size = size, normalize = true);
        output = sub(input, mean);
    }

    fragment local_variance_normalization( input: tensor<scalar>, size: integer[], bias: scalar = 0.0, epsilon: scalar = 0.0 ) -> ( output: tensor<scalar> )
    {
        sigma = sqrt(box(sqr(input), size = size, normalize = true));
        output = input / max(sigma + bias, epsilon);
    }

    fragment local_contrast_normalization( input: tensor<scalar>, size: integer[], bias: scalar = 0.0, epsilon: scalar = 0.0 ) -> ( output: tensor<scalar> )
    {
        centered = local_mean_normalization(input, size = size);
        output = local_variance_normalization(centered, size = size, bias = bias, epsilon = epsilon);
    }

    fragment l1_normalization( input: tensor<scalar>, axes: integer[], bias: scalar = 0.0, epsilon: scalar = 0.0 ) -> ( output: tensor<scalar> )
    {
        sigma = sum_reduce(abs(input), axes = axes);
        output = input / max(sigma + bias, epsilon);
    }

    fragment l2_normalization( input: tensor<scalar>, axes: integer[], bias: scalar = 0.0, epsilon: scalar = 0.0 ) -> ( output: tensor<scalar> )
    {
        sigma = sqrt(sum_reduce(sqr(input), axes = axes));
        output = input / max(sigma + bias, epsilon);
    }

    fragment batch_normalization( input: tensor<scalar>, mean: tensor<scalar>, variance: tensor<scalar>, offset: tensor<scalar>, scale: tensor<scalar>, epsilon: scalar )
    -> ( output: tensor<scalar> )
    {
        output = offset + scale * (input - mean) / sqrt(variance + epsilon);
    }
    
    
    )STDLIB" /* break the raw literal because of max length limit */ R"STDLIB(


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
    -> ( output: tensor<scalar> )
    {
        size = [for i in range_of(output_size) yield output_size[i] * sampling_rate[i]];
        resized = roi_resample(input, rois, batch_index, output_size = size,
                             method = resize_method);
        output = avg_pool(resized, size = sampling_rate, stride = sampling_rate);
    }

    fragment max_roi_align(
        input: tensor<scalar>,
        rois: tensor<scalar>,
        batch_index: tensor<integer>,
        output_size: integer[],
        sampling_rate: integer[],
        resize_method: string = 'symmetric' )
    -> ( output: tensor<scalar> )
    {
        size = [for i in range_of(output_size) yield output_size[i] * sampling_rate[i]];
        resized = roi_resample(input, rois, batch_index, output_size = size,
                             method = resize_method);
        output = max_pool(resized, size = sampling_rate, stride = sampling_rate);
    }


    # quantization operations

    fragment min_max_linear_quantize(
        x: tensor<scalar>,
        min: tensor<scalar>,
        max: tensor<scalar>,
        bits: integer,
        signed: logical,
        symmetric: logical )
    -> ( y: tensor<scalar> )
    {
        r = scalar(2 ^ bits - 1 - integer(signed && symmetric));
        z = clamp(x, min, max);
        p = scalar(2 ^ (bits - 1) - integer(symmetric) if signed else 0);
        q = round((z - min) / (max - min) * r) - p;
        y = (q + p) / r * (max - min) + min;
    }

    fragment zero_point_linear_quantize(
        x: tensor<scalar>,
        zero_point: tensor<integer>,
        scale: tensor<scalar>,
        bits: integer,
        signed: logical,
        symmetric: logical )
    -> ( y: tensor<scalar> )
    {
        z = cast<scalar>(zero_point);
        s = round(x / scale) + z;
        r = scalar(2 ^ (bits - 1) - 1 if signed else 2 ^ bits - 1);
        q = clamp(s, 0.0 if !signed else -r if symmetric else -r - 1.0, r);
        y = (q - z) * scale;
    }

    fragment linear_quantize(
        x: tensor<scalar>,
        min: tensor<scalar>,
        max: tensor<scalar>,
        bits: integer )
    -> ( y: tensor<scalar> )
    {
        y = min_max_linear_quantize(x, min = min, max = max, bits = bits,
                                    signed = false, symmetric = false);
    }

    fragment logarithmic_quantize(
        x: tensor<scalar>,
        max: tensor<scalar>,
        bits: integer )
    -> ( y: tensor<scalar> )
    {
        m = ceil(log2(max));
        r = scalar(2 ^ bits - 1);
        q = round(clamp(log2(abs(x)), m - r, m));
        y = sign(x) * 2.0 ^ q;
    }


    # misc operations

    fragment copy_n<?>( x: tensor<?>, times: integer ) -> ( y: tensor<?>[] )
    {
        y = [x] * times;
    }

    fragment add_n( x: tensor<scalar>[] ) -> ( y: tensor<scalar> )
    {
        y = x[0] + add_n(x[1:]) if length_of(x) > 0 else constant(shape = [1], value = [0.0]);
    }


    )STDLIB";


    inline const char* stdlib_source()
    {
        return _stdlib_source<void>::text;
    }

}   // namespace nnef

#endif
