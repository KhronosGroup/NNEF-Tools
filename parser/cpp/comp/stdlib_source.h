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

    template<typename T> struct stdlib_source { static const char* text; };
    template<typename T> const char* stdlib_source<T>::text = R"STDLIB(


    # tensor declaration operations

    fragment external( shape: extent[] ) -> ( output: tensor )
    fragment variable( shape: extent[], label: string ) -> ( output: tensor )
    fragment constant( shape: extent[], value: scalar[] ) ->  ( output: tensor )

    fragment update( variable: tensor, value: tensor ) -> ( result: tensor )


    # tensor shape operations

    fragment reshape( input: tensor, shape: extent[] ) -> ( output: tensor )
    fragment transpose( input: tensor, perm: extent[] ) -> ( output: tensor )
    fragment concat( values: tensor[], axis: extent ) -> ( value: tensor )
    fragment split( value: tensor, axis: extent, ratios: extent[] ) -> ( values: tensor[] )


    # element-wise arithmetic operations

    fragment add( x: tensor, y: tensor ) -> ( z: tensor )
    fragment sub( x: tensor, y: tensor ) -> ( z: tensor )
    fragment mul( x: tensor, y: tensor ) -> ( z: tensor )
    fragment div( x: tensor, y: tensor ) -> ( z: tensor )
    fragment pow( x: tensor, y: tensor ) -> ( z: tensor )

    fragment exp( x: tensor ) -> ( y: tensor )
    fragment log( x: tensor ) -> ( y: tensor )
    fragment abs( x: tensor ) -> ( y: tensor )
    fragment sign( x: tensor ) -> ( y: tensor )
    fragment rcp( x: tensor ) -> ( y: tensor )
    fragment neg( x: tensor ) -> ( y: tensor )

    # element-wise comparison operations

    fragment lt( x: tensor, y: tensor ) -> ( z: tensor<logical> )
    fragment gt( x: tensor, y: tensor ) -> ( z: tensor<logical> )
    fragment le( x: tensor, y: tensor ) -> ( z: tensor<logical> )
    fragment ge( x: tensor, y: tensor ) -> ( z: tensor<logical> )
    fragment eq( x: tensor, y: tensor ) -> ( z: tensor<logical> )
    fragment ne( x: tensor, y: tensor ) -> ( z: tensor<logical> )

    # element-wise logical operations

    fragment and( x: tensor<logical>, y: tensor<logical> ) -> ( z: tensor<logical> )
    fragment or( x: tensor<logical>, y: tensor<logical> ) -> ( z: tensor<logical> )
    fragment not( x: tensor<logical> ) -> ( y: tensor<logical> )

    # element-wise rounding operations

    fragment floor( x: tensor ) -> ( y: tensor )
    fragment ceil( x: tensor ) -> ( y: tensor )
    fragment round( x: tensor ) -> ( y: tensor )

    # element-wise select operation

    fragment select( condition: tensor<logical>, true_value: tensor, false_value: tensor ) -> ( output: tensor )

    # simplifier operations

    fragment sqr( x: tensor ) -> ( y: tensor )
    {
        y = x ^ 2.0
    }
    
    fragment sqrt( x: tensor ) -> ( y: tensor )
    {
        y = x ^ 0.5
    }
    
    fragment rsqr( x: tensor ) -> ( y: tensor )
    {
        y = x ^ -2.0
    }
    
    fragment rsqrt( x: tensor ) -> ( y: tensor )
    {
        y = x ^ -0.5
    }
    
    fragment log2( x: tensor ) -> ( y: tensor )
    {
        y = log(x) / log(2.0)
    }

    fragment min( x: tensor, y: tensor ) -> ( z: tensor )
    {
        z = select(x < y, x, y)
    }
    
    fragment max( x: tensor, y: tensor ) -> ( z: tensor )
    {
        z = select(x > y, x, y)
    }

    fragment clamp( x: tensor, a: tensor, b: tensor ) -> ( y: tensor )
    {
        y = max(min(x, b), a)
    }


    # matrix multiplication

    fragment matmul( A: tensor, B: tensor, trA: logical = false, trB: logical = false ) -> ( C: tensor )


    # sliding-window operations

    fragment conv(
        input: tensor,
        filter: tensor,
        bias: tensor = 0.0,
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [],
        groups: extent = 1 )
    -> ( output: tensor )

    fragment deconv(
        input: tensor,
        filter: tensor,
        bias: tensor = 0.0,
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [],
        groups: extent = 1 )
    -> ( output: tensor )


    fragment box(
        input: tensor,
        size: extent[],
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [],
        normalize: logical = false )
    -> ( output: tensor )

    fragment debox(
        input: tensor,
        size: extent[],
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [],
        normalize: logical = false )
    -> ( output: tensor )


    fragment argmax_pool(
        input: tensor,
        size: extent[],
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [] )
    -> ( index: tensor<extent> )


    fragment sample(
        input: tensor,
        index: tensor<extent>,
        size: extent[],
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [] )
    -> ( output: tensor )

    fragment desample(
        input: tensor,
        index: tensor<extent>,
        size: extent[],
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [] )
    -> ( output: tensor )


    # up/down-sampling operations

    fragment nearest_downsample( input: tensor, factor: extent[] ) -> ( output: tensor )
    {
        output = box(input, size = [1], stride = [1,1] + factor, padding = [(0,0)] * (length_of(factor) + 2))
    }

    fragment area_downsample( input: tensor, factor: extent[] ) -> ( output: tensor )
    {
        output = box(input, size = [1,1] + factor, stride = [1,1] + factor, padding = [(0,0)] * (length_of(factor) + 2), normalize = true)
    }

    fragment nearest_upsample( input: tensor, factor: extent[] ) -> ( output: tensor )
    {
        output = debox(input, size = [1,1] + factor, stride = [1,1] + factor, padding = [(0,0)] * (length_of(factor) + 2))
    }

    fragment multilinear_upsample( input: tensor, factor: extent[], method: string = 'symmetric', border: string = 'replicate' ) -> ( output: tensor )


    # reduce operations

    fragment sum_reduce( input: tensor, axes: extent[], normalize: logical = false ) -> ( output: tensor )
    fragment min_reduce( input: tensor, axes: extent[] ) -> ( output: tensor )
    fragment max_reduce( input: tensor, axes: extent[] ) -> ( output: tensor )

    fragment mean_reduce( input: tensor, axes: extent[] ) -> ( output: tensor )
    {
        output = sum_reduce(input, axes = axes, normalize = true)
    }

    fragment moments( input: tensor, axes: extent[] ) -> ( mean: tensor, variance: tensor )
    {
        mean = mean_reduce(input, axes = axes)
        variance = mean_reduce(sqr(input - mean), axes = axes)
    }


    # activation functions

    fragment relu( x: tensor ) -> ( y: tensor )
    {
        y = max(x, 0.0)
    }

    fragment sigmoid( x: tensor ) -> ( y: tensor )
    {
        y = 1.0 / (1.0 + exp(-x))
    }

    fragment sinh( x: tensor ) -> ( y: tensor )
    {
        y = 0.5 * (exp(x) - exp(-x))
    }

    fragment cosh( x: tensor ) -> ( y: tensor )
    {
        y = 0.5 * (exp(x) + exp(-x))
    }

    fragment tanh( x: tensor ) -> ( y: tensor )
    {
        y = sinh(x) / cosh(x)
    }

    fragment softabs( x: tensor, epsilon: scalar ) -> ( y: tensor )
    {
        y = sqrt(sqr(x) + epsilon)
    }

    fragment softmax( x: tensor, axes: extent[] = [1] ) -> ( y: tensor )
    {
        m = max_reduce(x, axes = axes)
        e = exp(x - m)
        y = e / sum_reduce(e, axes = axes)
    }

    fragment softplus( x: tensor ) -> ( y: tensor )
    {
        y = log(exp(x) + 1.0)
    }

    fragment elu( x: tensor ) -> ( y: tensor )
    {
        y = select(x < 0.0, exp(x) - 1.0, x)
    }

    fragment leaky_relu( x: tensor, alpha: scalar ) -> ( y: tensor )
    {
        y = select(x < 0.0, alpha * x, x)
    }


    # pooling operations

    fragment max_pool_with_index(
        input: tensor,
        size: extent[],
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [] )
    -> ( output: tensor, index: tensor<extent> )
    {
        index = argmax_pool(input, size = size, border = border, padding = padding, stride = stride, dilation = dilation)
        output = sample(input, index, size = size, border = border, padding = padding, stride = stride, dilation = dilation)
    }

    fragment max_pool(
        input: tensor,
        size: extent[],
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [] )
    -> ( output: tensor )
    {
        output, index = max_pool_with_index(input, size = size, border = border, padding = padding, stride = stride, dilation = dilation)
    }

    fragment avg_pool(
        input: tensor,
        size: extent[],
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [] )
    -> ( output: tensor )
    {
        output = box(input, size = size, border = border, padding = padding, stride = stride, dilation = dilation, normalize = true)
    }

    fragment rms_pool(
        input: tensor,
        size: extent[],
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [] )
    -> ( output: tensor )
    {
        output = sqrt(avg_pool(sqr(input), size = size, border = border, padding = padding, stride = stride, dilation = dilation))
    }


    # linear operations

    fragment linear(
        input: tensor,
        filter: tensor,
        bias: tensor = 0.0 )
    -> ( output: tensor )
    {
        output = matmul(input, filter, trB = true) + bias
    }

    fragment planewise_conv(
        input: tensor,
        filter: tensor,
        bias: tensor = 0.0,
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [] )
    -> ( output: tensor )
    {
        output = conv(input, filter, bias, border = border, padding = padding, stride = stride, dilation = dilation, groups = 0)
    }

    fragment planewise_deconv(
        input: tensor,
        filter: tensor,
        bias: tensor = 0.0,
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [] )
    -> ( output: tensor )
    {
        output = deconv(input, filter, bias, border = border, padding = padding, stride = stride, dilation = dilation, groups = 0)
    }

    fragment separable_conv(
        input: tensor,
        plane_filter: tensor,
        point_filter: tensor,
        bias: tensor = 0.0,
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [],
        groups: extent = 1 )
    -> ( output: tensor )
    {
        filtered = planewise_conv(input, plane_filter, border = border, padding = padding, stride = stride, dilation = dilation)
        output = conv(filtered, point_filter, bias, groups = groups)
    }

    fragment separable_deconv(
        input: tensor,
        plane_filter: tensor,
        point_filter: tensor,
        bias: tensor = 0.0,
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [],
        groups: extent = 1 )
    -> ( output: tensor )
    {
        filtered = deconv(input, point_filter, groups = groups)
        output = planewise_deconv(filtered, plane_filter, bias, border = border, padding = padding, stride = stride, dilation = dilation)
    }


    # normalization operations

    fragment local_response_normalization(
        input: tensor,
        size: extent[],
        alpha: scalar = 1.0,
        beta: scalar = 0.5,
        bias: scalar = 1.0 )
    -> ( output: tensor )
    {
        sigma = bias + alpha * box(sqr(input), size = size, normalize = true)
        output = input / (sigma ^ beta)
    }

    fragment local_mean_normalization( input: tensor, size: extent[] ) -> ( output: tensor )
    {
        mean = box(input, size = size, normalize = true)
        output = sub(input, mean)
    }

    fragment local_variance_normalization( input: tensor, size: extent[], bias: scalar = 0.0 ) -> ( output: tensor )
    {
        sigma = box(sqr(input), size = size, normalize = true)
        output = input / sqrt(sigma + bias)
    }

    fragment local_contrast_normalization( input: tensor, size: extent[], bias: scalar = 0.0 ) -> ( output: tensor )
    {
        centered = local_mean_normalization(input, size = size)
        output = local_variance_normalization(centered, size = size, bias = bias)
    }

    fragment l1_normalization( input: tensor, axes: extent[], bias: scalar = 0.0 ) -> ( output: tensor )
    {
        sigma = sum_reduce(abs(input), axes = axes)
        output = input / (sigma + bias)
    }

    fragment l2_normalization( input: tensor, axes: extent[], bias: scalar = 0.0 ) -> ( output: tensor )
    {
        sigma = sum_reduce(sqr(input), axes = axes)
        output = input / sqrt(sigma + bias)
    }

    fragment layer_normalization( input: tensor, axes: extent[], bias: scalar = 0.0 ) -> ( output: tensor )
    {
        mean, variance = moments(input, axes = axes)
        output = (input - mean) / (sqrt(variance + bias))
    }

    fragment divisive_normalization( input: tensor, size: extent[], bias: scalar = 0.0 ) -> ( output: tensor )
    {
        mean = mean_reduce(avg_pool(input, size = [1,1] + size), axes = [1])
        centered = input - mean
        sigma = mean_reduce(avg_pool(sqr(centered), size = [1,1] + size), axes = [1])
        output = centered / sqrt(sigma + bias)
    }

    fragment batch_normalization( input: tensor, mean: tensor, variance: tensor, offset: tensor, scale: tensor, epsilon: scalar )
    -> ( output: tensor )
    {
        output = offset + scale * (input - mean) / sqrt(variance + epsilon)
    }


    # quantization operations

    fragment linear_quantize( x: tensor, min: tensor, max: tensor, bits: extent ) -> ( y: tensor )
    {
        z = clamp(x, min, max)
        r = scalar(2 ^ bits - 1) / (max - min)
        y = round((z - min) * r) / r + min
    }

    fragment logarithmic_quantize( x: tensor, max: tensor, bits: extent ) -> ( y: tensor )
    {
        amax = 2.0 ^ (ceil(log2(max)))
        amin = 2.0 ^ (log2(amax) - scalar(bits))
        z = clamp(x, amin, amax)
        y = 2.0 ^ round(log2(z / amin))
    }

    fragment binary_quantize(
        x: tensor,
        threshold: tensor = 0.0,
        negative_value: tensor = -1.0,
        positive_value: tensor = 1.0 )
    -> ( y: tensor )
    {
        y = select(x < threshold, negative_value, positive_value)
    }

    fragment ternary_quantize(
        x: tensor,
        low_threshold: tensor,
        high_threshold: tensor,
        negative_value: tensor = -1.0,
        positive_value: tensor = 1.0,
        zero_value: tensor = 0.0 )
    -> ( y: tensor )
    {
        y = select(x < low_threshold, negative_value, select(x > high_threshold, positive_value, zero_value))
    }


    # misc operations

    fragment copy_n( x: tensor, times: extent ) -> ( y: tensor[] )
    {
        y = [x] * times
    }

    fragment add_n( x: tensor[] ) -> ( y: tensor )
    {
        y = x[0] + add_n(x[1:]) if length_of(x) > 0 else 0.0
    }


    )STDLIB";


    template<typename T> struct stdlib_layers { static const char* text; };
    template<typename T> const char* stdlib_layers<T>::text = R"LAYERS(


    # linear_layer layers

    fragment linear_layer(
        input: tensor,
        channels: extent,
        use_bias: logical = true,
        scope: string )
    -> ( output: tensor )
    {
        filter = variable(label = scope + '/filter', shape = [channels, shape_of(input)[1]])
        bias = variable(label = scope + '/bias', shape = [1, channels]) if use_bias else 0.0

        output = linear(input, filter, bias)
    }

    fragment conv_layer(
        input: tensor,
        channels: extent,
        size: extent[],
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [],
        groups: extent = 1,
        use_bias: logical = true,
        scope: string )
    -> ( output: tensor )
    {
        filter = variable(label = scope + '/filter', shape = [channels, shape_of(input)[1] / groups] + size)
        bias = variable(label = scope + '/bias', shape = [1, channels]) if use_bias else 0.0

        output = conv(input, filter, bias, border = border, padding = padding, stride = stride, dilation = dilation, groups = groups)
    }

    fragment deconv_layer(
        input: tensor,
        channels: extent,
        size: extent[],
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [],
        groups: extent = 1,
        use_bias: logical = true,
        scope: string )
    -> ( output: tensor )
    {
        filter = variable(label = scope + '/filter', shape = [channels, shape_of(input)[1] / groups] + size)
        bias = variable(label = scope + '/bias', shape = [1, channels]) if use_bias else 0.0

        output = deconv(input, filter, bias, border = border, padding = padding, stride = stride, dilation = dilation, groups = groups)
    }


    # pooling layers

    fragment max_pool_layer(
        input: tensor,
        size: extent[],
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [] )
    -> ( output: tensor )
    {
        output = max_pool(input, size = [1,1] + size,
                          border = border, padding = [(0,0), (0,0)] + padding,
                          stride = [1,1] + stride, dilation = [1,1] + dilation)
    }

    fragment avg_pool_layer(
        input: tensor,
        size: extent[],
        border: string = 'constant',
        padding: (extent,extent)[] = [],
        stride: extent[] = [],
        dilation: extent[] = [] )
    -> ( output: tensor )
    {
        output = avg_pool(input, size = [1,1] + size,
                          border = border, padding = [(0,0), (0,0)] + padding,
                          stride = [1,1] + stride, dilation = [1,1] + dilation)
    }


    # normalization layers

    fragment batch_normalization_layer(
        input: tensor,
        center: logical = true,
        scale: logical = true,
        epsilon: scalar,
        scope: string )
    -> ( output: tensor )
    {
        shape = [1, shape_of(input)[1]]

        gamma = variable(label = scope + '/gamma', shape = shape) if scale else 1.0
        beta = variable(label = scope + '/beta', shape = shape) if center else 0.0

        mean = variable(label = scope + '/mean', shape = shape)
        variance = variable(label = scope + '/variance', shape = shape)

        output = batch_normalization(input, mean, variance, beta, gamma,
                                     epsilon = epsilon)
    }


    )LAYERS";

}   // namespace nnef

#endif
