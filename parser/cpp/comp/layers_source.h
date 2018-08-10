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

#ifndef _NNEF_LAYERS_SOURCE_H_
#define _NNEF_LAYERS_SOURCE_H_


namespace nnef {

    template<typename T> struct _layers_source { static const char* text; };
    template<typename T> const char* _layers_source<T>::text = R"LAYERS(


    # linear_layer layers

    fragment linear_layer(
        input: tensor<scalar>,
        channels: integer,
        use_bias: logical = true,
        scope: string )
    -> ( output: tensor<scalar> )
    {
        filter = variable(label = scope + '/filter', shape = [channels, shape_of(input)[1]]);
        bias = variable(label = scope + '/bias', shape = [1, channels]) if use_bias else constant(shape = [1], value = [0.0]);

        output = linear(input, filter, bias);
    }

    fragment conv_layer(
        input: tensor<scalar>,
        channels: integer,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [],
        groups: integer = 1,
        use_bias: logical = true,
        scope: string )
    -> ( output: tensor<scalar> )
    {
        planes = shape_of(input)[1] / groups if groups != 0 else 1;
        filter = variable(label = scope + '/filter', shape = [channels, planes] + size);
        bias = variable(label = scope + '/bias', shape = [1, channels]) if use_bias else constant(shape = [1], value = [0.0]);

        output = conv(input, filter, bias, border = border, padding = padding, stride = stride, dilation = dilation, groups = groups);
    }

    fragment deconv_layer(
        input: tensor<scalar>,
        channels: integer,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [],
        output_shape: integer[] = [],
        groups: integer = 1,
        use_bias: logical = true,
        scope: string )
    -> ( output: tensor<scalar> )
    {
        planes = shape_of(input)[1] / groups if groups != 0 else 1;
        filter = variable(label = scope + '/filter', shape = [channels, planes] + size);
        bias = variable(label = scope + '/bias', shape = [1, channels]) if use_bias else constant(shape = [1], value = [0.0]);

        output = deconv(input, filter, bias, border = border, padding = padding, stride = stride,
                        dilation = dilation, output_shape = output_shape, groups = groups);
    }


    # pooling layers

    fragment max_pool_layer(
        input: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( output: tensor<scalar> )
    {
        output = max_pool(input, size = [1,1] + size, border = border,
                          padding = [(0,0), (0,0)] + padding if length_of(padding) != 0 else [],
                          stride = [1,1] + stride if length_of(stride) != 0 else [],
                          dilation = [1,1] + dilation if length_of(dilation) != 0 else []);
    }

    fragment avg_pool_layer(
        input: tensor<scalar>,
        size: integer[],
        border: string = 'constant',
        padding: (integer,integer)[] = [],
        stride: integer[] = [],
        dilation: integer[] = [] )
    -> ( output: tensor<scalar> )
    {
        output = avg_pool(input, size = [1,1] + size, border = border,
                          padding = [(0,0), (0,0)] + padding if length_of(padding) != 0 else [],
                          stride = [1,1] + stride if length_of(stride) != 0 else [],
                          dilation = [1,1] + dilation if length_of(dilation) != 0 else []);
    }


    # normalization layers

    fragment batch_normalization_layer(
        input: tensor<scalar>,
        center: logical = true,
        scale: logical = true,
        epsilon: scalar,
        scope: string )
    -> ( output: tensor<scalar> )
    {
        shape = [1, shape_of(input)[1]];

        gamma = variable(label = scope + '/gamma', shape = shape) if scale else constant(shape = [1], value = [1.0]);
        beta = variable(label = scope + '/beta', shape = shape) if center else constant(shape = [1], value = [0.0]);

        mean = variable(label = scope + '/mean', shape = shape);
        variance = variable(label = scope + '/variance', shape = shape);

        output = batch_normalization(input, mean, variance, beta, gamma,
                                     epsilon = epsilon);
    }


    )LAYERS";


    inline const char* layers_source()
    {
        return _layers_source<void>::text;
    }

}   // namespace nnef

#endif

