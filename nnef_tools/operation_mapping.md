# TensorFlow

The following table lists the correspondence between operations in TensorFlow and NNEF.

| TensorFlow | NNEF
| --- | ---
| tf.Variable | variable
| tf.get_variable | variable
| tf.placeholder | external
| tf.constant | constant
| tf.zeros | constant
| tf.ones | constant
| tf.zeros_like | constant
| tf.ones_like | constant
| tf.concat | concat
| tf.split | split
| tf.stack | stack
| tf.unstack | unstack
| tf.reshape | reshape
| tf.squeeze | squeeze
| tf.expand_dims | unsqueeze
| tf.transpose | transpose
| tf.add | add
| tf.subtract | sub
| tf.multiply | mul
| tf.divide | div
| tf.pow | pow
| tf.logical_and | and
| tf.logical_or | or
| tf.logical_not | not
| tf.negative | neg
| tf.identity | copy
| tf.abs | abs
| tf.sign | sign
| tf.exp | exp
| tf.log | log
| tf.sqrt | sqrt
| tf.rsqrt | rsqrt
| tf.square | sqr
| tf.floor | floor
| tf.ceil | ceil
| tf.round | round
| tf.where | select
| tf.greater | gt
| tf.greater_equal | ge
| tf.less | lt
| tf.less_equal | le
| tf.equal | eq
| tf.not_equal | ne
| tf.minimum | min
| tf.maximum | max
| tf.assign | update
| tf.reduce_sum | sum_reduce
| tf.reduce_mean | mean_reduce
| tf.reduce_max | max_reduce
| tf.argmax | argmax_reduce
| tf.matmul | matmul
| tf.add_n | add_n
| tf.sigmoid | sigmoid
| tf.nn.sigmoid | sigmoid
| tf.tanh | tanh
| tf.nn.tanh | tanh
| tf.nn.elu | elu
| tf.nn.relu | relu
| tf.nn.softsign | softsign
| tf.nn.softplus | softplus
| tf.nn.conv1d | conv
| tf.nn.conv2d | conv
| tf.nn.conv3d | conv
| tf.nn.convolution | conv
| tf.nn.conv2d_transpose | deconv
| tf.nn.conv3d_transpose | deconv
| tf.nn.depthwise_conv2d | conv
| tf.nn.depthwise_conv2d_native | conv
| tf.nn.separable_conv2d | separable_conv
| tf.nn.max_pool | max_pool
| tf.nn.max_pool_with_argmax | max_pool_with_index
| tf.nn.avg_pool | avg_pool
| tf.nn.bias_add | add
| tf.nn.lrn | local_response_normalization
| tf.nn.local_response_normalization | local_response_normalization
| tf.nn.batch_normalization | batch_normalization
| tf.nn.fused_batch_norm | batch_normalization
| tf.nn.l2_normalize | l2_normalization
| tf.nn.softmax | softmax
| tf.nn.moments | moments
| tf.image.resize_images | multilinear_upsample
|                        | nearest_upsample
|                        | nearest_downsample
|                        | area_downsample
| tf.image.resize_bilinear | multilinear_upsample
| tf.image.resize_nearest_neighbor | nearest_upsample
|                                  | nearest_downsample
| tf.image.resize_area | area_downsample


# Caffe
The following table lists the correspondence between operations in Caffe and NNEF.

| Caffe | NNEF | Notes
| --- | --- | ---
| Input | external
| Convolution | conv
| Pooling | max_pool | if pool == MAX and not global_pooling
|         | avg_pool | if pool == AVE and not global_pooling
|         | max_reduce | if pool == MAX and global_pooling
|         | mean_reduce | if pool == AVE and global_pooling
| Crop | slice
| Deconvolution | multilinear_upsample | if weight_filler.type == 'bilinear' and depth-wise
|               | deconv | otherwise
| InnerProduct | linear | if bias_term and not transpose and axis == 1
|              | add(matmul) | if bias_term and (transpose or axis != 1)
|              | matmul | if not bias_term
|              | | + reshape if axis != input-rank - 1 <br> + unsqueeze if axis == 0
| Dropout | | skipped in inference
| LRN | local_response_normalization
| MVN | local_contrast_normalization | if normalize_variance 
|     | local_mean_normalization | if not normalize_variance
| BatchNorm | batch_normalization | scale factor merged into mean and variance if not 1 <br> merged with following scale layer if any
| ReLU | relu | if negative_slope == 0
|      | leaky_relu | if negative_slope != 0
| PReLU | prelu
| ELU | elu | if alpha == 1
|     | select(x > 0.0, x, alpha * (exp(x) - 1.0)) | if alpha != 1
| Sigmoid | sigmoid
| TanH | tanh
| AbsVal | abs
| Power(a, b, n) | pow(a * x + b, n) | '*' or '+' omitted if the corresponding parameter is 1 or 0
| Exp(base, a, b) | exp(a * x + b)       | if base == -1
|                 | pow(base, a * x + b) | if base != -1
|                 |                      | '*' or '+' omitted if the corresponding parameter is 1 or 0
| Log(base, a, b) | log(a * x + b)             | if base == -1
|                 | log2(a * x + b)            | if base == 2
|                 | log(a * x + b) / log(base) | otherwise
|                 |                            | '*' or '+' omitted if the corresponding parameter is 1 or 0
| BNLL | softplus
| Threshold(x, t) | select(x > t, 1.0, 0.0)
| Bias(x) | add(x, weight) | + unsqueeze if axis > 0
| Scale(x) | mul(x + bias, weight) | '+' omitted if no bias_term <br> + unsqueeze if axis > 0  
| Flatten | reshape
| Reshape | reshape
| Split | copy_n
| Concat | concat
| Slice | split
| Eltwise | x_1 * ... * x_n | if operation == PROD
|         | x_1 + ... + x_n | if operation == SUM and coeff == []
|         | coeff_1 * x_1 + ... + coeff_n * x_n | if operation == SUM and coeff != []
|         | max(x_1, ... , x_n) | if operation == MAX
| Reduction | squeeze(sum_reduce) * coeff | if operation == SUM
|           | squeeze(sum_reduce(abs)) * coeff | if operation == ASUM
|           | squeeze(sum_reduce(sqr)) * coeff | if operation == SUMSQ
|           | squeeze(mean_reduce) * coeff | if operation == MEAN
|           |                              | '*' omitted if coeff == 1
| Silence | | skipped in inference
| ArgMax | argmax_reduce | if top_k == 1 and out_max_val == false
| Softmax | softmax

# ONNX

The following table lists the correspondence between operations in ONNX and NNEF.

| ONNX | NNEF | Notes
| --- | --- | ---
| Abs | abs
| Acos | -
| Acosh | -
| Add | add
| And | and
| ArgMax | argmax_reduce
| ArgMin | argmin_reduce
| Asin | -
| Asinh | -
| Atan | -
| Atanh | -
| AveragePool | avg_pool
| BatchNormalization | batch_normalization
| Cast | select | logical to integer/scalar
|      | ne | integer/scalar to logical
| Ceil | ceil
| Clip | clamp
| Compress | -
| Concat | concat
| Constant | constant
| ConstantOfShape | constant
| Conv | conv
| ConvTranspose | deconv
| Cos | -
| Cosh | -
| DepthToSpace | reshape(transpose(reshape))
| Div | div
| Dropout | copy
| Elu | elu | + arithmetic when alpha != 1.0
| Equal | eq
| Erf | -
| Exp | exp
| Expand | add(constant(0)) | workaround
| EyeLike | -
| Flatten | reshape
| Floor | floor
| GRU | -
| Gather | -
| Gemm | matmul
| GlobalAveragePool | mean_reduce
| GlobalLpPool | sum_reduce(abs) | if p == 1
|              | sqrt(sum_reduce(sqr)) | if p == 2
| GlobalMaxPool | max_reduce
| Greater | gt
| HardSigmoid | clamp(add(mul))
| HardMax | -
| Identity | copy
| If | -
| InstanceNormalization | div(moments) | + further arithmetic
| IsNan | -
| LRN | local_response_normalization
| LSTM | -
| LeakyRelu | leaky_relu
| Less | lt
| Log | log
| LogSoftmax | log(softmax)
| Loop | -
| LpNormalization | l1_normalization | if p == 1
|                 | l2_normalization | if p == 2
| LpPool | avg_pool(abs) | if p == 1
|        | sqrt(avg_pool(sqr)) | if p == 2
| MatMul | matmul
| Max | max
| MaxPool | max_pool
| MaxRoiPool | max_roi_pool
| MaxUnpool | desample
| Mean | div(add)
| Min | min
| Mul | mul
| Multinomial | -
| Neg | neg
| Not | not
| OneHot | -
| Or | or
| PRelu | prelu
| Pad | box | workaround
| Pow | pow
| RNN | -
| RandomNormal | -
| RandomNormalLike | -
| RandomUniform | -
| RandomUniformLike | -
| Reciprocal | rcp
| ReduceL1 | sum_reduce(abs)
| ReduceL2 | sqrt(sum_reduce(sqr))
| ReduceLogSum | log(sum_reduce)
| ReduceLogSumExp | log(sum_reduce(exp))
| ReduceMax | max_reduce
| ReduceMean | mean_reduce
| ReduceMin | min_reduce
| ReduceProd | -
| ReduceSum | sum_reduce
| ReduceSumSquare | sum_reduce(sqr)
| Relu | relu
| Reshape | reshape
| Scan | -
| Scatter | -
| Selu | -
| Shape | constant | if can be evaluated
| Shrink | -
| Sigmoid | sigmoid
| Sign | sign
| Sin | -
| Sinh | -
| Size | constant | if can be evaluated
| Slice | slice
| Softmax | softmax
| Softplus | softplus
| Softsign | -
| SpaceToDepth | -
| Split | split
| Sqrt | sqrt
| Squeeze | squeeze
| Sub | sub
| Sum | add
| Tan | -
| Tanh | tanh
| Tile | -
| TopK | -
| Transpose | transpose
| Unsqueeze | unsqueeze
| Upsample | multilinear_upsample
|          | nearest_upsample
| Where | select
| Xor | or(and(x,not(y)),and(y,not(x)))
