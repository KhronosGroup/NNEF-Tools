# TensorFlow

The following table lists the correspondence between operations in TensorFlow and NNEF.

| TensorFlow | NNEF
| --- | ---
| tf.Variable | variable
| tf.get_variable | variable
| tf.placeholder | external
| tf.constant | constant
| tf.zeros | zeros
| tf.ones | ones
| tf.zeros_like | zeros_like
| tf.ones_like | ones_like
| tf.concat | concat
| tf.split | split
| tf.stack | stack
| tf.unstack | unstack
| tf.reshape | reshape
| tf.squeeze | squueze
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
| tf.no_op | copy
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

| Caffe | NNEF
| --- | ---
| Convolution | conv
| Deconvolution | deconv
| InnerProduct | linear
| Pooling | avg_pool, max_pool
| LRN | local_response_normalization
| MVN | local_contrast_normalization
| BatchNorm | batch_normalization
| ReLU | relu, leaky_relu
| ELU | elu
| Sigmoid | sigmoid
| TanH | tanh
| Threshold | max
| Scale | mul
| Bias | add
| Flatten | reshape
| Reshape | reshape
| Split | copy_n
| Slice | split
| Concat | concat
| Softmax | softmax
| BNLL | softplus
| Eltwise | mul, add
| Power(a,b,n) | pow(a * x + b, n)
| Exp(base,a,b) | pow(base, a * x + b)
| Log(base,a,b) | log(a * x + b) / log(base)
| Reduction | sum_reduce, mean_reduce
| ArgMax | argmax_reduce


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
| GlobalLpPool | sum_reduce(abs) | if p = 1
|              | sqrt(sum_reduce(sqr)) | if p = 2
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
| LpPool | avg_pool(abs) | if p = 1
|        | sqrt(avg_pool(sqr)) | if p = 2
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