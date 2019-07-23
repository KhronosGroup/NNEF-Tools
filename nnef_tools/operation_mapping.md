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
| tf.sin | sin
| tf.cos | cos
| tf.pad | pad
| tf.tile | tile
| tf.reduce_any | any_reduce
| tf.reduce_all | all_reduce

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

# Caffe2

The following tables show the correspondence between operations in Caffe2 and NNEF.

All operations without outputs (e.g. Assert) are stripped from the graph.

Only the NCHW version of the operations are supported (as opposed to NHWC). 

**Normal operations:**

| Caffe2 | NNEF | Notes
| --- | --- | ---
| Abs | abs
| Add | add | + unsqueeze if axis != 0
| And | and | + unsqueeze if axis != 0
| ArgMax | argmax_reduce | + squeeze if not keepdims
| ArgMin | argmin_reduce | + squeeze if not keepdims
| AveragePool <br> AveragePool1D <br> AveragePool2D <br> AveragePool3D | avg_pool | if not global_pooling
| | mean_reduce | if global_pooling
| BatchMatMul | matmul | + reshape if input ranks are not equal or less than 2
| Cast | select | logical to scalar or integer
|      | ne | scalar to logical
|      | copy | cast to same type, may be optimized away
| Ceil | ceil
| ChannelShuffle | reshape(transpose(reshape))
| Clip | clamp | if both min and max is given
| | min | if only max is given
| | max | if only min is given
| | copy | if neither min nor max is given, may be optimized away
| Concat <br> DepthConcat <br> Append | concat
| Conditional | select
| Conv <br> Conv1D <br> Conv2D <br> Conv3D | conv
| ConvTranspose | deconv
| Copy <br> CopyFromCPUInput <br> CopyOnDeviceLike <br> EnsureCPUOutput <br> StopGradient | copy | may be optimized away
| Cos | cos
| Div | div | + unsqueeze if axis != 0
| DotProduct | mul | + sum_reduce + squeeze if input-rank > 1
| Dropout | copy | may be optimized away
| ElementwiseLinear(x, w, b) | x * w + b | + reshape if X.rank != 2 or axis != 1
| EQ | eq | + unsqueeze if axis != 0
| FC | linear | + reshape if X.rank != 2 or W.rank != 2 or axis != 1 or axis_w != 1
| FCTransposed | add(matmul) | + reshape if X.rank != 2 or W.rank != 2 or axis != 1 or axis_w != 1
| Flatten | reshape
| FlattenToVec | reshape
| Floor | floor
| GE | ge | + unsqueeze if axis != 0
| GT | gt | + unsqueeze if axis != 0
| L1Distance(a, b) | abs(a-b) | + sum_reduce + squeeze if input-rank > 1
| LE | le | + unsqueeze if axis != 0
| LT | lt | + unsqueeze if axis != 0
| LayerNorm(x) | mean_, std_ = moments(x);<br> y = (x - mean_) / sqrt(std_ + epsilon);<br> mean=squeeze(mean_);<br> std=squeeze(sqrt(std_ + epsilon))
| LeakyRelu | leaky_relu
| Log | log
| Logit(x) | x_ = clamp(x, eps, 1.0-eps);<br> y = log(x_ / (1 - x_))
| LpNorm | sum_reduce(abs) | if p = 1 and not average
| | mean_reduce(abs) | if p = 1 and average
| | sum_reduce(sqr) | if p = 2 and not average
| | mean_reduce(sqr) | if p = 2 and average
| | | + reshape if input-rank != 1
| LpPool(x) | pow(box(pow(abs(x), p)), 1.0/p)
| MatMul | matmul | + reshape if A.rank != 2 or B.rank != 2 or axis_a != 1 or axis_b != 1
| Max | max, \[max, ...\] | if input-count >= 2
| | copy | if input-count == 1, may be optimized away
| MaxPool <br> MaxPool1D <br> MaxPool2D <br> MaxPool3D | max_pool | if not global_pooling
| | max_reduce | if global_pooling
| MaxPoolWithIndex | max_pool_with_index | Caffe2 supports it only on GPU
| Mean | div(add_n) | if input-count >= 3
| | div(add) | if input-count == 2
| | copy | if input-count == 1, may be optimized away
| MergeDim | reshape
| Min | min, \[min, ...\] | if input-count >= 2
| | copy | if input-count == 1, may be optimized away
| Mul | mul | + unsqueeze if axis != 0
| NE | ne | + unsqueeze if axis != 0
| Negative | neg
| Normalize | l2_normalization
| NormalizeL1 | l1_normalization
| Not | not
| Or | or | + unsqueeze if axis != 0
| PadImage | pad
| PRelu | prelu
| Pow | pow | + unsqueeze if axis != 0
| PrependDim | reshape
| ReduceMin | min_reduce | + squeeze if not keepdims
| ReduceMax <br> ReduceFrontMax <br> ReduceBackMax <br> ColwiseMax <br> RowwiseMax | max_reduce | + squeeze if not keepdims
| ReduceSum <br> ReduceFrontSum <br> ReduceBackSum <br> ReduceTailSum <br> SumElements | sum_reduce | + squeeze if not keepdims
| ReduceMean <br> ReduceFrontMean <br> ReduceBackMean | mean_reduce | + squeeze if not keepdims
| ReduceL1 | sum_reduce(abs) | + squeeze if not keepdims
| ReduceL2 | sqrt(sum_reduce(sqr)) | + squeeze if not keepdims
| Relu | relu
| Reshape | reshape | if shape parameter is constant or the result of Shape or the 2nd result of Reshape (old_shape)
| ResizeLike | reshape
| ResizeNearest | nearest_upsample | if upsample (or same size) in both dimensions
| | nearest_downsample | if downsample (or same size) in both dimensions
| | nearest_upsample(nearest_downsample) | if downsample in one dimension and upsample in the other
| | copy | if output-shape = input-shape, may be optimized away
| Scale | mul
| RowMul(x, w) | mul | + reshape if w.rank != 1
| Selu(x, alpha, scale) | select(x > 0, x, exp(x) * alpha - alpha) * scale 
| Sigmoid | sigmoid
| Sign | sign
| Sin | sin
| Slice | slice
| Softsign(x) | x / (abs(x) + 1)
| Split | split | if split parameter is constant or the 2nd result of Concat (split_info)
| Sqr | sqr
| Sqrt | sqrt
| SquaredL2Distance | (x - y) ^ 2 / 2 | + sum_reduce + squeeze if input-rank > 1
| Squeeze | squeeze
| StumpFunc(x, threshold, low_value, high_value) | select(x > threshold, high_value, low_value)
| Sub | sub | + unsqueeze if axis != 0
| Sum | add_n | if input-count >= 3
| | add | if input-count == 2
| | copy | if input-count == 1, may be optimized away
| SumSqrElements | squeeze(sum_reduce(sqr)) | if not average
| | squeeze(mean_reduce(sqr)) | if average
| SumReduceLike | sum_reduce and/or squeeze | if output-shape != input-shape
| | copy | if output-shape = input-shape, may be optimized away
| Summarize(x) | min_ = min_reduce(x);<br> max_ = max_reduce(x);<br> mean_, std_ = moments(x);<br>min = reshape(min_);<br> max = reshape(max_);<br>mean = reshape(mean_);<br> std = reshape(std_);<br>y = concat(\[min, max, mean, sqrt(std * N / (N - 1))\]) | where N = count of x
| Swish(x) | x / (1 + exp(-x))
| Tanh | tanh
| ThresholdedRelu | select(x > alpha, x, 0.0)
| Tile | tile
| Transpose <br> NCHW2NHWC <br> NHWC2NCHW | transpose
| Where | select
| Xor(x, y) | or(and(x, not(y)), and(y, not(x))) | + unsqueeze if axis != 0

**Constants:** These operations/tensors are converted to constants.

| Caffe2 | NNEF | Notes
| --- | --- | ---
| Shape | constant\<integer\> | Can be used as Reshape's 2nd input (shape)
| Size | constant\<integer\>
| Reshape's 2nd output (old_shape) | constant\<integer\> | Can be used as Reshape's 2nd input (shape) 
| Concat's 2nd output (split_info) | constant\<integer\> | Can be used as Split's 2nd input (split)
| Range | constant\<scalar\>

**Variables:** These operations (in the param initializer network) are converted to variable tensors.

| Caffe2 | NNEF | Representation
| --- | --- | ---
| GivenTensorFill | variable\<scalar\> | float32
| GivenTensorDoubleFill | variable\<scalar\> | float64
| GivenTensorInt16Fill | variable\<integer\> | int16
| GivenTensorIntFill | variable\<integer\> | int32
| GivenTensorInt64Fill | variable\<integer\> | int64
| GivenTensorBoolFill | variable\<logical\> | bool

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
| Cast | select | logical to scalar or integer
|      | ne | scalar to logical
|      | copy | to same type
| Ceil | ceil
| Clip | clamp
| Compress | -
| Concat | concat
| Constant | constant
| ConstantOfShape | constant
| Conv | conv
| ConvTranspose | deconv
| Cos | cos
| Cosh | -
| DepthToSpace | reshape(transpose(reshape))
| Div | div
| Dropout | copy
| Elu | elu | + arithmetic when alpha != 1.0
| Equal | eq
| Erf | -
| Exp | exp
| Expand | tile
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
| Pad | pad
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
| Sin | sin
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
| Tile | tile
| TopK | -
| Transpose | transpose
| Unsqueeze | unsqueeze
| Upsample | multilinear_upsample
|          | nearest_upsample
| Where | select
| Xor | or(and(x,not(y)),and(y,not(x)))
