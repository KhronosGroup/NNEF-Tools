# NNEF Runner using Pytorch NNEF Backend

## Examples

For detailed usage, please see the `--help`.

How to run on random input:

```
./nnef_tools/run.py --input-model bvlc_googlenet.nnef
```

Other examples:

The `--stats` flag enables the generation of a graph.stats file, 
containing the statistics of all tensors in the graph, using the given inputs.

The `--permissive` flag allows the graph to run even when some of its operations are only imprecisely supported.

The `--resize` flag tries to make the graph batch-size agnostic. 
This is useful when you run the network with different batch size compared to the declared input shape.

The `--activations` flag enables the export of activation tensors.

The `--generate-weights` flag enables the generation of weights if the graph does not have them already.

```
./nnef_tools/run.py --input-model nnef_models/mobilenet_v1_1.0.tfpb.nnef  \
                        --input "Image('~/Documents/imagenet5/*.jpg', data_format=NHWC, range=[0, 255])" \
                        --stats 
              
./nnef_tools/run.py --input-model nnef_models/bvlc_googlenet.onnx.nnef \
                        --input "Image('~/Documents/imagenet5/*.jpg', data_format=NHWC, range=[0, 1])" \
                        --stats
              
./nnef_tools/run.py --input-model ~/Documents/neumann.nnef \
                        --input "Image('~/Documents/imagenet5/*.jpg', \
                                       data_format=NCHW, \
                                       norm=[[103.52999877929688, 116.27999877929688, 123.67500305175781], \
                                             [57.375, 57.119998931884766, 58.39500045776367]])" \
                        --stats \
                        --permissive \
                        --resize \
                        --custom-operations custom.torch_backend_my_company
              
./nnef_tools/run.py --input-model squeezenet.nnef \  
                        --generate-weights \
                        --activations \
                        --stats
``` 

## Implementation status

Limitations like maximum kernel-size, padding, stride, dilation that come from pytorch are not listed.

Used terms:
* Suppressible: When --permissive is specified, the operation can run, but it will be imprecise.
  E.g. 'reflect' border instead of 'reflect-even'.
* Active dimensions: Dimensions that are being pooled or normalized over.
  Technically: All dimensions that have (kernel) size != 1, padding != (0, 0), stride != 1 or dilation != 1.

|  NNEF operation                 | Supported | Limitations |
|:-------------------------------:|:---------:|:-----------:|
| external                        | YES       |             |
| variable                        | YES       |             |
| constant                        | YES       |             |
| update                          | YES       | Returns the updated value, but does not update anything. |
| reshape                         | YES       |             |
| transpose                       | YES       |             |
| concat                          | YES       |             |
| split                           | YES       |             |
| slice                           | YES       |             |
| squeeze                         | YES       |             |
| unsqueeze                       | YES       |             |
| stack                           | YES       |             |
| unstack                         | YES       |             |
| add                             | YES       |             |
| sub                             | YES       |             |
| mul                             | YES       |             |
| div                             | YES       |             |
| pow                             | YES       |             |
| exp                             | YES       |             |
| log                             | YES       |             |
| abs                             | YES       |             |
| sign                            | YES       |             |
| rcp                             | YES       |             |
| neg                             | YES       |             |
| copy                            | YES       |             |
| lt                              | YES       |             |
| gt                              | YES       |             |
| le                              | YES       |             |
| ge                              | YES       |             |
| eq                              | YES       |             |
| ne                              | YES       |             |
| and                             | YES       |             |
| or                              | YES       |             |
| not                             | YES       |             |
| floor                           | YES       |             |
| ceil                            | YES       |             |
| round                           | YES       |             |
| select                          | YES       |             |
| sqr                             | YES       |             |
| sqrt                            | YES       |             |
| rsqr                            | YES       |             |
| rsqrt                           | YES       |             |
| log2                            | YES       |             |
| min                             | YES       |             |
| max                             | YES       |             |
| clamp                           | YES       |             |
| matmul                          | YES       |             |
| conv                            | YES       | Only for 3 - 5D tensors. Border='reflect-even' is unsupported (suppressible). |
| deconv                          | YES       | Only for 3 - 5D tensors. Only border='constant' is supported (suppressible). |
| box                             | YES       | Only for 0 - 3 active dimensions. Dilation not supported. Border='reflect-even' is unsupported (suppressible). |
| debox                           | YES       | Only for 3 - 5D tensors, dimensions 0 - 1 can not be active. Only border='constant' and border='ignore' is supported (they do the same, suppressible). |
| argmax_pool                     | YES       | Only for 3 - 5D tensors, dimensions 0 - 1 can not be active. Border='reflect-even' is unsupported (suppressible). |
| sample                          | NO        |             |
| desample                        | YES       | See argmax_pool. |
| nearest_downsample              | YES       | Only for 3 - 5D tensors. |
| area_downsample                 | YES       | Only for 3 - 5D tensors. |
| nearest_upsample                | YES       | Only for 3 - 5D tensors. |
| multilinear_upsample            | YES       | Only the listed methods and borders are supported (suppressible). |
|                                 |           | For 3 - 5D tensors: ('symmetric', 'replicate'), ('aligned', \[anything\]) |
|                                 |           | For 3 - 4D tensors and factor = 2: ('symmetric', 'constant'), ('asymmetric', 'constant'), ('asymmetric', 'replicate'). |
| sum_reduce                      | YES       |             |
| max_reduce                      | YES       |             |
| min_reduce                      | YES       |             |
| argmax_reduce                   | YES       | Only for consecutive axes. |
| argmin_reduce                   | YES       | Only for consecutive axes. |
| mean_reduce                     | YES       |             |
| moments                         | YES       |             |
| relu                            | YES       |             |
| sigmoid                         | YES       |             |
| tanh                            | YES       |             |
| softabs                         | YES       |             |
| softmax                         | YES       |             |
| softplus                        | YES       |             |
| elu                             | YES       |             |
| prelu                           | YES       |             |
| leaky_relu                      | YES       |             |
| max_pool_with_index             | YES       | See argmax_pool. |
| max_pool                        | YES       | Only for 0 - 3 active dimensions. Border='reflect-even' is unsupported (suppressible). |
| avg_pool                        | YES       | See box (normalized mode). |
| rms_pool                        | YES       | See box (normalized mode). |
| linear                          | YES       |             |
| separable_conv                  | YES       | See conv.   |
| separable_deconv                | YES       | See deconv.   |
| local_response_normalization    | YES       | Only for 0 - 3 active dimensions. |
| local_mean_normalization        | YES       | Only for 0 - 3 active dimensions. |
| local_variance_normalization    | YES       | Only for 0 - 3 active dimensions. |
| local_contrast_normalization    | YES       | Only for 0 - 3 active dimensions. |
| l1_normalization                | YES       |             |
| l2_normalization                | YES       |             |
| batch_normalization             | YES       |             |
| avg_roi_pool                    | NO        |             |
| max_roi_pool                    | NO        |             |
| roi_resample                    | NO        |             |
| avg_roi_align                   | NO        |             |
| max_roi_align                   | NO        |             |
| linear_quantize                 | YES       |             |
| logarithmic_quantize            | YES       |             |
| copy_n                          | YES       |             |
| add_n                           | YES       |             |
| sin                             | YES       |             |
| cos                             | YES       |             |
| tile                            | YES       |             |
| pad                             | YES       |             |
| any_reduce                      | YES       |             |
| all_reduce                      | YES       |             |
