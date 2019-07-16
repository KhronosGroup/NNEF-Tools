NNEF model zoo
==============

The following collection of models were compiled by running the converter tools in this repository on publicly available models. Each entry provides a link to the original and the converted model.

* TensorFlow models have been acquired from [https://www.tensorflow.org/lite/guide/hosted_models]
* ONNX models have been acquired from [https://github.com/onnx/models]
* Caffe models have been acquired from [https://github.com/BVLC/caffe/wiki/Model-Zoo]
* Caffe2 models have been acquired from [https://github.com/caffe2/models]


AlexNet
-------

_Floating point models_

Name | Size | Original | Converted
--- | --- | --- | ---
BVLC AlexNet | 244 Mb | [Caffe](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/bvlc_alexnet.caffemodel.nnef.tgz)
BVLC AlexNet | 244 Mb | [ONNX](https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_alexnet.tar.gz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/bvlc_alexnet.onnx.nnef.tgz)


VGG
---

_Floating point models_

Name | Size | Original | Converted
--- | --- | --- | ---
VGG-16 | 553.6 MB Mb | [Caffe](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/vgg16.caffemodel.nnef.tgz)
VGG-19 | 574.8 MB Mb | [Caffe](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/vgg19.caffemodel.nnef.tgz)
VGG-16 | 527.8 MB Mb | [ONNX](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.onnx) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/vgg16.onnx.nnef.tgz)
VGG-19 | 548.1 MB Mb | [ONNX](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19/vgg19.onnx) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/vgg19.onnx.nnef.tgz)


GoogleNet
---------

_Floating point models_

Name | Size | Original | Converted
--- | --- | --- | ---
Inception v1 | 28 Mb | [Caffe2](https://github.com/caffe2/models/tree/master/inception_v1) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/inception_v1.caffe2.nnef.tgz)
Inception v1 | 28 Mb | [ONNX](https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v1.tar.gz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/inception_v1.onnx.nnef.tgz)
Inception v2 | 45 Mb | [Caffe2](https://github.com/caffe2/models/tree/master/inception_v2) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/inception_v2.caffe2.nnef.tgz)
Inception v2 | 45 Mb | [ONNX](https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v2.tar.gz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/inception_v2.onnx.nnef.tgz)
Inception v3 | 95.3 Mb | [TensorFlow](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/inception_v3.tfpb.nnef.tgz)
Inception v4 | 170.7 Mb | [TensorFlow](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/inception_v4.tfpb.nnef.tgz)
BVLC GoogleNet | 28 Mb | [Caffe](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/bvlc_googlenet.caffemodel.nnef.tgz)
BVLC GoogleNet | 28 Mb | [ONNX](https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_googlenet.tar.gz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/bvlc_googlenet.onnx.nnef.tgz)


_Quantized models_

Name | Size | Original | Converted
--- | --- | --- | ---
Inception v1 | 6.4 Mb | [TensorFlow-Lite](http://download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/inception_v1_quant.tflite.nnef.tgz)
Inception v2 | 11 Mb | [TensorFlow-Lite](http://download.tensorflow.org/models/inception_v2_224_quant_20181026.tgz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/inception_v2_quant.tflite.nnef.tgz)
Inception v3 | 23 Mb | [TensorFlow-Lite](http://download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/inception_v3_quant.tflite.nnef.tgz)
Inception v4 | 41 Mb | [TensorFlow-Lite](http://download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/inception_v4_quant.tflite.nnef.tgz)


ResNet
------

_Floating point models_

Name | Size | Original | Converted
--- | --- | --- | ---
Resnet v1-18 | 44.7 Mb | [ONNX](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.onnx) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v1_18.onnx.nnef.tgz)
Resnet v1-34 | 83.3 Mb | [ONNX](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v1/resnet34v1.onnx) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v1_34.onnx.nnef.tgz)
Resnet v1-50 | 97.8 Mb | [ONNX](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.onnx) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v1_50.onnx.nnef.tgz)
Resnet v1-101 | 170.6 MB Mb | [ONNX](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v1/resnet101v1.onnx) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v1_101.onnx.nnef.tgz)
Resnet v1-152 | 242.3 Mb | [Caffe](https://github.com/KaimingHe/deep-residual-networks) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v1_152.caffemodel.nnef.tgz)
Resnet v2-18 | 44.6 Mb | [ONNX](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.onnx) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v2_18.onnx.nnef.tgz)
Resnet v2-34 | 83.2 Mb | [ONNX](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v2/resnet34v2.onnx) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v2_34.onnx.nnef.tgz)
Resnet v2-50 | 97.7 Mb | [ONNX](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.onnx) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v2_50.onnx.nnef.tgz)
Resnet v2-101 | 170.4 Mb | [ONNX](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.onnx) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/resnet_v2_101.onnx.nnef.tgz)
Inception-Resnet v2 | 121 Mb | [TensorFlow](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/inception_resnet_v2.tfpb.nnef.tgz)


MobileNet
---------

_Floating point models_

Name | Size | Original | Converted
--- | --- | --- | ---
MobileNet v1-1.0 | 16.9 Mb | [TensorFlow](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/mobilenet_v1_1.0.tfpb.nnef.tgz)
MobileNet v1-1.0 | 17.2 Mb | [Caffe](https://github.com/shicai/MobileNet-Caffe) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/mobilenet_v1_1.0.caffemodel.nnef.tgz)
MobileNet v2-1.0 | 14.0 Mb | [TensorFlow](http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/mobilenet_v2_1.0.tfpb.nnef.tgz)
MobileNet v2-1.0 | 14.4 Mb | [Caffe](https://github.com/shicai/MobileNet-Caffe) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/mobilenet_v2_1.0.caffemodel.nnef.tgz)
MobileNet v2-1.0 | 13.6 Mb | [ONNX](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/mobilenet_v2_1.0.onnx.nnef.tgz)



_Quantized models_

Name | Size | Original | Converted
--- | --- | --- | ---
MobileNet v1-1.0 | 4.3 Mb | [TensorFlow-Lite](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/mobilenet_v1_1.0_quant.tflite.nnef.tgz)
MobileNet v2-1.0 | 3.4 Mb | [TensorFlow-Lite](http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/mobilenet_v2_1.0_quant.tflite.nnef.tgz)


SqueezeNet
----------

_Floating point models_

Name | Size | Original | Converted
--- | --- | --- | ---
SqueezeNet | 5.0 Mb | [TensorFlow](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/squeezenet.tfpb.nnef.tgz)
SqueezeNet 1.0 | 4.7 Mb | [ONNX](https://s3.amazonaws.com/download.onnx/models/opset_9/squeezenet.tar.gz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/squeezenet_v1.0.onnx.nnef.tgz)
SqueezeNet 1.1 | 4.7 Mb | [ONNX](https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.onnx) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/squeezenet_v1.1.onnx.nnef.tgz)
SqueezeNet 1.0 | 4.7 Mb | [Caffe](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.0) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/squeezenet_v1.0.caffemodel.nnef.tgz)
SqueezeNet 1.1 | 4.7 Mb | [Caffe](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/squeezenet_v1.1.caffemodel.nnef.tgz)


ShuffleNet
----------

_Floating point models_

Name | Size | Original | Converted
--- | --- | --- | ---
ShuffleNet | 5.3 Mb | [ONNX](https://s3.amazonaws.com/download.onnx/models/opset_9/shufflenet.tar.gz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/shufflenet.onnx.nnef.tgz)


NASNet
------

_Floating point models_

Name | Size | Original | Converted
--- | --- | --- | ---
NasNet mobile | 21.4 Mb | [TensorFlow](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz) | [NNEF](https://sfo2.digitaloceanspaces.com/nnef-public/nasnet_mobile.tfpb.nnef.tgz)
