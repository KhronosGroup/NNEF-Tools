# NNEF Tools

This is a repository of NNEF converters and other tools.

## Dependencies

Python2 version >= 2.7.12 and Python3 version >= 3.5.2 are supported.
Python3 is recommended.

For all tools an installed `nnef` package is needed:

```
cd parser/python
python setup.py install
cd ../..
```
You have to install dependencies only for the functionalities that you are using:

| Functionality                                 | Dependencies                                                        |
|-----------------------------------------------|---------------------------------------------------------------------|
|  TensorFlow Python code conversion (+tests)   | pip install typing six numpy "tensorflow-gpu<1.14
|  TensorFlow Protobuf conversion               | pip install typing six numpy protobuf
|  TensorFlow Protobuf conversion tests         | pip install typing six numpy protobuf "tensorflow-gpu<1.14"
|  TensorFlow Lite conversion                   | pip install typing six numpy flatbuffers
|  TensorFlow Lite conversion tests             | pip install typing six numpy flatbuffers "tensorflow-gpu<1.14"
|  TensorFlow activation export                 | pip install typing six numpy scikit-image "tensorflow-gpu<1.14"
|  ONNX conversion                              | pip install typing six numpy protobuf
|  ONNX conversion tests                        | pip install typing six numpy protobuf onnx torch
|  Caffe conversion                             | pip install typing six numpy protobuf
|  Caffe conversion tests                       | pip install typing six numpy protobuf, Caffe
|  Caffe activation export                      | pip install typing six numpy protobuf, Caffe
|  Caffe2 conversion                            | pip install typing six numpy protobuf
|  Caffe2 conversion tests                      | pip install typing six numpy protobuf torch


All pip dependencies (just for reference):
```
pip install typing six numpy scikit-image protobuf flatbuffers onnx torch "tensorflow-gpu<1.14"
```

### Remarks for ONNX conversion tests

Caffe2 (which we use as ONNX backend) is now in `torch`, that's why we need it.

### Remarks for TensorFlow export

For TensorFlow Python code conversion and some test cases, a working TensorFlow installation is needed.
Versions 1.8 - 1.13 should be fine. Prefer the GPU versions as they support more operations.
Some operations are not supported by older versions, but the tools should work in general.
If you use the CPU version or an older version, some test cases might break.   

## Command line tools

### The conversion tool: ```nnef_tools/convert.py```

It can be used to convert between different neural network frameworks.

The tool has an extensive help that can be printed with ```--help```.

Supported frameworks:
- NNEF
- Tensorflow Python code (+ checkpoint) 
- Tensorflow Protobuf model
- Tensorflow Lite FlatBuffers model
- ONNX
- Caffe
- Caffe2

In all conversions either the input or the output framework must be NNEF.

When converting from ONNX or Tensorflow Protobuf model, the shape of the input(s) might be incomplete or unknown in the model.
By default (if the rank of the inputs is known) all unknown dimensions are set to 1.
To specify an input shape manually, use the ```--input-shape``` option.
 
Tensorflow uses NHWC as default data format while NNEF uses NCHW. 
The needed conversion is done by inserting transpose operations 
before and after the affected operations in the graph.
Luckily most transposes are optimized away.
To avoid confusion, the data format of the input(s) and output(s) are by default not changed.
To change the data format of certain inputs/outputs, the ```--io-transformation``` option can be used. 

Examples:

```
./nnef_tools/convert.py --input-format=tensorflow-pb \
                        --output-format=nnef \
                        --input-model=tf_models/frozen_inception_v1.pb \
                        --output-model=out/nnef/frozen_inception_v1.nnef.tgz \
                        --input-shape="[2, 224, 224, 3]" \
                        --compress

./nnef_tools/convert.py --input-format=nnef \
                        --output-format=tensorflow-pb \
                        --input-model=out/nnef/frozen_inception_v1/model.nnef.tgz \
                        --output-model=out/tensorflow-pb/frozen_inception_v1.pb
```

Mappings between operations in various frameworks and NNEF can be found [here](operation_mapping.md).

### The activation exporter tool: ```nnef_tools/export_activation.py```

The activation exporter tool can be used to export the activations (the evaluated values of the tensors) 
of the TensorFlow graphs.
 
Before using the tool, one needs to convert the model to NNEF, and supply the resulting ```conversion.json``` to the exporter.
To make the converter write a  ```conversion.json``` we have to pass the ```--conversion-info``` flag to it. 

The tool has an extensive help that can be printed with ```--help```.

Example:

```
./nnef_tools/convert.py --input-format=tensorflow-pb \
                        --output-format=nnef \
                        --input-model=tf_models/frozen_inception_v1.pb \
                        --output-model=out/nnef/frozen_inception_v1.nnef.tgz \
                        --input-shape="[1,224,224,3]" \
                        --conversion-info \
                        --compress

./nnef_tools/export_activation.py  --input-format=tensorflow-pb \
                                   --input-model=tf_models/frozen_inception_v1.pb \
                                   --input-shape="[1,224,224,3]" \
                                   --output-path=out/nnef/frozen_inception_v1_activations \
                                   --conversion-info=out/nnef/frozen_inception_v1.nnef.tgz.conversion.json
```

## Conversion tests:

These tests convert the model to NNEF and then convert it back to the original framework. 
They compare the activations (heatmaps) of the original and the converted graph.

- ```./nnef_tests/conversion/*layer_test_cases.py```: Test cases for simple layers.

- ```./nnef_tests/conversion/*network_test_cases.py```: Test cases for full networks.


How to run all layer tests (about 5 minutes on i7-7700 + GTX 1050 Ti)?
```
python -m unittest discover -s 'nnef_tests/conversion' -p '*layer_test_cases.py'
```

### Conversion testing without running the networks

If you set the following environment variable
then the conversion tests will not run the networks and they will not check the resulting activations.

```
export NNEF_ACTIVATION_TESTING=0
```

In this case the following tests can run with reduced dependencies:

| Test                                       | Dependencies without activation testing |
|--------------------------------------------|-----------------------------------------|
|  onnx_network_test_cases.py                | pip install typing six numpy protobuf   |
|  tf_pb_network_test_cases.py               | pip install typing six numpy protobuf   |
|  tflite_network_test_cases.py              | pip install typing six numpy protobuf   |
|  caffe_network_test_cases.py               | pip install typing six numpy protobuf   |
|  caffe2_network_test_cases.py              | pip install typing six numpy protobuf   |


