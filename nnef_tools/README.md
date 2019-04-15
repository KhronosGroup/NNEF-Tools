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

| Functionality                              | Dependencies                                                        |
|--------------------------------------------|---------------------------------------------------------------------|
|  TensorFlow Python code conversion (+tests)| pip install typing six numpy "tensorflow-gpu<1.13                   |
|  TensorFlow Protobuf conversion            | pip install typing six numpy protobuf                               |
|  TensorFlow Protobuf activation tests      | pip install typing six numpy protobuf "tensorflow-gpu<1.13"         |
|  TensorFlow Lite conversion                | pip install typing six numpy flatbuffers                            |
|  TensorFlow Lite activation tests          | pip install typing six numpy flatbuffers "tensorflow-gpu<1.13"      |
|  TensorFlow activation export              | pip install typing six numpy scipy matplotlib "tensorflow-gpu<1.13" |
|  ONNX conversion                           | pip install typing six numpy protobuf                               |
|  ONNX activation tests                     | pip install typing six numpy protobuf onnx torch                    |

All dependencies (just for reference):
```
pip install typing six numpy scipy matplotlib protobuf flatbuffers onnx torch "tensorflow-gpu<1.13"
```

### Remarks for ONNX activation tests

Caffe2 is now in `torch`, that's why we need it.

### Remarks for TensorFlow export

For TensorFlow Python code conversion and some test cases, a working TensorFlow installation is needed.
Versions 1.8 - 1.12 should be fine. Prefer the GPU versions as they support more operations.
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

In all conversions either the input or the output framework must be NNEF.

When converting from ONNX or Tensorflow Protobuf model, the shape (or dtype) of the input(s) might not be set in the model.
In this case the ```--input-shape``` option must be used.
 
Tensorflow uses NHWC as default data format while NNEF uses NCHW. 
The needed conversion is done by inserting transpose operations 
before and after the affected operations in the graph.
Luckily most transposes are optimized away.
To avoid confusion, the data format of the input(s) and output(s) are by default not changed.
To change the data format of certain inputs/outputs, the ```--io-transformation``` option can be used. 

Examples:

```
./nnef_tools/convert.py --input-framework=tensorflow-pb \
                        --input-model=tf_models/frozen_inception_v1.pb \
                        --input-shape="(float32, [2, 224, 224, 3])" \
                        --output-framework=nnef \
                        --output-directory=out/nnef/frozen_inception_v1 \
                        --compress

./nnef_tools/convert.py --input-framework=nnef \
                        --input-model=out/nnef/frozen_inception_v1/model.nnef.tgz \
                        --output-framework=tensorflow-pb \
                        --output-directory=out/tensorflow-pb/frozen_inception_v1
```

Mappings between operations in various frameworks and NNEF can be found [here](operation_mapping.md).

### The activation exporter tool: ```nnef_tools/export_activation.py```

The activation exporter tool can be used to export the activations (the evaluated values of the tensors) 
of the TensorFlow graphs.
 
Before using the tool, one needs to convert the model to NNEF, and supply the resulting ```conversion.json``` to the exporter.  

The tool has an extensive help that can be printed with ```--help```.

Example:

```
./nnef_tools/convert.py --input-framework=tensorflow-pb \
                        --input-model=tf_models/frozen_inception_v1.pb \
                        --input-shape="{'input:0':('float32', [1,224,224,3])}" \
                        --output-framework=nnef \
                        --output-directory=out/nnef/frozen_inception_v1 \
                        --compress

./nnef_tools/export_activation.py  --input-framework=tensorflow-pb \
                                   --input-model=tf_models/frozen_inception_v1.pb \
                                   --input-shape="{'input:0':('float32', [1,224,224,3])}" \
                                   --output-directory=out/nnef/frozen_inception_v1/activations \
                                   --conversion-info=out/nnef/frozen_inception_v1/conversion.json
```

## Activation tests:

These tests convert the model to NNEF and then convert it back to the original framework. 
They compare the activations (heatmaps) of the original and the converted graph.

- ```./tests/activation/*layer_test_cases.py```: Test cases for simple layers.

- ```./tests/activation/*network_test_cases.py```: Test cases for full networks.


How to run all layer tests (about 5 minutes on i7-7700 + GTX 1050 Ti)?
```
python -m unittest discover -s 'tests/activation' -p '*layer_test_cases.py'
```
