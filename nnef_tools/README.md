# NNEF Tools

This is a repository of NNEF converters and other tools.

## Dependencies

Python2 version >= 2.7.12 and Python3 version >= 3.5.2 are supported.
Python3 is recommended.

All tools need an installed `nnef` package.

Other dependencies for all tools:

```pip install numpy typing six```

For activation export with image as input source:

```pip install matplotlib scipy```

For export from TensorFlow Python code, a working TensorFlow installation is needed.
It is also needed for the test cases of TensorFlow Protobuf and TensorFlow Lite. 
Versions 1.8 - 1.12 should be fine. Prefer the GPU versions as they support more operations.
Some operations are not supported by older versions, but the tools should work in general.

```pip install tensorflow-gpu==1.12```

For ONNX test cases the `onnx` and the `torch` package is required.
We use caffe2, but now it is inside the `torch` package.

```pip install onnx torch```

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
                                   --conversion-json=out/nnef/frozen_inception_v1/conversion.json
```

# Activation tests:

These tests convert the model to NNEF and then convert it back to the original framework. 
They compare the activations (heatmaps) of the original and the converted graph.

- ```./tests/activation/*layer_test_cases.py```: Test cases for simple layers.

- ```./tests/activation/*network_test_cases.py```: Test cases for full networks.


How to run all layer tests (about 5 minutes on i7-7700 + GTX 1050 Ti)?
```
python -m unittest discover -s 'tests/activation' -p '*layer_test_cases.py'
```
