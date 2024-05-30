# NNEF Tools

This package contains a set of tools for converting and transforming machine learning models.

## Dependencies

The python package supports extras for different functionalities:

| Functionality                  | Extra               | Additional packages                          |
|--------------------------------|---------------------|----------------------------------------------|
| TensorFlow Protobuf conversion | tensorflow-protobuf | tensorflow                                   |
| TensorFlow Lite conversion     | tensorflow-lite     | tensorflow, flatbuffers                      |
| ONNX conversion                | onnx                | protobuf, onnx, onnx-simplifier, onnxruntime |
| Caffe and Caffe2 conversion    | caffe               | protobuf, torch                              |
| Visualization of NNEF models   | visualization       | graphviz                                     |
| Full install                   | full                | _all packages listed above_                  |

Installing ONNX and Caffe dependencies (for reference):
```
pip install nnet_tools[onnx, caffe] 
```

## Usage

[Python package usage](package_info.md)
