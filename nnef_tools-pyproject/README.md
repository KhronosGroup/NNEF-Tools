# NNEF Tools

This package contains a set of tools for converting and optimizing machine learning models into NNEF format and compiling or executing them.

## Usage

[Python package usage](package_info.md)

## Dependencies

The python package supports extras for different functionalities:

| Functionality                  | Extra               | Additional packages                          |
|--------------------------------|---------------------|----------------------------------------------|
| TensorFlow Protobuf conversion | tensorflow-protobuf | tensorflow                                   |
| TensorFlow Lite conversion     | tensorflow-lite     | tensorflow, flatbuffers                      |
| ONNX conversion                | onnx                | protobuf, onnx, onnx-simplifier, onnxruntime |
| Visualization of NNEF models   | visualization       | graphviz                                     |
| Full install                   | full                | _all packages listed above_                  |

Installing ONNX dependencies (for reference):
```
pip install nnet_tools[onnx] 
```
