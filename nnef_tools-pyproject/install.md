# NNEF Tools


## Setup

### Installing from Git

To install the `nnef_tools` package into a Python environment, clone the NNEF-Tools repo and run

    python setup.py install

Afterwards, the scripts in the `nnef_tools` folder can be run using the `-m` option to `python`, as in the examples below. From the main folder of the repo, the scripts can also be run with the same `-m` option without installation.


## Dependencies

Python3 version >= 3.5.2 is supported.

For all tools an installed `nnef` package is required:


    cd parser/python
    python setup.py install
    cd ../..

You need to install dependencies only for the functionalities that you are using:

| Functionality                  | Dependencies                                                                 |
| ------------------------------ | ---------------------------------------------------------------------------- |
| TensorFlow Protobuf conversion | pip install future typing six numpy tensorflow                               |
| TensorFlow Lite conversion     | pip install future typing six numpy flatbuffers, tensorflow                  |
| ONNX conversion                | pip install future typing six numpy protobuf onnx onnx-simplifier onnxruntime |
| Caffe conversion               | pip install future typing six numpy protobuf torch                           |
| Caffe2 conversion              | pip install future typing six numpy protobuf torch                           |
| Execution of NNEF models       | pip install future typing six numpy torch                                    |
| Visualization of NNEF models   | pip install future typing six numpy graphviz                                 |


All pip dependencies (for reference):
```
pip install future typing six numpy protobuf flatbuffers onnx onnx-simplifier onnxruntime torch tensorflow
```
