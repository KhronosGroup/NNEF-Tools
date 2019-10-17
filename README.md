[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)<p align="center"><img src="https://www.khronos.org/assets/uploads/ceimg/made/assets/uploads/apis/NNEF_500px_Apr17_165_75.png" /></p>

# NNEF-Tools

NNEF reduces machine learning deployment fragmentation by enabling a rich mix of neural network training tools and inference engines to be used by applications across a diverse range of devices and platforms.

This repository contains tools to generate and consume NNEF documents, such as a parser (C++ and Python) that can be included in consumer applications and converters for deep learning frameworks.

* [NNEF Model Zoo](models#nnef-model-zoo)
* [NNEF Tests](#nnef-tests)
* [NNEF Tools](nnef_tools#nnef-tools)
* [NNEF Parser](parser#nnef-parser-project)

## NNEF Model Zoo
A **Model Zoo** is now available; the 'models' folder contains a variety of [NNEF models](models#nnef-model-zoo) converted from various sources.

## NNEF Tests
NNEF Tests folder contains tests to verify installation and unit tests.

## NNEF Tools
[NNEF Tools](nnef_tools#nnef-tools) folder contains tools to convert pre-trained models in `tensorFlow`/`caffe`/`caffe2`/`ONNX` to NNEF format.

## NNEF Parser
[NNEF Parser](parser#nnef-parser-project) folder contains `C++` and `Python` source code for a sample NNEF graph parser.

## Release Notes
### Change in shape inference compared to previous version (04.10.2019)

According to a change in version 1.0.1 of the NNEF specification, the `shape_of` operator in NNEF syntax is deprecated, and the parser does not support it. This enables the decoupling of parsing from shape inference, allowing parsing to succeed even if shape information is not available for all operations, such as custom defined operations before the graph definition. Shape inference can still be run after training, furthermore it can be customized (via function pointers) for custom defined operations.

### IMPORTANT BUG FIX (10.19.2018)

We have recently found a bug in the Python code that reads/writes the tensor binary files (the header contained 4 extra padding bytes therefore not conforming to the spec). We have updated the code to read/write and _check_ the proper header size. As a consequence, any files written out with the code that contained the bug cannot be read back with the updated code. To aid the usage of such existing files, we have created a script called `fix_nnef_binary_size.py` that can be used to remove the excess 4 bytes from existing NNEF files. The script is located in the root folder of this repo, it has no dependencies (not even the NNEF parser). It can be run on the main folder of an NNEF model, and it fixes all binary files in the folder. In case one runs it on an NNEF model that does not contain the bug, it does nothing. It can be used as follows:
```
python fix_nnef_binary_size.py my_nnef_model_folder
```
Such an invocation fixes the files in place. Optionally, a second argument can be supplied to the script to write the fixed files to a different output path. In this case, the script copies all non-binary files (such as graph.nnef) to the target folder, so the resulting folder contains the whole valid model.
