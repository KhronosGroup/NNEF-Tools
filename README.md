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

### Added new operators in spec version 1.0.4 (06.15.2021)

Following the update of the NNEF specification to version 1.0.4, conversion for the corresponding operators has been added. Furthermore, error handling of non-convertible models has been greately enhanced with error messages detailing the exact cause of failure listed for all non-convertible operations before conversion is started.

### Reworked NNEF Tools (10.21.2020)

The tools for converting models to NNEF and transforming NNEF models has been thoroughly reworked to make them more robust and unified and easier to maintain. The basic functionality of the main scripts has been kept, however their parameterization has been simplified and unified in some places; please refer to the readme and the help (`-h` option) of the respective scripts for more details. The scripts cover the following major areas of functionality: model conversion, optimization, execution and visualization. A GMAC calculator is also provide, and further utility scripts may be added in the future.  

### Change in quantization information in binary files (06.12.2020)

According to the change in version 1.0.3 of the NNEF specification, quantization algorithm information has been deprecated in the tensor binary file format. The tensor binary only stores the item-type of the tensor data, and the binary reader does not return quantization information (also used to be called 'compression' info). Furthermore, the mapping between stored item-types and data-types in the structural description has been clarified, so that the reader of a tensor binary can tell what the data-type of the read tensor is. This enhances the reader as it can now properly map the binary data to C++ or Python numpy types upon reading. The C++ code has been updated to perform such a mapping, and is now able to return a typed array instead of just plain bytes.

### Change in shape inference compared to previous version (04.10.2019)

According to a change in version 1.0.1 of the NNEF specification, the `shape_of` operator in NNEF syntax is deprecated, and the parser does not support it. This enables the decoupling of parsing from shape inference, allowing parsing to succeed even if shape information is not available for all operations, such as custom defined operations before the graph definition. Shape inference can still be run after training, furthermore it can be customized (via function pointers) for custom defined operations.

### TENSOR BINARY BUG FIX (10.19.2018)

There was a bug in the Python code that reads/writes the tensor binary files (the header contained 4 extra padding bytes therefore not conforming to the spec). The code has been updated to read/write and _check_ the proper header size. As a consequence, any files written out with the code that contained the bug cannot be read back with the updated code. To aid the usage of such existing files, a script was created called `fix_nnef_binary_size.py` that can be used to remove the excess 4 bytes from existing NNEF files. The script is located in the root folder of this repo, it has no dependencies (not even the NNEF parser). It can be run on the main folder of an NNEF model, and it fixes all binary files in the folder. In case one runs it on an NNEF model that does not contain the bug, it does nothing. It can be used as follows:
```
python fix_nnef_binary_size.py my_nnef_model_folder
```
Such an invocation fixes the files in place. Optionally, a second argument can be supplied to the script to write the fixed files to a different output path. In this case, the script copies all non-binary files (such as graph.nnef) to the target folder, so the resulting folder contains the whole valid model.
