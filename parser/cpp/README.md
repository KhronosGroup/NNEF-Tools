NNEF Parser Project
==========================

This repository C++ source code for a sample NNEF parser.

Introduction
------------

The code consists of a header only library that contains two example parsers (one for
flat and one for compositional NNEF syntax). This library can be used to build tools
that require parsing NNEF files. It requires a C++11 compatible compiler.

The code also contains an example main.cpp that showcases the usage of the parser library.

The tool itself parses and validates an NNEF graph structure document, and echoes an
optionally flattened version of it. The arguments required to the tool are as follows
* "graph file name": the file where the graph structure is described with NNEF syntax
* --flat: whether to use a flat parser (by default a compositional parser is used)
* --layers: whether to include layer level fragments (as described in the specification appendix)
* --binary: whether to check info in binary files. The binaries should be in the same folder as the graph file
* --quant: whether to check info in quantization files. The quantization file should be in the same folder as the graph file and should have the same name with .quant extension
* --atomic <op-names>: list of ops to treat as atomic; default list includes standard ops
* --no-atomic <op-names>: list of ops to treat as non-atomic

If the tool encounters an invalid document, it prints the first error and stops parsing.


Build
-----

The example can be compiled with cmake.

Example of build commands under Linux:
````
$ cd <NNEF parser root directory>
$ mkdir build && cd build
$ cmake ..
$ make
````

How to run
----------

Example of execution under Linux:
````
$ cd <NNEF parser build directory>
$ ./bin/nnef-validator ../examples/googlenet_comp.txt --layers
$ ./bin/nnef-validator ../examples/googlenet_flat.txt
````

Building the Python module
--------------------------

The python folder contains a Python wrapper for the C++ parser code. To build the python module, move into the python folder and run

`python setup.py install`

This invokes the system compiler for C++ (e.g. gcc, g++, clang depending on the operating system), 
builds and installs an 'nnef' python module. If that command succeeds, the nnef module can be used
within the Python interpreter.

Using the Python module
-----------------------

In the python interpreter, type

````
import nnef
attribs, ops = nnef.parse_file(input = 'example.nnef',
                               quantization = 'example.quant',
                               atomics=['relu', 'softmax'])
````

Optional quantization info can be supplied with the `quantization` argument. The argument `atomics` contains a 
list of op names that are not flattened by the compositional parser. By default, `atomics = nnef.StandardOperations`, 
which is a list that contains all standard operations.

To enable layer level fragments (described in the spec but not officially part of the standard) to be accepted by the parser, issue the command

`nnef._register_layer_ops()`

After invocation, `attribs` is a dictionary containing the name, inputs and outputs of the graph, 
the shapes, data-types and quantization information of tensors (in nested dictionaries).
`ops` is a list of operations (list of tuples containing the name and argument/result dictionaries of the op). 
See `python/sample.py` for more details.
