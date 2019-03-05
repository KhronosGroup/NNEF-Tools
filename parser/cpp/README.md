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
* --atomic <op-name>: the name of the operation to be treated as atomic
* --lower <op-name>: the name of the operation to be lowered

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
graph = nnef.load_model('example.nnef')
````

If the path (`example.nnef`) points to a folder (with a graph.nnef in it), the whole model with weights is loaded. 
If it points to a file, it is interpreted as the graph description only, and it is loaded without weights.

Alternatively, the method

```
graph = nnef.parse_string("version 1.0; graph ...", quantization = "...")
```

can be used to parse a graph and optional quantization info from strings.

After invocation, `graph` is a data structure (named tuple) containing the name, tensors, operations, inputs and outputs of the graph.
See `nnef.py` and `python/sample.py` for more details.
