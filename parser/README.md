NNEF Parser Project
==========================

This repository C++ source code for a sample NNEF parser.

Introduction
------------

The code consists of a C++ library that contains two example parsers (one for
flat and one for compositional NNEF syntax). This library can be used to build tools
that require parsing NNEF files. It requires a C++11 compatible compiler.

The code also contains an example main.cpp that showcases the usage of the parser library.

The tool itself parses and validates an NNEF graph structure document, and echoes an
optionally flattened version of it. The arguments required to the tool are as follows
* <path>: the path to the NNEF folder or the standalone graph file
* --stdlib \<file-name>: an alternate definition of standard operations (defaults to all-primitive definitions)
* --lower \<op-name>: the name of the operation to be lowered (if defined as compound)
* --shapes: turn on shape inference and shape validity checking

If the tool encounters an invalid document, it prints the first error and stops parsing.


Building with CMake
-------------------

The example can be compiled with cmake.

Example of build commands under Linux:
````
$ cd <NNEF parser root directory>
$ mkdir build && cd build
$ cmake ..
$ make
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

Alternatively, the methods

```
graph = nnef.parse_file("graph.nnef", quantization = "graph.quant")
```

and

```
graph = nnef.parse_string("version 1.0; graph ...", quantization = "...")
```

can be used to parse a graph and optional quantization info from files or strings.

After invocation, `graph` is a data structure (named tuple) containing the name, tensors, operations, inputs and outputs of the graph. See `nnef.py` and `python/sample.py` for more details.

The script `validate.py` is a Python implementation of the NNEF validator. Its command line arguments are the same as listed above.
