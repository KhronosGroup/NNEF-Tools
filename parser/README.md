NNEF Parser Project
==========================

This repository contains C++ and Python source code for a sample NNEF parser.

Introduction
------------

The code consists of a C++ library that contains two example parsers (one for
flat and one for compositional NNEF syntax). This library can be used to build tools
that require parsing NNEF files. It requires a C++11 compatible compiler. The `sample.cpp` 
contains a minimal example that showcases the use of the parser.

The Python code wraps the C++ parser and adds some further utilities to load and save NNEF documents easily. It also contains a script to validate NNEF documents (`validate.py`) and optionally print a lowered version of the graph. If the tool encounters an invalid document, it prints the first error and stops parsing. Type `python validate.py -h` to show the usage help.


Building the C++ library
------------------------

The C++ library can be compiled with cmake.

Example of build commands under Linux:
````
$ cd NNEF-Tools/parser/cpp
$ mkdir build && cd build
$ cmake ..
$ make
````


Using the C++ library
---------------------

Using the C++ parser is as simple as follows:

```
#include "nnef.h"

nnef::Graph graph;
std::string error;
bool success = nnef::load_graph("path/to/NNEF/folder", graph, error);
```

Upon succeess, the graph structure is filled, while in case of an error, the error string is filled. The fields inside the graph structure, and further parameters to the `load_graph` function are documented in `nnef.h`. After the graph is successfully loaded, shape inference can be performed in a subsequent call if required:

```
success = nnef::infer_shapes(graph, error);
```

Upon success, the shape fields of tensors are filled in.


Building the Python module
--------------------------

The python folder contains a Python wrapper for the C++ parser code. To build the python module, move into the python folder and run
```
cd NNEF-Tools/parser/python
python setup.py install
```

This invokes the system compiler for C++ (e.g. gcc, g++, clang depending on the operating system), 
builds and installs an 'nnef' python module. If that command succeeds, the nnef module can be used
within the Python interpreter.


Using the Python module
-----------------------

In the python interpreter, type

````
import nnef
graph = nnef.load_graph('example.nnef')
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

After invocation, `graph` is a data structure (named tuple) containing the name, tensors, operations, inputs and outputs of the graph. See `nnef.py` and `python/sample.py` for more details. If shape information is also required, it can be obtained by calling `nnef.infer_shapes(graph)`, which updates the shape information on the graph structure in place.
