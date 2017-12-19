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

If the tool encounters an invalid document, it prints the first error and stops parsing.


Build
-----

The example can be compiled with cmake.

Example of build commands under Linux:

$ cd <NNEF parser root directory>
$ mkdir build && cd build
$ cmake ..
$ make


How to run
----------

Example of execution under Linux:

cd <NNEF parser build directory>
./bin/nnef-parser ../examples/googlenet_comp.txt -layers
./bin/nnef-parser ../examples/googlenet_flat.txt
