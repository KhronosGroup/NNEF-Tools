NNEF Parser - repository
===================


Introduction
------------

The code consists of a C++ library that contains two example parsers (one for
flat and one for compositional NNEF syntax). This library can be used to build tools
that require parsing NNEF files. It requires a C++11 compatible compiler. 

The Python code wraps the C++ parser and adds some further utilities to load and save NNEF documents easily. It also contains a script to validate NNEF documents (`validate.py`) and optionally print a lowered version of the graph. If the tool encounters an invalid document, it prints the first error and stops parsing. Type `python validate.py -h` to show the usage help.

C++ Library
-----------

Documentation of the library: [cpp_api.md](cpp_api.md)


Python Package
--------------

Documentation of the Python package: [package_info.md](package_info.md)

