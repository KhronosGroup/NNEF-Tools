SkriptND project
===================

Introduction
------------

SkriptND is a domain specific language to decribe operations on N-dimensional arrays (tensors) and graphs composed from such operations. 
It allows complete definition of operations including types and shapes of inputs and outputs (shape propagation) 
and the mathematical formulae to compute the output of primitive operations (lowering to scalar computation) using compact notations.

The code consists of a C++ library for parsing SkriptND syntax. This library can be used to build tools
that require parsing NNEF files. It requires a C++17 compatible compiler. 

Furthermore, the repository contains Python wrapper code around the C++ parser and also adds some further utilities to load and save 
computational parameters easily and to generate sample execution code (C++) from the sources, and ultimately an executable python object.

C++ Library
-----------

Documentation of the C++ library: [cpp_api.md](cpp_api.md)

Python Package
--------------

Documentation of the Python package: [package_info.md](package_info.md)
