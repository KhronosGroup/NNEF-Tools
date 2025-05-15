
Building the C++ library
------------------------

The C++ library can be compiled with cmake.
The `examples/samples/sample.cpp` contains a minimal example that showcases the use of the parser.

Example of build commands under Linux:
````
$ cd nnef/cpp
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
