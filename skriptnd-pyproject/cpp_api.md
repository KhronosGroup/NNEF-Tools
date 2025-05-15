
Building the C++ library
------------------------

The C++ library can be compiled with cmake. Example of build commands under Linux:
````
$ cd skriptnd-pyproject/skriptnd/cpp
$ mkdir build && cd build
$ cmake ..
$ make
````

Using the C++ library
---------------------

Using the C++ parser is as follows.

```
#include "skriptnd.h"

auto model = sknd::read_model("path/to/model/folder", "graph-name", "path/to/stdlib/folder", error_handler);
if ( !model )
{
    std::cout << "Failed to read model" << std::endl;
    exit(-1);
}
```

The graph-name parameter may be left empty, in which case the first graph in the model is considered as the entry point of the model.

If required, all the graphs defined in the model can be listed by calling

```
auto graph_names = sknd::enum_graph_names("path/to/model/folder");
```

The stdlib path must point to folder that contains the .sknd module files with operator definitions that are considered as built-in.

The error handler must be a function of signature:

```
void( const sknd::Position& position, const std::string& message, const sknd::StackTrace& trace, bool warning );
```

Upon succeess, the model structure is filled, while in case of errors, the error handler is called for each error.
The fields inside the model structure are documented in `composer/model.h`.
