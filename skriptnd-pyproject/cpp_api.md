
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

Further optional parameters to the function `sknd::read_model` are as follows.
* atomic callback: function to decide if a compound operation is considered atomic (not expanded to components)
* unroll callback: function to decide if loops along tensor packs in a primitive operation are unrolled during lowering (replaced with a sequence of assignments)
* attributes: dictionary of attribute values to the main graph

The callbacks must be functions of signature:

```
bool( const std::string& name, const std::map<std::string,sknd::Typename>& dtypes,
      const std::map<std::string,sknd::ValueExpr>& attribs, const std::vector<sknd::TensorRef>& inputs
```

where the parameters of the callback are the properties of the operation in question, and the return value must be whether to consider the operation instance as atomic or to be unrolled. For both parameters, supplying a null pointer results in a constant false function.
