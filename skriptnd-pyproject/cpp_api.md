
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

auto model = sknd::read_model("path/to/model/folder", import_handler, error_handler, "entry-point");
if ( !model )
{
    std::cout << "Failed to read model" << std::endl;
    exit(-1);
}
```

The entry-point parameter may be the name of the main graph to be considered the entry point of the model, or may be left empty, in which case the first graph in the model is considered as the entry point.

The import handler tells how module imports are mapped from module names to input streams. It must be a function with the following signature:

```
std::unique_ptr<std::istream>( const std::string& module_name )
```

Its most basic implementation simply reads files from one or more import paths, for which a helper function is provided:

```
auto importer = []( const std::string& module_name )
{
    return sknd::try_import_from_paths(module_name, {"path/to/stdlib/"});
};
```

The stdlib path can point to a folder that contains the .sknd module files with operator definitions that are considered as built-in, but other paths may also be enumerated, such as the model folder itself or its sub-folders that may contain custom operator definitions.

The import handler may also implement a mechanism to import modules from bundles that somehow package the standard library files with the parser.

The error handler must be a function with the following signature:

```
void( const sknd::Position& position, const std::string& message, const sknd::StackTrace& trace, bool warning );
```

In case of compilation errors, the error handler is called for each error.

Further optional parameters to the function `sknd::read_model` are as follows:
* attributes: dictionary of attribute values to the main graph
* flags: compiler options that control some aspects of the built model

Upon success, the model structure is filled in (the returned value is wrapped in a `std::optional`). The model contains a hierarchy of graphs of tensors and operations. The main graph is built from the entry point of the model, while sub-graphs are formed if the model contains control-flow blocks or compound operations. In the latter case, the body of the compound operation is detailed in a separate sub-graph. The sub-graphs are context dependent, meaning they may reference tensors as inputs from their parent/context graph. Further details of the model structure are documented in `composer/model.h`.

Two methods are provided to alter the hierarchical graph structure related to compound operations: they can be _atomized_ or _inlined_ by calling the following methods:

```
atomize_compounds(*model, []( const sknd::Operation& op ){ /* return true if op should be atomized */ });
inline_compounds(*model, []( const sknd::Operation& op ){ /* return true if op should be inlined */ });
```

Atomization simply removes the sub-graph that details the components of a compound operation, leaving the operation as an atomic unit. Inlining substitutes the components into the main graph in place of the operation invocation.

Variables of the model must be loaded separately, by calling the method:

```
bool success = load_variables("path/to/model/folder", *model, error_handler);
```

More detailed usage example can be found in `skriptnd/cpp/src/sample.cpp`.
