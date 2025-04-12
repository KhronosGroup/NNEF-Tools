SkriptND Parser Project
===================

This package contains a sample SkriptND parser and sample executor, using a C++ backend.


Using the module
-----------------------

In the python interpreter, type

    import skriptnd as sknd
    graph = sknd.read_model('/model/folder/main.nds')

The path `'/model/folder/main.nds'` should point to the main .nds source file of the model, which may reference other
.nds files that it includes and .dat files for model weights.

Alternatively, the methods

    graph = sknd.parse_file('/model/folder/main.nds')

and

    graph = sknd.parse_string("graph G { ... }")

can be used to parse a graph from files or strings without loading the associated weights.

After invocation, `graph` is a data class containing the name, tensors, operations, inputs and outputs of the graph.
After the graph is obtained, it can be compiled with the command

    model = sknd.compile_model(graph)

The compiled model is a python executable object that can then be invoked with inputs (numpy arrays) and returns the 
model outputs (as a tuple of numpy arrays):

    import numpy as np
    input = np.random.random((10, 20))
    output, = model(input)

Note however that the executor uses a sample C++ code generator running on CPU (unoptimized), therefore it is only 
intended for testing/comparison purposes.  
