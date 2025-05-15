SkriptND Python package
===================

This package contains a SkriptND parser and sample executor, using a C++ backend.


Using the module
-----------------------

In the python interpreter, type

    import skriptnd as sknd
    model = sknd.read_model('path/to/model/folder')

The path `'path/to/model/folder'` should point to the folder containing the model's main .sknd file, which may reference other .sknd files that it includes and .dat files for model parameters.

Alternatively, the methods

    model = sknd.parse_file('path/to/model/folder/main.sknd')

and

    model = sknd.parse_string("graph G { ... }")

can be used to parse a model from files or strings without loading the associated parameters.

After invocation, `model` is a data class containing the graphs of the model, and each graph contains its name, tensors, operations, inputs and outputs. After the model is obtained, it can be compiled with the command

    compiled = sknd.compile_model(model)

The compiled model is a python executable object that can then be invoked with inputs (numpy arrays) and returns the 
model outputs (as a tuple of numpy arrays):

    import numpy as np
    input = np.random.random((10, 20))
    output, = compiled(input)

Note however that the executor uses a sample C++ code generator running on CPU (unoptimized), therefore it is only 
intended for testing/comparison purposes.  

Further usage examples can be found in `sample.py`.
