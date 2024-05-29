NNEF Parser Project
===================

This package contains a sample NNEF parser, using a C++ backend.


Using the module
-----------------------

In the python interpreter, type

    import nnef
    graph = nnef.load_graph('example.nnef')

If the path (`example.nnef`) points to a folder (with a graph.nnef in it), the whole model with weights is loaded. 
If it points to a file, it is interpreted as the graph description only, and it is loaded without weights.

Alternatively, the methods

    graph = nnef.parse_file("graph.nnef", quantization = "graph.quant")

and

    graph = nnef.parse_string("version 1.0; graph ...", quantization = "...")

can be used to parse a graph and optional quantization info from files or strings.

After invocation, `graph` is a data structure (named tuple) containing the name, tensors, operations, inputs and outputs of the graph.
If shape information is also required, it can be obtained by calling `nnef.infer_shapes(graph)`, which updates the shape information on the graph structure in place.
