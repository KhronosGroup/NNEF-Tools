# NNEF Tools

This package contains a set of tools for converting and transforming machine learning models.

## Usage

For basic usage, you have to supply an input format, an output format and an input model. The output model name defaults to the input model name suffixed with the output format, but it can also be supplied explicitly.

```
python -m nnef_tools.convert --input-format tf --output-format nnef --input-model my_model.pb --output-model my_model.nnef
```

### Setting input shapes

If the model has (partially) undefined shapes, the concrete shapes can be supplied with the `--input-shapes` argument. The input shapes must be a Python dict expression, with string keys of input tensor names and tuple values as shapes. It is enough to supply shapes for those inputs that we want to freeze. For example:

```
--input-shapes "{'input': (1, 224, 224, 3)}"
```

### Transposing inputs and outputs

When converting between TF and NNEF, the (default) dimension ordering differs, and the model may be transposed (for example in case of 2D convolutional models). However, the inputs and outputs are not automatically transposed, as the converter cannot reliably decide which input and outputs represent images. Transposing inputs and outputs can be turned on by the `--io-transpose` option. There are two ways to use it: either to transpose all inputs and outputs, or to select the ones to be transposed. All inputs and outputs can be transposed by using `--io-transpose` without any further arguments, while selecting inputs and outputs can be done by providing a list of names:

```
--io-transpose "input1" "input2" "output1"
```

### Retaining input/output names

During conversion, the converter may generate suitable names for tensors. However, it is possible to force to keep the names of input and output tensors using the `--keep-io-names` option.


### Folding constants

The original model may contain operations that are performed on constant tensors, mainly resulting from shapes that are known in conversion time, or that became known by setting with the `--input-shape` option. In this case, it can be useful to fold constant operations, because the resulting graph is simplified. Furthermore, without constant folding, the graph may not even be convertible due to the presence of non-convertible operations, but constant folding may eliminate them and make the model convertible. To use it, simply turn on the `--fold-constants` option.

### Optimizing the output model

The resulting model may contain operations or sequences of operations that can be merged or even eliminated as they result in a no-op. To do so, turn on the `--optimize` flag. This works for NNEF output.

The converter can also be run with the same input and output format. In this case, the tool only reads and writes the model, with an optional optimization phase in between if the `--optimize` flag is set and an optimizer is available for the given format.

### Handling unsupported operations

When running into an unsupported operation, the converter stops the conversion process. It is possible to override this behavior by enabling mirror-conversion (one-to-one copying the operation to the destination format) using the `--mirror-unsupported` flag. This may not result in a valid output model, but may be helpful for debugging.

## Further options

The following further options can be used when the output format is NNEF:
* The `--compress` option generates a compressed `tgz` file. It can also take a further compression level argument.
* The `--annotate-shapes` flag generates the graph description with the shapes of tensors annotated in comments.
* The `--output-names` option takes a list of tensor names, and considers those as outputs, and only converts the sub-graph required to compute those outputs.
* The `--tensor-mapping` option allows to save the mapping of tensor names (mapping from the input model to the output model) into a separate json file.


## Conversion from TF Python code

When starting from Python code, the first step is to export the graph into a graph-def protobuf (.pb) file, which can then be further converted to a different format. To do so, the package contains some utility functions to freeze the graph and save it. Simply import these utilities and call them in your Python code:

```
import nnef_tools.io.tf.graphdef as graphdef
# define your TF model here
with tf.Session() as sess:
    ...     # initialize variables and train graph
    graphdef.save_default_graph('path/to/save.pb', session=sess, outputs=...)
```

If your model contains dynamic shapes, you can save the graph with concrete shapes by providing the input shapes to the save function. Furthermore, constant operations can also be folded while saving the model:

```
graphdef.save_default_graph('path/to/save.pb', session=..., outputs=...,
                            input_shapes={'input': (1, 224, 224, 3)},
                            fold_constants=True)
```

Outputs can be specified as a list of tensors, or alternatively, they can be renamed by mapping tensors to strings as new names.

### Saving composite functions as a single operation

Often, when exporting a graph, it is desirable to convert a subgraph (compound operation) into a single operation. This can be done by defining the subgraph in a Python function and annotating it with `@composite_function` of the `graphdef` module:

```
@graphdef.composite_function
def my_compound_op( x, a, b ):
    return a * x + b
```

Then `graphdef.save_default_graph` will magically take care of the rest, by converting composite functions into `PyFunc` ops in the graph-def. Note however, that if you are exporting such graphs repeatedly, you have to call `graphdef.reset_composites()`  before the definition of the graph.

How exactly the signature of the function is converted depends on the invocation of the function: tensor arguments are converted to inputs, while non-tensor arguments are converted to attributes. It does not matter whether positional or keyword arguments are used. Outputs must be tensors:

```
graphdef.reset_composites()

# define the graph
x = tf.placeholder(shape=(2,3), dtype=tf.float32, name='input')
y = my_compound_op(x, a=4, b=5)   # x is treated as tensor, a and b as attributes

with tf.Session() as sess:
    graphdef.save_default_graph('path/to/save.pb', session=sess, outputs={y: 'output'})
```

When exporting models containing composite functions, if the model has dynamic shapes it is preferable to export it with concrete shapes and folding constants during export. This is because before converting composite functions to a single op, TF can still perform shape inference and constant folding automatically, but after the conversion, it cannot infer shapes and perform the computation of the `PyFunc` operations resulting from the composite functions. If there are no composite functions in the model, then concrete shapes can be provided later as well (during conversion), accompanied by constant folding.

Collapsing composites to a single op when saving the graph can be turned off by `collapse_composites=False`. See `custom/composite_export_example.py` for more examples.


#### **Important note**

Composite functions **must not** get tensor inputs from other sources than the function arguments (such as global or class member variables). In that case, the code must be reorganized to make the actual composite function be called with explicitly marked tensor arguments. The same practice is also useful for attributes. In general, composite functions should be stateless.


## Custom converter plugins

The coverage of the converter can be extended to custom operations. This is required for example, when one wants to convert a composite function. Such a function is exported to the protobuf model as a `PyFunc` operation, that records the name, attributes, inputs and outputs of the original composite function. However, a converter must be provided for that name. In the actual conversion process, the `PyFunc` node is replaced with an operator of the original name of the composite function, so that it can be referenced.

The conversion of operations is governed by `nnef_tools.conversion.Transform` instances mapped to operator types. To add a new operator to be converted, one needs to provide a map entry for the operator. This is done by providing a Python module to the converter that contains the mapping for custom operators in a dict with the standard name `CUSTOM_TRANSFORMS`. The module is injected to the converter with the `--custom-converters` option:

```
--custom-converters my.custom.plugin.module
```

where `my/custom/plugin/module.py` is a Python module accessible to the Python interpreter (either by providing an absolute path or by setting `PYTHON_PATH`). Its contents may look like the following:

```
from nnef_tools.conversion import Transform

def my_conversion_helper_func(converter, ...):
    ...

CUSTOM_TRANSFORMS = {
    'op_type_to_convert_from':
        Transform(
            type='op_type_to_convert_into',
            name='optional_name_of_resulting op',
            inputs=(
                # one entry for each input
            ),
            outputs=(
                # one entry for each output
            ),
            attribs={
                # one entry for each attribute
            }
        ),
}
```

Entries are for the resulting operator, and may be constant Python values or expressions to be evaluated by the Python interpreter. Such expressions are written as Python strings that start with the `!` character, for example `'!a+2'` evaluates the expression `a+2`. The expressions are evaluated in the context of the source operator (the one converted from) and the converter context (that is defined by the input and output formats). It consists of the following:
* The type of the source operator is accessed via the identifier `_type_`.
* The name of the source operator is accessed via the identifier `_name_`.
* Inputs of the source operator are accessed via the identifier `I`, which is a Python `list`. For example the expression `'!I[0]'` results in the first input.
* Outputs of the source operator are accessed via the identifier `O`, which is a Python `list`. For example, the expression `'len(O)'` results in the number of outptus.
* Attributes of the source operator are accessed via identifiers that match the names of the attributes. For example if the source operator has attribute `a` then the expression `'!a'` takes its value.
* Furthermore, the following can be used in building complex expressions:
    * All built-in Python operators and functions.
    * All public member functions (not starting with `_`) defined by the converter in effect.
    * All public functions (not starting with `_`) defined in the custom module. Such functions must take a converter as their first argument, but otherwise can take arbitrary arguments. The public methods of the converter can be used in their definition.

The `Transform` can further contain a `using={'id': '!expr', ...}` field, which may define intermediate expressions that are evaluated first and can be used in other expressions for attributes/inputs/outputs. If the dictionary is ordered, the entries may depend on each other.

Furthermore, by adding an optional `cond='!expr'` field to the `Transform`, it is possible to achieve conditional conversion, only when the given expression evaluates to `True`. Otherwise, the converter treats it as if there was no converter provided for the given operator. This is to allow conversion of operations with only certain attribute values.

See `custom/custom_transforms_example.py` for more details.

Similarly to the above mechanism, custom shape inference functions and custom operator definitions (fragments) can be plugged in to converters that convert from NNEF using the `--custom-shapes` and `--custom-fragments` option. This may be required for custom NNEF operators defined as fragments in the input when such fragments are not decomposed. The fragments and shape inference functions must be defined in python module(s) supplied after the `--custom-shape` or `--custom-fragments` option. The module may look like this:

```
def my_custom_shape_function(intput1_shape, ..., attrib1, ...)
    ...     # assert validity of input shapes / attribs
    ...     # return calculated output shape(s)

CUSTOM_SHAPES = {
    'my_custom_op': my_custom_shape_function,
}
```

or

```
op_fragment =
"""
# NNEF fragment declaration/definition goes here
"""

CUSTOM_FRAGMENTS = {
    'op-name': op_fragment,
}
```

Furthermore, the `--decompose` option can be used to let the NNEF parser decompose the (composite) operators listed after the option (as separate args).

Additionally, with a similar mechanism, custom optimization passes can also be injected to the converter. The optimizer can match sequential sub-graphs (chains), and replace them with another sequence of operations. To provide custom optimizer passes, the chains of operations to be replaced must be mapped onto functions that perform generate the replacement sequence after checking the chain to bre replaced for validity:

```
def replace_my_chain(a, b, c):   # a, b, c will contain the matched chain of ops in order when this is called
    ...     # check attributes of the chain a, b, c to see if it should really be replaced;
            # if not, return False (do not modify the graph before all checks)
    ...     # create new tensors and operations in the graph that will replace the chain
    ...     # either return nothing (None), or any non-False value

CUSTOM_OPTIMIZERS = {
    ('a', 'b', 'c'): replace_my_chain,      # use a tuple as key, since list is not hashable
}
```

See `custom/custom_optimizers_example.py` for mode info.

## Executing a model and saving activations

A separate tool (`execute.py`) is available for executing a model. It requires a model and a format to be specified.

The inputs may be read from the (binary) input stream and outputs may be written to the (binary) output stream. Tensor data files can be piped as inputs and outputs:

```
python -m nnef_tools.execute < input.dat my_model.pb --format tf > output.dat
```

Alternatively, inputs can be random generated, and selected activations may be written to a folder, allowing to specify a different name:

```
python -m nnef_tools.execute my_model.pb --format tf --random "uniform(0,1)" --seed 0 --output-path . --output-names "{'tensor-name1': 'save-name1', ...}"
```

Further options to the model executor:

* The `--batch-size` option can be used to perform batched execution if a model specifies batch size of 1 in its inputs, supplying the desired batch size. If the supplied batch size is 0, it means that the (common) batch size of the actual inputs is used. Furthermore, when the supplied batch size equals the one defined by the model, execution will be done one-by-one instead of a single batch, which may be useful for reducing the memory footprint.
* The `--statistics` flag (followed by an optional output file path) can be used to generate activation statistics and save it in json format.
* The `--tensor-mapping` option can be used to provide a tensor name mapping obtained from the conversion step to the executor, used in remapping tensor names when generating statistics. This may be useful for comparing executions of the same model in different formats.
* Inputs and outputs (or activations) may need transposing before feeding into execution or after execution upon saving. This can be achieved with the `--io-transpose` flag. If no further arguments are listed, all tensors are transposed, but the transposed tensors can be controlled by enumerating a list of tensor names (as separate args). Inputs read from the input stream are transposed from channels first to last, while the outputs that are written to the output stream or saved are transposed from channels last to first if the format dictates so (TF/Lite).
* The `--decompose` option can be used to let the NNEF parser decompose the (composite) operators listed after the option (as separate args).
* The `--custom-operators` option can be used to inject custom operators to the executor by supplying a python module after the option. The contents of the module may look like this:

```
def my_custom_op(input1, ..., attrib1, ...):
    ...     # calculate output using inputs / attribs

CUSTOM_OPERATORS = {
    'my_custom_op': my_custom_op,
}
```

See `custom/custom_operators_example.py` for more info.

Further tools are available for generating random tensors (`random_tensor.py`) and converting images to tensors (`image_tensor.py`). These tools write their results to the output stream and can be directed into a file or piped to `execute.py`.

## Visualizing a model

NNEF models can be visualized with the `visualize.py` tool. The tool generates and svg/pdf/png rendering of the NNEF graph:

```
python -m nnef_tools.visualize my_model.nnef --format svg
```

By default, the render only contains the names of operations and tensors. In case of and svg output, _tooltips_ contain more details about nodes (op attributes, tensor dtypes and shapes). The shapes are only calculated if the `--infer-shapes` flag is turned on. To include those details in the render itself, use the `--verbose` flag.

## GMAC calculation

The script `gmac.py` can be used to calculate the GMACs required to execute a model. By default, it only calculates linear operations (convolutions, matrix multiplies), but it is possible to add other groups of operations (pooling, normalization, reduction, up-sampling) into the calculation:

```
python -m nnef_tools.gmac my_model.nnef --include-pooling
```

The calculation requires shape inference, so in case of custom operators, the `--custom-shapes` option should be used (same as for `convert.py`).

## Troubleshooting

Several things can go wrong during various stages of conversion, and sometimes it's hard to find where it exactly happened. Here are a few tips on how to get started:
* If the export process starts from Python code in a framework such as TensorFlow or PyTorch, the first step is saving the model into a framework specific format, such as TensorFlow protobuf or ONNX in case of PyTorch.
    * Check the resulting model to see if it accurately reflects the framework code. TensorBoard or Netron viewer can be used for this purpose.
    * If there is an error in this step, try to turn off certain flags during saving. For example in `nnef_tools.io.tf.graphdef.save_default_graph`, try turning off `fold_constants` and `collapse_composites` flag. The first merges operations on constant tensors, the second one merges composite operators into a single piece. By turning them off, errors in these transformation steps can be excluded.
* If the conversion from any model format to NNEF fails, typical reasons are as follows:
    * Conversion of some operator is not implemented. In this case, adding a custom converter using the `--custom-converters` option can solve the problem.
    * There is a bug in the converter; for example it does not support some parameter/version of an operator. In this case file a bug for `nnef_tools`.
* After the conversion to NNEF succeeds, check the converted model by executing it (`nnef_tools.execute`) on some (maybe random) inputs.
    * Execution may itself fail if there are custom operators in the model, in which case custom executors can be injected with the `--custom-operators` option.
    * If executed on non-random inputs, the outputs can be compared to results obtained from executing the same model in the original framework, or after saving it and executing the saved model (`nnef_tools.execute`). By comparing the results of those three stages, it is possible to tell in which stage something goes wrong. However, make sure to feed the same inputs to all stages, and beware that NNEF dimension order (channels first) is different from TensorFlow dimension order (channels last).
    * If the failing stage is the saving step, see above for turning off certain options too see if those are the culprits.
    * If the failing stage is the conversion step, first make sure to isolate optimizations by not using the `--optimize` option. The same goes for the `--fold-constants` option to see if that causes problems.
    * If conversion fails even without optimization and constant folding, it is usually due to the conversion of one of the operations, which must be found. Ideally, one would compare the intermediate tensors after each operation in a sequence, but exact comparison is hard to do automatically due to non 1-1 mappings during the conversion. However, generating statistics (`nnef_tools.execute --statistics`) for the same input for both models allows comparison of how execution proceeds in the two models and finding where the first difference occurs.
* When in doubt about some of the tools and this documentation does not provide enough information, check the help of the command-line tool itself (`-h` or `--help`) option.
