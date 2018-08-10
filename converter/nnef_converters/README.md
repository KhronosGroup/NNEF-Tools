# NNEF Converters

The `nnef_converters` package consists of different command line tools for conversion between NNEF and the respective 
formats of neural network frameworks. It can convert the graph definitions as well as the trained models.

Current converters:
- `tf_to_nnef`: Converts Tensorflow Python source code (and checkpoint) to NNEF
- `nnef_to_tf`: Converts NNEF to Tensorflow Python source code (and checkpoint)
- `caffe_to_nnef`: Converts Caffe prototxt (and caffemodel) to NNEF
- `nnef_to_caffe`: Converts NNEF to Caffe prototxt (and caffemodel)

Extra tools for testing:
- `create_dummy_tf_checkpoint`: Creates a default initialized Tensorflow checkpoint for a Tensorflow graph 
- `create_dummy_caffe_model`: Creates a default initialized Caffe model for a Caffe graph

This package has example networks to experiment on.
They are located in the following directories:
- `nnef_converters/tf_converters/examples`
- `nnef_converters/caffe_converters/examples`

## Dependencies
The nnef_converters package can be installed systemwide or to a Python virtualenv.
In the later case, its dependencies must be installed to the same virtualenv.
Python 3 is recommended, but Python 2 is also supported.
We recommend using the latest available version of Python 2 or 3.
Linux and Mac OS X are fully supported. On Windows, only tf_to_nnef and nnef_to_tf has been tested.
The following dependencies must be installed:

- The official NNEF parser Python module (nnef) from
  [KhronosGroup's GitHub repository](https://github.com/KhronosGroup/NNEF-Tools/tree/master/parser/cpp)
- The numpy package: `pip install numpy`
- The neural network frameworks that you want to convert to/from. (You don't have to install all supported frameworks.)

Note that if a framework is installed from source, 
usually you have to manually add its python directory to your `PYTHONPATH` environment variable.

### Caffe remarks
In the case of Caffe, not only its python module, but also the `upgrade_net_proto_text` binary is used.
Its location is usually autodetected from the caffe python module's location, but if the detection fails, 
you have to manually set the `CAFFE_BIN_FOLDER` environment variable to its directory.
Its location can be for example: `/home/your.name/apps/caffe-1.0/build/tools`.

## Installation
Clone this repository to your computer and execute the following shell commands:
```sh
cd nnef_converters
python setup.py install
```

For a systemwide installation you may have to use `sudo python setup.py install`.
The setup script installs the command line tools to the virtualenv or to a system directory, 
so they can be used from any folder.

## Usage
All tools have `-h`/`--help`, `-v`/`--verbose` and `--version` options.
The `-v`/`--verbose` option emits optional debug messages 
and even writes debug comments to the generated files in some cases. 
All tools write to the current directory by default,
and their output can be redirected to another directory with the `-o`/`--output_path` option.
All * to NNEF tools output `.nnef.tgz` archives, with the `graph.nnef` and optionally the weight `.bin` files inside.

### Tensorflow to NNEF
The `tf_to_nnef` converter converts python source code (and checkpoint files) to NNEF.
To export your graph, you have to write a function which defines the graph and returns its outputs as a dictionary. 
We call this _network function_.  
The _network function_ can of course call other functions to create parts of the graph.
The _network function_ must be in a python file, which can optionally be inside a package hierarchy.

```python
import tensorflow as tf

def small_net():
    input_ = tf.placeholder(tf.float32, shape=[1, 64, 64, 3], name="input")
    filter_ = tf.get_variable(dtype=tf.float32, shape=[4, 4, 3, 16], name="filter")
    result = tf.nn.convolution(input=input_, filter=filter_, padding='SAME')
    return {"result": result}
```

If the `small_net` _network function_ is in `my_networks.py` in the current working directory 
(or in any directory in `PYTHONPATH`) then you can export its code with:

```sh
tf_to_nnef my_networks.small_net
```

#### Handling packages and directories
If `my_networks.py` is part of the `my_company.deep_learning` package,
 which is in the current directory (or in any directory in `PYTHONPATH`), then this command should be used:

```sh
tf_to_nnef my_company.deep_learning.my_networks.small_net
```

If the `my_company.deep_learning` package is not in the current directory (and not reachable from `PYTHONPATH`), 
you have to specify where can it be found with the `-p`/`--pythonpath` option.
Assuming it is in the `/home/xyz/packages` directory, you could use this command:

```sh
tf_to_nnef my_company.deep_learning.my_networks.small_net --pythonpath=/home/xyz/packages
```

#### Converting the pretrained weights too
If you would like to convert the pretrained weights of the network to NNEF, 
you can specify a checkpoint with the `-m`/`--model` option.
Its value can be any string accepted by `tf.train.Saver.restore`, e.g. a directory, or a checkpoint name prefix, 
like: `small_net_checkpoint`, or `small_net_checkpoint/model12.ckpt`. 
Example command:

```sh
tf_to_nnef my_networks.small_net --model=small_net_checkpoint
```

#### Printing debug info
The `-v`/`--verbose` option emits comments to the generated NNEF file, 
that helps locating the original tensorflow tensor for an NNEF tensor.

#### Shape transformations applied when converting
See "Tensorflow shape transformations".

### NNEF to Tensorflow
The `nnef_to_tf` converter can convert a `.nnef.tgz` archive (or an extracted directory or a single .nnef file) 
to Tensorflow python source (and checkpoint).
The checkpoint is created if and only if the nnef archive/directory has weights in it. 
It must have weights for either all of the variables or for none of them.

```sh
nnef_to_tf small_net.nnef.tgz
```

The `-v`/`--verbose` option generates comments as in `tf_to_nnef`.

#### Shape transformations applied when converting
See "Tensorflow shape transformations".

### Caffe to NNEF

The `caffe_to_nnef` tool converts a `prototxt` and optionally a `caffemodel` file to NNEF.

To export just the code:
```sh
caffe_to_nnef small_net.prototxt
```
To export the weights too:
```sh
caffe_to_nnef small_net.prototxt --model=small_net.caffemodel
```

#### Python layers
Sometimes you might use Python layers as inputs in your `prototxt` file.  
Python layers are exported as `external` operations.
If your Python layers are defined in a Python module in a directory that is not in `PYTHONPATH`,
you can specify that directory with the `-p`/`--pythonpath` option.
Alternatively you can just rewrite your Python layer to an Input layer.

### NNEF to Caffe

The `nnef_to_caffe` tool can convert a `.nnef.tgz` archive (or an extracted directory or a single .nnef file)
to a Caffe `prototxt` and optionally a `caffemodel` file.
The `caffemodel` is created if and only if the nnef archive/directory has weights in it. 
It must have weights for either all of the variables or for none of them.

```sh
nnef_to_caffe small_net.nnef.tgz
```

### Create dummy Tensorflow checkpoint

`create_dummy_tf_checkpoint` has the same parameters as `tf_to_nnef`, except the `-m`/`--model` parameter.

### Create dummy Caffe model

`create_dummy_caffe_model` has the same parameters as `caffe_to_nnef`, except the `-m`/`--model` parameter.

## Tensorflow shape transformations
### Different dimension orders
By default, Tensorflow uses many operations in NHWC mode, 
while NNEF and several frameworks can only use them in NCHW mode.
NHWC mode means that the operation expects its input(s) and writes its output(s) in 
\[batch, height, width, channel\] dimension order, while NCHW means \[batch, channel, height, width\].
The storage of convolutional weights and biases are also different in Tensorflow and NNEF.

### Transformation injection
To circumvent the problems caused by differing dimension orders, `tf_to_nnef` and `nnef_to_tf` inserts `transpose`, 
`unsqueeze` and other operations before and/or after the problematic operations.

### Graph optimization
But we love small and optimal graphs, so when possible we remove most of the generated operations from the graph. 
The four most common optimization patterns are the following (simplified examples):
1. conv, transpose, transpose_inverse, conv -> conv, conv
2. variable, unsqueeze, bias_add -> variable', bias_add
3. input, transpose, conv -> input', conv
4. conv, transpose, output -> conv, output'

In the first case, there is a transpose in the graph followed by its own inverse.
In this case the two transposes can simply be removed without causing any changes in the semantics of the graph.
We don't have to worry about this.

In the second case we have a variable with shape \[C\] and we would like to unsqueeze it to shape \[1, C\] 
before using it as a bias. 
The optimizer changes the variable's shape to \[1, C\] and removes the unsqueeze operation. 
This does not change the semantics, but probably you should know that the shape of your variable 
is now different then what it was in the source file.
To get the transforms applied to your variables as comments in the generated code,
you can use the `-v`/`--verbose` option.

In the third and fourth case the transpose is applied right after loading the input tensor 
or right before returning the output tensor. 
The optimizer simply removes the transpose, which means you will have to supply the inputs to the NNEF graph
in an order different to what you have used previously. 
For example you have supplied the input in NHWC order, but now you have to supply it in NCHW order.
The same can happen with outputs, maybe you have to process them in a different order. 
**You have to be notified about this**, so `tf_to_nnef` and `nnef_to_tf` outputs the list of applied transformations 
to each input and output tensor to stdout.

Let's have a look at this example:
```sh
nnef_to_tf network.nnef.tgz
```
```
Applied transformations:
Inputs:
input: [('transpose', [0, 2, 3, 1])]
Outputs:
result: [('transpose', [0, 2, 3, 1])]
```
The nodes `input`, and `result` both have been transposed with \[0, 2, 3, 1\], 
this probably means that semantically they are now NHWC instead of NCHW.

### Conclusion
For most traditional convolutional neural networks we can convert an NHWC Tensorflow graph 
to an NCHW NNEF graph without including any extra operations in the resulting graph.
Of course the user has to pay attention to supply the inputs and process the outputs in the correct order.

In some cases (mainly when defining graphs with strange reshapes, and tensors of different ranks), 
not all the generated transposes/etc. can be optimized away. 
This means that maybe some of the inputs/outputs will be transposed and some will not.

If you convert a Tensorflow graph which has only `data_format='NCHW'` (or 'NCW', or 'NCDHW') operations in it,
there shouldn't be many changes to the graph (other then the transposing of weights which is unavoidable).
