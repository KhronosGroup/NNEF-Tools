# NNEF Converters

The `nnef_converters` package consists of different command line tools for conversion between NNEF and the respective 
formats of neural network frameworks. It can convert the graph definitions as well as the trained models.

Current converters:
- `caffe_to_nnef`: Converts Caffe prototxt (and caffemodel) to NNEF
- `nnef_to_caffe`: Converts NNEF to Caffe prototxt (and caffemodel)

Extra tools for testing:
- `create_dummy_caffe_model`: Creates a default initialized Caffe model for a Caffe graph

This package has example networks to experiment on.
They are located in the following directories:
- `nnef_converters/caffe_converters/examples`

## Dependencies
The nnef_converters package can be installed systemwide or to a Python virtualenv.
In the later case, its dependencies must be installed to the same virtualenv.
Python 3 is recommended, but Python 2 is also supported.
We recommend using the latest available version of Python 2 or 3.
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

### Create dummy Caffe model

`create_dummy_caffe_model` has the same parameters as `caffe_to_nnef`, except the `-m`/`--model` parameter.


