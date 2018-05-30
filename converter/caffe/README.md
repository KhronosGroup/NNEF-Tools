Caffe To NNEF converter
==========================

This repository contains Pyhon source code for exporting NNEF from Caffe. 

**Requirements**:

Caffe (and its prerequisites), built with pycaffe enabled. It does not require GPU mode for execution.

If Caffe is built from source and not installed as an apt package:

* CAFFE_BIN_FOLDER environment variable needs to be set (pointing to caffe/build/tools)
* PYTHONPATH needs to include the Caffe python api (default: caffe/python)


**Usage**:

A Caffe model can be exported with the following command:

```python export_nnef_description.py --graph <graphname.prototxt> [--weights <weights.caffemodel>] [--outputs <output1> <output2> <outputN>] [--compress]```

If outputs are provided, only the listed tensors are exported as graph outputs.
If compression is turned on, the result is a single .tgz file, otherwise the conversion results (network structure and parameters) are output as separate files organized into folders.

Network activations (heatmaps) can be exported with the following command:

```python export_nnef_heatmaps.py --graph <graphname.prototxt> [--weights <weights.caffemodel>]```
