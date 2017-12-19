NNEF To TensorFlow
==========================

This repository contains Pyhon source code for exporting NNEF from TensorFlow.

The code consistst of two files: `tf2nnef.py` is the export code that does the actual work, and `sample_export.py` is an example that illustrates how to use the exporter, with comments on how to customize the export.

The exporter works by tracing TensorFlow function calls in Python. For this reason, it cannot be used as a standalone tool from outside Python, since it requires the Python function that builds the TensorFlow graph that is to be exported.
