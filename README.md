# NNEF-Tools

This repository contains tools to generate and consume NNEF documents, such as a parser that can be included in consumer applications and converters for deep learning frameworks.

Currently, there are two sets of converters; one provided by Khronos Group (in the `converter` folder) and one provided by the company Au-Zone (in the `au-zone` folder).

The ones provided by Khonos Group convert between Caffe and TensorFlow. The Caffe converter reads/writes models in protobuf format, while the TensorFlow converter reads/writes the Python script that builds the TensorFlow graph

The ones provided by Au-Zone convert between Caffe2 and TensorFlow, both reading/writing models in protobuf format.

**We are working on merging the above converters into a unified tool that can handle all supported formats under the same interface, and publishing it as a Python package.**
