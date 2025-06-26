[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center"><img src="https://www.khronos.org/images/jcogs_img/cache/nnef_500px_apr17_-_28de80_-_3c2b17797282ce265889b88b2035b24403f2d049.png" /></p>

# NNEF-Tools

NNEF 2.0 is built around SkriptND, a domain specific language for computations on N-dimensional arrays (tensors). It allows the detailed description of tensor operators, including custom ones, and graphs composed from them, and the compilation of models into executables via automated lowering passes. NNEF 2.0 models are essentially ScriptND source files accompanied by binary data files for model parameters, packaged into a model folder.

The tools are divided into two Python packages:
* [SkriptND parser and sample compiler](skriptnd-pyproject): this package contains a C++ parser implementation with a Python wrapper, along with a sample C++ based code generator that allows a SkriptND model to be compiled into an executable Python object
* [Conversion and compilation tools](nnef_tools-pyproject): this package contains conversion and optimization scripts from model formats of ML frameworks (ONNX, TensorFlow/Lite) to NNEF 2.0, along with integration to the TVM compiler stack as a frontend, allowing compilation of SkriptND models to various CPU and GPU backends (such as Vulkan, OpenCL, Cuda).
