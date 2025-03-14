from setuptools import Extension, setup
import numpy
from os import name as os_name
from sys import platform

setup(
    ext_modules=[
        Extension(
            "_skriptnd",
            sources=["skriptnd/skriptnd.cpp", "skriptnd/cpp/src/skriptnd.cpp"],
            include_dirs=["skriptnd/cpp/include", "skriptnd/cpp/include/core",
                          "skriptnd/cpp/include/frontend", "skriptnd/cpp/include/composer",
                          numpy.get_include()],
            language="c++",
            extra_compile_args=["/std:c++17", "/Zc:preprocessor"] if platform in ["win32", "cygwin"] else
                               ["-std=c++17", "-mmacosx-version-min=10.14"] if platform in ["darwin"] else
                               ["-std=c++17"] if os_name != "nt" else [],
        )
    ],
)
