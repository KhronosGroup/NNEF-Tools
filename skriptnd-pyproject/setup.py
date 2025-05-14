from setuptools import Extension, setup
import shutil
import numpy
import os
from sys import platform


if os.path.exists("build"):
    shutil.rmtree("build")
if os.path.exists("dist"):
    shutil.rmtree("dist")
if os.path.exists("skriptnd.egg-info"):
    shutil.rmtree("skriptnd.egg-info")


setup(
    ext_modules=[
        Extension(
            name="_skriptnd",
            sources=["skriptnd/skriptnd.cpp", "skriptnd/cpp/src/skriptnd.cpp"],
            include_dirs=["skriptnd/cpp/include", "skriptnd/cpp/include/core",
                          "skriptnd/cpp/include/frontend", "skriptnd/cpp/include/composer",
                          numpy.get_include()],
            language="c++",
            extra_compile_args=["/std:c++17", "/Zc:preprocessor"] if platform in ["win32", "cygwin"] else
                               ["-std=c++17", "-mmacosx-version-min=10.14"] if platform in ["darwin"] else
                               ["-std=c++17"] if os.name != "nt" else [],
        )
    ],
    package_data={"skriptnd.stdlib": ["*.sknd"]}
)
