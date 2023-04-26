# Copyright (c) 2017 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import shutil
import os


# This is to add numpy includes after installing numpy in the setup process.
# Based on https://stackoverflow.com/a/21621689
# Since numpy 1.13, the check for __NUMPY_SETUP__ is no longer necessary.
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        import numpy
        self.include_dirs.append(numpy.get_include())


if os.path.exists('build'):
    shutil.rmtree('build')
if os.path.exists('dist'):
    shutil.rmtree('dist')
if os.path.exists('nnef.egg-info'):
    shutil.rmtree('nnef.egg-info')


module = Extension('_nnef',
                   sources=['nnef.cpp'],
                   include_dirs=['../cpp/include'],
                   language='c++',
                   extra_compile_args=['-std=c++11'],
                   define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])

setup(name='nnef',
      version='1.0',
      description='A package for parsing NNEF files',
      url='https://github.com/KhronosGroup/NNEF-Tools',
      author='Viktor Gyenes',
      author_email='viktor.gyenes@aimotive.com',
      license='Apache 2.0',
      classifiers=
      [
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
      ],
      keywords='nnef',
      packages=['nnef'],
      cmdclass={'build_ext':build_ext},
      install_requires=['numpy>=1.13,<2.0'],
      setup_requires=['numpy>=1.13,<2.0'],
      ext_modules=[module]
      )
