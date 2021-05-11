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
import shutil
import os


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
                   extra_compile_args=['-std=c++11'])

setup(name='nnef',
      version='0.3',
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
      ext_modules=[module]
      )
