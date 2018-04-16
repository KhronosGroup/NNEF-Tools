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

import os
import shutil
from setuptools import setup, Extension


module = Extension('_nnef',
                   sources = ['nnef.cpp'],
                   language='c++',
                   extra_compile_args=['-std=c++11'])

include_path = 'include'
include_subdirs = ['common', 'comp', 'flat']

if not os.path.isdir(include_path):
    os.makedirs(include_path)
    for subdir in include_subdirs:
        shutil.copytree('../' + subdir, include_path + '/' + subdir)

setup(name = 'nnef',
	  version = '1.0',
	  description = 'A package for parsing NNEF files',
      url = 'https://github.com/KhronosGroup/NNEF-Tools',
      author = 'Viktor Gyenes',
      author_email = 'viktor.gyenes@aimotive.com',
      license = 'Apache 2.0',
      classifiers =
      [
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 3',
      ],
      keywords='nnef',
      py_modules=['nnef'],
      ext_modules=[module]
)

shutil.rmtree(include_path)
