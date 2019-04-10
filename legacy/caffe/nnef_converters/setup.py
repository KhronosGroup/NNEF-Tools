#!/usr/bin/env python

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

from __future__ import division, print_function

import os
from setuptools import setup, find_packages
from version import __version__ as version

packages = find_packages(where="..", include=("nnef_converters*",))
package_dir = {package: os.path.join("..", *package.split('.')) for package in packages}

setup(name='nnef_converters_legacy_caffe',
      version=version,
      description='NNEF Converters Legacy caffe',
      url='https://github.com/KhronosGroup/NNEF-Tools',
      author='Tamas Danyluk',
      author_email='tamas.danyluk@aimotive.com',
      license='Apache 2.0',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
      ],
      keywords='nnef',
      packages=packages,
      package_dir=package_dir,
      entry_points={
          'console_scripts': [
              'caffe_to_nnef = nnef_converters.caffe_converters.caffe_to_nnef.command:main',
              'nnef_to_caffe = nnef_converters.caffe_converters.nnef_to_caffe.command:main',
              'create_dummy_caffe_model = nnef_converters.caffe_converters.create_dummy_caffe_model:main',
          ]
      })

print()
print("Some tools need additional dependencies, that are not checked now:")
print("all tools: nnef, numpy")
print("caffe_to_nnef, nnef_to_caffe: caffe")
print()
print("Install successful!")
