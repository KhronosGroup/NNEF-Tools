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

from setuptools import setup, find_packages
import shutil
import os


if os.path.exists('build'):
    shutil.rmtree('build')
if os.path.exists('dist'):
    shutil.rmtree('dist')
if os.path.exists('nnef_tools.egg-info'):
    shutil.rmtree('nnef_tools.egg-info')


setup(name='nnef_tools',
      version='1.0',
      description='A package for managing NNEF files',
      url='https://github.com/KhronosGroup/NNEF-Tools',
      author='Viktor Gyenes',
      author_email='viktor.gyenes@aimotive.com',
      license='Apache 2.0',
      classifiers=
      [
          'Intended Audience :: Developers',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
      ],
      keywords='nnef',
      packages=['nnef_tools'] + ['nnef_tools.' + package for package in find_packages(where='nnef_tools')]
      )
