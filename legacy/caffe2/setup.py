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

import os, sys, re

from setuptools import setup, find_packages
from subprocess import Popen, PIPE

try:
    from pathlib2 import Path
except ImportError:
    try:
        from pathlib import Path
    except ImportError:
        raise ImportError('pathlib or pathlib2 are required')


def _copy(self, target):
    import shutil
    assert self.is_file()
    shutil.copy(str(self), str(target))
Path.copy = _copy


def get_version():
    """
    Returns project version as string from 'git describe' command.
    """
    pipe = Popen('git describe --tags --always', stdout=PIPE, shell=True)
    version = str(pipe.stdout.read().rstrip())
    return re.sub('-g\w+', '', version)

setup (
    name = 'nnef_converter_legacy_caffe2',
    version = get_version(),
    description = 'NNEF Converter',
    author = 'Au-Zone Technologies Inc.',
    author_email = 'support@au-zone.com',
    license = 'Apache 2.0',
    url = 'http://www.au-zone.com',
    packages = find_packages(),
    install_requires = ['nnef', 'networkx', 'protobuf', 'numpy']
)

