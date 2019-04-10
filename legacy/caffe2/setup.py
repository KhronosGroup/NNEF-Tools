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

