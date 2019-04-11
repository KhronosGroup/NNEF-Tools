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

from __future__ import division, print_function, absolute_import

import os
import shutil
import sys
import tarfile
import tempfile

if sys.version_info > (3, 0):
    import urllib.error
    import urllib.request

    URLError = urllib.error.URLError
    _urlopen = urllib.request.urlopen
else:
    import urllib2

    URLError = urllib2.URLError
    _urlopen = urllib2.urlopen


def _resolve_path(url, path):
    if not path or path[-1] == '/':
        filename = url.split('#')[0].split('?')[0].split('/')[-1]
        assert filename, "Cannot get filename from url: {}".format(url)
        path = path + filename
    return path


def download(url, path="", verbose=True):
    """

    :param url: Must not be urlencoded
    :param path: Can be empty to use same file name
    :param verbose: Print message?
    :return: final path
    """

    path = _resolve_path(url, path)

    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    if verbose:
        print("Downloading {} to {}: ".format(url, path), end="")
        sys.stdout.flush()

    try:
        f = _urlopen(url)
    except URLError:
        if verbose:
            print("\n\nThe file might have been moved on the server! "
                  "Sorry but we can not ensure the correctness of external urls.")
            raise
    try:
        with open(path, "wb+") as g:
            g.write(f.read())
    finally:
        f.close()

    if verbose:
        print("done.")
        sys.stdout.flush()


def download_once(url, path="", verbose=True):
    path = _resolve_path(url, path)

    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    if not os.path.exists(path):
        download(url, path, verbose=verbose)
    else:
        if verbose:
            print("Using existing {}".format(path))
            sys.stdout.flush()

    if verbose:
        print()
        sys.stdout.flush()

    return path


def download_and_untar_once(url, member, path, verbose=True):
    assert path and not path.endswith('/'), "Please specify full path"

    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    if not os.path.exists(path):
        temp_dir = tempfile.mkdtemp()
        try:
            tar_path = os.path.join(temp_dir, 'downloaded.tar')
            download_once(url, tar_path, verbose=verbose)
            with tarfile.open(tar_path, "r:*") as tar:
                assert not any(name.startswith('/')
                               or name.startswith('\\')
                               or ':' in name
                               or '..' in name
                               for name in tar.getnames()), "Tar file has unsafe items, not extracting it."
                if member and member[0] == '*':
                    members = [name for name in tar.getnames() if name.endswith(member[1:])]
                    assert len(members) == 1, "Number of members matching {} must be 1".format(member)
                    member = members[0]

                extract_dir = os.path.join(temp_dir, 'extracted')
                if verbose:
                    print("Extracting {} to {}: ".format(tar_path, extract_dir), end='')
                    sys.stdout.flush()
                tar.extractall(extract_dir)
                if verbose:
                    print("done.")
                    sys.stdout.flush()
                extracted_path = os.path.join(extract_dir, member)
                if verbose:
                    print("Moving {} to {}: ".format(extracted_path, path), end='')
                    sys.stdout.flush()
                shutil.move(extracted_path, path)
                if verbose:
                    print("done.")
                    sys.stdout.flush()
        finally:
            shutil.rmtree(temp_dir)
            if verbose:
                print('Removed {}.'.format(temp_dir))
    else:
        if verbose:
            print("Using cached {}".format(path))
            sys.stdout.flush()
    if verbose:
        print()
        sys.stdout.flush()
    return path
