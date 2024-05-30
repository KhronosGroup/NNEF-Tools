# Copyright (c) 2020 The Khronos Group Inc.
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
import tarfile


def tgz_compress(dir_path, file_path, compression_level=0):
    target_directory = os.path.dirname(file_path)
    if target_directory and not os.path.exists(target_directory):
        os.makedirs(target_directory)

    with tarfile.open(file_path, 'w:gz', compresslevel=compression_level) as tar:
        for file_ in os.listdir(dir_path):
            tar.add(dir_path + '/' + file_, file_)


def tgz_extract(file_path, dir_path):
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(dir_path)
