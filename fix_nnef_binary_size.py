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

import sys
import os
import struct


def fix_nnef_binary(in_fn, out_fn):
    header_size = 128

    with open(in_fn, 'rb') as file:
        file_size = os.fstat(file.fileno()).st_size
        header = file.read(header_size)
        excess = file.read(4)
        data = file.read()

    [magic1, magic2, major, minor] = bytearray(header[:4])
    if magic1 != 0x4E or magic2 != 0xEF or major != 1 or minor != 0:
        return False

    data_length, = struct.unpack('i', header[4:8])

    if file_size != header_size + data_length + 4:
        return False

    with open(out_fn, 'wb') as file:
        file.write(header)
        file.write(data)

    return True


def fix_nnef_binaries(in_path, out_path):
    for root, dirs, files in os.walk(in_path):
        for filename in files:
            if not filename.startswith('.'):
                in_fn = os.path.join(root, filename)
                out_fn = os.path.join(out_path, os.path.relpath(in_fn, in_path))
                if os.path.splitext(filename)[1] == '.dat':
                    if fix_nnef_binary(in_fn, out_fn):
                        print('Fixed file: ' + in_fn)
                elif out_fn != in_fn:
                    with open(in_fn, 'rb') as in_file, open(out_fn, 'wb') as out_file:
                        out_file.write(in_file.read())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('input path must be provided')
        exit(-1)
    elif len(sys.argv) > 3:
        print('too many arguments provided')
        exit(-1)

    fix_nnef_binaries(in_path=sys.argv[1], out_path=sys.argv[2] if len(sys.argv) == 3 else sys.argv[1])
