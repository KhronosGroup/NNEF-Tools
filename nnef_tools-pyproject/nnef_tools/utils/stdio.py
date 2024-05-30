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

import sys


def set_stdin_to_binary():
    if sys.version_info >= (3, 0):
        sys.stdin = sys.stdin.buffer
    elif sys.platform == 'win32':
        import os, msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)


def set_stdout_to_binary():
    if sys.version_info >= (3, 0):
        sys.stdout = sys.stdout.buffer
    elif sys.platform == 'win32':
        import os, msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)


def is_stdin_piped():
    return not sys.stdin.isatty()


def is_stdout_piped():
    return not sys.stdout.isatty()
