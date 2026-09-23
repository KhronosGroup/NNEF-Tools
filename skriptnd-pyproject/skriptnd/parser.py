# Copyright (c) 2017-2025 The Khronos Group Inc.
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

import _skriptnd as _sknd


_StdlibPath = __file__[:-9] + "stdlib/"


def _default_error_callback(position, message, stack, warning):
    print("{} in module '{}' [{}:{}]: {}".format("⚠️Warning" if warning else "🛑Error",
                                                 position.module, position.line, position.column, message))
    for op, pos in reversed(stack):
        print("\t...while calling operator {} in module '{}' [{}:{}]".format(op, pos.module, pos.line, pos.column))


def parse_file(path, attribs=None, error=None, flags=_sknd.DefaultCompilerFlags):
    return _sknd.parse_file(path, stdlib=_StdlibPath, attribs=attribs or {},
                            error_callback=error or _default_error_callback, flags=flags)


def parse_string(text, attribs=None, error=None, flags=_sknd.DefaultCompilerFlags):
    return _sknd.parse_string(text, stdlib=_StdlibPath, attribs=attribs or {},
                              error_callback=error or _default_error_callback, flags=flags)
