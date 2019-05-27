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

import json
import os


class Empty(object):
    pass


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


class CustomEncoderWithClass(json.JSONEncoder):
    def default(self, o):
        d = dict(o.__dict__)
        d["__class__"] = o.__class__.__name__
        return d


class CustomDecoder(object):
    def __init__(self, class_by_name):
        self._class_by_name = class_by_name

    def __call__(self, d):
        if "__class__" in d:
            o = Empty()
            d2 = dict(d)
            del d2["__class__"]
            o.__dict__.update(d2)
            o.__class__ = self._class_by_name[d["__class__"]]
            return o
        return d


def dump(obj, file_name, add_class_name=True, indent=True):
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    with open(file_name, 'w') as f:
        json.dump(obj, f,
                  indent=4 if indent else None,
                  sort_keys=True,
                  cls=CustomEncoderWithClass if add_class_name else CustomEncoder)


def load(file_name, classes=None):
    if classes is None:
        classes = []
    with open(file_name, 'r') as f:
        return json.load(f, object_hook=CustomDecoder({class_.__name__: class_ for class_ in classes}))


def dumps(obj, add_class_name=True, indent=True):
    return json.dumps(obj,
                      indent=4 if indent else None,
                      sort_keys=True,
                      cls=CustomEncoderWithClass if add_class_name else CustomEncoder)


def loads(s, classes=None):
    if classes is None:
        classes = []
    return json.loads(s, object_hook=CustomDecoder({class_.__name__: class_ for class_ in classes}))


__all__ = [
    "dump",
    "load",
    "dumps",
    "loads"
]
