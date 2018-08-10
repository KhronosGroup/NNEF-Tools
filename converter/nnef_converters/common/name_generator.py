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

from .types import *


class NameGenerator(object):
    def __init__(self, used_names=None):
        # type: (List[str])->None
        if used_names is None:
            used_names = []
        self.used_names = set(used_names)  # type: Set[str]
        self.generated_names = set()  # type: Set[str]
        self.counters = dict()  # type: Dict[str, int]

    def get_new_name(self, name):
        # type: (str)->str
        if name not in self.used_names:
            self.used_names.add(name)
            return name

        if name in self.generated_names:
            counter = int(name.split("_")[-1])
            self.counters[name] = max(self.counters.get(name, 1), counter)
            name = name[:name.rfind('_')]

        if name is self.counters:
            counter = self.counters[name]
        else:
            counter = 1

        name_candidate = name + "_" + str(counter)
        counter += 1
        while name_candidate in self.used_names:
            name_candidate = name + "_" + str(counter)
            counter += 1

        self.used_names.add(name_candidate)
        self.generated_names.add(name_candidate)
        self.counters[name] = counter
        return name_candidate

    def get_new_name_with_ending(self, name, ending):
        # type: (str)->str
        if name + ending not in self.used_names:
            self.used_names.add(name + ending)
            return name + ending

        if name is self.counters:
            counter = self.counters[name]
        else:
            counter = 1

        name_candidate = name + "_" + str(counter) + ending
        counter += 1
        while name_candidate in self.used_names:
            name_candidate = name + "_" + str(counter) + ending
            counter += 1

        self.used_names.add(name_candidate)
        self.generated_names.add(name_candidate)
        self.counters[name] = counter
        return name_candidate

