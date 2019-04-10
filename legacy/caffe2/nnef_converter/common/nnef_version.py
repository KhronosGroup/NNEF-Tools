# Copyright (c) 2018 The Khronos Group Inc.
# Copyright (c) 2018 Au-Zone Technologies Inc.
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

NNEF_VERSION_MAJOR = 1
NNEF_VERSION_MINOR = 0

class NNEFVersion(object):
    def __init__(self):
        self.major = NNEF_VERSION_MAJOR
        self.minor = NNEF_VERSION_MINOR
        self.version = str(self.major) + "." + str(self.minor)

    