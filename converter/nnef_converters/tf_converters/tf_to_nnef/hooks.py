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

# Maybe this hook system is now unnecessary

"""It means that the next hook will handle the event"""
PROPAGATE = object()

"""
Will be called when extras can be added to the tfdn. 
It's recommended to return PROPAGATE from this hook.
"""
HOOK_ADD_EXTRAS_TO_TFDN = "add_extras_to_tfdn"


def empty_hook_add_extras_to_tfdn(tfdn, tf_tensor=None):
    return PROPAGATE
