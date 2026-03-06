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

import warnings
import tvm
from packaging import version

ver = version.parse(tvm.__version__)
if ver.minor > 19:
    raise ImportError(f"TVM version 0.19 or lower is required, but found {tvm.__version__}")

if ver.minor != 19:
    warnings.warn(f"TVM version 0.19 is recommended, but found {tvm.__version__}. Some features may not work as expected.")

from .from_nnef import from_nnef
