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

from .skriptnd_frontend import from_skriptnd
import tvm


class VirtualMachine:

    def __init__(self, model, target=None, device=None):
        model = from_skriptnd(model)
        target = tvm.target.Target(target or 'llvm')
        exe = tvm.compile(model, target=target)
        self.device = tvm.device(device) if device else tvm.cpu()
        self.vm = tvm.relax.VirtualMachine(exe, self.device)

    def __call__(self, *inputs):
        tvm_input = [tvm.nd.array(input, device=self.device) for input in inputs]
        tvm_output = self.vm["main"](*tvm_input)
        if isinstance(tvm_output, tvm.nd.NDArray):
            tvm_output = (tvm_output,)
        return tuple(output.numpy() for output in tvm_output)
