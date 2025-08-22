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
        target = tvm.target.Target(target or 'llvm')
        if target.get_target_device_type() != tvm.runtime.Device.kDLCPU and device is None:
            raise ValueError("Device must be specified for CPU targets.")

        model = from_skriptnd(model)
        model = self.addTransforms(model, target)

        exe = tvm.compile(model, target=target)

        self.device = tvm.device(device) if device else tvm.cpu()
        self.vm = tvm.relax.VirtualMachine(exe, self.device)

    def __call__(self, *inputs):
        tvm_input = [tvm.nd.array(input, device=self.device) for input in inputs]
        tvm_output = self.vm["main"](*tvm_input)
        if isinstance(tvm_output, tvm.nd.NDArray):
            tvm_output = (tvm_output,)
        return tuple(output.numpy() for output in tvm_output)

    def addTransforms(self, model, target):
        if target.get_target_device_type() != tvm.runtime.Device.kDLCPU:
            import tvm.dlight as dl, tvm.relax as rx
            with (target):
                model = tvm.ir.transform.Sequential([
                    rx.get_pipeline("zero"),
                    dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                        dl.gpu.Fallback(),
                    ),
                ])(model)

        return model
