from .sknd_tvm_converter import from_sknd, convert_dtype
import tvm


class VirtualMachine:

    def __init__(self, model, target=None, device=None):
        model = from_sknd(model)
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
