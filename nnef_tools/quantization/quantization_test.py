import nnef
from nnef_tools.io.nnef.nnef_graph import *
from nnef_tools.io.nnef import nnef_io
import nnef_tools.backend.pytorch as backend
import numpy as np
import subprocess
import tflite_runtime.interpreter as tflite

from dequantize import dequantize, _const, _op, _var
graph = NNEFGraph()

def _conv(graph, input, weights_np, bias_np, idx, weights_quant=None, output_quant=None):
    weights = _var(graph,
                   "weights_{}".format(idx),
                   weights_np,
                   quantization=weights_quant)
    bias_quant = NNEFQuantization('tflite_quantize',
                                  attribs=dict(bits=32,
                                               scale=weights_quant.attribs['scale'] * input.quantization.attribs['scale'],
                                               zero_point=0, min=0.0, max=0.0)) if weights_quant != None and input.quantization != None else None

    bias = _var(graph,
                "bias_{}".format(idx),
                bias_np,
                quantization=bias_quant)

    return _op(graph, 'conv', idx,
               output_quant=output_quant,
               inputs=(input, weights, bias),
               attribs=dict(border='constant', padding=[]))

input = NNEFTensor(graph=graph,
                   shape=[1,1,3,3],
                   name="input",
                   dtype='scalar',
                   quantization=NNEFQuantization('tflite_quantize', attribs=dict(bits=8, scale=1/128, zero_point=128, min=0.0, max=0.0)))

conv0, conv_op0 = _conv(graph, input, idx=0,
                      weights_np=np.array(np.array([[[[128,128,128],[64,255,64],[128,128,128]]]]), dtype=np.uint8),
                      weights_quant=NNEFQuantization('tflite_quantize', attribs=dict(bits=8, scale=1/1, zero_point=128, min=0.0, max=0.0)),
                      bias_np=np.array(np.array([[0]]), dtype=np.int32),
                      output_quant=NNEFQuantization('tflite_quantize', attribs=dict(bits=8, scale=1/128, zero_point=128, min=0.0, max=0.0))
                      )

conv1, conv_op1 = _conv(graph, conv0, idx=1,
                      weights_np=np.array(np.array([[[[128,128,128],[128,255,128],[128,128,128]]]]), dtype=np.uint8),
                      weights_quant=NNEFQuantization('tflite_quantize', attribs=dict(bits=8, scale=1/128, zero_point=150, min=0.0, max=0.0)),
                      bias_np=np.array(np.array([[0]]), dtype=np.int32),
                      output_quant=NNEFQuantization('tflite_quantize', attribs=dict(bits=8, scale=1/128, zero_point=150, min=0.0, max=0.0))
                      )

graph.inputs = [input]
graph.outputs = [conv1]

writer = nnef_io.Writer(only_print_used_fragments=True)
writer(graph, "models/quant_test.nnef")
cmd = "NNEF-Tools/nnef_tools/convert.py --input-format nnef --output-format tensorflow-lite --input-model models/quant_test.nnef/ --output-model models/ --io-transformation IDENTITY --conversion-info"
subprocess.run(cmd.split(" "))

dequantize(graph)

for t in graph.inputs:
    print(t)

for t in graph.outputs:
    print(t)

for op in graph.operations:
    print(op)

inp = np.array([[[[128, 128, 128],[255,255,255],[128,128,128]]]], dtype=np.uint8)
output = backend.run(nnef_graph=graph, inputs=inp)
print(output)

interpreter = tflite.Interpreter(model_path="models/quant_test.nnef.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], inp)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
print(output)