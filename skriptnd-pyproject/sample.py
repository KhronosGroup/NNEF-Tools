import skriptnd as nd
import numpy as np

graph = nd.read_model('alexnet.nds')
model = nd.compile_model(graph)
input = np.random.random((1,3,224,224)).astype(np.float32)
output, = model(input)
print(output.dtype, output.shape)
