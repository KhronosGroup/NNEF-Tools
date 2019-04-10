import _nnef
import os
from nnef.binary import read_tensor


def parse_file(graph_fn, quant_fn=None, stdlib=None, lowered=[]):
    return _nnef.parse_file(graph_fn, quantization=quant_fn, stdlib=stdlib, lowered=lowered)


def parse_string(graph_str, quant_str=None, stdlib=None, lowered=[]):
    return _nnef.parse_string(graph_str, quantization=quant_str, stdlib=stdlib, lowered=lowered)


def load_graph(path, stdlib=None, lowered=[]):
    if os.path.isfile(path):
        return parse_file(path, stdlib=stdlib, lowered=lowered)

    graph_fn = os.path.join(path, 'graph.nnef')
    quant_fn = os.path.join(path, 'graph.quant')

    graph = parse_file(graph_fn, quant_fn if os.path.isfile(quant_fn) else None, stdlib=stdlib, lowered=lowered)

    for operation in graph.operations:
        if operation.name == 'variable':
            variable_filename = os.path.join(path, operation.attribs['label'] + '.dat')
            tensor_name = operation.outputs['output']
            with open(variable_filename) as variable_file:
                data, compression = read_tensor(variable_file)

            data_shape = list(data.shape);
            shape = operation.attribs['shape']
            if data_shape != shape:
                raise _nnef.Error('shape {} in variable file does not match shape {} defined in network structure'
                                  .format(data_shape, shape))

            tensor = graph.tensors[tensor_name]
            graph.tensors[tensor_name] = _nnef.Tensor(tensor.name, tensor.dtype, data_shape, data, compression, tensor.quantization)

    return graph
