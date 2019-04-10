import _nnef
from .parser import *
from .printer import *
from .binary import read_tensor, write_tensor
from .shapes import infer_shapes


Identifier = _nnef.Identifier   # subclass of str
Error = _nnef.Error             # subclass of exception

Graph = _nnef.Graph             # namedtuple('Graph', ['name': str, 'tensors': typing.Dict[str, Tensor], 'operations': typing.List[Operation],
                                #                       'inputs': typing.List[str], 'outputs': typing.List['str']])
Tensor = _nnef.Tensor           # namedtuple('Tensor', ['name': str, 'dtype': str, 'shape': typing.List[int], 'data': numpy.ndarray,
                                #                       'compression': typing.Dict[str, object], 'quantization': Dict[str, object]])
Operation = _nnef.Operation     # namedtuple('Operation', ['name': str, 'attribs': OrderedDict[str, object], 'inputs': OrderedDict[str, object],
                                #                           'outputs': OrderedDict[str, object], 'dtype': str])
