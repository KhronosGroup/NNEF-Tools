# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite_fb

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class MatrixSetDiagOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsMatrixSetDiagOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MatrixSetDiagOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def MatrixSetDiagOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # MatrixSetDiagOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def MatrixSetDiagOptionsStart(builder): builder.StartObject(0)
def MatrixSetDiagOptionsEnd(builder): return builder.EndObject()
