# automatically generated by the FlatBuffers compiler, do not modify

# namespace: lite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class EqualOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsEqualOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = EqualOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def EqualOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # EqualOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def EqualOptionsStart(builder): builder.StartObject(0)
def EqualOptionsEnd(builder): return builder.EndObject()
