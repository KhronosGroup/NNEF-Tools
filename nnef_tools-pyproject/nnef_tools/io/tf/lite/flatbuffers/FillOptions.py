# automatically generated by the FlatBuffers compiler, do not modify

# namespace: lite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class FillOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsFillOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FillOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def FillOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # FillOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def FillOptionsStart(builder): builder.StartObject(0)
def FillOptionsEnd(builder): return builder.EndObject()
