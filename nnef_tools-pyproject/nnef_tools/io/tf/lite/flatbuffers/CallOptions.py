# automatically generated by the FlatBuffers compiler, do not modify

# namespace: lite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class CallOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsCallOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = CallOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def CallOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # CallOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # CallOptions
    def Subgraph(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

def CallOptionsStart(builder): builder.StartObject(1)
def CallOptionsAddSubgraph(builder, subgraph): builder.PrependUint32Slot(0, subgraph, 0)
def CallOptionsEnd(builder): return builder.EndObject()
