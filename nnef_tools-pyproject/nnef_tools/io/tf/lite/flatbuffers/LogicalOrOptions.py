# automatically generated by the FlatBuffers compiler, do not modify

# namespace: lite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class LogicalOrOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsLogicalOrOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LogicalOrOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def LogicalOrOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # LogicalOrOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def LogicalOrOptionsStart(builder): builder.StartObject(0)
def LogicalOrOptionsEnd(builder): return builder.EndObject()
