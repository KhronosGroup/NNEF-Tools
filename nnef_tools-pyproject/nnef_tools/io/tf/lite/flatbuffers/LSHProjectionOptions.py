# automatically generated by the FlatBuffers compiler, do not modify

# namespace: lite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class LSHProjectionOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsLSHProjectionOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LSHProjectionOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def LSHProjectionOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # LSHProjectionOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # LSHProjectionOptions
    def Type(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def LSHProjectionOptionsStart(builder): builder.StartObject(1)
def LSHProjectionOptionsAddType(builder, type): builder.PrependInt8Slot(0, type, 0)
def LSHProjectionOptionsEnd(builder): return builder.EndObject()
