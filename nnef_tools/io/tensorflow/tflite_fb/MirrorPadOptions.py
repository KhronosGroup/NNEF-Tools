# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite_fb

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class MirrorPadOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsMirrorPadOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MirrorPadOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def MirrorPadOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # MirrorPadOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # MirrorPadOptions
    def Mode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def MirrorPadOptionsStart(builder): builder.StartObject(1)
def MirrorPadOptionsAddMode(builder, mode): builder.PrependInt8Slot(0, mode, 0)
def MirrorPadOptionsEnd(builder): return builder.EndObject()
