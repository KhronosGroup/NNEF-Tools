# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite_fb

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ReshapeOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsReshapeOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReshapeOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ReshapeOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # ReshapeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ReshapeOptions
    def NewShape(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # ReshapeOptions
    def NewShapeAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # ReshapeOptions
    def NewShapeLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ReshapeOptions
    def NewShapeIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def ReshapeOptionsStart(builder): builder.StartObject(1)
def ReshapeOptionsAddNewShape(builder, newShape): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(newShape), 0)
def ReshapeOptionsStartNewShapeVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ReshapeOptionsEnd(builder): return builder.EndObject()
