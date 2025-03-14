# automatically generated by the FlatBuffers compiler, do not modify

# namespace: lite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ConcatEmbeddingsOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsConcatEmbeddingsOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ConcatEmbeddingsOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def ConcatEmbeddingsOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # ConcatEmbeddingsOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ConcatEmbeddingsOptions
    def NumChannels(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ConcatEmbeddingsOptions
    def NumColumnsPerChannel(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # ConcatEmbeddingsOptions
    def NumColumnsPerChannelAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # ConcatEmbeddingsOptions
    def NumColumnsPerChannelLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ConcatEmbeddingsOptions
    def NumColumnsPerChannelIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # ConcatEmbeddingsOptions
    def EmbeddingDimPerChannel(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # ConcatEmbeddingsOptions
    def EmbeddingDimPerChannelAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # ConcatEmbeddingsOptions
    def EmbeddingDimPerChannelLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ConcatEmbeddingsOptions
    def EmbeddingDimPerChannelIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

def ConcatEmbeddingsOptionsStart(builder): builder.StartObject(3)
def ConcatEmbeddingsOptionsAddNumChannels(builder, numChannels): builder.PrependInt32Slot(0, numChannels, 0)
def ConcatEmbeddingsOptionsAddNumColumnsPerChannel(builder, numColumnsPerChannel): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(numColumnsPerChannel), 0)
def ConcatEmbeddingsOptionsStartNumColumnsPerChannelVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ConcatEmbeddingsOptionsAddEmbeddingDimPerChannel(builder, embeddingDimPerChannel): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(embeddingDimPerChannel), 0)
def ConcatEmbeddingsOptionsStartEmbeddingDimPerChannelVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ConcatEmbeddingsOptionsEnd(builder): return builder.EndObject()
