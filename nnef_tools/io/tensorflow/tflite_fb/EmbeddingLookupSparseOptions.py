# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite_fb

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class EmbeddingLookupSparseOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsEmbeddingLookupSparseOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = EmbeddingLookupSparseOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def EmbeddingLookupSparseOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # EmbeddingLookupSparseOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # EmbeddingLookupSparseOptions
    def Combiner(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def EmbeddingLookupSparseOptionsStart(builder): builder.StartObject(1)
def EmbeddingLookupSparseOptionsAddCombiner(builder, combiner): builder.PrependInt8Slot(0, combiner, 0)
def EmbeddingLookupSparseOptionsEnd(builder): return builder.EndObject()
