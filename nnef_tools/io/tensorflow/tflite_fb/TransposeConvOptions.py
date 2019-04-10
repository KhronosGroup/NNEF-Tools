# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite_fb

import flatbuffers

class TransposeConvOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsTransposeConvOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TransposeConvOptions()
        x.Init(buf, n + offset)
        return x

    # TransposeConvOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # TransposeConvOptions
    def Padding(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # TransposeConvOptions
    def StrideW(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # TransposeConvOptions
    def StrideH(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def TransposeConvOptionsStart(builder): builder.StartObject(3)
def TransposeConvOptionsAddPadding(builder, padding): builder.PrependInt8Slot(0, padding, 0)
def TransposeConvOptionsAddStrideW(builder, strideW): builder.PrependInt32Slot(1, strideW, 0)
def TransposeConvOptionsAddStrideH(builder, strideH): builder.PrependInt32Slot(2, strideH, 0)
def TransposeConvOptionsEnd(builder): return builder.EndObject()
