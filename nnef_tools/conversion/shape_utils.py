from __future__ import division, print_function, absolute_import


class ShapeUtils(object):
    @staticmethod
    def shape_nhwc_to_nchw(shape):
        return shape[0:1] + shape[-1:] + shape[1:-1]

    @staticmethod
    def shape_nchw_to_nhwc(shape):
        return shape[0:1] + shape[2:] + shape[1:2]

    @staticmethod
    def shape_hwcn_to_nchw(shape):
        return shape[-1:] + shape[-2:-1] + shape[:-2]

    @staticmethod
    def shape_hwcm_to_nchw(shape):
        return [shape[-2] * shape[-1], 1] + shape[:-2]

    @staticmethod
    def shape_nchw_to_hwcn(shape):
        return shape[2:] + shape[1:2] + shape[:1]

    @staticmethod
    def shape_nchw_to_hwcm(shape, input_channels):
        return shape[2:] + [input_channels, shape[0] // input_channels]

    @staticmethod
    def transpose_axes_nhwc_to_nchw(rank):
        return ShapeUtils.shape_nhwc_to_nchw(list(range(rank)))

    @staticmethod
    def transpose_axes_nchw_to_nhwc(rank):
        return ShapeUtils.shape_nchw_to_nhwc(list(range(rank)))

    @staticmethod
    def transpose_axes_hwcn_to_nchw(rank):
        return ShapeUtils.shape_hwcn_to_nchw(list(range(rank)))

    @staticmethod
    def transpose_axes_nchw_to_hwcn(rank):
        return ShapeUtils.shape_nchw_to_hwcn(list(range(rank)))
