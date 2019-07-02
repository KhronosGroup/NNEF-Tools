# Copyright (c) 2017 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division, print_function, absolute_import

import typing
from functools import partial

import numpy as np

from nnef_tools.io.caffe2 import caffe2_pb
from nnef_tools.io.caffe2.caffe2_graph import *
from nnef_tools.core import utils, graph_utils
from nnef_tools.shape_inference import shape_inference as infer

ShapeResult = typing.Union[
    typing.Tuple[typing.List[int], str],
    typing.Tuple[typing.Tuple[typing.List[int], ...], typing.Tuple[str, ...]]
]

DTYPE_FLOAT = 'FLOAT'
DTYPE_BOOL = 'BOOL'
DTYPE_INT32 = 'INT32'
DTYPE_INT64 = 'INT64'
DTYPE_UINT8 = 'UINT8'


def caffe2_pads_to_nnef_padding(pads):
    assert len(pads) % 2 == 0
    return list(zip(pads[:len(pads) // 2], pads[len(pads) // 2:]))


def nnef_padding_to_caffe2_pads(padding):
    ps, qs = utils.zip_inverse(2, padding)
    return ps + qs


def flatten_to_2d(shape, axis):
    return [utils.product(shape[:axis]), utils.product(shape[axis:])]


# Shape inference

def one_element_0d_shape(op, dtype=None):
    # type: (Caffe2Operation, str)->ShapeResult
    return [], dtype if dtype is not None else op.inputs[0].dtype


def one_element_1d_shape(op, dtype=None):
    # type: (Caffe2Operation, str)->ShapeResult
    return [1], dtype if dtype is not None else op.inputs[0].dtype


def first_input_shape(op, n=1, dtype=None):
    # type: (Caffe2Operation, typing.Union[int, str], str)->ShapeResult
    if n == 'auto':
        n = len(op.outputs)

    assert n == len(op.outputs)

    if n == 1:
        return op.inputs[0].shape, op.inputs[0].dtype if dtype is None else dtype
    else:
        return tuple(i.shape for i in op.inputs[:n]), tuple(i.dtype if dtype is None else dtype for i in op.inputs[:n])


def shape_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    graph_utils.replace_tensor_in_consumers(op.graph,
                                            op.output,
                                            Caffe2Tensor(graph=op.graph,
                                                         shape=[op.input.rank],
                                                         data=np.array(op.input.shape, dtype=np.int64),
                                                         dtype=DTYPE_INT64),
                                            remove=False)
    return [op.input.rank], DTYPE_INT64


def prepend_dim_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    shape = op.input.shape
    dim_size = op.attribs['dim_size']
    assert shape[0] % dim_size == 0
    return [dim_size] + [shape[0] // dim_size] + shape[1:], op.input.dtype


def arg_min_max_shape(op, dtype=None):
    # type: (Caffe2Operation, typing.Optional[str])->ShapeResult
    axis = op.attribs.get('axis', -1)
    keep_dims = op.attribs.get('keepdims', 1)
    return infer.reduce(op.inputs[0].shape, axes=[axis], squeeze=not keep_dims), \
           op.inputs[0].dtype if dtype is None else dtype


def reduce_shape(op, dtype=None):
    # type: (Caffe2Operation, typing.Optional[str])->ShapeResult
    axes = op.attribs['axes']
    keep_dims = op.attribs['keepdims']
    return infer.reduce(op.inputs[0].shape, axes=axes, squeeze=not keep_dims), \
           op.inputs[0].dtype if dtype is None else dtype


def no_output_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    return tuple(), tuple()


def conv_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    is_nhwc = op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    return infer.conv(input=op.inputs[0].shape,
                      filter=op.inputs[1].shape[1:-1] if is_nhwc else op.inputs[1].shape[2:],
                      padding=caffe2_pads_to_nnef_padding(op.attribs['pads']),
                      stride=op.attribs['strides'],
                      dilation=op.attribs['dilations'],
                      groups=op.attribs['group'],
                      format=(infer.Format.NHWC if is_nhwc else infer.Format.NCHW),
                      output_channels=op.inputs[1].shape[0]), op.inputs[0].dtype


def conv_transpose_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    is_nhwc = op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    return infer.conv(input=op.inputs[0].shape,
                      filter=op.inputs[1].shape[1:-1] if is_nhwc else op.inputs[1].shape[2:],
                      padding=caffe2_pads_to_nnef_padding(op.attribs['pads']),
                      stride=op.attribs['strides'],
                      dilation=[1] * (op.inputs[0].rank - 2),
                      groups=op.attribs['group'],
                      format=(infer.Format.NHWC if is_nhwc else infer.Format.NCHW),
                      output_channels=op.inputs[1].shape[-1 if is_nhwc else 1] * op.attribs['group'],
                      output_padding=[(0, a) for a in op.attribs['adjs']],
                      deconv=True), op.inputs[0].dtype


def pool_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    is_nhwc = op.attribs.get('order', 'NCHW').upper() == 'NHWC'

    if op.attribs['global_pooling']:
        return infer.reduce(
            op.inputs[0].shape,
            axes=list(range(1, op.inputs[0].rank - 1)) if is_nhwc else list(range(2, op.inputs[0].rank)),
            squeeze=False), op.inputs[0].dtype

    def expand(list, default):
        if is_nhwc:
            return [default] + list + [default]
        else:
            return [default, default] + list

    return infer.sliding_window(input=op.inputs[0].shape,
                                filter=expand(op.attribs['kernels'], 1),
                                padding=expand(caffe2_pads_to_nnef_padding(op.attribs['pads']), (0, 0)),
                                stride=expand(op.attribs['strides'], 1),
                                dilation=expand(op.attribs['dilations'], 1)), op.inputs[0].dtype


def max_pool_with_index_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    shape, dtype = pool_shape(op)
    return (shape, shape), (dtype, DTYPE_INT32)


def lrn_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    return (op.inputs[0].shape,) * len(op.outputs), (op.inputs[0].dtype,) * len(op.outputs)


def concat_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    if op.attribs['add_axis']:
        output_shape = infer.stack([input.shape for input in op.inputs], axis=op.attribs['axis'])
    else:
        output_shape = infer.concat([input.shape for input in op.inputs], axis=op.attribs['axis'])

    graph_utils.replace_tensor_in_consumers(
        op.graph,
        op.outputs[1],
        Caffe2Tensor(graph=op.graph,
                     shape=[len(op.inputs)],
                     data=np.array([input.shape[op.attribs['axis']] for input in op.inputs], dtype=np.int32),
                     dtype=DTYPE_INT32),
        remove=False)

    return (output_shape, [len(op.inputs)]), (op.inputs[0].dtype, DTYPE_INT32)


def dropout_shape(op):
    # type: (Caffe2Operation)->ShapeResult

    if not op.attribs.get('is_test', 0):
        raise utils.NNEFToolsException("Dropout: only is_test=1 is supported.")

    return (op.inputs[0].shape,) * len(op.outputs), (op.inputs[0].dtype,) * len(op.outputs)


def bbox_transform_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    rois, deltas, im_info = op.inputs
    M, fourK = deltas.shape
    N, _ = im_info.shape
    if len(op.outputs) == 1:
        return [M, fourK], op.inputs[0].dtype
    elif len(op.outputs) == 2:
        return ([M, fourK], [N]), (op.inputs[0].dtype, op.inputs[0].dtype)
    else:
        assert False


def batch_matmul_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    A, B = op.inputs

    A_shape = A.shape
    B_shape = B.shape

    if op.attribs.get('trans_a'):
        A_shape = A_shape[:-2] + list(reversed(A_shape[-2:]))

    if op.attribs.get('trans_b'):
        B_shape = B_shape[:-2] + list(reversed(B_shape[-2:]))

    if len(A_shape) == 1:
        A_shape = [None, A_shape[0]]
    if len(B_shape) == 1:
        B_shape = [B_shape[0], None]

    rank = max(len(A_shape), len(B_shape))
    A_shape = [1] * (rank - len(A_shape)) + A_shape
    B_shape = [1] * (rank - len(B_shape)) + B_shape

    assert all(a == b or a == 1 or b == 1 for a, b in zip(A_shape[:-2], B_shape[:-2]))
    assert A_shape[-1] == B_shape[-2]

    shape = utils.without_none([max(a, b) for a, b in zip(A_shape[:-2], B_shape[:-2])]
                               + [A_shape[-2], B_shape[-1]])
    if not shape:
        shape = [1]
    return shape, op.inputs[0].dtype


def fc_shape(op, transposed=False):
    # type: (Caffe2Operation, bool)->ShapeResult
    X, W, b = op.inputs
    axis = op.attribs.get('axis', 1)
    axis_w = op.attribs.get('axis_w', 1)

    if not transposed:
        shape = X.shape[:axis] + [utils.product(W.shape[:axis_w])]
    else:
        shape = X.shape[:axis] + [utils.product(W.shape[axis_w:])]

    return shape, op.inputs[0].dtype


def matmul_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    assert len(op.inputs) == 2
    A, B = op.inputs
    axis_a = op.attribs.get('axis_a', 1)
    axis_b = op.attribs.get('axis_b', 1)
    trans_a = op.attribs.get('trans_a', 0)
    trans_b = op.attribs.get('trans_b', 0)
    return infer.matmul(flatten_to_2d(A.shape, axis_a), flatten_to_2d(B.shape, axis_b),
                        transpose_a=trans_a, transpose_b=trans_b), \
           op.inputs[0].dtype


def brg_nchw_c_to_packed_int8_bgra_stylizer_deprocess_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    N, C, H, W = op.inputs[0].shape
    return [N, H, W, 4], DTYPE_UINT8


def packed_int8_bgra_nhwc_to_nchw_c_stylizer_preprocess_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    N, H, W, C = op.inputs[0].shape
    return [N, 3, H, W], DTYPE_FLOAT


def cast_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    dest_type_id = op.attribs['to']
    return op.inputs[0].shape, caffe2_pb.dtype_id_to_name(dest_type_id)


def conditional_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    cond, true_value, false_value = op.inputs
    return true_value.shape, true_value.dtype


def split_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    if len(op.inputs) == 1:
        sizes = op.attribs['split']
    elif len(op.inputs) == 2:
        if op.inputs[1].data is None:
            raise utils.NNEFToolsException('Split is not supported with calculated sizes.')
        sizes = op.inputs[1].data.tolist()
        op.attribs['split'] = sizes
    else:
        assert False

    op.inputs = (op.inputs[0],)

    output_shapes = tuple(infer.split(input=op.inputs[0].shape, axis=op.attribs['axis'], sizes=sizes))
    return output_shapes, (op.inputs[0].dtype,) * len(output_shapes)


def reshape_shape(op):
    # type: (Caffe2Operation)->ShapeResult

    if len(op.inputs) == 1:
        shape = op.attribs['shape']
    elif len(op.inputs) == 2:
        if op.inputs[1].data is None:
            raise utils.NNEFToolsException('Reshape is not supported with calculated shape.')
        shape = op.inputs[1].data.tolist()
    else:
        assert False

    graph_utils.replace_tensor_in_consumers(op.graph,
                                            op.outputs[1],
                                            Caffe2Tensor(graph=op.graph,
                                                         shape=[op.inputs[0].rank],
                                                         data=np.array(op.inputs[0].shape, dtype=np.int64),
                                                         dtype=DTYPE_INT64),
                                            remove=False)

    op.attribs['shape'] = shape
    op.inputs = (op.inputs[0],)

    return (infer.reshape(op.inputs[0].shape, shape=shape, zero_means_same=True), [op.inputs[0].rank]), \
           (op.inputs[0].dtype, DTYPE_INT64)


def resize_like_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    return op.inputs[1].shape, op.inputs[0].dtype


def squeeze_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    return infer.squeeze(op.inputs[0].shape, axes=op.attribs['dims']), op.inputs[0].dtype


def only_batch_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    return [op.inputs[0].shape[0]], op.inputs[0].dtype


def dot_product_with_padding_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    return [max(op.inputs[0].shape[0], op.inputs[1].shape[0])], op.inputs[0].dtype


def expand_dims_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    return infer.unsqueeze(op.inputs[0].shape, axes=op.attribs['dims']), op.inputs[0].dtype


def flatten_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    axis = op.attribs.get('axis', 1)
    return flatten_to_2d(op.inputs[0].shape, axis), op.inputs[0].dtype


def flatten_to_vec_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    return [utils.product(op.inputs[0].shape)], op.inputs[0].dtype


def generate_proposals_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    n = 1  # not precise
    return ([n, 5], [n]), (op.inputs[0].dtype, op.inputs[0].dtype)


def glu_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    shape = list(op.input.shape)
    shape[-1] //= 2
    return shape, op.input.dtype


def instance_norm_shape(op):
    # type: (Caffe2Operation)->ShapeResult

    if len(op.outputs) == 1:
        return op.inputs[0].shape, op.inputs[0].dtype
    else:
        N = op.inputs[0].shape[0]
        C = op.inputs[1].shape[0]
        return (op.inputs[0].shape, [N, C], [N, C]), (op.inputs[0].dtype,) * 3


def box_with_nms_limit_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    _count, num_classes = op.inputs[0].shape
    n = 1  # not precise

    shapes = ([n], [n, 4], [n], [n], [n], [num_classes])
    dtypes = (DTYPE_FLOAT, DTYPE_FLOAT, DTYPE_FLOAT, DTYPE_FLOAT, DTYPE_INT32, DTYPE_INT32)
    return shapes[:len(op.outputs)], dtypes[:len(op.outputs)]


def layer_norm_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    axis = op.attribs.get('axis', 1)
    return (op.inputs[0].shape, op.inputs[0].shape[:axis] + [1], op.inputs[0].shape[:axis] + [1]), \
           (op.inputs[0].dtype, op.inputs[0].dtype, op.inputs[0].dtype)


def merge_dim_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    if op.inputs[0].rank >= 2:
        return [op.inputs[0].shape[0] * op.inputs[0].shape[1]] + op.inputs[0].shape[2:], op.inputs[0].dtype
    else:
        return op.inputs[0].shape, op.inputs[0].dtype


def pad_image_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    is_nhwc = op.attribs.get('order', 'NCHW').upper() == 'NHWC'

    def expand(list, default):
        if is_nhwc:
            return [default] + list + [default]
        else:
            return [default, default] + list

    return infer.pad(input=op.inputs[0].shape,
                     padding=expand(caffe2_pads_to_nnef_padding(op.attribs['pads']), (0, 0))), op.inputs[0].dtype


def quant_decode_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    if len(op.inputs) == 2:
        return op.inputs[1].shape, op.inputs[1].dtype
    elif len(op.inputs) >= 3:
        return tuple(i.shape for i in op.inputs[1:]), tuple(i.dtype for i in op.inputs[1:])
    else:
        assert False


def resize_nearest_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    assert op.inputs[0].rank == 4
    width_scale = op.attribs.get('width_scale', 1.0)
    height_scale = op.attribs.get('height_scale', 1.0)
    is_nhwc = op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    shape = op.inputs[0].shape
    if is_nhwc:
        return [shape[0], int(shape[1] * height_scale), int(shape[2] * width_scale), shape[3]], op.inputs[0].dtype
    else:
        return [shape[0], shape[1], int(shape[2] * height_scale), int(shape[3] * width_scale)], op.inputs[0].dtype


def roi_align_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    is_nhwc = op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    x, rois = op.inputs
    R, _4or5 = rois.shape
    pooled_h = op.attribs.get('pooled_h', 1)
    pooled_w = op.attribs.get('pooled_w', 1)
    if is_nhwc:
        N, H, W, C = x.shape
        return [R, pooled_h, pooled_w, C], x.dtype
    else:
        N, C, H, W = x.shape
        return [R, C, pooled_h, pooled_w], x.dtype


def roi_pool_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    if not op.attribs.get('is_test', 0):
        raise utils.NNEFToolsException("RoIPool: only is_test=1 is supported.")

    is_nhwc = op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    x, rois = op.inputs
    num_rois, _5 = rois.shape
    pooled_h = op.attribs.get('pooled_h', 1)
    pooled_w = op.attribs.get('pooled_w', 1)
    if is_nhwc:
        N, H, W, C = x.shape
        shape = [num_rois, pooled_h, pooled_w, C]
    else:
        N, C, H, W = x.shape
        shape = [num_rois, C, pooled_h, pooled_w]

    if len(op.outputs) == 1:
        return shape, x.dtype
    else:
        return (shape, shape), (x.dtype, DTYPE_INT32)


def size_shape(op):
    # type: (Caffe2Operation)->ShapeResult

    graph_utils.replace_tensor_in_consumers(op.graph,
                                            op.outputs[0],
                                            Caffe2Tensor(graph=op.graph,
                                                         shape=[],
                                                         data=np.array(op.inputs[0].count, dtype=np.int64),
                                                         dtype=DTYPE_INT64),
                                            remove=False)

    return one_element_0d_shape(op)


def slice_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    # Currently, only slicing in a single dimension is supported in Caffe2

    if len(op.inputs) == 1:
        starts = op.attribs['starts']
        ends = op.attribs['ends']
    elif len(op.inputs) == 3:
        if op.inputs[1].data is None:
            raise utils.NNEFToolsException('Slice is not supported with calculated sizes.')
        if op.inputs[2].data is None:
            raise utils.NNEFToolsException('Slice is not supported with calculated sizes.')
        starts = op.inputs[1].data.tolist()
        ends = op.inputs[2].data.tolist()
    else:
        assert False

    op.attribs = {
        'starts': starts,
        'ends': ends,
    }

    op.inputs = (op.inputs[0],)

    return infer.slice(op.inputs[0].shape,
                       begin=starts,
                       end=[e + 1 if e < 0 else e for e in ends],
                       zero_means_all=True), op.input.dtype


def spatial_bn_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    if not op.attribs.get('is_test', 0):
        raise utils.NNEFToolsException("SpatialBN: only is_test=1 is supported.")

    is_nhwc = op.attribs.get('order', 'NCHW').upper() == 'NHWC'
    c = op.inputs[0].shape[-1] if is_nhwc else op.inputs[0].shape[1]
    if len(op.outputs) == 1:
        return op.inputs[0].shape, op.inputs[0].dtype

    assert len(op.outputs) == 5
    return (op.inputs[0].shape, [c], [c], [c], [c]), (op.inputs[0].dtype,) * 5


def range_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    if len(op.inputs) == 1:
        start = 0
        if op.inputs[0].data is None:
            raise utils.NNEFToolsException('Range is not supported with calculated sizes.')
        stop = op.inputs[0].data.tolist()
        step = 1
    elif len(op.inputs) == 2:
        if op.inputs[0].data is None:
            raise utils.NNEFToolsException('Range is not supported with calculated sizes.')
        start = op.inputs[0].data.tolist()
        if op.inputs[1].data is None:
            raise utils.NNEFToolsException('Range is not supported with calculated sizes.')
        stop = op.inputs[1].data.tolist()
        step = 1
    elif len(op.inputs) == 3:
        if op.inputs[0].data is None:
            raise utils.NNEFToolsException('Range is not supported with calculated sizes.')
        start = op.inputs[0].data.tolist()
        if op.inputs[1].data is None:
            raise utils.NNEFToolsException('Range is not supported with calculated sizes.')
        stop = op.inputs[1].data.tolist()
        if op.inputs[2].data is None:
            raise utils.NNEFToolsException('Range is not supported with calculated sizes.')
        step = op.inputs[2].data.tolist()
    else:
        assert False

    return [len(np.arange(start, stop, step))], op.inputs[0].dtype


def summarize_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    if len(op.outputs) == 0:
        return tuple(), tuple()
    elif len(op.outputs) == 1:
        return [4], op.inputs[0].dtype


def tile_shape(op):
    # type: (Caffe2Operation)->ShapeResult

    assert len(op.inputs) in [1, 2, 3]

    tiles = op.attribs.get('tiles', 1)
    axis = op.attribs.get('axis', 0)

    if len(op.inputs) >= 2:
        if op.inputs[1].data is None:
            raise utils.NNEFToolsException('Tile is not supported with calculated sizes.')
        tiles = op.inputs[1].data.item()
    if len(op.inputs) >= 3:
        if op.inputs[2].data is None:
            raise utils.NNEFToolsException('Tile is not supported with calculated sizes.')
        axis = op.inputs[2].data.item()

    repeats = [1] * op.inputs[0].rank
    repeats[axis] = tiles

    op.attribs['tiles'] = tiles
    op.attribs['axis'] = axis
    op.inputs = (op.inputs[0],)

    return infer.tile(op.inputs[0].shape, repeats), op.inputs[0].dtype


def topk_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    shape = list(op.input.shape)
    shape[-1] = int(op.attribs['k'])

    shapes = (shape, shape, [utils.product(shape)])
    dtypes = (op.input.dtype, DTYPE_INT64, DTYPE_INT64)
    return shapes[:len(op.outputs)], dtypes[:len(op.outputs)]


def transpose_shape(op):
    # type: (Caffe2Operation)->ShapeResult

    axes = op.attribs.get('axes')
    if not axes:
        axes = list(range(op.input.rank))[::-1]

    return infer.transpose(op.input.shape, axes), op.input.dtype


def where_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    assert not op.attribs.get('broadcast_on_rows')
    return op.inputs[1].shape, op.inputs[1].dtype


def channel_stats_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    is_nhwc = op.attribs.get('order', 'NCHW').upper() == 'NHWC'

    C = op.input.shape[-1 if is_nhwc else 1]

    return ([C], [C]), (op.inputs[0].dtype, op.inputs[0].dtype)


def sum_reduce_like_shape(op):
    # type: (Caffe2Operation)->ShapeResult
    input, reference = op.inputs

    axis = op.attribs['axis']
    target = [None if i < axis or i >= axis + reference.rank else min(input.shape[i], reference.shape[i - axis])
              for i in range(input.rank)]
    axes_keep = [i for i in range(input.rank) if target[i] == 1 and input.shape[i] != 1]
    axes_squeeze = [i for i in range(input.rank) if target[i] is None]

    tmp_shape = infer.reduce(input.shape, axes_keep, squeeze=False)
    shape = infer.reduce(tmp_shape, axes_squeeze, squeeze=True)

    return shape, input.dtype


# Infer shape

def infer_shape(op, custom_shapes):
    func = _DefaultShapeFuncs.get(op.name)
    if func is None:
        func = custom_shapes.get(op.name)
    if not func:
        raise utils.NNEFToolsException("shape inference not defined for operation '{}'".format(op.name))
    shapes, dtypes = func(op)

    if len(op.outputs) == 1:
        if isinstance(shapes, tuple):
            shapes = shapes[0]
        if isinstance(dtypes, tuple):
            dtypes = dtypes[0]
        op.outputs[0].shape = list(shapes)
        op.outputs[0].dtype = dtypes
    else:
        for tensor, shape, dtype in zip(op.outputs, shapes, dtypes):
            tensor.shape = list(shape)
            tensor.dtype = dtype


# The list only includes the somewhat relevant ops
# Operations that are unified away are listed in comments
_DefaultShapeFuncs = {
    'Abs': first_input_shape,
    'Add': first_input_shape,
    'Alias': None,
    'And': first_input_shape,
    'ArgMax': partial(arg_min_max_shape, dtype=DTYPE_INT64),
    'ArgMin': partial(arg_min_max_shape, dtype=DTYPE_INT64),
    'Assert': no_output_shape,
    'AveragePool': pool_shape,  # AveragePool1D, AveragePool2D, AveragePool3D
    'BBoxTransform': bbox_transform_shape,
    'BRGNCHWCToPackedInt8BGRAStylizerDeprocess': brg_nchw_c_to_packed_int8_bgra_stylizer_deprocess_shape,
    'BatchGather': None,
    'BatchMatMul': batch_matmul_shape,
    'BatchToSpace': None,
    'BooleanMask': None,
    'BooleanUnmask': None,
    'BoxWithNMSLimit': box_with_nms_limit_shape,
    'Cast': cast_shape,
    'Ceil': first_input_shape,
    'ChannelShuffle': first_input_shape,
    'ChannelStats': channel_stats_shape,
    'Clip': first_input_shape,
    'ClipTensorByScaling': None,
    'Col2Im': None,
    'Concat': concat_shape,  # DepthConcat, Append
    'Conditional': conditional_shape,
    'Conv': conv_shape,  # Conv1D, Conv2D, Conv3D
    'ConvTranspose': conv_transpose_shape,
    'Copy': first_input_shape,  # Copy, CopyFromCPUInput, CopyOnDeviceLike, EnsureCPUOutput, StopGradient
    'Cos': first_input_shape,
    'Div': first_input_shape,
    'DotProduct': only_batch_shape,
    'DotProductWithPadding': dot_product_with_padding_shape,
    'Dropout': dropout_shape,
    'EQ': partial(first_input_shape, dtype=DTYPE_BOOL),
    'ElementwiseLinear': first_input_shape,
    'Elu': first_input_shape,
    'Exp': first_input_shape,
    'ExpandDims': expand_dims_shape,
    'FC': fc_shape,
    'FCTransposed': partial(fc_shape, transposed=True),
    'Flatten': flatten_shape,
    'FlattenToVec': flatten_to_vec_shape,
    'FlexibleTopK': None,
    'Floor': first_input_shape,
    'GE': partial(first_input_shape, dtype=DTYPE_BOOL),
    'GRUUnit': None,
    'GT': partial(first_input_shape, dtype=DTYPE_BOOL),
    'Gather': None,
    'GenerateProposals': generate_proposals_shape,  # GenerateProposalsCPP
    'Glu': glu_shape,
    'Im2Col': None,
    'InstanceNorm': instance_norm_shape,
    'L1Distance': only_batch_shape,
    'LC': None,  # has other versions too
    'LE': partial(first_input_shape, dtype=DTYPE_BOOL),
    'LRN': lrn_shape,
    'LT': partial(first_input_shape, dtype=DTYPE_BOOL),
    'LayerNorm': layer_norm_shape,
    'LeakyRelu': first_input_shape,
    'Log': first_input_shape,
    'Logit': first_input_shape,
    'LpNorm': one_element_1d_shape,
    'LpPool': pool_shape,
    'MatMul': matmul_shape,
    'Max': first_input_shape,
    'MaxPool': pool_shape,  # MaxPool1D, MaxPool2D, MaxPool3D
    'MaxPoolWithIndex': max_pool_with_index_shape,  # not in doc, only in .cu file
    'Mean': first_input_shape,
    'MergeDim': merge_dim_shape,
    'Min': first_input_shape,
    'Mod': first_input_shape,
    'Mul': first_input_shape,
    'NanCheck': first_input_shape,
    'NE': first_input_shape,  # not in doc
    'Negative': first_input_shape,
    'Normalize': first_input_shape,
    'NormalizeL1': first_input_shape,
    'NormalizePlanarYUV': first_input_shape,
    'Not': first_input_shape,
    'Or': first_input_shape,
    'PRelu': first_input_shape,
    'PackedInt8BGRANHWCToNCHWCStylizerPreprocess': packed_int8_bgra_nhwc_to_nchw_c_stylizer_preprocess_shape,
    'PadImage': pad_image_shape,
    'Perplexity': one_element_1d_shape,
    'PiecewiseLinearTransform': first_input_shape,
    'Pow': first_input_shape,
    'PrependDim': prepend_dim_shape,
    'Print': no_output_shape,
    'QuantDecode': quant_decode_shape,
    'Range': range_shape,
    'ReduceMin': reduce_shape,  # not in doc
    'ReduceMax': reduce_shape,  # not in doc, ReduceFrontMax, ReduceBackMax, ColwiseMax, RowwiseMax
    # not in doc, ReduceFrontSum, ReduceBackSum, ReduceTailSum, SumElements, SumElementsInt
    'ReduceSum': reduce_shape,
    'ReduceMean': reduce_shape,  # ReduceFrontMean, ReduceBackMean
    'ReduceL1': reduce_shape,  # not in doc
    'ReduceL2': reduce_shape,  # not in doc
    'Relu': first_input_shape,
    'ReplaceNaN': first_input_shape,
    'Reshape': reshape_shape,
    'ResizeLike': resize_like_shape,
    'ResizeNearest': resize_nearest_shape,
    'RoIAlign': roi_align_shape,
    'RoIPool': roi_pool_shape,
    'RowMul': first_input_shape,
    'Scale': first_input_shape,
    'Selu': first_input_shape,
    'Shape': shape_shape,
    'Sigmoid': first_input_shape,
    'Sign': first_input_shape,
    'Sin': first_input_shape,
    'Size': size_shape,
    'Slice': slice_shape,
    'Softmax': first_input_shape,
    'Softplus': first_input_shape,
    'Softsign': first_input_shape,
    'SpaceToBatch': None,
    'SpatialBN': spatial_bn_shape,
    'Split': split_shape,  # DepthSplit
    'Sqr': first_input_shape,
    'Sqrt': first_input_shape,
    'SquaredL2Distance': only_batch_shape,
    'Squeeze': squeeze_shape,
    'StumpFunc': first_input_shape,
    'Sub': first_input_shape,
    'Sum': first_input_shape,
    'SumInt': first_input_shape,
    'SumSqrElements': one_element_0d_shape,
    'SumReduceLike': sum_reduce_like_shape,
    'Summarize': summarize_shape,
    'Swish': first_input_shape,
    'TT': None,
    'Tanh': first_input_shape,
    'ThresholdedRelu': first_input_shape,
    'Tile': tile_shape,
    'TopK': topk_shape,
    'Transpose': transpose_shape,  # NCHW2NHWC, NHWC2NCHW
    'Unique': None,
    'WallClockTime': partial(one_element_1d_shape, dtype=DTYPE_INT64),
    'WeightedSum': first_input_shape,
    'Where': where_shape,
    'Xor': first_input_shape,
    'ZeroGradient': no_output_shape,
}
