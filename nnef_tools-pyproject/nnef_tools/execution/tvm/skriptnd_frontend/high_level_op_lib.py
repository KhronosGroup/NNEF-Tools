"""High level operation library for NNEF 2.0 (SkriptND) to TVM Relax operation conversion"""

from __future__ import annotations

import logging
from types import MethodType

import numpy as np
import tvm
from tvm import relax, topi
from tvm.relax import op as _op

from . import ConverterError

def _gen_shape_list(var: relax.Var):
    # todo-- check for dynamic - should work
    return [s.value for s in var.struct_info.shape.values]


def _map_data_format(data_format: str, ndim: int) -> str:
    """
    Map SkriptND channel position format specifier to general data format

    :param data_format: NCX / NXC / XCN
    :param ndim: Num of spatial dimensions
    :return: N[C]S..[C] format string
    """
    spatials = "HWD"[:ndim]
    ch_first = data_format == "NCX"
    return f"N{'C' if ch_first else ''}{spatials}{'' if ch_first else 'C'}"


# todo exceptions .warn_msg if more info is needed
class NotInMapError(ConverterError):
    """Operation does not exist in the high level operation mappings"""
    pass


# todo exceptions .warn_msg if more info is needed
class DimensionError(ConverterError):
    """No valid relax implementation for the given dimension"""
    pass

class UnsupportedAttr(ConverterError):
    """Unsupported SkriptND attribute, or attribute value"""
    def __init__ (self, op, attr, value):
        super().__init__(op, attr, value)
        self.op = op
        self.attr = attr
        self.value = value

    def warn_msg(self):
        return f"Unsupported attribute '{self.attr}' with value '{self.value}' for operation '{self.op}'"


class convert_map:
    """Dictionary like class for storing conversion functions

    The naming of the functions are the normalized name of the SkriptND operation as such:
        `package.operation -> package_operation`

    Additional conversion functions can be added, the entries should have the following signatures:
        `dict[str, Callable[[self, *inputs, OperationInfo], relax.Call]]`

        - the dict key should be the exact name of the operation in the SkriptND script
        - the class wraps the function into the class, providing the instance as first parameter
        - the following function parameters should consist of the unrolled SkriptND inputs, and the OperationInfo object containing the operation attributes, preprocessed

    The class' getitem method returns the correct conversion function if possible, otherwise raises the appropriate error
    Only breaking errors are in the algo module, as no fallback is implemented for them, other errors will fall back to contraction conversions, as well as misc. mesh grid

    The conversion functions can handle the following issues:
        - unsupported operations
        - unsupported attributes, values
        - unsupported dimensions
        - unsupported tvm intrinsic usage (e.g. deconv has limited intrinsic support)

    The custom converter map has precedence over the default conversion functions, overload is possible.

    """

    def __init__(self, block_builder: relax.BlockBuilder, custom_convert_map: dict = None, intrinsic_map: dict = None):
        """

        :param block_builder: Current BlockBuilder instance
        :param custom_convert_map: Additional conversion functions
        :param intrinsic_map: intrinsic checking dict for operations that have limited intrinsic support
        """
        self.bb = block_builder
        self.use_intrinsic = intrinsic_map if intrinsic_map else {}
        self.custom_map = custom_convert_map if custom_convert_map else {}

        for key, value in self.custom_map.items():
            self.custom_map[key] = MethodType(value, self)


    # algo module - Algorithmic operators

    def algo_top_k(self, input, attribs):
        """TopK operation - sorted attrib is not supported"""
        k = attribs.ts_attrs["k"]
        axis = attribs.ts_attrs["axis"]
        largest = attribs.ts_attrs["largest"]
        sorted = attribs.ts_attrs["sorted"]

        if not sorted:
            raise UnsupportedAttr("TopK", "sorted", sorted)

        return _op.topk(input, k=k, axis=axis, largest=largest, ret_type="both")

    def algo_nonmax_suppress(self, boxes, scores, attribs):
        # TODO nonmax suppress, not yet supported firsthand by relax
        pass
        raise NotImplementedError

    # image module - Image processing operators
    # used generalization for all image processing operators with image resize

    def image_nearest_downsample(self, input, attribs):
        return self._image_resize_helper(input, attribs, "nearest_neighbor", True)

    def image_nearest_upsample(self, input, attribs):
        return self._image_resize_helper(input, attribs, "nearest_neighbor")

    def image_area_downsample(self, input, attribs):
        return self._image_resize_helper(input, attribs, "linear", True)

    def image_linear_upsample(self, input, attribs):
        return self._image_resize_helper(input, attribs, "linear")

    def image_nearest_resize(self, input, attribs):
        return self._image_resize_helper(input, attribs, "nearest_neighbor")

    def image_linear_resize(self, input, attribs):
        return self._image_resize_helper(input, attribs, "linear")

    def image_cubic_resize(self, input, attribs):
        return self._image_resize_helper(input, attribs, "cubic")

    def image_resize(self, input, attribs):
        return self._image_resize_helper(input, attribs, "")

    def image_rescale(self, input, attribs):
        return self._image_resize_helper(input, attribs, "")

    # todo max roi pool, avg roi pool ??


    # TODO write these with internal call to nn functions
    def layer_linear(self, input, attribs):
        channels = attribs.ts_attrs["channels"]
        use_bias = attribs.ts_attrs["use_bias"]

        # todo check variables?

        filter = None  # ("filter", shape=[channels, input.struct_info.shape[1]])
        bias = None  # _op.var("bias", shape=[channels]) if use_bias else None

        return _op.linear(input, filter, bias)

    # todo conv deconv bn

    def layout_constant(self, attribs):
        shape = attribs.ts_attrs["shape"]
        value = attribs.ts_attrs["value"]
        dtype = attribs.output_dtype[0]

        if not isinstance(value, relax.Expr):
            return relax.const(np.array(value).reshape(shape), dtype)

        return relax.op.full(shape, value, dtype)

    def layout_uniform(self, value, attribs):
        shape = attribs.ts_attrs["shape"]
        dtype = attribs.output_dtype[0]

        return _op.full(shape, value, dtype)

    def layout_range(self, attribs):
        first = attribs.ts_attrs["first"]
        last = attribs.ts_attrs["last"]
        stride = attribs.ts_attrs["stride"]
        dtype = attribs.output_dtype[0]

        return _op.arange(first, last, stride, dtype)

    def layout_shape(self, input, attribs):
        return _op.shape_to_tensor(_op.shape_of(input))

    def layout_cast(self, input, attribs):
        dtype = attribs.output_dtype[0]
        return _op.astype(input, dtype)

    def layout_reshape(self, input, attribs):
        axis = attribs.ts_attrs["axis"]
        rank = attribs.ts_attrs["rank"]
        shape = attribs.ts_attrs["shape"]

        old_shape = _gen_shape_list(input)
        new_shape = old_shape[:axis] + shape + old_shape[axis + rank:]

        return _op.reshape(input, new_shape)

    def layout_flatten(self, input, attribs):
        axis = attribs.ts_attrs["axis"]
        rank = attribs.ts_attrs["rank"]

        old_shape = _gen_shape_list(input)
        new_shape = old_shape[:axis] + [np.prod(old_shape[axis:axis + rank])] + old_shape[axis + rank:]

        return _op.reshape(input, new_shape)

    def layout_unflatten(self, input, attribs):
        axis = attribs.ts_attrs["axis"]
        shape = attribs.ts_attrs["shape"]

        old_shape = _gen_shape_list(input)
        new_shape = old_shape[:axis] + shape + old_shape[axis + 1:]

        return _op.reshape(input, new_shape)

    def layout_squeeze(self, input, attribs):
        axes = attribs.ts_attrs["axes"]

        return _op.squeeze(input, axis=axes)

    def layout_unsqueeze(self, input, attribs):
        axes = attribs.ts_attrs["axes"]

        return _op.expand_dims(input, axis=axes)

    def layout_concat(self, inputs, attribs):
        axis = attribs.ts_attrs["axis"]

        return _op.concat(inputs, axis=axis)

    def layout_split(self, input, attribs):
        axis = attribs.ts_attrs["axis"]
        count = attribs.ts_attrs["count"]
        sizes = attribs.ts_attrs["sizes"]

        if count is not None:
            indices_or_sections = count
        else:
            # cumsum of sizes returns the axes, we don't need the last one as SkND asserts that the sizes add up to the size
            indices_or_sections = [sum(sizes[:i + 1]) for i in range(len(sizes))][:-1]

        return _op.split(input, indices_or_sections, axis=axis)

    def layout_stack(self, inputs, attribs):
        axis = attribs.ts_attrs["axis"]
        squeeze = attribs.ts_attrs["squeeze"]

        if squeeze:
            return _op.concat(inputs, axis=axis)

        expanded = [_op.expand_dims(i, axis=axis) for i in inputs]
        return _op.concat(expanded, axis=axis)

    def layout_unstack(self, input, attribs):
        axis = attribs.ts_attrs["axis"]
        squeeze = attribs.ts_attrs["squeeze"]

        sections = input.struct_info.shape[axis]

        if not squeeze:
            return _op.split(input, sections, axis=axis)
        else:
            # squeeze extra dimension in result, not too slow
            result = _op.split(input, sections, axis=axis)
            result = self.bb.normalize(result)
            nres = len(result.struct_info.fields)
            return relax.Tuple([_op.squeeze(result[i], axis=axis) for i in range(nres)])

    def layout_tile(self, input, attribs):
        axes = attribs.ts_attrs["axes"]
        repeats = attribs.ts_attrs["repeats"]

        rank = len(_gen_shape_list(input))

        full_repeats = [1] * rank
        for ax, rep in zip(axes, repeats):
            full_repeats[ax] = rep

        return _op.tile(input, full_repeats)

    def layout_broadcast(self, input, attribs):
        axes = attribs.ts_attrs["axes"]
        shape = attribs.ts_attrs["shape"]

        dshape = _gen_shape_list(input)

        for ax, sh in zip(axes, shape):
            dshape[ax] = sh

        return _op.broadcast_to(input, dshape)

    def layout_slice(self, input, attribs):
        axes = attribs.ts_attrs["axes"]
        begin = attribs.ts_attrs["begin"]
        end = attribs.ts_attrs["end"]
        stride = attribs.ts_attrs["stride"]

        return _op.strided_slice(input, axes, begin, end, stride)

    def layout_pad(self, input, value, attribs):
        axes = attribs.ts_attrs["axes"]
        padding = attribs.ts_attrs["padding"]
        method = attribs.ts_attrs["method"]

        # generate padding value pairs

        rank = len(_gen_shape_list(input))
        pad_pairs = [0] * rank * 2

        for ax in axes:
            pad_pairs[ax * 2] = padding.pop(0)
            pad_pairs[ax * 2 + 1] = padding.pop(0)

        if not value:
            value = 0

        # TVM 0.20+sth broke this fml
        # using topi for all

        # method is not implemented properly in relax, can't use for other than constant pad
        # if method == "CONSTANT":
        #     return _op.nn.pad(input, pad_width=pad_pairs, pad_value=value, pad_mode=method)

        # use TOpI mirror_pad for other methods
        if method == "REPLICATE":
            method = "SYMMETRIC"

        before = pad_pairs[::2]
        after = pad_pairs[1::2]

        # constant, removeable after tvm fixes relax.pad
        if method == "CONSTANT":
            if isinstance(value, relax.Constant):
                val = value.data.numpy().tolist()
            else:
                val = value
            return self.bb.emit_te(topi.nn.pad, input, before, after, val)

        return self.bb.emit_te(topi.nn.mirror_pad, input, before, after, method)

    def layout_transpose(self, input, attribs):
        axis = attribs.ts_attrs["axis"]
        perm = attribs.ts_attrs["perm"]

        rank = len(_gen_shape_list(input))

        # handle negative values for both axis and perms
        if axis < 0:
            axis = rank + axis

        for p in perm:
            if p < 0:
                p = rank + p

        axes = [*range(axis), *perm]
        return _op.permute_dims(input, axes)

    def layout_space_to_batch_alter(self, input, attribs):
        """
        Alternative space to batch implementation, using TOpI, layout change makes it slower than the other implementation
        TODO remove?
        """
        block_size = attribs.ts_attrs["block_size"]
        dshape = _gen_shape_list(input)
        rank = len(dshape)
        pad_before = [0] * (rank - 2)
        pad_after = [0] * (rank - 2)

        # change layout from N C [Spatial] to N [Spatial] C
        # input_nd = _op.permute_dims(input, [0, *range(2, rank), 1])
        # self.bb.normalize(input_nd)


        input_nd = self.bb.emit_te(topi.layout_transform, input, "NCHW", "NHWC")
        transposed = self.bb.emit_te(topi.nn.space_to_batch_nd, input_nd, block_size, pad_before, pad_after)

        # return _op.permute_dims(transposed, [0, rank - 1, *range(1, rank - 1)])

        return self.bb.emit_te(topi.layout_transform, transposed, "NHWC", "NCHW")


    def layout_space_to_batch(self, input, attribs):
        """
        SkND style space to batch implementation, currently fastest of the 3 (relax hl, topi, contraction)
        """
        block_size = attribs.ts_attrs["block_size"]

        dshape = _gen_shape_list(input)
        d = len(dshape) - 2
        rs_shape = [(block_size[i], dshape[i + d] / block_size[i]) for i in range(len(block_size))]
        rs_shape = [int(item) for sublist in rs_shape for item in sublist]

        attribs.ts_attrs.update({"axis": 2, "rank": d, "shape": rs_shape})
        reshaped = self.layout_reshape(input, attribs)
        reshaped = self.bb.normalize(reshaped)
        attribs.ts_attrs.update({"axis": 0, "perm": [*range(2, 2 * d + 2, 2), 0, 1, *range(3, 2 * d + 2, 2)]})
        transposed = self.layout_transpose(reshaped, attribs)
        attribs.ts_attrs.update({"axis": 0, "rank": d + 1})
        transposed = self.bb.normalize(transposed)
        output = self.layout_flatten(transposed, attribs)
        return output


    def layout_nonzero(self, input, attribs):
        return _op.nonzero(input)

    def layout_gather(self, input, indices, attribs):
        axis = attribs.ts_attrs["axis"]
        return _op.take(input, indices, axis)

    def layout_gather_nd(self, input, indices, attribs):
        batch_dims = attribs.ts_attrs["batch_dims"]
        return _op.gather_nd(input, _op.astype(indices, "int64"), batch_dims)

    def layout_scatter(self, input, indices, updates, attribs):
        axis = attribs.ts_attrs["axis"]
        return _op.scatter_elements(input, indices, updates, axis)

    def layout_scatter_nd(self, input, indices, updates, attribs):
        batch_dims = attribs.ts_attrs["batch_dims"]
        if batch_dims > 0:
            raise UnsupportedAttr("scatter_nd", "batch_dims", batch_dims)

        return _op.scatter_nd(input, _op.astype(indices, "int64"), updates)

    def linalg_dot(self, left, right, bias, attribs):
        mul = _op.matmul(left, right)
        if bias is not None:
            return _op.add(mul, bias)
        return mul

    def linalg_matvec(self, left, right, bias, attribs):
        """ Matrix vector multiplication, Only 2D matrix and 1D vector supported"""
        transA = attribs.ts_attrs["transA"]

        if transA:
            left = _op.permute_dims(left, [1, 0])

        mul = _op.matmul(left, right)

        if bias is not None:
            return _op.add(mul, bias)
        return mul

    def linalg_matmul(self, left, right, bias, attribs):
        """ Matrix multiplication on last two dimensions on par with contr implementation,
        the most trivial cases being faster, multiple permutes, and bias making it worse

        Overload if needed"""
        transA = attribs.ts_attrs["transA"]
        transB = attribs.ts_attrs["transB"]

        if transA:
            shapeA = _gen_shape_list(left)
            perm_ind = list(range(len(shapeA)))
            perm_ind[-1], perm_ind[-2] = perm_ind[-2], perm_ind[-1]
            left = _op.permute_dims(left, perm_ind)

        if transB:
            shapeB = _gen_shape_list(right)
            perm_ind = list(range(len(shapeB)))
            perm_ind[-1], perm_ind[-2] = perm_ind[-2], perm_ind[-1]
            right = _op.permute_dims(right, perm_ind)

        mul = _op.matmul(left, right)

        if bias is not None:
            return _op.add(mul, bias)
        return mul

    def math_iden(self, input, attribs):
        return self.bb.emit_te(topi.identity, input)

    def math_neg(self, input, attribs):
        return _op.negative(input)

    def math_rcp(self, input, attribs):
        return self.math_div(relax.const(1.0), input, attribs)

    def math_sqr(self, input, attribs):
        return _op.square(input)

    def math_sqrt(self, input, attribs):
        return _op.sqrt(input)


    def math_rsqrt(self, input, attribs):
        return _op.rsqrt(input)

    def math_exp(self, input, attribs):
        return _op.exp(input)

    def math_log(self, input, attribs):
        return _op.log(input)


    def math_sin(self, input, attribs):
        return _op.sin(input)

    def math_cos(self, input, attribs):
        return _op.cos(input)

    def math_tan(self, input, attribs):
        return _op.tan(input)

    def math_sinh(self, input, attribs):
        return _op.sinh(input)

    def math_cosh(self, input, attribs):
        return _op.cosh(input)

    def math_tanh(self, input, attribs):
        return _op.tanh(input)

    def math_asin(self, input, attribs):
        return _op.asin(input)

    def math_acos(self, input, attribs):
        return _op.acos(input)

    def math_atan(self, input, attribs):
        return _op.atan(input)

    def math_asinh(self, input, attribs):
        return _op.asinh(input)

    def math_acosh(self, input, attribs):
        return _op.acosh(input)

    def math_atanh(self, input, attribs):
        return _op.atanh(input)

    def math_abs(self, input, attribs):
        return _op.abs(input)

    def math_sign(self, input, attribs):
        return _op.sign(input)

    def math_not(self, input, attribs):
        return _op.logical_not(input)

    def math_floor(self, input, attribs):
        return _op.floor(input)

    def math_ceil(self, input, attribs):
        return _op.ceil(input)

    def math_round(self, input, attribs):
        return _op.round(input)

    def math_add(self, left, right, attribs):
        return self.__aligned_bin_op(_op.add, left, right, attribs)

    def math_sub(self, left, right, attribs):
        return self.__aligned_bin_op(_op.subtract, left, right, attribs)

    def math_mul(self, left, right, attribs):
        return self.__aligned_bin_op(_op.multiply, left, right, attribs)

    def math_div(self, left, right, attribs):
        return self.__aligned_bin_op(_op.divide, left, right, attribs)

    def math_mod(self, left, right, attribs):
        return self.__aligned_bin_op(_op.mod, left, right, attribs)

    def math_pow(self, left, right, attribs):
        return self.__aligned_bin_op(_op.power, left, right, attribs)

    def math_min(self, left, right, attribs):
        return self.__aligned_bin_op(_op.minimum, left, right, attribs)

    def math_max(self, left, right, attribs):
        return self.__aligned_bin_op(_op.maximum, left, right, attribs)

    def math_lt(self, left, right, attribs):
        return self.__aligned_bin_op(_op.less, left, right, attribs)

    def math_gt(self, left, right, attribs):
        return self.__aligned_bin_op(_op.greater, left, right, attribs)

    def math_le(self, left, right, attribs):
        return self.__aligned_bin_op(_op.less_equal, left, right, attribs)

    def math_ge(self, left, right, attribs):
        return self.__aligned_bin_op(_op.greater_equal, left, right, attribs)

    def math_eq(self, left, right, attribs):
        return self.__aligned_bin_op(_op.equal, left, right, attribs)

    def math_ne(self, left, right, attribs):
        return self.__aligned_bin_op(_op.not_equal, left, right, attribs)

    def math_and(self, left, right, attribs):
        return self.__aligned_bin_op(_op.logical_and, left, right, attribs)

    def math_or(self, left, right, attribs):
        return self.__aligned_bin_op(_op.logical_or, left, right, attribs)

    def math_xor(self, left, right, attribs):
        return self.__aligned_bin_op(_op.logical_xor, left, right, attribs)

    def math_select(self, cond, left, right, attribs):
        rank = len(attribs.output_shape[0])
        if attribs.ts_attrs["cond_align"] is not None:
            cond = self._align_tensor(cond, attribs.ts_attrs["cond_align"], rank)
        if attribs.ts_attrs["lhs_align"] is not None:
            left = self._align_tensor(left, attribs.ts_attrs["lhs_align"], rank)
        if attribs.ts_attrs["rhs_align"] is not None:
            right = self._align_tensor(right, attribs.ts_attrs["rhs_align"], rank)

        return _op.where(cond, left, right)

    def math_sum_n(self, inputs, attribs):
        return self.bb.emit_te(topi.elemwise_sum, inputs)

    def math_min_reduce(self, input, attribs):
        axis = attribs.ts_attrs["axes"]
        keepdims = not attribs.ts_attrs["squeeze"]

        return _op.min(input, axis=axis, keepdims=keepdims)

    def math_max_reduce(self, input, attribs):
        axis = attribs.ts_attrs["axes"]
        keepdims = not attribs.ts_attrs["squeeze"]

        return _op.max(input, axis=axis, keepdims=keepdims)

    def math_sum_reduce(self, input, attribs):
        axis = attribs.ts_attrs["axes"]
        keepdims = not attribs.ts_attrs["squeeze"]

        return _op.sum(input, axis=axis, keepdims=keepdims)

    def math_mean_reduce(self, input, attribs):
        axis = attribs.ts_attrs["axes"]
        keepdims = not attribs.ts_attrs["squeeze"]

        return _op.mean(input, axis=axis, keepdims=keepdims)


    def math_any_reduce(self, input, attribs):
        axis = attribs.ts_attrs["axes"]
        keepdims = not attribs.ts_attrs["squeeze"]

        return self.bb.emit_te(topi.any, input, axis=axis, keepdims=keepdims)

    def math_all_reduce(self, input, attribs):
        axis = attribs.ts_attrs["axes"]
        keepdims = not attribs.ts_attrs["squeeze"]

        return self.bb.emit_te(topi.all, input, axis=axis, keepdims=keepdims)

    def math_argmin(self, input, attribs):
        axis = attribs.ts_attrs["axis"]
        keepdims = not attribs.ts_attrs["squeeze"]

        return _op.argmin(input, axis=axis, keepdims=keepdims)

    def math_argmax(self, input, attribs):
        axis = attribs.ts_attrs["axis"]
        keepdims = not attribs.ts_attrs["squeeze"]

        return _op.argmax(input, axis=axis, keepdims=keepdims)


    def misc_mesh_grid(self, *inputs, attribs):
        # TODO
        raise NotImplementedError

    def nn_linear(self, input, filter, bias, attribs):
        return _op.linear(input, filter, bias)

    def nn_conv(self, input, filter, bias, attribs):
        stride = attribs.ts_attrs["stride"]
        dilation = attribs.ts_attrs["dilation"]
        padding = attribs.ts_attrs["padding"]
        padding_align = attribs.ts_attrs["padding_align"]
        ceil_mode = attribs.ts_attrs["ceil_mode"]
        groups = attribs.ts_attrs["groups"]
        data_format = attribs.ts_attrs["data_format"]

        ndim = len(_gen_shape_list(input)) - 2

        try:
            op = getattr(_op.nn, f"conv{ndim}d")
        except AttributeError:
            raise DimensionError("convolution", ndim)

        if padding is None:
            padding = self._calculate_pad(_gen_shape_list(input)[2:],
                                          _gen_shape_list(filter)[2:],
                                          stride,
                                          dilation,
                                          padding_align,
                                          ceil_mode)

        conv = op(
            input,
            filter,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            data_layout=_map_data_format(data_format, ndim),
        )

        if bias is not None:
            return self._add_bias(conv, bias, data_format, ndim)

        return conv

    def nn_deconv(self, input, filter, bias, attribs):
        stride = attribs.ts_attrs["stride"]
        dilation = attribs.ts_attrs["dilation"]
        padding = attribs.ts_attrs["padding"]
        padding_align = attribs.ts_attrs["padding_align"]
        output_size = attribs.ts_attrs["output_size"] # todo
        groups = attribs.ts_attrs["groups"]
        data_format = attribs.ts_attrs["data_format"]

        ndim = input.struct_info.ndim - 2

        # Legalize for deconv is only partially supported in TVM currently
        intrin_check = self.use_intrinsic.get("deconv", lambda *_: False)
        if not intrin_check([input, filter, bias], attribs):
            if data_format != "NCX":
                raise UnsupportedAttr("deconv", "data_format", data_format)
            if dilation != [1] * ndim:
                raise UnsupportedAttr("deconv", "dilation", dilation)

        try:
            op = getattr(_op.nn, f"conv{ndim}d_transpose")
        except AttributeError:
            raise DimensionError("deconvolution", ndim)

        if padding is None:
            padding = self._calculate_pad(_gen_shape_list(input)[2:],
                                          _gen_shape_list(filter)[2:],
                                          stride,
                                          dilation,
                                          padding_align,
                                          inverse_op=True)


        deconv = op(
            input,
            filter,
            strides=stride,
            padding=padding,
            output_padding=(0,0),  # todo ???
            dilation=dilation,
            groups=groups,
            data_layout=_map_data_format(data_format, ndim),
        )

        if bias is not None:
            return self._add_bias(deconv, bias, data_format, ndim)

        return deconv

    def nn_depthwise_conv(self, input, filter, bias, attribs):
        stride = attribs.ts_attrs["stride"]
        dilation = attribs.ts_attrs["dilation"]
        padding = attribs.ts_attrs["padding"]
        padding_align = attribs.ts_attrs["padding_align"]
        ceil_mode = attribs.ts_attrs["ceil_mode"]
        data_format = attribs.ts_attrs["data_format"]

        ndim = len(_gen_shape_list(input)) - 2


        try:
            op = getattr(_op.nn, f"conv{ndim}d")
        except AttributeError:
            raise DimensionError("depthwise convolution", ndim)

        if padding is None:
            padding = self._calculate_pad(_gen_shape_list(input)[2:],
                                          _gen_shape_list(filter)[2:],
                                          stride,
                                          dilation,
                                          padding_align,
                                          ceil_mode)

        # convert filter from OMS.. to (O*M)1S..
        groups = _gen_shape_list(filter)[0]
        if isinstance(filter, relax.Constant):
            # if the filter is a constant, reshape it in compile time
            new_data = tvm.nd.array(np.array(filter.data.numpy()).reshape([-1, 1] + _gen_shape_list(filter)[2:]))
            filter = relax.const(new_data, filter.struct_info.dtype)
        else:
            filter = _op.reshape(filter, [-1, 1] + _gen_shape_list(filter)[2:]) # todo reshape in numpy constant to handle constant filters better


        conv = op(
            input,
            filter,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            data_layout=_map_data_format(data_format, ndim),
        )

        if bias is not None:
            return self._add_bias(conv, bias, data_format, ndim)

        return conv

    def nn_depthwise_deconv(self, input, filter, bias, attribs):
        stride = attribs.ts_attrs["stride"]
        dilation = attribs.ts_attrs["dilation"]
        padding = attribs.ts_attrs["padding"]
        padding_align = attribs.ts_attrs["padding_align"]
        output_size = attribs.ts_attrs["output_size"] # todo
        data_format = attribs.ts_attrs["data_format"]

        ndim = input.struct_info.ndim - 2

        # Legalize for deconv is only partially supported in TVM currently
        intrin_check = self.use_intrinsic.get("depthwise_deconv", lambda *_: False)
        if not intrin_check([input, filter, bias], attribs):
            if data_format != "NCX":
                raise UnsupportedAttr("depthwise_deconv", "data_format", data_format)
            if dilation != [1] * ndim:
                raise UnsupportedAttr("depthwise_deconv", "dilation", dilation)

        try:
            op = getattr(_op.nn, f"conv{ndim}d_transpose")
        except AttributeError:
            raise DimensionError("depthwise deconvolution", ndim)

        if padding is None:
            padding = self._calculate_pad(_gen_shape_list(input)[2:],
                                          _gen_shape_list(filter)[2:],
                                          stride,
                                          dilation,
                                          padding_align,
                                          inverse_op=True)

        # os_pred = [s * stride[i] + _gen_shape_list(filter)[i + 2] - 1 for i, s in enumerate(output_size)]

        # convert filter from OMS.. to (O*M)1S..
        groups = _gen_shape_list(filter)[0]
        if isinstance(filter, relax.Constant):
            # if the filter is a constant, reshape it in compile time
            new_data = tvm.nd.array(np.array(filter.data.numpy()).reshape([-1, 1] + _gen_shape_list(filter)[2:]))
            filter = relax.const(new_data, filter.struct_info.dtype)
        else:
            filter = _op.reshape(filter, [-1, 1] + _gen_shape_list(filter)[2:])

        deconv = op(
            input,
            filter,
            strides=stride,
            padding=padding,
            output_padding=(0,0),  # todo ???
            dilation=dilation,
            groups=groups,
            data_layout=_map_data_format(data_format, ndim),
        )

        if bias is not None:
            return self._add_bias(deconv, bias, data_format, ndim)

        return deconv

    def nn_separable_conv(self, input, depthwise_filter, pointwise_filter, bias, attribs):
        intermedate = self.nn_depthwise_conv(input, depthwise_filter, None, attribs)
        intermedate = self.bb.normalize(intermedate)

        conv = self.nn_conv(intermedate, pointwise_filter, bias, attribs)

        return conv

    def nn_max_pool(self, input, attribs):
        axes = attribs.ts_attrs["axes"]     # todo!!
        size = attribs.ts_attrs["size"]
        stride = attribs.ts_attrs["stride"]
        dilation = attribs.ts_attrs["dilation"]
        padding = attribs.ts_attrs["padding"]
        padding_align = attribs.ts_attrs["padding_align"]
        ceil_mode = attribs.ts_attrs["ceil_mode"]
        ignore_border = attribs.ts_attrs["ignore_border"]

        ndim = input.struct_info.ndim - 2

        try:
            op = getattr(_op.nn, f"max_pool{ndim}d")
            # op = getattr(topi.nn, f"pool{ndim}d")
        except AttributeError:
            raise DimensionError("max pooling", ndim)

        if padding is None:
            padding = self._calculate_pad(_gen_shape_list(input)[2:],
                                          size,
                                          stride,
                                          dilation,
                                          padding_align,
                                          ceil_mode)

        return op(  # TODO check attribs
            input,
            pool_size=size,
            strides=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=False,
            count_include_pad=False,
        )

    def nn_avg_pool(self, input, attribs):
        axes = attribs.ts_attrs["axes"]     # todo!!
        size = attribs.ts_attrs["size"]
        stride = attribs.ts_attrs["stride"]
        dilation = attribs.ts_attrs["dilation"]
        padding = attribs.ts_attrs["padding"]
        padding_align = attribs.ts_attrs["padding_align"]
        ignore_border = attribs.ts_attrs["ignore_border"]
        ceil_mode = attribs.ts_attrs["ceil_mode"]

        ndim = input.struct_info.ndim - 2

        try:
            op = getattr(_op.nn, f"avg_pool{ndim}d")
        except AttributeError:
            raise DimensionError("average pooling", ndim)

        if padding is None:
            padding = self._calculate_pad(_gen_shape_list(input)[2:],
                                          size,
                                          stride,
                                          dilation,
                                          padding_align,
                                          ceil_mode)

        return op(
            input,
            pool_size=size,
            strides=stride,
            padding=padding,
            # ceil_mode=ceil_mode,
            count_include_pad=not ignore_border,
        )

    def nn_relu(self, input, attribs):
        alpha = attribs.ts_attrs["alpha"]
        max = attribs.ts_attrs["max"]

        if alpha is None and max is None:
            return _op.nn.relu(input)
        elif alpha is not None and max is None:
            return _op.nn.leakyrelu(input, alpha)
        elif alpha is None and max is not None:
            return _op.clip(input, 0, max)
        else:
            raise UnsupportedAttr("relu", "alpha and max", (alpha, max))

    def nn_prelu(self, input, alpha, attribs):
        axis = attribs.ts_attrs["axis"]

        return self.bb.emit_te(topi.nn.prelu, input, alpha, axis)

    def nn_gelu(self, input, attribs):
        approximate = attribs.ts_attrs["approximate"]
        if approximate is None:
            return _op.nn.gelu(input)
        if approximate == "TANH":
            return _op.nn.gelu_tanh(input)
        if approximate == "SIGMOID":
            raise UnsupportedAttr("gelu", "approximate", approximate)

    def nn_sigmoid(self, input, attribs):
        return _op.sigmoid(input)

    def nn_erf(self, input, attribs):
        return _op.erf(input)

    def nn_batch_norm(self, input, mean, variance, bias, scale, attribs):
        epsilon = attribs.ts_attrs["epsilon"]

        if not bias:
            bias = relax.const(0, input.struct_info.dtype)
        if not scale:
            scale = relax.const(1, input.struct_info.dtype)

        return self.bb.emit_te(topi.nn.batch_norm, input, scale, bias, mean, variance, 1, epsilon,
                               training=False,
                           )


    def nn_local_response_norm(self, input, attribs):
        axes = attribs.ts_attrs["axes"]
        size = attribs.ts_attrs["size"]
        alpha = attribs.ts_attrs["alpha"]
        beta = attribs.ts_attrs["beta"]
        bias = attribs.ts_attrs["bias"]

        if len(axes) != 1:
            raise UnsupportedAttr("lrn", "axes", axes)

        return self.bb.emit_te(topi.nn.lrn, input, size[0], axes[0], alpha, beta, bias)

    def nn_softmax(self, input, attribs):
        axes = attribs.ts_attrs["axes"]

        if len(axes) != 1:
            raise UnsupportedAttr("softmax", "axes", axes)

        return _op.nn.softmax(input, axis=axes)


    def _align_tensor(self, tensor, left_align_val, desired_rank):
        shape = _gen_shape_list(tensor)
        new_shape = [1] * left_align_val + shape + [1] * (desired_rank - len(shape) - left_align_val)
        return _op.memory.view(tensor, shape=new_shape)

    def __aligned_bin_op(self, op, left, right, attribs):
        lalign = attribs.ts_attrs.get("lhs_align", None)
        ralign = attribs.ts_attrs.get("rhs_align", None)
        rank = len(attribs.output_shape[0])

        if lalign is not None:
            left = self._align_tensor(left, lalign, rank)
        if ralign is not None:
            right = self._align_tensor(right, ralign, rank)

        return op(left, right)

    def _image_resize_helper(self, input, attribs, method, downscale=False):
        axes = attribs.ts_attrs["axes"]
        factor = attribs.ts_attrs.get("factor", None)
        size = attribs.ts_attrs.get("size", None)

        symmetric = attribs.ts_attrs.get("symmetric", None)
        replicate_border = attribs.ts_attrs.get("replicate_border", None)

        rounding_method = attribs.ts_attrs.get("rounding_method", None)

        antialias = attribs.ts_attrs.get("antialias", None)

        coordinate_transform = attribs.ts_attrs.get("coordinate_transform", None)
        coeff_a = attribs.ts_attrs.get("coeff_a", -0.5)

        mode = attribs.ts_attrs.get("mode", None)

        if antialias is not None:
            logging.log(logging.DEBUG, "Antialising attrib is unused in SkND currently")


        assert factor or size, "factor or size must be provided"

        if method == "":
            method = {
                "NEAREST": "nearest_neighbor",
                "LINEAR": "linear",
                "CUBIC": "cubic",
            }[mode]

        if replicate_border is False:
            raise UnsupportedAttr("image_sample", "replicate_border", replicate_border)

        dshape = _gen_shape_list(input)
        rank = len(dshape) - 2

        try:
            op = getattr(topi.image, f"resize{rank}d")
        except AttributeError:
            raise DimensionError("image_nearest_downsample", "rank", rank)

        # generate new shape

        if factor:
            if not isinstance(factor, list):
                factor = [factor] * rank

            new_shape = [int(dshape[axes[i]] * factor[i] if not downscale else dshape[axes[i]] // factor[i]) for i in range(rank)]
        else:
            new_shape = size

        roi = [0, 0] * rank

        # NEAREST
        # TODO Viktor
        rounding = {
            "FLOOR": "floor",
            "CEIL": "ceil",
            "ROUND_PREFER_FLOOR": "floor",
            "ROUND_PREFER_CEIL": "round",
        }.get(rounding_method, "floor")


        # LINEAR (AREA)
        if symmetric is None and coordinate_transform is None:
            # default value if not linear
            coord_transform = "half_pixel"
        elif coordinate_transform is None:
            # if sampling, symmetric is defined
            coord_transform = "half_pixel" if symmetric else "asymmetric"
        else:
            # if resize, coordinate_transform is defined
            coord_transform = {
                "SYMMETRIC": "half_pixel",
                "ASYMMETRIC": "asymmetric",
                "ALIGNED": "align_corners",
            }[coordinate_transform]

        # CUBIC
        # method name in TVM docu is wrong, correct is `cubic`

        if rank == 2:
            return _op.image.resize2d(input,
                                      new_shape,
                                      method=method,
                                      coordinate_transformation_mode=coord_transform,
                                      rounding_method=rounding,
                                      cubic_alpha=coeff_a,
                                      )

        return self.bb.emit_te(
            op,
            input,
            roi,
            new_shape,
            method=method,
            coordinate_transformation_mode=coord_transform,
            rounding_method=rounding,
            bicubic_alpha=coeff_a,
        )


    def _calculate_pad(self, dsshape, ksshape, stride, dilation, padding_align="UPPER", ceil_mode=False,
                       inverse_op=False):
        fd = [(k - 1) * d + 1 for k, d in zip(ksshape, dilation)]

        if not inverse_op:
            if ceil_mode:
                # ceil div in first step
                padding = [(-(v // -sr) - 1) * sr + f - v for v, sr, f in zip(dsshape, stride, fd)]
            else:
                # floor div in first step
                padding = [((v // sr) - 1) * sr + f - v for v, sr, f in zip(dsshape, stride, fd)]
        else:
            padding = [(s-1) * sr + f - s * sr for s, sr, f in zip(dsshape, stride, fd)]

        if padding_align == "UPPER":
            before = [p // 2 for p in padding]
            after = [p - b for p, b in zip(padding, before)]
            padding = [*before, *after]
        else:
            before = [-(p // -2) for p in padding]
            after = [p - b for p, b in zip(padding, before)]
            padding = [*before, *after]

        return padding

    def _add_bias(self, op, bias, data_format, ndim):
        # broadcast bias to the correct shape
        if data_format == "NCX":
            if isinstance(bias, relax.Constant):
                # if the bias is a constant, reshape it in compile time to 1B1..(spatial dims)
                new_data = tvm.nd.array(np.array(bias.data.numpy()).reshape([1, -1] + [1] * ndim))
                bias = relax.const(new_data, bias.struct_info.dtype)
            else:
                bias = _op.expand_dims(bias, axis=[0] + list(range(2, ndim + 2)))
        else:
            if isinstance(bias, relax.Constant):
                # if the bias is a constant, reshape it in compile time to 11..(spatial dims)B
                new_data = tvm.nd.array(np.array(bias.data.numpy()).reshape([1] + [1] * ndim + [-1]))
                bias = relax.const(new_data, bias.struct_info.dtype)
            else:
                bias = _op.expand_dims(bias, axis=list(range(0, ndim + 1)))

        return _op.add(op, bias)

    # Magics
    def __getitem__(self, key):
        # todo write log if custom shadows?
        key_norm = key.replace(".", "_")
        if key in self.custom_map:
            if hasattr(self, key_norm):
                logging.log(logging.WARNING, f"Custom conversion {key} shadows built-in {key_norm}")
            return self.custom_map[key]
        try:
            return getattr(self, key_norm)
        except AttributeError:
            raise NotInMapError(key)

    def __contains__(self, item):
        return item in self.custom_map or hasattr(self, item)
