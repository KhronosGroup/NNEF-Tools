from src.nnef_tools import Transform


# define mapping from custom op names to converter transforms that maps them in this case to NNEF ops

CUSTOM_TRANSFORMS = {
    'sum_pool2d':
        Transform(
            type='box',
            inputs=(
                '!transpose_input(I[0], data_format)',
            ),
            outputs=(
                '!transpose_output(O[0], data_format)',
            ),
            attribs={
                'size': '!nxc_to_ncx(ensure_list(ksize), cond=is_nxc(data_format))',
                'stride': '!nxc_to_ncx(ensure_list(strides), cond=is_nxc(data_format))',
                'padding': '!convert_padding(padding, I[0].rank)',
                'normalize': False,
            }
        ),
    'lp_norm':
        Transform(
            type='!"l1_normalization" if p == 1 else "l2_normalization"',
            cond='!p == 1 or p == 2',
            inputs='!I[0]',
            outputs='!transpose_like(O[0], ref=I[0])',
            attribs={
                'axes': '!ensure_list(transpose_axis_like(axis, ref=I[0]))',
            }
        ),
}
