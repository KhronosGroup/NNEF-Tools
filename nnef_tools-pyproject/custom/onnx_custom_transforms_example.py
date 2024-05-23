from src.nnef_tools import Transform


def affine_grid_shape(theta, shape):
    return shape


CUSTOM_SHAPES = {
    'grid_sample': lambda input, grid: input,
    'affine_grid': affine_grid_shape,
}


CUSTOM_TRANSFORMS = {
    'affine_grid':
        Transform(
            type='affine_grid',
            using={
                'size': '!as_const(I[1])',
                'align': '!as_const(I[2])',
            },
            cond={
                '!align == 0': 'align_corners must be 0 (false)',
            },
            inputs=(
                '!I[0]',
            ),
            outputs=(
                '!O[0]',
            ),
            attribs={
                'shape': '!size',
            }
        ),
    'grid_sample':
        Transform(
            type='grid_sample',
            using={
                'mode': '!as_const(I[2])',
                'padding': '!as_const(I[3])',
                'align': '!as_const(I[4])',
            },
            cond={
                '!mode == 0': 'mode must be 0 (bilinear)',
                '!padding == 0': 'padding_mode must be 0 (zeros)',
                '!align == 0': 'align_corners must be 0 (false)',
            },
            inputs=(
                '!I[0]',
                '!I[1]',
            ),
            outputs=(
                '!O[0]',
            ),
        ),
}
