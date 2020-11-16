import torch


# define how the PyTorch interpreter should execute the op

def shuffle(input, groups):
    shape = list(input.shape)
    reshaped = input.reshape([shape[0], groups, shape[1] / groups] + shape[2:])
    transposed = reshaped.permute(0, 2, 1, *list(range(3, len(shape) + 1)))
    return transposed.reshape(shape)


# mapping from op names to executor functions

CUSTOM_OPERATORS = {
    'shuffle': shuffle,
}
