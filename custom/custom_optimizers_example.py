# Define how a sequence of ops is replaced by a new sequence.
# First test if the sequence of ops matched should be really replaced; return False if not.
# If yes, create new Tensors and Operations in the graph with the Tensor() and  Operation() constructors.
# DO NOT perform modifications to the graph before all checks passed!

def replace_shuffle(reshape1, transpose, reshape2):
    if reshape2.output.shape != reshape1.input.shape:
        return False

    if len(reshape1.output.shape) != len(reshape1.input.shape) + 1 or \
            reshape1.output.shape[0] != reshape1.input.shape[0] or \
            reshape1.output.shape[3:] != reshape1.input.shape[2:]:
        return False

    axes = transpose.attribs['axes']
    if axes[:3] != [0, 2, 1] or axes[3:] != list(range(3, len(axes))):
        return False

    groups = reshape1.output.shape[1]

    Operation(reshape1.graph, type='shuffle', attribs={'groups': groups},
              inputs=reshape1.input, outputs=reshape2.output, custom=True)


# List sequences of op types that should be matched and replaced if the replacer function does not return False
# An item in the list may be a set as well, in which case any of its items can count as a match in the sequence
# Use a tuple for the key sequence, because list is not hashable

CUSTOM_OPTIMIZERS = {
    ('reshape', 'transpose', 'reshape'): replace_shuffle,
}
