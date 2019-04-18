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

from nnef_tools.io.nnef.nnef_graph import *
from nnef_tools.shape_inference import shape_inference as infer


def unify_conv(op):
    # type: (NNEFOperation)->None
    input, filter, bias = op.inputs

    if not op.attribs['stride']:
        op.attribs['stride'] = [1] * (input.rank - 2)
    if not op.attribs['dilation']:
        op.attribs['dilation'] = [1] * (input.rank - 2)
    if not op.attribs['padding']:
        op.attribs['padding'] = infer.same_padding(upscaled_input=input.shape[2:],
                                                   filter=filter.shape[2:],
                                                   stride=op.attribs["stride"],
                                                   dilation=op.attribs["dilation"])
    if not op.attribs['groups']:
        op.attribs['groups'] = input.shape[1]


def unify_deconv(op):
    # type: (NNEFOperation)->None
    input, filter, bias = op.inputs

    if not op.attribs['stride']:
        op.attribs['stride'] = [1] * (input.rank - 2)
    if not op.attribs['dilation']:
        op.attribs['dilation'] = [1] * (input.rank - 2)

    if not op.attribs['groups']:
        if op.attribs['output_shape']:
            op.attribs['groups'] = op.attribs['output_shape'][1]
        else:
            print("Warning: Planewise deconvolution without output_size, "
                  "assuming that num(input channels) == num(output channels).")
            op.attribs['groups'] = filter.shape[0]

    output_channels = filter.shape[1] * op.attribs['groups']
    if op.attribs['output_shape']:
        assert op.attribs['output_shape'][1] == output_channels

    if not op.attribs['padding']:
        calculated_output_size = [i * s for i, s in zip(input.shape[2:], op.attribs['stride'])]
        op.attribs['padding'] = infer.same_padding(upscaled_input=calculated_output_size,
                                                   filter=filter.shape[2:],
                                                   stride=op.attribs['stride'],
                                                   dilation=op.attribs['dilation'])
    else:
        calculated_output_size = infer.conv(input=list(input.shape),
                                            filter=filter.shape[2:],
                                            padding=op.attribs['padding'],
                                            stride=op.attribs['stride'],
                                            dilation=op.attribs['dilation'],
                                            groups=op.attribs['groups'],
                                            output_channels=output_channels,
                                            format=infer.Format.NCHW,
                                            deconv=True)[2:]

    if not op.attribs['output_shape']:
        op.attribs['output_shape'] = [input.shape[0], output_channels] + calculated_output_size


def unify_box_and_pool(op):
    # type: (NNEFOperation)->None
    input = op.inputs[0]
    if not op.attribs['stride']:
        op.attribs['stride'] = [1] * input.rank
    if not op.attribs['dilation']:
        op.attribs['dilation'] = [1] * input.rank
    if not op.attribs['padding']:
        op.attribs['padding'] = infer.same_padding(upscaled_input=input.shape,
                                                   filter=op.attribs['size'],
                                                   stride=op.attribs['stride'],
                                                   dilation=op.attribs['dilation'])


def unify_debox(op):
    # type: (NNEFOperation)->None
    input = op.inputs[0]
    if not op.attribs['stride']:
        op.attribs['stride'] = [1] * input.rank
    if not op.attribs['dilation']:
        op.attribs['dilation'] = [1] * input.rank

    if not op.attribs['padding']:
        calculated_output_shape = [i * s for i, s in zip(input.shape, op.attribs['stride'])]
        op.attribs['padding'] = infer.same_padding(upscaled_input=calculated_output_shape,
                                                   filter=op.attribs['size'],
                                                   stride=op.attribs['stride'],
                                                   dilation=op.attribs['dilation'])
    else:
        calculated_output_shape = infer.sliding_window(input=input.shape,
                                                       filter=op.attribs['size'],
                                                       padding=op.attribs['padding'],
                                                       stride=op.attribs['stride'],
                                                       dilation=op.attribs['dilation'],
                                                       upscale=True)
    if not op.attribs['output_shape']:
        op.attribs['output_shape'] = calculated_output_shape


def unify(nnef_graph):
    # type: (NNEFGraph)->None
    """
    Warning: probably more unifiers will be added
    """
    unifier = {
        'conv': unify_conv,
        'deconv': unify_deconv,
        'avg_pool': unify_box_and_pool,
        'max_pool': unify_box_and_pool,
        'rms_pool': unify_box_and_pool,
        'box': unify_box_and_pool,
        'debox': unify_debox,
    }

    for op in list(nnef_graph.operations):
        if op.name in unifier:
            unifier[op.name](op)


__all__ = ['unify']
