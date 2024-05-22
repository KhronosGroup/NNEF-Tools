# Copyright (c) 2020 The Khronos Group Inc.
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

import argparse
from functools import reduce
from .io.nnef import Reader


def _volume(shape):
    return reduce(lambda x, y: x * y, shape, 1)


def _count_macs(op, include_pooling, include_upsampling, include_normalization, include_reduction):
    if len(op.inputs) == 0 or len(op.outputs) == 0:
        return 0
    
    input_volume = _volume(op.inputs[0].shape)
    output_volume = _volume(op.outputs[0].shape)

    if op.type in ['conv', 'deconv']:
        volume = input_volume if op.type == 'deconv' else output_volume
        filter_shape = op.inputs[1].shape
        return volume * _volume(filter_shape[1:])
    elif op.type in ['separable_conv', 'separable_deconv']:
        volume = input_volume if op.type == 'separable_deconv' else output_volume
        filter_shape = op.inputs[1].shape
        inter_channels = filter_shape[0]
        inter_volume = output_volume / op.outputs[0].shape[1] * inter_channels if op.type == 'separable_deconv' else \
                        input_volume / op.inputs[0].shape[1] * inter_channels
        return inter_volume * _volume(filter_shape[2:]) + volume * inter_channels
    elif op.type == 'linear':
        filter_shape = op.inputs[1].shape
        return output_volume * filter_shape[-1]
    elif op.type == 'matmul':
        filter_shape = op.inputs[1].shape
        return output_volume * (filter_shape[-1] if op.attribs['transposeB'] else filter_shape[-2])
    elif op.type in ['max_pool', 'avg_pool', 'rms_pool', 'max_pool_with_index', 'box', 'debox'] and include_pooling:
        volume = input_volume if op.type == 'debox' else output_volume
        kernel_size = op.attribs['size']
        return volume * _volume(kernel_size)
    elif op.type == 'multilinear_upsample' and include_upsampling:
        factor = op.attribs['factor']
        method = op.attribs['method']
        if method == 'symmetric':
            kernel_size = [2 * f for f in factor]
        elif method == 'asymmetric':
            kernel_size = [2 * f - 1 for f in factor]
        else:
            kernel_size = factor
        return input_volume * _volume(kernel_size)
    elif op.type in ['local_response_normalization', 'local_mean_normalization',
                     'local_variance_normalization', 'local_contrast_normalization'] and include_normalization:
        kernel_size = op.attribs['size']
        return output_volume * _volume(kernel_size)
    elif op.type in ['l1_normalization', 'l2_normalization', 'batch_normalization'] and include_normalization:
        return output_volume
    elif op.type in ['sum_reduce', 'max_reduce', 'min_reduce',
                     'mean_reduce', 'all_reduce', 'any_reduce'] and include_reduction:
        return input_volume
    else:
        return 0


def get_custom_shapes(module_names):
    import importlib

    CUSTOM_SHAPES = "CUSTOM_SHAPES"

    shapes = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, CUSTOM_SHAPES):
            shapes.update(getattr(module, CUSTOM_SHAPES))

    return shapes


def main(args):
    custom_shapes = get_custom_shapes(args.custom_shapes) if args.custom_shapes is not None else None
    reader = Reader(infer_shapes=True, custom_shapes=custom_shapes)
    graph = reader(args.model)

    macs = 0
    for op in graph.operations:
        macs += _count_macs(op, args.include_pooling, args.include_upsampling,
                            args.include_normalization, args.include_reduction)

    volume = 0
    for tensor in graph.tensors:
        volume += _volume(tensor.shape)

    gmacs = macs / 1000 / 1000 / 1000
    mbytes = volume * 4 / 1000 / 1000
    print('GMACs = {}'.format(gmacs))
    print('Total memory in Mbytes (supposing float32) = {}'.format(mbytes))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='The model to visualize')
    parser.add_argument('--include-pooling', action='store_true',
                        help='Whether to include pooling operations in the calculation')
    parser.add_argument('--include-upsampling', action='store_true',
                        help='Whether to include (linear) upsampling operations in the calculation')
    parser.add_argument('--include-normalization', action='store_true',
                        help='Whether to include normalization operations in the calculation')
    parser.add_argument('--include-reduction', action='store_true',
                        help='Whether to include reduction operations in the calculation')
    parser.add_argument('--custom-shapes', type=str, nargs='+',
                        help='Module(s) containing custom shape inference code (when converting to NNEF)')
    exit(main(parser.parse_args()))
