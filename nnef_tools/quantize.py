from nnef_tools.io.nnef.reader import Reader
from nnef_tools.io.nnef.writer import Writer
import numpy as np
import argparse
import json
import os


_CONV_OPS = ['conv', 'deconv', 'separable_conv', 'separable_deconv']


def make_quantization(min, max, signed, symmetric):
    if min > 0:
        min = 0
    if max < 0:
        max = 0

    if signed and symmetric:
        if -max < min:
            min = -max
        if -min > max:
            max = -min

    scale = 255 / (max - min)
    zero_point = int((0 - min) * scale)

    if signed:
        zero_point -= 127 if symmetric else 128

    return {'op-name': 'zero_point_linear_quantize', 'zero_point': zero_point, 'scale': scale,
            'signed': signed, 'symmetric': symmetric, 'bits': 8}


def quantize_params(params, zero_point, scale, signed, symmetric):
    min = (((-127 if symmetric else -128) if signed else 0) - zero_point) * scale
    max = ((127 if signed else 255) - zero_point) * scale

    params = np.clip(params, min, max)
    return np.floor((params - min) / scale).astype(np.int8 if signed else np.uint8)


def quantize_bias(params, scale):
    return np.floor(params / scale).astype(np.int32)


def is_conv_param(tensor):
    return all(op.type in _CONV_OPS for op in tensor.consumers)


def is_conv_bias(tensor):
    assert len(tensor.consumers) == 1
    return len(tensor.consumers) == 1 and tensor.consumer.type in _CONV_OPS and len(tensor.consumer.inputs) > 2 and \
           tensor is tensor.consumer.inputs[2]


def main(args):
    reader = Reader(infer_shapes=False)
    model = reader(args.model)

    stats_path = args.statistics or os.path.join(args.model, 'graph.stats')
    if not os.path.exists(stats_path):
        print("Could not find statistics file '{}'".format(stats_path))
        return -1

    with open(stats_path, 'r') as file:
        stats = json.load(file)

    for tensor in model.tensors:
        stat = stats.get(tensor.name)
        if stat is not None:

            if args.percentile is not None:
                lo = max(stat['mean'] - args.percentile * stat['std'], stat['min'])
                hi = min(stat['mean'] + args.percentile * stat['std'], stat['max'])
            else:
                lo = stat['min']
                hi = stat['max']

            tensor.quant = make_quantization(lo, hi, args.signed, args.symmetric)

            if tensor.data is not None:
                tensor.data = quantize_params(tensor.data, tensor.quant['zero_point'], tensor.quant['scale'],
                                              args.signed, args.symmetric)

    if args.wide_bias:
        for tensor in model.tensors:
            if len(tensor.quant) > 0 and tensor.data is not None and is_conv_bias(tensor):
                conv = tensor.consumer
                tensor.quant['bits'] = 32
                tensor.quant['zero_point'] = 0
                tensor.quant['scale'] = conv.inputs[0].quant['scale'] * conv.inputs[1].quant['scale']
                tensor.data = quantize_bias(tensor.data, tensor.quant['scale'])

    writer = Writer()
    writer(model, args.output)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='The model to quantize')
    parser.add_argument('--statistics', type=str, default=None,
                        help='The tensor statistics to use for quantization')
    parser.add_argument('--output', type=str, required=True,
                        help='The path of the output model')
    parser.add_argument('--signed', action='store_true',
                        help='Whether to generate signed int8 quantized values instead of uint8')
    parser.add_argument('--symmetric', action='store_true',
                        help='Whether to quantize symmetrically and force zero-point to 0')
    parser.add_argument('--wide-bias', action='store_true',
                        help='Whether to quantize biases into int32 values')
    parser.add_argument('--percentile', type=float, default=None,
                        help='Define ranges with approximate normal distribution percentiles;'
                             'provide number of standard deviations from mean to be used')
    exit(main(parser.parse_args()))
