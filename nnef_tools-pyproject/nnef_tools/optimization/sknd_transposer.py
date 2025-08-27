# Copyright (c) 2017-2025 The Khronos Group Inc.
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
from ..model import *
import six
import math
import copy
import numpy as np
import inspect
from ..model.utils import generate_missing_tensor_names_from_op_type


class Transform:

    def __init__(self, inputs=None, outputs=None, attribs=None, using=None):
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.attribs = attribs or {}
        self.using = using or {}


class Transposer:

    @staticmethod
    def find_public_methods(obj):
        methods = inspect.getmembers(obj, predicate=inspect.ismethod)
        return {name: func for name, func in methods if not name.startswith('_')}

    @staticmethod
    def unpack_transforms(transforms):
        unpacked = {}
        for key, transform in six.iteritems(transforms):
            assert isinstance(transform, Transform)

            if not isinstance(transform.inputs, tuple):
                transform.inputs = (transform.inputs,)

            if not isinstance(transform.outputs, tuple):
                transform.outputs = (transform.outputs,)

            if isinstance(key, tuple):
                for item in key:
                    unpacked[item] = transform
            else:
                unpacked[key] = transform

        return unpacked

    def __init__(self, source_format, target_format):
        self._transforms = _Transforms
        self._callables = self.find_public_methods(self)
        self._source_format = source_format
        self._target_format = target_format

    def __call__(self, model, inputs_to_transpose=None):
        if inputs_to_transpose is None:
            inputs_to_transpose = list(range(len(model.main.inputs)))

        self._transposed = set()

        for idx in inputs_to_transpose:
            input = model.main.inputs[idx]
            if len(input.shape) > 2:
                input.shape = self.transpose_shape(input.shape)
                self._transposed.add(input)

        for graph in model.graphs:
            for op in list(graph.operations):
                transform = self._transforms.get(op.type)
                if transform:
                    usings = {}
                    if transform.using:
                        for name, expr in transform.using.items():
                            usings[name] = self._evaluate(expr, op.attribs, op.inputs, op.outputs, usings)

                    attribs = dict(op.attribs)
                    for name, expr in transform.attribs.items():
                        value = self._evaluate(expr, op.attribs, op.inputs, op.outputs, usings)
                        if value is not None:
                            attribs[name] = value

                    inputs = tuple(self._evaluate(expr, op.attribs, op.inputs, op.outputs, usings)
                                   for expr in transform.inputs)

                    outputs = tuple(self._evaluate(expr, op.attribs, op.inputs, op.outputs, usings)
                                    for expr in transform.outputs)

                    op.inputs = tuple(inputs)
                    op.outputs = tuple(outputs)
                    op.attribs = attribs

                elif len(op.inputs) == 1 and len(op.outputs) == 1:
                    op.outputs = (self.transpose_output_like(op.output, op.input),)
                else:
                    op.inputs = tuple(self.undo_transpose(input) for input in op.inputs)

            graph.sort()

        generate_missing_tensor_names_from_op_type(model)

    def _evaluate(self, arg, attribs, inputs, outputs, usings):
        if isinstance(arg, str) and arg[0] == '!':
            return eval(arg[1:], {'I': inputs, 'O': outputs, **attribs, **usings, **self._callables,
                                  'np': np, 'math': math})
        else:
            return arg

    @staticmethod
    def _permute(items, perm):
        permuted = list(items)
        for i in range(len(perm)):
            permuted[i] = items[perm[i]]
        return type(items)(permuted)

    @staticmethod
    def _inverse_permute(items, perm):
        permuted = list(items)
        for i in range(len(perm)):
            permuted[perm[i]] = items[i]
        return type(items)(permuted)

    def source_format(self):
        return self._source_format

    def target_format(self):
        return self._target_format

    def permutation(self, rank):
        raise NotImplementedError()

    def inverse_permutation(self, rank):
        raise NotImplementedError()

    def transposed(self, tensor):
        return tensor in self._transposed

    def transpose_input(self, tensor):
        if tensor in self._transposed:
            return tensor
        else:
            return self._pre_transpose(tensor, perm=self.permutation(tensor.rank))

    def transpose_output(self, tensor):
        tensor.shape = self.transpose_shape(tensor.shape)
        self._transposed.add(tensor)
        return tensor

    def transpose_output_like(self, tensor, reference):
        return self.transpose_output(tensor) if self.transposed(reference) else tensor

    def _pre_transpose(self, input, perm):
        shape = self.transpose_shape(input.shape)
        output = Tensor(input.graph, dtype=input.dtype, shape=shape, quant=copy.deepcopy(input.quant))
        Operation(input.graph, type='layout.transpose', inputs=input, outputs=output, attribs={'perm': perm})
        return output

    def _post_transpose(self, output, perm):
        shape = self._inverse_permute(output.shape, perm)
        input = Tensor(output.graph, dtype=output.dtype, shape=shape, quant=copy.deepcopy(output.quant))
        Operation(input.graph, type='layout.transpose', inputs=input, outputs=output, attribs={'perm': perm})
        return input

    def transpose_shape(self, shape):
        raise NotImplementedError()

    def transpose_axis(self, axis, rank):
        raise NotImplementedError()

    def transpose_axes(self, axes, rank):
        return [self.transpose_axis(axis, rank) for axis in axes]

    def transpose_axis_like(self, axis, rank, reference):
        return self.transpose_axis(axis, rank) if self.transposed(reference) else axis

    def transpose_axes_like(self, axes, rank, reference):
        return self.transpose_axes(axes, rank) if self.transposed(reference) else axes

    def transpose_padding(self, padding):
        rank = len(padding) / 2
        return self.transpose_shape(padding[:rank]) + self.transpose_shape(padding[rank:])

    def align_to_offset(self, align, input_rank, output_rank):
        if align is None:
            return output_rank - input_rank
        elif align < 0:
            return output_rank - input_rank + align
        else:
            return align

    def needs_transpose(self, tensor, other):
        return not self.transposed(tensor) and self.transposed(other)

    def undo_transpose(self, tensor):
        if isinstance(tensor, TensorPack):
            if all(not self.transposed(item) for item in tensor):
                return tensor

            return [self.undo_transpose(item) for item in tensor]
        else:
            perm = self.inverse_permutation(tensor.rank)
            if perm == list(range(tensor.rank)):
                return tensor
            return self._pre_transpose(tensor, perm) if self.transposed(tensor) else tensor


class NXCtoNCX(Transposer):

    def __init__(self):
        super().__init__("NXC", "NCX")

    def permutation(self, rank):
        return [0, rank - 1] + list(range(1, rank - 1))

    def inverse_permutation(self, rank):
        return [0] + list(range(2, rank)) + [1]

    def transpose_shape(self, shape):
        return [shape[0], shape[-1], *shape[1:-1]]

    def transpose_axis(self, axis, rank):
        return 0 if axis == 0 else 1 if axis == -1 or axis == rank - 1 else axis + 1


_Transforms = Transposer.unpack_transforms({
    ('nn.conv', 'nn.deconv'):
        Transform(
            using={
                'needs_transpose': '!data_format == source_format()',
            },
            inputs=(
                '!transpose_input(I[0]) if needs_transpose else I[0]',
                '!I[1]',
                '!I[2]',
            ),
            outputs='!transpose_output(O[0]) if needs_transpose else O[0]',
            attribs={
                'data_format': '!target_format() if needs_transpose else None',
            },
        ),
    ('nn.max_pool', 'nn.sum_pool', 'nn.avg_pool', 'nn.rms_pool', 'nn.lp_pool'):
        Transform(
            using={
                'all_axes': '!axes == list(range(I[0].rank))',
            },
            inputs='!transpose_input(I[0])',
            outputs='!transpose_output(O[0])',
            attribs={
                'axes': '!axes if all_axes else transpose_axes(axes, I[0].rank)',
                'size': '!transpose_shape(size) if all_axes else size',
                'stride': '!transpose_shape(stride) if all_axes else stride',
                'dilation': '!transpose_shape(dilation) if all_axes else dilation',
                'padding': '!transpose_padding(padding) if all_axes else padding',
            },
        ),
    ('math.add', 'math.sub', 'math.mul', 'math.div', 'math.mod', 'math.pow', 'math.min', 'math.max',
     'math.lt', 'math.gt', 'math.le', 'math.ge', 'math.eq', 'math.ne', 'math.and', 'math.or', 'math.xor'):
        Transform(
            using={
                'same_rank': '!I[0].rank == I[1].rank',
                'lhs_offs': '!align_to_offset(lhs_align, I[0].rank, O[0].rank)',
                'rhs_offs': '!align_to_offset(rhs_align, I[1].rank, O[0].rank)',
            },
            inputs=(
                '!transpose_input(I[0]) if needs_transpose(I[0], I[1]) and same_rank else I[0]',
                '!transpose_input(I[1]) if needs_transpose(I[1], I[0]) and same_rank else I[1]',
            ),
            outputs='!transpose_output(O[0]) if transposed(I[0]) or transposed(I[1]) else O[0]',
            attribs={
                'lhs_align': '!transpose_axis(lhs_offs, O[0].rank)'
                             ' if needs_transpose(I[0], I[1]) and not same_rank else None',
                'rhs_align': '!transpose_axis(rhs_offs, O[0].rank)'
                             ' if needs_transpose(I[1], I[0]) and not same_rank else None',
            },
        ),
    'math.select':
        Transform(
            using={
                'same_rank': '!I[1].rank == I[2].rank',
                'cond_offs': '!align_to_offset(cond_align, I[0].rank, O[0].rank)',
                'lhs_offs': '!align_to_offset(lhs_align, I[1].rank, O[0].rank)',
                'rhs_offs': '!align_to_offset(rhs_align, I[2].rank, O[0].rank)',
            },
            inputs=(
                '!I[0]',
                '!transpose_input(I[1]) if needs_transpose(I[1], I[2]) and same_rank else I[1]',
                '!transpose_input(I[2]) if needs_transpose(I[2], I[1]) and same_rank else I[2]',
            ),
            outputs='!transpose_output(O[0]) if transposed(I[1]) or transposed(I[2]) else O[0]',
            attribs={
                'lhs_align': '!transpose_axis(lhs_offs, O[0].rank)'
                             ' if needs_transpose(I[1], I[2]) and not same_rank else None',
                'rhs_align': '!transpose_axis(rhs_offs, O[0].rank)'
                             ' if needs_transpose(I[2], I[1]) and not same_rank else None',
            },
        ),
    ('math.min_reduce', 'math.max_reduce', 'math.sum_reduce', 'math.prod_reduce',
     'math.mean_reduce', 'math.lp_reduce', 'math.any_reduce', 'math.all_reduce'):
        Transform(
            inputs='!I[0]',
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'axes': '!transpose_axes_like(axes, I[0].rank, I[0])',
            },
        ),
    ('math.sum_n', 'math.prod_n', 'math.min_n', 'math.max_n', 'math.argmin_n', 'math.argmax_n', 'math.any_n', 'math.all_n'):
        Transform(
            using={
                'all_transposed': '!all(transposed(t) for t in I[0])',
            },
            inputs='!I[0] if all_transposed else [undo_transpose(t) for t in I[0]]',
            outputs='!transpose_output(O[0]) if all_transposed else O[0]',
        ),
    ('layout.reshape', 'layout.flatten', 'layout.unflatten', 'layout.squeeze', 'layout.unsqueeze'):
        Transform(
            inputs='!undo_transpose(I[0])',
            outputs='!O[0]',
        ),
    'layout.transpose':
        Transform(
            using={
                'perm_ext': '![*range(axis), *perm, *range(axis + len(perm), I[0].rank)]',
            },
            inputs='!I[0]',
            outputs='!O[0]',
            attribs={
                'axis': 0,
                'perm': '!transpose_axes_like(perm_ext, I[0].rank, I[0])',
            },
        ),
    'layout.concat':
        Transform(
            using={
                'ref': '!I[0][1]',
            },
            inputs='!I[0]',
            outputs='!transpose_output_like(O[0], ref)',
            attribs={
                'axis': '!transpose_axis_like(axis, O[0].rank, ref)',
            },
        ),
    'layout.split':
        Transform(
            inputs='!I[0]',
            outputs='![transpose_output_like(t, I[0]) for t in O[0]]',
            attribs={
                'axis': '!transpose_axis_like(axis, I[0].rank, I[0])',
            },
        ),
    'layout.stack':
        Transform(
            using={
                'ref': '!I[0][0]',
            },
            inputs='!undo_transpose(I[0]) if not squeeze else I[0]',
            outputs='!O[0] if not squeeze else transpose_output_like(O[0], ref)',
            attribs={
                'axis': '!transpose_axis_like(axis, O[0].rank, ref) if squeeze else None',
            },
        ),
    'layout.unstack':
        Transform(
            inputs='!undo_transpose(I[0]) if squeeze else I[0]',
            outputs='!O[0] if squeeze else [transpose_output_like(t, I[0]) for t in O[0]]',
            attribs={
                'axis': '!transpose_axis_like(axis, I[0].rank, I[0]) if not squeeze else None',
            },
        ),
    'layout.tile':
        Transform(
            using={
                'all_axes': '!axes == list(range(I[0].rank))',
            },
            inputs='!I[0]',
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'axes': '!axes if all_axes else transpose_axes_like(axes, I[0].rank, I[0])',
                'repeats': '!transpose_shape(repeats) if all_axes else repeats',
            },
        ),
    'layout.broadcast':
        Transform(
            using={
                'all_axes': '!axes == list(range(I[0].rank))',
            },
            inputs='!I[0]',
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'axes': '!axes if all_axes else transpose_axes_like(axes, I[0].rank, I[0])',
                'shape': '!transpose_shape(shape) if all_axes else shape',
            },
        ),
    'layout.slice':
        Transform(
            using={
                'all_axes': '!axes == list(range(I[0].rank))',
            },
            inputs='!I[0]',
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'axes': '!axes if all_axes else transpose_axes_like(axes, I[0].rank, I[0])',
                'begin': '!transpose_shape(begin) if all_axes else begin',
                'end': '!transpose_shape(end) if all_axes else end',
                'stride': '!transpose_shape(stride) if all_axes else stride',
            },
        ),
    'layout.pad':
        Transform(
            using={
                'all_axes': '!axes == list(range(I[0].rank))',
            },
            inputs='!I[0]',
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'axes': '!axes if all_axes else transpose_axes_like(axes, I[0].rank, I[0])',
                'padding': '!transpose_padding(padding) if all_axes else padding',
            },
        ),
    'layout.shuffle':
        Transform(
            inputs='!I[0]',
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'axis': '!transpose_axis_like(axis, I[0].rank, I[0])',
            },
        ),
    'layout.gather':
        Transform(
            using={
                'keeps_rank': '!O[0].rank == I[0].rank',
            },
            inputs=(
                '!I[0] if keeps_rank else undo_transpose(I[0])',
                '!I[1]',
            ),
            outputs='!transpose_output_like(O[0], I[0]) if keeps_rank else O[0]',
            attribs={
                'axis': '!transpose_axis_like(axis, I[0].rank, I[0]) if keeps_rank else None',
            },
        ),
    'layout.scatter':
        Transform(
            using={
                'keeps_rank': '!O[0].rank == I[0].rank',
            },
            inputs=(
                '!I[0] if keeps_rank else undo_transpose(I[0])',
                '!I[1]',
                '!I[2]',
            ),
            outputs='!transpose_output_like(O[0], I[0]) if keeps_rank else O[0]',
            attribs={
                'axis': '!transpose_axis_like(axis, I[0].rank, I[0]) if keeps_rank else None',
            },
        ),
    ('layout.space_to_batch', 'layout.batch_to_space', 'layout.space_to_depth', 'layout.depth_to_space'):
        Transform(
            inputs='!I[0]',
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'data_format': '!"NCX" if transposed(I[0]) else None',
            },
        ),
    'nn.batch_norm':
        Transform(
            inputs=(
                '!I[0]',
                '!I[1]',
                '!I[2]',
                '!I[3]',
                '!I[4]',
            ),
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'channel_axis': '!transpose_axis_like(channel_axis, I[0].rank, I[0])',
            },
        ),
    'nn.mean_variance_norm':
        Transform(
            using={
                'all_axes': '!axes == list(range(I[0].rank))',
            },
            inputs=(
                '!I[0]',
                '!I[1]',
                '!I[2]',
            ),
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'axes': '!axes if all_axes else transpose_axes_like(axes, I[0].rank, I[0])',
            },
        ),
    'nn.local_response_norm':
        Transform(
            using={
                'all_axes': '!axes == list(range(I[0].rank))',
            },
            inputs='!I[0]',
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'axes': '!axes if all_axes else transpose_axes_like(axes, I[0].rank, I[0])',
                'size': '!transpose_shape(size) if all_axes else size',
            },
        ),
    ('nn.l1_norm', 'nn.l2_norm'):
        Transform(
            using={
                'all_axes': '!axes == list(range(I[0].rank))',
            },
            inputs='!I[0]',
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'axes': '!axes if all_axes else transpose_axes_like(axes, I[0].rank, I[0])',
            },
        ),
    'nn.softmax':
        Transform(
            using={
                'all_axes': '!axes == list(range(I[0].rank))',
            },
            inputs='!I[0]',
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'axes': '!axes if all_axes else transpose_axes_like(axes, I[0].rank, I[0])',
            },
        ),
    ('image.nearest_downsample', 'image.nearest_upsample', 'image.area_downsample', 'image.linear_upsample', 'image.rescale'):
        Transform(
            using={
                'all_axes': '!axes == list(range(I[0].rank))',
            },
            inputs='!I[0]',
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'axes': '!axes if all_axes else transpose_axes_like(axes, I[0].rank, I[0])',
                'factor': '!transpose_shape(factor) if all_axes else factor',
            },
        ),
    ('image.nearest_resize', 'image.linear_resize', 'image.cubic_resize', 'image.resize'):
        Transform(
            using={
                'all_axes': '!axes == list(range(I[0].rank))',
            },
            inputs='!I[0]',
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'axes': '!axes if all_axes else transpose_axes_like(axes, I[0].rank, I[0])',
                'size': '!transpose_shape(size) if all_axes else size',
            },
        ),
    ('quant.zero_point_linear_quantize', 'quant.min_max_linear_quantize'):
        Transform(
            inputs='!I[0]',
            outputs='!transpose_output_like(O[0], I[0])',
            attribs={
                'channel_axis': '!transpose_axis_like(channel_axis, I[0].rank, I[0])',
            },
        ),
})
