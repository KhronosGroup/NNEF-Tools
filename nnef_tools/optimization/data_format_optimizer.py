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

import copy
from collections import deque, namedtuple

import numpy as np
import six
import typing

from nnef_tools.conversion.conversion_info import ConversionInfo, TensorInfo
from nnef_tools.conversion.transforms \
    import Transform, Transpose, Squeeze, Unsqueeze, Reshape, unsqueezed_shape, squeezed_shape
from nnef_tools.core import graph_utils, matcher, utils
from nnef_tools.core.base_graph import BaseGraph, BaseTensor, BaseOperation


class DataFormatOptimizationDriver(object):
    @property
    def graph_type(self):
        raise NotImplementedError()

    @property
    def tensor_type(self):
        raise NotImplementedError()

    @property
    def op_type(self):
        raise NotImplementedError()

    @property
    def conv_grad_filter_op_names(self):
        raise NotImplementedError()

    @property
    def squeeze_op_name(self):
        raise NotImplementedError()

    @property
    def unsqueeze_op_name(self):
        raise NotImplementedError()

    @property
    def transpose_op_name(self):
        raise NotImplementedError()

    @property
    def reshape_op_name(self):
        raise NotImplementedError()

    @property
    def copy_op_name(self):
        raise NotImplementedError()

    def get_axes_from_squeeze(self, transpose):
        raise NotImplementedError()

    def set_axes_on_squeeze(self, transpose, axes):
        raise NotImplementedError()

    def get_axes_from_unsqueeze(self, transpose):
        raise NotImplementedError()

    def get_axes_from_transpose(self, transpose):
        raise NotImplementedError()

    def set_axes_on_transpose(self, transpose, axes):
        raise NotImplementedError()

    def get_shape_from_reshape(self, reshape):
        raise NotImplementedError()

    def create_tensor(self, graph, name, shape, dtype):
        raise NotImplementedError()

    def create_transpose_op(self, graph, input, output, axes):
        raise NotImplementedError()

    def create_copy_op(self, graph, input, output):
        raise NotImplementedError()

    def generate_missing_names(self, graph):
        raise NotImplementedError()

    def get_input_of_transform(self, transform):
        raise NotImplementedError()


class Transposer(object):
    @staticmethod
    def apply_permutation(list, perm):
        return utils.apply_permutation(list, perm)

    @staticmethod
    def apply_permutation_to_axis(axis, perm):
        return perm.index(axis)

    @staticmethod
    def apply_permutation_to_axes(axes, perm):
        return [Transposer.apply_permutation_to_axis(a, perm) for a in axes]

    @staticmethod
    def zip_inverse(output_count, iterable):
        return utils.zip_inverse(output_count, iterable)

    @staticmethod
    def inverse_permutation(perm):
        return utils.inverse_permutation(perm)


_transposer = Transposer()


class TransposableOperation(object):
    def __init__(self, name, dg_transpose):
        # type: (str, typing.Callable[[Transposer, BaseGraph, BaseOperation, typing.List[int]], None])->None
        self.name = name
        self.dg_transpose = dg_transpose


class _CustomTransform(Transform):
    pass


class _Identity(_CustomTransform):
    def __repr__(self):
        return "IDENTITY"


class _SmartNHWCToNCHW(_CustomTransform):
    def __repr__(self):
        return "SMART_NHWC_TO_NCHW"


class _SmartNCHWToNHWC(_CustomTransform):
    def __repr__(self):
        return "SMART_NCHW_TO_NHWC"


class _TFFilterGradToNNEF(_CustomTransform):
    pass


class _NNEFFilterGradToTF(_CustomTransform):
    pass


class _SmartTFNHWCToNCHW(_CustomTransform):
    pass


class _SmartTFNCHWToNCHW(_CustomTransform):
    pass


class _SmartNCHWToTFNHWC(_CustomTransform):
    pass


class _SmartNCHWToTFNCHW(_CustomTransform):
    pass


class IOTransform(object):
    Transpose = Transpose
    NDHWC_TO_NCDHW = Transpose([0, 4, 1, 2, 3])
    NCDHW_TO_NDHWC = Transpose([0, 2, 3, 4, 1])
    NHWC_TO_NCHW = Transpose([0, 3, 1, 2])
    NCHW_TO_NHWC = Transpose([0, 2, 3, 1])
    NWC_TO_NCW = Transpose([0, 3, 1])
    NCW_TO_NWC = Transpose([0, 2, 1])
    IDENTITY = _Identity()
    TF_FILTER_GRAD_TO_NNEF = _TFFilterGradToNNEF()
    NNEF_FILTER_GRAD_TO_TF = _NNEFFilterGradToTF()
    SMART_NCHW_TO_NHWC = _SmartNCHWToNHWC()
    SMART_NHWC_TO_NCHW = _SmartNHWCToNCHW()
    SMART_TF_NHWC_TO_NCHW = _SmartTFNHWCToNCHW()
    SMART_TF_NCHW_TO_NCHW = _SmartTFNCHWToNCHW()
    SMART_NCHW_TO_TF_NHWC = _SmartNCHWToTFNHWC()
    SMART_NCHW_TO_TF_NCHW = _SmartNCHWToTFNCHW()


TransformOrTransformDictType = typing.Union[Transform, typing.Dict[typing.Union[BaseTensor, str], Transform]]


# We need this extended passthrough remover for the onnx graph, because there can be a reshape with multiple inputs
def remove_passthrough_ex(g, op):
    # type: (BaseGraph, BaseOperation)->None
    op_input = op.inputs[0]
    op_output = op.outputs[0]

    g.remove_operation(op, unlink=True)
    graph_utils.replace_tensor_in_consumers(g, op_output, op_input, remove=True)


def _reshaped_shape(input_shape, reshape_shape):
    for i in range(len(reshape_shape)):
        assert reshape_shape[i] != 0 or i <= len(input_shape), "Invalid input_shape and reshape_shape combination"
    reshape_shape = [input_shape[i] if reshape_shape[i] == 0 else reshape_shape[i] for i in range(len(reshape_shape))]
    if -1 in reshape_shape:
        idx = reshape_shape.index(-1)
        reshape_shape2 = list(reshape_shape)
        reshape_shape2[idx] = 1
        rem = int(np.prod(input_shape)) % int(np.prod(reshape_shape2))
        assert rem == 0, "Invalid input_shape and reshape_shape combination"
        div = int(int(np.prod(input_shape)) / int(np.prod(reshape_shape2)))
        reshape_shape2[idx] = div
        return reshape_shape2
    return reshape_shape


class _TransformException(Exception):
    pass


def add_transform(transforms_by_name, tensor, transform):
    assert isinstance(tensor, BaseTensor)

    if tensor.name in transforms_by_name:
        transforms_by_name[tensor.name].append(transform)
    else:
        transforms_by_name[tensor.name] = [transform]


def _transform_tf_filter_grad_to_nnef(g, tensor, transforms_by_name, driver):
    # type: (BaseGraph, BaseTensor, typing.Dict[str, typing.List[Transform]], DataFormatOptimizationDriver)->None

    assert driver.conv_grad_filter_op_names

    cgf1_output = matcher.Tensor()
    cgf1 = matcher.Operation(name=driver.conv_grad_filter_op_names, outputs=cgf1_output)
    transpose1 = matcher.Operation(name=driver.transpose_op_name, inputs=cgf1_output)

    cgf2_output = matcher.Tensor()
    cgf2 = matcher.Operation(name=driver.conv_grad_filter_op_names, outputs=cgf2_output)
    transpose2_output = matcher.Tensor()
    transpose2 = matcher.Operation(name=driver.transpose_op_name, inputs=cgf2_output, outputs=transpose2_output)
    reshape2 = matcher.Operation(name=driver.reshape_op_name, inputs=transpose2_output)

    if tensor.producer is None:
        raise _TransformException("Cannot apply TF_FILTER_GRAD_TO_NNEF")

    m = matcher.match(g, tensor.producer, matcher.OrPattern(transpose1, reshape2))

    if transpose1 in m:
        cgf = m[cgf1]  # type: BaseOperation
        transpose = m[transpose1]  # type: BaseOperation
        if not (len(transpose.output.consumers) <= 1 and cgf.output not in g.outputs):
            raise _TransformException("Cannot apply TF_FILTER_GRAD_TO_NNEF")
        cgf.output.name = transpose.output.name
        add_transform(transforms_by_name,
                      cgf.output,
                      Transpose(utils.inverse_permutation(driver.get_axes_from_transpose(transpose))))
        graph_utils.replace_tensor_in_outputs(g, transpose.output, cgf.output)
        graph_utils.remove_subgraph(g, [transpose])
    elif reshape2 in m:
        cgf = m[cgf2]  # type: BaseOperation
        transpose = m[transpose2]  # type: BaseOperation
        reshape = m[reshape2]  # type: BaseOperation
        if not (len(reshape.output.consumers) <= 1
                and len(transpose.output.consumers) <= 1
                and cgf.output not in g.outputs):
            raise _TransformException("Cannot apply TF_FILTER_GRAD_TO_NNEF")
        if reshape.output.rank == 3:  # conv1d
            cgf.output.name = reshape.output.name
            add_transform(transforms_by_name, cgf.output, Unsqueeze([0]))
            add_transform(transforms_by_name,
                          cgf.output,
                          Transpose(utils.inverse_permutation(driver.get_axes_from_transpose(transpose))))
            graph_utils.replace_tensor_in_outputs(g, reshape.output, cgf.output)
            graph_utils.remove_subgraph(g, [transpose, reshape])
        else:  # depthwise
            cgf.output.name = reshape.output.name
            reshape_shape = driver.get_shape_from_reshape(reshape)
            tmp_shape = (reshape_shape[:-2] + [1, int(reshape_shape[-2] * reshape_shape[-1])])
            add_transform(transforms_by_name, cgf.output, Reshape(tmp_shape))
            add_transform(transforms_by_name,
                          cgf.output,
                          Transpose(utils.inverse_permutation(driver.get_axes_from_transpose(transpose))))
            graph_utils.replace_tensor_in_outputs(g, reshape.output, cgf.output)
            graph_utils.remove_subgraph(g, [transpose, reshape])
    else:
        raise _TransformException("Cannot apply TF_FILTER_GRAD_TO_NNEF")


def _transform_nnef_filter_grad_to_tf(g, tensor, transforms_by_name, driver):
    # type: (BaseGraph, BaseTensor, typing.Dict[str, typing.List[Transform]], DataFormatOptimizationDriver)->None

    assert driver.conv_grad_filter_op_names

    cgf1_output = matcher.Tensor()
    cgf1 = matcher.Operation(name=driver.conv_grad_filter_op_names, outputs=cgf1_output)
    transpose1 = matcher.Operation(name=driver.transpose_op_name, inputs=cgf1_output)

    cgf2_output = matcher.Tensor()
    cgf2 = matcher.Operation(name=driver.conv_grad_filter_op_names, outputs=cgf2_output)
    reshape2_output = matcher.Tensor()
    reshape2 = matcher.Operation(name=driver.reshape_op_name, inputs=cgf2_output, outputs=reshape2_output)
    transpose2 = matcher.Operation(name=driver.transpose_op_name, inputs=reshape2_output)

    if tensor.producer is None:
        raise _TransformException("Cannot apply TF_FILTER_GRAD_TO_NNEF")

    m = matcher.match(g, tensor.producer, matcher.OrPattern(transpose1, transpose2))

    if transpose1 in m:
        cgf = m[cgf1]  # type: BaseOperation
        transpose = m[transpose1]  # type: BaseOperation
        if not (len(transpose.output.consumers) <= 1 and cgf.output not in g.outputs):
            raise _TransformException("Cannot apply TF_FILTER_GRAD_TO_NNEF")
        cgf.output.name = transpose.output.name
        add_transform(transforms_by_name,
                      cgf.output,
                      Transpose(utils.inverse_permutation(driver.get_axes_from_transpose(transpose))))
        graph_utils.replace_tensor_in_outputs(g, transpose.output, cgf.output)
        graph_utils.remove_subgraph(g, [transpose])
    elif transpose2 in m:
        cgf = m[cgf2]  # type: BaseOperation
        reshape = m[reshape2]  # type: BaseOperation
        transpose = m[transpose2]  # type: BaseOperation

        if not (len(reshape.output.consumers) <= 1
                and len(transpose.output.consumers) <= 1
                and cgf.output not in g.outputs):
            raise _TransformException("Cannot apply TF_FILTER_GRAD_TO_NNEF")

        cgf.output.name = transpose.output.name
        add_transform(transforms_by_name,
                      cgf.output,
                      Transpose(utils.inverse_permutation(driver.get_axes_from_transpose(transpose))))
        add_transform(transforms_by_name, cgf.output, Reshape(cgf.output.shape))

        graph_utils.replace_tensor_in_outputs(g, transpose.output, cgf.output)
        graph_utils.remove_subgraph(g, [transpose, reshape])
    else:
        raise _TransformException("Cannot apply TF_FILTER_GRAD_TO_NNEF")


def transform_io(g, io_transform, transforms_by_name, driver):
    # type:(BaseGraph, TransformOrTransformDictType, typing.Dict[str, typing.List[Transform]], DataFormatOptimizationDriver)->None
    io_tensors_by_name = {t.name: t for t in list(g.inputs) + list(g.outputs)}

    transform_by_io_tensor = {}
    if isinstance(io_transform, dict):
        for k, v in six.iteritems(io_transform):
            assert isinstance(k, (str, driver.tensor_type)), \
                "io_transform: Key type must be {} or str".format(driver.tensor_type.__name__)
            assert isinstance(v, Transform), "io_transform: Value type must be Transform"

            if isinstance(k, BaseTensor):
                assert k in six.itervalues(io_tensors_by_name)
            else:
                assert k in io_tensors_by_name
                k = io_tensors_by_name[k]
            transform_by_io_tensor[k] = v
        for io_tensor in six.itervalues(io_tensors_by_name):
            assert io_tensor in transform_by_io_tensor, \
                "io_transform: Please specify transform for all io tensors. " \
                "You can use graph_optimizer.IDENTITY if no change is required."
    else:
        assert isinstance(io_transform, Transform), \
            "io_transform must be Transform or Dict[str, Transform] or Dict[NNEFTensor, Transform]"
        for t in six.itervalues(io_tensors_by_name):
            transform_by_io_tensor[t] = io_transform

    for tensor, transform in six.iteritems(transform_by_io_tensor):
        assert bool(tensor in g.inputs) != bool(tensor in g.outputs), \
            "Tensor must be input or output (and not both)"

        assert isinstance(transform, (Transpose, _CustomTransform)), \
            "Unsupported io_transform"

        if isinstance(transform, _Identity):
            continue

        if isinstance(transform, _SmartTFNCHWToNCHW):
            try:
                _transform_tf_filter_grad_to_nnef(g, tensor, transforms_by_name, driver)
            except _TransformException:
                pass
            continue

        if isinstance(transform, _SmartNHWCToNCHW):
            if tensor.rank <= 2:
                continue
            transform = Transpose([0, tensor.rank - 1] + list(range(tensor.rank))[1:-1])

        if isinstance(transform, _SmartTFNHWCToNCHW):
            try:
                _transform_tf_filter_grad_to_nnef(g, tensor, transforms_by_name, driver)
                continue
            except _TransformException:
                if tensor.rank <= 2:
                    continue
                transform = Transpose([0, tensor.rank - 1] + list(range(tensor.rank))[1:-1])

        if isinstance(transform, _SmartNCHWToTFNCHW):
            try:
                _transform_nnef_filter_grad_to_tf(g, tensor, transforms_by_name, driver)
            except _TransformException:
                pass
            continue

        if isinstance(transform, _SmartNCHWToNHWC):
            if tensor.rank <= 2:
                continue
            transform = Transpose([0] + list(range(tensor.rank))[2:] + [1])

        if isinstance(transform, _SmartNCHWToTFNHWC):
            try:
                _transform_nnef_filter_grad_to_tf(g, tensor, transforms_by_name, driver)
                continue
            except _TransformException:
                if tensor.rank <= 2:
                    continue
                transform = Transpose([0] + list(range(tensor.rank))[2:] + [1])

        if isinstance(transform, _TFFilterGradToNNEF):
            _transform_tf_filter_grad_to_nnef(g, tensor, transforms_by_name, driver)
            continue

        if isinstance(transform, _NNEFFilterGradToTF):
            _transform_nnef_filter_grad_to_tf(g, tensor, transforms_by_name, driver)
            continue

        assert isinstance(transform, Transpose), "Unsupported io_transform: {}".format(transform)
        assert len(transform.axes) == tensor.rank, "Transpose: invalid rank"

        if transform.is_identity():
            continue

        if tensor in g.inputs:
            assert tensor.name

            new_input_tensor = driver.create_tensor(graph=g,
                                                    name=tensor.name,
                                                    shape=utils.apply_permutation(tensor.shape, transform.axes),
                                                    dtype=tensor.dtype)
            add_transform(transforms_by_name, new_input_tensor, transform)

            transpose = driver.create_transpose_op(graph=g, input=new_input_tensor,
                                                   axes=utils.inverse_permutation(transform.axes),
                                                   output=driver.create_tensor(graph=g,
                                                                               name=None,
                                                                               shape=tensor.shape,
                                                                               dtype=tensor.dtype))

            graph_utils.replace_tensor_in_inputs(g, tensor, new_input_tensor)
            graph_utils.replace_tensor_in_consumers(g, tensor, transpose.output, remove=True)
        else:  # output
            transpose = driver.create_transpose_op(graph=g,
                                                   input=tensor,
                                                   axes=transform.axes,
                                                   output=driver.create_tensor(
                                                       graph=g,
                                                       name=tensor.name,
                                                       shape=utils.apply_permutation(tensor.shape, transform.axes),
                                                       dtype=tensor.dtype))
            add_transform(transforms_by_name, transpose.output, transform)
            tensor.name = None

            graph_utils.replace_tensor_in_outputs(g, tensor, transpose.output)


def merge_transforms_into_varlikes(g, transforms_by_name, merge_into_constants, merge_into_variables, driver):
    # type: (BaseGraph, typing.Dict[str, typing.List[Transform]], bool, bool, DataFormatOptimizationDriver)->None
    transform_ops = [driver.squeeze_op_name,
                     driver.unsqueeze_op_name,
                     driver.reshape_op_name,
                     driver.transpose_op_name,
                     driver.copy_op_name]

    def get_param(op):
        if op.name == driver.squeeze_op_name:
            return driver.get_axes_from_squeeze(tensor.consumers[0])
        elif op.name == driver.unsqueeze_op_name:
            return driver.get_axes_from_unsqueeze(tensor.consumers[0])
        elif op.name == driver.transpose_op_name:
            return driver.get_axes_from_transpose(tensor.consumers[0])
        elif op.name == driver.reshape_op_name:
            return driver.get_shape_from_reshape(tensor.consumers[0])
        elif op.name == driver.copy_op_name:
            return None
        else:
            assert False

    for tensor in list(g.tensors):
        while (((merge_into_variables and tensor.is_variable) or (merge_into_constants and tensor.is_constant))
               and len(tensor.consumers) >= 1
               and tensor.consumers[0].name in transform_ops
               and tensor is tensor.consumers[0].inputs[0]  # need to check for onnx graph
               and all(t not in g.outputs for t in [tensor, tensor.consumers[0].output])):

            op_name = tensor.consumers[0].name
            op_param = get_param(tensor.consumers[0])

            if not all(op.name == op_name and get_param(op) == op_param
                       for op in tensor.consumers[1:]):
                break

            if op_name == driver.squeeze_op_name:
                axes = op_param

                tensor.shape = squeezed_shape(tensor.shape, axes)
                if tensor.is_variable and tensor.data.size > 0:
                    tensor.data = np.squeeze(tensor.data, tuple(axes))
                elif tensor.is_constant:
                    pass  # good as it is

                add_transform(transforms_by_name, tensor, Squeeze(axes))
            elif op_name == driver.unsqueeze_op_name:
                axes = op_param

                tensor.shape = unsqueezed_shape(tensor.shape, axes)
                if tensor.is_variable and tensor.data.size > 0:
                    tensor.data = np.reshape(tensor.data, tensor.shape)
                elif tensor.is_constant:
                    pass  # good as it is

                add_transform(transforms_by_name, tensor, Unsqueeze(axes))
            elif op_name == driver.reshape_op_name:
                tensor.shape = _reshaped_shape(tensor.shape, op_param)

                if tensor.is_variable and tensor.data.size > 0:
                    tensor.data = np.reshape(tensor.data, tensor.shape)
                elif tensor.is_constant:
                    pass  # good as it is

                add_transform(transforms_by_name, tensor, Reshape(tensor.shape))
            elif op_name == driver.transpose_op_name:
                apply_transpose_to_varlike(tensor, op_param, transforms_by_name)
            elif op_name == driver.copy_op_name:
                pass
            else:
                assert False

            for op in list(tensor.consumers):
                remove_passthrough_ex(g, op)


def _is_squeeze_invariant_to_perm(squeeze_axes, perm):
    dummy_shape = list(range(len(perm)))
    perm_dummy_shape = utils.apply_permutation(dummy_shape, perm)
    perm_squeeze_axes = Transposer.apply_permutation_to_axes(squeeze_axes, perm)
    shape1 = squeezed_shape(dummy_shape, squeeze_axes, can_squeeze_not_one=True)
    shape2 = squeezed_shape(perm_dummy_shape, perm_squeeze_axes, can_squeeze_not_one=True)

    return shape1 == shape2


def apply_transpose_to_varlike(tensor, axes, transforms_by_name):
    # type: (BaseTensor, typing.List[int], typing.Dict[str, typing.List[Transform]])->None

    if tensor.rank <= 1:
        return

    old_shape = tensor.shape
    tensor.shape = utils.apply_permutation(old_shape, axes)
    if tensor.is_variable and tensor.data.size > 0:
        tensor.data = np.transpose(tensor.data, axes)
    elif tensor.is_constant:
        if len(tensor.data) > 1:
            tensor.data = (np.array(tensor.data)
                           .reshape(old_shape)
                           .transpose(axes)
                           .flatten()
                           .tolist())

    add_transform(transforms_by_name, tensor, Transpose(axes))


_FromUpAndTensor = namedtuple('_FromUpAndTensor', ['from_up', 'tensor'])


class _Subgraph(object):
    def __init__(self, started_down, boundary_elements, skipped_tensors, visited_tensors):
        # type: (bool, typing.List[_FromUpAndTensor], typing.List[BaseTensor], typing.List[BaseTensor])->None
        self.started_down = started_down
        self.boundary_elements = boundary_elements
        self.skipped_tensors = skipped_tensors
        self.visited_tensors = visited_tensors


def _find_subgraph(
        tensor,  # type: BaseTensor
        is_skippable,  # type: typing.Callable[[BaseTensor], bool]
        is_boundary,  # type: typing.Callable[[BaseTensor, bool, bool], bool] # (tensor, from_op, start_down) -> bool
        start_down=True  # type: bool
):
    # type: (...)->typing.Optional[_Subgraph]

    if not is_boundary(tensor, not start_down, start_down):
        return None

    boundary = [_FromUpAndTensor(from_up=not start_down, tensor=tensor)]  # op, from_up
    skipped = []
    visited_tensors = {tensor}
    q = deque()

    def add(tensor_, from_up_):
        if tensor_ not in visited_tensors:
            visited_tensors.add(tensor_)
            q.append((tensor_, from_up_))

    if start_down:
        for c in tensor.consumers:
            for t in c.outputs:
                add(t, True)
    else:
        if tensor.producer:
            for t in tensor.producer.inputs:
                add(t, False)

    while q:
        a, from_up = q.popleft()  # type: BaseTensor, bool

        if is_boundary(a, from_up, start_down):
            boundary.append(_FromUpAndTensor(from_up=from_up, tensor=a))
            if from_up:
                for t in a.producer.inputs:
                    add(t, False)
            else:
                for c in a.consumers:
                    for t in c.outputs:
                        add(t, True)
        elif is_skippable(a):
            skipped.append(a)
            for t in a.producer.inputs:
                add(t, False)
            for c in a.consumers:
                for t in c.outputs:
                    add(t, True)
        else:
            return None

    return _Subgraph(started_down=start_down,
                     boundary_elements=boundary,
                     skipped_tensors=skipped,
                     visited_tensors=list(visited_tensors))


_AxesAndSubgraph = namedtuple('_AxesAndSubgraph', ['axes', 'subgraph'])


def _find_inverse_transposes(g, transposable_op_names, merge_into_constants, merge_into_variables, driver):
    # type: (BaseGraph, typing.Set[str], bool, bool, DataFormatOptimizationDriver)->typing.List[_AxesAndSubgraph]

    results = []  # type: typing.List[_AxesAndSubgraph]
    visited_tensors = set()

    for tensor in g.tensors:
        if (tensor.producer is not None
                and tensor.producer.name == driver.transpose_op_name
                and tensor not in visited_tensors):
            axes = driver.get_axes_from_transpose(tensor.producer)

            def is_boundary(tensor2, from_up, started_down):
                if tensor2 in visited_tensors:
                    return False
                if started_down:
                    perm = utils.inverse_permutation(axes)
                else:
                    perm = axes
                if from_up:
                    is_boundary_transpose = (tensor2.producer is not None
                                             and tensor2.producer.name == driver.transpose_op_name
                                             and driver.get_axes_from_transpose(tensor2.producer) == perm)
                    is_boundary_squeeze = (
                            tensor2.producer is not None
                            and tensor2.producer.name == driver.squeeze_op_name
                            and _is_squeeze_invariant_to_perm(driver.get_axes_from_squeeze(tensor2.producer), perm)
                    )

                    return is_boundary_transpose or is_boundary_squeeze
                else:
                    is_boundary_transpose = (
                            tensor2.producer is not None
                            and tensor2.producer.name == driver.transpose_op_name
                            and driver.get_axes_from_transpose(tensor2.producer) == utils.inverse_permutation(perm)
                    )
                    is_boundary_varlike = (((merge_into_constants and tensor2.is_constant)
                                            or (merge_into_variables and tensor2.is_variable))
                                           and len(tensor2.shape) in [0, 1, len(axes)])

                    return is_boundary_transpose or is_boundary_varlike

            def is_skippable(tensor2):
                if tensor2 in visited_tensors:
                    return False
                return tensor2.producer is not None and tensor2.producer.name in transposable_op_names

            if len(tensor.consumers) > 0:
                subgraph = _find_subgraph(tensor, is_skippable=is_skippable, is_boundary=is_boundary, start_down=True)
                if subgraph is not None:
                    visited_tensors.update(subgraph.visited_tensors)
                    results.append(_AxesAndSubgraph(axes, subgraph))
                    continue

            subgraph = _find_subgraph(tensor, is_skippable=is_skippable, is_boundary=is_boundary, start_down=False)
            if subgraph is not None:
                visited_tensors.update(subgraph.visited_tensors)
                results.append(_AxesAndSubgraph(axes, subgraph))
    return results


def transform_remove_inverse_transposes(
        g,  # type: BaseGraph
        transforms_by_name,  # type:typing.Dict[str, typing.List[Transform]]
        merge_into_constants,  # type: bool
        merge_into_variables,  # type: bool
        driver,  # type: DataFormatOptimizationDriver
        transposable_ops=None,  # type: typing.Optional[typing.List[TransposableOperation]]
):
    # type: (...)-> None

    if transposable_ops is None:
        transposable_ops = []

    transposable_op_by_name = {}  # type: typing.Dict[str, TransposableOperation]
    transposable_op_by_name.update({top.name: top for top in transposable_ops})

    for op in g.operations:
        if op.name == driver.transpose_op_name and op.output.rank > len(driver.get_axes_from_transpose(op)):
            driver.set_axes_on_transpose(op, driver.get_axes_from_transpose(op)
                                         + list(range(op.output.rank))[len(driver.get_axes_from_transpose(op)):])

    matches = _find_inverse_transposes(g,
                                       transposable_op_names=set(six.iterkeys(transposable_op_by_name)),
                                       merge_into_constants=merge_into_constants,
                                       merge_into_variables=merge_into_variables,
                                       driver=driver)

    for axes, subgraph in matches:
        upper_perm = axes if subgraph.started_down else utils.inverse_permutation(axes)
        lower_perm = utils.inverse_permutation(upper_perm)

        upper_boundary = [be for be in subgraph.boundary_elements if not be.from_up]
        lower_boundary = [be for be in subgraph.boundary_elements if be.from_up]

        for _, tensor in upper_boundary:
            if tensor.producer is not None and tensor.producer.name == driver.transpose_op_name:
                if tensor in g.outputs:
                    graph_output = driver.create_tensor(
                        graph=g,
                        name=tensor.name,
                        shape=utils.apply_permutation(tensor.producer.input.shape, upper_perm),
                        dtype=tensor.producer.input.dtype)
                    driver.create_transpose_op(graph=g,
                                               input=tensor.producer.input,
                                               axes=list(upper_perm),
                                               output=graph_output)
                    graph_utils.replace_tensor_in_outputs(
                        g, tensor,
                        graph_output)
                elif (len(tensor.producer.input.consumers) == 1
                      and tensor.producer.input not in g.inputs
                      and tensor.producer.input not in g.outputs):
                    tensor.producer.input.name = tensor.name
                    add_transform(transforms_by_name, tensor.producer.input, Transpose(lower_perm))
                remove_passthrough_ex(g, tensor.producer)
            else:
                assert (merge_into_variables and tensor.is_variable) \
                       or (merge_into_constants and tensor.is_constant)

                apply_transpose_to_varlike(tensor, lower_perm, transforms_by_name)

        skipped_ops = set(tensor.producer for tensor in subgraph.skipped_tensors)  # type: typing.Set[BaseOperation]
        for op in skipped_ops:
            assert op.name in transposable_op_by_name
            transposable_op_by_name[op.name].dg_transpose(_transposer, g, op, lower_perm)
            for output in op.outputs:
                if output in g.outputs:
                    graph_output = driver.create_tensor(graph=g,
                                                        name=output.name,
                                                        shape=output.shape,
                                                        dtype=output.dtype)
                    driver.create_transpose_op(graph=g,
                                               input=output,
                                               axes=list(upper_perm),
                                               output=graph_output)

                    graph_utils.replace_tensor_in_outputs(g, output, graph_output)
                    output.name = None
                    output.shape = utils.apply_permutation(output.shape, lower_perm)
                else:
                    output.shape = utils.apply_permutation(output.shape, lower_perm)
                    add_transform(transforms_by_name, output, Transpose(lower_perm))

        for _, tensor in lower_boundary:
            if tensor.producer is not None and tensor.producer.name == driver.transpose_op_name:
                if tensor in g.outputs:
                    graph_output = driver.create_tensor(graph=g,
                                                        name=tensor.name,
                                                        shape=tensor.producer.input.shape,
                                                        dtype=tensor.producer.input.dtype)

                    driver.create_copy_op(graph=g,
                                          input=tensor.producer.input,
                                          output=graph_output)

                    graph_utils.replace_tensor_in_outputs(g, tensor, graph_output)
                remove_passthrough_ex(g, tensor.producer)
            elif tensor.producer is not None and tensor.producer.name == driver.squeeze_op_name:
                driver.set_axes_on_squeeze(
                    tensor.producer,
                    sorted(Transposer.apply_permutation_to_axes(driver.get_axes_from_squeeze(tensor.producer),
                                                                lower_perm)))
            else:
                assert False

    graph_utils.remove_unreachable(g)


def transform_remove_unneeded_copies(g, driver):
    # type: (BaseGraph, DataFormatOptimizationDriver)->None
    for op in list(g.operations):
        if (op.name == driver.copy_op_name
                and not ((op.input in g.inputs or op.input in g.outputs) and op.output in g.outputs)):
            if op.output in g.outputs:
                op.input.name = op.output.name
            remove_passthrough_ex(g, op)


def _get_conversion_info(g, transforms_by_name, old_name_by_tensor):
    # type: (BaseGraph, typing.Dict[str, typing.List[Transform]], typing.Dict[BaseTensor, str])->ConversionInfo

    return ConversionInfo([
        TensorInfo(source_name=old_name_by_tensor[t],
                   target_name=t.name,
                   target_shape=t.shape,
                   target_dtype=t.dtype,
                   is_input=t in g.inputs,
                   is_output=t in g.outputs,
                   is_variable=t.is_variable,
                   transforms=transforms_by_name[old_name_by_tensor[t]])
        for t in g.tensors
        if t in old_name_by_tensor and old_name_by_tensor[t] in transforms_by_name
    ])


def propagate_quantizations(g, driver):
    # type: (BaseGraph, DataFormatOptimizationDriver)->None

    g.sort()

    propagatable = [driver.reshape_op_name,
                    driver.squeeze_op_name,
                    driver.unsqueeze_op_name,
                    driver.transpose_op_name]

    for op in list(g.operations):
        # output->input
        orig_op = op
        while True:
            if op is None or op.name not in propagatable or op.output.quantization is None:
                break
            input = driver.get_input_of_transform(op)
            if len(input.consumers) != 1 or getattr(input, "quantization", None) is not None:
                if not op.output.quantization.is_close_to(getattr(input, "quantization", None)):
                    print('Warning: Could not propagate quantization up for transform: {}'.format(op.output.name))
                break
            input.quantization = copy.copy(op.output.quantization)
            op = input.producer
        op = orig_op
        # input->output
        if op is None or op.name not in propagatable:
            continue
        input = driver.get_input_of_transform(op)
        if input.quantization is None:
            continue
        if getattr(op.output, "quantization", None) is not None:
            if not input.quantization.is_close_to(getattr(op.output, "quantization", None)):
                print('Warning: Could not propagate quantization down for transform: {}'.format(op.output.name))
                print(input.quantization, getattr(op.output, "quantization", None))
            continue
        op.output.quantization = copy.copy(input.quantization)


def optimize_impl(g,  # type: BaseGraph
                  driver,  # type: DataFormatOptimizationDriver
                  remove_unneeded_copies=False,  # type: bool
                  remove_inverse_transposes=False,  # type: bool
                  merge_transforms_into_variables=False,  # type: bool
                  merge_transforms_into_constants=False,  # type: bool
                  transposable_ops=None,  # type: typing.Optional[typing.List[TransposableOperation]]
                  io_transform=None,  # type:typing.Optional[TransformOrTransformDictType]
                  verbose=False,  # type: bool
                  rename_tensors=False,  # type: bool
                  ):
    # type: (...)->ConversionInfo

    if any(getattr(tensor, 'quantization', None) is not None for tensor in g.tensors):
        propagate_quantizations(g, driver)

    transforms_by_name = {t.name: [] for t in g.tensors}

    if io_transform is not None:
        transform_io(g, io_transform, transforms_by_name, driver)
        g.assert_consistent()

    last_len = len(g.operations)
    effective_loops = 0

    while True:
        if merge_transforms_into_constants or merge_transforms_into_variables:
            merge_transforms_into_varlikes(g,
                                           driver=driver,
                                           transforms_by_name=transforms_by_name,
                                           merge_into_constants=merge_transforms_into_constants,
                                           merge_into_variables=merge_transforms_into_variables)
            g.assert_consistent()

        if remove_inverse_transposes:
            transform_remove_inverse_transposes(g,
                                                driver=driver,
                                                transforms_by_name=transforms_by_name,
                                                merge_into_constants=merge_transforms_into_constants,
                                                merge_into_variables=merge_transforms_into_variables,
                                                transposable_ops=transposable_ops)
            g.assert_consistent()

        if len(g.operations) == last_len:
            break

        effective_loops += 1
        last_len = len(g.operations)

    if remove_unneeded_copies:
        transform_remove_unneeded_copies(g, driver=driver)
        g.assert_consistent()

    if verbose and effective_loops > 1:
        print("Info: Effective optimization loops: {}".format(effective_loops))

    old_name_by_tensor = {t: t.name for t in g.tensors if t.name is not None}
    if rename_tensors:  # so we get consecutive op names
        for t in g.tensors:
            if t not in g.inputs and t not in g.outputs:  # TODO maybe we could reset input/output names too?
                t.name = None

    driver.generate_missing_names(g)

    info = _get_conversion_info(g, transforms_by_name, old_name_by_tensor)

    g.assert_consistent()

    return info


__all__ = [
    'DataFormatOptimizationDriver',
    'TransposableOperation',
    'optimize_impl',
    'IOTransform',
    'TransformOrTransformDictType',
    'Transposer',
]
