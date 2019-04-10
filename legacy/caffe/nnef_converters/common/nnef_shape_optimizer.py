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

from __future__ import division, print_function

import sys
from collections import deque, OrderedDict

import numpy as np

from . import dog
from . import utils

EXTRA_GENERATED_TRANSPOSE = 'generated_transpose'
EXTRA_GENERATED_SQUEEZE = 'generated_squeeze'
EXTRA_GENERATED_UNSQUEEZE = 'generated_unsqueeze'
EXTRA_GENERATED_RESHAPE = "generated_reshape"
EXTRA_APPLIED_TRANSFORMATIONS = "applied_transformations"  # value: (trafo_name, axes)


def unlink_and_push_down_arg(op, is_source_lang, dg_is_output_nnefdn, arg_name="input"):
    dn_input = op.args[arg_name]
    for dn_arg in op.get_arg_nodes():
        dn_arg.consumers.remove(op)
    replace_in_consumers(op.result, dn_input, is_source_lang, dg_is_output_nnefdn)


def replace_in_consumers(dn_old, dn_new, is_source_lang, dg_is_output_nnefdn):
    if dg_is_output_nnefdn(dn_old):
        if is_source_lang:
            dn_new.name = dn_old.name
        else:
            dn_new.source_name = dn_old.source_name

    def replace(x):
        if x == dn_old:
            return dn_new
        return x

    for consumer in dn_old.consumers:
        dn_new.consumers.append(consumer)
        consumer.args = utils.recursive_transform(consumer.args, replace)
        consumer.results = utils.recursive_transform(consumer.results, replace)


# is_skippable: (op) -> bool
# is_boundary: (op, from_up, started_down) -> bool
# return good, boundary=[[op, from_up]], skipped=[op], visited=[op]
def find_subgraph(op, is_skippable, is_boundary, start_down=True):
    if not is_boundary(op, not start_down, start_down):
        return False, None, None, None

    boundary = [(op, not start_down)]  # op, from_up
    skipped = []
    visited_ops = {op}
    q = deque()

    def add(op, from_up):
        if not op in visited_ops:
            visited_ops.add(op)
            q.append((op, from_up))

    if start_down:
        for r in op.get_result_nodes():
            for c in r.consumers:
                add(c, True)
    else:
        for a in op.get_arg_nodes():
            add(a.producer, False)

    while q:
        a, from_up = q.popleft()

        if is_boundary(a, from_up, start_down):
            boundary.append((a, from_up))
            if from_up:
                for arg in a.get_arg_nodes():
                    add(arg.producer, False)
            else:
                for res in a.get_result_nodes():
                    for c in res.consumers:
                        add(c, True)
        elif is_skippable(a):
            skipped.append(a)
            for arg in a.get_arg_nodes():
                add(arg.producer, False)
            for res in a.get_result_nodes():
                for c in res.consumers:
                    add(c, True)
        else:
            return False, None, None, None

    return True, boundary, skipped, list(visited_ops)


def apply_transpose_to_varlike(op_varlike, perm):
    if len(op_varlike.args["shape"]) <= 1:
        return

    old_shape = op_varlike.args["shape"]
    new_shape = utils.apply_permutation(old_shape, perm)

    op_varlike.args["shape"] = new_shape
    op_varlike.result.shape = new_shape
    add_transformation(op_varlike.result, ("transpose", perm))

    if op_varlike.name == "constant" and len(op_varlike.args["value"]) > 1:
        op_varlike.args["value"] = (np
                                    .array(op_varlike.args["value"])
                                    .reshape(old_shape)
                                    .transpose(perm)
                                    .flatten()
                                    .tolist())


def add_transformation(dn, trafo):
    assert isinstance(dn, dog.DataNode)
    trafos = dn.extra.setdefault(EXTRA_APPLIED_TRANSFORMATIONS, [])
    trafos.append(trafo)


def does_squeeze_return_the_same_after_perm(squeeze_axes, perm):
    dummy_shape = list(range(len(perm)))
    perm_dummy_shape = utils.apply_permutation(dummy_shape, perm)
    perm_squeeze_axes = utils.apply_permutation_to_axes(squeeze_axes, perm)
    shape1 = utils.apply_squeeze_shape(dummy_shape, squeeze_axes, can_squeeze_non_one=True)
    shape2 = utils.apply_squeeze_shape(perm_dummy_shape, perm_squeeze_axes, can_squeeze_non_one=True)

    return shape1 == shape2


def _get_ops_invariant_to_transpose():
    unary_ops = ['neg', 'rcp', 'exp', 'log', 'abs', 'sign', 'not', 'floor', 'ceil', 'round',
                 'sqr', 'sqrt', 'rsqr', 'rsqrt', 'log2']
    binary_ops = ['add', 'sub', 'mul', 'div', 'pow', 'lt', 'gt', 'le', 'ge', 'eq', 'ne', 'and', 'or',
                  'min', 'max']
    activation_ops = ['sigmoid', 'relu', 'leaky_relu', 'prelu', 'elu', 'tanh', 'softmax', 'softplus']
    reduce_ops = ['sum_reduce', 'min_reduce', 'max_reduce', 'mean_reduce']

    skippable_norm_ops = ['local_mean_normalization', 'local_variance_normalization',
                          'local_contrast_normalization', 'l1_normalization', 'l2_normalization']
    quantization_ops = ['linear_quantize', 'logarithmic_quantize']
    skippable_other_ops = ['select', 'slice', 'concat', 'stack', 'unstack', 'copy_n', 'add_n', 'moments', 'box',
                           '_nnef_pad', 'clamp', 'split']

    return set(unary_ops + binary_ops + activation_ops + reduce_ops + skippable_norm_ops + quantization_ops
               + skippable_other_ops)


ops_invariant_to_transpose = _get_ops_invariant_to_transpose()


# it's important that the transpose axes must be all written out
def transform_remove_inverse_gen_transposes(nnefops, is_source_lang, dg_is_output_nnefdn):
    can_remove_user_transposes = True

    if can_remove_user_transposes:
        def is_transpose(op):
            return op.name == "transpose"
    else:
        def is_transpose(op):
            return op.extra.get(EXTRA_GENERATED_TRANSPOSE)

    matches = []
    visited = set()
    for op in nnefops:
        if is_transpose(op) and op not in visited:
            def is_boundary(op2, from_up, started_down):
                if started_down:
                    perm = utils.get_inverse_permutation(op.args["axes"])
                else:
                    perm = op.args["axes"]
                if from_up:
                    is_boundary_transpose = (
                            is_transpose(op2) and op2.args["axes"] == perm and not dg_is_output_nnefdn(op2.result))
                    is_boundary_squeeze = (op2.name == "squeeze"
                                           and does_squeeze_return_the_same_after_perm(op2.args["axes"], perm))

                    return is_boundary_transpose or is_boundary_squeeze
                else:
                    is_boundary_transpose = (is_transpose(op2) and op2.args["axes"] == utils.get_inverse_permutation(
                        perm) and not dg_is_output_nnefdn(op2.result))
                    is_boundary_varlike = (op2.name in ["external", "constant", "variable"]
                                           and len(op2.args["shape"]) in [0, 1, len(op.args["axes"])])

                    return is_boundary_transpose or is_boundary_varlike

            def is_skippable(op2):
                return op2.name in ops_invariant_to_transpose

            found, boundary, skipped, new_visited = find_subgraph(op,
                                                                  is_skippable=is_skippable,
                                                                  is_boundary=is_boundary,
                                                                  start_down=True)
            if found:
                visited.update(new_visited)
                matches.append((op, True, boundary, skipped))
                continue

            found, boundary, skipped, new_visited = find_subgraph(op,
                                                                  is_skippable=is_skippable,
                                                                  is_boundary=is_boundary,
                                                                  start_down=False)
            if found:
                visited.update(new_visited)
                matches.append((op, False, boundary, skipped))

    to_remove = set()
    for op, started_down, boundary, skipped in matches:
        for op2, from_up in boundary:
            if started_down:
                perm = utils.get_inverse_permutation(op.args["axes"])
            else:
                perm = op.args["axes"]

            if from_up:
                if is_transpose(op2):
                    unlink_and_push_down_arg(op2, is_source_lang, dg_is_output_nnefdn, arg_name="input")
                    to_remove.add(op2)
                elif op2.name == "squeeze":
                    op2.args["axes"] = sorted(utils.apply_permutation_to_axes(op2.args["axes"], perm))
                else:
                    assert False
            else:
                if is_transpose(op2):
                    to_remove.add(op2)
                    unlink_and_push_down_arg(op2, is_source_lang, dg_is_output_nnefdn, arg_name="input")
                else:
                    assert op2.name in ["external", "constant", "variable"]
                    apply_transpose_to_varlike(op2, perm)
        if started_down:
            perm = utils.get_inverse_permutation(op.args["axes"])
        else:
            perm = op.args["axes"]
        for op2 in skipped:
            if op2.name == "slice":
                op2.args["axes"], op2.args["begin"], op2.args["end"] = utils.zip_inverse(3, sorted(zip(
                    utils.apply_permutation_to_axes(op2.args["axes"], perm), op2.args["begin"], op2.args["end"])))
            else:
                if 'axis' in op2.args:
                    op2.args["axis"] = utils.apply_permutation_to_axis(op2.args["axis"], perm)
                if 'axes' in op2.args:
                    op2.args["axes"] = sorted(utils.apply_permutation_to_axes(op2.args["axes"], perm))
                if 'size' in op2.args:
                    op2.args["size"] = utils.apply_permutation(op2.args["size"], perm)
                if op2.args.get('padding'):
                    op2.args["padding"] = utils.apply_permutation(op2.args["padding"], perm)
                if op2.args.get('stride'):
                    op2.args["stride"] = utils.apply_permutation(op2.args["stride"], perm)
                if op2.args.get('dilation'):
                    op2.args["dilation"] = utils.apply_permutation(op2.args["dilation"], perm)
            for res in op2.get_result_nodes():
                if res.shape:
                    res.shape = utils.apply_permutation(res.shape, perm)
                add_transformation(res, ("transpose", perm))

    return [op for op in nnefops if op not in to_remove]


def transform_merge_up_gen_reshape_to_varlike(nnefops, is_source_lang, dg_is_output_nnefdn):
    matches = []
    to_remove = set()
    for op in nnefops:
        if op.name in ["constant", "external", "variable"]:
            if not op.result.consumers:
                continue
            ok = True
            shape = None
            for nnefop_consumer in op.result.consumers:
                if (nnefop_consumer.extra.get(EXTRA_GENERATED_RESHAPE)
                        and (shape is None or shape == nnefop_consumer.args["shape"])):
                    if shape is None:
                        shape = nnefop_consumer.args["shape"]
                else:
                    ok = False
                    break
            if ok:
                matches.append(op)
                for op_reshape in op.result.consumers:
                    to_remove.add(op_reshape)

    for op_varlike in matches:
        shape = op_varlike.result.consumers[0].args["shape"]
        op_varlike.args["shape"] = shape
        op_varlike.result.shape = shape
        add_transformation(op_varlike.result, ("reshape", shape))

        old_consumers = list(op_varlike.result.consumers)
        for op_reshape in old_consumers:
            unlink_and_push_down_arg(op_reshape, is_source_lang, dg_is_output_nnefdn, arg_name="input")

    return [op for op in nnefops if op not in to_remove]


def transform_merge_up_gen_unsqueeze_to_varlike(nnefops, is_source_lang, dg_is_output_nnefdn):
    matches = []
    to_remove = set()
    for op in nnefops:
        if op.name in ["constant", "external", "variable"]:
            if not op.result.consumers:
                continue
            ok = True
            axes = None
            for nnefop_consumer in op.result.consumers:
                if (nnefop_consumer.extra.get(EXTRA_GENERATED_UNSQUEEZE)
                        and (axes is None or axes == nnefop_consumer.args["axes"])):
                    if axes is None:
                        axes = nnefop_consumer.args["axes"]
                else:
                    ok = False
                    break
            if ok:
                matches.append(op)
                for op_unsqueeze in op.result.consumers:
                    to_remove.add(op_unsqueeze)

    for op_varlike in matches:
        axes = op_varlike.result.consumers[0].args["axes"]
        shape = utils.apply_unsqueeze_shape(op_varlike.args["shape"], axes)
        op_varlike.args["shape"] = shape
        op_varlike.result.shape = shape
        add_transformation(op_varlike.result, ("unsqueeze", axes))

        old_consumers = list(op_varlike.result.consumers)
        for op_unsqueeze in old_consumers:
            unlink_and_push_down_arg(op_unsqueeze, is_source_lang, dg_is_output_nnefdn, arg_name="input")

    return [op for op in nnefops if op not in to_remove]


def transform_merge_up_gen_squeeze_to_varlike(nnefops, is_source_lang, dg_is_output_nnefdn):
    matches = []
    to_remove = set()
    for op in nnefops:
        if op.name in ["constant", "external", "variable"]:
            if not op.result.consumers:
                continue
            ok = True
            axes = None
            for nnefop_consumer in op.result.consumers:
                if (nnefop_consumer.extra.get(EXTRA_GENERATED_SQUEEZE)
                        and (axes is None or axes == nnefop_consumer.args["axes"])):
                    if axes is None:
                        axes = nnefop_consumer.args["axes"]
                else:
                    ok = False
                    break
            if ok:
                matches.append(op)
                for op_unsqueeze in op.result.consumers:
                    to_remove.add(op_unsqueeze)

    for op_varlike in matches:
        axes = op_varlike.result.consumers[0].args["axes"]
        shape = utils.apply_squeeze_shape(op_varlike.args["shape"], axes)
        op_varlike.args["shape"] = shape
        op_varlike.result.shape = shape
        add_transformation(op_varlike.result, ("squeeze", axes))

        old_consumers = list(op_varlike.result.consumers)
        for op_unsqueeze in old_consumers:
            unlink_and_push_down_arg(op_unsqueeze, is_source_lang, dg_is_output_nnefdn, arg_name="input")

    return [op for op in nnefops if op not in to_remove]


def transform_remove_unreachables(ops, dg_is_output_nnefdn):
    visited = set()

    q = deque()

    output_ops = []
    for op in ops:
        if any([dg_is_output_nnefdn(result) for result in op.get_result_nodes()]):
            output_ops.append(op)

    for op in utils.unique(output_ops):
        q.append(op)
        visited.add(op)

    while len(q) > 0:
        op = q.popleft()

        for dn in op.get_arg_nodes():
            if dn.producer and dn.producer not in visited:
                visited.add(dn.producer)
                q.append(dn.producer)

    return [op for op in ops if op in visited]


def transform_shape_optimizations(nnefops, is_source_lang, dg_is_output_nnefdn):
    effective_loops = 0
    while True:
        n = len(nnefops)
        nnefops = transform_merge_up_gen_squeeze_to_varlike(nnefops, is_source_lang, dg_is_output_nnefdn)
        nnefops = transform_merge_up_gen_reshape_to_varlike(nnefops, is_source_lang, dg_is_output_nnefdn)
        nnefops = transform_merge_up_gen_unsqueeze_to_varlike(nnefops, is_source_lang, dg_is_output_nnefdn)
        nnefops = transform_remove_inverse_gen_transposes(nnefops, is_source_lang, dg_is_output_nnefdn)
        nnefops = transform_remove_unreachables(nnefops, dg_is_output_nnefdn)  # Not really needed

        if n == len(nnefops):
            break
        else:
            effective_loops += 1

    if effective_loops >= 3:
        utils.print_warning("Performed optimizations in {} effective loops".format(effective_loops))

    return nnefops


def copy_transformations(nnefdn_by_otherdn):
    for otherdn, nnefdn in nnefdn_by_otherdn.items():
        if not nnefdn:
            continue

        otherdn.source_name = nnefdn.name
        trafos = nnefdn.extra.get(EXTRA_APPLIED_TRANSFORMATIONS)
        if trafos:
            otherdn.extra[EXTRA_APPLIED_TRANSFORMATIONS] = trafos


def add_comments(dns):
    for dn in dns:
        comment = dn.extra.get(dog.EXTRA_COMMENT, "")  # type: str
        if dn.source_name:
            comment += " " + str(dn.source_name)
        applied_trafo = dn.extra.get(EXTRA_APPLIED_TRANSFORMATIONS)
        if applied_trafo:
            comment += " " + str(applied_trafo)
        dn.extra[dog.EXTRA_COMMENT] = comment.lstrip() if comment else None


# nnefdns or otherdns also good
def get_trafo_dict(dns, output_dns,
                   input_op_name="external", variable_op_name="variable", constant_op_name="constant",
                   label_arg_name="label"):
    d = OrderedDict([
        ("input", OrderedDict()),
        ("output", OrderedDict()),
        ("variable", OrderedDict()),
        ("constant", OrderedDict()),
        ("activation", OrderedDict())])

    for dn in dns:
        name = dn.name
        trafos = dn.extra.get(EXTRA_APPLIED_TRANSFORMATIONS, [])
        d["activation"][name] = list(trafos)
        if dn.producer.name == variable_op_name:
            d["variable"][dn.producer.args[label_arg_name]] = list(trafos)
        elif dn.producer.name == input_op_name:
            d["input"][name] = list(trafos)
        elif dn.producer.name == constant_op_name:
            d["constant"][name] = list(trafos)
    for dn in output_dns:
        trafo_list = dn.extra.get(EXTRA_APPLIED_TRANSFORMATIONS, [])
        d["output"][dn.name] = list(trafo_list)
    return d


def print_trafos(trafo_dict, keys=None, file=None):
    if file is None:
        file = sys.stdout
    if keys is None:
        keys = trafo_dict.keys()
    if has_trafos(trafo_dict, keys):
        print('Applied transformations:', file=file)
        for k, v in trafo_dict.items():
            if k in keys and has_trafos(trafo_dict, [k]):
                print(k.title() + 's:', file=file)
                for k2, v2 in v.items():
                    if v2:
                        print(k2 + ':', v2, file=file)


def has_trafos(trafo_dict, keys=None):
    if keys is None:
        keys = trafo_dict.keys()
    for k, v in trafo_dict.items():
        if k in keys:
            for k2, v2 in v.items():
                if v2:
                    return True
    return False
