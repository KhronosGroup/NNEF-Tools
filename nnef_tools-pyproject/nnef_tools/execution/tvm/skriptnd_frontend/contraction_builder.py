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

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph_builder import OperationInfo, TensorInfo

import skriptnd as sknd
from tvm import tir, ir

from .expression_builder import Environment, ExprBuilder


class NotExpressibleError(Exception):
    """ Raised when the contractions cannot be expressed in TIR (missing contraction list, atomic compound op) """
    pass


def _separate_contractions(contrs) -> dict[str, list[sknd.Contraction]]:
    """
    Separate contractions by output tensor and return a dict with output tensor name as key
    """
    output_contractions = {}

    for c in contrs:
        out_name = c.left.tensor.name  # tensor / tensorpack
        if out_name not in output_contractions:
            output_contractions[out_name] = []
        output_contractions[out_name].append(c)
    return output_contractions


def build_contraction(contractions: list[sknd.Contraction],
                      inputs: list[TensorInfo],
                      outputs: list[TensorInfo],
                      attribs: OperationInfo,
                      backward_map: dict[sknd.Expr, str],
                      rx_dyn_shape_vars: dict[str, Any],
                      return_stmt=False
                      ) -> tuple[tir.Stmt, list[Any]] | tir.PrimFunc:
    """
    Convert a list of SkriptND contractions to a TIR PrimFunc

    :param contractions: contractions of a SkND operation
    :param inputs: TensorInfo of the inputs of the operation
    :param outputs: TensorInfo of the outputs of the operation
    :param attribs: OperationInfo of the operation
    :param backward_map: For some expressions, the backward map is needed, to get some of their data
    :param rx_dyn_shape_vars: dynamic shape vars of the operation, will be mirrored to TIR
    :param return_stmt: return the Stmt instead of the PrimFunc
    :return: TIR PrimFunc or tuple of Stmt and list of function parameters
    """

    if not contractions:
        raise NotExpressibleError()

    # duplicate dyn shape vars to avoid casting issues of TVM (int64 -> int32)
    dyn_shape_vars = {k: tir.Var(k, "int32") for k in rx_dyn_shape_vars.keys()}

    # create TIR Buffers for inputs and outputs
    in_buffers = {t.name: _create_tir_placeholder(t, dyn_shape_vars) for t in inputs}
    out_buffers = {t.name: _create_tir_placeholder(t, dyn_shape_vars) for t in outputs}

    buffers = {**in_buffers, **out_buffers}

    # separate contractions by output tensor
    output_contractions = _separate_contractions(contractions)

    alloc_buffs = []

    def generate_body():
        nonlocal buffers

        stmts = []
        for name, contr_list in output_contractions.items():
            env = Environment(buffers, backward_map)
            exp_converter = ExprBuilder(env)

            # single contraction, no init block
            if len(contr_list) == 1:
                contr = contr_list[0]
                init_value = None
                ins, outs = _collect_ins_outs(contr, buffers)

                stmt = _contr_to_stmt(contr, ins, outs, buffers, env, exp_converter, init_value)
                stmts.append(stmt)

                continue

            # |contr_list| > 1: either single reduction with init or unrolled tensorpack, init and internal tensors needed
            # init Block
            contr = contr_list[0]
            try:
                # TODO???? sth needed this before,
                #  when the operation did more init, like upsampling(?)
                if not contr.axes:
                    # print("No reduction axes, init block won't work properly")
                    raise ValueError("No reduction axes, init block won't work properly")

                # check if init contraction is scalar
                # if env does not contain idx vars, expression conversion will fail, so init is not scalar
                for name, expr in contr.locals:
                    env.vars[name] = exp_converter(expr)
                env.curr_conds = []
                init_value = exp_converter(contr.right)
            except ValueError:  # ValueError from Environment's internal dict # todo? custom error to make nice
                # calculate init tensor manually and return indexing expression
                ins, outs = _collect_ins_outs(contr, buffers)

                # get TensorInfo of output
                out_info = next(o for o in outputs if o.name == contr.left.tensor.name)

                # init buffer
                if contr_list[1].assignment not in ["=", ":="] and contr.axes:
                    # create init buffer in the shape of current output, change output of initializer to this
                    suffix = "_init" if len(contr_list) == 2 else "_0"

                    buffers[contr.left.tensor.name + suffix] = _create_tir_placeholder(out_info, dyn_shape_vars,
                                                                                       suffix)

                    alloc_buffs.append(buffers[contr.left.tensor.name + suffix])
                else:
                    # only if the contraction is not update
                    suffix = ""

                outs = {contr.left.tensor.name + suffix: contr.left}

                init_block = _contr_to_stmt(contr, ins, outs, buffers, env, exp_converter)
                stmts.append(init_block)
                init_value = contr.left.tensor.name + suffix

            # spacial blocks
            for i, contr in enumerate(contr_list[1:]):
                # if the last contraction is reached, emmit the final output
                if i == len(contr_list) - 2:
                    ins, outs = _collect_ins_outs(contr, buffers)
                    suffix = ""
                else:
                    # if not the last contraction, create a new tensor for the intermediate result
                    out_info = next(o for o in outputs if o.name == contr.left.tensor.name)

                    suffix = "_init" if len(contr_list) == 2 else f"_{i}"
                    buffers[contr.left.tensor.name + suffix] = _create_tir_placeholder(out_info,
                                                                                       dyn_shape_vars,
                                                                                       suffix)

                    alloc_buffs.append(buffers[contr.left.tensor.name + suffix])
                    ins, outs = _collect_ins_outs(contr, buffers)
                    outs = {contr.left.tensor.name + suffix: contr.left}

                stmt = _contr_to_stmt(contr, ins, outs, buffers, env, exp_converter, init_value)
                stmts.append(stmt)

                # change init_value to the intermediate tensor
                init_value = contr.left.tensor.name + suffix

        if len(stmts) > 1:
            return tir.SeqStmt(stmts)
        else:
            return stmts[0]

    body = generate_body()

    # don't add duplicates to inputs
    params = list(in_buffers.values()) + list(out_buffers.values())

    if attribs.use_handles:
        params = [b.data for b in params]
        pass

    # for loops the statement is returned flatten nesting
    if return_stmt:
        return body, params


    # create root block + blockRealize
    body = tir.BlockRealize([], True,
                            tir.Block([], [], [], "root", body, alloc_buffers=alloc_buffs))

    return tir.PrimFunc(
        params=params,
        body=body,
    )


def _collect_ins_outs(contr: sknd.Contraction, buffers) -> tuple[dict[str, sknd.TensorAccess], dict[str, sknd.TensorAccess]]:
    # inputs: on rhs of contr
    ins = {}
    for i, exp in enumerate(sknd.recursive_enumerate_expr(contr.right)):
        if isinstance(exp, sknd.TensorAccess):
            ins[exp.tensor.name] = exp

    # outputs: on lhs of contr
    outs = {contr.left.tensor.name: contr.left}

    return ins, outs


def _contr_to_stmt(contr: sknd.Contraction, ins, outs, buffers, env: Environment, exp_converter: ExprBuilder,
                   init_value=None) -> tir.Stmt:
    it_vars = {}
    it_itvars = {}
    for i, (name, max) in enumerate(contr.bounds):
        it_vars[name] = tir.Var(name, "int32")
        if i in contr.axes:
            type = 2  # CommReduce type
        else:
            type = 0  # DataPar type
        it_itvars[name] = tir.IterVar(ir.Range(0, exp_converter(max)), "v_" + name, type)

    # update environment with iteration variables
    env.update(it_vars=it_vars, it_itvars=it_itvars)

    # output_name = contr.left.tensor.name
    output_name = list(outs.keys())[0]

    for name, expr in contr.locals:
        env.vars[name] = exp_converter(expr)
    env.curr_conds = []

    # get lhs call indices
    l_indices, conds = exp_converter.get_call_indices(contr.left)

    # get rhs
    rhs = exp_converter(contr.right)
    if contr.assignment not in ["=", ":="]:
        rhs = _reduction_function(contr.assignment)(tir.BufferLoad(buffers[output_name], l_indices), rhs)

    # reads are all inputs
    reads = []
    for i in ins.values():
        idx, _ = exp_converter.get_call_indices(i)
        ranges = [ir.Range.from_min_extent(i, 1) for i in idx]
        reads.append(tir.BufferRegion(buffers[i.tensor.name], ranges))

    # get writes
    writes = []
    for name, i in outs.items():
        idx, _ = exp_converter.get_call_indices(i)
        ranges = [ir.Range.from_min_extent(i, 1) for i in idx]
        writes.append(tir.BufferRegion(buffers[name], ranges))

    if isinstance(init_value, str):
        init_stmt = tir.BufferStore(buffers[output_name], tir.BufferLoad(buffers[init_value], l_indices), l_indices)
    elif init_value is not None:
        init_stmt = tir.BufferStore(buffers[output_name], init_value, l_indices)
    else:
        init_stmt = None

    # condition checked stmt
    body = tir.BufferStore(buffers[output_name], rhs, l_indices)
    if conds or env.call_conds:  # todo del debug

        # body = tir.SeqStmt([body, tir.Evaluate(tir.call_extern("void", "printf", "%d %d %d %d -> %d %d\n", *[v.var for v in it_itvars.values()], *l_indices[2:]))])

        body = tir.IfThenElse(tir.all(*conds, *env.call_conds), body, None)

    # condition checked init
    if init_stmt is not None:
        if conds:
            init_stmt = tir.IfThenElse(tir.all(*conds), init_stmt, None)
    inner_block = tir.Block(
        iter_vars=list(it_itvars.values()),
        reads=reads,
        writes=writes,
        name_hint=output_name,
        body=body,
        init=init_stmt,
    )

    block_realize = tir.BlockRealize(
        iter_values=list(it_vars.values()),
        predicate=True,
        # tir.all(*conds, *env.call_conds) if conds or env.call_conds else True, # todo fails because of affine
        block=inner_block,
    )

    def genarate_loops(stmt, indices):
        if len(indices) == len(contr.bounds):
            return stmt
        else:
            current = contr.bounds[len(indices)]
            loop_var = it_vars[current[0]]
            start = tir.const(0, "int32")
            end = exp_converter(current[1])
            if isinstance(end, tir.Var):
                if end.dtype != "int32":
                    end = tir.Cast("int32", end)
            indices[current[0]] = loop_var
            body = genarate_loops(stmt, indices)
            # todo diff type?
            return tir.For(loop_var, start, end, tir.ForKind.SERIAL, body)

    # reset environment
    env.reset()

    return genarate_loops(block_realize, {})


def _create_tir_placeholder(tensor: TensorInfo, dyn_shape_vars: dict[str, tir.Var], suffix="") -> tir.Buffer:
    shape = tensor.get_shape_values(dyn_shape_vars)
    return tir.decl_buffer(shape, tensor.dtype, tensor.name + suffix)


def _reduction_function(assignment):
    # todo error handle other methods as well?
    try:
        return {
            "+=": tir.add,
            "*=": tir.multiply,
            "&=": tir.all,
            "|=": tir.any,
            "<=": tir.min,
            ">=": tir.max,
        }[assignment]
    except KeyError:  # KeyError means not supported operation, in case of expansion
        raise ValueError(f"Assignment {assignment} not supported")
