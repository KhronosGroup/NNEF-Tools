from __future__ import annotations
from typing import Union, Callable

import logging
from collections import ChainMap
from dataclasses import dataclass, field
from os import PathLike, path

import numpy as np
import tvm
from tvm import relax, tir

import skriptnd as sknd

from . import convert_dtype, _atomics
from .expression_builder import ExprBuilder, Environment
from .high_level_op_lib import convert_map, NotInMapError, DimensionError, UnsupportedAttr
from .contraction_builder import build_contraction, NotExpressibleError

UNEXPRESSIBLE_OPERATIONS = ["top_k", "nonmax_suppress"]


def from_skriptnd(model: str | sknd.Model | PathLike,
                  keep_params_in_input: bool = False,
                  keep_ts_names: bool = False,
                  contr_only: bool = False,
                  atomics: dict[str, Callable[[...], bool] | bool] = None,
                  keep_intrin: dict[str, Callable[[...], bool] | bool] = None,
                  custom_converter_map: dict[str, Callable] = None,
                  ) -> tvm.IRModule:
    """
    Convert a SkriptND model to a TVM IRModule

    The Model has to be unrolled, and the atomic operations have to be resolved.
    Some defaults are provided in the package, but can be overridden by the user.

    :param model: Either a path to a file, a string containing the model, or a SkriptND Model object
    :param keep_ts_names: whether to try to keep the names of the object in the module, default False (currently only main function name depends on it)
    :param contr_only: whether to force the conversion to use contractions only, default False
    :param atomics: dictionary of atomic operations that can be handled by the consecutive transformation passes
    :param keep_intrin: dictionary of intrinsics that can be handled by a custom codegen
    :param custom_converter_map: dictionary of custom converters for high level operations, for detailed info see high_level_op_lib.py
    :return: TVM IRModule eqivalent of the input model
    """

    # if input is str, try to resolve it
    if isinstance(model, (str, PathLike)):
        if isinstance(model, PathLike) or path.isfile(model):
            model = sknd.read_model(model, unroll=True,
                                    atomic={**_atomics, **(atomics or {})} if not contr_only else False)
        else:
            # raise ValueError("TO DO")   # TODO
            try:
                model = sknd.parse_string(model, unroll=True,
                                          atomic={**_atomics, **(atomics or {})} if not contr_only else False)
            except Exception as e:
                raise ValueError(f"Invalid model string: {model}") from e

    if not isinstance(model, sknd.Model):
        raise ValueError(f"Invalid model: {model}")

    return ModuleBuilder(custom_converter_map, keep_ts_names, contr_only,
                         keep_intrin).convert_model(model)


def _get_shape_var_name(shape_elem, idx, info):
    # todo rewrite with expr_converter?
    if isinstance(shape_elem, sknd.PlaceholderExpr):
        return shape_elem.id

    # Currently the support only needs PlaceholderExpr, these are deprecated
    if isinstance(shape_elem, sknd.ShapeAccess):
        return shape_elem.tensor.shape[shape_elem.dim].id
    elif isinstance(shape_elem, sknd.UnaryExpr):
        if shape_elem.op == "~":
            return f"{info.name}:{idx}"
    else:
        raise ValueError(f"Invalid shape expression {type(shape_elem)}")


class ModuleBuilder:
    """
    Convert a SkriptND module to a TVM Relax IRModule

    The class contains the general conversion variables
    """

    def __init__(self,
                 custom_cm: dict,
                 keep_ts_names: bool,
                 use_contractions: bool,
                 keep_intrin: dict):
        # Module level attributes
        self.block_builder = relax.BlockBuilder()
        self.convert_map = convert_map(self.block_builder, custom_cm, keep_intrin)
        self.keep_ts_names = keep_ts_names  # TODO tedd bele tobbi valtozo letrehozasnal is myb?
        self.use_contractions = use_contractions

        # Subgraphs need to be either tPrimFunc or rFunction save them for starters
        self.subgraphs = {}

    def convert_model(self, module: sknd.Model):
        """
        Convert a SkriptND module to a TVM Relax IRModule
        """
        for graph in module.graphs[1:]:
            self.subgraphs[graph.name] = GraphBuilder(self, graph, self.use_contractions)

        main = GraphBuilder(self, module.graphs[0], self.use_contractions)
        main.subgraphs = self.subgraphs

        main.to_relax_func(name=module.graphs[0].name if self.keep_ts_names else "main")

        return self.block_builder.get()


class GraphBuilder:
    """
    Convert a SkriptND graph to a TVM Relax function (or PrimFunc)
    """

    def __init__(self, outer, graph: sknd.Graph, force_tir):
        # Graph level attributes
        self.module_builder: ModuleBuilder = outer
        self.graph = graph
        self.inputs = {}
        self.outputs = {}
        self.tensors: ChainMap[str, TensorInfo] = ChainMap({}, self.inputs, self.outputs)
        self.env = Environment()
        self.force_tir = force_tir

        self.subgraphs = None

        # setup functions
        self._load_tensors()

    def to_relax_func(self, issubgraph: bool = False, name: str = None):
        """
        Create a Relax function from the current graph, adding it into the BlockBuilder instance

        :param issubgraph: whether to make the resulting function private
        :param name: the name of the function, if None, the name of the graph will be used
        """

        # create Relax Tensors from TensorInfo, can contain tuples
        params = self._create_rx_vars(self.tensors.maps[1])

        # create main Relax function
        with self.module_builder.block_builder.function(
                self.graph.name if not name else name,
                params=list(params.values()),
                private=issubgraph,
        ):
            with self.module_builder.block_builder.dataflow():
                # create flattened relax tensors from TensorInfo,
                internals = ChainMap(self._create_rx_vars(self.tensors.maps[0], params), params)

                self.env.update(buffers=internals)

                for op in self.graph.operations:
                    op_attribs = OperationInfo(op.name,
                                               op.outputs,
                                               op.attribs,
                                               ExprBuilder(self.env, namespace=relax.op))

                    # convert operation, the result will be a relax.Call or a tir.PrimFunc
                    output = self._convert_op(op, internals, op_attribs)

                    # PrimFuncs need to be called
                    if isinstance(output, tir.PrimFunc):
                        glob_var = self.module_builder.block_builder.add_func(output, op_attribs.op_name)

                        # create the Relax call of the primFunc
                        output = self._call_relax(glob_var, op, internals, op_attribs)

                    # TODO check for tir at other todo as well
                    output = self.module_builder.block_builder.normalize(output)

                    # check if output is Tuple
                    if not isinstance(output.struct_info, relax.TupleStructInfo):
                        internals[op.outputs[0].name] = output
                        # output = relax.Tuple([output])
                    else:

                        i = 0
                        for o in op.outputs:
                            if isinstance(o, sknd.TensorPack):
                                pack_out = []
                                for item in o.items:
                                    internals[item.name] = output[i]
                                    pack_out.append(output[i])
                                    i += 1
                                # internals[o.name] = relax.Tuple(pack_out)
                            else:
                                internals[o.name] = output[i]
                                i += 1

                # collect outputs
                outputs = []
                for o in self.graph.outputs:
                    # bundle TensorPack into Tuple
                    if isinstance(o, sknd.TensorPack):
                        outputs.append(relax.Tuple([internals[t.name] for t in o.items]))
                    else:
                        outputs.append(internals[o.name])

                if len(outputs) == 1:
                    outputs = outputs[0]

                out_var = self.module_builder.block_builder.emit_output(outputs)

            self.module_builder.block_builder.emit_func_output(out_var)

    def to_tir_primfunc(self, name: str) -> tvm.ir.GlobalVar:
        """
        Create a TIR PrimFunc from the current graph and add it to the BlockBuilder instance, returning the GlobalVar

        :param name: the name of the function
        :return: the GlobalVar of the created function
        """
        self.force_tir = True

        params = self._create_tir_vars(self.tensors.maps[1])

        internals = ChainMap(self._create_tir_vars(self.tensors.maps[0]), params)

        body: list[tir.Stmt] = []
        for op in self.graph.operations:
            op_attribs = OperationInfo(op.name,
                                       op.outputs,
                                       op.attribs,
                                       ExprBuilder(namespace=relax.op),
                                       use_handles=True)
            stmt, op_pars = self._convert_op(op, internals, op_attribs, True)

            body.append(stmt)
            for p in op_pars:
                params[p.name] = p


        if len(body) > 1:
            body: tir.Stmt = tir.SeqStmt(body)
        elif len(body) == 1:
            # if isinstance(body[0], tir.Evaluate):
            #     return gv
            body: tir.Stmt = body[0]
        else:
            raise ValueError("No operations in graph")


        body = tir.BlockRealize(
            [],
            True,
            tir.Block(
                [],
                [],
                [],
                "root",
                body
            )
        )

        pf = tir.PrimFunc(
            params=list(params.values()),
            body=body,
        )
        #.with_attr("global_symbol", name)

        return self.module_builder.block_builder.add_func(pf, name)

    def _load_tensors(self):
        # todo check TP enumerate?
        for t in self.graph.inputs:
            if isinstance(t, sknd.TensorPack):
                # flattened
                # for i, o in enumerate(t.items):
                #     inner.inputs[o.name] = inner.module_builder.TensorInfo(o)
                # packed
                items = [TensorInfo(o) for o in t.items]
                # add items to internals, and the pack to input
                self.inputs[t.name] = tuple(items)
                for i in items:
                    self.tensors[i.name] = i

            else:
                self.inputs[t.name] = TensorInfo(t)

        for t in self.graph.outputs:
            if isinstance(t, sknd.TensorPack):
                for i, o in enumerate(t.items):
                    self.outputs[o.name] = TensorInfo(o)
            else:
                self.outputs[t.name] = TensorInfo(t)

        for t in self.graph.tensors:
            if t.name not in self.tensors:
                self.tensors[t.name] = TensorInfo(t)

    def _create_rx_vars(self, tensors: dict, rx_vars: dict = None):
        """Create Relax Tensors from a dict of SkND like TensorInfo"""
        # TODO rewrite if wanna change the loop order
        rx_tensors = {}
        for name, info in tensors.items():
            if isinstance(info, tuple):
                shapes = [i.shape for i in info]
                for j, shape in enumerate(shapes):
                    shape = list(shape)
                    for i, s in enumerate(shape):
                        if isinstance(s, sknd.Expr):
                            var_name = _get_shape_var_name(s, i, info)
                            if var_name not in self.env:
                                var = tvm.te.var(var_name, "int64")
                                self.env.vars[var_name] = var
                            shape[i] = self.env[var_name]
                    shapes[j] = tuple(shape)

                assert not any([isinstance(s, sknd.Expr) for shape in shapes for s in shape]), \
                    "Invalid shape, at least one shape is still ts.Expr"

                tens_struct_infos = [relax.TensorStructInfo(shape, i.dtype) for shape, i in zip(shapes, info)]
                tup_struct_info = relax.TupleStructInfo(tens_struct_infos)
                rx_tensors[name] = relax.Var(info[0].name.split(":")[0], tup_struct_info)

            else:
                # check if part of tuple, then get them with getitem
                if len(name.split(":")) > 1:
                    rx_tensors[name] = rx_vars[name.split(":")[0]][int(name.split(":")[1]) - 1]
                else:
                    shape = list(info.shape)
                    # Change dynamic SkND shapes to TVM vars
                    for i, s in enumerate(shape):
                        if isinstance(s, sknd.Expr):
                            var_name = _get_shape_var_name(s, i, info)
                            if var_name not in self.env:
                                var = tvm.te.var(var_name, "int64")
                                self.env.vars[var_name] = var
                            shape[i] = self.env[var_name]

                    assert not any([isinstance(s, sknd.Expr) for s in shape]), \
                        "Invalid shape, at least one shape is still ts.Expr"
                    pass

                    if info.const_value is not None:
                        if isinstance(info.const_value, (int, float, bool, str)):
                            rx_tensors[name] = relax.const(np.full(info.shape, info.const_value, dtype=info.dtype))
                        elif isinstance(info.const_value, sknd.ListExpr):
                            rx_tensors[name] = relax.const(np.array(info.const_value.items).reshape(info.shape),
                                                           dtype=info.dtype)
                        else:  # todo check uniform expr?
                            rx_tensors[name] = relax.const(info.const_value, dtype=info.dtype)
                    else:
                        struct_info = relax.TensorStructInfo(shape, info.dtype)
                        rx_tensors[name] = relax.Var(info.name, struct_info)

        return rx_tensors

    def _create_tir_vars(self, tensors: dict):
        """Create TIR Buffers from a dict of SkND like TensorInfo"""
        tir_tensors = {}
        for name, info in tensors.items():
            if isinstance(info, tuple):
                raise ValueError("Tuple not supported in TIR")
            else:
                shape = list(info.shape)
                # Change dynamic SkND shapes to TVM vars
                for i, s in enumerate(shape):
                    if isinstance(s, sknd.Expr):
                        var_name = _get_shape_var_name(s, i, info)
                        if var_name not in self.env:
                            var = tvm.te.var(var_name, "int32")
                            self.env.vars[var_name] = var
                        shape[i] = self.env[var_name]

                assert not any([isinstance(s, sknd.Expr) for s in shape]), \
                    "Invalid shape, at least one shape is still ts.Expr"

                tir_tensors[name] = tir.decl_buffer(tuple(shape), info.dtype, name)

        return tir_tensors

    def _convert_op(self, op: sknd.Operation, internals: dict, op_attribs, get_stmt=False) -> relax.Call | tir.PrimFunc:

        if op.name in ["if", "do"]:
            # TODO
            call = self._handle_subgraph_op(op, op_attribs, internals)
            # raise NotImplementedError("if/do not implemented")
            return call  # TODO return fucntion.

        # TODO check if TensorPack
        # assignment
        if op.name == "":
            logging.warning(f"Empty operation in {op}")  # todo ??
            # internals[op.outputs[0].name] = inputs[0]
            return internals[op.inputs[0].name]

        ## TODO ?? prettify

        output = None
        if not self.force_tir or op.name in UNEXPRESSIBLE_OPERATIONS:
            # try converting high level op
            try:
                output = self.__hlop(op, op_attribs, internals)
            except (NotInMapError, DimensionError, UnsupportedAttr) as e:
                if op.name in UNEXPRESSIBLE_OPERATIONS:
                    # if top_k or nonmax_suppress high level fails,
                    #   it can't be solved via the converter, re-raise error
                    raise e
                if isinstance(e, NotInMapError):
                    logging.log(logging.DEBUG, e)
                else:
                    logging.warning(str(e) + f" in {op.name}")

        if output is None:
            # convert operation by contractions, if it fails, leave error uncaught, as it can't be solved
            output = self.__contraction(op, op_attribs, internals, get_stmt)

        return output

    def __contraction(self, op: sknd.Operation, op_attribs, internals: ChainMap, get_stmt):
        """ Create a PrimFunc from a list of Contractions, then call_tir to create a Relax Tensor function """

        # todo check if tensorpack
        # collect TensorInfo of inputs to create TIR buffers
        input_infos = []  # [inner.tensors[i.name] for i in op.inputs]
        for i in op.inputs:
            if i is None:  # input is optional
                continue
            if isinstance(i, sknd.Tensor):
                input_infos.append(self.tensors[i.name])
            elif isinstance(i, sknd.TensorPack):
                input_infos.extend([self.tensors[t.name] for t in i.items])
            else:
                raise ValueError(f"Invalid input type {type(i)}")

        # todo check if append intermediate neccessary
        # extend with constants
        input_infos.extend([self.tensors[i.name] for i in op.constants])

        pass

        # save backward mapping from expr to name
        backward_map = {}
        # collect SubscriptExprs in the op, as they can be constants that need to be inputs
        sub_exprs = self._collect_expr_of_type(op, sknd.SubscriptExpr)
        if sub_exprs:
            for name, expr in sub_exprs.items():
                if isinstance(expr.pack, sknd.ListExpr):
                    # todo check if list elements are ts expr?
                    rx_const = relax.const(expr.pack.items, convert_dtype(expr.dtype))
                    internals[name] = rx_const
                    input_infos.append(TensorInfo(None,
                                                  name=name,
                                                  const_value=expr.pack.items,
                                                  dtype=convert_dtype(expr.dtype),
                                                  shape=(len(expr.pack.items),),
                                                  max_shape=(len(expr.pack.items),)))
                else:
                    raise ValueError(f"Invalid SubscriptExpr pack type {type(expr.pack)}")

                backward_map[expr.__repr__()] = name

        op_attribs.backward_map = backward_map

        # output infos
        output_infos = []
        for o in op.outputs:
            if isinstance(o, sknd.TensorPack):
                output_infos.extend([self.tensors[t.name] for t in o.items])
            else:
                output_infos.append(self.tensors[o.name])

        primfunc = build_contraction(op.contractions,
                                     input_infos,
                                     output_infos,
                                     op_attribs,
                                     backward_map,  # TODO remeve redundant
                                     self.env.vars,
                                     get_stmt,
                                     )

        if isinstance(primfunc, tir.PrimFunc):
            return primfunc.without_attr("global_symbol")

        return primfunc

    def _collect_expr_of_type(self, op: sknd.Operation, expr_type, name="internal"):
        exprs = {}
        for contr in op.contractions:
            root_expr_list = [contr.right, contr.left] + [t[1] for t in contr.locals]
            expr_list = [e for re in root_expr_list for e in sknd.recursive_enumerate_expr(re)]
            for i, exp in enumerate(expr_list):
                if isinstance(exp, expr_type):
                    exprs[f"{name}_{i}"] = exp

        return exprs

    def __hlop(self, op: sknd.Operation, op_attribs, internals: dict):

        inputs = []  # [internals[i.name] for i in op.inputs]
        for i in op.inputs:
            if i is None:  # input is optional save None
                inputs.append(None)
                continue
            if isinstance(i, sknd.Tensor):
                inputs.append(internals[i.name])
            elif isinstance(i, sknd.TensorPack):
                inputs.append([internals[t.name] for t in i.items])
            else:
                raise ValueError(f"Invalid input type {type(i)}")

        call = self.module_builder.convert_map[op.name](*inputs, attribs=op_attribs)
        return call

    def _call_tir(self, gv: tvm.ir.GlobalVar, op: sknd.Operation, internals: dict):
        inputs = []
        for i in op.inputs:
            if i is None:  # input is optional
                continue
            if isinstance(i, sknd.Tensor):
                inputs.append(internals[i.name])
            elif isinstance(i, sknd.TensorPack):
                inputs.extend([internals[t.name] for t in i.items])
            else:
                raise ValueError(f"Invalid input type {type(i)}")

        inputs = list(dict.fromkeys(inputs))
        inputs = [i.data for i in inputs]

        # create handles from inputs

        return tir.call_tir(gv, *inputs)

    def _call_relax(self, gv: tvm.ir.GlobalVar, op: sknd.Operation, internals: dict, op_attribs):

        # todo check TensorPack
        # todo nicer than this
        # create relax tensor info
        out_sinfo = []
        for i, o in enumerate(op.outputs):
            if isinstance(o, sknd.TensorPack):
                out_shape = [self.tensors[o_it.name].get_shape_values(self.env) for o_it in
                             o.items]
                out_sinfo.extend([relax.TensorStructInfo(s, op_attribs.output_dtype[i]) for s in out_shape])
            else:
                out_shape = self.tensors[o.name].get_shape_values(self.env)
                out_sinfo.append(relax.TensorStructInfo(out_shape, op_attribs.output_dtype[i]))

        # Collect Relax inputs
        inputs = []
        for i in op.inputs:
            if i is None:  # input is optional
                continue
            if isinstance(i, sknd.Tensor):
                inputs.append(internals[i.name])
            elif isinstance(i, sknd.TensorPack):
                inputs.extend([internals[t.name] for t in i.items])
            else:
                raise ValueError(f"Invalid input type {type(i)}")

        # expand with the added SubscriptExprs
        inputs.extend([internals[name] for name in op_attribs.backward_map.values()])

        # Remove duplicates while keeping order
        inputs = list(dict.fromkeys(inputs))

        # todo check tirvars param for dyn shape
        call = relax.call_tir(gv, inputs, out_sinfo)
        return call

    def _handle_subgraph_op(self, op: sknd.Operation, op_attribs, internals):
        if op.name == "if":
            assert len(op.attribs["cond_graphs"]) == 1, "Multiple conditions not supported"
            assert len(op.attribs["branch_graphs"]) == 2, "Exactly 2 (true&false) branches are supported"

            # load subgraphs as relax functions
            for n, g in self.subgraphs.items():
                g.to_relax_func(issubgraph=True, name=n)
            pass

            # TODO check how not 0th index
            c_graph = op.attribs["cond_graphs"][0]
            c_inputs = [op.inputs[i] for i in op.attribs["cond_inputs"]]
            _cond = relax.Call(
                self.module_builder.block_builder.get().get_global_var(c_graph.name.replace(".", "_")),
                [internals[var.name] for var in c_inputs])
            t_graph = op.attribs["branch_graphs"][0]
            t_inputs = [op.inputs[i] for i in op.attribs["branch_inputs"][:2]]
            _then = relax.Call(
                self.module_builder.block_builder.get().get_global_var(t_graph.name.replace(".", "_")),
                [internals[var.name] for var in t_inputs])
            e_graph = op.attribs["branch_graphs"][1]
            e_inputs = [op.inputs[i] for i in op.attribs["branch_inputs"][2:]]
            _else = relax.Call(
                self.module_builder.block_builder.get().get_global_var(e_graph.name.replace(".", "_")),
                [internals[var.name] for var in e_inputs])
            call = relax.If(
                _cond,
                _then,
                _else
            )
            internals[op.outputs[0].name] = self.module_builder.block_builder.normalize(call)
            return call
        if op.name == "do":
            if op.attribs["nscans"] != 0:
                raise NotImplementedError("Iterating over tuples with variables is not supported in TVM")
            if "cond_graph" in op_attribs.ts_attrs:
                return self.__while_loop(op, op_attribs, internals)

            else:
                return self.__for_count_loop(op, op_attribs, internals)

    def __for_count_loop(self, op: sknd.Operation, op_attribs, internals: dict):

        for n, g in self.subgraphs.items():
            g.to_tir_primfunc(n)

        params = {}
        # todo dyn shape?
        for i in op.inputs:
            if i is None:
                continue
            if isinstance(i, sknd.Tensor):
                params[i.name] = tir.decl_buffer(i.shape, convert_dtype(i.dtype), i.name)
            elif isinstance(i, sknd.TensorPack):
                for t in i.items:
                    params[t.name] = (tir.decl_buffer(t.shape, convert_dtype(t.dtype), t.name))
            else:
                raise ValueError(f"Invalid input type {type(i)}")

        if "iters" in op_attribs.ts_attrs:
            iters = op_attribs.ts_attrs["iters"]
        else:
            iters = tir.BufferLoad(params[op.inputs[1].name], [] if op.inputs[1].shape == () else [0])

        b_graph = op.attribs["body_graph"]
        b_inputs = [op.inputs[i] for i in op.attribs["body_inputs"]]
        # b_tir_inps = inner._create_tir_vars({k: v for k, v in inner.tensors.items() if k in [t.name for t in b_inputs] and k not in params})
        b_tir_inps = []
        for t in b_inputs:
            if t.name in params:
                b_tir_inps.append(params[t.name])
            else:
                b_tir_inps.append(tir.decl_buffer(t.shape, convert_dtype(t.dtype), t.name))
        b_gv = self.module_builder.block_builder.get().get_global_var(b_graph.name.replace(".", "_"))
        # b_pf = inner.module_builder.block_builder.get()[b_gv]

        b_out = []

        if op_attribs.ts_attrs["nvars"] == 0:
            b_out.append(tir.decl_buffer((), "int32", "body_out"))
        else:
            for i in range(0, op_attribs.ts_attrs["nvars"]):
                b_out.append(list(params.values())[i])

        loop_var = tir.Var("loop_var", "int32")

        for_stmt = tir.For(
            loop_var,
            0,
            iters,
            tir.ForKind.SERIAL,
            tir.Evaluate(tir.call_tir(b_gv, *[buf.data for buf in b_tir_inps + b_out])),
        )

        par = list(params.values())
        outs = [tir.decl_buffer(t.shape, convert_dtype(t.dtype), t.name) for t in op.outputs]
        copy_stmts = []
        for i, o in enumerate(outs):
            copy_stmts.append(tir.BufferStore(o, tir.BufferLoad(b_out[i], []), []))  # todo check indexing

        main_stmt = tir.SeqStmt([for_stmt, *copy_stmts])

        main_func_tir: tir.PrimFunc = tir.PrimFunc(
            params=par + outs,
            body=tir.BlockRealize(
                [],
                True,
                tir.Block(
                    [],
                    [],
                    [],
                    "loop",
                    main_stmt,
                )
            )
        ).with_attr("global_symbol", op_attribs.op_name)

        gv = self.module_builder.block_builder.add_func(main_func_tir, op_attribs.op_name)

        op_attribs.backward_map = {}

        call = self._call_relax(gv, op, internals, op_attribs)
        internals[op.outputs[0].name] = self.module_builder.block_builder.normalize(call)  # todo?
        return call

    def __while_loop(self, op: sknd.Operation, op_attribs, internals: dict):
        # todo check scoping for CUDA or other threading backends

        # assert len(op.attribs["cond_graph"]) == 1, "Multiple conditions not supported"
        cond_shape = op.attribs["cond_graph"].outputs[0].shape
        assert len(op.attribs["cond_graph"].outputs) == 1 and \
               len(cond_shape) == 0 or cond_shape == (1,), \
            "Only scalar conditions supported"

        cond_index = [] if cond_shape == () else [0]

        # load subgraphs as TIR functions
        for n, g in self.subgraphs.items():
            g.to_tir_primfunc(n)

        # op level params
        params = {}
        # todo dyn shape?
        for i in op.inputs:
            if i is None:
                continue
            if isinstance(i, sknd.Tensor):
                params[i.name] = tir.decl_buffer(i.shape, convert_dtype(i.dtype), i.name)
            elif isinstance(i, sknd.TensorPack):
                for t in i.items:
                    params[t.name] = (tir.decl_buffer(t.shape, convert_dtype(t.dtype), t.name))
            else:
                raise ValueError(f"Invalid input type {type(i)}")

        c_graph = op.attribs["cond_graph"]
        c_inputs = [op.inputs[i] for i in op.attribs["cond_inputs"]]

        # c_tir_inps = inner._create_tir_vars({k: v for k, v in inner.tensors.items() if k in [t.name for t in c_inputs] and k not in params})
        c_tir_inps = []
        for t in c_inputs:
            if t.name in params:
                c_tir_inps.append(params[t.name])
            else:
                c_tir_inps.append(tir.decl_buffer(t.shape, convert_dtype(t.dtype), t.name))

        c_gv = self.module_builder.block_builder.get().get_global_var(c_graph.name.replace(".", "_"))
        # c_pf = inner.module_builder.block_builder.get()[c_gv]

        c_out = tir.decl_buffer(cond_shape, "bool", "condition")

        b_graph = op.attribs["body_graph"]
        b_inputs = [op.inputs[i] for i in op.attribs["body_inputs"]]
        # b_tir_inps = inner._create_tir_vars({k: v for k, v in inner.tensors.items() if k in [t.name for t in b_inputs] and k not in params})
        b_tir_inps = []
        for t in b_inputs:
            if t.name in params:
                b_tir_inps.append(params[t.name])
            else:
                b_tir_inps.append(tir.decl_buffer(t.shape, convert_dtype(t.dtype), t.name))
        b_gv = self.module_builder.block_builder.get().get_global_var(b_graph.name.replace(".", "_"))
        # b_pf = inner.module_builder.block_builder.get()[b_gv]

        b_out = []

        if op_attribs.ts_attrs["nvars"] == 0:
            b_out.append(tir.decl_buffer((), "int32", "body_out"))
        else:
            for i in range(0, op_attribs.ts_attrs["nvars"]):
                b_out.append(list(params.values())[i])

        # if len(b_out) == 1:
        #     b_out = b_out[0]

        # TODO should be list?
        # b_out = tir.decl_buffer((), "int32", "body_out")

        # add aditional stmt, where we handle the swap of variables
        # swaps = []
        # for i in range(0, op_attribs.ts_attrs["nvars"]):
        #     swaps.append((list(params.values())[0], b_out))

        while_stmt = tir.While(
            tir.BufferLoad(c_out, cond_index),
            tir.SeqStmt([
                tir.Evaluate(tir.call_tir(b_gv, *[buf.data for buf in b_tir_inps + b_out])),
                tir.Evaluate(tir.call_tir(c_gv, *[buf.data for buf in c_tir_inps + [c_out]])),

            ])
        )

        # create encapsulating block, whose params are the params of the operation,
        # and separate buffers, which mirror the body's outputs, copying the data from the body's outputs

        par = list(params.values())
        outs = [tir.decl_buffer(t.shape, convert_dtype(t.dtype), t.name) for t in op.outputs]
        copy_stmts = []
        for i, o in enumerate(outs):
            copy_stmts.append(tir.BufferStore(o, tir.BufferLoad(b_out[i], cond_index), cond_index))

        main_stmt = tir.SeqStmt([tir.Evaluate(tir.call_tir(c_gv, *[buf.data for buf in c_tir_inps + [c_out]])),
                                 while_stmt, *copy_stmts])

        main_func_tir: tir.PrimFunc = tir.PrimFunc(
            params=par + outs,
            body=tir.BlockRealize(
                [],
                True,
                tir.Block(
                    [],
                    [],
                    [],
                    "loop",
                    main_stmt,
                    alloc_buffers=[c_out]  # todo check if b_out is needed
                )
            )
        ).with_attr("global_symbol", op_attribs.op_name)

        gv = self.module_builder.block_builder.add_func(main_func_tir, op_attribs.op_name)

        op_attribs.backward_map = {}

        call = self._call_relax(gv, op, internals, op_attribs)
        internals[op.outputs[0].name] = self.module_builder.block_builder.normalize(call)  # todo?
        return call


@dataclass(init=False)
class TensorInfo:
    name: str
    shape: tuple
    max_shape: tuple
    dtype: str
    const_value: Union[int, float, bool, str]

    def __init__(self, tensor: sknd.Tensor, name=None, shape=None, max_shape=None, dtype=None, const_value=None):
        self.name = tensor.name if name is None else name
        self.shape = tensor.shape if shape is None else shape
        self.max_shape = tensor.max_shape if max_shape is None else max_shape
        self.dtype = convert_dtype(tensor.dtype) if dtype is None else dtype
        self.const_value = tensor.value if const_value is None else const_value

    def get_shape_values(self, env):
        def _handle_dyn(s, i):
            if isinstance(s, sknd.Expr):
                var_name = _get_shape_var_name(s, i, self)
                if var_name in env:
                    return env[var_name]
                raise ValueError(f"Dynamic shape variable {var_name} not found")
            else:
                return s

        return [_handle_dyn(s, i) for i, s in enumerate(self.shape)]


@dataclass(init=False)
class OperationInfo:
    op_name: str
    output_shape: list[list]
    output_dtype: list[str]
    ts_attrs: dict
    backward_map: dict = field(default_factory=dict)
    use_handles = False

    def __init__(self, name: str, outs: list[sknd.Tensor], attrs: dict, exp_converter: ExprBuilder,
                 use_handles=False):
        self.op_name = name
        self.output_shape = [o.shape if not o == [] else [()] for o in outs]
        self.output_dtype = [convert_dtype(o.dtype) for o in outs]
        self.use_handles = use_handles

        self.ts_attrs = {}
        for k, v in attrs.items():
            if k.endswith("_graph") or k.endswith("_graphs"):
                self.ts_attrs[k] = v
            else:
                self.ts_attrs[k] = exp_converter(v)
            pass
