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

import _skriptnd as _sknd
import numpy as np
import re


def valid_id(name):
    id = re.sub('[^~_0-9a-zA-Z]+', '_', name)
    if len(id) > 0 and id[0].isdigit():
        id = "__" + id
    return id


class Printer:

    def __init__(self, inline_subgraphs=False, module=None):
        self._inline_subgraphs = inline_subgraphs
        self._module_scope = module + '.' if module else None

    def __call__(self, model, file=None):
        self._block_scope = ''
        self._used_ids = {self._make_id(tensor.name) for tensor in model.tensors}
        self._next_shape_id = 0

        self._print_graph(model, model.graphs[0], idx=0, file=file)

        if not self._inline_subgraphs:
            for i, graph in enumerate(model.graphs[1:]):
                self._print_graph(model, graph, idx=i + 1, file=file)

    @staticmethod
    def _strip_scope(name, scope):
        return name[len(scope):] if scope and name.startswith(scope) and len(name) > len(scope) else name

    def _make_id(self, name):
        return valid_id(self._strip_scope(name, self._block_scope))

    def _can_inline(self, tensor):
        return tensor.shape is not None and len(tensor.shape) == 0 and \
               tensor.value is not None and not isinstance(tensor.value, (np.ndarray, list))

    def _format_value(self, value, include_max=False):
        if value is None:
            return "~"  # null tensor
        elif isinstance(value, _sknd.PlaceholderExpr):
            s = '~'
            if include_max:
                s += '|'
                s += self._format_value(value.max_value)
            return s
        elif isinstance(value, _sknd.ShapeAccess):
            s = self._make_id(value.tensor.name)
            if value.item is not None:
                s += '[' + str(value.item) + ']'
            s += '.shape'
            if value.dim is not None:
                s += '[' + str(value.dim) + ']'
            return s
        elif isinstance(value, _sknd.SizeAccess):
            return self._make_id(value.pack.name) + '.size'
        elif isinstance(value, _sknd.Tensor):
            if self._can_inline(value):
                if isinstance(value.value, bool):
                    return "true" if value.value else "false"
                else:
                    return str(value.value)
            else:
                return self._make_id(value.name)
        elif isinstance(value, _sknd.TensorPack):
            if not value.name.startswith('.'):
                return self._make_id(value.name)
            else:
                return "[" + ", ".join(self._format_value(v) for v in value) + "]"
        elif isinstance(value, _sknd.Graph):
            return value.name
        elif isinstance(value, np.ndarray):
            return "[" + ", ".join(self._format_value(v.item()) for v in value.flat) + "]"
        elif isinstance(value, (list, tuple)):
            return "[" + ", ".join(self._format_value(v) for v in value) + "]"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, str):
            return '"' + value + '"'
        else:
            return str(value)

    def _format_shape(self, shape, include_max=False, ignore_dynamic_shape=False):
        return "[" + ",".join('~' if ignore_dynamic_shape and isinstance(s, _sknd.Expr) else
                              self._format_value(s, include_max)
                              for s in shape) + "]"

    def _format_param(self, name, dtype, shape, default=None, optional=False, packed=False, repeats=None,
                      include_shape_max=False, ignore_dynamic_shape=False):
        text = self._make_id(name) + ": "
        if optional:
            text += "optional "
        text += dtype.name.lower()
        if shape is not None:
            text += self._format_shape(shape, include_shape_max, ignore_dynamic_shape)
        if packed:
            text += ".."
        if repeats is not None:
            text += "(" + str(repeats) + ")"
        if default is not None:
            text += " = " + self._format_value(default)
        return text

    def _format_result(self, result, repeats):
        if result is None:
            return "~"
        elif isinstance(result, _sknd.Tensor):
            return self._make_id(result.name)
        elif isinstance(result, _sknd.TensorPack):
            if not result.name.startswith('.'):
                id = self._make_id(result.name)
                return f"{id}..({repeats})" if repeats is not None else id
            else:
                return "[" + ", ".join(self._format_result(r, None) for r in result) + "]"
        elif isinstance(result, tuple):
            return ", ".join(self._format_result(r, repeats) for r in result)
        elif isinstance(result, list):
            return "[" + ", ".join(self._format_result(r, None) for r in result) + "]"
        else:
            assert False

    def _format_subgraph(self, target, inputs):
        if isinstance(target, _sknd.Tensor):
            return self._format_value(target)
        elif self._inline_subgraphs:
            if len(target.operations) == 0:
                if len(target.inputs) == 1:
                    return self._format_value(target.inputs[0])
                else:
                    return '{ yield ' + ', '.join(self._format_value(input)
                                                  for input in target.inputs) + '; }'
            else:
                label = self._make_id(target.name)
                text = label + ': {\n'
                for op in target.operations:
                    text += "\t\t\t"
                    text += self._format_operation(op.outputs, op.name, op.dtypes.values(),
                                                   op.attribs, op.inputs) + ";\n"
                text += '\t\t\tyield ' + ', '.join(self._make_id(output.name) for output in target.outputs) + ';\n'
                text += '\t\t}'
                return text
        else:
            args = list(inputs)
            args = [item for item in args
                    if (isinstance(item, _sknd.TensorPack) and len(item) > 0)
                    or (isinstance(item, _sknd.Tensor) and not self._can_inline(item))]
            name = self._make_id(self._strip_scope(target.name, self._module_scope))
            return self._format_invocation(name, args)

    def _format_invocation(self, name, args, dtypes=None, attribs=None, alias=None, label=None):
        text = ""
        if label:
            text += label + ": "
        text += name
        if dtypes:
            text += "<" + ",".join(dtype.name.lower() for dtype in dtypes) + ">"
        if attribs and any(v is not None for k, v in attribs.items()):
            text += "{" + ", ".join(k + "=" + self._format_value(v) for k, v in attribs.items() if v is not None) + "}"
        text += "(" + ", ".join(self._format_value(a) for a in args) + ")"
        if alias:
            text += " as " + alias
        return text

    def _format_operation(self, results, name, dtypes, attribs, args, alias=None):
        if name == 'do':
            nvars = attribs['nvars']
            nscans = attribs['nscans']
            static_iters = attribs.get('iters')
            dynamic_iters = args[nvars + nscans]
            repeats = self._format_value(static_iters) \
                if static_iters is not None and dynamic_iters is not None else None
            text = self._format_result(results, repeats)
            text += " = "
        else:
            text = self._format_result(results, None) + " = "

        if name == 'if':
            conditions = attribs['cond_graphs']
            branches = attribs['branch_graphs']
            cond_input_indices = attribs['cond_inputs']
            branch_input_indices = attribs['branch_inputs']
            cond_input_offset = 0
            branch_input_offset = 0
            for i, (condition, branch) in enumerate(zip(conditions, branches)):
                cond_inputs = [args[idx] for idx in cond_input_indices[cond_input_offset:cond_input_offset + len(condition.inputs)]] \
                    if isinstance(condition, _sknd.Graph) else None
                branch_inputs = [args[idx] for idx in branch_input_indices[branch_input_offset:branch_input_offset + len(branch.inputs)]] \
                    if isinstance(branch, _sknd.Graph) else None
                text += (('if ' if i == 0 else ' elif ') + self._format_subgraph(condition, cond_inputs) +
                         ' then ' + self._format_subgraph(branch, branch_inputs))
                cond_input_offset += 1 if isinstance(condition, _sknd.Tensor) else len(condition.inputs)
                branch_input_offset += 1 if isinstance(branch, _sknd.Tensor) else len(branch.inputs)

            branch = branches[-1]
            branch_inputs = [args[idx] for idx in branch_input_indices[branch_input_offset:branch_input_offset + len(branch.inputs)]] \
                if isinstance(branch, _sknd.Graph) else None
            text += ' else ' + self._format_subgraph(branch, branch_inputs)
        elif name == 'do':
            condition = attribs.get('cond_graph')
            cond_input_indices = attribs.get('cond_inputs')
            body = attribs['body_graph']
            body_input_indices = attribs['body_inputs']
            nvars = attribs['nvars']
            nscans = attribs['nscans']
            static_iters = attribs.get('iters')
            dynamic_iters = args[nvars + nscans]
            index_name = attribs.get('index')
            pretest = attribs.get('pretest', False)

            index = _sknd.Tensor(name=index_name, dtype=_sknd.Dtype.Int, shape=(), canonic_shape=(), max_shape=()) if index_name else None

            subgraph_inputs = body.inputs[:nvars + nscans] + (index,) + args[nvars + nscans + 1:]
            cond_inputs = [subgraph_inputs[idx] for idx in cond_input_indices] \
                if condition and isinstance(condition, _sknd.Graph) else None

            body_inputs = [subgraph_inputs[idx] for idx in body_input_indices] \
                if isinstance(body, _sknd.Graph) else None
            body_inputs[:nvars] = body.inputs[:nvars]

            if nvars > 0:
                ids = [self._make_id(tensor.name) for tensor in body.inputs[:nvars]]
                inits = args[:nvars]
                shapes = [f": {res.dtype.name.lower()}{self._format_shape(res.shape)}"
                          if res.shape is not None and init.shape is not None and
                             len(res.shape) != 0 and len(init.shape) == 0 else ''
                          for init, res in zip(args[:nvars], results[:nvars])]
                text += 'with '
                text += ', '.join(f'{id}{shape} = {self._format_value(value)}'
                                  for id, shape, value in zip(ids, shapes, inits))

            if nscans > 0:
                if nvars > 0:
                    text += ' '
                ids = [self._make_id(tensor.name) for tensor in body.inputs[nvars:nvars+nscans]]
                scans = args[nvars:nvars+nscans]
                text += 'for '
                text += ', '.join(f'{id} : {self._format_value(value)}' for id, value in zip(ids, scans))

            if condition and pretest:
                if nvars > 0 or nscans > 0:
                    text += ' '
                text += 'while ' + self._format_subgraph(condition, cond_inputs)

            if nvars > 0 or nscans > 0 or (condition and pretest):
                text += ' '
            text += 'do'

            if index_name is not None or dynamic_iters is not None or static_iters is not None:
                text += '..('
                if index_name is not None:
                    text += self._make_id(index_name) + ' -> '
                if dynamic_iters is not None:
                    text += self._format_value(dynamic_iters)
                elif static_iters is not None:
                    text += self._format_value(static_iters)
                text += ')'

            text += ' ' + self._format_subgraph(body, body_inputs)

            if condition and not pretest:
                text += ' while ' + self._format_subgraph(condition, cond_inputs)

        elif name == '':    # assignment
            text += self._format_value(args[0] if len(args) != 0 else attribs[''])
        else:
            text += self._format_invocation(name, args, dtypes, attribs, alias)

        return text

    def _print_graph(self, model, graph, idx, file):
        self._block_scope = graph.name + '.'

        print("graph " + self._make_id(self._strip_scope(graph.name, self._module_scope)) + " {", file=file)

        print("\t@input {", file=file)
        for input in graph.inputs:
            if isinstance(input, _sknd.TensorPack):
                print("\t\t" + self._format_param(input.name, input.dtype, input.shape,
                                                  packed=True, repeats=len(input),
                                                  include_shape_max=True) + ";", file=file)
            elif not self._can_inline(input):
                print("\t\t" + self._format_param(input.name, input.dtype, input.shape,
                                                  include_shape_max=True) + ";", file=file)
        print("\t}", file=file)

        print("\t@output {", file=file)
        for output in graph.outputs:
            if isinstance(output, _sknd.TensorPack):
                print("\t\t" + self._format_param(output.name, output.dtype, output.shape,
                                                  packed=True, repeats=len(output),
                                                  ignore_dynamic_shape=True) + ";", file=file)
            else:
                print("\t\t" + self._format_param(output.name, output.dtype, output.shape,
                                                  ignore_dynamic_shape=True) + ";", file=file)
        print("\t}", file=file)

        if self._inline_subgraphs and idx == 0:
            variables = model.variables
            constants = model.constants
        else:
            variables = graph.variables
            constants = graph.constants

        variables = list(variables)
        constants = list(tensor for tensor in constants if not self._can_inline(tensor))

        if len(constants) > 0:
            print("\t@constant {", file=file)
            for tensor in constants:
                print("\t\t" + self._format_param(tensor.name, tensor.dtype, tensor.shape,
                                                  default=tensor.value) + ";", file=file)
            print("\t}", file=file)

        if len(variables) > 0:
            print("\t@variable {", file=file)
            for tensor in variables:
                print("\t\t" + self._format_param(tensor.name, tensor.dtype, tensor.shape) + ";", file=file)
            print("\t}", file=file)

        print("\t@compose {", file=file)
        for op in graph.operations:
            print("\t\t" + self._format_operation(op.outputs, op.name, op.dtypes.values(), op.attribs, op.inputs)
                  + ";", file=file)
        print("\t}", file=file)

        print("}\n", file=file)


def print_model(model, file=None, inline_subgraphs=False, module=None):
    printer = Printer(inline_subgraphs=inline_subgraphs, module=module)
    printer(model, file)
